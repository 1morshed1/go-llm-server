# 02 — The engine interface

> Code: `internal/engine/engine.go`, `internal/engine/mock.go`, `internal/engine/rng.go`

## The idea

An inference server has to talk to *something* that runs the model. That
something is the **engine** (vLLM calls it the "executor" / "model runner",
Ollama shells out to `llama.cpp`, TGI has "backends"). The rest of the server —
queueing, batching, streaming, metrics — does not care *how* a token is
produced, only that it can ask for one.

So we draw a line. Above the line: pure Go systems code that runs anywhere.
Below the line: the model compute, which in production is C++/CUDA. The line is
a Go interface:

```go
type Engine interface {
    Info() ModelInfo
    Prefill(ctx, req, uncachedPromptTokens) (*Sequence, error)
    Step(ctx, seqs []*Sequence) ([]StepResult, error)
    Release(seq *Sequence)
}
```

Four methods. That's the entire contract between "the server" and "the model."

## Prefill vs. decode — the two phases you must internalize

This split shows up *everywhere* in LLM serving, so it's worth getting crisp:

| Phase | What happens | Cost | Sets... |
|-------|--------------|------|---------|
| **Prefill** | Model reads the whole prompt at once, fills the KV cache | ∝ prompt length, done once | **TTFT** (time to first token) |
| **Decode** | Model emits one token, attending over the KV cache; repeat | ~constant per step per sequence; **batches cheaply** | **throughput** (tokens/sec) |

Two consequences drive the whole design:

1. **Prefill dominates TTFT.** A long prompt takes a while before the first
   token appears. This is *why prefix caching matters* — if the prompt's prefix
   is already cached, prefill can skip it. Our `Prefill` takes
   `uncachedPromptTokens` precisely so the cost reflects a cache hit (doc 04).

2. **Decode batches almost for free.** A GPU running a decode step for 16
   sequences takes barely longer than for 1, because it's the same matmuls with
   a wider batch dimension. That is the entire economic basis of continuous
   batching (doc 06). Our `Step` takes a *slice* of sequences and advances them
   all in one call, with a latency that grows slowly with batch size.

## The mock engine

`MockEngine` implements the interface and *simulates* those cost profiles
instead of doing real matmuls:

- **Prefill latency** = `PrefillPerToken × uncachedPromptTokens`. Cached prefix
  tokens are free → prefix caching visibly lowers TTFT.
- **Step latency** = `StepBase + StepPerSeq × batchSize`, with
  `StepPerSeq ≪ StepBase`. So one step for 16 sequences (`18 + 16×1 = 34 ms`)
  produces 16 tokens, vs one step for 1 sequence (`18 + 1 = 19 ms`) producing 1.
  Per-token cost drops from 19 ms to ~2.1 ms — the batching win, made
  measurable.
- **Output** is deterministic pseudo-text: a per-sequence SplitMix64 PRNG seeded
  from the prompt picks readable words, so demos look plausible and tests are
  repeatable.
- Every simulated sleep goes through `sleepCtx`, which returns early on context
  cancellation — so client disconnects and timeouts abort promptly.

### Why simulate instead of wiring in llama.cpp now?

The full argument is decision **D1** in [doc 01](01-architecture-and-decisions.md).
In short: a real backend needs a C++ toolchain, a multi-GB model file, and
ideally a GPU, none of which teach you anything about *serving*, and all of
which make the code slow, non-deterministic, and hard to test. The interface is
the real lesson (dependency inversion + testability). The simulation lets the
*actual* scheduler and batcher run at full speed against a controllable cost
model.

## How a real backend slots in

You would add `internal/engine/llama.go` with a CGo build tag:

```go
//go:build llama

package engine

/*
#cgo LDFLAGS: -lllama
#include "llama.h"
*/
import "C"

type LlamaEngine struct { /* *C.llama_context, batch handles, ... */ }

func (e *LlamaEngine) Info() ModelInfo { /* from model metadata */ }
func (e *LlamaEngine) Prefill(ctx, req, uncached) (*Sequence, error) {
    // llama_batch with the prompt tokens; llama_decode once.
}
func (e *LlamaEngine) Step(ctx, seqs) ([]StepResult, error) {
    // one llama_batch containing one new token slot per sequence;
    // llama_decode once for the whole batch; sample each logits row.
}
func (e *LlamaEngine) Release(seq) { /* free the seq's KV cells */ }
```

Notice how the `Step`-over-a-batch shape maps directly onto `llama_decode`
processing a batch: **the interface was designed around how real engines
actually work**, which is what makes it a faithful teaching model rather than a
convenient fiction. `cmd/server/main.go` would select the backend by config;
nothing above the line changes.

## Decisions & trade-offs

- **The engine reports the model's behaviour; the batcher enforces policy.**
  `StepResult.EOS` means "the *model* chose to stop." Limits like `MaxTokens`
  and stop-strings live in the batcher. Keeping "what the model does" separate
  from "what the server allows" means either can change without touching the
  other.
- **`Sequence` is opaque to callers.** The batcher holds `*Sequence` values but
  never inspects them. That's what lets a real engine store native pointers in
  there without leaking into the serving layer.
- **Determinism over realism in output.** The mock's *text* is nonsense; its
  *timing shape* is what's faithful. We optimized for the property that lets us
  test and benchmark the serving layer, and were explicit that the token text
  is not meaningful.
