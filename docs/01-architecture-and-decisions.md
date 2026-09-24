# 01 — Architecture & the big decisions

This doc records the *shape* of the system and the load-bearing decisions
behind it. It doubles as the project's **decision log** — new decisions get
appended here as later phases add them.

## The layered architecture

```
        ┌───────────────────────────────────────────────┐
        │                 API layer                      │  transport
        │  HTTP (Gin) · OpenAI-compatible · SSE stream   │  concerns only
        └───────────────────────┬───────────────────────┘
                                 │ domain.Request
        ┌───────────────────────▼───────────────────────┐
        │                 Scheduler                      │  when to run
        │  priority queue · admission control · timeout  │
        └───────────────────────┬───────────────────────┘
                                 │ pulls next request
        ┌───────────────────────▼───────────────────────┐
        │              Continuous batcher                │  how to run
        │  active batch · forward-pass loop · streaming  │  many at once
        └───────────┬───────────────────────┬───────────┘
                    │                        │
      ┌─────────────▼──────────┐  ┌──────────▼───────────┐
      │     KV-cache manager   │  │   Inference engine   │  the compute
      │  blocks · prefix cache │  │   (Engine interface) │  boundary
      └────────────────────────┘  └──────────────────────┘
```

Everything above the engine is **pure Go systems code** that runs anywhere.
The engine is the one place that would, in production, touch C++/CUDA.

## Decision log

### D1 — Build the serving layer against an `Engine` *interface*, and ship a simulated engine as the default backend.

**Context.** The project brief says "use `llama.cpp` via CGo or ONNX Runtime as
the compute backend." That's the right choice *in production*. But wiring it in
first has enormous costs that have nothing to do with what this project teaches:

- A CGo build of `llama.cpp` needs a C++ toolchain and platform-specific build
  flags; it turns `go build` into a 10-minute compile.
- You need a real model file (a 7B GGUF is ~4 GB) just to start the server.
- Without a GPU the numbers are dominated by CPU matmul, which is not what we're
  studying and makes tests slow and non-deterministic.

**Decision.** Define a small `engine.Engine` interface that captures exactly
what the serving layer needs from a model:

```go
type Engine interface {
    Prefill(ctx, req) (state, error)   // process the prompt (the expensive first step)
    Decode(ctx, state) (Token, done, error) // produce ONE next token
    Release(state)                     // free per-request compute state
    Info() ModelInfo                   // name, context length, vocab size
}
```

Ship a **`MockEngine`** as the default: it implements the interface and
*simulates* a transformer's observable behaviour — prefill latency roughly
proportional to prompt length, a per-token decode latency, an EOS token, and
deterministic output seeded from the prompt. A real `LlamaEngine` (CGo) would
implement the same four methods.

**Why this is the *right* design, not a shortcut.**

- **Dependency inversion.** The scheduler and batcher depend on an abstraction,
  not on `llama.cpp`. That's the Dependency Inversion Principle, and it's how
  you keep a large system decoupled.
- **Testability.** A deterministic engine makes the batcher, scheduler, and API
  unit-testable with no external state. This is *the* reason to have the seam.
- **Benchmarkability.** We can drive thousands of requests through the real
  scheduling/batching code on a laptop and measure the batching win, because
  the mock's timings are configurable and repeatable.
- **It mirrors reality.** vLLM, TGI, and Ollama all abstract the "executor" /
  "backend" behind an interface for exactly these reasons.

**Trade-off / what we give up.** We are not measuring *real* model tokens/sec on
real hardware. The benchmarks in [doc 10](10-benchmarks.md) therefore measure
*the serving layer's* behaviour (queueing, batching throughput, TTFT under
load) using a controllable simulated compute cost — which is precisely the thing
this project is about. Swapping in a real engine is a documented, self-contained
change; [doc 02](02-the-engine-interface.md) sketches it.

### D2 — Use `domain` types as the lingua franca; no transport types below the API layer.

The `internal/domain` package has no imports of `net/http`, `gin`, JSON tags
that matter for logic, or Prometheus. Every lower layer speaks `domain.Request`
and `domain.StreamEvent`. This keeps each layer independently testable and means
you can read the domain package to understand the whole system's nouns. (See
D2 in practice: the API layer has its own `openai` DTO types that it *maps* to
`domain.Request` — the wire format never leaks inward.)

### D3 — One request carries one output channel; the batcher is a pure producer.

Rather than have the batcher hold references to HTTP response writers (coupling
compute to transport, and making cancellation messy), each `domain.Request`
carries a `Stream chan StreamEvent`. The batcher only ever *produces* onto it;
the API handler only ever *consumes*. Cancellation flows the other way via
`Request.Ctx`. This producer/consumer split is what lets streaming, timeouts,
and client-disconnect handling all compose cleanly.

### D4 — Gin for HTTP.

The brief specifies Gin, and it's a reasonable default: fast router, good
middleware story, ubiquitous in Go shops. Nothing about the design depends on
it — handlers map wire types to `domain` types and back, so the router could be
swapped for `net/http` or chi without touching a lower layer.

### D5 — `internal/` for everything.

All packages live under `internal/` so they can't be imported by other modules.
This is a deliberate signal: these are implementation details of *this server*,
not a reusable library. It keeps us free to refactor package boundaries without
worrying about external consumers.

---

*Later phases append their decisions below as they are made.*
