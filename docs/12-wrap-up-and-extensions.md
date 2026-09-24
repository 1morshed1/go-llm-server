# 12 — Wrap-up: what you built, and where to go next

You've now read (and hopefully run) a complete LLM serving layer. This final doc
ties the threads together, maps what you built onto the real systems it models,
and lays out the honest edges — what's intentionally *not* here and how you'd add
it.

## The whole system in one breath

A request arrives at the **API layer** (REST or gRPC), which renders the chat
messages into a prompt, tokenizes it, marks the shared prefix, and builds a
`domain.Request` carrying its own output channel. The **scheduler** admits it (or
sheds load with 429), orders it in a priority queue, and times it out if it waits
too long. The **continuous batcher** pulls it into an active batch, asking the
**KV-cache manager** for memory (reusing a cached prefix when possible), runs one
batched forward pass per iteration on the **engine**, and streams each token back
through the request's channel — letting finished requests leave and new ones join
mid-flight. **Prometheus metrics** observe all of it; **Grafana** draws it.

Every arrow in that paragraph is a package, and every package has a doc.

## How it maps to the real systems

| This project | vLLM / TGI / Ollama |
|--------------|---------------------|
| `engine.Engine` interface + `MockEngine` | the model executor / backend (llama.cpp, CUDA graphs) |
| `batcher` continuous loop | vLLM's iteration-level scheduler ("continuous batching") |
| `kvcache` blocks + prefix cache | PagedAttention + automatic prefix caching |
| `scheduler` priority queue + admission | the waiting queue + `max_num_seqs` / preemption |
| `prompt` templating | the model's chat template |
| OpenAI REST + SSE | vLLM/Ollama's OpenAI-compatible server |
| gRPC `Generate` stream | TGI/Triton's gRPC generate |
| Prometheus metrics | the same metric families these servers export |

The concepts transfer directly. When you next read vLLM's scheduler or Ollama's
runner, you'll recognize the shapes.

## What's intentionally *not* here (and how you'd add it)

Being explicit about scope is part of the lesson. None of these are missing by
accident:

- **A real compute backend.** The whole point of the `Engine` interface (doc 02).
  Add `internal/engine/llama.go` behind a `//go:build llama` tag with CGo
  bindings to `llama.cpp`; select it by config in `main`. Nothing above the
  interface changes.
- **A model manager (load/unload, multiple models, mmap, hot-swap).** The brief
  sketches this. Today we serve one model. You'd add a `ModelManager` mapping
  model names to `Engine` instances, resolve `req.Model` to an engine in the
  batcher, and support atomic swap-in of a new version. The `Engine` interface is
  already the right seam for it.
- **Chunked prefill.** We pay prefill synchronously at admission (doc 06). Real
  servers interleave prefill and decode in the same step to avoid a long prompt
  stalling the batch. The `Step` API could be extended to carry prefill chunks.
- **Preemption + lazy KV growth.** We reserve generation room up front (doc 04);
  vLLM grows lazily and preempts a running sequence back to the queue under
  pressure. A natural next step once you're comfortable with the block manager.
- **Auto prefix detection.** We require the caller to mark the prefix; vLLM
  detects shared prefixes across live requests with a radix tree.
- **Auth, rate limits per key, request logging, tracing.** Production edges. The
  narrow `Submitter` interface and the metrics seam are where they'd attach.
- **Kubernetes manifests + GPU scheduling.** The image is K8s-ready; doc 09 lists
  the Deployment/probe/ServiceMonitor pieces and the GPU node-pool bits.
- **Speculative decoding, quantization, LoRA adapters.** Model-side techniques
  that live behind the engine interface.

## Suggested exercises

If you want to *prove* you understand it, try these in order of difficulty:

1. **Turn off prefix caching** (`--prefix-cache=false`) and watch TTFT and the
   hit-rate gauge change. Explain the delta from first principles.
2. **Shrink the KV cache** (`--kv-blocks 32`) and drive load until you see
   `ErrNoCapacity` back-pressure. Watch `kv_cache_utilization` pin at 1.0 while
   compute sits idle — the capacity wall, live.
3. **Add a new sampling parameter** end-to-end: wire it from the OpenAI/gRPC
   request through `prompt.GenSpec` to `SamplingParams`. Notice how few files you
   touch.
4. **Implement a second `Engine`** that adds latency jitter, and confirm the
   batcher, scheduler, and API are completely unaffected.
5. **Write the `ModelManager`** and serve two models from one process.

## The one idea to keep

Everything here rests on a single move: **keep the transport and the compute at
the edges, behind interfaces, and let a small set of `domain` types flow through
a core of channels and goroutines.** That's why a simulated engine can drive real
scheduling code, why a second protocol cost nothing to add, and why each piece
fits in your head. It's not an LLM idea — it's a systems idea. The LLM is just a
particularly good place to see why it matters.
