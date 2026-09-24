# go-llm-server

A from-scratch **LLM inference server** written in Go — a teaching-grade,
simplified version of what [vLLM](https://github.com/vllm-project/vllm),
[Ollama](https://github.com/ollama/ollama), and
[TGI](https://github.com/huggingface/text-generation-inference) do.

It implements the *serving layer* that sits between a model and the HTTP API
your applications call:

- **OpenAI-compatible REST API** (`POST /v1/chat/completions`, streaming + non-streaming)
- **Request scheduler** with a priority queue, admission control, and per-request timeouts
- **Continuous batching** (the vLLM trick that gives 2–3× throughput over naive batching)
- **KV-cache manager** with **prefix caching** (reuse the KV of shared system prompts)
- **Pluggable inference engine** behind a single Go interface
- **Prometheus metrics** + a **Grafana dashboard**
- **Docker Compose** stack (server + Prometheus + Grafana)

> **The compute backend is pluggable.** The interesting, hard, and portable part
> of an inference server is *everything around* the matrix multiplications:
> queuing, batching, cache management, streaming, backpressure, observability.
> This repo ships a **simulated engine** by default so the whole system runs,
> is testable, and is benchmarkable on any laptop — no GPU, no 4 GB model
> download, no CGo toolchain. A real `llama.cpp` backend slots in behind the
> same `engine.Engine` interface. See
> [`docs/02-the-engine-interface.md`](docs/02-the-engine-interface.md) for why
> this is the *right* design, not just a convenient one.

## Learning path

This project is built to be **read**, not just run. The `docs/` folder is a
guided tour, written in the order the code was built. Start here:

| Doc | What you'll learn |
|-----|-------------------|
| [00 — Overview & how to read this repo](docs/00-overview.md) | Orientation and the mental model |
| [01 — Architecture & the big decisions](docs/01-architecture-and-decisions.md) | Why the system is shaped this way |
| [02 — The engine interface](docs/02-the-engine-interface.md) | Dependency inversion; the mock vs. real backend |
| [03 — Tokenization](docs/03-tokenization.md) | What a tokenizer actually does |
| [04 — The KV cache & prefix caching](docs/04-kv-cache.md) | The single biggest memory concept in LLM serving |
| [05 — The scheduler](docs/05-scheduler.md) | Priority queues, admission control, backpressure |
| [06 — Continuous batching](docs/06-continuous-batching.md) | The core throughput trick, explained line by line |
| [07 — The HTTP API & streaming](docs/07-http-api-and-streaming.md) | OpenAI compatibility and Server-Sent Events |
| [08 — Observability](docs/08-observability.md) | The metrics that matter and why |
| [09 — Deployment](docs/09-deployment.md) | Docker, Compose, Prometheus, Grafana |
| [10 — Benchmarks](docs/10-benchmarks.md) | Measuring TTFT, throughput, and the batching win |

Every doc also records **decisions and trade-offs** as we go — that decision
log is the real learning material.

## Quick start

```bash
# Run the server (uses the simulated engine by default)
go run ./cmd/server

# In another terminal — a non-streaming request:
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock-7b","messages":[{"role":"user","content":"Hello!"}]}' | jq

# A streaming request (Server-Sent Events):
curl -N localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock-7b","stream":true,"messages":[{"role":"user","content":"Tell me a story"}]}'
```

Other endpoints: `GET /healthz`, `GET /v1/models`, `GET /metrics`.

## Project layout

```
cmd/server/          Entry point: wiring and graceful shutdown
internal/
  domain/            Core types shared by every layer (no transport code)
  config/            Configuration (flags + environment variables)
  tokenizer/         Tokenizer interface + a simple word-piece tokenizer
  engine/            Engine interface + the simulated backend
  kvcache/           KV-cache manager + prefix caching
  scheduler/         Priority queue, admission control, timeouts
  batcher/           Continuous batcher (the forward-pass loop)
  server/            HTTP handlers, chat templating, SSE streaming
  metrics/           Prometheus metrics
deploy/              Dockerfile, docker-compose, Prometheus & Grafana config
scripts/             Load-generator / benchmark helper
docs/                The guided tour + decision log
```

## Status

See the [decision log](docs/01-architecture-and-decisions.md) and each phase
doc for what is implemented and what is intentionally out of scope.
