# go-llm-server

[![CI](https://github.com/1morshed1/go-llm-server/actions/workflows/ci.yml/badge.svg)](https://github.com/1morshed1/go-llm-server/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch **LLM inference server** written in Go — a teaching-grade,
simplified version of what [vLLM](https://github.com/vllm-project/vllm),
[Ollama](https://github.com/ollama/ollama), and
[TGI](https://github.com/huggingface/text-generation-inference) do.

It implements the *serving layer* that sits between a model and the HTTP API
your applications call:

- **OpenAI-compatible REST API** (`POST /v1/chat/completions`, streaming + non-streaming)
- **gRPC streaming API** (a second front door onto the same scheduler)
- **Request scheduler** with a priority queue, admission control, and per-request timeouts
- **Continuous batching** (the vLLM trick that gives 2–3× throughput over naive batching)
- **KV-cache manager** with **prefix caching** (reuse the KV of shared system prompts)
- **Pluggable inference engine** behind a single Go interface
- **Prometheus metrics** + a **Grafana dashboard**
- **Docker Compose** stack (server + Prometheus + Grafana)

Measured on the simulated engine (`make bench`): **8.75× throughput** scaling to
batch 16, **3.21× continuous vs. naive batching** on mixed traffic, and a
**98.4% prefix-cache hit rate** cutting average TTFT by **26%**. See
[docs/10](docs/10-benchmarks.md).

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
| [11 — The gRPC API](docs/11-grpc.md) | A second transport, and what it proves about the design |
| [12 — Wrap-up & extensions](docs/12-wrap-up-and-extensions.md) | How it maps to real systems, and what to build next |

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

The gRPC API listens on `:50051` (disable with `--grpc-addr ""`); the contract is
`api/proto/inference.proto`. See [docs/11](docs/11-grpc.md).

## Project layout

```
cmd/server/          Entry point: wiring and graceful shutdown
api/proto/           gRPC service definition (inference.proto)
internal/
  domain/            Core types shared by every layer (no transport code)
  config/            Configuration (flags + environment variables)
  tokenizer/         Tokenizer interface + a simple word-piece tokenizer
  engine/            Engine interface + the simulated backend
  kvcache/           KV-cache manager + prefix caching
  scheduler/         Priority queue, admission control, timeouts
  batcher/           Continuous batcher (the forward-pass loop)
  prompt/            Shared chat templating, validation, request building
  server/            HTTP handlers, OpenAI compatibility, SSE streaming
  grpcapi/           gRPC streaming server (+ generated pb/)
  metrics/           Prometheus metrics
deploy/              Dockerfile, docker-compose, Prometheus & Grafana config
scripts/bench/       In-process benchmark suite
scripts/loadgen/     HTTP load generator
docs/                The guided tour + decision log
```

## Testing & tooling

```bash
make test     # run all tests
make race     # tests under the race detector
make bench    # in-process benchmark suite (docs/10)
make check    # fmt + vet + race — run before committing
make help     # list all targets
```

## Status

The serving layer is complete end-to-end: REST + gRPC APIs, scheduling,
continuous batching, KV cache with prefix caching, metrics, and a deployable
stack, all covered by tests and benchmarks. What's intentionally deferred — a
real `llama.cpp` backend, a multi-model manager/hot-swap, chunked prefill,
preemption, auth, and Kubernetes manifests — is catalogued with implementation
notes in [docs/12](docs/12-wrap-up-and-extensions.md). The full rationale for
every design choice lives in the [decision log](docs/01-architecture-and-decisions.md).

## License

[MIT](LICENSE)
