# 09 — Deployment

> Code: `Dockerfile`, `.dockerignore`, `deploy/docker-compose.yml`,
> `deploy/prometheus/prometheus.yml`, `deploy/grafana/provisioning/**`

The goal here: one command brings up the server *and* a full observability stack
so you can watch the metrics from doc 08 move in real time.

## The container image

`Dockerfile` is a standard two-stage Go build:

1. **build stage** (`golang:1.25`): download modules (cached separately from
   source for fast rebuilds), then compile a **fully static binary** with
   `CGO_ENABLED=0`.
2. **runtime stage** (`gcr.io/distroless/static-debian12:nonroot`): copy in just
   the binary. No shell, no package manager, runs as non-root — a tiny image with
   a tiny attack surface.

Because the default (mock) engine has no native dependencies, the static build
"just works" and the final image is minimal. A real `llama.cpp` backend would
require `CGO_ENABLED=1` and a C toolchain in the build stage, and a runtime base
with the shared libs — one more concrete way the pluggable engine keeps the
common path simple.

> Verify the exact build the image uses, without a Docker daemon:
> ```
> CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /tmp/server ./cmd/server
> ```

## The Compose stack

```bash
docker compose -f deploy/docker-compose.yml up --build
```

Three services on one network:

| Service | Port | Role |
|---------|------|------|
| `server` | 8080 | the inference server (this project) |
| `prometheus` | 9090 | scrapes `server:8080/metrics` every 2s |
| `grafana` | 3000 | dashboards, Prometheus pre-wired |

The server is started with a little simulated latency (`STEP_BASE`,
`STEP_PER_SEQ`, `PREFILL_PER_TOKEN` env vars) so the graphs show realistic
curves as you send traffic.

## Zero-click observability (provisioning)

Everything is **provisioned** — configured from files on startup — so nothing
needs manual clicking:

- `deploy/prometheus/prometheus.yml` — one scrape job pointed at the server by
  its Compose service name (`server:8080`), which resolves on the shared network.
- `deploy/grafana/provisioning/datasources/datasource.yml` — registers
  Prometheus as Grafana's default datasource with a **fixed uid** (`prometheus`)
  so the dashboard can reference it deterministically.
- `deploy/grafana/provisioning/dashboards/` — a provider that loads
  `go-llm-server.json` on startup, set as Grafana's home dashboard.

Grafana runs with anonymous admin access (local demo only — never do this in
production) so the dashboard opens immediately at `http://localhost:3000`.

## The dashboard

`go-llm-server.json` is built entirely on the metric names from doc 08:

- **Aggregate throughput** — `rate(llm_tokens_generated_total[1m])`
- **TTFT p50/p95** — `histogram_quantile(…, llm_ttft_seconds_bucket)`
- **Request duration p50/p95**
- **Batch size** — the live `llm_active_batch_size` plus the p50 per-step batch
- **KV-cache utilization** and **prefix-cache hit rate** gauges
- **Queue depth**
- **Finish reasons & rejections** — `rate` of `llm_requests_finished_total` by
  `reason`, plus the 429 rate

Drive traffic with the load generator (`scripts/loadgen.go`, doc 10) and watch
batch size and throughput climb together while TTFT holds steady — continuous
batching, live.

## Seeing it work

```bash
# terminal 1
docker compose -f deploy/docker-compose.yml up --build
# terminal 2 — generate load
go run ./scripts/loadgen.go -url http://localhost:8080 -concurrency 32 -requests 2000
# browser
open http://localhost:3000   # dashboard
open http://localhost:9090   # raw Prometheus, if you want to poke at queries
```

## Toward Kubernetes

The brief mentions Kubernetes. The image is K8s-ready as-is; a Deployment would
add: a **readiness probe** on `/healthz` (already returns queue depth), a
**PodMonitor/ServiceMonitor** pointing at `/metrics`, resource
requests/limits, and — for a real GPU backend — a node selector/toleration for
GPU node pools and a `nvidia.com/gpu` resource request. Those manifests are left
as a natural next step rather than shipped half-finished.

## Decisions & trade-offs

- **distroless static base.** Smallest sensible, secure-by-default image for a
  CGO-free Go binary. If you switch to a CGO backend, move to a `debian:slim`
  runtime with the needed libs.
- **2s scrape interval.** Great for a live demo; in production you'd relax it to
  15–30s to cut storage and load.
- **Anonymous Grafana.** Convenience for local exploration only; explicitly
  called out as not for production.
- **Compose over raw `docker run`.** One file documents the whole topology and
  wires the network, which is itself part of the lesson.
