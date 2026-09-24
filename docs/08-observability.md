# 08 — Observability

> Code: `internal/metrics/recorder.go`, `internal/metrics/prometheus.go`

You can't operate what you can't see. An inference server has a handful of
metrics that tell you almost everything about its health; this doc explains each
one, *why it matters*, and the design of the metrics seam.

## The seam: an interface, not a global

The hot-path code (the batcher) never imports Prometheus. It calls a small
`Recorder` interface. Two implementations exist:

- `Noop` — discards everything (tests, or running without a backend);
- `Prometheus` — the real thing, registered into its **own registry** (not the
  global default) so multiple instances can coexist without "duplicate metric"
  panics — which is exactly what lets tests build a real recorder freely.

This is the same dependency-inversion move as the engine interface: the code that
*emits* measurements doesn't depend on the code that *stores* them.

## The metrics that matter

### Throughput
- **`llm_tokens_generated_total`** (counter) — every output token. `rate()` of
  this in Grafana is your aggregate tokens/sec, the headline throughput number.
- **`llm_tokens_per_second`** (histogram) — per-request decode throughput, so you
  can see the *distribution*, not just the average.

### Latency
- **`llm_ttft_seconds`** (histogram) — **Time To First Token**, the latency a user
  actually feels before text starts appearing. The brief's target is <200 ms for
  a 7B model. Prefill dominates it, which is why prefix caching (doc 04) moves
  this number. Watch the p95/p99, not the mean.
- **`llm_request_duration_seconds`** (histogram) — end-to-end generation time.

### Saturation (is the server full?)
- **`llm_queue_depth`** (gauge) — requests waiting. Sustained growth means you're
  overloaded; this is what admission control caps.
- **`llm_active_batch_size`** (gauge) + **`llm_batch_size`** (histogram) — how
  many sequences decode together. If this stays near 1 under load, continuous
  batching isn't helping and something upstream is serializing.
- **`llm_kv_cache_utilization_ratio`** (gauge) — fraction of KV memory in use.
  This is the real capacity limit; when it pins near 1.0, new requests can't be
  admitted no matter how idle the compute is. **The single most important
  saturation signal in LLM serving.**

### Efficiency
- **`llm_prefix_cache_hit_rate`** (gauge) — fraction of prefix lookups that hit.
  High hit rate ⇒ big TTFT and memory savings on shared system prompts.

### Traffic & errors
- **`llm_requests_received_total`**, **`llm_requests_rejected_total`**,
  **`llm_requests_finished_total{reason}`** — volume, how much you're shedding
  (429s), and *how* requests end (`stop` / `length` / `canceled`). A rising
  `canceled` share often means clients are giving up — a latency problem in
  disguise.

## Conventions we followed (and why)

Prometheus has strong naming conventions, and following them makes your metrics
work with the whole ecosystem's tooling and dashboards:

- a common **`llm_` prefix** namespaces everything;
- **base units** — seconds, not milliseconds; a ratio in [0,1], not a percent
  (Grafana multiplies by 100 for display). This is why the gauge is
  `..._utilization_ratio`, a deliberate deviation from the brief's
  `..._percent` name in favour of convention;
- **`_total` suffix** on counters;
- **labels used sparingly** — only `reason`, with three fixed values. High-
  cardinality labels (like per-request IDs) would explode memory; we avoid them.

Histograms carry hand-picked buckets matched to the expected ranges (TTFT in the
tens-of-ms to seconds, batch size 1..16), because a histogram is only as useful
as its buckets.

## Where the numbers come from

- **Counters/histograms** are emitted by the batcher as things happen
  (`ObserveTTFT` on the first token, `IncTokensGenerated` per token,
  `IncRequestsFinished` at completion).
- **Gauges** are point-in-time and refreshed every loop iteration by
  `publishGauges`, which reads live state from the scheduler (`Len`) and the KV
  manager (`Stats`). Pulling gauges from the authoritative source each tick keeps
  them honest.

The `/metrics` endpoint (wired in `main`, mounted by the server) exposes all of
this in Prometheus text format; [doc 09](09-deployment.md) points Prometheus at
it and ships a Grafana dashboard built on these exact names.
