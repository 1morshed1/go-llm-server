# 06 — Continuous batching

> Code: `internal/batcher/batcher.go`, `internal/batcher/finish.go`,
> `internal/batcher/batcher_test.go`

This is the centerpiece. If you understand this doc, you understand the single
most important idea in modern LLM serving.

## Why batch at all?

A GPU is a wide parallel machine. Running one sequence through a decode step uses
a sliver of it; running 16 through the *same* step costs almost the same
wall-clock time but produces 16× the tokens. So **throughput is all about how
many sequences you can keep in each forward pass.** Our mock models this exactly:
`Step` latency is `StepBase + StepPerSeq × batchSize`, with `StepPerSeq ≪
StepBase`, so per-token cost plummets as the batch fills.

## Naive batching and why it's bad

The obvious way to batch: gather N requests, run them together until *all* N
finish, then return and gather the next N.

```
batch = [A(needs 20 tok), B(needs 500 tok), C(needs 30 tok)]
        └─ A done at step 20, but must wait...
        └─ C done at step 30, but must wait...
        └─ B finally done at step 500 → NOW the batch returns
```

Two killers:
1. **Head-of-line blocking**: A and C sit idle for hundreds of steps.
2. **No mid-flight admission**: a new request D can't start until the whole
   batch drains at step 500.

GPU utilization and latency both suffer.

## Continuous batching: batch the *step*, not the *request*

The fix (vLLM's "iteration-level scheduling", also in TGI/Ollama): don't tie the
batch's lifetime to whole requests. Each **iteration**:

1. **Top up** the batch from the queue — new requests JOIN immediately.
2. **One step** advances *every* active sequence by one token.
3. Any sequence that finished this step **LEAVES right away**, freeing its KV
   block and its slot.

```
step 20:  A finishes → leaves → D joins in its slot
step 30:  C finishes → leaves → E joins
step 500: B finishes → leaves
          (D, E, ... have been generating the whole time, not waiting)
```

No request waits for another to finish. The batch is always as full as the queue
and KV memory allow. That's where the 2–3× throughput comes from.

## Reading `batcher.go`

The loop in `Run` is the whole story:

```
for {
    admitNew(ctx)                 // step 1: fill empty slots (new requests join)
    if len(active) == 0 {         // nothing to do → block, don't spin
        select { Ready | idleTick | ctx.Done }
        continue
    }
    step(ctx)                     // step 2+3: one forward pass; finishers leave
    publishGauges()
}
```

### `admitNew` / `tryAdmit` — joining the batch

To admit a request we must reserve KV memory and run prefill:

- `kv.Allocate(req)` may **reuse a cached prefix** (returns `CachedPrefixLen`),
  or fail with `ErrNoCapacity` when memory is full.
- On `ErrNoCapacity` we **hold** the popped request and retry it next iteration
  *before* popping new ones — so it keeps its place and we don't busy-loop
  allocating. If the batch is empty and it *still* won't fit, it's larger than
  the entire KV budget and can never run, so we fail it with a clear error.
- `Engine.Prefill(req, uncached)` pays the prompt-processing cost (skipping the
  cached prefix). The first token comes from the first `step`, so **TTFT =
  prefill + first step**.

### `step` — one forward pass

- Build parallel slices of `activeReq` and `Sequence` in a stable order (so
  `results[i]` maps back to `ars[i]`).
- Drop any request whose client disconnected *before* spending a forward pass on
  it (`req.Ctx.Err()`).
- `eng.Step(ctx, seqs)` advances all sequences by one token.
- For each result: append+stream the token (record TTFT on the first), then ask
  `finishReason` whether to stop.

### Policy lives in the batcher, not the engine

`finishReason` is where **serving policy** is applied, deliberately separate from
the model:

- model's own EOS → `stop`
- reached `MaxTokens` → `length`
- generated text contains a stop string → `stop`

The engine only ever reports *its* EOS; the server layers `MaxTokens` and stop
strings on top. Either can change without touching the other.

### `finishReq` — leaving the batch, exactly once

When a request ends (any reason), `finishReq`:
1. releases the engine sequence and **frees its KV blocks** (so the slot and
   memory are immediately reusable — this is what lets a new request join),
2. records completion metrics,
3. emits the terminal `Done` event and **closes the stream exactly once**.

We do the frees and metrics *before* emitting `Done`, so the moment a client
sees the request end, the server's state (freed memory, updated counters) is
already consistent. The terminal send prefers a non-blocking send so a buffered
client always learns the finish reason, even after canceling.

## Concurrency model: one goroutine owns the batch

All the batcher's mutable state — the `active` map, the `held` request — is
touched **only by the single goroutine running `Run`.** No mutex. This is a
deliberate design choice: instead of sharing the batch across goroutines behind
locks, we confine it to one goroutine and communicate with the outside world
through channels (the scheduler's queue in, each request's `Stream` out) and
through internally-locked components (`kvcache`, `scheduler`, `metrics`). That's
the Go way — *"do not communicate by sharing memory; share memory by
communicating"* — and it makes the hottest, most intricate loop in the system
lock-free and easy to reason about. `TestManyConcurrentRequestsAllComplete` runs
40 requests through it under `-race` to prove it.

## Decisions & trade-offs

- **Prefill is paid synchronously at admission ("prefill-priority").** While we
  prefill a joining request, active requests aren't decoding that instant. Real
  servers do *chunked prefill* to interleave the two. We chose the simpler model
  and documented it; mock prefill is short, so the effect is small.
- **One token per sequence per step.** Real engines can do speculative decoding
  (multiple tokens/step). Out of scope; the interface wouldn't need to change
  much.
- **`MaxBatchSize` on top of the KV limit.** Two independent ceilings: an
  explicit batch cap and the (usually tighter) memory cap. Either can bind first.
- **Held-request backpressure instead of re-queueing.** When memory is full we
  hold the one popped request rather than pushing it back (which would disturb
  ordering and could hit a full queue). Simple and preserves fairness.
