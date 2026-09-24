# 05 — The scheduler

> Code: `internal/scheduler/scheduler.go`, `internal/scheduler/heap.go`,
> `internal/scheduler/scheduler_test.go`

The scheduler answers one question: **when does a request run?** It sits between
the API layer (produces requests) and the batcher (consumes them) and does three
things — admission control, prioritization, and timeouts.

## Admission control (backpressure)

A server with an unbounded queue is a trap. Under overload it accepts everything,
the queue grows without limit, and *every* request eventually times out — the
worst possible behaviour, because clients waited *and* got nothing.

The fix is **backpressure**: cap the queue depth and reject early when it's full.
`Submit` returns `ErrQueueFull` once the queue holds `MaxQueueDepth` requests,
and the API layer maps that to **HTTP 429 Too Many Requests**. A fast, honest
"no" lets clients retry elsewhere or back off. This is the same principle as a
bounded channel or a full thread pool — saying no is a feature.

## Prioritization: FCFS with a *bounded* short-prompt boost

Two instincts pull in opposite directions:

- **FCFS (first-come-first-served)** is *fair*: no request starves.
- **SPT (shortest-processing-time first)** minimizes *average* latency: serving
  cheap requests first gets more people answered sooner — but it can *starve*
  long requests forever if short ones keep arriving.

We want most of SPT's latency win without its starvation risk. The trick is a
**bounded** boost. Each request's priority starts as its arrival sequence number
(pure FCFS), and short prompts get a fixed bonus subtracted:

```go
priority = arrivalSeq
if len(promptTokens) <= ShortPromptThreshold {
    priority -= ShortPromptBoost   // e.g. 16
}
```

Because the bonus is a **constant**, a short prompt can overtake at most
`ShortPromptBoost` earlier arrivals — never more. So:

- short prompts get served sooner (better average latency), but
- a long prompt can be jumped by at most `ShortPromptBoost` requests, after which
  it *will* run. No starvation.

`TestShortPromptBoost` shows a short prompt jumping three long ones;
`TestBoostIsBounded` shows it *cannot* jump past the boost limit. That pair of
tests is the whole design in two examples.

Implementation: a `container/heap` min-heap on `(priority, arrivalSeq)`. The
`heap.go` file is a textbook example of Go's heap interface — implement `Len`,
`Less`, `Swap`, `Push`, `Pop` and you get an O(log n) priority queue.

## Timeouts: fail waiting requests proactively

The brief says "cancel if waiting > 30s." Doing this only *when a request
reaches the front* is too late — a client could wait 30s and then still get
served, or wait far longer if the queue is deep. So a **background reaper**
goroutine wakes every `ReapInterval`, scans the queue, and fails any request
older than `MaxWait` with `FinishCanceled`. `TryPop` also skips (and fails) any
expired request it happens to pop, as a belt-and-suspenders second check.

## How the batcher and scheduler talk

The batcher doesn't busy-poll. The scheduler exposes:

- `TryPop() (*Request, bool)` — non-blocking; take the next runnable request.
- `Ready() <-chan struct{}` — signalled on every `Submit`.

The batcher fills its free slots with `TryPop` in a loop, and when it has nothing
to do it blocks in a `select` on `Ready()` (plus a fallback tick). `signal()`
does a non-blocking send on a capacity-1 channel, which *coalesces* a burst of
submissions into a single wake-up — a common and important Go idiom (don't send
N wake-ups when one will do).

## The close-exactly-once discipline

Every request's `Stream` channel must be closed exactly once, by whoever
finishes the request:

- **Rejected at submit** (queue full): no `Stream` events; the handler gets an
  error and responds 429.
- **Timed out in the queue**: the *scheduler* sends a `Done{FinishCanceled}` and
  closes the stream (`fail`).
- **Actually run**: the *batcher* sends the terminal event and closes (doc 06).

The handler only ever *reads* the stream; it never closes it. Keeping ownership
of the close unambiguous is what prevents double-close panics and lost events in
concurrent Go — worth internalizing as a general pattern.

## Decisions & trade-offs

- **Bounded boost instead of aging.** Another classic anti-starvation technique
  is *aging* (a request's priority improves the longer it waits). Bounded boost
  is simpler to reason about and to test, and gives a hard starvation guarantee
  by construction. Aging is noted as a natural alternative.
- **Reaper rebuilds the heap on eviction.** Removing arbitrary elements from a
  binary heap is awkward; the reaper filters the slice and calls `heap.Init`
  (O(n)). Fine at these queue sizes; a production system with huge queues might
  use an indexed heap or lazy deletion.
- **Priority computed at submit, not re-evaluated.** Simpler and stable. Since
  the boost already bounds jumping, there's no need to recompute priorities as
  the queue evolves.
