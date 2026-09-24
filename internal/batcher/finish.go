package batcher

import (
	"context"
	"errors"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
)

// errTooLarge is used when a request's KV requirement exceeds the entire cache
// budget, so it can never be admitted no matter how much frees up.
var errTooLarge = errors.New("request KV footprint exceeds cache capacity")

// finishReq completes a request exactly once: it emits the terminal stream
// event, closes the stream, releases the engine sequence and KV allocation,
// removes the request from the active set, and records completion metrics.
//
// This is the batcher's half of the "close-exactly-once" discipline (the
// scheduler owns the other half for requests it fails from the queue). Because
// only the single Run goroutine calls this, there is no risk of a double close.
func finishReq(ctx context.Context, b *Batcher, ar *activeReq, reason domain.FinishReason, err error) {
	// Release compute + memory and record metrics FIRST, so that by the time an
	// observer sees the terminal Done event, resources are already freed and the
	// completion metrics are already recorded. This ordering is what makes the
	// system's externally-visible state consistent the instant a request ends.
	if ar.seq != nil {
		b.eng.Release(ar.seq)
	}
	if ar.req.KVHandle != nil {
		if h, ok := ar.req.KVHandle.(*kvcache.Handle); ok {
			b.kv.Free(h)
		}
		ar.req.KVHandle = nil
	}
	delete(b.active, ar.req.ID)

	b.metrics.IncRequestsFinished(string(reason))
	dur := time.Since(ar.start)
	b.metrics.ObserveRequestDuration(dur.Seconds())
	if n := len(ar.req.Generated); n > 0 && dur > 0 {
		b.metrics.ObserveTokensPerSecond(float64(n) / dur.Seconds())
	}

	// Emit the terminal event, then close exactly once. Prefer a non-blocking
	// send so a still-listening (buffered) client always learns the finish
	// reason, even when its context is already canceled. Only if that can't
	// proceed do we fall back to a guarded blocking send that bails if the
	// client or server is truly gone.
	if ar.req.Stream != nil {
		ev := domain.StreamEvent{Done: true, Reason: reason, Err: err}
		select {
		case ar.req.Stream <- ev:
		default:
			select {
			case ar.req.Stream <- ev:
			case <-ar.req.Ctx.Done():
			case <-ctx.Done():
			}
		}
		close(ar.req.Stream)
	}
}
