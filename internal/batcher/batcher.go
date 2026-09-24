// Package batcher implements CONTINUOUS BATCHING — the technique that makes an
// LLM server efficient, and the centerpiece of this project.
//
// THE PROBLEM WITH NAIVE BATCHING
//
// To use a GPU well you must run many sequences per forward pass. The naive
// way: collect N requests, run them together until ALL finish, then return.
// The flaw: requests finish at different times. If request A wants 20 tokens and
// B wants 500, A is stuck in the batch for 480 extra steps doing nothing, and no
// new request can start until the whole batch drains. Utilization collapses.
//
// CONTINUOUS BATCHING (a.k.a. iteration-level scheduling)
//
// Instead of batching whole requests, we batch a single STEP. Each iteration:
//
//   1. top up the batch from the queue (new requests JOIN mid-flight),
//   2. run ONE decode step across all active sequences (each advances 1 token),
//   3. any sequence that finished this step LEAVES immediately, freeing its KV
//      and its slot for the next queued request.
//
// So a finished request never blocks the others, and a new request never waits
// for the batch to drain. This is how vLLM/TGI get 2–3× the throughput of naive
// batching at the same latency. docs/06-continuous-batching.md walks this file
// line by line.
package batcher

import (
	"context"
	"strings"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
	"github.com/1morshed1/go-llm-server/internal/metrics"
	"github.com/1morshed1/go-llm-server/internal/scheduler"
)

// Config tunes the batcher.
type Config struct {
	// MaxBatchSize caps how many sequences decode concurrently. In a real server
	// this is bounded by KV-cache memory; here it's an explicit ceiling on top of
	// the cache limit.
	MaxBatchSize int
	// IdleTick is the fallback wake interval when the batch is empty (belt and
	// suspenders alongside the scheduler's Ready() signal).
	IdleTick time.Duration
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{MaxBatchSize: 16, IdleTick: 50 * time.Millisecond}
}

// activeReq is a request currently being decoded, plus the bookkeeping the
// batcher needs: its engine sequence, when it started (for TTFT and duration),
// whether we've emitted its first token yet, and the generated text so far (for
// stop-string detection).
type activeReq struct {
	req       *domain.Request
	seq       *engine.Sequence
	start     time.Time
	firstSent bool
	genText   strings.Builder
}

// Batcher owns the decode loop. All of its mutable state (the active map, the
// held request) is touched ONLY by the single goroutine running Run, so it needs
// no locks of its own — a deliberate, simplifying design choice (see docs/06).
type Batcher struct {
	cfg     Config
	eng     engine.Engine
	kv      *kvcache.Manager
	sched   *scheduler.Scheduler
	metrics metrics.Recorder

	active map[string]*activeReq
	held   *domain.Request // a popped request awaiting KV capacity (see admit)
}

// New builds a Batcher. If rec is nil, metrics are discarded.
func New(cfg Config, eng engine.Engine, kv *kvcache.Manager, sched *scheduler.Scheduler, rec metrics.Recorder) *Batcher {
	if cfg.MaxBatchSize <= 0 {
		cfg = DefaultConfig()
	}
	if rec == nil {
		rec = metrics.Noop{}
	}
	return &Batcher{
		cfg:     cfg,
		eng:     eng,
		kv:      kv,
		sched:   sched,
		metrics: rec,
		active:  make(map[string]*activeReq),
	}
}

// Run drives the decode loop until ctx is cancelled. It returns after failing
// any still-active requests as canceled, so callers can wait for a clean drain.
func (b *Batcher) Run(ctx context.Context) {
	idle := time.NewTicker(b.cfg.IdleTick)
	defer idle.Stop()

	for {
		if ctx.Err() != nil {
			b.drainCanceled()
			return
		}

		b.admitNew(ctx)

		// Nothing to do? Block until a request arrives, the fallback tick fires,
		// or we're told to shut down. This is what keeps an idle server from
		// burning a CPU core spinning.
		if len(b.active) == 0 {
			b.publishGauges()
			select {
			case <-ctx.Done():
				b.drainCanceled()
				return
			case <-b.sched.Ready():
			case <-idle.C:
			}
			continue
		}

		b.step(ctx)
		b.publishGauges()
	}
}

// admitNew fills empty batch slots from the scheduler. A request that can't get
// KV capacity right now is "held" and retried next iteration (before popping
// new ones), so we never lose its place or spin allocating.
func (b *Batcher) admitNew(ctx context.Context) {
	// First, retry a previously held request.
	if b.held != nil {
		if b.tryAdmit(ctx, b.held) {
			b.held = nil
		} else if len(b.active) == 0 {
			// The batch is empty yet this request still won't fit — it is larger
			// than the entire KV budget and can never run. Fail it clearly rather
			// than looping forever.
			finishReq(ctx, b, &activeReq{req: b.held, start: time.Now()}, domain.FinishCanceled, errTooLarge)
			b.held = nil
		} else {
			return // hold; free some memory by decoding first
		}
	}

	for len(b.active) < b.cfg.MaxBatchSize {
		req, ok := b.sched.TryPop()
		if !ok {
			return
		}
		if !b.tryAdmit(ctx, req) {
			b.held = req // out of capacity; retry this exact request next round
			return
		}
	}
}

// tryAdmit attempts to move one request into the active batch: reserve KV
// (maybe reusing a cached prefix), run prefill (cost ∝ uncached tokens), and
// register it. Returns false ONLY when KV capacity is unavailable (so the
// caller can hold and retry). A request cancelled before/while prefilling is
// finished here and counts as "admitted" (handled), returning true.
func (b *Batcher) tryAdmit(ctx context.Context, req *domain.Request) bool {
	// Client already gone before we even started? Finish immediately.
	if req.Ctx.Err() != nil {
		finishReq(ctx, b, &activeReq{req: req, start: time.Now()}, domain.FinishCanceled, nil)
		return true
	}

	handle, err := b.kv.Allocate(req)
	if err != nil {
		return false // ErrNoCapacity — caller will hold and retry
	}
	req.KVHandle = handle

	uncached := len(req.PromptTokens) - handle.CachedPrefixLen
	ar := &activeReq{req: req, start: time.Now()}
	seq, err := b.eng.Prefill(req.Ctx, req, uncached)
	if err != nil {
		// Context cancelled during prefill: release and finish.
		b.kv.Free(handle)
		finishReq(ctx, b, ar, domain.FinishCanceled, nil)
		return true
	}
	ar.seq = seq
	b.active[req.ID] = ar
	return true
}

// step runs ONE decode iteration across the whole active batch.
func (b *Batcher) step(ctx context.Context) {
	// Snapshot the active requests in a stable order so results line up.
	ars := make([]*activeReq, 0, len(b.active))
	seqs := make([]*engine.Sequence, 0, len(b.active))
	for _, ar := range b.active {
		// Drop clients that disconnected since the last step, before spending a
		// forward pass on them.
		if ar.req.Ctx.Err() != nil {
			finishReq(ctx, b, ar, domain.FinishCanceled, nil)
			continue
		}
		ars = append(ars, ar)
		seqs = append(seqs, ar.seq)
	}
	if len(seqs) == 0 {
		return
	}

	b.metrics.ObserveBatchSize(len(seqs))

	results, err := b.eng.Step(ctx, seqs)
	if err != nil {
		// Only the run context cancels Step; treat as shutdown.
		return
	}

	for i, res := range results {
		ar := ars[i]

		// The engine produced a token. Emit it unless the model signalled EOS
		// (whose token is empty and shouldn't be streamed).
		if !res.EOS {
			ar.req.Generated = append(ar.req.Generated, res.Token)
			ar.genText.WriteString(res.Token.Text)

			if !b.stream(ctx, ar, domain.StreamEvent{Token: res.Token}) {
				// Client went away mid-stream.
				finishReq(ctx, b, ar, domain.FinishCanceled, nil)
				continue
			}
			if !ar.firstSent {
				ar.firstSent = true
				b.metrics.ObserveTTFT(time.Since(ar.start).Seconds())
			}
			b.metrics.IncTokensGenerated(1)
		}

		// Decide whether this request is done. Policy lives HERE, not in the
		// engine: the engine reports its own EOS; the server layers MaxTokens
		// and stop-strings on top.
		if reason, done := b.finishReason(ar, res.EOS); done {
			finishReq(ctx, b, ar, reason, nil)
		}
	}
}

// finishReason applies serving policy to decide if generation should stop.
func (b *Batcher) finishReason(ar *activeReq, modelEOS bool) (domain.FinishReason, bool) {
	if modelEOS {
		return domain.FinishStop, true
	}
	if ar.req.Params.MaxTokens > 0 && len(ar.req.Generated) >= ar.req.Params.MaxTokens {
		return domain.FinishLength, true
	}
	for _, s := range ar.req.Params.Stop {
		if s != "" && strings.Contains(ar.genText.String(), s) {
			return domain.FinishStop, true
		}
	}
	return "", false
}

// stream pushes an event to the request's channel, but never blocks forever: it
// gives up if the client disconnects or the server shuts down. Returns false if
// the event could not be delivered (client/server gone).
func (b *Batcher) stream(ctx context.Context, ar *activeReq, ev domain.StreamEvent) bool {
	select {
	case ar.req.Stream <- ev:
		return true
	case <-ar.req.Ctx.Done():
		return false
	case <-ctx.Done():
		return false
	}
}

// publishGauges refreshes the point-in-time gauges each loop iteration.
func (b *Batcher) publishGauges() {
	b.metrics.SetQueueDepth(b.sched.Len())
	b.metrics.SetActiveBatch(len(b.active))
	st := b.kv.Stats()
	b.metrics.SetKVCacheUtilization(st.Utilization)
	b.metrics.SetPrefixHitRate(st.HitRate)
}

// drainCanceled fails every still-active request on shutdown.
func (b *Batcher) drainCanceled() {
	ctx := context.Background()
	for _, ar := range b.active {
		finishReq(ctx, b, ar, domain.FinishCanceled, nil)
	}
	if b.held != nil {
		finishReq(ctx, b, &activeReq{req: b.held, start: time.Now()}, domain.FinishCanceled, nil)
		b.held = nil
	}
}
