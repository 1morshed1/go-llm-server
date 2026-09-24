// Package scheduler decides WHEN a request runs.
//
// It sits between the API layer (which produces requests) and the batcher
// (which consumes them). Its three jobs map onto three classic systems concerns:
//
//   - ADMISSION CONTROL (backpressure): if the queue is already too deep,
//     reject new work immediately with a clear error instead of accepting an
//     unbounded backlog. A server that says "no" fast is far more useful than
//     one that quietly builds a 10,000-deep queue and times everything out.
//
//   - PRIORITIZATION: order the waiting requests. We serve earlier arrivals
//     first (FCFS, which is fair), but give short prompts a BOUNDED head start
//     (they're cheap to serve, so letting them jump a little improves average
//     latency without starving long prompts). See the priority discussion below.
//
//   - TIMEOUTS: a request that has waited too long is failed proactively, so a
//     client never waits 30s only to then get served. A background reaper
//     enforces this.
//
// See docs/05-scheduler.md for the full discussion.
package scheduler

import (
	"container/heap"
	"errors"
	"sync"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
)

// ErrQueueFull is returned by Submit when admission control rejects a request
// because the queue is at capacity. The API layer maps this to HTTP 429.
var ErrQueueFull = errors.New("scheduler: queue full")

// Config tunes the scheduler.
type Config struct {
	// MaxQueueDepth is the admission-control threshold. Submit rejects when the
	// queue already holds this many requests.
	MaxQueueDepth int
	// MaxWait is how long a request may sit in the queue before it is failed
	// with FinishCanceled. (The brief's "cancel if waiting > 30s".)
	MaxWait time.Duration
	// ShortPromptThreshold: prompts with at most this many tokens are "short"
	// and get the priority boost.
	ShortPromptThreshold int
	// ShortPromptBoost bounds how far a short prompt may jump ahead of earlier
	// arrivals. A short prompt can overtake at most this many already-queued
	// requests — which is why the boost speeds up cheap work without starving
	// expensive work. See docs/05.
	ShortPromptBoost int
	// ReapInterval is how often the background reaper scans for timed-out
	// requests.
	ReapInterval time.Duration
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		MaxQueueDepth:        256,
		MaxWait:              30 * time.Second,
		ShortPromptThreshold: 32,
		ShortPromptBoost:     16,
		ReapInterval:         time.Second,
	}
}

// Scheduler is a priority queue with admission control and timeouts.
type Scheduler struct {
	cfg Config

	mu    sync.Mutex
	pq    priorityQueue
	seq   int64 // monotonic arrival counter, used for FCFS tie-breaking
	notify chan struct{}
	closed bool

	stopReaper chan struct{}
}

// New builds a Scheduler and starts its reaper goroutine.
func New(cfg Config) *Scheduler {
	if cfg.MaxQueueDepth <= 0 {
		cfg = DefaultConfig()
	}
	s := &Scheduler{
		cfg:        cfg,
		notify:     make(chan struct{}, 1),
		stopReaper: make(chan struct{}),
	}
	heap.Init(&s.pq)
	go s.reapLoop()
	return s
}

// Submit applies admission control, assigns a priority, and enqueues the
// request. It returns ErrQueueFull if the queue is at capacity.
func (s *Scheduler) Submit(req *domain.Request) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return errors.New("scheduler: closed")
	}
	if s.pq.Len() >= s.cfg.MaxQueueDepth {
		return ErrQueueFull
	}

	req.EnqueuedAt = time.Now()
	req.Priority = s.priorityFor(req)

	heap.Push(&s.pq, &item{req: req, seq: s.seq})
	s.seq++
	s.signal()
	return nil
}

// priorityFor computes the ordering key: lower runs sooner. We start from the
// arrival sequence number (pure FCFS) and subtract a bounded boost for short
// prompts. Because the boost is bounded by ShortPromptBoost, a short prompt can
// overtake at most that many earlier requests — bounded queue-jumping, never
// starvation. Caller holds s.mu (uses s.seq).
func (s *Scheduler) priorityFor(req *domain.Request) int {
	base := int(s.seq) // arrival order
	if len(req.PromptTokens) <= s.cfg.ShortPromptThreshold {
		base -= s.cfg.ShortPromptBoost
	}
	return base
}

// TryPop removes and returns the highest-priority request that has NOT timed
// out. Timed-out requests encountered along the way are failed and skipped.
// Returns (nil, false) when the queue is empty.
func (s *Scheduler) TryPop() (*domain.Request, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.pq.Len() > 0 {
		it := heap.Pop(&s.pq).(*item)
		if s.expired(it.req) {
			// Fail it and keep looking for a live request.
			go fail(it.req, domain.FinishCanceled, nil)
			continue
		}
		return it.req, true
	}
	return nil, false
}

// Len returns the current queue depth (for the queue-depth metric).
func (s *Scheduler) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pq.Len()
}

// Ready returns a channel that is signalled whenever a request is enqueued. The
// batcher selects on it to wake up promptly instead of busy-polling.
func (s *Scheduler) Ready() <-chan struct{} { return s.notify }

// Close stops the reaper and fails everything still queued.
func (s *Scheduler) Close() {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	close(s.stopReaper)
	drained := make([]*domain.Request, 0, s.pq.Len())
	for s.pq.Len() > 0 {
		drained = append(drained, heap.Pop(&s.pq).(*item).req)
	}
	s.mu.Unlock()
	for _, r := range drained {
		fail(r, domain.FinishCanceled, nil)
	}
}

// --- internals ---

func (s *Scheduler) expired(req *domain.Request) bool {
	return time.Since(req.EnqueuedAt) > s.cfg.MaxWait
}

// signal does a non-blocking send on notify (cap 1), coalescing bursts.
func (s *Scheduler) signal() {
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

// reapLoop periodically fails requests that have waited past MaxWait, so slow
// clients don't discover their timeout only when they finally reach the front.
func (s *Scheduler) reapLoop() {
	t := time.NewTicker(s.cfg.ReapInterval)
	defer t.Stop()
	for {
		select {
		case <-s.stopReaper:
			return
		case <-t.C:
			s.reapExpired()
		}
	}
}

func (s *Scheduler) reapExpired() {
	s.mu.Lock()
	if s.pq.Len() == 0 {
		s.mu.Unlock()
		return
	}
	var expired []*domain.Request
	kept := s.pq[:0:0] // new backing array; we rebuild the heap
	for _, it := range s.pq {
		if s.expired(it.req) {
			expired = append(expired, it.req)
		} else {
			kept = append(kept, it)
		}
	}
	if len(expired) > 0 {
		s.pq = kept
		heap.Init(&s.pq)
	}
	s.mu.Unlock()

	for _, r := range expired {
		fail(r, domain.FinishCanceled, nil)
	}
}

// fail sends a terminal event on the request's stream and closes it. This is
// the scheduler's way of completing a request it will never run. The batcher
// uses the same close-exactly-once discipline for requests it does run.
func fail(req *domain.Request, reason domain.FinishReason, err error) {
	if req.Stream == nil {
		return
	}
	// Best-effort terminal event, then close. Guard against a canceled reader
	// by also selecting on the request context.
	select {
	case req.Stream <- domain.StreamEvent{Done: true, Reason: reason, Err: err}:
	case <-req.Ctx.Done():
	}
	close(req.Stream)
}
