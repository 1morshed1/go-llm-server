package batcher

import (
	"context"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
	"github.com/1morshed1/go-llm-server/internal/scheduler"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// countingRecorder captures a few metrics so tests can assert on behaviour like
// "continuous batching actually engaged" (max observed batch size > 1).
type countingRecorder struct {
	mu           sync.Mutex
	maxBatch     int
	tokens       int
	finished     map[string]int
	ttftObserved int
}

func newRecorder() *countingRecorder { return &countingRecorder{finished: map[string]int{}} }

func (r *countingRecorder) ObserveBatchSize(n int) {
	r.mu.Lock()
	if n > r.maxBatch {
		r.maxBatch = n
	}
	r.mu.Unlock()
}
func (r *countingRecorder) IncTokensGenerated(n int) { r.mu.Lock(); r.tokens += n; r.mu.Unlock() }
func (r *countingRecorder) IncRequestsFinished(reason string) {
	r.mu.Lock()
	r.finished[reason]++
	r.mu.Unlock()
}
func (r *countingRecorder) ObserveTTFT(float64)            { r.mu.Lock(); r.ttftObserved++; r.mu.Unlock() }
func (r *countingRecorder) ObserveRequestDuration(float64) {}
func (r *countingRecorder) ObserveTokensPerSecond(float64) {}
func (r *countingRecorder) IncRequestsRejected()           {}
func (r *countingRecorder) IncRequestsReceived()           {}
func (r *countingRecorder) SetQueueDepth(int)              {}
func (r *countingRecorder) SetActiveBatch(int)             {}
func (r *countingRecorder) SetKVCacheUtilization(float64)  {}
func (r *countingRecorder) SetPrefixHitRate(float64)       {}

// locked accessors for use from the test goroutine.
func (r *countingRecorder) getMaxBatch() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.maxBatch
}
func (r *countingRecorder) getFinished(reason string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.finished[reason]
}

// stack bundles a fully wired serving pipeline for tests.
type stack struct {
	tok    tokenizer.Tokenizer
	sched  *scheduler.Scheduler
	kv     *kvcache.Manager
	rec    *countingRecorder
	cancel context.CancelFunc
	done   chan struct{}
}

func newStack(t *testing.T, mockCfg engine.MockConfig, batchCfg Config) *stack {
	t.Helper()
	tok := tokenizer.NewSimple(32000)
	eng := engine.NewMock(mockCfg, tok)
	kv := kvcache.New(kvcache.DefaultConfig())
	sched := scheduler.New(scheduler.DefaultConfig())
	rec := newRecorder()
	b := New(batchCfg, eng, kv, sched, rec)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { b.Run(ctx); close(done) }()

	s := &stack{tok: tok, sched: sched, kv: kv, rec: rec, cancel: cancel, done: done}
	t.Cleanup(func() {
		cancel()
		<-done
		sched.Close()
	})
	return s
}

func (s *stack) submit(id, prompt string, maxTokens int, stop []string) *domain.Request {
	toks := s.tok.Encode(prompt)
	req := &domain.Request{
		ID:           id,
		Model:        "mock-7b",
		Prompt:       prompt,
		PromptTokens: toks,
		Params:       domain.SamplingParams{MaxTokens: maxTokens, Stop: stop},
		Ctx:          context.Background(),
		Stream:       make(chan domain.StreamEvent, maxTokens+8),
	}
	if err := s.sched.Submit(req); err != nil {
		panic(err)
	}
	return req
}

// collect drains a request's stream and returns the generated text tokens and
// the finish reason.
func collect(req *domain.Request) (tokens []string, reason domain.FinishReason) {
	for ev := range req.Stream {
		if ev.Done {
			return tokens, ev.Reason
		}
		tokens = append(tokens, ev.Token.Text)
	}
	return tokens, reason
}

func fastMock() engine.MockConfig {
	c := engine.DefaultMockConfig()
	c.PrefillPerToken = 0
	c.StepBase = 0
	c.StepPerSeq = 0
	c.MinNewTokens = 6
	c.MaxNewTokens = 10
	return c
}

func TestSingleRequestCompletes(t *testing.T) {
	s := newStack(t, fastMock(), DefaultConfig())
	req := s.submit("r1", "Hello there", 100, nil)
	tokens, reason := collect(req)
	if reason != domain.FinishStop {
		t.Errorf("reason=%q, want stop", reason)
	}
	if len(tokens) < 6 || len(tokens) > 10 {
		t.Errorf("generated %d tokens, want within [6,10]", len(tokens))
	}
}

func TestMaxTokensEnforced(t *testing.T) {
	c := fastMock()
	c.MinNewTokens = 100 // model wants to generate a lot...
	c.MaxNewTokens = 200
	s := newStack(t, c, DefaultConfig())
	req := s.submit("r1", "Hello", 3, nil) // ...but the server caps at 3
	tokens, reason := collect(req)
	if reason != domain.FinishLength {
		t.Errorf("reason=%q, want length", reason)
	}
	if len(tokens) != 3 {
		t.Errorf("generated %d tokens, want exactly 3", len(tokens))
	}
}

func TestManyConcurrentRequestsAllComplete(t *testing.T) {
	s := newStack(t, fastMock(), DefaultConfig())
	const n = 40
	reqs := make([]*domain.Request, n)
	for i := 0; i < n; i++ {
		reqs[i] = s.submit("r"+strconv.Itoa(i), "prompt "+strconv.Itoa(i), 100, nil)
	}
	var wg sync.WaitGroup
	for _, req := range reqs {
		wg.Add(1)
		go func(r *domain.Request) {
			defer wg.Done()
			_, reason := collect(r)
			if reason != domain.FinishStop {
				t.Errorf("req %s reason=%q, want stop", r.ID, reason)
			}
		}(req)
	}
	wg.Wait()

	// The whole point: many requests decoded together, so the batch grew past 1.
	if s.rec.getMaxBatch() <= 1 {
		t.Errorf("continuous batching did not engage: maxBatch=%d", s.rec.getMaxBatch())
	}
	if got := s.rec.getFinished("stop"); got != n {
		t.Errorf("finished stop=%d, want %d", got, n)
	}
}

func TestStopSequence(t *testing.T) {
	// Force a long natural length, then stop on a token the mock always emits
	// ("the" is in the word bank and appears frequently).
	c := fastMock()
	c.MinNewTokens = 100
	c.MaxNewTokens = 100
	s := newStack(t, c, DefaultConfig())
	req := s.submit("r1", "Hello", 100, []string{"the"})
	tokens, reason := collect(req)
	if reason != domain.FinishStop {
		t.Errorf("reason=%q, want stop from stop-sequence", reason)
	}
	if len(tokens) >= 100 {
		t.Errorf("stop sequence did not cut generation short: got %d tokens", len(tokens))
	}
}

func TestClientCancellationFreesResources(t *testing.T) {
	c := fastMock()
	c.StepBase = 5 * time.Millisecond // slow enough to cancel mid-generation
	c.MinNewTokens = 1000
	c.MaxNewTokens = 1000
	s := newStack(t, c, DefaultConfig())

	ctx, cancel := context.WithCancel(context.Background())
	toks := s.tok.Encode("Hello")
	req := &domain.Request{
		ID: "cancelme", Model: "mock-7b", Prompt: "Hello", PromptTokens: toks,
		Params: domain.SamplingParams{MaxTokens: 1000},
		Ctx:    ctx, Stream: make(chan domain.StreamEvent, 8),
	}
	s.sched.Submit(req)

	// Read a couple of tokens, then cancel (simulating a client disconnect).
	<-req.Stream
	cancel()

	// Drain until closed; must end as canceled and free its KV.
	var reason domain.FinishReason
	for ev := range req.Stream {
		if ev.Done {
			reason = ev.Reason
		}
	}
	if reason != domain.FinishCanceled {
		t.Errorf("reason=%q, want canceled", reason)
	}
	// Give the loop a moment to free, then check the cache drained back down.
	time.Sleep(50 * time.Millisecond)
	if used := s.kv.Stats().Used; used != 0 {
		t.Errorf("KV not fully freed after cancellation: used=%d", used)
	}
}
