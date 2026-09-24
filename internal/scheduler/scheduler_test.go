package scheduler

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
)

func mkReq(id string, promptTokens int) *domain.Request {
	toks := make([]domain.Token, promptTokens)
	return &domain.Request{
		ID:           id,
		PromptTokens: toks,
		Ctx:          context.Background(),
		Stream:       make(chan domain.StreamEvent, 4), // buffered so fail() never blocks
	}
}

func TestAdmissionControl(t *testing.T) {
	s := New(Config{MaxQueueDepth: 2, MaxWait: time.Minute, ReapInterval: time.Hour})
	defer s.Close()

	if err := s.Submit(mkReq("a", 10)); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(mkReq("b", 10)); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(mkReq("c", 10)); err != ErrQueueFull {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
}

func TestFCFSOrdering(t *testing.T) {
	// All prompts long (no boost) → pure FCFS.
	s := New(Config{MaxQueueDepth: 10, MaxWait: time.Minute, ShortPromptThreshold: 1, ShortPromptBoost: 0, ReapInterval: time.Hour})
	defer s.Close()

	for i := 0; i < 5; i++ {
		if err := s.Submit(mkReq(strconv.Itoa(i), 100)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 5; i++ {
		req, ok := s.TryPop()
		if !ok {
			t.Fatalf("expected request %d", i)
		}
		if req.ID != strconv.Itoa(i) {
			t.Errorf("FCFS violated: position %d has %s", i, req.ID)
		}
	}
}

func TestShortPromptBoost(t *testing.T) {
	s := New(Config{MaxQueueDepth: 100, MaxWait: time.Minute, ShortPromptThreshold: 5, ShortPromptBoost: 4, ReapInterval: time.Hour})
	defer s.Close()

	// Enqueue three long prompts first...
	s.Submit(mkReq("long0", 100))
	s.Submit(mkReq("long1", 100))
	s.Submit(mkReq("long2", 100))
	// ...then a short one. With boost=4 it should jump ahead of the long ones
	// enqueued before it (it can overtake up to 4 earlier arrivals).
	s.Submit(mkReq("short", 3))

	req, _ := s.TryPop()
	if req.ID != "short" {
		t.Errorf("expected short prompt to be served first, got %s", req.ID)
	}
}

func TestBoostIsBounded(t *testing.T) {
	// A short prompt should NOT overtake requests further back than the boost.
	s := New(Config{MaxQueueDepth: 100, MaxWait: time.Minute, ShortPromptThreshold: 5, ShortPromptBoost: 2, ReapInterval: time.Hour})
	defer s.Close()

	// 5 long prompts, then a short one with boost 2 → it can pass at most the
	// last 2 long prompts, so the first long prompt still goes first.
	for i := 0; i < 5; i++ {
		s.Submit(mkReq("long"+strconv.Itoa(i), 100))
	}
	s.Submit(mkReq("short", 3))

	req, _ := s.TryPop()
	if req.ID != "long0" {
		t.Errorf("boost not bounded: expected long0 first, got %s", req.ID)
	}
}

func TestTimeoutReaping(t *testing.T) {
	s := New(Config{MaxQueueDepth: 10, MaxWait: 20 * time.Millisecond, ReapInterval: 5 * time.Millisecond})
	defer s.Close()

	req := mkReq("slow", 10)
	s.Submit(req)

	// Wait for the reaper to fail it.
	select {
	case ev := <-req.Stream:
		if !ev.Done || ev.Reason != domain.FinishCanceled {
			t.Errorf("expected FinishCanceled Done event, got %+v", ev)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("reaper did not time out the request")
	}
	// It should also be gone from the queue.
	if s.Len() != 0 {
		t.Errorf("expected empty queue after reaping, got %d", s.Len())
	}
}

func TestReadySignal(t *testing.T) {
	s := New(DefaultConfig())
	defer s.Close()
	s.Submit(mkReq("x", 10))
	select {
	case <-s.Ready():
	case <-time.After(time.Second):
		t.Fatal("expected Ready signal after Submit")
	}
}
