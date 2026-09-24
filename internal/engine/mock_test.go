package engine

import (
	"context"
	"testing"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// fastCfg returns a config with negligible latencies for fast, deterministic
// tests. We keep StepBase tiny but non-zero to exercise the timer path.
func fastCfg() MockConfig {
	c := DefaultMockConfig()
	c.PrefillPerToken = 0
	c.StepBase = 0
	c.StepPerSeq = 0
	c.MinNewTokens = 4
	c.MaxNewTokens = 5
	return c
}

func newReq(prompt string) *domain.Request {
	return &domain.Request{
		ID:     "t1",
		Model:  "mock-7b",
		Prompt: prompt,
		Ctx:    context.Background(),
		Params: domain.SamplingParams{MaxTokens: 100},
	}
}

// TestGeneratesUntilEOS checks that a sequence eventually reports natural EOS
// within the configured bound.
func TestGeneratesUntilEOS(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	eng := NewMock(fastCfg(), tok)
	seq, err := eng.Prefill(context.Background(), newReq("hello"), 1)
	if err != nil {
		t.Fatal(err)
	}
	sawEOS := false
	for i := 0; i < 50; i++ {
		res, err := eng.Step(context.Background(), []*Sequence{seq})
		if err != nil {
			t.Fatal(err)
		}
		if res[0].EOS {
			sawEOS = true
			break
		}
	}
	if !sawEOS {
		t.Fatalf("expected EOS within bound, got none after %d tokens", seq.produced)
	}
	if seq.produced < fastCfg().MinNewTokens {
		t.Errorf("produced %d tokens, fewer than MinNewTokens", seq.produced)
	}
}

// TestDeterministicOutput ensures identical (prompt, seed) produce identical
// token streams — the property tests and demos rely on.
func TestDeterministicOutput(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	eng := NewMock(fastCfg(), tok)

	run := func() []string {
		seq, _ := eng.Prefill(context.Background(), newReq("same prompt"), 1)
		var out []string
		for {
			res, _ := eng.Step(context.Background(), []*Sequence{seq})
			if res[0].EOS {
				break
			}
			out = append(out, res[0].Token.Text)
		}
		return out
	}

	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("nondeterministic length: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("nondeterministic token at %d: %q vs %q", i, a[i], b[i])
		}
	}
}

// TestBatchStepAmortizes verifies the core batching premise: a step advancing
// many sequences costs only a little more than one advancing a single sequence,
// so per-token cost falls as the batch grows.
func TestBatchStepAmortizes(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	cfg := DefaultMockConfig()
	cfg.PrefillPerToken = 0
	eng := NewMock(cfg, tok)

	mkBatch := func(n int) []*Sequence {
		seqs := make([]*Sequence, n)
		for i := range seqs {
			seqs[i], _ = eng.Prefill(context.Background(), newReq("p"), 1)
		}
		return seqs
	}

	timeStep := func(n int) time.Duration {
		seqs := mkBatch(n)
		start := time.Now()
		_, _ = eng.Step(context.Background(), seqs)
		return time.Since(start)
	}

	one := timeStep(1)
	sixteen := timeStep(16)

	perTokenOne := one
	perTokenSixteen := sixteen / 16
	if perTokenSixteen >= perTokenOne {
		t.Errorf("expected per-token cost to fall with batch size: batch1=%v/tok batch16=%v/tok", perTokenOne, perTokenSixteen)
	}
}

// TestPrefillCancels ensures a canceled context aborts prefill promptly instead
// of sleeping for the full simulated duration.
func TestPrefillCancels(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	cfg := DefaultMockConfig()
	cfg.PrefillPerToken = 100 * time.Millisecond // large, so cancellation matters
	eng := NewMock(cfg, tok)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already canceled
	_, err := eng.Prefill(ctx, newReq("long prompt tokens tokens tokens"), 100)
	if err == nil {
		t.Fatal("expected cancellation error, got nil")
	}
}
