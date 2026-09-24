package engine

import (
	"context"
	"hash/fnv"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// MockConfig controls the simulated engine's timing and behaviour. Every
// latency is configurable so that:
//   - tests can set them to ~0 for speed and determinism, and
//   - benchmarks can dial in realistic values to study queueing and batching.
type MockConfig struct {
	Name          string
	ContextLength int
	VocabSize     int

	// PrefillPerToken is the simulated cost of processing one prompt token
	// during prefill. TTFT ≈ PrefillPerToken × (uncached prompt tokens).
	PrefillPerToken time.Duration

	// StepBase is the fixed cost of one batched decode step regardless of how
	// many sequences are in the batch. StepPerSeq is the small additional cost
	// per sequence in the batch. Because StepPerSeq << StepBase, a step that
	// advances 16 sequences costs only a little more than one advancing 1 — so
	// per-token throughput rises with batch size. This is the crux of the
	// batching win (see docs/06-continuous-batching.md).
	StepBase   time.Duration
	StepPerSeq time.Duration

	// MinNewTokens/MaxNewTokens bound how many tokens a sequence naturally
	// generates before emitting EOS (the actual count is chosen deterministically
	// from the prompt). The batcher still enforces the request's MaxTokens on
	// top of this.
	MinNewTokens int
	MaxNewTokens int
}

// DefaultMockConfig returns timings chosen to resemble a small model on modest
// hardware, tuned so the batching win is clearly visible in benchmarks.
func DefaultMockConfig() MockConfig {
	return MockConfig{
		Name:            "mock-7b",
		ContextLength:   4096,
		VocabSize:       32000,
		PrefillPerToken: 400 * time.Microsecond, // ~40ms for a 100-token prompt
		StepBase:        18 * time.Millisecond,  // ~55 tok/s at batch size 1
		StepPerSeq:      1 * time.Millisecond,   // batch of 16 → ~470 tok/s
		MinNewTokens:    24,
		MaxNewTokens:    96,
	}
}

// MockEngine is a deterministic, dependency-free implementation of Engine that
// simulates a transformer's observable behaviour. It generates readable
// pseudo-text so streamed responses look plausible in demos.
type MockEngine struct {
	cfg   MockConfig
	tok   tokenizer.Tokenizer
	words []string
}

// lorem is the small word bank the mock samples from to produce readable output.
var lorem = []string{
	"the", "model", "processes", "each", "token", "in", "sequence", "and",
	"attention", "flows", "across", "the", "cache", "while", "the", "server",
	"batches", "requests", "together", "to", "keep", "the", "hardware", "busy",
	"streaming", "results", "back", "as", "soon", "as", "they", "are", "ready",
	"which", "is", "why", "throughput", "improves", "with", "continuous",
	"batching", "and", "prefix", "reuse", "of", "shared", "prompts",
}

// NewMock builds a MockEngine. The tokenizer is used to encode the generated
// words into tokens so decode counts and text line up with the rest of the
// system.
func NewMock(cfg MockConfig, tok tokenizer.Tokenizer) *MockEngine {
	if cfg.VocabSize == 0 {
		cfg = DefaultMockConfig()
	}
	return &MockEngine{cfg: cfg, tok: tok, words: lorem}
}

func (m *MockEngine) Info() ModelInfo {
	return ModelInfo{
		Name:          m.cfg.Name,
		ContextLength: m.cfg.ContextLength,
		VocabSize:     m.cfg.VocabSize,
	}
}

// Prefill simulates prompt processing. Its latency is proportional to the
// number of UNCACHED prompt tokens: tokens served from the prefix cache cost
// nothing here, which is exactly how prefix caching lowers TTFT in a real
// server. The sequence's natural length is derived deterministically from the
// prompt so identical prompts (with the same seed) generate identically.
func (m *MockEngine) Prefill(ctx context.Context, req *domain.Request, uncachedPromptTokens int) (*Sequence, error) {
	if uncachedPromptTokens < 0 {
		uncachedPromptTokens = 0
	}
	cost := time.Duration(uncachedPromptTokens) * m.cfg.PrefillPerToken
	if err := sleepCtx(ctx, cost); err != nil {
		return nil, err
	}

	// Seed a per-sequence RNG from the prompt text plus the request's seed, so
	// output is deterministic per (prompt, seed) but differs across prompts.
	seed := hashString(req.Prompt) ^ uint64(req.Params.Seed)
	r := newRNG(seed)

	// Choose a natural generation length in [Min, Max], deterministically.
	span := m.cfg.MaxNewTokens - m.cfg.MinNewTokens
	if span < 1 {
		span = 1
	}
	target := m.cfg.MinNewTokens + r.intn(span)

	return &Sequence{
		req:        req,
		rng:        r,
		targetLen:  target,
		produced:   0,
		vocabWords: m.words,
	}, nil
}

// Step advances every sequence by one token in a single simulated batched
// forward pass. The step latency depends on the batch size, not on any single
// sequence, which is what models GPU batch parallelism.
func (m *MockEngine) Step(ctx context.Context, seqs []*Sequence) ([]StepResult, error) {
	if len(seqs) == 0 {
		return nil, nil
	}
	stepCost := m.cfg.StepBase + time.Duration(len(seqs))*m.cfg.StepPerSeq
	if err := sleepCtx(ctx, stepCost); err != nil {
		return nil, err
	}

	results := make([]StepResult, len(seqs))
	for i, s := range seqs {
		s.produced++
		// The model naturally stops when it reaches its target length.
		if s.produced >= s.targetLen {
			results[i] = StepResult{Token: m.tok.EOS(), EOS: true}
			continue
		}
		// Otherwise emit a readable word (with a leading space after the first
		// token) so the streamed text looks like prose.
		word := s.vocabWords[s.rng.intn(len(s.vocabWords))]
		text := word
		if s.produced > 1 {
			text = " " + word
		}
		// Encode the word through the real tokenizer so IDs and counts are
		// consistent with how prompts were tokenized.
		toks := m.tok.Encode(text)
		tok := domain.Token{ID: firstIDOrZero(toks), Text: text}
		results[i] = StepResult{Token: tok, EOS: false}
	}
	return results, nil
}

// Release frees per-sequence state. The mock holds only Go memory, so this
// simply drops references to help the GC; a real engine would free KV blocks
// and any native handles here.
func (m *MockEngine) Release(seq *Sequence) {
	if seq == nil {
		return
	}
	seq.rng = nil
	seq.vocabWords = nil
}

// --- helpers ---

// sleepCtx sleeps for d but returns early with the context's error if the
// context is canceled first. Every simulated latency goes through here so that
// client disconnects and timeouts cancel promptly instead of blocking for the
// full simulated duration.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		// Still respect an already-canceled context.
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			return nil
		}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func hashString(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

func firstIDOrZero(toks []domain.Token) int {
	if len(toks) == 0 {
		return 0
	}
	return toks[0].ID
}

var _ Engine = (*MockEngine)(nil)
