// Package engine defines the boundary between the Go serving layer and the
// actual model compute, and provides a simulated backend that implements it.
//
// This is the single most important abstraction in the project. Read
// docs/02-the-engine-interface.md alongside this file.
//
// THE PREFILL / DECODE SPLIT
//
// Transformer inference has two distinct phases, and every serious inference
// server treats them separately because they have very different cost profiles:
//
//   - PREFILL ("prompt processing"): the model reads the entire prompt and
//     builds the KV cache for it. Cost is roughly proportional to the number of
//     prompt tokens. This is a one-time, parallel operation and it dominates
//     Time-To-First-Token (TTFT).
//
//   - DECODE ("generation"): the model produces output one token at a time,
//     each step attending over the whole KV cache. Cost per step is roughly
//     constant per sequence, and — crucially — MANY sequences can be decoded in
//     a single batched step almost as cheaply as one. That parallelism is the
//     entire reason continuous batching works (see docs/06).
//
// The Engine interface exposes exactly this shape: Prefill once, then Step
// repeatedly over a batch of sequences.
package engine

import (
	"context"

	"github.com/1morshed1/go-llm-server/internal/domain"
)

// ModelInfo describes a loaded model. Returned to clients via /v1/models and
// used by the server to validate requests against the context window.
type ModelInfo struct {
	Name          string // e.g. "mock-7b"
	ContextLength int    // max prompt+generation tokens the model supports
	VocabSize     int    // size of the token vocabulary
}

// Sequence is the per-request compute state owned by the engine while a request
// is being generated. For a real backend this would hold pointers into the
// model's KV cache and the current position; for the mock it holds the RNG and
// counters that make generation deterministic. The batcher treats it as opaque.
type Sequence struct {
	req        *domain.Request
	rng        *rng     // deterministic PRNG seeded from the prompt
	targetLen  int      // how many tokens this sequence will emit before natural EOS
	produced   int      // how many tokens emitted so far
	vocabWords []string // word bank the mock samples from (for readable output)
}

// Request returns the domain request this sequence is generating for.
func (s *Sequence) Request() *domain.Request { return s.req }

// StepResult is what one decode step produces for one sequence.
type StepResult struct {
	Token domain.Token // the newly generated token
	// EOS is true when the MODEL naturally chose to stop (it emitted an
	// end-of-sequence token). Serving policy limits like MaxTokens and stop
	// strings are enforced by the BATCHER, not here — the engine only reports
	// the model's own behaviour. This separation keeps "what the model does"
	// and "what the server allows" in different places (see docs/06).
	EOS bool
}

// Engine is the compute backend. A real implementation would wrap llama.cpp
// (via CGo) or ONNX Runtime; MockEngine simulates one. Anything implementing
// these four methods can drive the entire serving layer unchanged.
type Engine interface {
	// Info returns static metadata about the loaded model.
	Info() ModelInfo

	// Prefill processes a request's prompt and returns a fresh Sequence ready
	// to decode. uncachedPromptTokens is how many leading prompt tokens were
	// NOT served from the KV prefix cache; the engine's prefill cost is
	// proportional to it (that's the mechanism by which prefix caching lowers
	// TTFT — see docs/04). Prefill emits no output token; the first token comes
	// from the first Step.
	Prefill(ctx context.Context, req *domain.Request, uncachedPromptTokens int) (*Sequence, error)

	// Step advances EVERY sequence in seqs by exactly one token in a single
	// (simulated) batched forward pass. The returned slice is aligned with the
	// input slice. The step's wall-clock latency models GPU batch parallelism:
	// it grows only slightly with batch size, so tokens-per-second rises sharply
	// as the batch fills — the throughput win this whole project is about.
	Step(ctx context.Context, seqs []*Sequence) ([]StepResult, error)

	// Release frees the compute state held by a finished sequence.
	Release(seq *Sequence)
}
