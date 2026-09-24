// Package domain holds the core vocabulary of the inference server.
//
// These types are deliberately kept free of any HTTP, gRPC, or Prometheus
// concerns. Everything else in the codebase (the API layer, the scheduler,
// the batcher, the engine) speaks in terms of these types. Keeping the
// "domain language" in one dependency-free package is a classic technique for
// keeping a system understandable: you can read this one file and know what
// the whole server is *about* before reading a single line of transport code.
package domain

import (
	"context"
	"time"
)

// Token is a single unit produced by the tokenizer. In a real LLM a token is
// an integer ID into the model's vocabulary (e.g. the word " cache" might be
// token 8145). We keep both the ID (what the model computes on) and the Text
// (what we stream back to the user) so the rest of the system never has to
// re-consult the tokenizer to know what a token "says".
type Token struct {
	ID   int    // vocabulary index
	Text string // the decoded piece of text this token represents
}

// Role identifies who authored a chat message, mirroring the OpenAI chat API.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one turn in a chat conversation.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// SamplingParams controls how the next token is chosen from the model's output
// probability distribution. These are the standard knobs every inference API
// exposes. We validate and clamp them at the API boundary so the engine can
// trust them.
type SamplingParams struct {
	// Temperature scales the logits before sampling. 0 => greedy (always pick
	// the most likely token, fully deterministic). Higher => more random.
	Temperature float64
	// TopP (nucleus sampling): sample only from the smallest set of tokens
	// whose cumulative probability exceeds TopP. 1.0 disables it.
	TopP float64
	// MaxTokens caps how many tokens we will generate for this request.
	MaxTokens int
	// Stop sequences: if the generated text ends with any of these, stop.
	Stop []string
	// Seed makes sampling reproducible for a given prompt (used by the mock
	// engine to produce deterministic output for tests and demos).
	Seed int64
}

// FinishReason explains why generation stopped, mirroring the OpenAI API.
type FinishReason string

const (
	// FinishStop: the model emitted an end-of-sequence token or a stop string.
	FinishStop FinishReason = "stop"
	// FinishLength: we hit MaxTokens before the model chose to stop.
	FinishLength FinishReason = "length"
	// FinishCanceled: the request was canceled (client disconnect, timeout,
	// or admission-control eviction).
	FinishCanceled FinishReason = "canceled"
)

// Request is the unit of work that flows through the whole system: it is
// created by the API layer, queued by the scheduler, executed by the batcher,
// and streamed back to the client. One Request == one generation.
//
// Note how a Request carries its *own* output channel (Stream). This is the
// heart of how the server stays concurrent: the batcher never calls back into
// the HTTP handler. It just pushes tokens onto the channel and moves on. The
// handler, running in its own goroutine, ranges over that channel and writes
// bytes to the client. Producer and consumer are fully decoupled.
type Request struct {
	// ID uniquely identifies this request for the life of the process. Used as
	// a map key in the batcher and as a label in logs/metrics.
	ID string

	// Model is the requested model name (e.g. "mock-7b"). The model manager
	// resolves this to an actual engine.
	Model string

	// Prompt is the full flattened prompt (system + chat turns rendered into a
	// single string) as produced by the API layer's chat template.
	Prompt string

	// PromptTokens is Prompt after tokenization. Split into two halves so the
	// KV-cache manager can do prefix caching: the shared, cacheable portion
	// (usually the system prompt) and the request-specific remainder.
	PromptTokens []Token

	// PrefixLen is how many of the leading PromptTokens form the "cacheable
	// prefix" (typically the system prompt). The KV manager hashes exactly
	// this many tokens to look for a cache hit. 0 means "no declared prefix".
	PrefixLen int

	// Params controls sampling.
	Params SamplingParams

	// Stream is the channel the batcher pushes generated tokens onto. The API
	// handler consumes it. The batcher closes it exactly once, when generation
	// finishes (for any reason). A nil Stream is not allowed.
	Stream chan StreamEvent

	// Ctx is the request's context. It is canceled if the client disconnects
	// or the request-level deadline fires. Every long-running stage checks it.
	Ctx context.Context

	// Priority is used by the scheduler's priority queue. LOWER numbers are
	// served first (like `nice` on Unix). See scheduler for how it is derived.
	Priority int

	// EnqueuedAt records when the scheduler accepted the request. Used to
	// compute queue-wait time and to enforce the max-wait timeout.
	EnqueuedAt time.Time

	// --- fields owned by the batcher while the request is "active" ---

	// Generated accumulates the tokens produced so far. The batcher appends to
	// this on every forward pass in which this request is part of the batch.
	Generated []Token

	// KVHandle is the request's allocation in the KV cache, assigned when it
	// joins the active batch and released when it leaves.
	KVHandle interface{} // *kvcache.Handle; kept as interface{} to avoid an import cycle
}

// StreamEvent is one item pushed onto a Request's Stream channel. It carries
// either a newly generated token or the terminal signal that generation is
// done. Modelling "done" as an explicit event (rather than only relying on the
// channel closing) lets us attach a FinishReason and an error to the ending.
type StreamEvent struct {
	// Token is the newly generated token. Valid only when Done is false.
	Token Token
	// Done marks the final event. When true, Token is zero and Reason/Err are
	// meaningful.
	Done bool
	// Reason explains why generation stopped. Set only when Done is true.
	Reason FinishReason
	// Err is non-nil if generation failed (as opposed to finishing normally).
	Err error
}

// Usage reports token accounting for a completed request, mirroring the
// OpenAI API's `usage` object.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
