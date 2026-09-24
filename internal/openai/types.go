// Package openai defines the wire types for the OpenAI-compatible HTTP API.
//
// These are TRANSPORT types: they carry JSON tags and mirror OpenAI's schema
// exactly, so existing OpenAI client libraries and tools work unchanged against
// this server. The server layer maps them to/from the transport-free
// domain.Request (decision D2 in docs/01) — the wire format never leaks into the
// scheduler, batcher, or engine.
//
// Compatibility is a feature: "OpenAI-compatible" is the de-facto standard that
// vLLM, Ollama, TGI, LM Studio, and others all speak, because it means clients
// don't need a new SDK to talk to you.
package openai

import (
	"encoding/json"
	"errors"
)

// ChatCompletionRequest is the body of POST /v1/chat/completions.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []Message     `json:"messages"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	Stop        StopSequences `json:"stop,omitempty"`
	Seed        *int64        `json:"seed,omitempty"`
}

// Message is one chat turn on the wire.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// StopSequences accepts either a single string or an array of strings, matching
// OpenAI's polymorphic `stop` field. Custom JSON unmarshaling is the idiomatic
// Go way to handle a field that can be one of several shapes.
type StopSequences []string

func (s *StopSequences) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	// Try a single string first.
	var single string
	if err := json.Unmarshal(data, &single); err == nil {
		*s = []string{single}
		return nil
	}
	// Then an array of strings.
	var many []string
	if err := json.Unmarshal(data, &many); err == nil {
		*s = many
		return nil
	}
	return errors.New("stop must be a string or an array of strings")
}

// ChatCompletionResponse is the non-streaming response body.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"` // "chat.completion"
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Choice is one completion in a non-streaming response.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage reports token accounting.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionChunk is one Server-Sent Event in a streaming response.
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"` // "chat.completion.chunk"
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

// ChunkChoice carries an incremental delta in a streaming response.
type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"` // null until the final chunk
}

// Delta is the incremental content in a streaming chunk.
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// ErrorResponse is the OpenAI-style error envelope.
type ErrorResponse struct {
	Error ErrorBody `json:"error"`
}

// ErrorBody describes an error.
type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// ModelList is the body of GET /v1/models.
type ModelList struct {
	Object string      `json:"object"` // "list"
	Data   []ModelCard `json:"data"`
}

// ModelCard describes one available model.
type ModelCard struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "model"
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}
