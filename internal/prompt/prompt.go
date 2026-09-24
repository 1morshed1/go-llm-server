// Package prompt turns chat messages into a domain.Request. It is the single
// place where chat templating, tokenization, prefix marking, validation, and
// sampling-parameter clamping live, so BOTH the REST (server) and gRPC
// (grpcapi) transports build requests identically — no drift between the two
// APIs. Each transport only maps its own wire types to a neutral GenSpec and
// maps a BuildError back to its own status representation.
package prompt

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// Message is a transport-neutral chat turn.
type Message struct {
	Role    string
	Content string
}

// GenSpec is a transport-neutral generation request. Pointer fields are
// "unset" when nil, so we can apply defaults distinct from a client's zero.
type GenSpec struct {
	Model       string
	Messages    []Message
	MaxTokens   *int
	Temperature *float64
	TopP        *float64
	Stop        []string
	Seed        int64
}

// BuildError is a validation failure. Code is a stable machine string each
// transport maps to its own status (HTTP 400 / gRPC InvalidArgument).
type BuildError struct {
	Code    string
	Message string
}

func (e *BuildError) Error() string { return e.Message }

const (
	CodeEmptyMessages   = "empty_messages"
	CodeContextExceeded = "context_exceeded"
)

// Build validates and assembles a domain.Request ready for the scheduler.
// ctxLen is the model's context window; defaultMaxTokens is used when the
// request omits max_tokens.
func Build(ctx context.Context, tok tokenizer.Tokenizer, ctxLen, defaultMaxTokens int, spec GenSpec) (*domain.Request, *BuildError) {
	if len(spec.Messages) == 0 {
		return nil, &BuildError{CodeEmptyMessages, "messages must not be empty"}
	}

	promptStr, prefixText := Render(spec.Messages)
	promptTokens := tok.Encode(promptStr)
	prefixLen := len(tok.Encode(prefixText))

	if len(promptTokens) >= ctxLen {
		return nil, &BuildError{CodeContextExceeded,
			fmt.Sprintf("prompt has %d tokens, exceeding the model's context length of %d", len(promptTokens), ctxLen)}
	}

	maxTokens := defaultMaxTokens
	if spec.MaxTokens != nil {
		maxTokens = *spec.MaxTokens
	}
	if maxAllowed := ctxLen - len(promptTokens); maxTokens > maxAllowed {
		maxTokens = maxAllowed
	}
	if maxTokens < 1 {
		maxTokens = 1
	}

	temperature := 1.0
	if spec.Temperature != nil {
		temperature = clamp(*spec.Temperature, 0, 2)
	}
	topP := 1.0
	if spec.TopP != nil {
		topP = clamp(*spec.TopP, 0, 1)
	}

	return &domain.Request{
		ID:           NewID(),
		Model:        spec.Model,
		Prompt:       promptStr,
		PromptTokens: promptTokens,
		PrefixLen:    prefixLen,
		Params: domain.SamplingParams{
			Temperature: temperature,
			TopP:        topP,
			MaxTokens:   maxTokens,
			Stop:        spec.Stop,
			Seed:        spec.Seed,
		},
		Ctx:    ctx,
		Stream: make(chan domain.StreamEvent, maxTokens+8),
	}, nil
}

// Render flattens chat messages into one prompt string using a simple
// role-tagged template, returning the leading system portion separately as the
// cacheable prefix (see docs/04). A real server would use the model's own chat
// template; the shape is what matters.
func Render(messages []Message) (promptStr, prefixText string) {
	var sb, prefix strings.Builder
	prefixEnded := false
	for _, m := range messages {
		block := fmt.Sprintf("<|%s|>\n%s\n", m.Role, m.Content)
		sb.WriteString(block)
		if !prefixEnded {
			if m.Role == string(domain.RoleSystem) {
				prefix.WriteString(block)
			} else {
				prefixEnded = true
			}
		}
	}
	sb.WriteString("<|assistant|>\n")
	return sb.String(), prefix.String()
}

// NewID returns an OpenAI-style request id like "chatcmpl-<hex>".
func NewID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "chatcmpl-" + hex.EncodeToString(b[:])
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
