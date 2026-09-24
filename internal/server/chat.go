package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/openai"
)

// handleChatCompletions implements POST /v1/chat/completions for both streaming
// and non-streaming requests.
func (s *Server) handleChatCompletions(c *gin.Context) {
	s.metrics.IncRequestsReceived()

	var body openai.ChatCompletionRequest
	if err := c.ShouldBindJSON(&body); err != nil {
		writeError(c, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}

	req, herr := s.buildDomainRequest(c.Request.Context(), &body)
	if herr != nil {
		writeError(c, herr.status, herr.errType, herr.msg)
		return
	}

	// Admission control happens at submit. A full queue → 429, the standard
	// backpressure signal that tells clients to slow down or shed load.
	if err := s.sub.Submit(req); err != nil {
		if queueFullErr != nil && errors.Is(err, queueFullErr) {
			s.metrics.IncRequestsRejected()
			writeError(c, http.StatusTooManyRequests, "rate_limit_error", "server is overloaded; retry later")
			return
		}
		writeError(c, http.StatusServiceUnavailable, "server_error", err.Error())
		return
	}

	if body.Stream {
		s.streamResponse(c, req, body.Model)
	} else {
		s.bufferedResponse(c, req, body.Model)
	}
}

// bufferedResponse drains the whole generation, then sends one JSON body — the
// non-streaming path.
func (s *Server) bufferedResponse(c *gin.Context, req *domain.Request, model string) {
	var sb strings.Builder
	var reason domain.FinishReason
	var genCount int

	for ev := range req.Stream {
		if ev.Done {
			reason = ev.Reason
			if ev.Err != nil {
				writeError(c, http.StatusInternalServerError, "server_error", ev.Err.Error())
				return
			}
			break
		}
		sb.WriteString(ev.Token.Text)
		genCount++
	}

	// If the client disconnected, the context is done and there's nothing to
	// write back.
	if c.Request.Context().Err() != nil {
		return
	}

	promptToks := len(req.PromptTokens)
	c.JSON(http.StatusOK, openai.ChatCompletionResponse{
		ID:      req.ID,
		Object:  "chat.completion",
		Created: nowUnix(),
		Model:   model,
		Choices: []openai.Choice{{
			Index:        0,
			Message:      openai.Message{Role: string(domain.RoleAssistant), Content: sb.String()},
			FinishReason: mapFinishReason(reason),
		}},
		Usage: openai.Usage{
			PromptTokens:     promptToks,
			CompletionTokens: genCount,
			TotalTokens:      promptToks + genCount,
		},
	})
}

// httpError bundles an HTTP status with an OpenAI error type/message.
type httpError struct {
	status  int
	errType string
	msg     string
}

// buildDomainRequest validates and clamps the wire request, renders the chat
// messages into a single prompt, tokenizes it, marks the cacheable prefix, and
// produces a domain.Request ready for the scheduler.
func (s *Server) buildDomainRequest(ctx context.Context, body *openai.ChatCompletionRequest) (*domain.Request, *httpError) {
	if len(body.Messages) == 0 {
		return nil, &httpError{http.StatusBadRequest, "invalid_request_error", "messages must not be empty"}
	}

	prompt, prefixText := renderPrompt(body.Messages)
	promptTokens := s.tok.Encode(prompt)
	prefixLen := len(s.tok.Encode(prefixText))

	info := s.eng.Info()
	if len(promptTokens) >= info.ContextLength {
		return nil, &httpError{http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("prompt has %d tokens, exceeding the model's context length of %d", len(promptTokens), info.ContextLength)}
	}

	// Sampling params: apply defaults, then clamp to safe ranges. Clamping at the
	// boundary means the engine can trust its inputs and never has to defend
	// against absurd values.
	maxTokens := s.opts.DefaultMaxTokens
	if body.MaxTokens != nil {
		maxTokens = *body.MaxTokens
	}
	if maxAllowed := info.ContextLength - len(promptTokens); maxTokens > maxAllowed {
		maxTokens = maxAllowed
	}
	if maxTokens < 1 {
		maxTokens = 1
	}

	temperature := 1.0
	if body.Temperature != nil {
		temperature = clampFloat(*body.Temperature, 0, 2)
	}
	topP := 1.0
	if body.TopP != nil {
		topP = clampFloat(*body.TopP, 0, 1)
	}
	var seed int64
	if body.Seed != nil {
		seed = *body.Seed
	}

	req := &domain.Request{
		ID:           genID(),
		Model:        body.Model,
		Prompt:       prompt,
		PromptTokens: promptTokens,
		PrefixLen:    prefixLen,
		Params: domain.SamplingParams{
			Temperature: temperature,
			TopP:        topP,
			MaxTokens:   maxTokens,
			Stop:        []string(body.Stop),
			Seed:        seed,
		},
		Ctx: ctx,
		// Buffer the stream so a slow client never blocks the batcher loop; the
		// batcher also selects on ctx, so this is belt-and-suspenders.
		Stream: make(chan domain.StreamEvent, maxTokens+8),
	}
	return req, nil
}

// renderPrompt flattens chat messages into a single prompt string using a simple
// role-tagged template, and returns the leading system portion separately as the
// cacheable prefix (see docs/04 on prefix caching). A real server would use the
// model's own chat template; the shape is what matters here.
func renderPrompt(messages []openai.Message) (prompt, prefixText string) {
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
	sb.WriteString("<|assistant|>\n") // primer that cues the model to reply
	return sb.String(), prefix.String()
}

// mapFinishReason converts our domain finish reason to the OpenAI string. A
// canceled request maps to "stop" for wire compatibility (OpenAI has no
// "canceled"); the distinction is preserved in metrics.
func mapFinishReason(r domain.FinishReason) string {
	switch r {
	case domain.FinishLength:
		return "length"
	case domain.FinishStop, domain.FinishCanceled:
		return "stop"
	default:
		return "stop"
	}
}

func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// genID returns an OpenAI-style request id like "chatcmpl-<hex>".
func genID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "chatcmpl-" + hex.EncodeToString(b[:])
}
