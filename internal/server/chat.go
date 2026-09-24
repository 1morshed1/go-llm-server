package server

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/openai"
	"github.com/1morshed1/go-llm-server/internal/prompt"
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

// buildDomainRequest maps the OpenAI wire request onto the shared prompt.Build
// (which does templating, tokenization, prefix marking, validation, and
// clamping) and maps any validation error back to an HTTP status. The actual
// logic lives in internal/prompt so REST and gRPC build requests identically.
func (s *Server) buildDomainRequest(ctx context.Context, body *openai.ChatCompletionRequest) (*domain.Request, *httpError) {
	msgs := make([]prompt.Message, len(body.Messages))
	for i, m := range body.Messages {
		msgs[i] = prompt.Message{Role: m.Role, Content: m.Content}
	}
	var seed int64
	if body.Seed != nil {
		seed = *body.Seed
	}
	spec := prompt.GenSpec{
		Model:       body.Model,
		Messages:    msgs,
		MaxTokens:   body.MaxTokens,
		Temperature: body.Temperature,
		TopP:        body.TopP,
		Stop:        []string(body.Stop),
		Seed:        seed,
	}
	info := s.eng.Info()
	req, berr := prompt.Build(ctx, s.tok, info.ContextLength, s.opts.DefaultMaxTokens, spec)
	if berr != nil {
		return nil, &httpError{http.StatusBadRequest, "invalid_request_error", berr.Message}
	}
	return req, nil
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
