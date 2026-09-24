package server

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/openai"
)

// streamResponse implements the streaming path using Server-Sent Events (SSE),
// exactly as the OpenAI streaming API does. SSE is a dead-simple protocol: the
// server keeps the HTTP response open and writes `data: <json>\n\n` lines,
// flushing after each so the client sees tokens as they're produced. The stream
// ends with the sentinel line `data: [DONE]`.
//
// The key to low latency is FLUSHING: without it, Go buffers the response and
// the client sees nothing until the buffer fills or the handler returns —
// defeating the entire purpose of streaming.
func (s *Server) streamResponse(c *gin.Context, req *domain.Request, model string) {
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		writeError(c, http.StatusInternalServerError, "server_error", "streaming unsupported")
		return
	}

	h := c.Writer.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // tell nginx not to buffer the stream
	c.Writer.WriteHeader(http.StatusOK)

	created := nowUnix()

	// OpenAI sends an initial chunk carrying just the assistant role, before any
	// content. Clients rely on this to open the message.
	first := openai.ChatCompletionChunk{
		ID: req.ID, Object: "chat.completion.chunk", Created: created, Model: model,
		Choices: []openai.ChunkChoice{{Index: 0, Delta: openai.Delta{Role: string(domain.RoleAssistant)}}},
	}
	if !writeSSE(c, flusher, first) {
		return
	}

	for ev := range req.Stream {
		if ev.Done {
			reason := mapFinishReason(ev.Reason)
			final := openai.ChatCompletionChunk{
				ID: req.ID, Object: "chat.completion.chunk", Created: created, Model: model,
				Choices: []openai.ChunkChoice{{Index: 0, Delta: openai.Delta{}, FinishReason: &reason}},
			}
			writeSSE(c, flusher, final)
			writeRaw(c, flusher, "data: [DONE]\n\n")
			return
		}
		chunk := openai.ChatCompletionChunk{
			ID: req.ID, Object: "chat.completion.chunk", Created: created, Model: model,
			Choices: []openai.ChunkChoice{{Index: 0, Delta: openai.Delta{Content: ev.Token.Text}}},
		}
		if !writeSSE(c, flusher, chunk) {
			// Client went away; the batcher observes req.Ctx and will stop and
			// free resources. We just stop writing. (The range will end when the
			// batcher closes the stream after cancellation.)
			return
		}
	}
}

// writeSSE marshals v and writes it as one SSE data frame, flushing immediately.
// Returns false if the write failed (client disconnected).
func writeSSE(c *gin.Context, flusher http.Flusher, v any) bool {
	data, err := json.Marshal(v)
	if err != nil {
		return false
	}
	return writeRaw(c, flusher, "data: "+string(data)+"\n\n")
}

func writeRaw(c *gin.Context, flusher http.Flusher, s string) bool {
	if _, err := c.Writer.WriteString(s); err != nil {
		return false
	}
	flusher.Flush()
	return true
}

func nowUnix() int64 { return time.Now().Unix() }
