// Package server is the HTTP/API layer: it turns HTTP requests into
// domain.Requests, submits them to the scheduler, and streams results back.
//
// It is the ONLY package that knows about HTTP and JSON. Everything below it
// speaks domain types (decision D2 in docs/01). See docs/07 for the full tour,
// including how Server-Sent Events streaming works.
package server

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/metrics"
	"github.com/1morshed1/go-llm-server/internal/openai"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// Submitter is the slice of the scheduler the API layer needs. Depending on a
// narrow interface (rather than the concrete *scheduler.Scheduler) keeps the
// server testable with a fake and documents exactly what it uses.
type Submitter interface {
	Submit(*domain.Request) error
	Len() int
}

// ErrQueueFull lets the server recognise admission-control rejection without
// importing the scheduler package (avoiding a dependency just for one error).
// The wiring in main passes the scheduler's error value through; we compare by
// message as a fallback. In practice we compare the sentinel from scheduler.
// To keep it simple and explicit, the server treats scheduler.ErrQueueFull via
// errors.Is using the value injected below.
var queueFullErr error

// SetQueueFullError injects the scheduler's ErrQueueFull sentinel so the handler
// can map it to HTTP 429. Called once from main.
func SetQueueFullError(err error) { queueFullErr = err }

// Options tunes API-layer behaviour.
type Options struct {
	DefaultMaxTokens int // used when a request omits max_tokens
}

// Server holds the dependencies the handlers need.
type Server struct {
	sub     Submitter
	eng     engine.Engine
	tok     tokenizer.Tokenizer
	metrics metrics.Recorder
	opts    Options
}

// New builds a Server.
func New(sub Submitter, eng engine.Engine, tok tokenizer.Tokenizer, rec metrics.Recorder, opts Options) *Server {
	if rec == nil {
		rec = metrics.Noop{}
	}
	if opts.DefaultMaxTokens <= 0 {
		opts.DefaultMaxTokens = 128
	}
	return &Server{sub: sub, eng: eng, tok: tok, metrics: rec, opts: opts}
}

// Router builds the gin engine with all routes registered. metricsHandler, if
// non-nil, is mounted at /metrics (wired by main so this package doesn't depend
// on the Prometheus client).
func (s *Server) Router(metricsHandler http.Handler) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	r.GET("/healthz", s.handleHealth)
	r.GET("/v1/models", s.handleModels)
	r.POST("/v1/chat/completions", s.handleChatCompletions)
	if metricsHandler != nil {
		r.GET("/metrics", gin.WrapH(metricsHandler))
	}
	return r
}

// handleHealth is a liveness/readiness probe. It reports the model and current
// queue depth so a load balancer or operator gets a useful signal, not just 200.
func (s *Server) handleHealth(c *gin.Context) {
	info := s.eng.Info()
	c.JSON(http.StatusOK, gin.H{
		"status":         "ok",
		"model":          info.Name,
		"context_length": info.ContextLength,
		"queue_depth":    s.sub.Len(),
	})
}

// handleModels implements GET /v1/models, mirroring OpenAI so clients can
// discover what's served.
func (s *Server) handleModels(c *gin.Context) {
	info := s.eng.Info()
	c.JSON(http.StatusOK, openai.ModelList{
		Object: "list",
		Data: []openai.ModelCard{{
			ID:      info.Name,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "go-llm-server",
		}},
	})
}

// writeError sends an OpenAI-style error envelope with the given status.
func writeError(c *gin.Context, status int, errType, msg string) {
	c.JSON(status, openai.ErrorResponse{Error: openai.ErrorBody{
		Message: msg,
		Type:    errType,
	}})
}
