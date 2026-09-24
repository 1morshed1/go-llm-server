// Package grpcapi implements the gRPC streaming transport, alongside the REST
// API in internal/server. Both transports submit the SAME domain.Request to the
// SAME scheduler — gRPC is just a second front door. This demonstrates the
// payoff of keeping transport out of the core (decision D2): adding a whole new
// protocol touches nothing below this package.
//
// See docs/11-grpc.md for why gRPC suits token streaming.
package grpcapi

import (
	"context"
	"errors"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/grpcapi/pb"
	"github.com/1morshed1/go-llm-server/internal/metrics"
	"github.com/1morshed1/go-llm-server/internal/prompt"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// Submitter is the slice of the scheduler this transport needs (same narrow
// interface idea as the REST layer).
type Submitter interface {
	Submit(*domain.Request) error
	Len() int
}

// Server implements pb.InferenceServiceServer.
type Server struct {
	pb.UnimplementedInferenceServiceServer

	sub              Submitter
	eng              engine.Engine
	tok              tokenizer.Tokenizer
	metrics          metrics.Recorder
	defaultMaxTokens int
	queueFull        error // scheduler.ErrQueueFull, injected by main
}

// New builds a gRPC Server. queueFull is the scheduler's ErrQueueFull sentinel
// so admission-control rejection maps to codes.ResourceExhausted.
func New(sub Submitter, eng engine.Engine, tok tokenizer.Tokenizer, rec metrics.Recorder, defaultMaxTokens int, queueFull error) *Server {
	if rec == nil {
		rec = metrics.Noop{}
	}
	if defaultMaxTokens <= 0 {
		defaultMaxTokens = 128
	}
	return &Server{sub: sub, eng: eng, tok: tok, metrics: rec, defaultMaxTokens: defaultMaxTokens, queueFull: queueFull}
}

// Generate runs one completion and streams tokens back. It is a server-streaming
// RPC: the framework hands us a stream we call Send on repeatedly, then return.
func (s *Server) Generate(req *pb.GenerateRequest, stream pb.InferenceService_GenerateServer) error {
	s.metrics.IncRequestsReceived()
	ctx := stream.Context() // cancelled if the client goes away

	msgs := make([]prompt.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = prompt.Message{Role: m.Role, Content: m.Content}
	}
	spec := prompt.GenSpec{
		Model:    req.Model,
		Messages: msgs,
		Stop:     req.Stop,
		Seed:     req.Seed,
	}
	if req.MaxTokens > 0 {
		mt := int(req.MaxTokens)
		spec.MaxTokens = &mt
	}
	if req.Temperature != 0 {
		spec.Temperature = &req.Temperature
	}
	if req.TopP != 0 {
		spec.TopP = &req.TopP
	}

	info := s.eng.Info()
	dreq, berr := prompt.Build(ctx, s.tok, info.ContextLength, s.defaultMaxTokens, spec)
	if berr != nil {
		return status.Error(codes.InvalidArgument, berr.Message)
	}

	if err := s.sub.Submit(dreq); err != nil {
		if s.queueFull != nil && errors.Is(err, s.queueFull) {
			s.metrics.IncRequestsRejected()
			return status.Error(codes.ResourceExhausted, "server is overloaded; retry later")
		}
		return status.Error(codes.Unavailable, err.Error())
	}

	promptTokens := int32(len(dreq.PromptTokens))
	var completion int32
	for ev := range dreq.Stream {
		if ev.Done {
			if ev.Err != nil {
				return status.Error(codes.Internal, ev.Err.Error())
			}
			return stream.Send(&pb.GenerateResponse{
				Done:             true,
				FinishReason:     mapFinishReason(ev.Reason),
				PromptTokens:     promptTokens,
				CompletionTokens: completion,
			})
		}
		completion++
		if err := stream.Send(&pb.GenerateResponse{Token: ev.Token.Text}); err != nil {
			// Client disconnected: the batcher observes ctx and frees resources.
			return err
		}
	}
	return nil
}

// Info returns model metadata and current queue depth.
func (s *Server) Info(ctx context.Context, _ *pb.InfoRequest) (*pb.InfoResponse, error) {
	i := s.eng.Info()
	return &pb.InfoResponse{
		Model:         i.Name,
		ContextLength: int32(i.ContextLength),
		VocabSize:     int32(i.VocabSize),
		QueueDepth:    int32(s.sub.Len()),
	}, nil
}

func mapFinishReason(r domain.FinishReason) string {
	switch r {
	case domain.FinishLength:
		return "length"
	case domain.FinishCanceled:
		return "canceled"
	default:
		return "stop"
	}
}
