package grpcapi

import (
	"context"
	"net"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/1morshed1/go-llm-server/internal/batcher"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/grpcapi/pb"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
	"github.com/1morshed1/go-llm-server/internal/metrics"
	"github.com/1morshed1/go-llm-server/internal/scheduler"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// newGRPCTest wires the full pipeline behind a real gRPC server on a random
// local port and returns a connected client.
func newGRPCTest(t *testing.T) pb.InferenceServiceClient {
	t.Helper()
	tok := tokenizer.NewSimple(32000)
	mc := engine.DefaultMockConfig()
	mc.PrefillPerToken, mc.StepBase, mc.StepPerSeq = 0, 0, 0
	mc.MinNewTokens, mc.MaxNewTokens = 5, 8
	eng := engine.NewMock(mc, tok)
	kv := kvcache.New(kvcache.DefaultConfig())
	sched := scheduler.New(scheduler.DefaultConfig())
	b := batcher.New(batcher.DefaultConfig(), eng, kv, sched, metrics.Noop{})

	ctx, cancel := context.WithCancel(context.Background())
	bdone := make(chan struct{})
	go func() { b.Run(ctx); close(bdone) }()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	gs := grpc.NewServer()
	pb.RegisterInferenceServiceServer(gs, New(sched, eng, tok, metrics.Noop{}, 32, scheduler.ErrQueueFull))
	go gs.Serve(lis)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		conn.Close()
		gs.Stop()
		cancel()
		<-bdone
		sched.Close()
	})
	return pb.NewInferenceServiceClient(conn)
}

func TestGRPCInfo(t *testing.T) {
	client := newGRPCTest(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := client.Info(ctx, &pb.InfoRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "mock-7b" || resp.ContextLength == 0 {
		t.Fatalf("bad info: %+v", resp)
	}
}

func TestGRPCGenerateStreams(t *testing.T) {
	client := newGRPCTest(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := client.Generate(ctx, &pb.GenerateRequest{
		Model:     "mock-7b",
		MaxTokens: 6,
		Messages:  []*pb.ChatMessage{{Role: "user", Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	var tokens int
	var sawDone bool
	for {
		resp, err := stream.Recv()
		if err != nil {
			// io.EOF after the final message is normal stream termination.
			break
		}
		if resp.Done {
			sawDone = true
			if resp.FinishReason == "" {
				t.Error("done message missing finish_reason")
			}
			if resp.PromptTokens == 0 {
				t.Error("done message missing prompt_tokens")
			}
			break
		}
		if resp.Token != "" {
			tokens++
		}
	}
	if !sawDone {
		t.Error("never received a done message")
	}
	if tokens == 0 {
		t.Error("no tokens streamed")
	}
}

func TestGRPCEmptyMessagesRejected(t *testing.T) {
	client := newGRPCTest(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := client.Generate(ctx, &pb.GenerateRequest{Model: "mock-7b"})
	if err != nil {
		t.Fatal(err)
	}
	// The error surfaces on the first Recv.
	_, err = stream.Recv()
	if err == nil {
		t.Fatal("expected InvalidArgument error for empty messages")
	}
}
