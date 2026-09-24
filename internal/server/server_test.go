package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/1morshed1/go-llm-server/internal/batcher"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
	"github.com/1morshed1/go-llm-server/internal/metrics"
	"github.com/1morshed1/go-llm-server/internal/openai"
	"github.com/1morshed1/go-llm-server/internal/scheduler"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// newTestServer wires the full pipeline behind an httptest.Server so tests
// exercise the real request path end-to-end.
func newTestServer(t *testing.T) *httptest.Server {
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
	done := make(chan struct{})
	go func() { b.Run(ctx); close(done) }()

	SetQueueFullError(scheduler.ErrQueueFull)
	srv := New(sched, eng, tok, metrics.Noop{}, Options{DefaultMaxTokens: 32})
	ts := httptest.NewServer(srv.Router(nil))

	t.Cleanup(func() {
		ts.Close()
		cancel()
		<-done
		sched.Close()
	})
	return ts
}

func TestHealthz(t *testing.T) {
	ts := newTestServer(t)
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestModels(t *testing.T) {
	ts := newTestServer(t)
	resp, err := http.Get(ts.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var list openai.ModelList
	json.NewDecoder(resp.Body).Decode(&list)
	if len(list.Data) != 1 || list.Data[0].ID != "mock-7b" {
		t.Fatalf("unexpected models: %+v", list)
	}
}

func TestChatCompletionNonStreaming(t *testing.T) {
	ts := newTestServer(t)
	body := `{"model":"mock-7b","max_tokens":6,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	var out openai.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "chat.completion" || len(out.Choices) != 1 {
		t.Fatalf("bad response: %+v", out)
	}
	if out.Choices[0].Message.Content == "" {
		t.Error("expected non-empty content")
	}
	if out.Usage.CompletionTokens == 0 || out.Usage.PromptTokens == 0 {
		t.Errorf("bad usage: %+v", out.Usage)
	}
	if out.Usage.TotalTokens != out.Usage.PromptTokens+out.Usage.CompletionTokens {
		t.Error("usage totals inconsistent")
	}
}

func TestChatCompletionStreaming(t *testing.T) {
	ts := newTestServer(t)
	body := `{"model":"mock-7b","stream":true,"max_tokens":6,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type %q", ct)
	}

	var contentChunks, doneSeen int
	sawDONE := false
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDONE = true
			break
		}
		var chunk openai.ChatCompletionChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("bad chunk JSON: %v", err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("bad object %q", chunk.Object)
		}
		if chunk.Choices[0].Delta.Content != "" {
			contentChunks++
		}
		if chunk.Choices[0].FinishReason != nil {
			doneSeen++
		}
	}
	if !sawDONE {
		t.Error("did not see [DONE] sentinel")
	}
	if contentChunks == 0 {
		t.Error("no content chunks streamed")
	}
	if doneSeen != 1 {
		t.Errorf("expected exactly one finish_reason chunk, got %d", doneSeen)
	}
}

func TestEmptyMessagesRejected(t *testing.T) {
	ts := newTestServer(t)
	body := `{"model":"mock-7b","messages":[]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d, want 400", resp.StatusCode)
	}
}

func TestStopFieldAcceptsStringAndArray(t *testing.T) {
	// Both forms must unmarshal without error.
	for _, body := range []string{
		`{"model":"m","messages":[{"role":"user","content":"x"}],"stop":"END"}`,
		`{"model":"m","messages":[{"role":"user","content":"x"}],"stop":["A","B"]}`,
	} {
		var req openai.ChatCompletionRequest
		if err := json.NewDecoder(bytes.NewReader([]byte(body))).Decode(&req); err != nil {
			t.Errorf("failed to decode stop field: %v", err)
		}
	}
}
