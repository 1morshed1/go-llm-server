// Command loadgen drives HTTP load against a running server and reports latency
// percentiles and throughput. Use it to watch the Grafana dashboards move and to
// measure the server end-to-end (unlike ./scripts/bench, which runs in-process).
//
// Example:
//
//	go run ./scripts/loadgen -url http://localhost:8080 -concurrency 32 -requests 2000
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	url := flag.String("url", "http://localhost:8080", "server base URL")
	concurrency := flag.Int("concurrency", 16, "number of concurrent workers")
	total := flag.Int("requests", 500, "total requests to send")
	maxTokens := flag.Int("max-tokens", 48, "max_tokens per request")
	stream := flag.Bool("stream", true, "use streaming (measures true TTFT)")
	sharedSys := flag.Bool("shared-system", true, "send a shared system prompt (exercises prefix cache)")
	flag.Parse()

	client := &http.Client{Timeout: 60 * time.Second}
	var sent int64

	ttfts := make([]time.Duration, 0, *total)
	durations := make([]time.Duration, 0, *total)
	var mu sync.Mutex
	var okCount, errCount int64

	start := time.Now()
	var wg sync.WaitGroup
	for w := 0; w < *concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := atomic.AddInt64(&sent, 1)
				if i > int64(*total) {
					return
				}
				ttft, dur, err := doRequest(client, *url, *maxTokens, *stream, *sharedSys, int(i))
				mu.Lock()
				if err != nil {
					errCount++
				} else {
					okCount++
					ttfts = append(ttfts, ttft)
					durations = append(durations, dur)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	wall := time.Since(start)

	fmt.Printf("requests: %d ok, %d errors in %s (%.1f req/s)\n",
		okCount, errCount, wall.Round(time.Millisecond), float64(okCount)/wall.Seconds())
	printPercentiles("TTFT", ttfts)
	printPercentiles("total latency", durations)
}

// doRequest sends one chat completion and returns TTFT and total duration.
func doRequest(client *http.Client, base string, maxTokens int, stream, sharedSys bool, i int) (ttft, dur time.Duration, err error) {
	msgs := []map[string]string{}
	if sharedSys {
		msgs = append(msgs, map[string]string{"role": "system", "content": "You are a helpful, concise assistant. Follow instructions carefully and avoid speculation."})
	}
	msgs = append(msgs, map[string]string{"role": "user", "content": fmt.Sprintf("Question %d: %s", i, phrases[rand.Intn(len(phrases))])})

	body, _ := json.Marshal(map[string]any{
		"model": "mock-7b", "stream": stream, "max_tokens": maxTokens, "messages": msgs,
	})

	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return 0, 0, fmt.Errorf("status %d", resp.StatusCode)
	}

	if stream {
		sc := bufio.NewScanner(resp.Body)
		first := true
		for sc.Scan() {
			line := sc.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			if first {
				ttft = time.Since(start)
				first = false
			}
			if strings.TrimPrefix(line, "data: ") == "[DONE]" {
				break
			}
		}
	} else {
		io_copy_discard(resp.Body)
		ttft = time.Since(start) // no streaming → TTFT == total
	}
	return ttft, time.Since(start), nil
}

func printPercentiles(label string, ds []time.Duration) {
	if len(ds) == 0 {
		fmt.Printf("%s: no samples\n", label)
		return
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
	p := func(q float64) time.Duration { return ds[int(float64(len(ds)-1)*q)].Round(time.Millisecond) }
	fmt.Printf("%-14s p50=%s  p90=%s  p95=%s  p99=%s  max=%s\n",
		label, p(0.50), p(0.90), p(0.95), p(0.99), ds[len(ds)-1].Round(time.Millisecond))
}

// io_copy_discard drains a reader without importing io just for one call.
func io_copy_discard(r interface{ Read([]byte) (int, error) }) {
	buf := make([]byte, 4096)
	for {
		if _, err := r.Read(buf); err != nil {
			return
		}
	}
}

var phrases = []string{
	"explain continuous batching",
	"what is a KV cache",
	"summarize prefix caching",
	"how does streaming work",
	"describe admission control",
}
