// Command bench measures the serving layer's behaviour in-process (no HTTP, no
// Docker), so the numbers are reproducible on any machine and depend only on the
// mock engine's configured timing model. It produces the tables in
// docs/10-benchmarks.md.
//
// It runs three experiments:
//
//  1. Throughput vs. batch size — the fundamental batching curve.
//  2. Continuous vs. naive batching — the headline 2–3× win, on identical
//     mixed-length traffic.
//  3. Prefix caching on vs. off — TTFT and hit-rate effect for a shared system
//     prompt.
//
// Run:  go run ./scripts/bench
package main

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/1morshed1/go-llm-server/internal/batcher"
	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
	"github.com/1morshed1/go-llm-server/internal/metrics"
	"github.com/1morshed1/go-llm-server/internal/scheduler"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// realisticMock returns the timing model used for all benchmarks. These are the
// numbers that determine the results; they're chosen to resemble a small model
// on modest hardware. Natural EOS is pushed far out so each request generates
// exactly its MaxTokens — making the experiments precise and deterministic.
func realisticMock() engine.MockConfig {
	c := engine.DefaultMockConfig()
	c.PrefillPerToken = 400 * time.Microsecond
	c.StepBase = 18 * time.Millisecond
	c.StepPerSeq = 1 * time.Millisecond
	c.MinNewTokens = 100000
	c.MaxNewTokens = 100001
	return c
}

func main() {
	fmt.Println("# Benchmark results (mock engine)")
	fmt.Printf("\nTiming model: StepBase=%v, StepPerSeq=%v, PrefillPerToken=%v\n",
		realisticMock().StepBase, realisticMock().StepPerSeq, realisticMock().PrefillPerToken)

	throughputVsBatch()
	continuousVsNaive()
	prefixCacheEffect()
}

// ---------------------------------------------------------------------------
// Experiment 1: throughput vs. batch size
// ---------------------------------------------------------------------------

func throughputVsBatch() {
	fmt.Println("\n## 1. Throughput vs. batch size")
	fmt.Println("\nAll requests generate 48 tokens; enough are offered to keep the batch full.")
	fmt.Println("\n| Max batch | Tokens/sec | Speedup vs batch=1 |")
	fmt.Println("|-----------|-----------|--------------------|")

	base := 0.0
	for _, bs := range []int{1, 4, 8, 16} {
		tps := runContinuous(benchParams{
			numRequests: bs * 6, // enough offered load to keep the batch full
			maxBatch:    bs,
			maxTokens:   func(int) int { return 48 },
			sharedSys:   false,
		}).tokensPerSec
		if bs == 1 {
			base = tps
		}
		fmt.Printf("| %9d | %9.1f | %17.2fx |\n", bs, tps, tps/base)
	}
}

// ---------------------------------------------------------------------------
// Experiment 2: continuous vs. naive batching
// ---------------------------------------------------------------------------

func continuousVsNaive() {
	fmt.Println("\n## 2. Continuous vs. naive batching (mixed request lengths)")
	fmt.Println("\n64 requests, lengths mixed 90% short (16 tok) / 10% long (128 tok), max batch 16.")

	params := benchParams{
		numRequests: 64,
		maxBatch:    16,
		maxTokens:   mixedLengths,
		sharedSys:   false,
	}

	cont := runContinuous(params)
	naive := runNaive(params)

	fmt.Println("\n| Strategy    | Wall time | Tokens/sec |")
	fmt.Println("|-------------|-----------|-----------|")
	fmt.Printf("| naive       | %9s | %9.1f |\n", round(naive.wall), naive.tokensPerSec)
	fmt.Printf("| continuous  | %9s | %9.1f |\n", round(cont.wall), cont.tokensPerSec)
	fmt.Printf("\n**Continuous batching is %.2fx faster** on this traffic.\n",
		cont.tokensPerSec/naive.tokensPerSec)
}

// ---------------------------------------------------------------------------
// Experiment 3: prefix caching
// ---------------------------------------------------------------------------

func prefixCacheEffect() {
	fmt.Println("\n## 3. Prefix caching (shared system prompt)")
	fmt.Println("\n64 requests sharing one system prompt; 24 tokens each; max batch 8.")

	on := runContinuous(benchParams{numRequests: 64, maxBatch: 8, maxTokens: func(int) int { return 24 }, sharedSys: true, prefixCache: true})
	off := runContinuous(benchParams{numRequests: 64, maxBatch: 8, maxTokens: func(int) int { return 24 }, sharedSys: true, prefixCache: false})

	fmt.Println("\n| Prefix cache | Avg TTFT | Hit rate | Tokens/sec |")
	fmt.Println("|--------------|----------|----------|-----------|")
	fmt.Printf("| off          | %8s |     0.0%% | %9.1f |\n", round(off.avgTTFT), off.tokensPerSec)
	fmt.Printf("| on           | %8s | %7.1f%% | %9.1f |\n", round(on.avgTTFT), on.hitRate*100, on.tokensPerSec)
	fmt.Printf("\n**Prefix caching cut average TTFT by %.0f%%** by skipping prefill of the shared prompt.\n",
		(1-on.avgTTFT.Seconds()/off.avgTTFT.Seconds())*100)
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

type benchParams struct {
	numRequests int
	maxBatch    int
	maxTokens   func(i int) int
	sharedSys   bool
	prefixCache bool
}

type result struct {
	wall         time.Duration
	tokensPerSec float64
	avgTTFT      time.Duration
	hitRate      float64
}

func mixedLengths(i int) int {
	if i%10 == 0 {
		return 128 // 10% long requests
	}
	return 16 // 90% short
}

const systemPrompt = "You are a meticulous assistant. Answer accurately and concisely, cite assumptions, avoid speculation, and prefer clarity over cleverness in every response you give to the user today."

func makeReq(tok tokenizer.Tokenizer, id int, shared bool, maxTokens int) *domain.Request {
	user := fmt.Sprintf("Question number %d please", id)
	prompt := user
	prefixLen := 0
	if shared {
		prompt = systemPrompt + "\n" + user
		prefixLen = len(tok.Encode(systemPrompt + "\n"))
	}
	return &domain.Request{
		ID:           fmt.Sprintf("r%d", id),
		Model:        "mock-7b",
		Prompt:       prompt,
		PromptTokens: tok.Encode(prompt),
		PrefixLen:    prefixLen,
		Params:       domain.SamplingParams{MaxTokens: maxTokens},
		Ctx:          context.Background(),
		Stream:       make(chan domain.StreamEvent, maxTokens+8),
	}
}

// runContinuous drives requests through the real production batcher.
func runContinuous(p benchParams) result {
	tok := tokenizer.NewSimple(32000)
	eng := engine.NewMock(realisticMock(), tok)
	kvCfg := kvcache.DefaultConfig()
	kvCfg.NumBlocks = 8192 // large, so KV memory isn't the bottleneck here
	kvCfg.EnablePrefixCache = p.prefixCache
	kv := kvcache.New(kvCfg)
	sched := scheduler.New(scheduler.Config{MaxQueueDepth: 100000, MaxWait: time.Hour, ReapInterval: time.Hour})
	b := batcher.New(batcher.Config{MaxBatchSize: p.maxBatch, IdleTick: 5 * time.Millisecond}, eng, kv, sched, metrics.Noop{})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { b.Run(ctx); close(done) }()

	reqs := make([]*domain.Request, p.numRequests)
	for i := range reqs {
		reqs[i] = makeReq(tok, i, p.sharedSys, p.maxTokens(i))
	}

	start := time.Now()
	var wg sync.WaitGroup
	ttfts := make([]time.Duration, p.numRequests)
	totalTokens := 0
	var mu sync.Mutex
	for idx, req := range reqs {
		if err := sched.Submit(req); err != nil {
			panic(err)
		}
		wg.Add(1)
		go func(i int, r *domain.Request) {
			defer wg.Done()
			n := 0
			var first time.Duration
			for ev := range r.Stream {
				if ev.Done {
					break
				}
				if n == 0 {
					first = time.Since(start)
				}
				n++
			}
			mu.Lock()
			ttfts[i] = first
			totalTokens += n
			mu.Unlock()
		}(idx, req)
	}
	wg.Wait()
	wall := time.Since(start)

	cancel()
	<-done
	sched.Close()

	return result{
		wall:         wall,
		tokensPerSec: float64(totalTokens) / wall.Seconds(),
		avgTTFT:      avg(ttfts),
		hitRate:      kv.Stats().HitRate,
	}
}

// runNaive models classic STATIC batching: requests are processed in fixed
// groups; a group runs until ALL its members finish before the next group
// starts, and finished members keep occupying (padding) the batch — so no new
// request joins mid-flight. This is the behaviour continuous batching replaces.
func runNaive(p benchParams) result {
	tok := tokenizer.NewSimple(32000)
	eng := engine.NewMock(realisticMock(), tok)
	kvCfg := kvcache.DefaultConfig()
	kvCfg.NumBlocks = 8192
	kv := kvcache.New(kvCfg)
	ctx := context.Background()

	reqs := make([]*domain.Request, p.numRequests)
	for i := range reqs {
		reqs[i] = makeReq(tok, i, p.sharedSys, p.maxTokens(i))
	}

	start := time.Now()
	totalTokens := 0
	for i := 0; i < len(reqs); i += p.maxBatch {
		end := i + p.maxBatch
		if end > len(reqs) {
			end = len(reqs)
		}
		group := reqs[i:end]

		seqs := make([]*engine.Sequence, len(group))
		limits := make([]int, len(group))
		produced := make([]int, len(group))
		for j, r := range group {
			h, _ := kv.Allocate(r)
			r.KVHandle = h
			s, _ := eng.Prefill(ctx, r, len(r.PromptTokens)-h.CachedPrefixLen)
			seqs[j] = s
			limits[j] = r.Params.MaxTokens
		}

		remaining := len(group)
		for remaining > 0 {
			// Step the FULL group every iteration — finished members are padded,
			// not removed, so the batch width (and its cost) stays high. That is
			// exactly the waste continuous batching eliminates.
			res, _ := eng.Step(ctx, seqs)
			for j := range group {
				if produced[j] >= limits[j] {
					continue // already done; contributes cost but no useful token
				}
				produced[j]++
				totalTokens++
				if produced[j] >= limits[j] || res[j].EOS {
					remaining--
				}
			}
		}
		for j, r := range group {
			eng.Release(seqs[j])
			kv.Free(r.KVHandle.(*kvcache.Handle))
		}
	}
	wall := time.Since(start)
	return result{wall: wall, tokensPerSec: float64(totalTokens) / wall.Seconds()}
}

func avg(ds []time.Duration) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	var sum time.Duration
	for _, d := range ds {
		sum += d
	}
	return sum / time.Duration(len(ds))
}

// round trims a duration to milliseconds for tidy tables.
func round(d time.Duration) time.Duration { return d.Round(time.Millisecond) }
