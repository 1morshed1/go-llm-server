// Package config centralizes all tunables and loads them from flags and
// environment variables.
//
// Philosophy: one place to see every knob, sensible defaults so `go run
// ./cmd/server` just works, and 12-factor-style env overrides so the same
// binary can be configured differently per deployment without a rebuild. Flags
// win over env, env wins over defaults.
package config

import (
	"flag"
	"os"
	"strconv"
	"time"

	"github.com/1morshed1/go-llm-server/internal/batcher"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
	"github.com/1morshed1/go-llm-server/internal/scheduler"
)

// Config aggregates every component's configuration in one struct so the wiring
// in main reads top-to-bottom with no surprises.
type Config struct {
	Addr      string // HTTP listen address, e.g. ":8080"
	GRPCAddr  string // gRPC listen address, e.g. ":50051"; empty disables gRPC
	VocabSize int    // tokenizer vocabulary size
	Model     engine.MockConfig
	KV        kvcache.Config
	Scheduler scheduler.Config
	Batcher   batcher.Config
}

// Load parses flags (falling back to environment variables, then to defaults)
// and returns a fully-populated Config. args is normally os.Args[1:].
func Load(args []string) (Config, error) {
	// Start from every component's own defaults.
	c := Config{
		Addr:      env("ADDR", ":8080"),
		GRPCAddr:  env("GRPC_ADDR", ":50051"),
		VocabSize: envInt("VOCAB_SIZE", 32000),
		Model:     engine.DefaultMockConfig(),
		KV:        kvcache.DefaultConfig(),
		Scheduler: scheduler.DefaultConfig(),
		Batcher:   batcher.DefaultConfig(),
	}
	// Apply env overrides to the nested defaults.
	c.Model.Name = env("MODEL_NAME", c.Model.Name)
	c.Model.ContextLength = envInt("CONTEXT_LENGTH", c.Model.ContextLength)
	c.Model.StepBase = envDur("STEP_BASE", c.Model.StepBase)
	c.Model.StepPerSeq = envDur("STEP_PER_SEQ", c.Model.StepPerSeq)
	c.Model.PrefillPerToken = envDur("PREFILL_PER_TOKEN", c.Model.PrefillPerToken)
	c.KV.NumBlocks = envInt("KV_NUM_BLOCKS", c.KV.NumBlocks)
	c.KV.BlockSize = envInt("KV_BLOCK_SIZE", c.KV.BlockSize)
	c.KV.EnablePrefixCache = envBool("KV_PREFIX_CACHE", c.KV.EnablePrefixCache)
	c.Batcher.MaxBatchSize = envInt("MAX_BATCH_SIZE", c.Batcher.MaxBatchSize)
	c.Scheduler.MaxQueueDepth = envInt("MAX_QUEUE_DEPTH", c.Scheduler.MaxQueueDepth)
	c.Scheduler.MaxWait = envDur("MAX_WAIT", c.Scheduler.MaxWait)

	// Flags override env. Register the most common ones.
	fs := flag.NewFlagSet("go-llm-server", flag.ContinueOnError)
	fs.StringVar(&c.Addr, "addr", c.Addr, "HTTP listen address")
	fs.StringVar(&c.GRPCAddr, "grpc-addr", c.GRPCAddr, "gRPC listen address (empty disables gRPC)")
	fs.StringVar(&c.Model.Name, "model", c.Model.Name, "model name reported by the API")
	fs.IntVar(&c.Batcher.MaxBatchSize, "max-batch", c.Batcher.MaxBatchSize, "maximum concurrent sequences per forward pass")
	fs.IntVar(&c.KV.NumBlocks, "kv-blocks", c.KV.NumBlocks, "number of KV-cache blocks")
	fs.BoolVar(&c.KV.EnablePrefixCache, "prefix-cache", c.KV.EnablePrefixCache, "enable prefix caching")
	fs.IntVar(&c.Scheduler.MaxQueueDepth, "max-queue", c.Scheduler.MaxQueueDepth, "admission-control queue depth")
	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	// Keep tokenizer/engine vocab in sync.
	c.Model.VocabSize = c.VocabSize
	return c, nil
}

// --- tiny env helpers (kept local; not worth a dependency) ---

func env(key, def string) string {
	if v, ok := os.LookupEnv(key); ok {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	if v, ok := os.LookupEnv(key); ok {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

func envDur(key string, def time.Duration) time.Duration {
	if v, ok := os.LookupEnv(key); ok {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
