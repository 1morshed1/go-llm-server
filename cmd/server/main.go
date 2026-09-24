// Command server is the entry point: it wires every component together and runs
// the HTTP server with graceful shutdown.
//
// Read this file LAST. Once you understand the pieces (domain, tokenizer,
// engine, kvcache, scheduler, batcher, server), this is just the assembly: it
// shows the dependency graph and the startup/shutdown order in one screen.
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/1morshed1/go-llm-server/internal/batcher"
	"github.com/1morshed1/go-llm-server/internal/config"
	"github.com/1morshed1/go-llm-server/internal/engine"
	"github.com/1morshed1/go-llm-server/internal/kvcache"
	"github.com/1morshed1/go-llm-server/internal/metrics"
	"github.com/1morshed1/go-llm-server/internal/scheduler"
	"github.com/1morshed1/go-llm-server/internal/server"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

func main() {
	cfg, err := config.Load(os.Args[1:])
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	// --- build the pipeline, bottom-up ---
	tok := tokenizer.NewSimple(cfg.VocabSize)
	eng := engine.NewMock(cfg.Model, tok)
	kv := kvcache.New(cfg.KV)
	sched := scheduler.New(cfg.Scheduler)

	// Metrics: a Prometheus recorder plus its /metrics handler. The serving code
	// only ever sees the metrics.Recorder interface.
	rec := metrics.NewPrometheus()

	batch := batcher.New(cfg.Batcher, eng, kv, sched, rec)

	// The batcher runs in its own goroutine for the life of the process; batchCtx
	// is cancelled during shutdown so it drains cleanly.
	batchCtx, stopBatcher := context.WithCancel(context.Background())
	batcherDone := make(chan struct{})
	go func() {
		batch.Run(batchCtx)
		close(batcherDone)
	}()

	// --- API layer ---
	server.SetQueueFullError(scheduler.ErrQueueFull)
	srv := server.New(sched, eng, tok, rec, server.Options{DefaultMaxTokens: 128})
	router := srv.Router(rec.Handler())

	httpServer := &http.Server{
		Addr:              cfg.Addr,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
	}

	// --- run with graceful shutdown ---
	go func() {
		info := eng.Info()
		log.Printf("go-llm-server listening on %s (model=%s, ctx=%d, max_batch=%d, kv_blocks=%d, prefix_cache=%v)",
			cfg.Addr, info.Name, info.ContextLength, cfg.Batcher.MaxBatchSize, cfg.KV.NumBlocks, cfg.KV.EnablePrefixCache)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("http server: %v", err)
		}
	}()

	// Wait for an interrupt/termination signal.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Println("shutdown: draining...")

	// Shutdown order matters:
	// 1. Stop accepting new HTTP requests (and give in-flight handlers a moment).
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("http shutdown: %v", err)
	}
	// 2. Stop the scheduler (fails anything still queued).
	sched.Close()
	// 3. Cancel the batcher so it fails any active requests and returns.
	stopBatcher()
	<-batcherDone
	log.Println("shutdown: complete")
}
