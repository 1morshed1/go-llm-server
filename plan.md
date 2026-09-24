## Project D: LLM Inference Server from Scratch (Go)

### Why This Project

This is the single best project for bridging your worlds. You are building the software that sits between a raw model file (GGUF/ONNX) and the HTTP API that applications call. Think of it as a simplified vLLM or Ollama — but you're writing the serving layer in Go.

You already understand the ML side: tokenization, attention, KV caches, batching. Most Go engineers don't. And you already use inference servers (Ollama, vLLM, TGI) as black boxes. This project opens that box.

What makes it valuable: every company deploying LLMs needs inference infrastructure. The engineers who understand *both* the model internals and the serving systems are the ones architecting these deployments. Right now, your CV says "I call APIs." After this project, it says "I understand what happens when I call those APIs, and I can build the serving layer myself."

You won't implement the actual matrix multiplications (that's C++/CUDA work). Instead, you'll use `llama.cpp` or ONNX Runtime as the compute backend and build everything around it in Go: the HTTP/gRPC API, request queuing, continuous batching, KV cache management, streaming, model loading, and health monitoring.

### Architecture

```
  Clients                                                                     
   │                                                                          
   │  POST /v1/chat/completions (OpenAI-compatible)                           
   │  or gRPC streaming                                                       
   ▼                                                                          
┌──────────────────────────────────────────────────────────────────────────────┐
│  Go Inference Server                                                         │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  API Layer (Gin)                                                     │   │
│  │                                                                      │   │
│  │  - OpenAI-compatible REST API                                        │   │
│  │  - gRPC streaming endpoint                                           │   │
│  │  - Health check, model info endpoints                                │   │
│  │  - Request validation (max tokens, temperature bounds)               │   │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Request Scheduler                                                   │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────┐          │   │
│  │  │  Priority Queue                                        │          │   │
│  │  │                                                        │          │   │
│  │  │  - Incoming requests sorted by arrival time            │          │   │
│  │  │  - Priority boost for short prompts (faster to serve)  │          │   │
│  │  │  - Admission control: reject if queue > threshold      │          │   │
│  │  │  - Request timeout: cancel if waiting > 30s            │          │   │
│  │  └────────────────────────────────────────────────────────┘          │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────┐          │   │
│  │  │  Continuous Batcher                                    │          │   │
│  │  │                                                        │          │   │
│  │  │  - Groups requests into batches for parallel inference │          │   │
│  │  │  - Dynamic batch size based on available KV cache      │          │   │
│  │  │  - New requests join mid-batch (continuous batching)   │          │   │
│  │  │  - Completed requests exit without blocking others     │          │   │
│  │  └────────────────────────────────────────────────────────┘          │   │
│  └──────────────────────────────────┬───────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Inference Engine (wraps llama.cpp via CGo or ONNX Runtime)         │   │
│  │                                                                      │   │
│  │  ┌─────────────────┐    ┌──────────────────────────────────┐        │   │
│  │  │  Tokenizer       │    │  KV Cache Manager               │        │   │
│  │  │                  │    │                                  │        │   │
│  │  │  - BPE/SentPiece │    │  - Per-request cache allocation │        │   │
│  │  │  - Token count   │    │  - Cache eviction (LRU) when    │        │   │
│  │  │    estimation    │    │    memory pressure hits         │        │   │
│  │  │  - Special token │    │  - Prefix caching: reuse cache  │        │   │
│  │  │    handling      │    │    for shared system prompts    │        │   │
│  │  └─────────────────┘    └──────────────────────────────────┘        │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────┐       │   │
│  │  │  Model Manager                                           │       │   │
│  │  │                                                          │       │   │
│  │  │  - Load/unload GGUF models from disk                     │       │   │
│  │  │  - Support multiple models simultaneously                │       │   │
│  │  │  - Memory-map model weights (mmap)                       │       │   │
│  │  │  - Hot-swap: load new model version without downtime     │       │   │
│  │  └──────────────────────────────────────────────────────────┘       │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Observability                                                       │   │
│  │                                                                      │   │
│  │  Prometheus metrics:                                                 │   │
│  │  - tokens_generated_total (counter)                                  │   │
│  │  - tokens_per_second (gauge, per-request and aggregate)              │   │
│  │  - time_to_first_token_seconds (histogram)                           │   │
│  │  - request_queue_depth (gauge)                                       │   │
│  │  - kv_cache_utilization_percent (gauge)                              │   │
│  │  - batch_size (histogram)                                            │   │
│  │  - model_memory_bytes (gauge)                                        │   │
│  │                                                                      │   │
│  │  Grafana dashboard: throughput, latency, cache utilization, queue    │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Deployment: Docker + Kubernetes (with GPU node pools if available)          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### What Makes This Hard (and Impressive)

**Continuous Batching:**

Naive batching waits for N requests to arrive, processes them all, and returns. The problem: if request A needs 50 tokens and request B needs 500 tokens, A waits for B to finish. Continuous batching solves this — when A finishes generating its last token, it exits the batch immediately and a new request C joins in its place:

```go
type ContinuousBatcher struct {
    activeBatch   map[string]*InferenceRequest  // request_id -> request
    pendingQueue  chan *InferenceRequest
    maxBatchSize  int
    mu            sync.Mutex
}

func (b *ContinuousBatcher) Run(ctx context.Context, engine *InferenceEngine) {
    for {
        b.mu.Lock()
        
        // Fill empty slots in the batch from the pending queue
        for len(b.activeBatch) < b.maxBatchSize {
            select {
            case req := <-b.pendingQueue:
                b.activeBatch[req.ID] = req
                engine.AllocateKVCache(req)
            default:
                break  // no more pending requests
            }
        }
        
        if len(b.activeBatch) == 0 {
            b.mu.Unlock()
            time.Sleep(1 * time.Millisecond)
            continue
        }
        
        // Run one forward pass for the entire batch
        // (all active requests advance by one token)
        outputs := engine.ForwardPass(b.activeBatch)
        
        // Check which requests are done
        for id, output := range outputs {
            req := b.activeBatch[id]
            req.GeneratedTokens = append(req.GeneratedTokens, output.Token)
            
            // Stream the token to the client
            req.StreamChan <- output.Token
            
            if output.IsEOS || len(req.GeneratedTokens) >= req.MaxTokens {
                // This request is complete — remove from batch
                close(req.StreamChan)
                engine.FreeKVCache(req)
                delete(b.activeBatch, id)
            }
        }
        
        b.mu.Unlock()
    }
}
```

This is how vLLM works. Understanding and implementing it puts you in a very small group of engineers.

**Prefix Caching (KV Cache Reuse):**

If 100 requests share the same system prompt, the KV cache for that prompt is computed 100 times. Prefix caching computes it once and reuses it:

```go
type KVCacheManager struct {
    prefixCache  map[uint64]*KVCacheBlock  // hash(token_sequence) -> cached KV
    blockPool    []*KVCacheBlock
    totalMemory  int64
    usedMemory   int64
}

func (m *KVCacheManager) AllocateForRequest(req *InferenceRequest) *KVCacheHandle {
    // Check if the system prompt's KV is already cached
    systemPromptHash := hashTokens(req.SystemPromptTokens)
    
    if cached, exists := m.prefixCache[systemPromptHash]; exists {
        // Reuse cached KV for the system prompt portion
        // Only compute KV for the user message (new tokens)
        handle := m.allocateBlocks(req.UserMessageTokens)
        handle.PrefixBlocks = cached
        return handle
    }
    
    // No cache hit — allocate full KV
    handle := m.allocateBlocks(req.AllTokens)
    
    // Cache the system prompt portion for future reuse
    m.prefixCache[systemPromptHash] = handle.ExtractPrefix(len(req.SystemPromptTokens))
    
    return handle
}
```

### Benchmarks to Document

- **Tokens/second:** at various batch sizes (1, 4, 8, 16) with a 7B parameter model.
- **Time to First Token (TTFT):** the latency before the first token starts streaming. Target: under 200ms for a 7B model.
- **Throughput improvement from continuous batching:** compare naive batching vs continuous batching with the same traffic pattern. Show 2-3x improvement.
- **Prefix cache hit rate:** with 10 unique system prompts across 1000 requests, show the memory savings and latency improvement.

### Why This Bridges Your Worlds

You already know what tokenization is, what a KV cache is, what attention does. Most Go engineers building inference servers are learning this from scratch. Your ML background means you can make smarter architectural decisions (you know *why* prefix caching matters, not just how to implement it). And the serving layer itself is pure Go systems engineering — goroutines, memory management, gRPC streaming, Prometheus metrics.

---
---