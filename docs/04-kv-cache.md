# 04 — The KV cache & prefix caching

> Code: `internal/kvcache/kvcache.go`, `internal/kvcache/kvcache_test.go`

This is the most important systems concept in LLM serving. Take your time here.

## The idea: why a cache exists at all

When a transformer generates token *N*, its attention layers look back at the
**keys and values** ("KV") of every earlier token, 1..*N*-1. Those K and V
tensors depend only on the tokens that came before, so recomputing them at every
step would be wasteful — quadratic in sequence length. Instead they are computed
once per token and **kept in the KV cache**. Generation then reads the cache and
appends one new entry each step.

The catch: **the KV cache is big.** Its size is
`num_layers × 2 × num_kv_heads × head_dim × bytes × tokens`. For a 7B model
that's on the order of hundreds of KB *per token*. A single 2000-token
conversation can need hundreds of MB. On a real GPU you run out of KV-cache
memory long before you run out of compute.

> **The one sentence to remember:** KV-cache capacity — not FLOPs — is what
> caps how many requests you can serve concurrently. Every design choice in a
> serving layer is downstream of that.

## Idea 1: paged allocation (PagedAttention)

The naive approach gives each request one big contiguous KV buffer sized for its
worst case. That wastes memory (you reserve for the max but usually use less) and
fragments it (a finished request leaves a hole the wrong shape for the next).

vLLM's insight — **PagedAttention** — is to borrow virtual memory's trick: chop
KV memory into fixed-size **blocks** (e.g. 16 tokens each) and give each request
a *list* of blocks. Now:

- no fragmentation — any free block fits any request;
- requests grow by grabbing another block, not by reallocating;
- and blocks can be **shared** between requests, which unlocks idea 2.

Our `Manager` models exactly this: `NumBlocks` blocks of `BlockSize` tokens
each. A request's `Handle` records how many `privateBlocks` it holds. We account
in whole blocks, just like the real thing.

## Idea 2: prefix caching (KV reuse)

Picture a chatbot: 1,000 requests all begin with the same 300-token system
prompt. That prompt's KV is **identical** for every request — computing it 1,000
times is pure waste, both in prefill time (→ TTFT) and in memory.

Prefix caching computes it **once**, stores those blocks in a shared cache keyed
by a hash of the prefix's token IDs, and lets later requests **reuse** them:

```
request 1  [system prompt KV .......][user KV]   ← MISS: compute & cache the prefix
request 2  [   reuse cached prefix   ][user KV]   ← HIT: skip prefill of the prefix
request 3  [   reuse cached prefix   ][user KV]   ← HIT
```

In our `Allocate`:

1. If the request declares a prefix (`req.PrefixLen > 0`), hash the IDs of those
   leading tokens.
2. **Hit** (`hash` already in the cache): reuse the shared blocks — allocate
   private blocks only for the *non-prefix* tokens plus generation room, bump the
   entry's refcount, and return `CachedPrefixLen = prefix length`. The batcher
   passes that to `Engine.Prefill`, so prefill *skips the cached tokens* →
   lower TTFT.
3. **Miss**: allocate blocks for the whole prompt, then **promote** the prefix's
   blocks into the shared cache so the *next* request hits.

The `CachedPrefixLen` returned on a hit is the bridge between this package and
the [engine](02-the-engine-interface.md): fewer uncached tokens ⇒ cheaper
prefill ⇒ measurably lower TTFT. That's the win, made concrete.

## Eviction: LRU under pressure

Cached prefixes occupy memory. When a new allocation needs blocks and none are
free, `ensureFree` evicts the **least-recently-used prefix entry whose refcount
is 0** (i.e. no active request is using it), and repeats until there's room. If
everything left is *pinned* by active requests, `Allocate` returns
`ErrNoCapacity` — and the batcher simply leaves the request queued and tries
again next iteration. Refcounting is what guarantees we never evict a prefix a
running request still depends on.

## The accounting, precisely

- `used` = private blocks held by active requests **+** blocks held by all
  prefix-cache entries (referenced or not, until evicted).
- `free = NumBlocks - used`.
- A **hit** adds only `blocksFor(promptTokens − cachedPrefixLen) + genReserve`.
- A **miss** adds `blocksFor(prefixTokens) + blocksFor(rest) + genReserve`, and
  the `blocksFor(prefixTokens)` portion becomes the shared entry.
- `Free` returns the private blocks immediately and decrements the prefix
  refcount; the prefix entry lingers (evictable) so future requests can still
  hit it.

`Stats()` exposes `Utilization`, `HitRate`, `PrefixCount`, and the token budget —
these feed the Prometheus gauges in [doc 08](08-observability.md).

## Decisions & trade-offs

- **Reserve generation room up front (`GenerationReserveTokens`).** Real vLLM
  grows a sequence's blocks lazily and *preempts* (evicts a running sequence back
  to the queue) when it runs out mid-generation. That's powerful but intricate.
  We reserve a fixed amount per request so capacity is predictable and admission
  control can reason about it. The cost is some utilization (we may reserve more
  than a short generation uses); the benefit is a model you can hold in your
  head. This is called out as a deliberate simplification.
- **Whole-block accounting.** We never split a block across requests. Simpler,
  and it's how PagedAttention works too.
- **Hash the prefix, carry no bytes.** We store block *counts*, not actual KV
  bytes (there are none — the engine is simulated). A real integration would map
  each `Handle` to concrete block indices in device memory; the bookkeeping here
  is exactly what you'd keep alongside that.
- **Prefix must be declared by the caller (`PrefixLen`).** We let the API layer
  mark the system-prompt portion as the cacheable prefix, rather than trying to
  auto-detect the longest shared prefix across live requests (which vLLM does
  with a radix tree). Declaring it keeps the hashing O(prefix) and the code
  legible; auto-detection is noted as a natural extension.
