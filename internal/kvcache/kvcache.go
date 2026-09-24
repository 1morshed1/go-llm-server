// Package kvcache models the memory that dominates LLM serving: the KV cache.
//
// WHAT THE KV CACHE IS
//
// When a transformer generates token N, it must attend over the keys and values
// (the "KV") of tokens 1..N-1. Recomputing those every step would be quadratic,
// so they are computed once and CACHED. That cache is large — for a 7B model it
// can be tens to hundreds of KB *per token* — so on real hardware the KV cache,
// not compute, is usually what limits how many requests you can run at once. If
// you understand nothing else about LLM serving, understand this: **KV-cache
// capacity is the batch-size ceiling.**
//
// TWO IDEAS THIS PACKAGE IMPLEMENTS
//
//  1. PAGED ALLOCATION (like vLLM's PagedAttention). Instead of one big
//     contiguous buffer per request, memory is split into fixed-size BLOCKS
//     (e.g. 16 tokens each). A request holds a list of blocks. This kills
//     fragmentation and lets blocks be shared between requests — which enables:
//
//  2. PREFIX CACHING. If 100 requests share the same system prompt, its KV is
//     identical for all of them. We compute it once, store the blocks in a
//     shared prefix cache keyed by a hash of the prefix's token IDs, and let
//     later requests REUSE those blocks instead of recomputing (which also skips
//     that portion of prefill, lowering TTFT — see docs/04).
//
// When memory is tight, unreferenced prefix-cache entries are evicted LRU.
//
// This is a faithful-but-simplified model: we account in whole blocks, reserve
// a fixed amount of room for generation up front, and never split a KV tensor
// across the wire. See docs/04-kv-cache.md for the full discussion and the
// simplifications we made (and why).
package kvcache

import (
	"errors"
	"hash/fnv"
	"sync"
	"time"

	"github.com/1morshed1/go-llm-server/internal/domain"
)

// ErrNoCapacity is returned by Allocate when there is not enough free KV memory
// even after evicting everything evictable. The batcher treats this as "can't
// admit this request right now" and leaves it queued.
var ErrNoCapacity = errors.New("kvcache: not enough capacity")

// Config configures the cache's size and behaviour.
type Config struct {
	// BlockSize is how many tokens' worth of KV one block holds.
	BlockSize int
	// NumBlocks is the total number of blocks (the whole memory budget).
	NumBlocks int
	// GenerationReserveTokens is how much room to set aside per request for the
	// tokens it will GENERATE (prefill only covers the prompt). Real servers
	// grow lazily and preempt on exhaustion; we reserve up front so capacity
	// accounting is simple and admission control can reason about it.
	GenerationReserveTokens int
	// EnablePrefixCache toggles prefix caching (useful for A/B benchmarks).
	EnablePrefixCache bool
}

// DefaultConfig: 512 blocks × 16 tokens = 8192 tokens of KV budget.
func DefaultConfig() Config {
	return Config{
		BlockSize:               16,
		NumBlocks:               512,
		GenerationReserveTokens: 128,
		EnablePrefixCache:       true,
	}
}

// Handle is a request's allocation. The batcher stores it on
// domain.Request.KVHandle and passes it back to Free when the request finishes.
type Handle struct {
	privateBlocks   int    // blocks owned exclusively by this request
	prefixHash      uint64 // the prefix entry this handle references (0 = none)
	hasPrefixRef    bool   // whether this handle holds a ref on a prefix entry
	CachedPrefixLen int    // prompt tokens served from cache (0 on miss / no prefix)
}

// prefixEntry is one cached shared prefix.
type prefixEntry struct {
	hash     uint64
	blocks   int       // blocks its KV occupies (stay used until evicted)
	tokenLen int       // how many prompt tokens this prefix covers
	refcount int       // active requests currently reusing it
	lastUsed time.Time // for LRU eviction when refcount == 0
}

// Manager owns the block budget and the prefix cache. All methods are safe for
// concurrent use; in practice the batcher goroutine is the main caller.
type Manager struct {
	cfg Config

	mu       sync.Mutex
	used     int // blocks currently occupied (private + all prefix entries)
	prefixes map[uint64]*prefixEntry

	// counters for observability
	hits   int64
	misses int64
}

// New builds a Manager.
func New(cfg Config) *Manager {
	if cfg.BlockSize <= 0 || cfg.NumBlocks <= 0 {
		cfg = DefaultConfig()
	}
	return &Manager{cfg: cfg, prefixes: make(map[uint64]*prefixEntry)}
}

// blocksFor returns how many blocks are needed to hold n tokens.
func (m *Manager) blocksFor(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + m.cfg.BlockSize - 1) / m.cfg.BlockSize
}

func (m *Manager) freeBlocks() int { return m.cfg.NumBlocks - m.used }

// Allocate reserves KV memory for a request, reusing a cached prefix when
// possible. On success it returns a Handle and the number of prompt tokens that
// were served from the cache (which the batcher passes to Engine.Prefill so the
// prefill cost skips the cached portion). On failure it returns ErrNoCapacity.
func (m *Manager) Allocate(req *domain.Request) (*Handle, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	promptTokens := len(req.PromptTokens)
	prefixTokens := req.PrefixLen
	if prefixTokens > promptTokens {
		prefixTokens = promptTokens
	}
	genBlocks := m.blocksFor(m.cfg.GenerationReserveTokens)

	// --- try a prefix-cache hit ---
	var (
		hitEntry        *prefixEntry
		cachedPrefixLen int
	)
	var hash uint64
	if m.cfg.EnablePrefixCache && prefixTokens > 0 {
		hash = hashTokenIDs(req.PromptTokens[:prefixTokens])
		if e, ok := m.prefixes[hash]; ok {
			hitEntry = e
			cachedPrefixLen = e.tokenLen
		}
	}

	h := &Handle{CachedPrefixLen: cachedPrefixLen}

	if hitEntry != nil {
		// HIT: the prefix's blocks are already resident and shared. We only need
		// private blocks for the non-prefix prompt tokens plus generation room.
		privateBlocks := m.blocksFor(promptTokens-cachedPrefixLen) + genBlocks
		if err := m.ensureFree(privateBlocks); err != nil {
			return nil, err
		}
		m.used += privateBlocks
		hitEntry.refcount++
		hitEntry.lastUsed = time.Now()
		h.privateBlocks = privateBlocks
		h.prefixHash = hash
		h.hasPrefixRef = true
		m.hits++
		return h, nil
	}

	// MISS (or prefix caching off / no declared prefix): allocate blocks for the
	// whole prompt plus generation room.
	if m.cfg.EnablePrefixCache && prefixTokens > 0 {
		m.misses++
	}
	prefixBlocks := 0
	if m.cfg.EnablePrefixCache && prefixTokens > 0 {
		prefixBlocks = m.blocksFor(prefixTokens)
	}
	restPromptBlocks := m.blocksFor(promptTokens-prefixTokens) + genBlocks
	total := prefixBlocks + restPromptBlocks
	if err := m.ensureFree(total); err != nil {
		return nil, err
	}
	m.used += total
	h.privateBlocks = restPromptBlocks

	// Promote the prefix into the shared cache so FUTURE requests hit it. The
	// prefix blocks are now owned by the cache entry (refcount 1 = this request).
	if prefixBlocks > 0 {
		m.prefixes[hash] = &prefixEntry{
			hash:     hash,
			blocks:   prefixBlocks,
			tokenLen: prefixTokens,
			refcount: 1,
			lastUsed: time.Now(),
		}
		h.prefixHash = hash
		h.hasPrefixRef = true
	}
	return h, nil
}

// Free releases a request's allocation. Private blocks are returned to the pool
// immediately; a referenced prefix entry has its refcount decremented and stays
// resident (evictable later) so subsequent requests can still hit it.
func (m *Manager) Free(h *Handle) {
	if h == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.used -= h.privateBlocks
	if m.used < 0 {
		m.used = 0
	}
	if h.hasPrefixRef {
		if e, ok := m.prefixes[h.prefixHash]; ok {
			e.refcount--
			e.lastUsed = time.Now()
			if e.refcount < 0 {
				e.refcount = 0
			}
		}
	}
}

// ensureFree evicts LRU unreferenced prefix entries until `need` blocks are
// free, or returns ErrNoCapacity. Caller must hold m.mu.
func (m *Manager) ensureFree(need int) error {
	if need <= m.freeBlocks() {
		return nil
	}
	for need > m.freeBlocks() {
		victim := m.lruEvictable()
		if victim == nil {
			return ErrNoCapacity // everything left is pinned by active requests
		}
		m.used -= victim.blocks
		delete(m.prefixes, victim.hash)
	}
	return nil
}

// lruEvictable returns the least-recently-used prefix entry with refcount 0, or
// nil if none can be evicted. Caller must hold m.mu.
func (m *Manager) lruEvictable() *prefixEntry {
	var victim *prefixEntry
	for _, e := range m.prefixes {
		if e.refcount != 0 {
			continue
		}
		if victim == nil || e.lastUsed.Before(victim.lastUsed) {
			victim = e
		}
	}
	return victim
}

// Stats is a snapshot for metrics and health endpoints.
type Stats struct {
	Capacity     int     // total blocks
	Used         int     // blocks in use
	Utilization  float64 // Used / Capacity in [0,1]
	PrefixCount  int     // number of cached prefixes
	Hits         int64
	Misses       int64
	HitRate      float64 // Hits / (Hits + Misses)
	BlockSize    int
	TokenBudget  int // Capacity × BlockSize
}

// Stats returns a consistent snapshot.
func (m *Manager) Stats() Stats {
	m.mu.Lock()
	defer m.mu.Unlock()
	total := m.hits + m.misses
	var rate float64
	if total > 0 {
		rate = float64(m.hits) / float64(total)
	}
	util := float64(m.used) / float64(m.cfg.NumBlocks)
	return Stats{
		Capacity:    m.cfg.NumBlocks,
		Used:        m.used,
		Utilization: util,
		PrefixCount: len(m.prefixes),
		Hits:        m.hits,
		Misses:      m.misses,
		HitRate:     rate,
		BlockSize:   m.cfg.BlockSize,
		TokenBudget: m.cfg.NumBlocks * m.cfg.BlockSize,
	}
}

// hashTokenIDs hashes a token slice by its IDs (order-sensitive). Identical
// prefixes therefore hash identically — the basis of prefix-cache hits.
func hashTokenIDs(tokens []domain.Token) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, t := range tokens {
		v := uint64(t.ID)
		for i := 0; i < 8; i++ {
			buf[i] = byte(v >> (8 * i))
		}
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}
