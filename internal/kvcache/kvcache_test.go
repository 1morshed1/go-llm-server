package kvcache

import (
	"testing"

	"github.com/1morshed1/go-llm-server/internal/domain"
	"github.com/1morshed1/go-llm-server/internal/tokenizer"
)

// helper: build a request whose prompt is `system + user`, declaring the system
// portion as the cacheable prefix.
func mkReq(tok tokenizer.Tokenizer, system, user string) *domain.Request {
	prompt := system + user
	toks := tok.Encode(prompt)
	prefixLen := len(tok.Encode(system))
	return &domain.Request{
		PromptTokens: toks,
		PrefixLen:    prefixLen,
	}
}

func TestPrefixHitSkipsPromptTokens(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	m := New(DefaultConfig())

	sys := "You are a helpful assistant that answers concisely. "
	r1 := mkReq(tok, sys, "What is Go?")
	h1, err := m.Allocate(r1)
	if err != nil {
		t.Fatal(err)
	}
	if h1.CachedPrefixLen != 0 {
		t.Errorf("first request should MISS (CachedPrefixLen=0), got %d", h1.CachedPrefixLen)
	}

	// Second request with the SAME system prompt should hit.
	r2 := mkReq(tok, sys, "What is Rust?")
	h2, err := m.Allocate(r2)
	if err != nil {
		t.Fatal(err)
	}
	if h2.CachedPrefixLen == 0 {
		t.Errorf("second request should HIT and reuse the prefix, got CachedPrefixLen=0")
	}
	if h2.CachedPrefixLen != len(tok.Encode(sys)) {
		t.Errorf("CachedPrefixLen=%d, want %d", h2.CachedPrefixLen, len(tok.Encode(sys)))
	}

	st := m.Stats()
	if st.Hits != 1 || st.Misses != 1 {
		t.Errorf("stats hits=%d misses=%d, want 1/1", st.Hits, st.Misses)
	}
}

func TestFreeReturnsPrivateBlocks(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	m := New(DefaultConfig())
	before := m.Stats().Used

	r := mkReq(tok, "sys prompt ", "hello")
	h, _ := m.Allocate(r)
	if m.Stats().Used <= before {
		t.Fatal("expected Used to grow after Allocate")
	}
	m.Free(h)
	// After free, private blocks are returned; the prefix entry (unreferenced)
	// remains resident but evictable.
	if m.Stats().Used == 0 && m.Stats().PrefixCount == 0 {
		// acceptable, but we expect the prefix entry to linger for reuse
	}
	if m.Stats().PrefixCount != 1 {
		t.Errorf("expected prefix entry to linger after Free, PrefixCount=%d", m.Stats().PrefixCount)
	}
}

func TestEvictionUnderPressure(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	// Tiny cache to force eviction quickly.
	cfg := Config{BlockSize: 4, NumBlocks: 8, GenerationReserveTokens: 4, EnablePrefixCache: true}
	m := New(cfg)

	// Create several distinct prefixes, freeing each so entries are unreferenced
	// and thus evictable. The cache should never exceed capacity.
	for i := 0; i < 20; i++ {
		sys := "prefix number " + string(rune('A'+i)) + " here "
		r := mkReq(tok, sys, "q")
		h, err := m.Allocate(r)
		if err != nil {
			// Out of capacity is acceptable only if everything is pinned; here we
			// free immediately, so it should not happen.
			t.Fatalf("unexpected allocate failure at i=%d: %v", i, err)
		}
		if m.Stats().Used > cfg.NumBlocks {
			t.Fatalf("used %d exceeded capacity %d", m.Stats().Used, cfg.NumBlocks)
		}
		m.Free(h)
	}
}

func TestOutOfCapacityWhenPinned(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	cfg := Config{BlockSize: 4, NumBlocks: 4, GenerationReserveTokens: 4, EnablePrefixCache: true}
	m := New(cfg)

	// Hold onto handles (do not Free) so blocks stay pinned. Eventually Allocate
	// must fail with ErrNoCapacity rather than over-committing memory.
	var held []*Handle
	failed := false
	for i := 0; i < 10; i++ {
		r := mkReq(tok, "sys ", "user message "+string(rune('a'+i)))
		h, err := m.Allocate(r)
		if err == ErrNoCapacity {
			failed = true
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		held = append(held, h)
	}
	if !failed {
		t.Fatal("expected ErrNoCapacity when all blocks are pinned")
	}
	_ = held
}

func TestHitRateOverManyRequests(t *testing.T) {
	tok := tokenizer.NewSimple(32000)
	m := New(DefaultConfig())
	sys := "shared system prompt for all requests. "
	for i := 0; i < 100; i++ {
		r := mkReq(tok, sys, "unique question "+string(rune('a'+i%26)))
		h, err := m.Allocate(r)
		if err != nil {
			t.Fatal(err)
		}
		m.Free(h)
	}
	st := m.Stats()
	// 1 miss (first) + 99 hits => hit rate ~0.99.
	if st.HitRate < 0.9 {
		t.Errorf("hit rate %.2f lower than expected for a shared prefix", st.HitRate)
	}
}
