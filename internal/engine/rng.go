package engine

// rng is a tiny deterministic pseudo-random number generator (SplitMix64).
//
// Why not use math/rand? We want each SEQUENCE to have its own independent,
// fully reproducible stream seeded from its prompt, with no shared global
// state and no locking. SplitMix64 is a well-known, high-quality generator
// that is a few lines long and perfect for this: same seed => same sequence,
// every time, on every machine. That determinism is what makes the mock
// engine's output stable for tests and demos.
type rng struct{ state uint64 }

func newRNG(seed uint64) *rng { return &rng{state: seed} }

// next returns the next 64-bit value and advances the state.
func (r *rng) next() uint64 {
	r.state += 0x9E3779B97F4A7C15
	z := r.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// intn returns a value in [0, n). n must be > 0.
func (r *rng) intn(n int) int { return int(r.next() % uint64(n)) }
