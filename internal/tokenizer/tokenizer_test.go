package tokenizer

import "testing"

// TestRoundTrip is the most important property: whatever we encode must decode
// back to exactly the original text. Streaming correctness depends on it.
func TestRoundTrip(t *testing.T) {
	tok := NewSimple(32000)
	cases := []string{
		"Hello, world!",
		"The quick brown fox.",
		"multiple   spaces\tand\nnewlines",
		"emoji 🚀 and числа 123",
		"",
	}
	for _, in := range cases {
		got := tok.Decode(tok.Encode(in))
		if got != in {
			t.Errorf("round-trip mismatch:\n in=%q\nout=%q", in, got)
		}
	}
}

// TestDeterministic ensures the same text always yields the same IDs. Prefix
// caching relies on this: identical system prompts must hash identically.
func TestDeterministic(t *testing.T) {
	tok := NewSimple(32000)
	a := tok.Encode("a shared system prompt")
	b := tok.Encode("a shared system prompt")
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID {
			t.Errorf("id mismatch at %d: %d vs %d", i, a[i].ID, b[i].ID)
		}
	}
}

// TestIDRange ensures ordinary tokens never collide with reserved special IDs.
func TestIDRange(t *testing.T) {
	tok := NewSimple(32000)
	for _, tk := range tok.Encode("the quick brown fox jumps over the lazy dog") {
		if tk.ID < firstOrdinaryID || tk.ID >= tok.VocabSize() {
			t.Errorf("id %d out of ordinary range [%d,%d)", tk.ID, firstOrdinaryID, tok.VocabSize())
		}
	}
}
