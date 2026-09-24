// Package tokenizer turns text into tokens and back.
//
// WHY THIS EXISTS
//
// Models don't operate on characters or words — they operate on *tokens*,
// integer IDs into a fixed vocabulary. Before the model can process a prompt,
// the prompt must be split into these pieces ("encoding"); after the model
// emits token IDs, they must be turned back into text ("decoding"). The
// serving layer needs the tokenizer for three concrete jobs:
//
//  1. Count tokens, to enforce context-length limits and bill "usage".
//  2. Produce the token sequence the KV-cache manager hashes for prefix caching.
//  3. Decode streamed token IDs back into the text we send to the client.
//
// A REAL tokenizer (BPE like GPT, or SentencePiece/Unigram like Llama) learns a
// merge table from data so that common substrings become single tokens. We do
// NOT need that fidelity to build and study the serving layer, so we ship a
// small, deterministic, dependency-free tokenizer that has the properties that
// actually matter to the rest of the system:
//
//   - Deterministic: the same text always yields the same IDs (so prefix-cache
//     hashing is stable across requests and process restarts).
//   - Reversible: concatenating the token texts reconstructs the input exactly
//     (so streaming decode is trivially correct).
//   - Special-token aware: it models BOS/EOS and reserved IDs like a real one.
//
// See docs/03-tokenization.md for the full discussion and how a real BPE
// tokenizer would slot in behind the Tokenizer interface.
package tokenizer

import (
	"hash/fnv"
	"unicode"

	"github.com/1morshed1/go-llm-server/internal/domain"
)

// Reserved special-token IDs. Real vocabularies reserve a block of low IDs for
// control tokens; we mirror that so IDs produced for ordinary text never
// collide with special tokens.
const (
	PadID = 0 // padding (unused here, reserved for realism)
	BOSID = 1 // beginning-of-sequence
	EOSID = 2 // end-of-sequence
	// firstOrdinaryID is where IDs for real text pieces begin.
	firstOrdinaryID = 256
)

// Tokenizer is the seam the rest of the system depends on. Swapping our simple
// implementation for a real BPE/SentencePiece tokenizer means implementing
// these three methods — nothing else in the codebase changes.
type Tokenizer interface {
	// Encode splits text into tokens. It does NOT add BOS/EOS; callers decide
	// on special tokens explicitly (see EncodePrompt).
	Encode(text string) []domain.Token
	// Decode concatenates the text of a token slice back into a string.
	Decode(tokens []domain.Token) string
	// VocabSize is the number of distinct token IDs the model understands.
	VocabSize() int
	// EOS returns the end-of-sequence token.
	EOS() domain.Token
}

// Simple is a deterministic, reversible tokenizer. It splits text into runs of
// "word" characters (letters/digits), individual punctuation marks, and runs of
// whitespace, then maps each piece's text to a stable ID by hashing it into the
// ordinary-ID range. Because each token keeps its exact source text, decoding
// is just concatenation and always reproduces the input.
type Simple struct {
	vocabSize int
}

// NewSimple returns a Simple tokenizer with the given vocabulary size. 32000 is
// a realistic default (it's Llama's vocab size).
func NewSimple(vocabSize int) *Simple {
	if vocabSize <= firstOrdinaryID {
		vocabSize = 32000
	}
	return &Simple{vocabSize: vocabSize}
}

func (t *Simple) VocabSize() int { return t.vocabSize }

func (t *Simple) EOS() domain.Token { return domain.Token{ID: EOSID, Text: ""} }

// idFor maps a text piece to a stable ID in [firstOrdinaryID, vocabSize).
// FNV-1a is a fast, well-distributed non-cryptographic hash — perfect here
// because we only need determinism and spread, not security.
func (t *Simple) idFor(piece string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(piece))
	span := t.vocabSize - firstOrdinaryID
	return firstOrdinaryID + int(h.Sum32()%uint32(span))
}

// classify groups adjacent runes that belong together. Whitespace runs stay
// together, word characters (letters/digits) stay together, and every other
// rune (punctuation, symbols) becomes its own token. This mirrors how real
// tokenizers keep whitespace attached and split on word boundaries, while
// staying simple enough to read in one sitting.
func (t *Simple) Encode(text string) []domain.Token {
	var tokens []domain.Token
	runes := []rune(text)
	i := 0
	for i < len(runes) {
		r := runes[i]
		switch {
		case unicode.IsSpace(r):
			j := i
			for j < len(runes) && unicode.IsSpace(runes[j]) {
				j++
			}
			piece := string(runes[i:j])
			tokens = append(tokens, domain.Token{ID: t.idFor(piece), Text: piece})
			i = j
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			j := i
			for j < len(runes) && (unicode.IsLetter(runes[j]) || unicode.IsDigit(runes[j])) {
				j++
			}
			piece := string(runes[i:j])
			tokens = append(tokens, domain.Token{ID: t.idFor(piece), Text: piece})
			i = j
		default:
			piece := string(r)
			tokens = append(tokens, domain.Token{ID: t.idFor(piece), Text: piece})
			i++
		}
	}
	return tokens
}

// Decode reconstructs text by concatenating token texts. Special tokens (BOS,
// EOS) carry empty text, so they naturally contribute nothing to the output.
func (t *Simple) Decode(tokens []domain.Token) string {
	var b []byte
	for _, tok := range tokens {
		b = append(b, tok.Text...)
	}
	return string(b)
}

// Compile-time assertion that Simple satisfies the interface.
var _ Tokenizer = (*Simple)(nil)
