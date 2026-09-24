# 03 — Tokenization

> Code: `internal/tokenizer/tokenizer.go`, `internal/tokenizer/tokenizer_test.go`

## The idea

Models don't see text. They see **tokens** — integer IDs into a fixed
vocabulary. `"caching"` might be one token `8145`, or split into `"cach"` +
`"ing"`, depending on the model's vocabulary. Two directions:

- **Encode**: text → `[]token`, before the model can process it.
- **Decode**: `[]token` → text, to turn the model's output back into words.

Real tokenizers (GPT's **BPE**, Llama's **SentencePiece**) *learn* a merge
table from a large corpus so that frequent substrings collapse into single
tokens. That's what makes "the" one token instead of three characters, and it's
why token counts don't equal word counts or character counts.

## Why the serving layer needs it

The server itself needs the tokenizer for three jobs — none of which require a
*good* tokenizer, only a *consistent* one:

1. **Counting** — to enforce the model's context window (prompt + output ≤
   context length) and to report `usage` (prompt/completion/total tokens) in the
   API response, just like OpenAI.
2. **Prefix-cache hashing** — the KV-cache manager hashes the *token IDs* of a
   shared prefix to detect reuse. This only works if identical text always
   produces identical IDs. (doc 04)
3. **Streaming decode** — the engine emits token IDs; we must turn each into the
   exact text to stream to the client.

## Our implementation: `Simple`

We ship a deterministic, reversible, dependency-free tokenizer. It:

- splits text into runs of word characters, runs of whitespace, and individual
  punctuation marks;
- maps each piece's text to a stable ID via **FNV-1a hashing** into the
  ordinary-ID range `[256, vocabSize)`;
- reserves low IDs `0/1/2` for `PAD`/`BOS`/`EOS`, mirroring real vocabularies;
- keeps each token's exact source text on the `Token`, so `Decode` is just
  concatenation and **always reproduces the input** (the round-trip test).

This gives us the three properties that matter (determinism, reversibility,
special-token awareness) in ~100 readable lines.

### What it is *not*

It does not learn merges, so its token counts won't match a real model's, and
its IDs are hashed rather than looked up in a trained vocabulary (so two
different strings could in principle collide on an ID — harmless here because we
carry the text alongside the ID). We are explicit about this because *knowing
the limitation is the point*: the serving layer we're studying doesn't depend on
tokenizer quality, only on its contract.

## How a real tokenizer slots in

Implement the `Tokenizer` interface with, say, a Go BPE library or CGo bindings
to `tokenizers`/`sentencepiece`:

```go
type Tokenizer interface {
    Encode(text string) []domain.Token
    Decode(tokens []domain.Token) string
    VocabSize() int
    EOS() domain.Token
}
```

`NewSimple(...)` becomes `NewBPE(vocabFile)`. Nothing else changes — the KV
cache hashes whatever IDs it's given, and streaming decodes whatever tokens it
receives.

## Decisions & trade-offs

- **Carry text on every token.** Real pipelines decode IDs lazily through the
  vocabulary. We store the text on the `Token` so decode is trivially correct
  and streaming never has to consult the tokenizer mid-generation. Costs a
  little memory; buys simplicity and correctness.
- **Hash instead of a learned vocab.** Determinism and spread are all the
  serving layer needs; a learned vocab would add a dependency and a model file
  for zero benefit to what we're studying.
- **Whitespace as its own tokens.** This is what makes the round trip exact.
  Real BPE attaches leading spaces to words (the famous `"▁the"`); we keep them
  separate for clarity. Documented so the difference isn't a surprise.
