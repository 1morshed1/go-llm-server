# 07 — The HTTP API & streaming

> Code: `internal/server/*.go`, `internal/openai/types.go`,
> `internal/config/config.go`, `cmd/server/main.go`

This layer turns HTTP into `domain.Request`s and streams results back. It's the
only package that knows about HTTP and JSON.

## OpenAI compatibility is the point

We implement the OpenAI Chat Completions schema exactly. Why copy someone else's
API? Because **"OpenAI-compatible" is the de-facto standard** — vLLM, Ollama,
TGI, LM Studio, LiteLLM and countless tools all speak it. Matching it means any
existing OpenAI client library works against our server with only a base-URL
change. Compatibility is a feature, not a lack of imagination.

The wire types live in `internal/openai`. They carry JSON tags and mirror
OpenAI's shapes (including the polymorphic `stop`, which can be a string *or* an
array — handled with a custom `UnmarshalJSON`, the idiomatic Go way to accept a
field of several shapes). Per decision D2, these transport types never leak
below the server: `buildDomainRequest` maps them into a `domain.Request`.

## From request to domain object (`buildDomainRequest`)

Four jobs happen at the boundary, so nothing downstream has to worry about them:

1. **Chat templating** (`renderPrompt`): flatten `messages` into one prompt
   string with role tags (`<|system|>… <|user|>… <|assistant|>`). The leading
   system portion is returned separately as the **cacheable prefix**, and its
   token length becomes `req.PrefixLen` — this is what wires the API to
   [prefix caching](04-kv-cache.md).
2. **Tokenization**: encode the prompt so the KV manager and engine have tokens.
3. **Validation**: reject empty `messages` (400) and prompts longer than the
   model's context window (400).
4. **Clamping**: apply defaults and clamp `temperature` to [0,2], `top_p` to
   [0,1], and `max_tokens` to fit the remaining context. **Clamp at the boundary
   so the engine can trust its inputs** and never has to defend against absurd
   values — a good rule for any system's edge.

## Two response paths

### Non-streaming (`bufferedResponse`)

Drain the whole `req.Stream`, accumulate the text, and send one JSON body with
`choices` and a `usage` block (prompt/completion/total tokens) — byte-for-byte
the OpenAI shape. If the client disconnected mid-generation, the context is done
and we simply stop.

### Streaming with Server-Sent Events (`streamResponse`)

Streaming is what makes a chat UI feel alive — tokens appear as they're
generated. We use **SSE**, the same protocol OpenAI uses: keep the response open
and write `data: <json>\n\n` frames, ending with `data: [DONE]`.

```
data: {"...":"delta with role: assistant"}

data: {"...":"delta with content: 'The'"}

data: {"...":"delta with content: ' cache'"}

data: {"...":"finish_reason: 'stop'"}

data: [DONE]
```

Three details that matter:

- **Flush after every frame.** Without `flusher.Flush()`, Go buffers the response
  and the client sees nothing until the buffer fills or the handler returns —
  which defeats streaming entirely. This is the single most common streaming bug.
- **Initial role chunk.** OpenAI sends a first chunk with just
  `delta: {role: "assistant"}`; clients rely on it to open the message.
- **`X-Accel-Buffering: no`.** Tells reverse proxies like nginx not to buffer the
  stream, so tokens aren't held back at the proxy.

If a write fails, the client is gone: we stop writing, and the batcher — which is
watching `req.Ctx` — cancels the generation and frees its KV. Client disconnects
cost nothing.

## Backpressure surfaces here

`Submit` returning `scheduler.ErrQueueFull` becomes **HTTP 429 Too Many
Requests**, the standard "slow down" signal. main injects the sentinel via
`SetQueueFullError` so the server can recognise it with `errors.Is` without
importing the scheduler just for one error value.

## Config and wiring (`config`, `main`)

`config.Load` gathers every knob in one place: component defaults, overridden by
environment variables, overridden by flags. `cmd/server/main.go` is the
assembly — build bottom-up (tokenizer → engine → kv → scheduler → batcher →
server), start the batcher goroutine, serve HTTP, and shut down **in order**:

1. stop accepting HTTP (drain in-flight handlers),
2. close the scheduler (fail queued requests),
3. cancel the batcher (fail active requests, return).

Getting shutdown order right is what makes `Ctrl-C` clean instead of a source of
panics and leaked goroutines.

## Decisions & trade-offs

- **Depend on a narrow `Submitter` interface, not `*scheduler.Scheduler`.** The
  server only needs `Submit` and `Len`, so that's all it asks for — easier to
  test and a clear contract.
- **Buffer each request's `Stream` (cap ≈ max_tokens).** A slow client can't
  stall the batcher loop; combined with the batcher's `select` on `ctx`, delivery
  is fully decoupled from generation.
- **`canceled` maps to `"stop"` on the wire.** OpenAI has no "canceled" finish
  reason, so we map to the nearest compatible value and keep the true reason in
  metrics. Wire compatibility beats inventing fields.
- **Gin, but only at the edge.** Handlers are thin; all logic is in
  `buildDomainRequest`/`streamResponse`. Swapping Gin for `net/http` or chi would
  touch only this package.
