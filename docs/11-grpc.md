# 11 — The gRPC streaming API

> Code: `api/proto/inference.proto`, `internal/grpcapi/server.go`,
> `internal/grpcapi/pb/*` (generated), `internal/prompt/prompt.go`

The server has two front doors: the OpenAI-compatible REST API (docs 07) and a
gRPC streaming API. Both submit the **same `domain.Request` to the same
scheduler**. This chapter is as much about *what adding a second protocol taught
us about the architecture* as it is about gRPC itself.

## Why gRPC for inference?

Token streaming is a **server-streaming** problem: one request in, many
responses out. gRPC models that natively:

- **Typed contract.** The `.proto` file *is* the API. Clients in any language
  generate a strongly-typed stub from it — no hand-written JSON parsing, no
  guessing the shape of an SSE `data:` line.
- **HTTP/2 streams.** Multiplexed, binary-framed, with flow control
  (backpressure) built into the transport. We call `stream.Send(token)` in a
  loop; the framework handles framing and windowing.
- **Cancellation is first-class.** `stream.Context()` is cancelled the instant
  the client goes away — which we pass straight through as the request's
  `Ctx`, so the batcher frees resources immediately, exactly like the REST path.

REST/SSE is the right choice for browser and OpenAI-ecosystem clients; gRPC is
the right choice for internal service-to-service calls where you want a typed
contract and efficient streaming. Real servers (TGI, Triton) offer both for the
same reasons.

## The contract

`api/proto/inference.proto` defines one service:

```proto
service InferenceService {
  rpc Generate(GenerateRequest) returns (stream GenerateResponse); // server-streaming
  rpc Info(InfoRequest) returns (InfoResponse);                    // unary probe
}
```

`GenerateResponse` is either a token (`token` set) or the terminal message
(`done = true`, carrying `finish_reason` and usage counts) — the same
token-then-terminal shape as our `domain.StreamEvent`, which is no accident.

Generate the Go code (checked in, so a plain `go build` needs no toolchain):

```bash
protoc --go_out=. --go_opt=module=github.com/1morshed1/go-llm-server \
       --go-grpc_out=. --go-grpc_opt=module=github.com/1morshed1/go-llm-server \
       api/proto/inference.proto
```

## The implementation is thin — and that's the lesson

`internal/grpcapi/server.go` is short because it does **no core work**. Its
`Generate`:

1. maps `pb.GenerateRequest` → a neutral `prompt.GenSpec`,
2. calls `prompt.Build` (the *same* templating/validation/clamping the REST path
   uses),
3. `sub.Submit(req)` to the *same* scheduler,
4. ranges over `req.Stream`, calling `stream.Send` per token, and sends a
   terminal message on `Done`.

Errors map to gRPC status codes the way REST maps to HTTP: empty messages →
`InvalidArgument` (like HTTP 400), queue full → `ResourceExhausted` (like HTTP
429), engine error → `Internal` (like HTTP 500).

### What this proves about the design

Adding an entire second network protocol required **zero changes** to the
scheduler, batcher, KV cache, or engine. That is the concrete payoff of decision
D2 (keep transport out of the core) and D3 (requests carry their own stream): the
core speaks `domain` types and channels, so a new transport is just a new mapping
at the edge. When you hear "decouple transport from domain," *this* is what it
buys you.

## Refactor: the shared `prompt` package

Before gRPC, prompt rendering + validation lived inside the REST handler. Adding
gRPC would have meant copying it — and copies drift, so the two APIs would
slowly disagree about, say, how `max_tokens` is clamped. Instead we extracted
`internal/prompt`:

- `Render(messages)` — chat templating + cacheable-prefix marking,
- `Build(...)` — the whole "wire-neutral spec → validated `domain.Request`"
  pipeline, returning a `BuildError` with a stable `Code`.

Both transports now map their wire types to `prompt.GenSpec` and map
`BuildError` to their own status vocabulary. **One behaviour, two front doors** —
the second protocol is what *forced* the good factoring, which is a common and
healthy pattern: duplication becomes a shared package the moment there's a second
caller.

## Trying it

The server listens on `:50051` by default (set `--grpc-addr ""` to disable). Any
gRPC client generated from the proto can call it; the integration test in
`internal/grpcapi/server_test.go` shows a full round trip — spin up the server,
open a `Generate` stream, and read tokens until the `done` message.

## Decisions & trade-offs

- **Server-streaming, not bidi.** Generation is one-request/many-responses, so
  server-streaming fits exactly; bidirectional streaming would add complexity
  (mid-stream client input) we don't need.
- **Check in the generated code.** So `go build` and CI need no `protoc`. The
  regenerate command is documented above and in the Makefile.
- **Separate port from HTTP.** gRPC and REST on different ports keeps routing
  trivial; a single-port multiplexer (cmux/h2c) is possible but adds machinery
  for no lesson here.
- **No TLS in the demo.** `insecure` credentials for local use; production would
  terminate TLS at the gRPC server or a mesh sidecar.
