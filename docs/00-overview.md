# 00 — Overview & how to read this repo

Welcome. This document orients you before you touch any code.

## What we are building

An **LLM inference server**: the program that turns a machine-learning model
into a network service. When you call the OpenAI API, or run `ollama run`, or
deploy vLLM, a server like this is what accepts your HTTP request, decides when
to run it, feeds it through the model one token at a time, and streams the
result back.

We are building the **serving layer** — not the model math. The reason is in
[doc 02](02-the-engine-interface.md), but the one-line version is: *the model
math (attention, matmuls) is done in C++/CUDA and is largely a solved,
downloadable component; the serving layer is where the systems engineering
lives, and it's what distinguishes a toy from production infrastructure.*

## The mental model: a request's journey

The single most useful thing to hold in your head is the life of one request.
Everything in the codebase exists to move a request through these stages:

```
   HTTP request in
        │
        ▼
 ┌──────────────┐   1. Parse JSON, validate params, render the chat messages
 │  API layer   │      into a single prompt string, tokenize it, build a
 │  (server/)   │      domain.Request with its own Stream channel.
 └──────┬───────┘
        │  submit(req)
        ▼
 ┌──────────────┐   2. Admission control: is the queue too full? reject early.
 │  Scheduler   │      Otherwise push onto a PRIORITY QUEUE (short prompts and
 │ (scheduler/) │      earlier arrivals go first). Enforce a max wait time.
 └──────┬───────┘
        │  pull next
        ▼
 ┌──────────────┐   3. Ask the KV-cache manager for space. Maybe reuse a
 │   Batcher    │      cached prefix (shared system prompt). Add the request
 │  (batcher/)  │      to the ACTIVE BATCH. Loop: run one forward pass across
 │              │      ALL active requests, append one token to each, stream
 │              │      it out. Finished requests LEAVE the batch immediately;
 │              │      new ones JOIN in their place. (Continuous batching.)
 └──────┬───────┘
        │  push StreamEvent per token
        ▼
 ┌──────────────┐   4. The API handler goroutine ranges over req.Stream and
 │  API layer   │      writes each token to the client (as JSON or as an SSE
 │  (server/)   │      `data:` line). On Done, it closes the response.
 └──────────────┘
        │
        ▼
   tokens streamed out
```

Two design ideas make this concurrent and clean:

1. **The request carries its own output channel** (`domain.Request.Stream`).
   The batcher never calls back into the HTTP layer; it just pushes tokens onto
   a channel. The handler, in its own goroutine, reads from that channel. This
   is Go's "share memory by communicating" in action.

2. **Each stage speaks only in `domain` types.** The scheduler doesn't know
   about HTTP. The batcher doesn't know about JSON. The engine doesn't know a
   scheduler exists. Each layer is independently understandable and testable.

## How to read the docs

Read them in numeric order. They were written in the order the code was built,
so each one only assumes what came before. Each doc has three recurring
sections:

- **The idea** — the concept, explained from first principles.
- **The code** — how we implemented it and why we made the choices we did.
- **Decisions & trade-offs** — what we deliberately did *not* do, and why.
  This is the part worth re-reading; naming the road not taken is how you learn
  to make these calls yourself.

## How to read the code

Start at `internal/domain/domain.go`. It's the vocabulary. Then follow a
request's journey in the order above: `server/` → `scheduler/` → `batcher/` →
`engine/`. The `cmd/server/main.go` file at the very end shows how all the
pieces are wired together — read it once you know the pieces.

Every non-obvious line in the code has a comment explaining *why*, not just
*what*. The comments and these docs are meant to be read together.
