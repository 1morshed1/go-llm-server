# syntax=docker/dockerfile:1

# --- build stage ---------------------------------------------------------------
# Compile a fully static binary. CGO is off, so the mock engine has no native
# dependencies and the result runs on a scratch/distroless base. (A real
# llama.cpp backend would flip CGO_ENABLED=1 and need a C toolchain here — one
# more reason the pluggable engine keeps the default build this simple.)
FROM golang:1.25 AS build
WORKDIR /src

# Cache module downloads separately from source for faster rebuilds.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/server ./cmd/server

# --- runtime stage -------------------------------------------------------------
# distroless/static: no shell, no package manager, tiny attack surface. It ships
# with a non-root "nonroot" user and CA certs.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/server /server
EXPOSE 8080
USER nonroot:nonroot
ENTRYPOINT ["/server"]
CMD ["--addr", ":8080"]
