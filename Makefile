# go-llm-server — common developer tasks.
# Run `make help` for the list.

BINARY := bin/server
PKG    := ./...

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the server binary into bin/
	go build -o $(BINARY) ./cmd/server

.PHONY: run
run: ## Run the server (uses the simulated engine)
	go run ./cmd/server

.PHONY: test
test: ## Run all tests
	go test $(PKG)

.PHONY: race
race: ## Run all tests with the race detector
	go test -race $(PKG)

.PHONY: cover
cover: ## Run tests with coverage and open an HTML report
	go test -coverprofile=coverage.out $(PKG)
	go tool cover -html=coverage.out -o coverage.html
	@echo "wrote coverage.html"

.PHONY: vet
vet: ## Run go vet
	go vet $(PKG)

.PHONY: fmt
fmt: ## Format all Go code
	gofmt -w .

.PHONY: check
check: fmt vet race ## Format, vet, and race-test (run before committing)

.PHONY: bench
bench: ## Run the in-process benchmark suite (tables in docs/10)
	go run ./scripts/bench

.PHONY: loadgen
loadgen: ## Run the HTTP load generator against a running server
	go run ./scripts/loadgen -url http://localhost:8080 -concurrency 32 -requests 2000

.PHONY: proto
proto: ## Regenerate gRPC code from the .proto (needs protoc + plugins)
	protoc --go_out=. --go_opt=module=github.com/1morshed1/go-llm-server \
	       --go-grpc_out=. --go-grpc_opt=module=github.com/1morshed1/go-llm-server \
	       api/proto/inference.proto

.PHONY: docker
docker: ## Build the container image
	docker build -t go-llm-server:local .

.PHONY: stack
stack: ## Bring up the full observability stack (server + Prometheus + Grafana)
	docker compose -f deploy/docker-compose.yml up --build

.PHONY: clean
clean: ## Remove build and coverage artifacts
	rm -rf bin coverage.out coverage.html
