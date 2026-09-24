package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Prometheus is a Recorder backed by Prometheus metrics. It registers all
// metrics into its OWN registry (not the global default) so multiple instances
// can coexist — notably in tests — without "duplicate metrics" panics.
//
// Metric naming follows Prometheus conventions: a common `llm_` prefix, base
// units (seconds, not milliseconds), `_total` suffix on counters, and labels
// used sparingly (only `reason`, which has a tiny fixed cardinality). See
// docs/08-observability.md for what each one tells you and why it's here.
type Prometheus struct {
	reg *prometheus.Registry

	requestsReceived prometheus.Counter
	requestsRejected prometheus.Counter
	requestsFinished *prometheus.CounterVec
	tokensGenerated  prometheus.Counter

	ttft       prometheus.Histogram
	duration   prometheus.Histogram
	tokensPerS prometheus.Histogram
	batchSize  prometheus.Histogram

	queueDepth    prometheus.Gauge
	activeBatch   prometheus.Gauge
	kvUtil        prometheus.Gauge
	prefixHitRate prometheus.Gauge
}

// NewPrometheus builds and registers all metrics.
func NewPrometheus() *Prometheus {
	reg := prometheus.NewRegistry()
	factory := prometheus.WrapRegistererWithPrefix("llm_", reg)

	p := &Prometheus{reg: reg}

	p.requestsReceived = newCounter(factory, "requests_received_total", "Total chat-completion requests received.")
	p.requestsRejected = newCounter(factory, "requests_rejected_total", "Requests rejected by admission control (queue full).")
	p.tokensGenerated = newCounter(factory, "tokens_generated_total", "Total output tokens generated across all requests.")
	p.requestsFinished = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "requests_finished_total",
		Help: "Requests that finished, labelled by reason (stop|length|canceled).",
	}, []string{"reason"})
	factory.MustRegister(p.requestsFinished)

	p.ttft = newHistogram(factory, "ttft_seconds", "Time to first token, in seconds.",
		[]float64{0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 2, 5})
	p.duration = newHistogram(factory, "request_duration_seconds", "Total generation time per request, in seconds.",
		[]float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30})
	p.tokensPerS = newHistogram(factory, "tokens_per_second", "Per-request decode throughput (tokens/second).",
		[]float64{10, 25, 50, 100, 150, 200, 300, 400, 600, 800})
	p.batchSize = newHistogram(factory, "batch_size", "Number of sequences per forward pass.",
		prometheus.LinearBuckets(1, 1, 16))

	p.queueDepth = newGauge(factory, "queue_depth", "Current scheduler queue depth.")
	p.activeBatch = newGauge(factory, "active_batch_size", "Current number of sequences decoding.")
	p.kvUtil = newGauge(factory, "kv_cache_utilization_ratio", "KV-cache utilization as a fraction [0,1].")
	p.prefixHitRate = newGauge(factory, "prefix_cache_hit_rate", "Cumulative prefix-cache hit rate [0,1].")

	return p
}

// Handler returns the HTTP handler that exposes these metrics at /metrics.
func (p *Prometheus) Handler() http.Handler {
	return promhttp.HandlerFor(p.reg, promhttp.HandlerOpts{})
}

// --- Recorder implementation ---

func (p *Prometheus) ObserveTTFT(s float64)            { p.ttft.Observe(s) }
func (p *Prometheus) ObserveRequestDuration(s float64) { p.duration.Observe(s) }
func (p *Prometheus) ObserveTokensPerSecond(t float64) { p.tokensPerS.Observe(t) }
func (p *Prometheus) ObserveBatchSize(n int)           { p.batchSize.Observe(float64(n)) }
func (p *Prometheus) IncTokensGenerated(n int)         { p.tokensGenerated.Add(float64(n)) }
func (p *Prometheus) IncRequestsFinished(reason string) {
	p.requestsFinished.WithLabelValues(reason).Inc()
}
func (p *Prometheus) IncRequestsRejected()            { p.requestsRejected.Inc() }
func (p *Prometheus) IncRequestsReceived()            { p.requestsReceived.Inc() }
func (p *Prometheus) SetQueueDepth(n int)             { p.queueDepth.Set(float64(n)) }
func (p *Prometheus) SetActiveBatch(n int)            { p.activeBatch.Set(float64(n)) }
func (p *Prometheus) SetKVCacheUtilization(f float64) { p.kvUtil.Set(f) }
func (p *Prometheus) SetPrefixHitRate(r float64)      { p.prefixHitRate.Set(r) }

// --- small constructors that register as they build ---

func newCounter(f prometheus.Registerer, name, help string) prometheus.Counter {
	c := prometheus.NewCounter(prometheus.CounterOpts{Name: name, Help: help})
	f.MustRegister(c)
	return c
}

func newGauge(f prometheus.Registerer, name, help string) prometheus.Gauge {
	g := prometheus.NewGauge(prometheus.GaugeOpts{Name: name, Help: help})
	f.MustRegister(g)
	return g
}

func newHistogram(f prometheus.Registerer, name, help string, buckets []float64) prometheus.Histogram {
	h := prometheus.NewHistogram(prometheus.HistogramOpts{Name: name, Help: help, Buckets: buckets})
	f.MustRegister(h)
	return h
}

var _ Recorder = (*Prometheus)(nil)
