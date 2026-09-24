// Package metrics defines the observability seam and its Prometheus
// implementation.
//
// The Recorder interface is what the batcher (and other layers) call to report
// what's happening. Defining it as an interface — with a no-op default — means
// the hot-path code never imports Prometheus directly and can be tested without
// a metrics registry. The concrete Prometheus implementation lives in
// prometheus.go and is wired in by main. See docs/08-observability.md for what
// each metric means and why it was chosen.
package metrics

// Recorder is the set of measurements the serving layer emits. Histograms take
// observations; gauges are set to a current value; counters are incremented.
type Recorder interface {
	// ObserveTTFT records time-to-first-token, in seconds, for one request.
	ObserveTTFT(seconds float64)
	// ObserveRequestDuration records total generation time, in seconds.
	ObserveRequestDuration(seconds float64)
	// ObserveTokensPerSecond records a completed request's decode throughput.
	ObserveTokensPerSecond(tps float64)
	// ObserveBatchSize records the number of sequences in one forward pass.
	ObserveBatchSize(n int)

	// IncTokensGenerated adds to the total tokens-generated counter.
	IncTokensGenerated(n int)
	// IncRequestsFinished increments the finished-requests counter, labelled by
	// finish reason ("stop", "length", "canceled").
	IncRequestsFinished(reason string)
	// IncRequestsRejected increments the admission-control rejection counter.
	IncRequestsRejected()
	// IncRequestsReceived increments the total-requests-received counter.
	IncRequestsReceived()

	// SetQueueDepth sets the current scheduler queue-depth gauge.
	SetQueueDepth(n int)
	// SetActiveBatch sets the current active-batch-size gauge.
	SetActiveBatch(n int)
	// SetKVCacheUtilization sets KV-cache utilization as a fraction [0,1].
	SetKVCacheUtilization(fraction float64)
	// SetPrefixHitRate sets the cumulative prefix-cache hit rate [0,1].
	SetPrefixHitRate(rate float64)
}

// Noop is a Recorder that discards everything. Handy for tests and for running
// the serving layer without a metrics backend.
type Noop struct{}

func (Noop) ObserveTTFT(float64)            {}
func (Noop) ObserveRequestDuration(float64) {}
func (Noop) ObserveTokensPerSecond(float64) {}
func (Noop) ObserveBatchSize(int)           {}
func (Noop) IncTokensGenerated(int)         {}
func (Noop) IncRequestsFinished(string)     {}
func (Noop) IncRequestsRejected()           {}
func (Noop) IncRequestsReceived()           {}
func (Noop) SetQueueDepth(int)              {}
func (Noop) SetActiveBatch(int)             {}
func (Noop) SetKVCacheUtilization(float64)  {}
func (Noop) SetPrefixHitRate(float64)       {}

var _ Recorder = Noop{}
