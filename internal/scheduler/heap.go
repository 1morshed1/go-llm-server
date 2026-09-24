package scheduler

import "github.com/1morshed1/go-llm-server/internal/domain"

// item is one entry in the priority queue. We keep the arrival `seq` separately
// so ties in Priority break in FCFS order (earlier arrival first), giving a
// stable, fair ordering.
type item struct {
	req *domain.Request
	seq int64
}

// priorityQueue implements container/heap.Interface as a MIN-heap on
// (Priority, seq): the smallest Priority is served first, ties broken by
// earliest arrival. This is the standard way to build a priority queue in Go —
// implement the five methods and let the heap package do the rest.
type priorityQueue []*item

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	if pq[i].req.Priority != pq[j].req.Priority {
		return pq[i].req.Priority < pq[j].req.Priority
	}
	return pq[i].seq < pq[j].seq
}

func (pq priorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *priorityQueue) Push(x any) {
	*pq = append(*pq, x.(*item))
}

func (pq *priorityQueue) Pop() any {
	old := *pq
	n := len(old)
	it := old[n-1]
	old[n-1] = nil // avoid leaking the pointer
	*pq = old[:n-1]
	return it
}
