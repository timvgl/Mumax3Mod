package timer

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"time"
)

var (
	clocks sync.Map
	//clocks     map[string]*clock
	firstStart time.Time
)

func Start(key string) {
	_, ok := clocks.Load(key)
	if !ok {
		clocks.Store(key, new(clock))
		firstStart = time.Now()
	}

	if c, ok := clocks.Load(key); ok {
		c.(*clock).Start()
	} else {
		clocks.Store(key, new(clock))
		// do not start, first run = warmup time
	}
}

func Stop(key string) {
	if c, ok := clocks.Load(key); ok {
		c.(*clock).Stop()
	}
}

type clock struct {
	total       time.Duration
	started     time.Time
	invocations int
}

func (c *clock) Start() {
	c.started = time.Now()
	c.invocations++
}

func (c *clock) Stop() {
	if (c.started == time.Time{}) {
		return // not started
	}
	d := time.Since(c.started)
	c.total += d
	c.started = time.Time{}
}

// entry for sorted output by Print()
type entry struct {
	name        string
	total       time.Duration
	invocations int
	pct         float32
}

func (e *entry) String() string {
	perOp := time.Duration(int64(e.total) / int64(e.invocations))
	return fmt.Sprint(pad(e.name), pad(fmt.Sprint(e.invocations, "x")), perOp, "/op\t", e.pct, " %\t", e.total, " total")
}

func pad(s string) string {
	if len(s) >= 20 {
		return s
	}
	return s + "                    "[:20-len(s)]
}

type entries []entry

func (l entries) Len() int           { return len(l) }
func (l entries) Less(i, j int) bool { return l[i].total > l[j].total }
func (l entries) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }

/*
	func Print(out io.Writer) {
		if clocks == nil {
			return
		}
		wallTime := time.Since(firstStart)
		lines := make(entries, 0, len(clocks))
		var accounted time.Duration
		for k, v := range clocks {
			pct := 100 * float32(int64(v.total)) / float32(int64(wallTime))
			lines = append(lines, entry{k, v.total, v.invocations, pct})
			accounted += v.total
		}

		unaccounted := wallTime - accounted
		pct := 100 * float32(int64(unaccounted)) / float32(int64(wallTime))
		lines = append(lines, entry{"NOT TIMED", unaccounted, 1, pct})

		sort.Sort(lines)

		for _, l := range lines {
			fmt.Fprintln(out, &l)
		}
	}
*/
func Print(out io.Writer) {
	// Calculate the wall time since tracking started
	wallTime := time.Since(firstStart)

	// Initialize a slice to hold the entries
	var lines entries

	// Variable to accumulate the total accounted time
	var accounted time.Duration

	// Iterate over all entries in the sync.Map
	clocks.Range(func(key, value interface{}) bool {
		// Type assert the key to a string
		k, ok := key.(string)
		if !ok {
			// If the key is not a string, skip this entry
			return true
		}

		// Type assert the value to a clock struct
		v, ok := value.(clock)
		if !ok {
			// If the value is not of type clock, skip this entry
			return true
		}

		// Calculate the percentage of wall time accounted for by this clock
		pct := 100 * float32(v.total.Nanoseconds()) / float32(wallTime.Nanoseconds())

		// Append the entry to the lines slice
		lines = append(lines, entry{
			name:        k,
			total:       v.total,
			invocations: v.invocations,
			pct:         pct,
		})

		// Accumulate the accounted time
		accounted += v.total

		return true // Continue iterating
	})

	// Calculate the unaccounted time
	unaccounted := wallTime - accounted
	pct := 100 * float32(unaccounted.Nanoseconds()) / float32(wallTime.Nanoseconds())

	// Append the "NOT TIMED" entry
	lines = append(lines, entry{
		name:        "NOT TIMED",
		total:       unaccounted,
		invocations: 1,
		pct:         pct,
	})

	// Sort the entries in descending order of total time
	sort.Sort(lines)

	// Output each entry to the provided io.Writer
	for _, l := range lines {
		fmt.Fprintln(out, &l)
	}
}
