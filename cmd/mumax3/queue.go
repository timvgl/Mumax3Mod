package main

// File que for distributing multiple input files over GPUs.

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/engine"
)

var (
	exitStatus       atom = 0
	numOK, numFailed atom = 0, 0
	QueueIndex       int  = 0
)

func get_sim_index() {
	for i := 0; i <= 255; i++ {
		_, ok := os.LookupEnv(fmt.Sprintf("MumaxQueue_%d", i))
		if !ok {
			QueueIndex = i
			os.Setenv(fmt.Sprintf("MumaxQueue_%d", i), "running")
			fmt.Println("env " + fmt.Sprintf("MumaxQueue_%d", i) + " set to running")
			break
		} else if i == 255 {
			panic("Could not find empty queue space.")
		}
	}
}

func RunQueue(files []string) {
	get_sim_index()
	var wg sync.WaitGroup
	wg.Add(len(files))
	s := NewStateTab(files)
	s.PrintTo(os.Stdout)
	go s.ListenAndServe(*engine.Flag_port)
	s.Run(&wg) // Run nimmt WG als Parameter
	wg.Wait()  // hier blockieren, bis alle Done()
	fmt.Println(numOK.get(), "OK, ", numFailed.get(), "failed")
	os.Unsetenv(fmt.Sprintf("MumaxQueue_%d", QueueIndex))
}

// StateTab holds the queue state (list of jobs + statuses).
// All operations are atomic.
type stateTab struct {
	lock sync.Mutex
	jobs []job
	next int
}

// Job info.
type job struct {
	inFile string // input file to run
	uid    int
	port   string
}

// NewStateTab constructs a queue for the given input files.
// After construction, it is accessed atomically.
func NewStateTab(inFiles []string) *stateTab {
	s := new(stateTab)
	s.jobs = make([]job, len(inFiles))
	for i, f := range inFiles {
		s.jobs[i] = job{inFile: f, uid: i}
	}
	return s
}

// StartNext advances the next job and marks it running, setting its webAddr to indicate the GUI url.
// A copy of the job info is returned, the original remains unmodified.
// ok is false if there is no next job.
func (s *stateTab) StartNext(port string) (next job, ok bool) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if s.next >= len(s.jobs) {
		return job{}, false
	}
	s.jobs[s.next].port = port
	jobCopy := s.jobs[s.next]
	s.next++
	return jobCopy, true
}

// Finish marks the job with j's uid as finished.
func (s *stateTab) Finish(j job) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.jobs[j.uid].port = ""
}
func getGPUUsage(gpu int) (int, error) {
	// Der nvidia-smi Befehl liefert die GPU-Auslastung ohne Header und Einheiten.
	cmd := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", strconv.Itoa(gpu))
	output, err := cmd.Output()
	if err != nil {
		return 0, err
	}
	// Entferne etwaige Leerzeichen und Zeilenumbrüche
	s := strings.TrimSpace(string(output))
	usage, err := strconv.Atoi(s)
	if err != nil {
		return 0, err
	}
	return usage, nil
}

func waitForLowGPUUsage(gpu int) {
	for {
		usage, err := getGPUUsage(gpu)
		if err != nil {
			fmt.Printf("And error occured during retrieve of GPU load %d: %v\n", gpu, err)
			// Im Fehlerfall etwas warten, bevor erneut abgefragt wird.
			time.Sleep(time.Second)
			continue
		}

		//fmt.Printf("GPU %d: current volatility: %d%% (Threshold: %d%%)\n", gpu, usage, *flag_GPUTreshold)
		if usage < *flag_GPUTreshold {
			// GPU-Nutzung ist nun niedrig genug
			break
		}
		// Kurze Pause, bevor erneut abgefragt wird.
		time.Sleep(time.Second)
	}
}

// Runs all the jobs in stateTab.
func (s *stateTab) Run(wg *sync.WaitGroup) {
	nGPU := cu.DeviceGetCount()
	idle := initGPUs(nGPU)
	for {
		_, ok := os.LookupEnv(fmt.Sprintf("MumaxQueue_%d", QueueIndex))
		if ok {
			if os.Getenv(fmt.Sprintf("MumaxQueue_%d", QueueIndex)) == "pause" {
				for {
					_, ok := os.LookupEnv(fmt.Sprintf("MumaxQueue_%d", QueueIndex))
					if ok {
						if os.Getenv(fmt.Sprintf("MumaxQueue_%d", QueueIndex)) == "running" {
							break
						}
					} else {
						break
					}
					time.Sleep(time.Second) // Sleep for a short duration before rechecking
				}
			}
		}
		gpu := <-idle
		port := fmt.Sprintf("%v", gpu+35367)
		j, ok := s.StartNext(port)
		if !ok {
			break
		}
		go func() {
			defer wg.Done()
			run(j.inFile, gpu, j.port)
			s.Finish(j)
			waitForLowGPUUsage(gpu)
			idle <- gpu
		}()
	}
	// drain remaining tasks (one already done)
	for i := 1; i < nGPU; i++ {
		<-idle
	}
}

type atom int32

func (a *atom) set(v int) { atomic.StoreInt32((*int32)(a), int32(v)) }
func (a *atom) get() int  { return int(atomic.LoadInt32((*int32)(a))) }
func (a *atom) inc()      { atomic.AddInt32((*int32)(a), 1) }

func run(inFile string, gpu int, port string) {
	// overridden flags
	gpuFlag := fmt.Sprint(`-gpu=`, gpu)
	portFlag := fmt.Sprint("-webUIPort=", port)

	// pass through flags
	flags := []string{portFlag}
	gotGPUExample := false
	flag.Visit(func(f *flag.Flag) {
		if (f.Name != "gpu" || f.Name == "gpu" && *flag_example) && f.Name != "failfast" && f.Name != "webUIPort" && f.Name != "template" && f.Name != "flat" && f.Name != "pipelineLenth" && f.Name != "encapsle" {
			flags = append(flags, fmt.Sprintf("-%v=%v", f.Name, f.Value))
		}
		if f.Name == "gpu" && *flag_example {
			gotGPUExample = true
		}
	})
	if !gotGPUExample {
		flags = append(flags, gpuFlag)
	}
	flags = append(flags, inFile)
	cmd := exec.Command(os.Args[0], flags...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Println(inFile, err)
		log.Printf("%s\n", output)
		exitStatus.set(1)
		numFailed.inc()
		if *flag_failfast {
			os.Exit(1)
		}
	} else {
		numOK.inc()
	}
}

func initGPUs(nGpu int) chan int {
	if nGpu == 0 {
		log.Fatal("no GPUs available")
	}
	maxPerGPU := *flag_maxPerGPU
	totalSlots := nGpu * maxPerGPU
	idle := make(chan int, totalSlots)

	// Für jede GPU maxPerGPU-mal einen Slot vergeben
	for gpu := 0; gpu < nGpu; gpu++ {
		for slot := 0; slot < maxPerGPU; slot++ {
			// Optional: vor dem ersten Befüllen sicherstellen, dass GPU frei ist
			waitForLowGPUUsage(gpu)
			idle <- gpu
		}
	}
	return idle
}

func (s *stateTab) PrintTo(w io.Writer) {
	s.lock.Lock()
	defer s.lock.Unlock()
	for i, j := range s.jobs {
		fmt.Fprintf(w, "%3d %v %v\n", i, j.inFile, j.port)
	}
}

func (s *stateTab) RenderHTML(w io.Writer) {
	s.lock.Lock()
	defer s.lock.Unlock()
	fmt.Fprintln(w, ` 
<!DOCTYPE html> <html> <head> 
	<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
	<meta http-equiv="refresh" content="1">
`+engine.CSS+`
	</head><body>
	<span style="color:gray; font-weight:bold; font-size:1.5em"> mumax<sup>3</sup> queue status </span><br/>
	<hr/>
	<pre>
`)

	hostname := "localhost"
	hostname, _ = os.Hostname()
	for _, j := range s.jobs {
		if j.port != "" {
			fmt.Fprint(w, `<b>`, j.uid, ` <a href="`, "http://", hostname+j.port, `">`, j.inFile, " ", j.port, "</a></b>\n")
		} else {
			fmt.Fprint(w, j.uid, " ", j.inFile, "\n")
		}
	}

	fmt.Fprintln(w, `</pre><hr/></body></html>`)
}

func (s *stateTab) ListenAndServe(addr string) {
	http.Handle("/", s)
	go http.ListenAndServe(addr, nil)
}

func (s *stateTab) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.RenderHTML(w)
}
