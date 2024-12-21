package engine

import (
	"fmt"
	"maps"
	"math"
	"runtime"
	"slices"
	"sync"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
)

/*
	bufsCPU := make([]*data.Slice, 0)
	bufsGPUIP := make([]*data.Slice, 0)
	bufsGPUOP := make([]*data.Slice, 0)*/

var (
	FFT_T_OP           = []*fftOperation4D{}
	FFT_T_OPRunning    = make(map[*fftOperation4D]bool)
	FFT_T_OPDataCopied = make(map[*fftOperation4D]bool)
	mu                 = new(sync.Mutex)
	condDataCopied     = sync.NewCond(mu)
	condCalcComplete   = sync.NewCond(mu)
	bufsCPU_map        = make(map[Quantity][]*data.Slice)
	bufsGPUIP_map      = make(map[Quantity][]*data.Slice)
	bufsGPUOP_map      = make(map[Quantity][]*data.Slice)
)

func init() {
	DeclFunc("FFT4D", FFT4D, "performs FFT in x, y z and t")
	//DeclFunc("FFT2D", FFT2D, "performs FFT in x, y and z")
}

type fftOperation4D struct {
	fieldOp
	name     string
	q        Quantity
	dF       float64
	maxF     float64
	minF     float64
	start    float64
	period   float64
	count    int
	args     []float64
	fDataSet bool
}

func FFT4D(q Quantity, period float64, args ...float64) {
	if len(args) > 3 {
		panic("FFT4D needs between two and five arguments for FFT.")
	}
	dF := 0.
	maxF := 0.
	minF := 0.
	FFT_T_OP_Obj := fftOperation4D{fieldOp{q, q, q.NComp()}, "k_x_y_z_f_" + NameOf(q), q, dF, maxF, minF, Time, period, -1, args, false}
	FFT_T_OP = append(FFT_T_OP, &FFT_T_OP_Obj)
	FFT_T_OPRunning[&FFT_T_OP_Obj] = false
	FFT_T_OPDataCopied[&FFT_T_OP_Obj] = false
}

func (s *fftOperation4D) Eval() {
	mu.Lock()
	FFT_T_OPDataCopied[s] = false
	condDataCopied.Broadcast()
	FFT_T_OPRunning[s] = true
	condCalcComplete.Broadcast()
	mu.Unlock()
	if !s.fDataSet {
		period := s.period
		args := s.args
		dF := 0.
		minF := 0.
		maxF := 0.
		if len(args) == 0 {
			if running {
				dF = 1 / currentRunningTime
			} else if stepping {
				panic("For Step mode FFT needs at least three arguments.\n In case of three arguments the third one is determining delta F")
			} else if runningWhile {
				panic("For RunWhile mode FFT needs at least three arguments.\n In case of three arguments the third one is determining delta F")
			} else {
				panic("An error occured in setting up FFT for time.")
			}
			maxF = 1 / (2 * period)
			minF = 0.
		} else if len(args) == 1 {
			if running {
				dF = 1 / currentRunningTime
				maxF = args[0]
				minF = 0.
			} else if stepping || runningWhile {
				dF = args[0]
				maxF = 1 / (2 * period)
				minF = 0.
			} else {
				panic("An error occured in setting up FFT for time.")
			}
		} else if len(args) == 2 {
			if running {
				dF = args[1]
				maxF = args[0]
				minF = 0.
			} else if stepping || runningWhile {
				dF = args[1]
				maxF = args[0]
				minF = 0.
			} else {
				panic("An error occured in setting up FFT for time.")
			}
		} else if len(args) == 3 {
			if stepping || runningWhile || running {
				dF = args[2]
				maxF = args[0]
				minF = args[1]
			} else {
				panic("An error occured in setting up FFT for time.")
			}
		}
		s.dF = dF
		s.maxF = maxF
		s.minF = minF
		s.fDataSet = true
	}
	runtime.LockOSThread()
	cuda.SetCurrent_Ctx()

	FFTOP := FFT3D_FFT_T(s.q)
	FFTOP.evalIntern()

	dataT := FFT3DData[s.q]
	cores := runtime.NumCPU()
	filesPerCore := int(int((s.maxF-s.minF)/s.dF) / cores)
	lowerEnd := 0
	if int((s.maxF-s.minF)/s.dF)%cores != 0 {
		cores += 1
	}
	bufsCPU := make([]*data.Slice, 0)
	bufsGPUIP := make([]*data.Slice, 0)
	bufsGPUOP := make([]*data.Slice, 0)
	bufInitalized := slices.Contains(slices.Collect(maps.Keys(bufsCPU_map)), s.q)
	if !bufInitalized {
		for core := range cores {
			bufsCPU = append(bufsCPU, data.NewSlice(dataT.NComp(), dataT.Size()))
			bufsGPUIP = append(bufsGPUIP, cuda.BufferFFT_T(dataT.NComp(), dataT.Size(), fmt.Sprint(NameOf(s.q)+"_%d", core)))
			bufsGPUOP = append(bufsGPUOP, cuda.BufferFFT_T(dataT.NComp(), dataT.Size(), fmt.Sprint(NameOf(s.q)+"_%d", core)))
		}
		bufsCPU_map[s.q] = bufsCPU
		bufsGPUIP_map[s.q] = bufsGPUIP
		bufsGPUOP_map[s.q] = bufsGPUOP
	}
	mu.Lock()
	FFT_T_OPDataCopied[s] = true
	condDataCopied.Broadcast()
	mu.Unlock()
	if bufInitalized {
		bufsCPU = bufsCPU_map[s.q]
		bufsGPUIP = bufsGPUIP_map[s.q]
		bufsGPUOP = bufsGPUOP_map[s.q]
	}
	NxNyNz, startK, endK, transformedAxis := FFTOP.Axis()
	wg := sync.WaitGroup{}
	for core := range cores {
		upperEnd := 0
		if lowerEnd+filesPerCore < int((s.maxF-s.minF)/s.dF) {
			upperEnd = (core + 1) * filesPerCore
		} else {
			upperEnd = int((s.maxF - s.minF) / s.dF)
		}
		wg.Add(1)
		go func(core int, s *fftOperation4D, startIndex, endIndex int, bufCPU, bufGPUIP, bufGPUOP, dataT *data.Slice, FFTOP *fftOperation3D, NxNyNz [3]int, startK, endK [3]float64, transformedAxis []string) {
			runtime.LockOSThread()
			cuda.SetCurrent_Ctx()
			for i := startIndex; i <= endIndex; i++ {
				in, err := httpfs.Open(OD() + fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+NameOf(s.q), i) + ".ovf")
				if err == nil {
					err := oommf.ReadOVF2DataBinary4Optimized(in, bufCPU)
					if err != nil {
						data.Zero(bufCPU)
					}
					in.Close()
				} else {
					data.Zero(bufCPU)
				}
				data.Copy(bufGPUIP, bufCPU)
				angle := -2i * complex64(complex(math.Pi*float64(i)*Time*s.dF, 0))
				cuda.FFT_T_Step(bufGPUOP, bufGPUIP, dataT, real(angle), imag(angle), int((s.maxF-s.minF)/s.dF))
				info := data.Meta{Freq: s.minF + float64(i)*s.dF, Name: "k_x_y_z_f_" + NameOf(s.q), Unit: UnitOf(FFTOP), CellSize: MeshOf(FFTOP).CellSize()}
				saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+NameOf(s.q), i)+".ovf", bufGPUOP.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false)
			}
			wg.Done()
		}(core, s, lowerEnd, upperEnd, bufsCPU[core], bufsGPUIP[core], bufsGPUOP[core], dataT, FFTOP, NxNyNz, startK, endK, transformedAxis)
		lowerEnd += filesPerCore
	}
	wg.Wait()
	/*
		for core := range cores {
			bufsCPU[core].Free()
			cuda.Recycle(bufsGPUIP[core])
			cuda.Recycle(bufsGPUOP[core])
		}
	*/
	//fmt.Println(fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+NameOf(s.q), i) + ".ovf")
	mu.Lock()
	FFT_T_OPRunning[s] = false
	condCalcComplete.Broadcast()
	mu.Unlock()
}

func (s *fftOperation4D) needUpdate() bool {
	t := Time - s.start
	return s.period != 0 && t-float64(s.count)*s.period >= s.period
}

func (s *fftOperation4D) waitUntilCopied() {
	mu.Lock()
	defer mu.Unlock()

	// While val is still the old value, we wait.
	// cond.Wait() will release the mutex and put the goroutine to sleep,
	// allowing other goroutines to change "val" and then wake us up.
	for !FFT_T_OPDataCopied[s] {
		condDataCopied.Wait()
	}
	// Once val != old, we can return the new val.
}

func (s *fftOperation4D) waitUntilPreviousCalcDone() {
	mu.Lock()
	defer mu.Unlock()

	// While val is still the old value, we wait.
	// cond.Wait() will release the mutex and put the goroutine to sleep,
	// allowing other goroutines to change "val" and then wake us up.
	for FFT_T_OPRunning[s] {
		condCalcComplete.Wait()
	}

	// Once val != old, we can return the new val.
}

func DoFFT4D() {
	for i, _ := range FFT_T_OP {
		if FFT_T_OP[i].needUpdate() {
			FFT_T_OP[i].waitUntilPreviousCalcDone()
			mu.Lock()
			FFT_T_OPDataCopied[FFT_T_OP[i]] = false
			condDataCopied.Broadcast()
			mu.Unlock()
			go FFT_T_OP[i].Eval()
			FFT_T_OP[i].waitUntilCopied()
			tmpOp := FFT_T_OP[i]
			tmpOp.count++
			FFT_T_OP[i] = tmpOp
		}
	}
}
