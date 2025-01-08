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
	FFT_T_data_map     = make(map[Quantity][]*data.Slice)
	FFT_T_in_mem       = true
	minFrequency       float64
	maxFrequency       float64
	dFrequency         float64
)

func init() {
	DeclFunc("FFT4D", FFT4D, "performs FFT in x, y z and t")
	DeclVar("FFT_T_IN_MEM", &FFT_T_in_mem, "")
	DeclVar("minFrequency", &minFrequency, "")
	DeclVar("maxFrequency", &maxFrequency, "")
	DeclVar("dFrequency", &dFrequency, "")
	//DeclFunc("FFT2D", FFT2D, "performs FFT in x, y and z")
}

type fftOperation4D struct {
	fieldOp
	name     string
	q        Quantity
	dF       *float64
	maxF     *float64
	minF     *float64
	start    float64
	period   float64
	count    int
	fDataSet bool
	fft3D    *fftOperation3D
}

func FFT4D(q Quantity, period float64) {
	FFT_T_OP_Obj := fftOperation4D{fieldOp{q, q, q.NComp()}, "k_x_y_z_f_" + NameOf(q), q, &dFrequency, &maxFrequency, &minFrequency, Time, period, -1, false, FFT3D_FFT_T(q)}
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
		dF := s.dF
		minF := s.minF
		maxF := s.maxF
		if dF == nil {
			if running {
				dF = new(float64)
				*dF = 1 / currentRunningTime
			} else if stepping {
				panic("For Step mode FFT dFrequency has to be set")
			} else if runningWhile {
				panic("For RunWhile mode FFT dFrequency has to be set")
			} else {
				panic("An error occured in setting up FFT for time.")
			}
		}

		if minF == nil {
			minF = new(float64)
			*minF = 0
		}
		if maxF == nil {
			maxF = new(float64)
			*maxF = 1 / period
		}

		s.dF = dF
		s.maxF = maxF
		s.minF = minF
		s.fDataSet = true
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cuda.SetCurrent_Ctx()

	FFTOP := s.fft3D
	FFTOP.evalIntern()

	dataT := FFT3DData[s.q]
	cores := runtime.NumCPU()
	amountFiles := int((*s.maxF - *s.minF) / *s.dF)
	if cores > amountFiles {
		if amountFiles < 32 {
			cores = 1
		} else {
			cores = amountFiles / 32
		}
	}
	if int(*s.maxF-*s.minF)%int(*s.dF) != 0 {
		amountFiles += 1
	}
	filesPerCore := int(amountFiles / cores)
	lowerEnd := 0
	if int((*s.maxF-*s.minF) / *s.dF)%cores != 0 {
		cores += 1
	}
	bufsCPU := make([]*data.Slice, 0)
	bufsGPUIP := make([]*data.Slice, 0)
	bufsGPUOP := make([]*data.Slice, 0)
	FFT_T_data := make([]*data.Slice, 0)
	bufInitalized := slices.Contains(slices.Collect(maps.Keys(bufsCPU_map)), s.q)
	if !bufInitalized {
		cuda.IncreaseBufMax(cores * 3)
		for core := range cores {
			bufsCPU = append(bufsCPU, data.NewSlice(dataT.NComp(), dataT.Size()))
			bufsGPUIP = append(bufsGPUIP, cuda.BufferFFT_T(dataT.NComp(), dataT.Size(), fmt.Sprintf(NameOf(s.q)+"_%d", core)))
			bufsGPUOP = append(bufsGPUOP, cuda.BufferFFT_T(dataT.NComp(), dataT.Size(), fmt.Sprintf(NameOf(s.q)+"_%d", core)))
		}
		if FFT_T_in_mem {
			cuda.IncreaseBufMax(dataT.NComp() * amountFiles)
			for range amountFiles {
				FFT_T_data = append(FFT_T_data, cuda.BufferFFT_T(dataT.NComp(), dataT.Size(), NameOf(s.q)))
			}
			FFT_T_data_map[s.q] = FFT_T_data
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
		if FFT_T_in_mem {
			FFT_T_data = FFT_T_data_map[s.q]
		}
	}
	NxNyNz, startK, endK, transformedAxis := FFTOP.Axis()
	wg := sync.WaitGroup{}
	fftT := Time
	for core := range cores {
		upperEnd := 0
		if lowerEnd+filesPerCore < amountFiles {
			upperEnd = (core + 1) * filesPerCore
		} else {
			upperEnd = amountFiles
		}
		wg.Add(1)
		if !FFT_T_in_mem {
			go func(core int, s *fftOperation4D, startIndex, endIndex int, bufCPU, bufGPUIP, bufGPUOP, dataT *data.Slice, FFTOP *fftOperation3D, NxNyNz [3]int, startK, endK [3]float64, transformedAxis []string, fftT float64) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				cuda.SetCurrent_Ctx()
				cuda.Create_Stream(NameOf(s.q) + fmt.Sprintf("_%d", core))
				for i := startIndex; i < endIndex; i++ {
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
					phase := -2 * math.Pi * float64(i) * (*s.minF + float64(i)**s.dF) * fftT
					cuda.FFT_T_Step(bufGPUOP, bufGPUIP, dataT, float32(phase), amountFiles, fmt.Sprintf(NameOf(s.q)+"_%d", core))
					info := data.Meta{Freq: *s.minF + float64(i)**s.dF, Name: "k_x_y_z_f_" + NameOf(s.q), Unit: UnitOf(FFTOP), CellSize: MeshOf(FFTOP).CellSize()}
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+NameOf(s.q), i)+".ovf", bufGPUOP.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false)
				}
				wg.Done()
			}(core, s, lowerEnd, upperEnd, bufsCPU[core], bufsGPUIP[core], bufsGPUOP[core], dataT, FFTOP, NxNyNz, startK, endK, transformedAxis, fftT)
		} else {
			go func(core int, s *fftOperation4D, startIndex, endIndex int, bufGPUIP, dataT *data.Slice, FFTOP *fftOperation3D, NxNyNz [3]int, startK, endK [3]float64, transformedAxis []string, amountFiles int, fftT float64) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				cuda.SetCurrent_Ctx()
				cuda.Create_Stream(NameOf(s.q) + fmt.Sprintf("_%d", core))
				for i := startIndex; i < endIndex; i++ {
					data.Copy(bufGPUIP, FFT_T_data[i], fmt.Sprintf(NameOf(s.q)+"_%d", core))
					//angle := -2i * complex64(complex(math.Pi*float64(i)*Time*s.dF, 0))
					phase := -2 * math.Pi * (*s.minF + float64(i)**s.dF) * fftT
					cuda.FFT_T_Step(FFT_T_data[i], bufGPUIP, dataT, float32(phase), amountFiles, fmt.Sprintf(NameOf(s.q)+"_%d", core))
				}
				cuda.Destroy_Stream(fmt.Sprintf(NameOf(s.q)+"_%d", core))
				wg.Done()
			}(core, s, lowerEnd, upperEnd, bufsGPUIP[core], dataT, FFTOP, NxNyNz, startK, endK, transformedAxis, amountFiles, fftT)
		}
		lowerEnd += filesPerCore
	}
	wg.Wait()
	if FFT_T_in_mem {
		FFT_T_data_map[s.q] = FFT_T_data
	}
	mu.Lock()
	FFT_T_OPRunning[s] = false
	condCalcComplete.Broadcast()
	mu.Unlock()
}

func (s *fftOperation4D) SaveResults() {
	FFTOP := s.fft3D
	NxNyNz, startK, endK, transformedAxis := FFTOP.Axis()
	for i := range len(FFT_T_data_map[s.q]) {
		info := data.Meta{Freq: *s.minF + float64(i)**s.dF, Name: "k_x_y_z_f_" + NameOf(s.q), Unit: UnitOf(FFTOP), CellSize: MeshOf(FFTOP).CellSize()}
		saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+NameOf(s.q), i)+".ovf", FFT_T_data_map[s.q][i].HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false)
	}
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
	for i := range FFT_T_OP {
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

func WaitFFTs4DDone() {
	for i := range FFT_T_OP {
		FFT_T_OP[i].waitUntilPreviousCalcDone()
		if FFT_T_in_mem {
			FFT_T_OP[i].SaveResults()
		}
	}
}
