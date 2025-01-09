package engine

import (
	"fmt"
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
	//bufsCPU_map        = make(map[Quantity][]*data.Slice)
	bufsCPU_map    sync.Map
	bufsGPUIP_map  sync.Map
	bufsGPUOP_map  sync.Map
	FFT_T_data_map sync.Map
	FFT_T_in_mem           = true
	minFrequency   float64 = math.NaN()
	maxFrequency   float64 = math.NaN()
	dFrequency     float64 = math.NaN()
	fft4DLabel     string  = ""
)

func init() {
	DeclFunc("FFT4D", FFT4D, "performs FFT in x, y z and t")
	DeclVar("FFT_T_IN_MEM", &FFT_T_in_mem, "")
	DeclVar("minFrequency", &minFrequency, "")
	DeclVar("maxFrequency", &maxFrequency, "")
	DeclVar("dFrequency", &dFrequency, "")
	DeclVar("FFT4D_Label", &fft4DLabel, "")
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
	fDataSet bool
	fft3D    *fftOperation3D
	label    string
}

func GetKeys[T any](m *sync.Map) []T {
	var keys []T

	m.Range(func(key, value interface{}) bool {
		if k, ok := key.(T); ok {
			keys = append(keys, k)
		}
		return true // continue iteration
	})

	return keys
}

func FFT4D(q Quantity, period float64) {
	QTTYName := ""
	if fft4DLabel != "" {
		QTTYName = fft4DLabel
	} else {
		QTTYName = NameOf(q)
	}
	FFT_T_OP_Obj := fftOperation4D{fieldOp{q, q, q.NComp()}, "k_x_y_z_f_" + NameOf(q), q, dFrequency, maxFrequency, minFrequency, Time, period, -1, false, FFT3D_FFT_T(q), QTTYName}
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
		if math.IsNaN(dF) {
			if running {
				dF = 1 / currentRunningTime
			} else if stepping {
				panic("For Step mode FFT dFrequency has to be set")
			} else if runningWhile {
				panic("For RunWhile mode FFT dFrequency has to be set")
			} else {
				panic("An error occured in setting up FFT for time.")
			}
		}

		if math.IsNaN(minF) {
			minF = 0
		}
		if math.IsNaN(maxF) {
			maxF = 1 / period
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
	amountFiles := int((s.maxF - s.minF) / s.dF)
	if cores > amountFiles {
		if amountFiles < 32 {
			cores = 1
		} else {
			cores = amountFiles / 32
		}
	}
	if int(s.maxF-s.minF)%int(s.dF) != 0 {
		amountFiles += 1
	}
	filesPerCore := int(amountFiles / cores)
	lowerEnd := 0
	if int((s.maxF-s.minF)/s.dF)%cores != 0 {
		cores += 1
	}
	bufsCPU := make([]*data.Slice, 0)
	bufsGPUIP := make([]*data.Slice, 0)
	bufsGPUOP := make([]*data.Slice, 0)
	FFT_T_data := make([]*data.Slice, 0)
	bufInitalized := slices.Contains(GetKeys[Quantity](&bufsCPU_map), s.q)
	if !bufInitalized {
		cuda.IncreaseBufMax(cores * 3 * dataT.NComp())
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
			FFT_T_data_map.Store(s.q, FFT_T_data)
		}
		bufsCPU_map.Store(s.q, bufsCPU)
		bufsGPUIP_map.Store(s.q, bufsGPUIP)
		bufsGPUOP_map.Store(s.q, bufsGPUOP)
	}
	mu.Lock()
	FFT_T_OPDataCopied[s] = true
	condDataCopied.Broadcast()
	mu.Unlock()
	if bufInitalized {
		var ok bool
		var tmpVar any
		tmpVar, ok = bufsCPU_map.Load(s.q)
		if ok {
			bufsCPU = tmpVar.([]*data.Slice)
		} else {
			panic("Could not load CPU buffers.")
		}
		tmpVar, ok = bufsGPUIP_map.Load(s.q)
		if ok {
			bufsGPUIP = tmpVar.([]*data.Slice)
		} else {
			panic("Could not load CPU IP buffers.")
		}
		tmpVar, ok = bufsGPUOP_map.Load(s.q)
		if ok {
			bufsGPUOP = tmpVar.([]*data.Slice)
		} else {
			panic("Could not load GPU OP buffers.")
		}
		if FFT_T_in_mem {
			tmpVar, ok = FFT_T_data_map.Load(s.q)
			if ok {
				FFT_T_data = tmpVar.([]*data.Slice)
			} else {
				panic("Could not load FFT_T buffers.")
			}
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
					in, err := httpfs.Open(OD() + fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+s.label, i) + ".ovf")
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
					phase := -2 * math.Pi * float64(i) * (s.minF + float64(i)*s.dF) * fftT
					cuda.FFT_T_Step(bufGPUOP, bufGPUIP, dataT, float32(phase), amountFiles, fmt.Sprintf(NameOf(s.q)+"_%d", core))
					info := data.Meta{Freq: s.minF + float64(i)*s.dF, Name: "k_x_y_z_f_" + s.label, Unit: UnitOf(FFTOP), CellSize: MeshOf(FFTOP).CellSize()}
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+s.label, i)+".ovf", bufGPUOP.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false)
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
					phase := -2 * math.Pi * (s.minF + float64(i)*s.dF) * fftT
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
		FFT_T_data_map.Store(s.q, FFT_T_data)
	}
	mu.Lock()
	FFT_T_OPRunning[s] = false
	condCalcComplete.Broadcast()
	mu.Unlock()
}

func (s *fftOperation4D) SaveResults() {
	FFTOP := s.fft3D
	NxNyNz, startK, endK, transformedAxis := FFTOP.Axis()
	tmpVar, ok := FFT_T_data_map.Load(s.q)
	FFT_T_data := make([]*data.Slice, 0)
	if ok {
		FFT_T_data = tmpVar.([]*data.Slice)
	} else {
		panic("FFT_T data could not be found during export.")
	}
	for i := range len(FFT_T_data) {
		info := data.Meta{Freq: s.minF + float64(i)*s.dF, Name: "k_x_y_z_f_" + s.label, Unit: UnitOf(FFTOP), CellSize: MeshOf(FFTOP).CellSize()}
		saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+s.label, i)+".ovf", FFT_T_data[i].HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false)
	}
}

func (s *fftOperation4D) Clear_Buffers() {
	FFT_T_data_buffer_length := 0
	if FFT_T_in_mem {
		tmpVar, ok := FFT_T_data_map.Load(s.q)
		if ok {
			FFT_T_data := tmpVar.([]*data.Slice)
			for i := range len(FFT_T_data) {
				cuda.Recycle_FFT_T(FFT_T_data[i], NameOf(s.q))
				FFT_T_data_buffer_length += 1
			}
			FFT_T_data_map.Delete(s.q)
		}
	}
	tmpVar, ok := bufsCPU_map.Load(s.q)
	lengthBuffers := 0
	if ok {
		bufsCPU := tmpVar.([]*data.Slice)
		for i := range len(bufsCPU) {
			bufsCPU[i].Free()
			lengthBuffers += 1
		}
		bufsCPU_map.Delete(s.q)
	}
	tmpVar, ok = bufsGPUIP_map.Load(s.q)
	if ok {
		bufsGPUIP := tmpVar.([]*data.Slice)
		for i := range len(bufsGPUIP) {
			cuda.Recycle_FFT_T(bufsGPUIP[i], fmt.Sprintf(NameOf(s.q)+"_%d", i))
		}
		bufsGPUIP_map.Delete(s.q)
	}
	tmpVar, ok = bufsGPUOP_map.Load(s.q)
	if ok {
		bufsGPUOP := tmpVar.([]*data.Slice)
		for i := range len(bufsGPUOP) {
			cuda.Recycle_FFT_T(bufsGPUOP[i], fmt.Sprintf(NameOf(s.q)+"_%d", i))
		}
		bufsGPUOP_map.Delete(s.q)
	}
	cuda.DecreaseBufMax(s.q.NComp()*3*lengthBuffers + FFT_T_data_buffer_length)
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
		FFT_T_OP[i].Clear_Buffers()
	}
}
