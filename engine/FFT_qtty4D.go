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
	bufsTable_map  sync.Map
	FFT_T_in_mem           = true
	minFrequency   float64 = math.NaN()
	maxFrequency   float64 = math.NaN()
	dFrequency     float64 = math.NaN()
	fft4DLabel     string  = ""
	kx             float64 = math.NaN()
	ky             float64 = math.NaN()
	kz             float64 = math.NaN()
)

func init() {
	DeclFunc("FFT4D", FFT4D, "performs FFT in x, y z and t")
	DeclVar("FFT_T_IN_MEM", &FFT_T_in_mem, "")
	DeclVar("minFrequency", &minFrequency, "")
	DeclVar("maxFrequency", &maxFrequency, "")
	DeclVar("dFrequency", &dFrequency, "")
	DeclVar("FFT4D_Label", &fft4DLabel, "")
	//DeclVar("kx", &kx, "")
	//DeclVar("ky", &ky, "")
	//DeclVar("kz", &kz, "")
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
	ikx      float64
	iky      float64
	ikz      float64
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

func FFT4D(q Quantity, period float64) *fftOperation4D {
	QTTYName := ""
	if fft4DLabel != "" {
		QTTYName = fft4DLabel
	} else {
		QTTYName = NameOf(q)
	}
	ikx := math.NaN()
	iky := math.NaN()
	ikz := math.NaN()
	fft3D := FFT3D_FFT_T(q)
	/*
		s, _, KXYZEnd, _ := fft3D.Axis()
		if !math.IsNaN(kx) {
			ikx = KXYZEnd[0] / kx
			if int(ikx) > s[0] {
				panic("kx is too large for the given mesh. ")
			}
		}
		if !math.IsNaN(ky) {
			iky = (1 + ky/KXYZEnd[1]) * float64(s[1]) / 2
			if int(iky) > s[1] {
				panic("ky is too large for the given mesh. ")
			}
		}
		if !math.IsNaN(kz) {
			ikz = (1 + kz/KXYZEnd[2]) * float64(s[2]) / 2
			if int(ikz) > s[2] {
				panic("kz is too large for the given mesh. ")
			}
		}
		if !kx_ky_kz_prop_defined(q, ikx, iky, ikz) {
			panic("kx, ky, kz are not properly defined.")
		}*/
	FFT_T_OP_Obj := fftOperation4D{fieldOp{q, q, q.NComp()}, "k_x_y_z_f_" + NameOf(q), q, dFrequency, maxFrequency, minFrequency, Time, period, -1, false, fft3D, QTTYName, ikx, iky, ikz}
	FFT_T_OP = append(FFT_T_OP, &FFT_T_OP_Obj)
	FFT_T_OPRunning[&FFT_T_OP_Obj] = false
	FFT_T_OPDataCopied[&FFT_T_OP_Obj] = false
	return &FFT_T_OP_Obj
}

func XOR3(a, b, c bool) bool {
	return (a != b) != c
}

func XOR3_2True(a, b, c bool) bool {
	// Count the number of true inputs
	count := 0
	if a {
		count++
	}
	if b {
		count++
	}
	if c {
		count++
	}

	// Return true if count is 1 or 2
	return count == 2
}

func kx_ky_kz_prop_defined(q Quantity, ikx, iky, ikz float64) bool {
	// Determine if kx, ky, kz are NaN (undefined)
	isNaN_kx := math.IsNaN(ikx)
	isNaN_ky := math.IsNaN(iky)
	isNaN_kz := math.IsNaN(ikz)

	// Count NaNs
	numNaNs := 0
	if isNaN_kx {
		numNaNs++
	}
	if isNaN_ky {
		numNaNs++
	}
	if isNaN_kz {
		numNaNs++
	}
	// Count dimensions equal to 1
	dim1 := SizeOf(q)[0] == 1
	dim2 := SizeOf(q)[1] == 1
	dim3 := SizeOf(q)[2] == 1

	numDims1 := 0
	if dim1 {
		numDims1++
	}
	if dim2 {
		numDims1++
	}
	if dim3 {
		numDims1++
	}

	// Define the two cases
	// Case 1: Exactly 1 NaN (two k values defined) and all dimensions > 1
	case1 := (numNaNs == 1) && (numDims1 == 0)

	// Case 2: Exactly 2 NaNs (one k value defined) and exactly 1 dimension == 1
	case2 := (numNaNs == 2) && (numDims1 == 1)

	// Ensure that the NaNs correspond to the dimensions that are not 1
	// For Case 2, if a dimension is 1, its corresponding k should be NaN
	// Example: If dim1 == 1, then isNaN_kx should be true
	// We need to verify that exactly one of the mappings matches

	// Count correct mappings
	correctMappings := 0
	if (dim1 && isNaN_kx) || (!dim1 && !isNaN_kx) {
		correctMappings++
	}
	if (dim2 && isNaN_ky) || (!dim2 && !isNaN_ky) {
		correctMappings++
	}
	if (dim3 && isNaN_kz) || (!dim3 && !isNaN_kz) {
		correctMappings++
	}

	// For Case 1: All mappings should be correct
	correctCase1 := (numNaNs == 1) && (correctMappings == 3)

	// For Case 2: Exactly one mapping should be mismatched (since one dim is 1 and two k's are NaN)
	correctCase2 := (numNaNs == 2) && (numDims1 == 1) && (correctMappings == 2)

	return (case1 && correctCase1) || (case2 && correctCase2)
}

/*
func (s *fftOperation4D) average() [][]float64 {
	//(a != b) != c
	averegedData := make([][]float32, s.q.NComp())
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
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cuda.SetCurrent_Ctx()
	cuda.Create_Stream(NameOf(s.q) + "_table")
	s.waitUntilPreviousCalcDone()
	FFT_T_data := make([]*data.Slice, 0)
	tmpVar, ok := FFT_T_data_map.Load(s.q)
	if ok {
		FFT_T_data = tmpVar.([]*data.Slice)
	} else {
		panic("FFT_T data could not be found during export for table. FFT needs to be calculated in GPU memory.")
	}
	bufsTable := make([]*data.Slice, 0)
	bufInitalized := slices.Contains(GetKeys[Quantity](&bufsTable_map), s.q)
	if !bufInitalized {
		sizeBufTable := FFT_T_data[0].Size()
		if !math.IsNaN(s.ikx) {
			sizeBufTable[0] = 1
		}
		if !math.IsNaN(s.iky) {
			sizeBufTable[1] = 1
		}
		if !math.IsNaN(s.ikz) {
			sizeBufTable[2] = 1
		}
		cuda.IncreaseBufMax(cores * FFT_T_data[0].NComp())
		for _ = range cores {
			bufsTable = append(bufsTable, cuda.BufferFFT_T(FFT_T_data[0].NComp(), sizeBufTable, NameOf(s.q)+"_table"))
		}
		bufsTable_map.Store(s.q, bufsTable)
	} else {
		tmpVar, ok = bufsTable_map.Load(s.q)
		if ok {
			bufsTable = tmpVar.([]*data.Slice)
		} else {
			panic("Could not load table buffers.")
		}
	}
	wg := sync.WaitGroup{}
	for core := range cores {
		upperEnd := 0
		if lowerEnd+filesPerCore < amountFiles {
			upperEnd = (core + 1) * filesPerCore
		} else {
			upperEnd = amountFiles
		}
		wg.Add(1)
		go func(core int, s *fftOperation4D, startIndex, endIndex int) {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			cuda.SetCurrent_Ctx()
			cuda.Create_Stream(NameOf(s.q) + fmt.Sprintf("_%d_table", core))
			for i := startIndex; i < endIndex; i++ {
				cuda.ExtractSlice(bufsTable[core], FFT_T_data[i], s.ikx, s.iky, s.ikz, NameOf(s.q)+fmt.Sprintf("_%d_table", core))
				reducedData := bufsTable[core].HostCopy().Host()
				for i := range reducedData {
					averegedData[i][startIndex:endIndex] = reducedData[i]
				}
			}
			cuda.Destroy_Stream(fmt.Sprintf(NameOf(s.q)+"_%d", core))
			wg.Done()
		}(core, s, lowerEnd, upperEnd)
		lowerEnd += filesPerCore
	}
	wg.Wait()

}
*/

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
