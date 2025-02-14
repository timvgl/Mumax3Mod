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
	cropKSpace             = func(q Quantity) Quantity { return q }
)

func init() {
	DeclFunc("FFT4D", FFT4D, "performs FFT in x, y z and t")
	DeclFunc("FFT_T", FFT_T, "performs FFT in t")
	DeclVar("FFT_T_IN_MEM", &FFT_T_in_mem, "")
	DeclVar("minFrequency", &minFrequency, "")
	DeclVar("maxFrequency", &maxFrequency, "")
	DeclVar("dFrequency", &dFrequency, "")
	DeclVar("FFT4D_Label", &fft4DLabel, "")
	DeclFunc("Tidy_up_FFT_4D", WaitFFTs4DDone, "Waits until all FFTs are done and clears buffers")
	DeclVar("cropKSpace", &cropKSpace, "")
}

type fftOperation4D struct {
	fieldOp
	name       string
	q          Quantity
	dF         float64
	maxF       float64
	minF       float64
	start      float64
	period     float64
	count      int
	fDataSet   bool
	qOP        Quantity
	label      string
	kspace     bool
	dataT      *data.Slice
	polar      bool
	phi        bool
	abs        bool
	cropKSpace func(q Quantity) Quantity
}

func UnpackUntilFFT3D(q Quantity) *fftOperation3D {
	if s, ok := q.(*fftOperation3D); ok {
		return s
	} else if s, ok := q.(*cropped); ok {
		return UnpackUntilFFT3D(s.parent)
	} else if s, ok := q.(*expanded); ok {
		return UnpackUntilFFT3D(s.parent)
	} else if s, ok := q.(*component); ok {
		return UnpackUntilFFT3D(s.parent)
	} else if s, ok := q.(ScalarField); ok {
		return UnpackUntilFFT3D(s.Quantity)
	} else {
		panic("UnpackUntilFFT3D: unknown type")
	}
}

func operatorAxis(size [3]int, startK, endK [3]float64, transformedAxis []string, operator func(q Quantity) Quantity, q Quantity) ([3]int, [3]float64, [3]float64, []string) {
	if s, ok := q.(*fftOperation3D); ok {
		return s.Axis()
	} else if operator(q) == q {
		return size, startK, endK, transformedAxis
	} else {
		size := SizeOf(q)
		newSize := [3]int{0, 0, 0}
		newStartK := [3]float64{math.NaN(), math.NaN(), math.NaN()}
		newEndK := [3]float64{math.NaN(), math.NaN(), math.NaN()}
		if s, ok := q.(*cropped); ok {
			newSize = s.Mesh().Size()
			newStartK[0] = startK[0] + float64(s.x1)*float64(size[0])*MeshOf(q).CellSize()[0]
			newStartK[1] = startK[1] + float64(s.y1)*float64(size[1])*MeshOf(q).CellSize()[1]
			newStartK[2] = startK[2] + float64(s.z1)*float64(size[2])*MeshOf(q).CellSize()[2]
			newEndK[0] = endK[0] - float64(size[0]-s.x2)*float64(size[0])*MeshOf(q).CellSize()[0]
			newEndK[1] = endK[1] - float64(size[1]-s.y2)*float64(size[1])*MeshOf(q).CellSize()[1]
			newEndK[2] = endK[2] - float64(size[2]-s.z2)*float64(size[2])*MeshOf(q).CellSize()[2]

		} else if s, ok := q.(*expanded); ok {
			return UnpackUntilFFT3D(s.parent)
		} else if s, ok := q.(*component); ok {
			return UnpackUntilFFT3D(s.parent)
		} else if s, ok := q.(ScalarField); ok {
			return UnpackUntilFFT3D(s.Quantity)
		} else {
			panic("UnpackUntilFFT3D: unknown type")
		}

		return operatorAxis(newSize, newStartK, newEndK, transformedAxis, operator, q)
	}
}

func (s *fftOperation4D) ToPolar() *fftOperation4D {
	s.polar = true
	return s
}

func (s *fftOperation4D) SavePhi() *fftOperation4D {
	s.polar = true
	s.phi = true
	return s
}
func (s *fftOperation4D) SaveAbs() *fftOperation4D {
	s.polar = true
	s.abs = true
	return s
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
	fft3D := cropKSpace(FFT3D_FFT_T(q))

	var dataT *data.Slice
	FFT_T_OP_Obj := fftOperation4D{fieldOp{q, q, q.NComp()}, "k_x_y_z_f_" + NameOf(q), q, dFrequency, maxFrequency, minFrequency, Time, period, -1, false, fft3D, QTTYName, true, dataT, false, false, false, cropKSpace}
	FFT_T_OP = append(FFT_T_OP, &FFT_T_OP_Obj)
	FFT_T_OPRunning[&FFT_T_OP_Obj] = false
	FFT_T_OPDataCopied[&FFT_T_OP_Obj] = false
	return &FFT_T_OP_Obj
}

func FFT_T(q Quantity, period float64) *fftOperation4D {
	QTTYName := ""
	if fft4DLabel != "" {
		QTTYName = fft4DLabel
	} else {
		QTTYName = NameOf(q)
	}
	fft3D := FFT3D_FFT_T(q)
	var dataT *data.Slice
	FFT_T_OP_Obj := fftOperation4D{fieldOp{q, q, q.NComp()}, "f_" + NameOf(q), q, dFrequency, maxFrequency, minFrequency, Time, period, -1, false, fft3D, QTTYName, false, dataT, false, false, false, cropKSpace}
	FFT_T_OP = append(FFT_T_OP, &FFT_T_OP_Obj)
	FFT_T_OPRunning[&FFT_T_OP_Obj] = false
	FFT_T_OPDataCopied[&FFT_T_OP_Obj] = false
	return &FFT_T_OP_Obj
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
			maxF = 1 / (2 * period)
		}

		s.dF = dF
		s.maxF = maxF
		s.minF = minF
		s.fDataSet = true
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	cuda.SetCurrent_Ctx()

	var dataT *data.Slice
	var FFTOP *fftOperation3D
	if s.kspace {
		qOP := s.qOP
		FFTOP = UnpackUntilFFT3D(qOP)
		FFTOP.evalIntern()
		dataT = FFT3DData[s.q]
	} else {
		dataT = cuda.BufferFFT_T(s.q.NComp(), SizeOf(s.q), NameOf(s.q)+"_real")
		s.dataT = dataT
		s.q.EvalTo(dataT)
	}
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
	var FFT_T_data *data.Slice
	FFTType := "complex"
	if !s.kspace {
		FFTType = "real"
	}
	bufInitalized := slices.Contains(GetKeys[string](&bufsCPU_map), NameOf(s.q)+"_"+FFTType)

	if !bufInitalized {
		size := dataT.Size()
		if !s.kspace {
			size[0] *= 2
		}
		if !FFT_T_in_mem {
			cuda.IncreaseBufMax(cores * 3 * dataT.NComp())
			for core := range cores {
				bufsCPU = append(bufsCPU, data.NewSlice(dataT.NComp(), dataT.Size()))
				cuda.Create_Stream(NameOf(s.q) + "_" + FFTType + fmt.Sprintf("_%d", core))
				bufsGPUIP = append(bufsGPUIP, cuda.BufferFFT_T(dataT.NComp(), dataT.Size(), fmt.Sprintf(NameOf(s.q)+"_%d", core)))
				bufsGPUOP = append(bufsGPUOP, cuda.BufferFFT_T(dataT.NComp(), size, fmt.Sprintf(NameOf(s.q)+"_%d", core)))
			}
		} else {
			cuda.IncreaseBufMax(dataT.NComp())
			cuda.Create_Stream(NameOf(s.q) + "_" + FFTType)
			FFT_T_data = cuda.BufferFFT_T_F(dataT.NComp(), size, amountFiles, NameOf(s.q))
			FFT_T_data_map.Store(NameOf(s.q)+"_"+FFTType, FFT_T_data)
		}

		bufsCPU_map.Store(NameOf(s.q)+"_"+FFTType, bufsCPU)
		bufsGPUIP_map.Store(NameOf(s.q)+"_"+FFTType, bufsGPUIP)
		bufsGPUOP_map.Store(NameOf(s.q)+"_"+FFTType, bufsGPUOP)
	}
	mu.Lock()
	FFT_T_OPDataCopied[s] = true
	condDataCopied.Broadcast()
	mu.Unlock()
	if bufInitalized {
		var ok bool
		var tmpVar any
		if !FFT_T_in_mem {
			tmpVar, ok = bufsCPU_map.Load(NameOf(s.q) + "_" + FFTType)
			if ok {
				bufsCPU = tmpVar.([]*data.Slice)
			} else {
				panic("Could not load CPU buffers.")
			}
			tmpVar, ok = bufsGPUIP_map.Load(NameOf(s.q) + "_" + FFTType)
			if ok {
				bufsGPUIP = tmpVar.([]*data.Slice)
			} else {
				panic("Could not load CPU IP buffers.")
			}
			tmpVar, ok = bufsGPUOP_map.Load(NameOf(s.q) + "_" + FFTType)
			if ok {
				bufsGPUOP = tmpVar.([]*data.Slice)
			} else {
				panic("Could not load GPU OP buffers.")
			}
		} else {
			tmpVar, ok = FFT_T_data_map.Load(NameOf(s.q) + "_" + FFTType)
			if ok {
				FFT_T_data = tmpVar.(*data.Slice)
			} else {
				panic("Could not load FFT_T buffers.")
			}
		}
	}
	var (
		NxNyNz          [3]int
		startK          [3]float64
		endK            [3]float64
		transformedAxis []string
	)
	if s.kspace {
		NxNyNz, startK, endK, transformedAxis = FFTOP.Axis()
	} else {
		meshQ := MeshOf(s.q)
		NxNyNz = meshQ.Size()
		startK = [3]float64{0, 0, 0}
		endK = [3]float64{float64(NxNyNz[0]) * meshQ.CellSize()[0], float64(NxNyNz[1]) * meshQ.CellSize()[1], float64(NxNyNz[1]) * meshQ.CellSize()[1]}
	}
	fftT := Time
	if !FFT_T_in_mem {
		wg := sync.WaitGroup{}
		for core := range cores {
			upperEnd := 0
			if lowerEnd+filesPerCore < amountFiles {
				upperEnd = (core + 1) * filesPerCore
			} else {
				upperEnd = amountFiles
			}
			wg.Add(1)
			go func(core int, s *fftOperation4D, startIndex, endIndex int, bufCPU, bufGPUIP, bufGPUOP, dataT *data.Slice, FFTOP *fftOperation3D, NxNyNz [3]int, startK, endK [3]float64, transformedAxis []string, fftT float64) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				cuda.SetCurrent_Ctx()
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
					if s.kspace {
						cuda.FFT_T_Step_Complex(bufGPUOP, bufGPUIP, dataT, float32(phase), amountFiles, fmt.Sprintf(NameOf(s.q)+"_"+FFTType+"_%d", core))
						info := data.Meta{Freq: s.minF + float64(i)*s.dF, Name: "k_x_y_z_f_" + s.label, Unit: UnitOf(FFTOP), CellSize: MeshOf(FFTOP).CellSize()}
						saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+s.label, i)+".ovf", bufGPUOP.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false, "real+imag")
					} else {
						cuda.FFT_T_Step_Real(bufGPUOP, bufGPUIP, dataT, float32(phase), amountFiles, fmt.Sprintf(NameOf(s.q)+"_"+FFTType+"_%d", core))
						info := data.Meta{Freq: s.minF + float64(i)*s.dF, Name: "f_" + s.label, Unit: UnitOf(s.q), CellSize: MeshOf(s.q).CellSize()}
						saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "f_"+s.label, i)+".ovf", bufGPUOP.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false, "real+imag")
					}
				}
				wg.Done()
			}(core, s, lowerEnd, upperEnd, bufsCPU[core], bufsGPUIP[core], bufsGPUOP[core], dataT, FFTOP, NxNyNz, startK, endK, transformedAxis, fftT)
			lowerEnd += filesPerCore
		}
		wg.Wait()

	} else {
		if s.kspace {
			cuda.FFT_T_Step_MEM_Complex(FFT_T_data, FFT_T_data, dataT, float32(s.minF), float32(s.dF), float32(fftT), NameOf(s.q)+"_"+FFTType)
		} else {
			cuda.FFT_T_Step_MEM_Real(FFT_T_data, FFT_T_data, dataT, float32(s.minF), float32(s.dF), float32(fftT), NameOf(s.q)+"_"+FFTType)
		}
	}
	if FFT_T_in_mem {
		FFT_T_data_map.Store(NameOf(s.q)+"_"+FFTType, FFT_T_data)
	}
	mu.Lock()
	FFT_T_OPRunning[s] = false
	condCalcComplete.Broadcast()
	mu.Unlock()
}

func (s *fftOperation4D) SaveResults() {
	var (
		NxNyNz          [3]int
		startK          [3]float64
		endK            [3]float64
		transformedAxis []string
		FFTType         string
	)
	if !s.kspace {
		FFTType = "real"
		NxNyNz = MeshOf(s.q).Size()
		Cellsize := MeshOf(s.q).CellSize()
		startK = [3]float64{0, 0, 0}
		endK = [3]float64{float64(NxNyNz[0]) * Cellsize[0], float64(NxNyNz[1]) * Cellsize[1], float64(NxNyNz[2]) * Cellsize[2]}
		transformedAxis = []string{""}
	} else {
		FFTType = "complex"
		FFTOP := s.fft3D
		NxNyNz, startK, endK, transformedAxis = FFTOP.Axis()
	}

	tmpVar, ok := FFT_T_data_map.Load(NameOf(s.q) + "_" + FFTType)
	var FFT_T_data *data.Slice
	if ok {
		FFT_T_data = tmpVar.(*data.Slice)
	} else {
		panic("FFT_T data could not be found during export.")
	}
	size := FFT_T_data.Size()
	if s.polar {
		cuda.ComplexToPolar(FFT_T_data, FFT_T_data)
	}
	for i := range FFT_T_data.LengthF {
		if s.kspace {
			FFTOP := s.fft3D
			info := data.Meta{Freq: s.minF + float64(i)*s.dF, Name: "k_x_y_z_f_" + s.label, Unit: UnitOf(FFTOP), CellSize: MeshOf(FFTOP).CellSize()}
			if s.polar {
				polarBuffer := cuda.Buffer(FFT_T_data.NComp(), size)
				if s.phi || s.abs {
					data.CopyPart(polarBuffer, FFT_T_data, 0, size[X], 0, size[Y], 0, size[Z], i, i+1, 0, 0, 0, 0)
				}
				if !s.phi && !s.abs {
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_polar_"+s.label, i)+".ovf", FFT_T_data.HostCopyPart(0, size[X], 0, size[Y], 0, size[Z], i, i+1), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false, "abs+phi")
				} else if s.phi {
					sizePhi := size
					sizePhi[0] /= 2
					phiBuffer := cuda.Buffer(FFT_T_data.NComp(), sizePhi)
					cuda.Imag(phiBuffer, polarBuffer) //same operation as imag - alternating real and imag or abs and phi
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_phi_"+s.label, i)+".ovf", phiBuffer.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, false, false, "phi")
					cuda.Recycle(phiBuffer)
				}
				if s.abs {
					sizeAbs := size
					sizeAbs[0] /= 2
					absBuffer := cuda.Buffer(FFT_T_data.NComp(), sizeAbs)
					cuda.Real(absBuffer, polarBuffer) //same operation as imag - alternating real and imag or abs and phi
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_abs_"+s.label, i)+".ovf", absBuffer.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, false, false, "abs")
					cuda.Recycle(absBuffer)
				}
				cuda.Recycle(polarBuffer)
			} else {
				saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "k_x_y_z_f_"+s.label, i)+".ovf", FFT_T_data.HostCopyPart(0, size[X], 0, size[Y], 0, size[Z], i, i+1), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false, "real+imag")
			}
		} else {
			info := data.Meta{Freq: s.minF + float64(i)*s.dF, Name: "f_" + s.label, Unit: UnitOf(s.q), CellSize: MeshOf(s.q).CellSize()}
			if s.polar {
				polarBuffer := cuda.Buffer(FFT_T_data.NComp(), size)
				if s.phi || s.abs {
					data.CopyPart(polarBuffer, FFT_T_data, 0, size[X], 0, size[Y], 0, size[Z], i, i+1, 0, 0, 0, 0)
				}
				if !s.phi && !s.abs {
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "f_polar_"+s.label, i)+".ovf", FFT_T_data.HostCopyPart(0, size[X], 0, size[Y], 0, size[Z], i, i+1), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false, "abs+phi")
				} else if s.phi {
					sizePhi := size
					sizePhi[0] /= 2
					phiBuffer := cuda.Buffer(FFT_T_data.NComp(), sizePhi)
					cuda.Imag(phiBuffer, polarBuffer) //same operation as imag - alternating real and imag or abs and phi
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "f_phi_"+s.label, i)+".ovf", phiBuffer.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, false, false, "phi")
					cuda.Recycle(phiBuffer)
				}
				if s.abs {
					sizeAbs := size
					sizeAbs[0] /= 2
					absBuffer := cuda.Buffer(FFT_T_data.NComp(), sizeAbs)
					cuda.Real(absBuffer, polarBuffer) //same operation as imag - alternating real and imag or abs and phi
					saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "f_abs_"+s.label, i)+".ovf", absBuffer.HostCopy(), info, outputFormat, NxNyNz, startK, endK, transformedAxis, false, false, "abs")
					cuda.Recycle(absBuffer)
				}
				cuda.Recycle(polarBuffer)
			} else {
				saveAsFFT_sync(OD()+fmt.Sprintf(FilenameFormat, "f_"+s.label, i)+".ovf", FFT_T_data.HostCopyPart(0, size[X], 0, size[Y], 0, size[Z], i, i+1), info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, false, "real+imag")
			}
		}

	}
}

func (s *fftOperation4D) Clear_Buffers() {
	FFTType := "complex"
	if !s.kspace {
		FFTType = "real"
	}
	FFT_T_data_buffer_length := 0
	if FFT_T_in_mem {
		tmpVar, ok := FFT_T_data_map.Load(NameOf(s.q) + "_" + FFTType)
		if ok {
			FFT_T_data := tmpVar.(*data.Slice)
			cuda.Recycle_FFT_T(FFT_T_data, NameOf(s.q))
			FFT_T_data_buffer_length += 1
			FFT_T_data_map.Delete(NameOf(s.q) + "_" + FFTType)
		}
	}
	tmpVar, ok := bufsCPU_map.Load(NameOf(s.q) + "_" + FFTType)
	lengthBuffers := 0
	if ok {
		bufsCPU := tmpVar.([]*data.Slice)
		for i := range len(bufsCPU) {
			bufsCPU[i].Free()
			lengthBuffers += 1
		}
		bufsCPU_map.Delete(NameOf(s.q) + "_" + FFTType)
	}
	tmpVar, ok = bufsGPUIP_map.Load(NameOf(s.q) + "_" + FFTType)
	if ok {
		bufsGPUIP := tmpVar.([]*data.Slice)
		for i := range len(bufsGPUIP) {
			cuda.Recycle_FFT_T(bufsGPUIP[i], fmt.Sprintf(NameOf(s.q)+"_%d", i))
		}
		bufsGPUIP_map.Delete(NameOf(s.q) + "_" + FFTType)
	}
	tmpVar, ok = bufsGPUOP_map.Load(NameOf(s.q) + "_" + FFTType)
	if ok {
		bufsGPUOP := tmpVar.([]*data.Slice)
		for i := range len(bufsGPUOP) {
			cuda.Recycle_FFT_T(bufsGPUOP[i], fmt.Sprintf(NameOf(s.q)+"_%d", i))
			cuda.Destroy_Stream(fmt.Sprintf(NameOf(s.q)+"_%d", i))

		}
		bufsGPUOP_map.Delete(NameOf(s.q) + "_" + FFTType)
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
			if !FFT_T_OP[i].kspace {
				if FFT_T_OP[i].dataT != nil {
					cuda.Recycle_FFT_T(FFT_T_OP[i].dataT, NameOf(FFT_T_OP[i].q))
				}
			}
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
