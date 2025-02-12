// Package cuda provides GPU interaction
package cuda

import (
	"fmt"
	"log"
	"runtime"
	"sync"

	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/util"
)

var (
	DriverVersion int        // cuda driver version
	DevName       string     // GPU name
	TotalMem      int64      // total GPU memory
	GPUInfo       string     // Human-readable GPU description
	Synchronous   bool       // for debug: synchronize stream0 at every kernel launch
	cudaCtx       cu.Context // global CUDA context
	cudaCC        int        // compute capablity (used for fatbin)
	GPUIndex      int
)

// Locks to an OS thread and initializes CUDA for that thread.
func Init(gpu int) {
	if cudaCtx != 0 {
		return // needed for tests
	}

	runtime.LockOSThread()
	tryCuInit()
	dev := cu.Device(gpu)
	cudaCtx = cu.CtxCreate(cu.CTX_SCHED_YIELD, dev)
	cudaCtx.SetCurrent()

	M, m := dev.ComputeCapability()
	cudaCC = 10*M + m
	DriverVersion = cu.Version()
	DevName = dev.Name()
	TotalMem = dev.TotalMem()
	GPUInfo = fmt.Sprintf("%s(%dMB), CUDA Driver %d.%d, cc=%d.%d",
		DevName, (TotalMem)/(1024*1024), DriverVersion/1000, (DriverVersion%1000)/10, M, m)

	if M < 2 {
		log.Fatalln("GPU has insufficient compute capability, need 2.0 or higher.")
	}
	if Synchronous {
		log.Println("DEBUG: synchronized CUDA calls")
	}
	GPUIndex = gpu
	// test PTX load so that we can catch CUDA_ERROR_NO_BINARY_FOR_GPU early
	fatbinLoad(madd2_map, "madd2")
}
func SetCurrent_Ctx() {
	cudaCtx.SetCurrent()
}

/*
type cudaContext struct {
	DriverVersion int        // cuda driver version
	DevName       string     // GPU name
	TotalMem      int64      // total GPU memory
	GPUInfo       string     // Human-readable GPU description
	cudaCtx       cu.Context // global CUDA context
	cudaCC        int        // compute capablity (used for fatbin)
	GPUIndex      int
}

func Set_cudaCtx(cudaCtxTmp cudaContext) {
	cudaCtx = cudaCtxTmp.cudaCtx
}

func Set_cudaCC(cudaCtxTmp cudaContext) {
	cudaCC = cudaCtxTmp.cudaCC
}

func Get_cudaCtx(cudaCtxTmp cudaContext) cu.Context {
	return cudaCtxTmp.cudaCtx
}

func InitFFT4D(gpu int) cudaContext {

	runtime.LockOSThread()
	tryCuInit()
	dev := cu.Device(gpu)
	cudaCtxFFT4D := cu.CtxCreate(cu.CTX_SCHED_YIELD, dev)
	cudaCtxFFT4D.SetCurrent()

	M, m := dev.ComputeCapability()
	cudaCCFFT4D := 10*M + m
	DriverVersionFFT4D := cu.Version()
	DevNameFFT4D := dev.Name()
	TotalMemFFT4D := dev.TotalMem()
	GPUInfoFFT4D := fmt.Sprintf("%s(%dMB), CUDA Driver %d.%d, cc=%d.%d",
		DevNameFFT4D, (TotalMemFFT4D)/(1024*1024), DriverVersionFFT4D/1000, (DriverVersionFFT4D%1000)/10, M, m)

	if M < 2 {
		log.Fatalln("GPU has insufficient compute capability, need 2.0 or higher.")
	}
	GPUIndex = gpu
	// test PTX load so that we can catch CUDA_ERROR_NO_BINARY_FOR_GPU early
	fatbinLoad(madd2_map, "madd2")
	return cudaContext{DriverVersionFFT4D, DevNameFFT4D, TotalMemFFT4D, GPUInfoFFT4D, cudaCtxFFT4D, cudaCCFFT4D, GPUIndex}
}
*/

// cu.Init(), but error is fatal and does not dump stack.
func tryCuInit() {
	defer func() {
		err := recover()
		if err == cu.ERROR_UNKNOWN {
			log.Println("\n Try running: sudo nvidia-modprobe -u \n")
		}
		util.FatalErr(err)
	}()
	cu.Init(0)
}

// Global stream used for everything
const stream0 = cu.Stream(0)

var StreamMap sync.Map

// Hahaha not quiet - so we need a new stream for the async FFT in time - one Stream per FFT parallelization
func Create_Stream(key string) {
	if !Check_Stream(key) {
		StreamMap.Store(key, cu.StreamCreate())
	}
}

func Check_Stream(key string) bool {
	_, ok := StreamMap.Load(key)
	return ok
}

func Destroy_Stream(key string) {
	if len(key) > 0 {
		val, ok := StreamMap.Load(key)
		if ok {
			streamTmp := (val.(cu.Stream))
			streamTmp.Destroy()
			StreamMap.Delete(key)
		} else {
			panic(fmt.Sprintf("Couldnot find stream for key %v.", key))
		}
	} else {
		panic("Cannot destroy not found stream.")
	}
}

func Get_Stream(key string) cu.Stream {
	if len(key) > 0 && key != "nil" {
		val, ok := StreamMap.Load(key)
		if ok {
			return val.(cu.Stream)
		} else {
			panic(fmt.Sprintf("Couldnot find stream for key %v.", key))
		}
	} else if key == "nil" {
		panic("Nil is not valid for Get_Stream")
	} else {
		return stream0
	}
}

// Synchronize the global stream
// This is called before and after all memcopy operations between host and device.
func Sync() {
	stream0.Synchronize()
}

func SyncFFT_T(key string) {
	if len(key) > 0 && key != "nil" {
		val, ok := StreamMap.Load(key)
		if ok {
			val.(cu.Stream).Synchronize()
		} else {
			panic(fmt.Sprintf("Couldnot find stream for key %v.", key))
		}
	} else if key == "nil" {
	} else {
		stream0.Synchronize()
	}
}
