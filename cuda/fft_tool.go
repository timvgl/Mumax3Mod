package cuda

import (
	//"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	//"github.com/mumax/3/util"
	//"github.com/mumax/3/engine"
)

var (
	FFT3DR2CPlan fft3DR2CPlan
	FFT3DR2CPlanInitialized = false
)

func InitializeR2CFFT(Nx, Ny, Nz int) {
	FFT3DR2CPlan = newFFT3DR2C(Nx, Ny, Nz)
	FFT3DR2CPlanInitialized = true
}

func PerformR2CFFT(src, dst *data.Slice) {
	if FFT3DR2CPlanInitialized {
		FFT3DR2CPlan.ExecAsync(src, dst)
	} else {
		panic("Plan for FFT has not been initalized.")
	}
}

