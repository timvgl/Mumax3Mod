package cuda

import (
	//"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	//"github.com/mumax/3/util"
	//"github.com/mumax/3/engine"
)

var (
	FFT3DR2CPlan            fft3DR2CPlan
	FFT3DR2CPlanInitialized = false
	//FFT2DR2CPlan fftManyR2CPlan
	//FFT2DR2CPlanInitialized = false
)

func Initialize3DR2CFFT(Nx, Ny, Nz int) {
	FFT3DR2CPlan = newFFT3DR2C(Nx, Ny, Nz)
	FFT3DR2CPlanInitialized = true
}

func Perform3DR2CFFT(src, dst *data.Slice) {
	if FFT3DR2CPlanInitialized {
		FFT3DR2CPlan.ExecAsync(src, dst)
	} else {
		panic("Plan for FFT has not been initalized.")
	}
}

/*
func Initialize2DR2CFFT(Nx, Ny, Nz int) {
	FFT2DR2CPlan = newFFTManyR2C(Nx, Ny, Nz)
	FFT2DR2CPlanInitialized = true
}

func Perform2DR2CFFT(src, dst *data.Slice) {
	if FFT2DR2CPlanInitialized {
		FFT2DR2CPlan.ExecAsync(src, dst)
	} else {
		panic("Plan for FFT has not been initalized.")
	}
}
*/
