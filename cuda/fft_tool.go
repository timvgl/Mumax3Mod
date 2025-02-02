package cuda

import (
	//"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	//"github.com/mumax/3/engine"
	//"github.com/mumax/3/engine"
)

var (
// FFT3DR2CPlanInitialized = false
// FFT2DR2CPlan fftManyR2CPlan
// FFT2DR2CPlanInitialized = false
)

func Initialize3DR2CFFT(Nx, Ny, Nz int) fft3DR2CPlan {
	return newFFT3DR2C(Nx, Ny, Nz)
}

func Initialize3DR2CFFT_FFT_T(Nx, Ny, Nz int, streamFFT_T_ cu.Stream) fft3DR2CPlan {
	return newFFT3DR2C_FFT_T(Nx, Ny, Nz, streamFFT_T_)
}

func Perform3DR2CFFT(src, dst *data.Slice, FFT3DR2CPlanInterface interface{}) {
	FFT3DR2CPlan := FFT3DR2CPlanInterface.(fft3DR2CPlan)
	FFT3DR2CPlan.ExecAsync(src, dst)
}

func Perform3DR2CFFT_T(src, dst *data.Slice, FFT3DR2CPlanInterface interface{}, key string) {
	FFT3DR2CPlan := FFT3DR2CPlanInterface.(fft3DR2CPlan)
	FFT3DR2CPlan.ExecAsync_FFT_T(src, dst, key)
}

func OutputSizeFloatsFFT3D(FFT3DR2CPlanInterface interface{}) (int, int, int) {
	FFT3DR2CPlan := FFT3DR2CPlanInterface.(fft3DR2CPlan)
	return FFT3DR2CPlan.OutputSizeFloats()
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
