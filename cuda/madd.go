package cuda

import (
	"fmt"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// multiply: dst[i] = a[i] * b[i]
// a and b must either have the same number of components
// or a can have an arbitrary amount of components, while b has one component
func Mul(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp || a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == 1)
	cfg := make1DConf(N)
	if b.NComp() == a.NComp() {
		for c := 0; c < nComp; c++ {
			k_mul_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg)
		}
	}
	if b.NComp() != a.NComp() && b.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_mul_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(0), N, cfg)
		}
	}
}

func MulMSlice(dst *data.Slice, a, b MSlice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp || a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == 1)
	cfg := make1DConf(N)
	if b.NComp() == a.NComp() {
		for c := 0; c < nComp; c++ {
			k_mul_mslice_async(dst.DevPtr(c), a.DevPtr(c), a.Mul(c), b.DevPtr(c), b.Mul(c), N, cfg)
		}
	}
	if b.NComp() != a.NComp() && b.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_mul_mslice_async(dst.DevPtr(c), a.DevPtr(c), a.Mul(c), b.DevPtr(0), b.Mul(0), N, cfg)
		}
	}
}

// divide: dst[i] = a[i] / b[i]
// divide-by-zero yields zero.
func Div(dst, a, b *data.Slice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp || a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == 1)
	cfg := make1DConf(N)
	if b.NComp() == a.NComp() {
		for c := 0; c < nComp; c++ {
			k_pointwise_div_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), N, cfg)
		}
	}
	if b.NComp() != a.NComp() && b.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_pointwise_div_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(0), N, cfg)
		}
	}
}
func DivMSlice(dst *data.Slice, a, b MSlice) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == nComp || a.Len() == N && a.NComp() == nComp && b.Len() == N && b.NComp() == 1)
	cfg := make1DConf(N)
	if b.NComp() == a.NComp() {
		for c := 0; c < nComp; c++ {
			k_pointwise_div_mslice_async(dst.DevPtr(c), a.DevPtr(c), a.Mul(c), b.DevPtr(c), b.Mul(c), N, cfg)
		}
	}
	if b.NComp() != a.NComp() && b.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_pointwise_div_mslice_async(dst.DevPtr(c), a.DevPtr(c), a.Mul(c), b.DevPtr(0), b.Mul(0), N, cfg)
		}
	}
}

// Add: dst = src1 + src2.
func Add(dst, src1, src2 *data.Slice) {
	Madd2(dst, src1, src2, 1, 1)
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2
func Madd2(dst, src1, src2 *data.Slice, factor1, factor2 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N)
	util.AssertMsg(src1.NComp() == nComp && src2.NComp() == nComp || src1.NComp() == nComp && src2.NComp() == 1 || src2.NComp() == nComp && src1.NComp() == 1, fmt.Sprintf("Comp: %v vs %v vs %v", nComp, src1.NComp(), src2.NComp()))
	cfg := make1DConf(N)
	if src1.NComp() == nComp && src2.NComp() == nComp {
		for c := 0; c < nComp; c++ {
			k_madd2_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
				src2.DevPtr(c), factor2, N, cfg)
		}
	} else if src1.NComp() == nComp && src2.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_madd2_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
				src2.DevPtr(0), factor2, N, cfg)
		}
	} else if src2.NComp() == nComp && src1.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_madd2_async(dst.DevPtr(c), src1.DevPtr(0), factor1,
				src2.DevPtr(c), factor2, N, cfg)
		}
	}
}

func Madd2Comp(dst, src1, src2 *data.Slice, factor1, factor2 []float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N)
	util.AssertMsg(src1.NComp() == len(factor1) && src2.NComp() == len(factor2) && src1.NComp() == nComp && src2.NComp() == nComp || src1.NComp() == nComp && src2.NComp() == 1 || src2.NComp() == nComp && src1.NComp() == 1, fmt.Sprintf("Comp: %v vs %v vs %v", nComp, src1.NComp(), src2.NComp()))
	cfg := make1DConf(N)
	if src1.NComp() == nComp && src2.NComp() == nComp {
		for c := 0; c < nComp; c++ {
			k_madd2_async(dst.DevPtr(c), src1.DevPtr(c), factor1[c],
				src2.DevPtr(c), factor2[c], N, cfg)
		}
	} else if src1.NComp() == nComp && src2.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_madd2_async(dst.DevPtr(c), src1.DevPtr(c), factor1[c],
				src2.DevPtr(0), factor2[0], N, cfg)
		}
	} else if src2.NComp() == nComp && src1.NComp() == 1 {
		for c := 0; c < nComp; c++ {
			k_madd2_async(dst.DevPtr(c), src1.DevPtr(0), factor1[0],
				src2.DevPtr(c), factor2[c], N, cfg)
		}
	}
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3[i] * factor3
func Madd3(dst, src1, src2, src3 *data.Slice, factor1, factor2, factor3 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_madd3_async(dst.DevPtr(c), src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2, src3.DevPtr(c), factor3, N, cfg)
	}
}

// scale:  x[i] = scale*x[i] + offset
func Scale(x *data.Slice, scale float32, offset []float64) {
	nComp := x.NComp()
	util.Assert(nComp == len(offset))
	N := x.Len()
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_scale_async(x.DevPtr(c), scale, -float32(offset[c]), N, cfg)
	}
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3[i] * factor3 + src4[i] * factor4
func Madd4(dst, src1, src2, src3, src4 *data.Slice, factor1, factor2, factor3, factor4 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N && src4.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp && src4.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_madd4_async(dst.DevPtr(c),
			src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2,
			src3.DevPtr(c), factor3,
			src4.DevPtr(c), factor4, N, cfg)
	}
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3[i] * factor3 + src4[i] * factor4 + src5[i] * factor5
func Madd5(dst, src1, src2, src3, src4, src5 *data.Slice, factor1, factor2, factor3, factor4, factor5 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N && src4.Len() == N && src5.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp && src4.NComp() == nComp && src5.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_madd5_async(dst.DevPtr(c),
			src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2,
			src3.DevPtr(c), factor3,
			src4.DevPtr(c), factor4,
			src5.DevPtr(c), factor5, N, cfg)
	}
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3[i] * factor3 + src4[i] * factor4 + src5[i] * factor5 + src6[i] * factor6
func Madd6(dst, src1, src2, src3, src4, src5, src6 *data.Slice, factor1, factor2, factor3, factor4, factor5, factor6 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N && src4.Len() == N && src5.Len() == N && src6.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp && src4.NComp() == nComp && src5.NComp() == nComp && src6.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_madd6_async(dst.DevPtr(c),
			src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2,
			src3.DevPtr(c), factor3,
			src4.DevPtr(c), factor4,
			src5.DevPtr(c), factor5,
			src6.DevPtr(c), factor6, N, cfg)
	}
}

// multiply-add: dst[i] = src1[i] * factor1 + src2[i] * factor2 + src3[i] * factor3 + src4[i] * factor4 + src5[i] * factor5 + src6[i] * factor6 + src7[i] * factor7
func Madd7(dst, src1, src2, src3, src4, src5, src6, src7 *data.Slice, factor1, factor2, factor3, factor4, factor5, factor6, factor7 float32) {
	N := dst.Len()
	nComp := dst.NComp()
	util.Assert(src1.Len() == N && src2.Len() == N && src3.Len() == N && src4.Len() == N && src5.Len() == N && src6.Len() == N && src7.Len() == N)
	util.Assert(src1.NComp() == nComp && src2.NComp() == nComp && src3.NComp() == nComp && src4.NComp() == nComp && src5.NComp() == nComp && src6.NComp() == nComp && src7.NComp() == nComp)
	cfg := make1DConf(N)
	for c := 0; c < nComp; c++ {
		k_madd7_async(dst.DevPtr(c),
			src1.DevPtr(c), factor1,
			src2.DevPtr(c), factor2,
			src3.DevPtr(c), factor3,
			src4.DevPtr(c), factor4,
			src5.DevPtr(c), factor5,
			src6.DevPtr(c), factor6,
			src7.DevPtr(c), factor7, N, cfg)
	}
}
