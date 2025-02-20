package cuda

import (
	"math"

	"github.com/mumax/3/data"
)

func Fill1DWithCoords(dst *data.Slice, factor float32) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_fill1DWithCoords_async(dst.DevPtr(c), factor, prod(size), cfg)
	}
}

func checkSizeGovaluate(aSize, bSize [3]int) {
	for val, _ := range aSize {
		if aSize[val] != bSize[val] && (aSize[val] != 1 && bSize[val] != 1) {
			panic("Size mismatch")
		}
	}
}

func AddGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_addGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func AddGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_addGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func AddGovaluate1X3(dst *data.Slice, b interface{}, a *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_addGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func SubGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_subGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func SubGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_addGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), -b.(float32), prod(size), cfg)
	}
}

func SubGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_subGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func MulGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_mulGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func MulGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_mulGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func MulGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_mulGovaluate3X1_async(dst.DevPtr(c), b.DevPtr(c), a.(float32), prod(size), cfg)
	}
}

func DivGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_divGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func DivGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_divGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func DivGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_divGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func PowGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_powGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func PowGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_powGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func PowGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_powGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func ModGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_modGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func ModGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_modGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func ModGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_modGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func Atan2Govaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_atan2Govaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func Atan2Govaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_atan2Govaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func Atan2Govaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_atan2Govaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func HypotGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_hypotGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func HypotGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_hypotGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func HypotGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_hypotGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func RemainderGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_remainderGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func RemainderGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_remainderGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func RemainderGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_remainderGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func YnGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_YnGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func YnGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_YnGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func YnGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_YnGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

func LdexpGovaluate3X3(dst, a, b *data.Slice) {
	size := dst.Size()
	sizeA := a.Size()
	sizeB := b.Size()
	checkSizeGovaluate(sizeA, sizeB)
	cfg := make3DConf(size)
	for c := range dst.NComp() {
		k_ldexpGovaluate3X3_async(dst.DevPtr(c), a.DevPtr(c), b.DevPtr(c), size[X], size[Y], size[Z], sizeA[X], sizeA[Y], sizeA[Z], sizeB[X], sizeB[Y], sizeB[Z], cfg)
	}
}

func LdexpGovaluate3X1(dst, a *data.Slice, b interface{}) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_ldexpGovaluate3X1_async(dst.DevPtr(c), a.DevPtr(c), b.(float32), prod(size), cfg)
	}
}

func LdexpGovaluate1X3(dst *data.Slice, a interface{}, b *data.Slice) {
	size := dst.Size()
	cfg := make1DConf(prod(size))
	for c := range dst.NComp() {
		k_ldexpGovaluate1X3_async(dst.DevPtr(c), a.(float32), b.DevPtr(c), prod(size), cfg)
	}
}

// These functions assume that the following helper functions are defined elsewhere:
// - func prod(size [3]int) int            // returns the total number of elements
// - func make1DConf(n int) int             // returns the kernel launch configuration for a 1D kernel
// - and that data.Slice methods Size(), NComp(), and DevPtr(c int) are available.

// AbsGovaluate calls the asynchronous absolute-value kernel.
func AbsGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_absGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// AcosGovaluate calls the asynchronous arccosine kernel.
func AcosGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_acosGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// AcoshGovaluate calls the asynchronous inverse hyperbolic cosine kernel.
func AcoshGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_acoshGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// AsinGovaluate calls the asynchronous arcsine kernel.
func AsinGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_asinGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// AsinhGovaluate calls the asynchronous inverse hyperbolic sine kernel.
func AsinhGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_asinhGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// AtanGovaluate calls the asynchronous arctangent kernel.
func AtanGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_atanGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// AtanhGovaluate calls the asynchronous inverse hyperbolic tangent kernel.
func AtanhGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_atanhGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// CbrtGovaluate calls the asynchronous cube-root kernel.
func CbrtGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_cbrtGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// CeilGovaluate calls the asynchronous ceiling kernel.
func CeilGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_ceilGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// CosGovaluate calls the asynchronous cosine kernel.
func CosGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_cosGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// CoshGovaluate calls the asynchronous hyperbolic cosine kernel.
func CoshGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_coshGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// ErfGovaluate calls the asynchronous error function kernel.
func ErfGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_erfGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// ErfcGovaluate calls the asynchronous complementary error function kernel.
func ErfcGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_erfcGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// ExpGovaluate calls the asynchronous exponential kernel.
func ExpGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_expGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// Exp2Govaluate calls the asynchronous base-2 exponential kernel.
func Exp2Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_exp2Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// Expm1Govaluate calls the asynchronous expm1 (exponential minus one) kernel.
func Expm1Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_expm1Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// FloorGovaluate calls the asynchronous floor kernel.
func FloorGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_floorGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// GammaGovaluate calls the asynchronous gamma function kernel.
func GammaGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_gammaGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// J0Govaluate calls the asynchronous Bessel J0 kernel.
func J0Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_j0Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// J1Govaluate calls the asynchronous Bessel J1 kernel.
func J1Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_j1Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// LogGovaluate calls the asynchronous natural logarithm kernel.
func LogGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_logGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// Log10Govaluate calls the asynchronous base-10 logarithm kernel.
func Log10Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_log10Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// Log1pGovaluate calls the asynchronous log1p (log(1+x)) kernel.
func Log1pGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_log1pGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// Log2Govaluate calls the asynchronous base-2 logarithm kernel.
func Log2Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_log2Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// LogbGovaluate calls the asynchronous logarithm of the absolute value kernel.
func LogbGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_logbGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// SinGovaluate calls the asynchronous sine kernel.
func SinGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_sinGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// SinhGovaluate calls the asynchronous hyperbolic sine kernel.
func SinhGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_sinhGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// SqrtGovaluate calls the asynchronous square-root kernel.
func SqrtGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_sqrtGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// TanGovaluate calls the asynchronous tangent kernel.
func TanGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_tanGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// TanhGovaluate calls the asynchronous hyperbolic tangent kernel.
func TanhGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_tanhGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// TruncGovaluate calls the asynchronous truncation kernel.
func TruncGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_truncGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// Y0Govaluate calls the asynchronous Bessel Y0 kernel.
func Y0Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_y0Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// Y1Govaluate calls the asynchronous Bessel Y1 kernel.
func Y1Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_y1Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// IlogbGovaluate calls the asynchronous ilogb kernel.
func IlogbGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_ilogbGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// HeavisideGovaluate calls the asynchronous Heaviside kernel.
func HeavisideGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_heavisideGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// NormGovaluate calls the asynchronous norm (absolute value) kernel.
func NormGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_normGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// SincGovaluate calls the asynchronous sinc kernel.
func SincGovaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_sincGovaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

// SincGovaluate calls the asynchronous sinc kernel.
func Pow10Govaluate(val *data.Slice) {
	size := val.Size()
	cfg := make1DConf(prod(size))
	for c := range val.NComp() {
		k_pow10Govaluate_async(val.DevPtr(c), prod(size), cfg)
	}
}

func MaxGovaluate(val *data.Slice) float64 {
	size := val.Size()
	cfg := make1DConf(prod(size))
	out := reduceBuf(0)
	returnFloat := 0.
	for c := range val.NComp() {
		k_maxGovaluate_async(val.DevPtr(c), out, 0, prod(size), cfg)
		returnFloat += math.Pow(float64(copyback(out)), 2)
	}
	return math.Sqrt(returnFloat)
}

func MinGovaluate(val *data.Slice) float64 {
	size := val.Size()
	cfg := make1DConf(prod(size))
	out := reduceBuf(0)
	returnFloat := 0.
	for c := range val.NComp() {
		k_minGovaluate_async(val.DevPtr(c), out, 0, prod(size), cfg)
		returnFloat += math.Pow(float64(copyback(out)), 2)
	}
	return math.Sqrt(returnFloat)
}
