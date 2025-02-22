package cuda

import (
	"github.com/mumax/3/data"
)

// Calculate [f(t)+bf+melFroce-eta*g(t)]/rho
func RightSide(dst, f, g *data.Slice, eta, rho MSlice, bf, melForce, thermalElasticNoise *data.Slice) {

	N := dst.Len()
	cfg := make1DConf(N)

	//eta*g(t)
	k_mul_mslice_async(dst.DevPtr(0), eta.DevPtr(0), eta.Mul(0), g.DevPtr(0), 1, N, cfg)
	k_mul_mslice_async(dst.DevPtr(1), eta.DevPtr(0), eta.Mul(0), g.DevPtr(1), 1, N, cfg)
	k_mul_mslice_async(dst.DevPtr(2), eta.DevPtr(0), eta.Mul(0), g.DevPtr(2), 1, N, cfg)

	//dst = bf-eta*g(t)
	Madd2(dst, bf, dst, 1, -1)

	//dst=f(t)+bf-eta*g(t)
	Madd2(dst, f, dst, 1, 1)

	//dst = f(t) + bf + melForce - eta*g(t)
	Madd2(dst, melForce, dst, 1, 1)

	//dst = f(t) + bf + melForce - eta*g(t) + thermalElasticNoise
	Madd2(dst, dst, thermalElasticNoise, 1, 1)

	//dst = [f(t)+bf-eta*g(t)]/rho
	k_pointwise_div_mslice_async(dst.DevPtr(0), dst.DevPtr(0), 1, rho.DevPtr(0), rho.Mul(0), N, cfg)
	k_pointwise_div_mslice_async(dst.DevPtr(1), dst.DevPtr(1), 1, rho.DevPtr(0), rho.Mul(0), N, cfg)
	k_pointwise_div_mslice_async(dst.DevPtr(2), dst.DevPtr(2), 1, rho.DevPtr(0), rho.Mul(0), N, cfg)
}
