package cuda

import (
	"github.com/mumax/3/data"
)

func KineticEnergy(dst, du *data.Slice, rho MSlice, mesh *data.Mesh) {
	N := mesh.Size()
	cfg := make3DConf(N)
	k_KineticEnergy_async(dst.DevPtr(X),
		du.DevPtr(X), du.DevPtr(Y), du.DevPtr(Z), rho.DevPtr(0), rho.Mul(0), N[X], N[Y], N[Z], cfg)
}
