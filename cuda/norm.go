package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Normalize vec to unit length, unless length or vol are zero.
func Norm(dst, vec *data.Slice, mesh *data.Mesh) {
	util.Argument(vec == nil || vec.NComp() == 3)
	N := vec.Size()
	cfg := make3DConf(N)
	k_pointwise_norm_async(dst.DevPtr(0), vec.DevPtr(X), vec.DevPtr(Y), vec.DevPtr(Z), N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}
