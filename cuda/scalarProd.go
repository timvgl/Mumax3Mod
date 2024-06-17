package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func ScalarProd(out, a, b *data.Slice, mesh *data.Mesh) {
	util.Argument(a.Size() == b.Size())
	util.Argument(a.Size() == out.Size())

	N := mesh.Size()
	cfg := make3DConf(N)

	k_scalarProd_async(out.DevPtr(X),
							a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
							b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
							N[X], N[Y], N[Z], cfg)
}