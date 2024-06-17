package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func Grad(out, a *data.Slice, mesh *data.Mesh) {
	util.Argument(out.Size() == mesh.Size())

	cellsize := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	rcsx := float32(1.0 / cellsize[X])
	rcsy := float32(1.0 / cellsize[Y])
	rcsz := float32(1.0 / cellsize[Z])
	k_pointwise_grad_async(out.DevPtr(X), out.DevPtr(Y), out.DevPtr(Z), a.DevPtr(X), rcsx, rcsy, rcsz, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

func Grad2(out, a *data.Slice, mesh *data.Mesh) {
	util.Argument(out.Size() == mesh.Size())

	cellsize := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	rcsx := float32(1.0 / cellsize[X])
	rcsy := float32(1.0 / cellsize[Y])
	rcsz := float32(1.0 / cellsize[Z])
	k_pointwise_grad_async(out.DevPtr(X), out.DevPtr(Y), out.DevPtr(Z), a.DevPtr(Y), rcsx, rcsy, rcsz, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}

func Grad3(out, a *data.Slice, mesh *data.Mesh) {
	util.Argument(out.Size() == mesh.Size())

	cellsize := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	rcsx := float32(1.0 / cellsize[X])
	rcsy := float32(1.0 / cellsize[Y])
	rcsz := float32(1.0 / cellsize[Z])
	k_pointwise_grad_async(out.DevPtr(X), out.DevPtr(Y), out.DevPtr(Z), a.DevPtr(Z), rcsx, rcsy, rcsz, N[X], N[Y], N[Z], mesh.PBC_code(), cfg)
}