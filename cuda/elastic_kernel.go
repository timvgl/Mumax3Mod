package cuda

import (
	"github.com/mumax/3/data"
)

func SecondDerivative(dst, u *data.Slice, mesh *data.Mesh, c1, c2, c3 MSlice) {
	N := mesh.Size()
	w := mesh.CellSize()
	wx := float32(1 / w[0])
	wy := float32(1 / w[1])
	wz := float32(1 / w[2])
	cfg := make3DConf(N)
	pbc := mesh.PBC_code()
	// k_Elastodynamic1_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)
	// k_Elastodynamic2_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)
	// k_Elastodynamic_freebndry_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)

	//Vacuum method
	// k_Elastodynamic_2D_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// 	u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// 	c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// 	pbc, cfg)

	//Adjusted differential equation at edges
	k_Elastos_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
		c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
		pbc, cfg)
	//k_Elastodynamic3_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
	// u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z), N[X], N[Y], N[Z], wx, wy, wz,
	// c1.DevPtr(0), c1.Mul(0), c2.DevPtr(0), c2.Mul(0), c3.DevPtr(0), c3.Mul(0),
	// pbc, cfg)
}

func StressWurtzitMtx(dst1, dst2, dst3, dst4, u, m *data.Slice, C11, C12, C13, C33, C44, B1, B2 MSlice, mesh *data.Mesh, cubic bool) {
	Nx, Ny, Nz := mesh.Size()[0], mesh.Size()[1], mesh.Size()[2]
	dx, dy, dz := mesh.CellSize()[0], mesh.CellSize()[1], mesh.CellSize()[2]
	cfg1D := make1DConf(Nx*Ny*2 + Ny*Nz*2 + Nz*Nx*2)
	k_adaptUNeumannBndry_async(u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		C11.DevPtr(0), C11.Mul(0),
		C12.DevPtr(0), C12.Mul(0),
		C13.DevPtr(0), C13.Mul(0),
		C33.DevPtr(0), C33.Mul(0),
		C44.DevPtr(0), C44.Mul(0),
		B1.DevPtr(0), B1.Mul(0),
		B2.DevPtr(0), B2.Mul(0),
		float32(dx), float32(dy), float32(dz),
		Nx, Ny, Nz,
		mesh.PBC_code(),
		cubic,
		cfg1D)
	Sync()
	cfg3D := make3DConf(mesh.Size())
	k_stressWurtzit_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
		dst3.DevPtr(X), dst3.DevPtr(Y), dst3.DevPtr(Z),
		dst4.DevPtr(X), dst4.DevPtr(Y), dst4.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		C11.DevPtr(0), C11.Mul(0),
		C12.DevPtr(0), C12.Mul(0),
		C13.DevPtr(0), C13.Mul(0),
		C33.DevPtr(0), C33.Mul(0),
		C44.DevPtr(0), C44.Mul(0),
		B1.DevPtr(0), B1.Mul(0),
		B2.DevPtr(0), B2.Mul(0),
		float32(dx), float32(dy), float32(dz),
		Nx, Ny, Nz,
		mesh.PBC_code(),
		cubic,
		cfg3D)
}

func ForceWurtzitMtx(dst, normStress, shearStress *data.Slice, mesh *data.Mesh) {
	Nx, Ny, Nz := mesh.Size()[0], mesh.Size()[1], mesh.Size()[2]
	dx, dy, dz := mesh.CellSize()[0], mesh.CellSize()[1], mesh.CellSize()[2]
	cfg := make3DConf(mesh.Size())
	k_divergenceStress_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		normStress.DevPtr(X), normStress.DevPtr(Y), normStress.DevPtr(Z),
		shearStress.DevPtr(X), shearStress.DevPtr(Y), shearStress.DevPtr(Z),
		Nx, Ny, Nz,
		float32(dx), float32(dy), float32(dz),
		mesh.PBC_code(),
		cfg)
}
