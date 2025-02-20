package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	Nx              int
	Ny              int
	Nz              int
	Dx              float64
	Dy              float64
	Dz              float64
	Tx              float64
	Ty              float64
	Tz              float64
	PBCx            int
	PBCy            int
	PBCz            int
	AutoMeshx       bool
	AutoMeshy       bool
	AutoMeshz       bool
	gotMeshDefValue bool = false
)

var globalmesh_ data.Mesh // mesh for m and everything that has the same size

func init() {
	DeclFunc("SetGridSize", SetGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("SetCellSize", SetCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclFunc("SetMesh", SetMesh, `Sets GridSize, CellSize and PBC at the same time`)
	DeclFunc("SetPBC", SetPBC, "Sets the number of repetitions in X,Y,Z to create periodic boundary "+
		"conditions. The number of repetitions determines the cutoff range for the demagnetization.")
	DeclVar("Nx", &Nx, "")
	DeclVar("Ny", &Ny, "")
	DeclVar("Nz", &Nz, "")
	DeclVar("Tx", &Tx, "")
	DeclVar("Ty", &Ty, "")
	DeclVar("Tz", &Tz, "")
	DeclVar("Dx", &Dx, "")
	DeclVar("Dy", &Dy, "")
	DeclVar("Dz", &Dz, "")
}

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh_
}

func arg(msg string, test bool) {
	if !test {
		panic(UserErr(msg + ": illegal arugment"))
	}
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
// TODO: dedup arguments from globals
func SetMesh(NxTmp, NyTmp, NzTmp int, cellSizeX, cellSizeY, cellSizeZ float64, pbcx, pbcy, pbcz int) {
	Nx = NxTmp
	Ny = NyTmp
	Nz = NzTmp
	Dx = cellSizeX
	Dy = cellSizeY
	Dz = cellSizeZ
	Tx = float64(Nx) * Dx
	Ty = float64(Ny) * Dy
	Tz = float64(Nz) * Dz
	PBCx = pbcx
	PBCy = pbcy
	PBCz = pbcz
	SetBusy(true)
	defer SetBusy(false)

	arg("GridSize", Nx > 0 && Ny > 0 && Nz > 0)
	arg("CellSize", cellSizeX > 0 && cellSizeY > 0 && cellSizeZ > 0)
	arg("PBC", pbcx >= 0 && pbcy >= 0 && pbcz >= 0)

	prevSize := globalmesh_.Size()
	pbc := []int{pbcx, pbcy, pbcz}

	if globalmesh_.Size() == [3]int{0, 0, 0} {
		// first time mesh is set
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.alloc()
		U.alloc()
		DU.alloc()
		regions.alloc()
		//UOVERLAY.alloc()
	} else {
		// here be dragons
		LogOut("resizing...")

		// free everything
		conv_.Free()
		conv_ = nil
		mfmconv_.Free()
		mfmconv_ = nil
		cuda.FreeBuffers()

		// resize everything
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.resize()
		U.resize()
		DU.resize()
		regions.resize()
		//UOVERLAY.resize()
		Geometry.Buffer.Free()
		Geometry.Buffer = data.NilSlice(1, Mesh().Size())
		Geometry.setGeom(Geometry.shape)

		// remove excitation extra terms if they don't fit anymore
		// up to the user to add them again
		if Mesh().Size() != prevSize {
			B_ext.RemoveExtraTerms()
			J.RemoveExtraTerms()
		}

		if Mesh().Size() != prevSize {
			B_therm.noise.Free()
			B_therm.noise = nil
		}
	}
	lazy_gridsize = []int{Nx, Ny, Nz}
	lazy_cellsize = []float64{cellSizeX, cellSizeY, cellSizeZ}
	lazy_pbc = []int{pbcx, pbcy, pbcz}
}

func printf(f float64) float32 {
	return float32(f)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	lazy_gridsize []int
	lazy_cellsize []float64
	lazy_pbc      = []int{0, 0, 0}
)

func SetGridSize(NxTmp, NyTmp, NzTmp int) {
	Nx = NxTmp
	Ny = NyTmp
	Nz = NzTmp
	if gotMeshDefValue == true {
		Tx = Dx * float64(Nx)
		Ty = Dy * float64(Ny)
		Tz = Dz * float64(Nz)
	} else {
		gotMeshDefValue = true
	}
	lazy_gridsize = []int{Nx, Ny, Nz}
	if lazy_cellsize != nil {
		SetMesh(Nx, Ny, Nz, lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z], lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func SetCellSize(cx, cy, cz float64) {
	Dx = cx
	Dy = cy
	Dz = cz
	if gotMeshDefValue == true {
		Tx = Dx * float64(Nx)
		Ty = Dy * float64(Ny)
		Tz = Dz * float64(Nz)
	} else {
		gotMeshDefValue = true
	}
	lazy_cellsize = []float64{cx, cy, cz}
	if lazy_gridsize != nil {
		SetMesh(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z], cx, cy, cz, lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func SetPBC(nx, ny, nz int) {
	PBCx = nx
	PBCy = ny
	PBCz = nz
	lazy_pbc = []int{nx, ny, nz}
	if lazy_gridsize != nil && lazy_cellsize != nil {
		SetMesh(lazy_gridsize[X], lazy_gridsize[Y], lazy_gridsize[Z],
			lazy_cellsize[X], lazy_cellsize[Y], lazy_cellsize[Z],
			lazy_pbc[X], lazy_pbc[Y], lazy_pbc[Z])
	}
}

func generateNDT(N int, D, T float64) (int, float64, float64) {
	if N != 0 && D != 0 && T == 0 {
		T = float64(N) * D
	} else if N != 0 && D == 0 && T != 0 {
		D = T / float64(N)
	} else if N == 0 && D != 0 && T != 0 {
		N = int(T / D)
	}
	return N, D, T
}

// check if mesh is set
func checkMesh() {
	if globalmesh_.Size() == [3]int{0, 0, 0} {
		if Nx != 0 && Dx != 0 || Nx != 0 && Tx != 0 || Dx != 0 && Tx != 0 {
			Nx, Dx, Tx = generateNDT(Nx, Dx, Tx)
		} else {
			panic("need to set mesh first")
		}
		if Ny != 0 && Dy != 0 || Ny != 0 && Ty != 0 || Dy != 0 && Ty != 0 {
			Ny, Dy, Ty = generateNDT(Ny, Dy, Ty)
		} else {
			panic("need to set mesh first")
		}
		if Nz != 0 && Dz != 0 || Nz != 0 && Tz != 0 || Dz != 0 && Tz != 0 {
			Nz, Dz, Tz = generateNDT(Nz, Dz, Tz)
		} else {
			panic("need to set mesh first")
		}
		SetMesh(Nx, Ny, Nz, Dx, Dy, Dz, PBCx, PBCy, PBCz)
		//panic("need to set mesh first")
	}
}
