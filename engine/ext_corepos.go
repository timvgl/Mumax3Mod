package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
	"fmt"
)

var (
	evaluteCorePosGPU 					bool = false
	devMode								bool = false
	memoryAllocated 					bool = false
	vals 								*data.Slice
	idx 								*data.Slice
	blk_vals 							*data.Slice
	blk_idxs 							*data.Slice
	CorePosRT					 			 = make([]float64, 3) //core Position of Vortex
	CorePosRTR							float64
	postStepModeVortexCore				bool = false
	ActivateCorePosScriptAccessState 	bool = false
	//elasticCorePos = NewVectorValue("ext_elastic_corepos", "m", "Vortex core position (x,y) + polarization (z)", elasticCorePos)
)
func init() {
	NewVectorValue("ext_corepos", "m", "Vortex core position (x,y) + polarization (z)", corePosTable)
	DeclVar("evaluteCorePosGPU", &evaluteCorePosGPU, "")
	DeclVar("__devMode__", &devMode, "")
	NewScalarValue("CorePosR", "m", "Radius of CorePosRT", func() float64 { 
																			corePosTable()
																			return math.Sqrt(math.Pow(CorePosRT[0], 2) + math.Pow(CorePosRT[1], 2))
																		})
	DeclFunc("Activate_corePosScriptAccess", Activate_corePosScriptAccess, "Activates the availability of the vortex core position as live data")
	DeclFunc("Activate_corePosScriptAccessR", Activate_corePosScriptAccessR, "")
	DeclVar("CorePosRT", &CorePosRT, "Vortex core position in real time")
	DeclVar("CorePosRTR", &CorePosRTR, "Radius of CorePosRT")


}

func allocateMemoryForMax() {
	if memoryAllocated == false {
		vals = cuda.NewSlice(1, [3]int{int(cu.MAX_BLOCK_DIM_X), 1, 1})
		cuda.Zero(vals)
		idx = cuda.NewSliceInt(1, [3]int{int(cu.MAX_BLOCK_DIM_X)*3 +2, 1, 1})
		cuda.Zero(idx)
		blk_vals = cuda.NewSlice(1, [3]int{int(cu.MAX_GRID_DIM_X), 1, 1})
		cuda.Zero(blk_vals)
		blk_idxs = cuda.NewSliceInt(1, [3]int{int(cu.MAX_GRID_DIM_X)*3 +2, 1, 1})
		cuda.Zero(blk_idxs)
		memoryAllocated = true
	}
}

func corePosGPU()[]float64 {
	util.AssertMsg(!devMode, "This code is not working. Please don't calculate the core position on the GPU.")
	fmt.Println("Getting pos on GPU")
	allocateMemoryForMax()
	var m = M.Buffer()
	var idx, max, max_n1X, max_p1X, max_n1Y, max_p1Y = cuda.MaxvecCellZComp(m, vals, idx, blk_vals, blk_idxs, M.Mesh())

	pos := make([]float64, 3)
	var maxX = idx[0]
	var maxY = idx[1]

	s := M.Mesh().Size()
	Nx, Ny := s[X], s[Y]

	// sub-cell interpolation in X and Y, but not Z
	pos[X] = float64(maxX) + interpolate_maxpos(
		max, -1, abs(max_n1X), 1, abs(max_p1X)) -
		float64(Nx)/2 + 0.5
	pos[Y] = float64(maxY) + interpolate_maxpos(
		max, -1, abs(max_n1Y), 1, abs(max_p1Y)) -
		float64(Ny)/2 + 0.5

	c := Mesh().CellSize()
	pos[X] *= c[X]
	pos[Y] *= c[Y]
	pos[Z] = float64(max) // 3rd coordinate is core polarization

	pos[X] += GetShiftPos() // add simulation window shift
	CorePosRT = pos
	fmt.Println(pos)
	return pos
}

func corePosCPU()[]float64 { 
	m := ValueOf(M_full)
	defer cuda.Recycle(m)

	m_z := m.Comp(Z).HostCopy().Scalars()
	s := m.Size()
	Nx, Ny, Nz := s[X], s[Y], s[Z]

	max := float32(math.Inf(-1))
	var maxX, maxY, maxZ int

	for z := 0; z < Nz; z++ {
		// Avoid the boundaries so the neighbor interpolation can't go out of bounds.
		for y := 1; y < Ny-1; y++ {
			for x := 1; x < Nx-1; x++ {
				m := abs(m_z[z][y][x])
				if m > max {
					maxX, maxY, maxZ = x, y, z
					max = m
				}
			}
		}
	}

	pos := make([]float64, 3)
	mz := m_z[maxZ]

	// sub-cell interpolation in X and Y, but not Z
	pos[X] = float64(maxX) + interpolate_maxpos(
		max, -1, abs(mz[maxY][maxX-1]), 1, abs(mz[maxY][maxX+1])) -
		float64(Nx)/2 + 0.5
	pos[Y] = float64(maxY) + interpolate_maxpos(
		max, -1, abs(mz[maxY-1][maxX]), 1, abs(mz[maxY+1][maxX])) -
		float64(Ny)/2 + 0.5

	c := Mesh().CellSize()
	pos[X] *= c[X]
	pos[Y] *= c[Y]
	msat, rM := Msat.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	pos[Z] = float64(m_z[maxZ][maxY][maxX]) / float64(msat.HostCopy().Scalars()[maxZ][maxY][maxX]) // 3rd coordinate is core polarization

	pos[X] += GetShiftPos() // add simulation window shift
	CorePosRT = pos
	return pos
}

func corePosTable()[]float64 {
	if evaluteCorePosGPU == true && postStepModeVortexCore == false {
		Activate_corePosScriptAccess()
		CorePosRT = corePosCPU()
	} else if evaluteCorePosGPU == true && postStepModeVortexCore == true {
	} else if evaluteCorePosGPU == false && postStepModeVortexCore == false {
		CorePosRT = corePosCPU()
	}
	return CorePosRT
}

func corePos()[]float64 {
	if evaluteCorePosGPU == true {
		CorePosRT = corePosGPU()
	} else {
		CorePosRT = corePosCPU()
	} 
	return CorePosRT
}

func interpolate_maxpos(f0, d1, f1, d2, f2 float32) float64 {
	b := (f2 - f1) / (d2 - d1)
	a := ((f2-f0)/d2 - (f0-f1)/(-d1)) / (d2 - d1)
	return float64(-b / (2 * a))
}

func abs(x float32) float32 {
	if x > 0 {
		return x
	} else {
		return -x
	}
}

func Activate_corePosScriptAccessR() {
	if ActivateCorePosScriptAccessState == false {
		Activate_corePosScriptAccess()
	}
	PostStep(func() {
		CorePosRTR = math.Sqrt(math.Pow(CorePosRT[0], 2) + math.Pow(CorePosRT[1], 2))
	})
}

func Activate_corePosScriptAccess() {
	postStepModeVortexCore = true
	ActivateCorePosScriptAccessState = true
	PostStep(func() {CorePosRT = corePos()})
}

