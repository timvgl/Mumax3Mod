package engine

import (
	"sort"
	"sync"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	regions                 = Regions{info: info{1, "Regions", ""}} // global Regions map
	limitRenderX            = RegionRange1D{from: 0, to: -1}
	limitRenderY            = RegionRange1D{from: 0, to: -1}
	limitRenderZ            = RegionRange1D{from: 0, to: -1}
	homogeniousRegionZ bool = false
	eraseAllRegions    bool = false
)

const NREGION = 256 // maximum number of Regions, limited by size of byte.

func init() {
	DeclFunc("DefRegion", DefRegion, "Define a material region with given index (0-255) and shape")
	DeclROnly("Regions", &regions, "Outputs the region index for each cell")
	DeclFunc("DefRegionCell", DefRegionCell, "Set a material region (first argument) in one cell "+
		"by the index of the cell (last three arguments)")
	DeclFunc("ReDefRegion", ReDefRegion, "")
	DeclFunc("LimitRenderRegionX", LimitRenderRegionX, "")
	DeclFunc("LimitRenderRegionY", LimitRenderRegionY, "")
	DeclFunc("LimitRenderRegionZ", LimitRenderRegionZ, "")
	DeclVar("homogeniousRegionZ", &homogeniousRegionZ, "")
	DeclFunc("EraseAllRegions", DeleteAllRegions, "")

}

// stores the region index for each cell
type Regions struct {
	gpuCache *cuda.Bytes                 // TODO: rename: buffer
	hist     []func(x, y, z float64) int // history of region set operations
	indices  []int
	info
}

type RegionRange1D struct {
	from int
	to   int
}

type Pair struct {
	Value  int
	Object func(x, y, z float64) int // Replace interface{} with the actual type of your objects
}

func (r *Regions) alloc() {
	mesh := r.Mesh()
	r.gpuCache = cuda.NewBytes(mesh.NCell())
	DefRegion(0, universe)
}

func (r *Regions) resize() {
	newSize := Mesh().Size()
	r.gpuCache.Free()
	r.gpuCache = cuda.NewBytes(prod(newSize))
	for _, f := range r.hist {
		r.render(f)
	}
}
func LimitRenderRegionX(start, end int) {
	if start < 0 || end < -1 {
		panic("Start or end out of sample. Please provide valid cell index.")
	}
	limitRenderX = RegionRange1D{from: start, to: end}
}

func LimitRenderRegionY(start, end int) {
	if start < 0 || end < -1 {
		panic("Start or end out of sample. Please provide valid cell index.")
	}
	limitRenderY = RegionRange1D{from: start, to: end}
}

func LimitRenderRegionZ(start, end int) {
	if start < 0 || end < -1 {
		panic("Start or end out of sample. Please provide valid cell index.")
	}
	limitRenderZ = RegionRange1D{from: start, to: end}
}

func GetExistingIndices() []int {
	pairs := make([]Pair, len(regions.indices))
	for i := range regions.indices {
		pairs[i] = Pair{Value: regions.indices[i], Object: regions.hist[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Value < pairs[j].Value
	})
	for i := range pairs {
		regions.indices[i] = pairs[i].Value
		regions.hist[i] = pairs[i].Object
	}
	return regions.indices
}

// Define a region with id (0-255) to be inside the Shape.
func DefRegion(id int, s Shape) {
	defRegionId(id)
	f := func(x, y, z float64) int {
		if s(x, y, z) {
			return id
		} else {
			return -1
		}
	}
	regions.render(f)
	regions.hist = append(regions.hist, f)
	regions.indices = append(regions.indices, id)
}

func ReDefRegion(startId, endId int) {
	// Checks validity of input region IDs
	defRegionId(startId)
	defRegionId(endId)

	hist_len := len(regions.hist) // Only consider hist before this Redef to avoid recursion
	f := func(x, y, z float64) int {
		value := -1
		for i := hist_len - 1; i >= 0; i-- {
			f_other := regions.hist[i]
			region := f_other(x, y, z)
			if region >= 0 {
				value = region
				break
			}
		}
		if value == startId {
			return endId
		} else {
			return value
		}
	}
	regions.redefine(startId, endId)

	deleteIndex := -1
	for i, v := range regions.indices {
		if v == startId {
			deleteIndex = i
			break
		}
	}
	if deleteIndex != -1 {
		regions.indices = append(regions.indices[:deleteIndex], regions.indices[deleteIndex+1:]...)
		regions.hist = append(regions.hist[:deleteIndex], regions.hist[deleteIndex+1:]...)
	} else {
		panic("Could not find region index of first index.")
	}
	deleteIndex = -1
	for i, v := range regions.indices {
		if v == endId {
			deleteIndex = i
			break
		}
	}
	if deleteIndex != -1 {
		regions.hist[deleteIndex] = f
	} else {
		panic("Could not find region index of second index.")
	}
	//Regions.hist = append(Regions.hist, f)
}

func DeleteAllRegions() {
	regions.hist = make([]func(x, y, z float64) int, 0)
	regions.indices = make([]int, 0)
	eraseAllRegions = true
	regions.redefine(0, 0)
	eraseAllRegions = false
}

func (r *Regions) redefine(startId, endId int) {
	// Loop through all cells, if their region ID matches startId, change it to endId
	n := Mesh().Size()
	l := r.RegionListCPU() // need to start from previous state
	arr := reshapeBytes(l, r.Mesh().Size())

	var (
		xStart = limitRenderX.from
		yStart = limitRenderY.from
		zStart = limitRenderZ.from
	)
	if limitRenderX.to == -1 || limitRenderX.to > n[X] {
		limitRenderX.to = n[X]
	}
	if limitRenderY.to == -1 || limitRenderY.to > n[Y] {
		limitRenderY.to = n[Y]
	}
	if limitRenderZ.to == -1 || limitRenderZ.to > n[Z] {
		limitRenderZ.to = n[Z]
	}

	var (
		xEnd = limitRenderX.to
		yEnd = limitRenderY.to
		zEnd = limitRenderZ.to
	)

	wg := sync.WaitGroup{}
	for iz := zStart; iz < zEnd; iz++ {
		if zEnd-zStart == 1 {
			for iy := yStart; iy < yEnd; iy++ {
				wg.Add(1)
				go func(iz, iy int) {
					for ix := xStart; ix < xEnd; ix++ {
						if arr[iz][iy][ix] == byte(startId) || eraseAllRegions && arr[iz][iy][ix] != byte(startId) {
							arr[iz][iy][ix] = byte(endId)
						}
					}
					wg.Done()
				}(iz, iy)
			}
		} else {
			wg.Add(1)
			go func(iz int) {
				for iy := yStart; iy < yEnd; iy++ {
					wg.Add(1)
					go func(iz, iy int) {
						for ix := xStart; ix < xEnd; ix++ {
							if arr[iz][iy][ix] == byte(startId) || eraseAllRegions && arr[iz][iy][ix] != byte(startId) {
								arr[iz][iy][ix] = byte(endId)
							}
						}
						wg.Done()
					}(iz, iy)
				}
				wg.Done()
			}(iz)
		}
	}
	wg.Wait()

	r.gpuCache.Upload(l)
}

func (r *Regions) RegionListCPU() []byte {
	regionsList := make([]byte, r.Mesh().NCell())
	r.gpuCache.Download(regionsList)
	return regionsList
}

// renders (rasterizes) shape, filling it with region number #id, between x1 and x2
// TODO: a tidbit expensive
func (r *Regions) render(f func(x, y, z float64) int) {
	n := Mesh().Size()
	l := r.HostList() // need to start from previous state
	arr := reshapeBytes(l, r.Mesh().Size())

	var (
		xStart = limitRenderX.from
		yStart = limitRenderY.from
		zStart = limitRenderZ.from
	)
	if limitRenderX.to == -1 || limitRenderX.to > n[X] {
		limitRenderX.to = n[X]
	}
	if limitRenderY.to == -1 || limitRenderY.to > n[Y] {
		limitRenderY.to = n[Y]
	}
	if limitRenderZ.to == -1 || limitRenderZ.to > n[Z] {
		limitRenderZ.to = n[Z]
	}

	var (
		xEnd = limitRenderX.to
		yEnd = limitRenderY.to
		zEnd = limitRenderZ.to
	)

	wg := sync.WaitGroup{}
	for iz := zStart; iz < zEnd; iz++ {
		if zEnd-zStart == 1 || homogeniousRegionZ {
			for iy := yStart; iy < yEnd; iy++ {
				wg.Add(1)
				go func(iz, iy int) {
					for ix := xStart; ix < xEnd; ix++ {
						r := Index2Coord(ix, iy, iz)
						region := f(r[X], r[Y], r[Z])
						if region >= 0 {
							arr[iz][iy][ix] = byte(region)
						}
					}
					wg.Done()
				}(iz, iy)
			}
			break
		} else {
			wg.Add(1)
			go func(iz int) {
				for iy := yStart; iy < yEnd; iy++ {
					wg.Add(1)
					go func(iz, iy int) {
						for ix := xStart; ix < xEnd; ix++ {
							r := Index2Coord(ix, iy, iz)
							region := f(r[X], r[Y], r[Z])
							if region >= 0 {
								arr[iz][iy][ix] = byte(region)
							}
						}
						wg.Done()
					}(iz, iy)
				}
				wg.Done()
			}(iz)
		}
	}
	wg.Wait()
	if homogeniousRegionZ {
		for iz := zStart; iz < zEnd; iz++ {
			wg.Add(1)
			go func(iz int) {
				for iy := yStart; iy < yEnd; iy++ {
					copy(arr[iz][iy], arr[0][iy])
				}
				wg.Done()
			}(iz)
		}
	}

	//log.Print("Regions.upload")
	r.gpuCache.Upload(l)
}

// get the region for position R based on the history
func (r *Regions) get(R data.Vector) int {
	// reverse order, last one set wins.
	for i := len(r.hist) - 1; i >= 0; i-- {
		f := r.hist[i]
		region := f(R[X], R[Y], R[Z])
		if region >= 0 {
			return region
		}
	}
	return 0
}

func (r *Regions) HostArray() [][][]byte {
	return reshapeBytes(r.HostList(), r.Mesh().Size())
}

func (r *Regions) HostList() []byte {
	RegionsList := make([]byte, r.Mesh().NCell())
	regions.gpuCache.Download(RegionsList)
	return RegionsList
}

func DefRegionCell(id int, x, y, z int) {
	defRegionId(id)
	index := data.Index(Mesh().Size(), x, y, z)
	regions.gpuCache.Set(index, byte(id))
}

// Load Regions from ovf file, use first component.
// Regions should be between 0 and 256
func (r *Regions) LoadFile(fname string) {
	inSlice := LoadFileDSliceMyDir(fname)
	n := r.Mesh().Size()
	inSlice = data.Resample(inSlice, n)
	inArr := inSlice.Tensors()[0]
	l := r.HostList()
	arr := reshapeBytes(l, n)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				val := inArr[iz][iy][ix]
				if val < 0 || val > 256 {
					util.Fatal("Regions.LoadFile(", fname, "): all values should be between 0 & 256, have: ", val)
				}
				arr[iz][iy][ix] = byte(val)
			}
		}
	}
	r.gpuCache.Upload(l)
}

func (r *Regions) average() []float64 {
	s, recycle := r.Slice()
	if recycle {
		defer cuda.Recycle(s)
	}
	return sAverageUniverse(s)
}

func (r *Regions) Average() float64 { return r.average()[0] }

// Set the region of one cell
func (r *Regions) SetCell(ix, iy, iz int, region int) {
	size := Mesh().Size()
	i := data.Index(size, ix, iy, iz)
	r.gpuCache.Set(i, byte(region))
}

func (r *Regions) GetCell(ix, iy, iz int) int {
	size := Mesh().Size()
	i := data.Index(size, ix, iy, iz)
	return int(r.gpuCache.Get(i))
}

func defRegionId(id int) {
	if id < 0 || id > NREGION {
		util.Fatalf("region id should be 0 -%v, have: %v", NREGION, id)
	}
	checkMesh()
}

// normalized volume (0..1) of region.
// TODO: a tidbit too expensive
func (r *Regions) volume(region_ int) float64 {
	region := byte(region_)
	vol := 0
	list := r.HostList()
	for _, reg := range list {
		if reg == region {
			vol++
		}
	}
	V := float64(vol) / float64(r.Mesh().NCell())
	return V
}

// Get the region data on GPU
func (r *Regions) Gpu() *cuda.Bytes {
	return r.gpuCache
}

var unitMap regionwise // unit map used to output Regions quantity

func init() {
	unitMap.init(1, "unit", "", nil)
	for r := 0; r < NREGION; r++ {
		unitMap.setRegion(r, []float64{float64(r)})
	}
}

// Get returns the Regions as a slice of floats, so it can be output.
func (r *Regions) Slice() (*data.Slice, bool) {
	buf := cuda.Buffer(1, r.Mesh().Size())
	cuda.RegionDecode(buf, unitMap.gpuLUT1(), regions.Gpu())
	return buf, true
}

func (r *Regions) EvalTo(dst *data.Slice) { EvalTo(r, dst) }

var _ Quantity = &regions

// Re-interpret a contiguous array as a multi-dimensional array of given size.
func reshapeBytes(array []byte, size [3]int) [][][]byte {
	Nx, Ny, Nz := size[X], size[Y], size[Z]
	util.Argument(Nx*Ny*Nz == len(array))
	sliced := make([][][]byte, Nz)
	for i := range sliced {
		sliced[i] = make([][]byte, Ny)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = array[(i*Ny+j)*Nx+0 : (i*Ny+j)*Nx+Nx]
		}
	}
	return sliced
}

func (b *Regions) shift(dx int) {
	// TODO: return if no Regions defined
	r1 := b.Gpu()
	r2 := cuda.NewBytes(b.Mesh().NCell()) // TODO: somehow recycle
	defer r2.Free()
	newreg := byte(0) // new region at edge
	cuda.ShiftBytes(r2, r1, b.Mesh(), dx, newreg)
	r1.Copy(r2)

	n := Mesh().Size()
	x1, x2 := shiftDirtyRange(dx)

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := x1; ix < x2; ix++ {
				r := Index2Coord(ix, iy, iz) // includes shift
				reg := b.get(r)
				if reg != 0 {
					b.SetCell(ix, iy, iz, reg) // a bit slowish, but hardly reached
				}
			}
		}
	}
}

func (b *Regions) shiftY(dy int) {
	// TODO: return if no Regions defined
	r1 := b.Gpu()
	r2 := cuda.NewBytes(b.Mesh().NCell()) // TODO: somehow recycle
	defer r2.Free()
	newreg := byte(0) // new region at edge
	cuda.ShiftBytesY(r2, r1, b.Mesh(), dy, newreg)
	r1.Copy(r2)

	n := Mesh().Size()
	y1, y2 := shiftDirtyRange(dy)

	for iz := 0; iz < n[Z]; iz++ {
		for ix := 0; ix < n[X]; ix++ {
			for iy := y1; iy < y2; iy++ {
				r := Index2Coord(ix, iy, iz) // includes shift
				reg := b.get(r)
				if reg != 0 {
					b.SetCell(ix, iy, iz, reg) // a bit slowish, but hardly reached
				}
			}
		}
	}
}

func (r *Regions) Mesh() *data.Mesh { return Mesh() }

func prod(s [3]int) int {
	return s[0] * s[1] * s[2]
}
