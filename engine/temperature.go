package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"

	//"github.com/mumax/3/util"
	"math"
)

var (
	Temp           = NewScalarParam("Temp", "K", "Temperature")
	E_therm        = NewScalarValue("E_therm", "J", "Thermal energy", GetThermalEnergy)
	Edens_therm    = NewScalarField("Edens_therm", "J/m3", "Thermal energy density", AddThermalEnergyDensity)
	B_therm        thermField // Thermal effective field (T)
	F_therm        thermForce
	useTempElastic = false
)

var AddThermalEnergyDensity = makeEdensAdder(&B_therm, -1)
var AddThermalElasticEnergyDensity = makeEdensAdder(&F_therm, -1)

// thermField calculates and caches thermal noise.
type thermField struct {
	seed      int64            // seed for generator
	generator curand.Generator //
	noise     *data.Slice      // noise buffer
	step      int              // solver step corresponding to noise
	dt        float64          // solver timestep corresponding to noise
}

type thermForce struct {
	seed      int64            // seed for generator
	generator curand.Generator //
	noise     *data.Slice      // noise buffer
	step      int              // solver step corresponding to noise
	dt        float64          // solver timestep corresponding to noise
}

func init() {
	DeclFunc("ThermSeed", ThermSeed, "Set a random seed for thermal noise")
	registerEnergy(GetThermalEnergy, AddThermalEnergyDensity)
	B_therm.step = -1 // invalidate noise cache
	F_therm.step = -1
	DeclROnly("B_therm", &B_therm, "Thermal field (T)")
	DeclROnly("F_therm", &F_therm, "Thermal forcedensity (F/m3)")
	DeclVar("useTempElastic", &useTempElastic, "")
}

func (b *thermField) AddTo(dst *data.Slice) {
	if !Temp.isZero() {
		b.update()
		cuda.Add(dst, dst, b.noise)
	}
}

func (b *thermField) AddToRegion(dst *data.Slice) {
	if !Temp.isZero() {
		b.update()
		noiseRed := cuda.Buffer(dst.NComp(), dst.RegionSize())
		cuda.Crop(noiseRed, b.noise, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Add(dst, dst, noiseRed)
		cuda.Recycle(noiseRed)
	}
}

func (b *thermForce) AddTo(dst *data.Slice) {
	if !Temp.isZero() {
		b.update()
		cuda.Add(dst, dst, b.noise)
	}
}

func (b *thermField) update() {
	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.
	if FixDt != 0 {
		Dt_si = FixDt
	}

	if b.generator == 0 {
		b.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		b.generator.SetSeed(b.seed)
	}
	if b.noise == nil {
		b.noise = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		// when noise was (re-)allocated it's invalid for sure.
		B_therm.step = -1
		B_therm.dt = -1
	}

	if Temp.isZero() {
		cuda.Memset(b.noise, 0, 0, 0)
		b.step = NSteps
		b.dt = Dt_si
		return
	}

	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt {
		return
	}

	// after a bad step the timestep is rescaled and the noise should be rescaled accordingly, instead of redrawing the random numbers
	if NSteps == b.step && Dt_si != b.dt {
		for c := 0; c < 3; c++ {
			cuda.Madd2(b.noise.Comp(c), b.noise.Comp(c), b.noise.Comp(c), float32(math.Sqrt(b.dt/Dt_si)), 0.)
		}
		b.dt = Dt_si
		return
	}

	if FixDt == 0 {
		Refer("leliaert2017")
		//uncomment to not allow adaptive step
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * cellVolume() * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	ms := Msat.MSlice()
	defer ms.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt, ms, temp, alpha)
	}

	b.step = NSteps
	b.dt = Dt_si
}

func (b *thermForce) update() {

	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.

	if b.generator == 0 {
		b.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		b.generator.SetSeed(b.seed)
	}
	if b.noise == nil {
		b.noise = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		// when noise was (re-)allocated it's invalid for sure.
		B_therm.step = -1
		B_therm.dt = -1
	}

	if Temp.isZero() {
		cuda.Memset(b.noise, 0, 0, 0)
		b.step = NSteps
		b.dt = Dt_si
		return
	}

	if !useTempElastic {
		return
	}

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt {
		return
	}

	// after a bad step the timestep is rescaled and the noise should be rescaled accordingly, instead of redrawing the random numbers
	if NSteps == b.step && Dt_si != b.dt {
		for c := 0; c < 3; c++ {
			cuda.Madd2(b.noise.Comp(c), b.noise.Comp(c), b.noise.Comp(c), float32(math.Sqrt(b.dt/Dt_si)), 0.)
		}
		b.dt = Dt_si
		return
	}

	if FixDt == 0 {
		Refer("leliaert2017")
		//uncomment to not allow adaptive step
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	eta := Eta.MSlice()
	defer eta.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.SetTemperatureElastic(dst.Comp(i), noise, eta, temp, float32(Dt_si), float32(cellVolume()))
	}

	b.step = NSteps
	b.dt = Dt_si
}

func GetThermalEnergy() float64 {
	if Temp.isZero() || relaxing {
		return 0
	} else {
		return -cellVolume() * dot(&M_full, &B_therm)
	}
}

func GetThermalEnergyPower() float64 {
	if Temp.isZero() || relaxing {
		return 0
	} else {
		buf := cuda.Buffer(F_therm.noise.NComp(), F_therm.noise.Size())
		F_therm.EvalTo(buf)
		return dot(&DU, &F_therm) * cellVolume()
	}
}

// Seeds the thermal noise generator
func ThermSeed(seed int) {
	B_therm.seed = int64(seed)
	if B_therm.generator != 0 {
		B_therm.generator.SetSeed(B_therm.seed)
	}
	F_therm.seed = int64(seed)
	if F_therm.generator != 0 {
		F_therm.generator.SetSeed(F_therm.seed)
	}
}

func (b *thermField) Mesh() *data.Mesh       { return Mesh() }
func (b *thermField) NComp() int             { return 3 }
func (b *thermField) Name() string           { return "Thermal field" }
func (b *thermField) Unit() string           { return "T" }
func (b *thermField) average() []float64     { return qAverageUniverse(b) }
func (b *thermField) EvalTo(dst *data.Slice) { EvalTo(b, dst) }
func (b *thermField) Slice() (*data.Slice, bool) {
	b.update()
	return b.noise, false
}

func (b *thermForce) Mesh() *data.Mesh       { return Mesh() }
func (b *thermForce) NComp() int             { return 3 }
func (b *thermForce) Name() string           { return "Thermal force" }
func (b *thermForce) Unit() string           { return "N/m3" }
func (b *thermForce) average() []float64     { return qAverageUniverse(b) }
func (b *thermForce) EvalTo(dst *data.Slice) { EvalTo(b, dst) }
func (b *thermForce) AddToRegion(dst *data.Slice) {
	if !Temp.isZero() {
		b.update()
		noiseRed := cuda.Buffer(dst.NComp(), dst.RegionSize())
		cuda.Crop(noiseRed, b.noise, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Add(dst, dst, noiseRed)
		cuda.Recycle(noiseRed)
	}
}
func (b *thermForce) Slice() (*data.Slice, bool) {
	b.update()
	return b.noise, false
}
