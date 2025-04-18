package engine

import (
	"reflect"

	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha                            = NewScalarParam("alpha", "", "Landau-Lifshitz damping constant")
	Xi                               = NewScalarParam("xi", "", "Non-adiabaticity of spin-transfer-torque")
	Pol                              = NewScalarParam("Pol", "", "Electrical current polarization")
	Lambda                           = NewScalarParam("Lambda", "", "Slonczewski Λ parameter")
	EpsilonPrime                     = NewScalarParam("EpsilonPrime", "", "Slonczewski secondairy STT term ε'")
	FrozenSpins                      = NewScalarParam("frozenspins", "", "Defines spins that should be fixed") // 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values
	FreeLayerThickness               = NewScalarParam("FreeLayerThickness", "m", "Slonczewski free layer thickness (if set to zero (default), then the thickness will be deduced from the mesh size)")
	FixedLayer                       = NewExcitation("FixedLayer", "", "Slonczewski fixed layer polarization")
	Torque                           = NewVectorField("torque", "T", "Total torque/γ0", SetTorque)
	LLTorque                         = NewVectorField("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque                         = NewVectorField("STTorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	J                                = NewExcitation("J", "A/m2", "Electrical current density")
	MaxTorque                        = NewScalarValue("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)
	GammaLL                  float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                          = true
	DisableZhangLiTorque             = false
	DisableSlonczewskiTorque         = false
	fixedLayerPosition               = FIXEDLAYER_TOP // instructs mumax3 how free and fixed layers are stacked along +z direction
)

func init() {
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	DeclLValue("FixedLayerPosition", &flposition{}, "Position of the fixed layer: FIXEDLAYER_TOP, FIXEDLAYER_BOTTOM (default=FIXEDLAYER_TOP)")
	DeclROnly("FIXEDLAYER_TOP", FIXEDLAYER_TOP, "FixedLayerPosition = FIXEDLAYER_TOP instructs mumax3 that fixed layer is on top of the free layer")
	DeclROnly("FIXEDLAYER_BOTTOM", FIXEDLAYER_BOTTOM, "FixedLayerPosition = FIXEDLAYER_BOTTOM instructs mumax3 that fixed layer is underneath of the free layer")
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

func SetTorqueRegion(dst, m, u *data.Slice, useFullSample bool, pbcX, pbcY, pbcZ int) {
	if useFullSample {
		backupM := cuda.Buffer(M.Buffer().NComp(), M.Buffer().Size())
		data.Copy(backupM, M.Buffer())
		data.CopyPart(M.Buffer(), m, 0, m.Size()[X], 0, m.Size()[Y], 0, m.Size()[Z], 0, 1, dst.StartX, dst.StartY, dst.StartZ, 0)
		defer func() {
			data.Copy(M.Buffer(), backupM)
			cuda.Recycle(backupM)
		}()
	}
	SetLLTorqueRegion(dst, m, u, useFullSample, pbcX, pbcY, pbcZ)
	AddSTTorqueRegion(dst, m, useFullSample)
	FreezeSpinsRegion(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

func SetLLTorqueRegion(dst, m, u *data.Slice, useFullSample bool, pbcX, pbcY, pbcZ int) {
	SetEffectiveFieldRegion(dst, m, u, useFullSample, pbcX, pbcY, pbcZ) // calc and store B_eff
	alpha := Alpha.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst, m, dst, alpha) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, m, dst)
	}
}

// Adds the current spin transfer torque to dst
func AddSTTorque(dst *data.Slice) {
	if J.isZero() {
		return
	}
	util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	if !DisableZhangLiTorque {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		xi := Xi.MSlice()
		defer xi.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		cuda.AddZhangLiTorque(dst, M.Buffer(), msat, j, alpha, xi, pol, Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		fixedP := FixedLayer.MSlice()
		defer fixedP.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		lambda := Lambda.MSlice()
		defer lambda.Recycle()
		epsPrime := EpsilonPrime.MSlice()
		defer epsPrime.Recycle()
		thickness := FreeLayerThickness.MSlice()
		defer thickness.Recycle()
		cuda.AddSlonczewskiTorque2(dst, M.Buffer(),
			msat, j, fixedP, alpha, pol, lambda, epsPrime,
			thickness,
			CurrentSignFromFixedLayerPosition[fixedLayerPosition],
			Mesh())
	}
}

func AddSTTorqueRegion(dst, m *data.Slice, useFullSample bool) {
	if !useFullSample {
		if J.isZero() {
			return
		}
		util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
		jspin, rec := J.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		if rec {
			defer cuda.Recycle(jspin)
		}
		fl, rec := FixedLayer.SliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
		if rec {
			defer cuda.Recycle(fl)
		}
		if !DisableZhangLiTorque {
			msat := Msat.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer msat.Recycle()
			j := J.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer j.Recycle()
			alpha := Alpha.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer alpha.Recycle()
			xi := Xi.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer xi.Recycle()
			pol := Pol.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer pol.Recycle()
			cuda.AddZhangLiTorque(dst, m, msat, j, alpha, xi, pol, Crop(&M.varVectorField, dst.StartX, dst.EndX, dst.StartY, dst.EndY, dst.StartZ, dst.EndZ).Mesh())
			cuda.Recycle(m)
		}
		if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
			msat := Msat.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer msat.Recycle()
			j := J.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer j.Recycle()
			fixedP := FixedLayer.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer fixedP.Recycle()
			alpha := Alpha.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer alpha.Recycle()
			pol := Pol.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer pol.Recycle()
			lambda := Lambda.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer lambda.Recycle()
			epsPrime := EpsilonPrime.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer epsPrime.Recycle()
			thickness := FreeLayerThickness.MSliceRegion(dst.RegionSize(), dst.StartX, dst.StartY, dst.StartZ)
			defer thickness.Recycle()
			cuda.AddSlonczewskiTorque2(dst, m,
				msat, j, fixedP, alpha, pol, lambda, epsPrime,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Crop(&M.varVectorField, dst.StartX, dst.EndX, dst.StartY, dst.EndY, dst.StartZ, dst.EndZ).Mesh())
			cuda.Recycle(m)
		}
	} else {
		B := cuda.Buffer(M.NComp(), M.Buffer().Size())
		cuda.Zero(B)
		AddSTTorque(B)
		BCropped := cuda.Buffer(dst.NComp(), dst.Size())
		cuda.Crop(BCropped, B, dst.StartX, dst.StartY, dst.StartZ)
		cuda.Add(dst, dst, BCropped)
		cuda.Recycle(BCropped)
		cuda.Recycle(B)
	}
}

func FreezeSpins(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		cuda.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
	}
}

func FreezeSpinsRegion(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		cuda.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
	}
}

func GetMaxTorque() float64 {
	torque := ValueOf(Torque)
	defer cuda.Recycle(torque)
	return cuda.MaxVecNorm(torque)
}

func GetAverageTorque() float32 {
	torque := ValueOf(Torque)
	defer cuda.Recycle(torque)
	avergeTorque := sAverageMagnet(torque)
	return float32(math.Sqrt(math.Pow(avergeTorque[0], 2) + math.Pow(avergeTorque[1], 2) + math.Pow(avergeTorque[2], 2)))
}

type FixedLayerPosition int

const (
	FIXEDLAYER_TOP FixedLayerPosition = iota + 1
	FIXEDLAYER_BOTTOM
)

var (
	CurrentSignFromFixedLayerPosition = map[FixedLayerPosition]float64{
		FIXEDLAYER_TOP:    1.0,
		FIXEDLAYER_BOTTOM: -1.0,
	}
)

type flposition struct{}

func (*flposition) Eval() interface{} { return fixedLayerPosition }
func (*flposition) SetValue(v interface{}) {
	drainOutput()
	fixedLayerPosition = v.(FixedLayerPosition)
}
func (*flposition) Type() reflect.Type { return reflect.TypeOf(FixedLayerPosition(FIXEDLAYER_TOP)) }
