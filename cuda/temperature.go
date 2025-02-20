package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Set Bth to thermal noise (Brown).
// see temperature.cu
func SetTemperature(Bth, noise *data.Slice, k2mu0_Mu0VgammaDt float64, Msat, Temp, Alpha MSlice) {
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	k_settemperature2_async(Bth.DevPtr(0), noise.DevPtr(0), float32(k2mu0_Mu0VgammaDt),
		Msat.DevPtr(0), Msat.Mul(0),
		Temp.DevPtr(0), Temp.Mul(0),
		Alpha.DevPtr(0), Alpha.Mul(0),
		N, cfg)
}

func SetTemperatureElastic(Fth, noise *data.Slice, eta, Temp MSlice, deltaT, cellVolume float32) {
	util.Argument(Fth.NComp() == 1 && noise.NComp() == 1)

	N := Fth.Len()
	cfg := make1DConf(N)

	k_settemperature_elastic_async(Fth.DevPtr(0), noise.DevPtr(0),
		eta.DevPtr(0), eta.Mul(0),
		Temp.DevPtr(0), Temp.Mul(0),
		deltaT, cellVolume,
		N, cfg)
}
