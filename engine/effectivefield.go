package engine

// Effective field

import "github.com/mumax/3/data"

var B_eff = NewVectorField("B_eff", "T", "Effective field", SetEffectiveField)

// Sets dst to the current effective field, in Tesla.
// This is the sum of all effective field terms,
// like demag, exchange, ...
func SetEffectiveField(dst *data.Slice) {
	SetDemagField(dst)    // set to B_demag...
	AddExchangeField(dst) // ...then add other terms
	AddAnisotropyField(dst)
	AddMagnetoelasticField(dst)
	B_ext.AddTo(dst)
	if !relaxing {
		B_therm.AddTo(dst)
	}
	AddCustomField(dst)
}

func SetEffectiveFieldRegion(dst, m, u *data.Slice, useFullSample bool, pbcX, pbcY, pbcZ int) {
	SetDemagFieldRegion(dst, m, useFullSample, pbcX, pbcY, pbcZ) // set to B_demag...
	AddExchangeFieldRegion(dst, m, useFullSample)                // ...then add other terms
	AddAnisotropyFieldRegion(dst, m, useFullSample)
	AddMagnetoelasticFieldRegion(dst, m, u, pbcX, pbcY, pbcZ, useFullSample)
	B_ext.AddToRegion(dst)
	if !relaxing {
		B_therm.AddToRegion(dst)
	}
	AddCustomFieldRegion(dst)
}
