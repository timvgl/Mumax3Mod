package engine

import (
	"fmt"
	"slices"

	"github.com/mumax/3/data"
	"github.com/mumax/3/logUI"
)

var (
	Categories          = []string{}
	DeclVarCalcDyn      = map[string][]Quantity{}
	DeclVarCalcDynAlias = map[string][]string{}
)

func init() {
	DeclFunc("Calc", Calc, "")
	DeclFunc("CalcAs", CalcAs, "")
}

func Calc(q Quantity, Category string) {
	if _, isFFT := q.(interface {
		FFTOutputSize() [3]int
	}); !isFFT {
		if !slices.Contains(Categories, Category) && Category != "FFT" {
			Categories = append(Categories, Category)
		} else if slices.Contains(Categories, Category) && Category != "FFT" {
			//pass
		} else {
			panic("Cannot call FFT3D or FFT3DAs function from within Calc. Call FFT3DAs/FFT3D alone instead.")
		}
	} else {
		panic("FFT data has to be assigned to FFT category")
	}
	if !slices.Contains(slices.Concat(DeclVarFFTDynAlias, DeclVarCalcDynAlias[Category]), NameOf(q)) {
		if q.NComp() == 3 {
			NewVectorField(NameOf(q), UnitOf(q), "", func(dst *data.Slice) { q.EvalTo(dst) })
		} else if q.NComp() == 1 {
			NewScalarField(NameOf(q), UnitOf(q), "", func(dst *data.Slice) { q.EvalTo(dst) })
		} else {
			panic("Invalid amount of  components.")
		}
		DeclVarFFTDyn = append(DeclVarFFTDyn, q)
		DeclVarCalcDyn[Category] = append(DeclVarCalcDyn[Category], q)
		DeclVarCalcDynAlias[Category] = append(DeclVarCalcDynAlias[Category], NameOf(q))
	}
}

func CalcAs(q Quantity, Name, Category string) {
	defer func() {
		if r := recover(); r == "identifier "+Name+" already defined" {
		}
	}()
	if _, isFFT := q.(interface {
		FFTOutputSize() [3]int
	}); !isFFT {
		if !slices.Contains(Categories, Category) && Category != "FFT" {
			Categories = append(Categories, Category)
		} else if Category != "FFT" {
		} else {
			panic("Cannot call FFT3D or FFT3DAs function from within CalcAs. Call FFT3DAs/FFT3D alone instead.")
		}
	} else {
		panic("FFT data has to be assigned to FFT category")
	}
	if !slices.Contains(slices.Concat(DeclVarFFTDynAlias, DeclVarCalcDynAlias[Category]), Name) && !slices.Contains(slices.Concat(DeclVarFFTDyn, DeclVarCalcDyn[Category]), q) {
		if q.NComp() == 3 {
			NewVectorField(Name, UnitOf(q), "", func(dst *data.Slice) { q.EvalTo(dst) })
		} else if q.NComp() == 1 {
			NewScalarField(Name, UnitOf(q), "", func(dst *data.Slice) { q.EvalTo(dst) })
		} else {
			panic("Invalid amount of  components.")
		}
		DeclVarFFTDyn = append(DeclVarFFTDyn, q)
		DeclVarCalcDyn[Category] = append(DeclVarCalcDyn[Category], q)
		DeclVarCalcDynAlias[Category] = append(DeclVarCalcDynAlias[Category], Name)
	} else if !slices.Contains(slices.Concat(DeclVarFFTDynAlias, DeclVarCalcDynAlias[Category]), Name) && slices.Contains(slices.Concat(DeclVarFFTDyn, DeclVarCalcDyn[Category]), q) {
		logUI.Log.Warn(fmt.Sprintf("Quantity %s behind %s exists already under %s.", NameOf(q), Name, NameOf(q)))
	}
}
