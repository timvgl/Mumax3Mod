package cmaes

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestNewOptimizer(t *testing.T) {
	mean := []float64{0, 0}
	sigma0 := 1.3
	optimizer, err := NewOptimizer(mean, sigma0)
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
		return
	}
	if optimizer.dim != 2 {
		t.Errorf("should be 2, but got %d", optimizer.dim)
	}
	if optimizer.popsize != 6 {
		t.Errorf("should be 6, but got %d", optimizer.popsize)
	}
	if optimizer.mu != 3 {
		t.Errorf("should be 3, but got %d", optimizer.mu)
	}
	if math.Abs(optimizer.muEff-2.0286114646100617) > 0.0001 {
		t.Errorf("should be 2.0286114646100617, but got %f", optimizer.muEff)
	}
	if math.Abs(optimizer.c1-0.1548153998964136) > 0.0001 {
		t.Errorf("should be 0.1548153998964136, but got %f", optimizer.c1)
	}
	if math.Abs(optimizer.cmu-0.05785908507191633) > 0.0001 {
		t.Errorf("should be 0.05785908507191633, but got %f", optimizer.cmu)
	}
	expectedWeights := []float64{0.63704257, 0.28457026, 0.07838717, -0.28638378, -0.76495809, -1.15598178}
	if !floats.EqualApprox(optimizer.weights.RawVector().Data, expectedWeights, 0.0001) {
		t.Errorf("should be %#v, but got %#v", expectedWeights, optimizer.weights.RawVector().Data)
	}
	if math.Abs(optimizer.cSigma-0.44620498737831715) > 0.0001 {
		t.Errorf("should be 0.44620498737831715, but got %f", optimizer.cSigma)
	}
	if math.Abs(optimizer.dSigma-1.4462049873783172) > 0.0001 {
		t.Errorf("should be 1.4462049873783172, but got %f", optimizer.dSigma)
	}
	if math.Abs(optimizer.cc-0.6245545390268264) > 0.0001 {
		t.Errorf("should be 0.6245545390268264, but got %f", optimizer.cc)
	}
	if math.Abs(optimizer.chiN-1.254272742818995) > 0.0001 {
		t.Errorf("should be 1.254272742818995, but got %f", optimizer.chiN)
	}
}

func TestOptimizer_Ask(t *testing.T) {
	mean := []float64{1, 2}
	sigma0 := 1.3
	optimizer, err := NewOptimizer(
		mean, sigma0,
		OptimizerOptionSeed(0),
	)
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
	}
	x, err := optimizer.Ask()
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
		return
	}
	if len(x) != 2 {
		t.Errorf("dim should be 2, but got %d", len(x))
	}
}

func TestOptimizer_Tell(t *testing.T) {
	mean := []float64{0, 0}
	sigma0 := 1.3
	optimizer, err := NewOptimizer(
		mean, sigma0,
		OptimizerOptionSeed(0),
	)
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
	}

	solutions := []*Solution{
		{
			Params: []float64{1.91231201, -1.71265425},
			Value:  9.439823102089013,
		},
		{
			Params: []float64{1.34432608, -1.46615684},
			Value:  31.240107864789625,
		},
		{
			Params: []float64{0.18029599, -1.40324121},
			Value:  43.56283645181167,
		},
		{
			Params: []float64{0.56156551, -0.28068763},
			Value:  301.54946482426044,
		},
		{
			Params: []float64{1.91285964, -0.15018778},
			Value:  343.3623991306441,
		},
		{
			Params: []float64{1.35673101, 0.31811501},
			Value:  540.0660546114952,
		},
	}
	err = optimizer.Tell(solutions)
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
		return
	}

	expectedC := []float64{1.18753118, -0.51807317, 0, 1.40915646}
	if !floats.EqualApprox(optimizer.c.RawSymmetric().Data, expectedC, 0.0001) {
		t.Errorf("should be %#v, but got %#v", expectedC, optimizer.c.RawSymmetric().Data)
	}

	expectedPSigma := []float64{1.47322504, -1.47627395}
	if !floats.EqualApprox(optimizer.pSigma.RawVector().Data, expectedPSigma, 0.0001) {
		t.Errorf("should be %#v, but got %#v", expectedPSigma, optimizer.pSigma.RawVector().Data)
	}

	expectedPC := []float64{1.63987933, -1.64327314}
	if !floats.EqualApprox(optimizer.pc.RawVector().Data, expectedPC, 0.0001) {
		t.Errorf("should be %#v, but got %#v", expectedPC, optimizer.pc.RawVector().Data)
	}

	expectedMean := []float64{1.61491227, -1.61825441}
	if !floats.EqualApprox(optimizer.mean.RawVector().Data, expectedMean, 0.0001) {
		t.Errorf("should be %#v, but got %#v", expectedMean, optimizer.mean.RawVector().Data)
	}

	if math.Abs(optimizer.sigma-1.5949829983784432) > 0.0001 {
		t.Errorf("should be 1.5949829983784432, but got %f", optimizer.sigma)
	}
}

func TestOptimizer_IsFeasible(t *testing.T) {
	tests := []struct {
		name   string
		bounds *mat.Dense
		value  *mat.VecDense
		want   bool
	}{
		{
			name: "feasible",
			bounds: mat.NewDense(2, 2, []float64{
				-1, 1,
				-2, -1,
			}),
			value: mat.NewVecDense(2, []float64{-0.5, -1.5}),
			want:  true,
		},
		{
			name: "out of lower bound",
			bounds: mat.NewDense(2, 2, []float64{
				-1, 1,
				-2, -1,
			}),
			value: mat.NewVecDense(2, []float64{-1.5, -1.5}),
			want:  false,
		},
		{
			name: "out of upper bound",
			bounds: mat.NewDense(2, 2, []float64{
				-1, 1,
				-2, -1,
			}),
			value: mat.NewVecDense(2, []float64{-0.5, 1}),
			want:  false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			optimizer, err := NewOptimizer(
				[]float64{0, 0}, 1.3,
				OptimizerOptionBounds(tt.bounds),
			)
			if err != nil {
				t.Errorf("should be nil, but got %s", err)
			}

			feasible := optimizer.isFeasible(tt.value)
			if tt.want != feasible {
				t.Errorf("should be %v, but got %v", tt.want, feasible)
			}
		})
	}
}

func TestOptimizer_RepairInfeasibleParams(t *testing.T) {
	tests := []struct {
		name     string
		bounds   *mat.Dense
		value    *mat.VecDense
		repaired *mat.VecDense
	}{
		{
			name: "out of lower bound",
			bounds: mat.NewDense(2, 2, []float64{
				-1, 1,
				-2, -1,
			}),
			value:    mat.NewVecDense(2, []float64{-1.5, -1.5}),
			repaired: mat.NewVecDense(2, []float64{-1, -1.5}),
		},
		{
			name: "out of upper bound",
			bounds: mat.NewDense(2, 2, []float64{
				-1, 1,
				-2, -1,
			}),
			value:    mat.NewVecDense(2, []float64{-0.5, 1}),
			repaired: mat.NewVecDense(2, []float64{-0.5, -1}),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			optimizer, err := NewOptimizer(
				[]float64{0, 0}, 1.3,
				OptimizerOptionBounds(tt.bounds),
			)
			if err != nil {
				t.Errorf("should be nil, but got %s", err)
			}

			err = optimizer.repairInfeasibleParams(tt.value)
			if err != nil {
				t.Errorf("should be nil, but got %s", err)
			}
			if !floats.Same(tt.value.RawVector().Data, tt.repaired.RawVector().Data) {
				t.Errorf("should be %v, but got %v", tt.value.RawVector().Data, tt.repaired.RawVector().Data)
			}
		})
	}
}

func TestOptimizer_ShouldStop_FunValHist(t *testing.T) {
	optimizer, err := NewOptimizer(
		[]float64{0, 0}, 1.3,
	)
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
		return
	}
	popsize := optimizer.PopulationSize()
	rng := rand.New(rand.NewSource(0))

	for i := 0; i < optimizer.funHistTerm+1; i++ {
		if optimizer.ShouldStop() {
			t.Error("ShouldStop() should be false, but got true")
			return
		}

		solutions := make([]*Solution, popsize)
		for j := 0; j < popsize; j++ {
			solutions[j] = &Solution{
				Params: []float64{rng.NormFloat64(), rng.NormFloat64()},
				Value:  0.01,
			}
		}
		err = optimizer.Tell(solutions)
		if err != nil {
			t.Errorf("should be nil, but got %s", err)
			return
		}
	}
	if !optimizer.ShouldStop() {
		t.Error("ShouldStop() should be true, but got false")
		return
	}
}

func TestOptimizer_ShouldStop_DivergentBehavior(t *testing.T) {
	optimizer, err := NewOptimizer(
		[]float64{0, 0}, 1e-4,
	)
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
		return
	}
	popsize := optimizer.PopulationSize()
	rng := rand.New(rand.NewSource(0))

	solutions := make([]*Solution, popsize)
	for i := 0; i < popsize; i++ {
		solutions[i] = &Solution{
			Params: []float64{100 * rng.NormFloat64(), 100 * rng.NormFloat64()},
			Value:  0.01,
		}
	}
	err = optimizer.Tell(solutions)
	if err != nil {
		t.Errorf("should be nil, but got %s", err)
		return
	}
	if !optimizer.ShouldStop() {
		t.Error("ShouldStop() should be true, but got false")
		return
	}
}
