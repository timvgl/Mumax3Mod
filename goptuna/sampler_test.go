package goptuna_test

import (
	"errors"
	"log"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/mumax/3/goptuna"
	"github.com/mumax/3/goptuna/internal/testutil"
)

func TestRandomSamplerOptionSeed(t *testing.T) {
	tests := []struct {
		name         string
		distribution interface{}
	}{
		{
			name: "uniform distribution",
			distribution: goptuna.UniformDistribution{
				High: 10,
				Low:  0,
			},
		},
		{
			name: "int uniform distribution",
			distribution: goptuna.IntUniformDistribution{
				High: 10,
				Low:  0,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sampler1 := goptuna.NewRandomSampler()
			sampler2 := goptuna.NewRandomSampler(goptuna.RandomSamplerOptionSeed(2))
			sampler3 := goptuna.NewRandomSampler(goptuna.RandomSamplerOptionSeed(2))

			s1, err := sampler1.Sample(nil, goptuna.FrozenTrial{}, "foo", tt.distribution)
			if err != nil {
				t.Errorf("should not be err, but got %s", err)
			}
			s2, err := sampler2.Sample(nil, goptuna.FrozenTrial{}, "foo", tt.distribution)
			if err != nil {
				t.Errorf("should not be err, but got %s", err)
			}
			s3, err := sampler3.Sample(nil, goptuna.FrozenTrial{}, "foo", tt.distribution)
			if err != nil {
				t.Errorf("should not be err, but got %s", err)
			}
			if s1 == s2 {
				t.Errorf("should not be the same but both are %f", s1)
			}
			if s2 != s3 {
				t.Errorf("should be equal, but got %f and %f", s2, s3)
			}
		})
	}
}

func TestRandomSampler_SampleLogUniform(t *testing.T) {
	sampler := goptuna.NewRandomSampler()
	study, err := goptuna.CreateStudy("", os.Stdout, goptuna.StudyOptionSampler(sampler))
	if err != nil {
		t.Errorf("should not be err, but got %s", err)
		return
	}

	distribution := goptuna.LogUniformDistribution{
		Low:  1e-7,
		High: 1,
	}

	points := make([]float64, 100)
	for i := 0; i < 100; i++ {
		trialID, err := study.Storage.CreateNewTrial(study.ID)
		if err != nil {
			t.Errorf("should not be err, but got %s", err)
			return
		}
		trial, err := study.Storage.GetTrial(trialID)
		if err != nil {
			t.Errorf("should not be err, but got %s", err)
			return
		}
		sampled, err := study.Sampler.Sample(study, trial, "x", distribution)
		if err != nil {
			t.Errorf("should not be err, but got %s", err)
			return
		}
		if sampled < distribution.Low || sampled > distribution.High {
			t.Errorf("should not be less than %f, and larger than %f, but got %f",
				distribution.High, distribution.Low, sampled)
			return
		}
		points[i] = sampled
	}

	for i := range points {
		if points[i] < distribution.Low {
			t.Errorf("should be higher than %f, but got %f",
				distribution.Low, points[i])
			return
		}
		if points[i] > distribution.High {
			t.Errorf("should be lower than %f, but got %f",
				distribution.High, points[i])
			return
		}
	}
}

func TestRandomSampler_SampleDiscreteUniform(t *testing.T) {
	sampler := goptuna.NewRandomSampler()
	study, err := goptuna.CreateStudy("", os.Stdout, goptuna.StudyOptionSampler(sampler))
	if err != nil {
		t.Errorf("should not be err, but got %s", err)
		return
	}

	distribution := goptuna.DiscreteUniformDistribution{
		Low:  -10,
		High: 10,
		Q:    0.1,
	}

	points := make([]float64, 100)
	for i := 0; i < 100; i++ {
		trialID, err := study.Storage.CreateNewTrial(study.ID)
		if err != nil {
			t.Errorf("should not be err, but got %s", err)
			return
		}
		trial, err := study.Storage.GetTrial(trialID)
		if err != nil {
			t.Errorf("should not be err, but got %s", err)
			return
		}
		sampled, err := study.Sampler.Sample(study, trial, "x", distribution)
		if err != nil {
			t.Errorf("should not be err, but got %s", err)
			return
		}
		if sampled < distribution.Low || sampled > distribution.High {
			t.Errorf("should not be less than %f, and larger than %f, but got %f",
				distribution.High, distribution.Low, sampled)
			return
		}
		points[i] = sampled
	}

	for i := range points {
		points[i] -= distribution.Low
		points[i] /= distribution.Q
		roundPoint := math.Round(points[i])
		if !testutil.AlmostEqualFloat64(roundPoint, points[i], 1e-6) {
			t.Errorf("should be almost the same, but got %f and %f",
				roundPoint, points[i])
			return
		}
	}
}

type queueRelativeSampler struct {
	params []map[string]float64
	errors []error
	index  int
}

func (s *queueRelativeSampler) SampleRelative(
	study *goptuna.Study,
	trial goptuna.FrozenTrial,
	searchSpace map[string]interface{},
) (p map[string]float64, e error) {
	p = s.params[s.index]
	if len(s.errors) > 0 {
		e = s.errors[s.index]
	}
	s.index++
	return
}

func TestRelativeSampler(t *testing.T) {
	sampler := goptuna.NewRandomSampler()
	relativeSampler := &queueRelativeSampler{
		params: []map[string]float64{
			{
				"uniform":     3,
				"log_uniform": 100,
				"int":         7,
				"discrete":    5.5,
				"categorical": 2, // choice3
			},
		},
	}

	study, err := goptuna.CreateStudy(
		"",
		os.Stdout,
		goptuna.StudyOptionSampler(sampler),
		goptuna.StudyOptionRelativeSampler(relativeSampler),
	)
	if err != nil {
		t.Errorf("should not be err, but got %s", err)
		return
	}

	// First trial cannot trigger relative sampler.
	err = study.Optimize(func(trial goptuna.Trial) (f float64, e error) {
		_, _ = trial.SuggestFloat("uniform", -10, 10)
		_, _ = trial.SuggestLogFloat("log_uniform", 1e-10, 1e10)
		_, _ = trial.SuggestInt("int", -10, 10)
		_, _ = trial.SuggestDiscreteFloat("discrete", -10, 10, 0.5)
		_, _ = trial.SuggestCategorical("categorical", []string{"choice1", "choice2", "choice3"})
		return 0.0, nil
	}, 1)

	// Second trial call relative sampler.
	err = study.Optimize(func(trial goptuna.Trial) (f float64, e error) {
		uniformParam, _ := trial.SuggestFloat("uniform", -10, 10)
		if uniformParam != 3 {
			t.Errorf("should be 3, but got %f", uniformParam)
		}

		logUniformParam, _ := trial.SuggestLogFloat("log_uniform", 1e-10, 1e10)
		if logUniformParam != 100 {
			t.Errorf("should be 100, but got %f", logUniformParam)
		}

		intParam, _ := trial.SuggestInt("int", -10, 10)
		if intParam != 7 {
			t.Errorf("should be 7, but got %d", intParam)
		}

		discreteParam, _ := trial.SuggestDiscreteFloat("discrete", -10, 10, 0.5)
		if discreteParam != 5.5 {
			t.Errorf("should be 5.5, but got %f", discreteParam)
		}

		categoricalParam, _ := trial.SuggestCategorical("categorical", []string{"choice1", "choice2", "choice3"})
		if categoricalParam != "choice3" {
			t.Errorf("should be 'choice3', but got %s", categoricalParam)
		}
		return 0.0, nil
	}, 1)
}

func TestRelativeSampler_UnsupportedSearchSpace(t *testing.T) {
	sampler := goptuna.NewRandomSampler()
	relativeSampler := &queueRelativeSampler{
		params: []map[string]float64{
			nil,
		},
		errors: []error{
			goptuna.ErrUnsupportedSearchSpace,
		},
	}

	study, err := goptuna.CreateStudy(
		"",
		os.Stdout,
		goptuna.StudyOptionSampler(sampler),
		goptuna.StudyOptionRelativeSampler(relativeSampler),
		goptuna.StudyOptionLogger(&goptuna.StdLogger{
			Logger: log.New(os.Stdout, "", log.LstdFlags),
			Level:  goptuna.LoggerLevelDebug,
			Color:  true,
		}),
	)
	if err != nil {
		t.Errorf("should not be err, but got %s", err)
		return
	}

	// First trial cannot trigger relative sampler.
	err = study.Optimize(func(trial goptuna.Trial) (f float64, e error) {
		_, _ = trial.SuggestFloat("x1", -10, 10)
		_, _ = trial.SuggestLogFloat("x2", 1e-10, 1e10)
		return 0.0, nil
	}, 1)
	if err != nil {
		t.Errorf("should not be err, but got %s", err)
		return
	}

	// Second trial. RelativeSampler return ErrUnsupportedSearchSpace.
	err = study.Optimize(func(trial goptuna.Trial) (f float64, e error) {
		_, e = trial.SuggestFloat("x1", -10, 10)
		if e != nil {
			t.Errorf("err should be nil, but got %s", e)
		}
		_, e = trial.SuggestLogFloat("x2", 1e-10, 1e10)
		if e != nil {
			t.Errorf("err should be nil, but got %s", e)
		}
		return 0.0, nil
	}, 1)
	if err != nil {
		t.Errorf("should not be err, but got %s", err)
		return
	}
}

func TestIntersectionSearchSpace(t *testing.T) {
	tests := []struct {
		name         string
		study        func() *goptuna.Study
		expectedKeys []string
		want         map[string]interface{}
		wantErr      bool
	}{
		{
			name: "No trial",
			study: func() *goptuna.Study {
				study, err := goptuna.CreateStudy("sampler_test", os.Stdout)
				if err != nil {
					panic(err)
				}
				return study
			},
			want:    map[string]interface{}{},
			wantErr: false,
		},
		{
			name: "One trial",
			study: func() *goptuna.Study {
				study, err := goptuna.CreateStudy("sampler_test", os.Stdout)
				if err != nil {
					panic(err)
				}

				if err = study.Optimize(func(trial goptuna.Trial) (v float64, e error) {
					x, _ := trial.SuggestInt("x", 0, 10)
					y, _ := trial.SuggestFloat("y", -3, 3)
					return float64(x) + y, nil
				}, 1); err != nil {
					panic(err)
				}
				return study
			},
			want: map[string]interface{}{
				"x": goptuna.IntUniformDistribution{
					High: 10,
					Low:  0,
				},
				"y": goptuna.UniformDistribution{
					High: 3,
					Low:  -3,
				},
			},
			wantErr: false,
		},
		{
			name: "Second trial (only 'y' parameter is suggested in this trial)",
			study: func() *goptuna.Study {
				study, err := goptuna.CreateStudy("sampler_test", os.Stdout)
				if err != nil {
					panic(err)
				}

				// First Trial
				if err = study.Optimize(func(trial goptuna.Trial) (v float64, e error) {
					x, _ := trial.SuggestInt("x", 0, 10)
					y, _ := trial.SuggestFloat("y", -3, 3)
					return float64(x) + y, nil
				}, 1); err != nil {
					panic(err)
				}

				// Second Trial
				if err = study.Optimize(func(trial goptuna.Trial) (v float64, e error) {
					y, _ := trial.SuggestFloat("y", -3, 3)
					return y, nil
				}, 1); err != nil {
					panic(err)
				}
				return study
			},
			want: map[string]interface{}{
				"y": goptuna.UniformDistribution{
					High: 3,
					Low:  -3,
				},
			},
			wantErr: false,
		},
		{
			name: "Failed or pruned trials are not considered in the calculation of an intersection search space.",
			study: func() *goptuna.Study {
				study, err := goptuna.CreateStudy("sampler_test", os.Stdout)
				if err != nil {
					panic(err)
				}

				// First Trial
				if err = study.Optimize(func(trial goptuna.Trial) (v float64, e error) {
					x, _ := trial.SuggestInt("x", 0, 10)
					y, _ := trial.SuggestFloat("y", -3, 3)
					return float64(x) + y, nil
				}, 1); err != nil {
					panic(err)
				}

				// Second Trial
				if err = study.Optimize(func(trial goptuna.Trial) (v float64, e error) {
					y, _ := trial.SuggestFloat("y", -3, 3)
					return y, nil
				}, 1); err != nil {
					panic(err)
				}

				// Failed trial (ignore error)
				_ = study.Optimize(func(trial goptuna.Trial) (v float64, e error) {
					_, _ = trial.SuggestFloat("y", -3, 3)
					return 0.0, errors.New("something error")
				}, 1)
				// Pruned trial
				if err = study.Optimize(func(trial goptuna.Trial) (v float64, e error) {
					_, _ = trial.SuggestFloat("y", -3, 3)
					return 0.0, goptuna.ErrTrialPruned
				}, 1); err != nil {
					panic(err)
				}
				return study
			},
			want: map[string]interface{}{
				"y": goptuna.UniformDistribution{
					High: 3,
					Low:  -3,
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := goptuna.IntersectionSearchSpace(tt.study())
			if (err != nil) != tt.wantErr {
				t.Errorf("IntersectionSearchSpace() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(got) != len(tt.want) {
				t.Errorf("IntersectionSearchSpace() return %d items, but want %d", len(got), len(tt.want))
			}
			for key := range tt.want {
				if distribution, ok := got[key]; !ok {
					t.Errorf("IntersectionSearchSpace() should have %s key", key)
				} else if !reflect.DeepEqual(distribution, tt.want[key]) {
					t.Errorf("IntersectionSearchSpace()[%s] = %v, want %v", key, distribution, tt.want[key])
				}
			}
		})
	}
}
