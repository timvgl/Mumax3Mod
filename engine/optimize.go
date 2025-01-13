package engine

import (
	"fmt"
	"log"
	"sync"

	"github.com/c-bata/goptuna"
	"github.com/c-bata/goptuna/tpe"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var target = []float64{2, 4, 6}

var (
	BackupQTTY  sync.Map
	SaveCounter = make(map[string]int)
)

func init() {
	DeclFunc("OptimizeQuantity", OptimizeQuantity, "")
	DeclFunc("MakeQuantityList", MakeQuantityList, "")
	DeclFunc("MakeAreaList", MakeAreaList, "")
	DeclFunc("CreateParameterSpace", CreateParameterSpace, "")
	DeclFunc("MakeIntSlice", MakeIntSlice, "")
	DeclFunc("CreateParameter", CreateParameter, "")
}

type optimize struct {
	target, output Quantity
	variables      []Quantity
	variablesStart []vectorScalar
	variablesEnd   []vectorScalar
	areaStart      [][3]int
	areaEnd        [][3]int
	period         float64
	StudyName      string
	Trials         int
	time           float64
}

type vectorScalar struct {
	vector    [3]float64
	useScalar bool
}

type dummyQuantity struct {
	path    string
	name    string
	storage *data.Slice
	ncomp   int
	time    float64
}

func (d dummyQuantity) Name() string {
	return d.name
}
func (d dummyQuantity) NComp() int {
	return d.ncomp
}

func (d dummyQuantity) EvalTo(dst *data.Slice) {
	data.Copy(dst, d.storage)
}

func ImportOVFAsQuantity(file, name string, i ...int) Quantity {
	if len(i) != 0 && len(i) != 1 {
		panic("Either zero or one integer is allowed")
	}
	fname := ""
	if len(i) == 1 {
		fname = fmt.Sprintf(file, i[0])
	} else {
		fname = file
	}
	cpuBuf, meta := LoadFileMeta(fname)
	buf := cuda.Buffer(cpuBuf.NComp(), cpuBuf.Size())
	data.Copy(buf, cpuBuf)
	qtty := dummyQuantity{fname, name, buf, 3, meta.Time}
	cpuBuf.Free()
	return qtty
}

func (s optimize) generate_vector_suggest(trial goptuna.Trial, q Quantity, j int) ([]float32, []float32, []float32, error) {
	var (
		vector0 []float32
		vector1 []float32
		vector2 []float32
	)
	length := s.areaEnd[j][0] - s.areaStart[j][0]
	length *= s.areaEnd[j][1] - s.areaStart[j][1]
	length *= s.areaEnd[j][2] - s.areaStart[j][2]
	paramName := fmt.Sprintf("%s_%d_%d", NameOf(q))
	for i := range length {
		v0, err0 := trial.SuggestFloat(fmt.Sprintf(paramName, i, 0), s.variablesStart[i].vector[0], s.variablesEnd[i].vector[0])
		if err0 != nil {
			return nil, nil, nil, err0
		}
		vector0 = append(vector0, float32(v0))
		if !s.variablesStart[i].useScalar {
			v1, err1 := trial.SuggestFloat(fmt.Sprintf(paramName, i, 1), s.variablesStart[i].vector[1], s.variablesEnd[i].vector[1])
			if err1 != nil {
				return nil, nil, nil, err1
			}
			vector1 = append(vector1, float32(v1))
			v2, err2 := trial.SuggestFloat(fmt.Sprintf(paramName, i, 2), s.variablesStart[i].vector[2], s.variablesEnd[i].vector[2])
			if err2 != nil {
				return nil, nil, nil, err2
			}
			vector2 = append(vector2, float32(v2))
		}
	}
	return vector0, vector1, vector2, nil
}

func (s optimize) get_params_as_data_slice(trial goptuna.Trial, q Quantity, j int) ([]float32, []float32, []float32, error) {
	var (
		vector0 []float32
		vector1 []float32
		vector2 []float32
	)
	length := s.areaEnd[j][0] - s.areaStart[j][0]
	length *= s.areaEnd[j][1] - s.areaStart[j][1]
	length *= s.areaEnd[j][2] - s.areaStart[j][2]
	paramName := fmt.Sprintf("%s_%d_%d", NameOf(q))
	for i := range length {
		v0, err0 := trial.SuggestFloat(fmt.Sprintf(paramName, i, 0), s.variablesStart[i].vector[0], s.variablesEnd[i].vector[0])
		if err0 != nil {
			return nil, nil, nil, err0
		}
		vector0 = append(vector0, float32(v0))
		if !s.variablesStart[i].useScalar {
			v1, err1 := trial.SuggestFloat(fmt.Sprintf(paramName, i, 1), s.variablesStart[i].vector[1], s.variablesEnd[i].vector[1])
			if err1 != nil {
				return nil, nil, nil, err1
			}
			vector1 = append(vector1, float32(v1))
			v2, err2 := trial.SuggestFloat(fmt.Sprintf(paramName, i, 2), s.variablesStart[i].vector[2], s.variablesEnd[i].vector[2])
			if err2 != nil {
				return nil, nil, nil, err2
			}
			vector2 = append(vector2, float32(v2))
		}
	}
	return vector0, vector1, vector2, nil
}

func (s optimize) objective(trial goptuna.Trial) (float64, error) {
	// Here we treat each element of array1 as a parameter to be learned.
	// If your array1 depends on fewer or more parameters in a more complex way,
	// replace these lines with your function's logic.
	for ii, q := range s.variables {
		size := [3]int{s.areaEnd[ii][0] - s.areaStart[ii][0], s.areaEnd[ii][1] - s.areaStart[ii][1], s.areaEnd[ii][2] - s.areaStart[ii][2]}
		vector0, vector1, vector2, err := s.generate_vector_suggest(trial, q, ii)
		if err != nil {
			return 0, err
		}
		if q.NComp() == 3 {
			suggestData := make([][]float32, 3)
			suggestData[0] = vector0
			suggestData[1] = vector1
			suggestData[2] = vector2
			buf := data.SliceFromArray(suggestData, size)
			SetScalarExcitation(NameOf(q), ScalarExcitationSlice{NameOf(q), s.areaStart[ii], s.areaEnd[ii], buf})
			buf.Free()
		}
	}
	RestoreBackup(append(s.variables, s.output))
	Time = s.time
	Run(s.period)

	input := cuda.Buffer(s.target.NComp(), SizeOf(s.target))
	output := cuda.Buffer(s.output.NComp(), SizeOf(s.output))
	s.target.EvalTo(input)
	s.output.EvalTo(output)
	DiffSq := cuda.MaxVecDiff(input, output)

	// The goal is to MINIMIZE this sum of squared differences.
	return DiffSq, nil
}

func MakeQuantityList(q ...Quantity) []Quantity {
	return q
}

func CreateParameter(q ...float64) vectorScalar {
	util.AssertMsg(len(q) == 3 || len(q) == 1, "CreateParameter must have 3 or 1 parameters")
	if len(q) == 3 {
		return vectorScalar{[3]float64{q[0], q[1], q[2]}, false}
	}
	return vectorScalar{[3]float64{q[0], q[0], q[0]}, true}
}

func CreateParameterSpace(q ...vectorScalar) []vectorScalar {
	return q
}

func MakeIntSlice(a, b, c int) [3]int {
	return [3]int{a, b, c}
}

func MakeAreaList(a ...[3]int) [][3]int {
	return a
}

func OptimizeQuantity(output Quantity, target dummyQuantity, variables []Quantity, areaStart, areaEnd [][3]int, parametersStart, parametersEnd []vectorScalar, studyName string, trials int) {
	util.AssertMsg(len(variables) == len(parametersStart) && len(variables) == len(parametersEnd), "variables, parametersStart and parametersEnd must have the same length")
	util.AssertMsg(len(areaStart) == len(areaEnd) && len(areaStart) == len(variables), "areaStart and areaEnd must have the same length like variables")
	for i := range variables {
		util.AssertMsg(parametersStart[i].useScalar == parametersEnd[i].useScalar, "start or end value for parameter space not properly defined.")
	}
	optim := optimize{target, output, variables, parametersStart, parametersEnd, areaStart, areaEnd, target.time - Time, studyName, trials, Time}
	CreateBackup(append(variables, output))
	sampler := tpe.NewSampler()
	study, err := goptuna.CreateStudy(
		optim.StudyName,
		goptuna.StudyOptionSampler(sampler),
	)
	if err != nil {
		log.Fatalf("Failed to create study: %v", err)
	}
	err = study.Optimize(optim.objective, optim.Trials)
	if err != nil {
		log.Fatalf("Failed to run optimization: %v", err)
	}

	bestValue, err := study.GetBestValue()
	bestParams, err := study.GetBestParams()
	if err != nil {
		log.Fatalf("Error getting best result: %v", err)
	}

	fmt.Println("=== Optimization Complete ===")
	fmt.Printf("Minimum sum of squared error = %f\n", bestValue)
	// Construct array1 from best params
	for k := range variables {
		size := [3]int{areaEnd[k][X] - areaStart[k][X], areaEnd[k][Y] - areaStart[k][Y], areaEnd[k][Z] - areaStart[k][Z]}
		varData := make([][]float32, 3)
		for c := range output.NComp() {
			for i := range prod(size) {
				varData[c] = append(varData[c], bestParams[fmt.Sprintf(NameOf(variables[k])+"_%d_%d", i, c)].(float32))
			}
		}
		buf := data.SliceFromArray(varData, size)
		info := data.Meta{Time: Time, Name: NameOf(output), Unit: UnitOf(output), CellSize: MeshOf(output).CellSize()}
		idx, ok := SaveCounter[NameOf(output)]
		if !ok {
			idx = 0
		}
		saveAs_sync(OD()+fmt.Sprintf(FilenameFormat, NameOf(output), idx)+".ovf", buf, info, outputFormat)
		SaveCounter[NameOf(output)] = idx + 1
	}

}

func CreateBackup(qs []Quantity) {
	for _, q := range qs {
		backup := cuda.Buffer(q.NComp(), SizeOf(q))
		q.EvalTo(backup)
		BackupQTTY.Store(NameOf(q), backup)
	}
}

func RestoreBackup(qs []Quantity) {
	for _, q := range qs {
		backup, ok := BackupQTTY.Load(NameOf(q))
		if ok {
			if s, ok := q.(interface {
				SetArray(src *data.Slice)
			}); ok {
				s.SetArray(backup.(*data.Slice))
			} else {
				log.Fatalf("Quantity %s does not have SetArray method", NameOf(q))
			}
		}
	}
}

/*
func OptimizeQuantityHandler() {
	// Create a TPE sampler (you could also use CMA-ES or random).
	sampler := tpe.NewSampler()

	// Create a new study. We'll just keep this in-memory for simplicity.
	study, err := goptuna.CreateStudy(
		"array-match-study",
		goptuna.StudyOptionSampler(sampler),
	)
	if err != nil {
		log.Fatalf("Failed to create study: %v", err)
	}

	// Run optimization for 50 trials.
	// Goptuna will vary p1, p2, p3 to minimize the objective.
	err = study.Optimize(objective, 1000)
	if err != nil {
		log.Fatalf("Failed to run optimization: %v", err)
	}

	// Retrieve the best trial result.
	bestValue, err := study.GetBestValue()
	bestParams, err := study.GetBestParams()
	if err != nil {
		log.Fatalf("Error getting best result: %v", err)
	}

	fmt.Println("=== Optimization Complete ===")
	fmt.Printf("Minimum sum of squared error = %f\n", bestValue)
	fmt.Printf("Best parameters found:\n")
	for k, v := range bestParams {
		fmt.Printf("  %s = %.4f\n", k, v.(float64))
	}

	// Construct array1 from best params
	array1 := []float64{
		bestParams["p1"].(float64),
		bestParams["p2"].(float64),
		bestParams["p3"].(float64),
	}
	fmt.Printf("Resulting array1: %v\n", array1)

	// Compare to target
	fmt.Printf("Target array2: %v\n", target)

	// Optionally, compute final difference
	var sumSq float64
	for i := range array1 {
		diff := array1[i] - target[i]
		sumSq += diff * diff
	}
	rmse := math.Sqrt(sumSq / float64(len(array1)))
	fmt.Printf("Final RMSE: %.4f\n", rmse)
}
*/
