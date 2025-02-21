package engine

import (
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/goptuna"
	"github.com/mumax/3/goptuna/tpe"
	"github.com/mumax/3/util"
)

var (
	BackupQTTY    sync.Map
	SaveCounter   = make(map[string]int)
	OptimizeTable = false
	barIdx        = 0
	bar_map       = make(map[string]*ProgressBar)
)

func init() {
	DeclFunc("OptimizeQuantity", OptimizeQuantity, "")
	DeclFunc("MakeQuantityList", MakeQuantityList, "")
	DeclFunc("CreateParameterSpace", CreateParameterSpace, "")
	DeclFunc("CreateParameter", CreateParameter, "")
	DeclFunc("ImportOVFAsQuantity", ImportOVFAsQuantity, "")
	DeclFunc("CreateFunction", CreateFunction, "")
	DeclFunc("MergeFitFunctions", MergeFitFunctions, "")
	DeclFunc("MergeParameterSpace", MergeParameterSpace, "")

}

type optimize struct {
	output          Quantity
	target          Quantity
	variables       []Quantity
	variablesStart  []map[string]vectorScalar
	variablesEnd    []map[string]vectorScalar
	areaStart       [][3]int
	areaEnd         [][3]int
	period          float64
	StudyName       string
	Trials          int
	time            float64
	functions       []StringFunction
	parsedFunctions [][3]*Function
	bar             *ProgressBar
}

type vectorScalar struct {
	vector        [3]float64
	useScalar     bool
	parameterName string
}

type dummyQuantity struct {
	path     string
	name     string
	storage  *data.Slice
	ncomp    int
	time     float64
	size     [3]int
	cellsize [3]float64
	comp     float64
}

type StringFunction struct {
	functions [3]string
	useScalar bool
}

func (s *StringFunction) IsScalar() bool {
	return s.useScalar
}

func (d dummyQuantity) Time() float64 {
	return d.time
}

func (d dummyQuantity) Mesh() *data.Mesh {
	return data.NewMesh(d.size[X], d.size[Y], d.size[Z], d.cellsize[X], d.cellsize[Y], d.cellsize[Z])
}

func (d dummyQuantity) Name() string {
	return d.name
}
func (d dummyQuantity) NComp() int {
	if !math.IsNaN(d.comp) {
		return 1
	} else {
		return d.ncomp
	}
}

func (d dummyQuantity) Comp(c int) dummyQuantity {
	d.comp = float64(c)
	return d
}

func (d dummyQuantity) EvalTo(dst *data.Slice) {
	if !math.IsNaN(d.comp) {
		data.CopyComp(dst, d.storage, int(d.comp))
	} else {
		data.Copy(dst, d.storage)
	}
}

func ImportOVFAsQuantity(file, name string, i ...int) dummyQuantity {
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
	qtty := dummyQuantity{fname, name, buf, 3, meta.Time, buf.Size(), meta.CellSize, math.NaN()}
	cpuBuf.Free()
	return qtty
}

func (d dummyQuantity) Free() {
	cuda.Recycle(d.storage)
}

func (s optimize) generateSliceFromExpr(xDep, yDep, zDep bool, vars map[string]interface{}, cellsize [3]float64, j int, function *Function) *data.Slice {
	return GenerateSliceFromExpr(xDep, yDep, zDep, vars, cellsize, function, s.areaStart[j], s.areaEnd[j])
}

func (s optimize) generateSliceFromFunction(trial goptuna.Trial, q Quantity, j int, cellsize [3]float64, comp int) (*data.Slice, error) {
	function, vars, err := GenerateExprFromFunctionString(s.functions[j].functions[comp])

	xDep := false
	yDep := false
	zDep := false
	worldVars := World.GetRuntimeVariables()
	for key := range vars {
		if key != "x_length" && key != "y_length" && key != "z_length" && key != "x_factor" && key != "y_factor" && key != "z_factor" && key != "t" && math.IsNaN(vars[key].(float64)) {
			varStartVector, ok1 := s.variablesStart[j][key]
			varEndVector, ok2 := s.variablesEnd[j][key]
			if ok1 && ok2 {
				value, err := trial.SuggestFloat(fmt.Sprintf("%s_%s_%d", NameOf(q), key, comp), varStartVector.vector[comp], varEndVector.vector[comp])
				if err != nil {
					return nil, err
				}
				vars[key] = value
			} else if worldVar, ok := worldVars[strings.ToLower(key)]; ok {
				vars[key] = worldVar
			} else {
				panic(fmt.Sprintf("Variable %s not found in parameter space", key))
			}

		} else if strings.ToLower(key) == "t" {
			panic("Time as a variable is not allowed in the function.")
		} else if key == "x_factor" {
			xDep = true
		} else if key == "y_factor" {
			yDep = true
		} else if key == "z_factor" {
			zDep = true
		}
	}

	if err != nil {
		panic(fmt.Sprintf("Failed to create function: %v", err))
	}
	s.parsedFunctions[j][comp] = function

	vector := s.generateSliceFromExpr(xDep, yDep, zDep, vars, cellsize, j, function)

	return vector, nil
}

func (s optimize) generate_vector_suggest_function(trial goptuna.Trial, q Quantity, j int) (*data.Slice, *data.Slice, *data.Slice, error) {
	var (
		vector0 *data.Slice
		vector1 *data.Slice
		vector2 *data.Slice
	)
	cellsize := MeshOf(q).CellSize()
	var err error
	vector0, err = s.generateSliceFromFunction(trial, q, j, cellsize, 0)
	if err != nil {
		return nil, nil, nil, err
	}
	if !s.functions[j].useScalar {
		vector1, err = s.generateSliceFromFunction(trial, q, j, cellsize, 1)
		if err != nil {
			return nil, nil, nil, err
		}
		vector2, err = s.generateSliceFromFunction(trial, q, j, cellsize, 2)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	return vector0, vector1, vector2, nil
}

func (s optimize) objective(trial goptuna.Trial) (float64, error) {
	// Here we treat each element of array1 as a parameter to be learned.
	// If your array1 depends on fewer or more parameters in a more complex way,
	// replace these lines with your function's logic.
	//start := time.Now()
	for ii, q := range s.variables {
		size := [3]int{s.areaEnd[ii][0] - s.areaStart[ii][0], s.areaEnd[ii][1] - s.areaStart[ii][1], s.areaEnd[ii][2] - s.areaStart[ii][2]}
		vector0, vector1, vector2, err := s.generate_vector_suggest_function(trial, q, ii)
		if err != nil {
			return 0, err
		}
		if q.NComp() == 3 {
			suggestData := make([]*data.Slice, 3)
			suggestData[0] = vector0
			suggestData[1] = vector1
			suggestData[2] = vector2
			bufGPU := data.SliceFromSlices(suggestData, size)
			//defer cuda.Recycle(bufGPU)
			SetExcitation(NameOf(q), ExcitationSlice{NameOf(q), s.areaStart[ii], s.areaEnd[ii], bufGPU, q.NComp(), false, s.functions[ii]})
		} else {
			suggestData := make([]*data.Slice, 1)
			suggestData[0] = vector0
			bufGPU := data.SliceFromSlices(suggestData, size)
			//defer cuda.Recycle(bufGPU)
			SetScalarExcitation(NameOf(q), ScalarExcitationSlice{NameOf(q), s.areaStart[ii], s.areaEnd[ii], bufGPU, false, s.functions[ii]})
		}
	}
	//t := time.Now()
	//elapsed := t.Sub(start)
	//fmt.Println(fmt.Sprintf("Needed %d ms for getting vector", elapsed.Milliseconds()))
	RestoreBackup([]Quantity{s.output})

	Time = s.time
	RunNoOutput(s.period)

	target := cuda.Buffer(s.target.NComp(), SizeOf(s.target))
	defer cuda.Recycle(target)
	output := cuda.Buffer(s.output.NComp(), SizeOf(s.output))
	defer cuda.Recycle(output)
	s.target.EvalTo(target)
	s.output.EvalTo(output)
	var DiffSq float64
	if target.NComp() == 3 {
		DiffSq = cuda.MaxVecDiff(target, output)
	} else {
		buffer := cuda.Buffer(1, SizeOf(s.target))
		cuda.Madd2(buffer, target, output, 1, -1)
		cuda.Mul(buffer, buffer, buffer)
		DiffSq = float64(cuda.Sum(buffer))
		cuda.Recycle(buffer)
	}
	barIdx += 1
	s.bar.Update(float64(barIdx))

	// The goal is to MINIMIZE this sum of squared differences.
	return DiffSq, nil
}

func MakeQuantityList(q ...Quantity) []Quantity {
	return q
}

func CreateParameter(name string, q ...float64) vectorScalar {
	util.AssertMsg(len(q) == 3 || len(q) == 1, "CreateParameter must have 3 or 1 parameters")
	if len(q) == 3 {
		return vectorScalar{[3]float64{q[0], q[1], q[2]}, false, name}
	}
	return vectorScalar{[3]float64{q[0], q[0], q[0]}, true, name}
}

func CreateParameterSpace(qs ...vectorScalar) map[string]vectorScalar {
	qMap := make(map[string]vectorScalar)
	for _, q := range qs {
		qMap[q.parameterName] = q
	}
	return qMap
}

func MergeParameterSpace(q ...map[string]vectorScalar) []map[string]vectorScalar {
	return q
}

func CreateFunction(a ...string) StringFunction {
	if len(a) == 1 {
		b := [3]string{a[0], "", ""}
		return StringFunction{b, true}
	} else if len(a) == 3 {
		b := [3]string{a[0], a[1], a[2]}
		return StringFunction{b, false}
	} else {
		panic("Need to have either one or three functions, depending on the quantity.")
	}

}

func MergeFitFunctions(a ...StringFunction) []StringFunction {
	return a
}

func OptimizeQuantity(output Quantity, target dummyQuantity, variables []Quantity, functions []StringFunction, parametersStart, parametersEnd []map[string]vectorScalar, studyName string, trials int, keep_bar bool, reduceTime float64) {
	util.AssertMsg(len(variables) == len(parametersStart) && len(variables) == len(parametersEnd), "variables, parametersStart and parametersEnd must have the same length")
	//util.AssertMsg(len(areaStart) == len(areaEnd) && len(areaStart) == len(variables), "areaStart and areaEnd must have the same length like variables")
	util.AssertMsg(len(functions) == len(variables), "need as many functions as variables.")
	areaStart := make([][3]int, 0)
	areaEnd := make([][3]int, 0)
	for i := range variables {
		for j := range parametersStart[i] {
			util.AssertMsg(parametersStart[i][j].useScalar == parametersEnd[i][j].useScalar, "start or end value for parameter space not properly defined.")
			util.AssertMsg(functions[i].useScalar == parametersEnd[i][j].useScalar, fmt.Sprintf("Function type does not match parameter type. %s", strings.Join(functions[i].functions[:], ", ")))
			util.AssertMsg((variables[i].NComp() == 1) == functions[i].useScalar, fmt.Sprintf("%s has different amount of components than function.", NameOf(variables[i])))
		}
		if s, ok := variables[i].(*cropped); ok {
			areaStart = append(areaStart, [3]int{s.x1, s.y1, s.z1})
			areaEnd = append(areaEnd, [3]int{s.x2, s.y2, s.z2})
		} else if s, ok := variables[i].(*expanded); ok {
			areaStart = append(areaStart, [3]int{s.x1, s.y1, s.z1})
			areaEnd = append(areaEnd, [3]int{s.x2, s.y2, s.z2})
		} else {
			areaStart = append(areaStart, [3]int{0, 0, 0})
			tmpMesh := MeshOf(variables[i])
			areaEnd = append(areaEnd, tmpMesh.Size())
		}
	}
	var bar *ProgressBar
	if keep_bar {
		var ok bool
		bar, ok = bar_map[NameOf(output)]
		if !ok {
			bar = NewProgressBar(0., float64(trials), "🧲", false)
			bar_map[NameOf(output)] = bar
		}
	} else {
		bar = NewProgressBar(0., float64(trials), "🧲", false)
	}
	optim := optimize{output, target, variables, parametersStart, parametersEnd, areaStart, areaEnd, target.time - reduceTime - Time, studyName, trials, Time, functions, make([][3]*Function, len(variables)), bar}
	CreateBackup([]Quantity{output})
	logFile, err := os.OpenFile(OD()+"/optimization.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("Failed to open log file: %v\n", err)
		return
	}
	defer logFile.Close()

	log.SetOutput(logFile)
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	logFile.WriteString(target.path + "\n")

	sampler := tpe.NewSampler()
	study, err := goptuna.CreateStudy(
		optim.StudyName,
		logFile,
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
	if err != nil {
		panic(err)
	}
	bestParams, err := study.GetBestParams()
	if err != nil {
		log.Fatalf("Error getting best result: %v", err)
	}

	var optiTable *DataTable
	if !CheckIfFileExistsOD("optimize.txt") {
		CreateNewTableOrg := CreateNewTable
		CreateNewTable = true
		OptimizeTable = true
		optiTable = NewTable("optimize")
		optiTable.init()
		CreateNewTable = CreateNewTableOrg
		fprint(optiTable, "# t (s)")
		for k := range variables {
			if variables[k].NComp() == 3 {
				for c := range variables[k].NComp() {
					vars := optim.parsedFunctions[k][c].required
					sort.Strings(vars)
					for _, par := range vars {
						if par != "x" && par != "y" && par != "z" && par != "t" {
							fprint(optiTable, "\t", NameOf(variables[k])+"_"+par+"_"+string('x'+c))
						}
					}
				}
			} else if variables[k].NComp() == 1 {
				vars := optim.parsedFunctions[k][0].required
				sort.Strings(vars)
				for _, par := range vars {
					if par != "x" && par != "y" && par != "z" && par != "t" {
						fprint(optiTable, "\t", NameOf(variables[k])+"_"+par)
					}
				}
			}
		}
		fprint(optiTable, "\t", "error")
		fprintln(optiTable)
		optiTable.Flush()
	} else {
		CreateNewTableOrg := CreateNewTable
		CreateNewTable = false
		RewriteHeaderTableOrg := RewriteHeaderTable
		RewriteHeaderTable = false
		optiTable = NewTable("optimize")
		optiTable.init()
		CreateNewTable = CreateNewTableOrg
		RewriteHeaderTable = RewriteHeaderTableOrg
	}
	optiTable.Flushlock.Lock() // flush during write gives errShortWrite
	defer optiTable.Flushlock.Unlock()
	// Construct array1 from best params
	fprint(optiTable, target.time-reduceTime)
	for k := range variables {
		size := [3]int{areaEnd[k][X] - areaStart[k][X], areaEnd[k][Y] - areaStart[k][Y], areaEnd[k][Z] - areaStart[k][Z]}
		varData := make([]*data.Slice, variables[k].NComp())
		cellsize := MeshOf(variables[k]).CellSize()
		for c := range variables[k].NComp() {
			xDep := false
			yDep := false
			zDep := false
			varsSlice := optim.parsedFunctions[k][c].required
			vars, err := InitializeVars(varsSlice)
			if err != nil {
				panic(err)
			}
			sort.Strings(varsSlice)
			for _, par := range varsSlice {
				if par != "x_length" && par != "y_length" && par != "z_length" && par != "x_factor" && par != "y_factor" && par != "z_factor" && par != "t" {
					val, ok := bestParams[fmt.Sprintf("%s_%s_%d", NameOf(variables[k]), par, c)].(float64)
					if !ok {
						panic(fmt.Sprintf("Parameter %s_%s_%d not found in bestParams", NameOf(variables[k]), par, c))
					}
					vars[par] = val
					fprint(optiTable, "\t", val)
				} else if par == "x_length" {
					xDep = true
				} else if par == "y_length" {
					yDep = true
				} else if par == "z_length" {
					zDep = true
				}
			}
			varData[c] = optim.generateSliceFromExpr(xDep, yDep, zDep, vars, cellsize, k, optim.parsedFunctions[k][c])
		}
		bufGPU := data.SliceFromSlices(varData, size)
		info := data.Meta{Time: target.time - reduceTime, Name: NameOf(variables[k]), Unit: UnitOf(variables[k]), CellSize: cellsize}
		idx, ok := SaveCounter[NameOf(variables[k])]
		if !ok {
			idx = 0
		}
		saveAs_sync(OD()+fmt.Sprintf(FilenameFormat, NameOf(variables[k]), idx)+".ovf", bufGPU.HostCopy(), info, outputFormat)
		SaveCounter[NameOf(variables[k])] = idx + 1
		if variables[k].NComp() == 3 {
			SetExcitation(NameOf(variables[k]), ExcitationSlice{NameOf(variables[k]), optim.areaStart[k], optim.areaEnd[k], bufGPU, variables[k].NComp(), false, optim.functions[k]})
			defer EraseSetExcitation(NameOf(variables[k]))
		} else {
			SetScalarExcitation(NameOf(variables[k]), ScalarExcitationSlice{NameOf(variables[k]), optim.areaStart[k], optim.areaEnd[k], bufGPU, false, optim.functions[k]})
			defer EraseSetScalarExcitation(NameOf(variables[k]))
		}
	}
	fprint(optiTable, "\t", bestValue)
	fprintln(optiTable)
	optiTable.Flush()
	RestoreBackup([]Quantity{output})
	Time = optim.time
	HideProgressBarManualF(true)
	Run(optim.period)
	HideProgressBarManualF(false)
	ResetProgressBarMode()
	idx, ok := SaveCounter[NameOf(output)]
	if !ok {
		idx = 0
	}
	SaveAs(output, fmt.Sprintf(FilenameFormat, NameOf(output), idx)+".ovf")
	SaveCounter[NameOf(output)] = idx + 1
	DeleteBackup([]Quantity{output})
	cuda.Recycle(target.storage)
	barIdx = 0
	optim.bar.Finish()
}

func DeleteBackup(qs []Quantity) {
	for _, q := range qs {
		qUnpack := UnpackQuantity(q)
		tmp, ok := BackupQTTY.Load(NameOf(qUnpack))
		if ok {
			buf := tmp.(*data.Slice)
			cuda.Recycle(buf)
			BackupQTTY.Delete(NameOf(qUnpack))
		}
	}
}

func CreateBackup(qs []Quantity) {
	for _, q := range qs {
		qUnpack := UnpackQuantity(q)
		backup := cuda.Buffer(qUnpack.NComp(), SizeOf(qUnpack))
		qUnpack.EvalTo(backup)
		BackupQTTY.Store(NameOf(qUnpack), backup)
	}
}

func UnpackQuantity(q Quantity) Quantity {
	if s, ok := q.(*cropped); ok {
		return UnpackQuantity(s.parent)
	} else if s, ok := q.(*expanded); ok {
		return UnpackQuantity(s.parent)
	} else if s, ok := q.(*component); ok {
		return UnpackQuantity(s.parent)
	} else if s, ok := q.(ScalarField); ok {
		return UnpackQuantity(s.Quantity)
	} else {
		return q
	}
}

func RestoreBackup(qs []Quantity) {
	for _, q := range qs {
		qUnpacked := UnpackQuantity(q)
		backup, ok := BackupQTTY.Load(NameOf(qUnpacked))
		if ok {
			if s, ok := qUnpacked.(interface {
				SetArray(src *data.Slice)
			}); ok {
				s.SetArray(backup.(*data.Slice))
			} else if s, ok := qUnpacked.(interface {
				Buffer() *data.Slice
			}); ok {
				data.Copy(s.Buffer(), backup.(*data.Slice))
			} else {
				log.Fatalf("Quantity %s does not have SetArray or Buffer method", NameOf(q))
			}
		}
	}
}
