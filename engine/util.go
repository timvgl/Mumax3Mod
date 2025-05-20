package engine

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/dump"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
)

var (
	GradMx               = NewVectorField("GradMx", "J", "", GetGradMx)
	GradMy               = NewVectorField("GradMy", "J", "", GetGradMy)
	GradMz               = NewVectorField("GradMz", "J", "", GetGradMz)
	GSDir                string
	Suffix               string
	absPath              string = ""
	newestOvfFileIndex   int
	failGetNewestOvfFile bool = true
	gotValidOvfFile      bool = false
	newestOvfFile        string
)

func init() {
	DeclFunc("Expect", Expect, "Used for automated tests: checks if a value is close enough to the expected value")
	DeclFunc("ExpectV", ExpectV, "Used for automated tests: checks if a vector is close enough to the expected value")
	DeclFunc("Fprintln", Fprintln, "Print to file")
	DeclFunc("Sign", sign, "Signum function")
	DeclFunc("Vector", Vector, "Constructs a vector with given components")
	DeclConst("Mu0", mag.Mu0, "Permittivity of vaccum (Tm/A)")
	DeclFunc("Print", myprint, "Print to standard output")
	DeclFunc("ExpandFloat", func(d float64) string { return strconv.FormatFloat(d, 'f', -1, 64) }, "Print to standard output")
	DeclFunc("LoadFile", LoadFileDSlice, "Load a data file (ovf or dump)")
	DeclFunc("LoadFileMyDir", LoadFileDSliceMyDir, "Load a data file from the ouput directory")
	DeclFunc("LoadFileWithoutMem", LoadFileWithoutMem, "")
	DeclFunc("SetQuantityWithoutMemToConfig", SetQuantityWithoutMemToConfig, "")
	DeclFunc("LoadFileVector", LoadFileAsConfig, "")
	DeclFunc("Index2Coord", Index2Coord, "Convert cell index to x,y,z coordinate in meter")
	DeclFunc("NewSlice", NewSlice, "Makes a 4D array with a specified number of components (first argument) "+
		"and a specified size nx,ny,nz (remaining arguments)")
	DeclFunc("NewVectorMask", NewVectorMask, "Makes a 3D array of vectors")
	DeclFunc("NewScalarMask", NewScalarMask, "Makes a 3D array of scalars")
	DeclFunc("IsFile", CheckIfFileExists, "Checks if a file at given path exists and returns bool")
	DeclFunc("IsFileMyDir", CheckIfFileExistsOD, "Checks if a file at given path exists and returns bool")
	DeclFunc("ConcStr", ConcStr, "")
	DeclVar("GSDir", &GSDir, "")
	DeclFunc("EraseOD", EraseOD, "")
	DeclVar("Suffix", &Suffix, "")
	DeclFunc("int", castInt, "")
	DeclFunc("iceil", func(v float64) int { return int(math.Ceil(v)) }, "")
	DeclFunc("ifloor", func(v float64) int { return int(math.Floor(v)) }, "")
	DeclFunc("float", castFloat, "")
	DeclFunc("string", castString, "")
	DeclFunc("NUndoneToLog", NUndoneToLog, "")
	DeclFunc("exec", RunExec, "")
	DeclFunc("execDir", RunExecDir, "")
	DeclVar("absPath", &absPath, "")
	DeclFunc("WriteNUndoneToLog", WriteNUndoneToLog, "")
	DeclFunc("getNewestOvfFile", get_newest_ovf_file, "")
	DeclVar("failGetNewestOvfFile", &failGetNewestOvfFile, "")
	DeclFunc("OD", OD, "")
	DeclVar("newestOvfFileIndex", &newestOvfFileIndex, "")
	DeclVar("gotValidOvfFile", &gotValidOvfFile, "")
	DeclVar("newestOvfFile", &newestOvfFile, "")
	DeclFunc("GetTime", LoadFileTime, "")

}

func castableInt(str string) bool {
	_, err := strconv.Atoi(str)
	if err != nil {
		return false
	}
	return true
}

func ValidOvfFile(fname string) bool {
	defer func() bool {
		if r := recover(); r != nil {
			return false
		} else {
			return true
		}
	}()
	in, err := httpfs.Open(fname)
	if err != nil {
		return false
	}
	if path.Ext(fname) == ".dump" {
		_, _, err = dump.Read(in)
	} else {
		_, _, err = oommf.Read(in)
	}
	return err == nil
}

func get_newest_ovf_file(q Quantity) string {
	dir := OD()
	d, err := os.Open(dir)
	if err != nil {
		util.PanicErr(err)
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	biggestOvfIndex := 0
	foundFile := false
	for _, name := range names {
		var (
			index    int
			gotIndex bool = false
		)
		for i, subString := range name {
			if castableInt(string(subString)) {
				gotIndex = true
				index = i
				break
			}
		}
		if gotIndex && name[:index] == NameOf(q) {
			ovfFileNameList := strings.Split(strings.TrimLeft(name[index+1:], "0"), ".")
			tmpOvfIndex, err := strconv.Atoi(strings.Join(ovfFileNameList[:len(ovfFileNameList)-1], "."))
			if err == nil && tmpOvfIndex > biggestOvfIndex {
				biggestOvfIndex = tmpOvfIndex
				foundFile = true
			}
		}
	}
	if foundFile {
		for i := biggestOvfIndex; i >= 0; i-- {
			if ValidOvfFile(fmt.Sprintf("%s%s%0*d%s", OD(), NameOf(q), 6, i, ".ovf")) {
				biggestOvfIndex = i
				break
			}
			if i == 0 {
				foundFile = false
			}
		}
	}
	if foundFile {
		newestOvfFileIndex = biggestOvfIndex
		gotValidOvfFile = true
		return fmt.Sprintf("%s%0*d%s", NameOf(q), 6, biggestOvfIndex, ".ovf")
	} else if failGetNewestOvfFile {
		panic("Could not find suitable ovf file.")
	} else {
		return ""
	}
}

func WriteNUndoneToLog() {
	LogOut(fmt.Sprintf("NUndone: %v", NUndone))
}

func RunExec(cmdStr string) {
	fmt.Println(cmdStr)
	path, err := exec.LookPath(strings.Split(cmdStr, " ")[0])
	if err != nil {
		panic(err)
	}
	cmd := exec.Command(path, strings.Split(cmdStr, " ")[1:]...)
	var outb, errb bytes.Buffer
	cmd.Stdout = &outb
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		panic(err)
	}
	fmt.Println("out:", outb.String(), "err:", errb.String())
}

func RunExecDirSweep(cmdStr, path string) {
	if strings.Contains(cmdStr, "%v") || strings.Contains(cmdStr, "%s") {
		argsSprintf := make([]any, strings.Count(cmdStr, "%v")+strings.Count(cmdStr, "%s"))
		for i := range len(argsSprintf) {
			argsSprintf[i] = absPath + ODSweep()
		}
		RunExec(fmt.Sprintf(cmdStr, argsSprintf...))
	} else {
		RunExec(cmdStr + " " + absPath + ODSweep())
	}
}

func RunExecDir(cmdStr string) {
	if strings.Contains(cmdStr, "%v") || strings.Contains(cmdStr, "%s") {
		argsSprintf := make([]any, strings.Count(cmdStr, "%v")+strings.Count(cmdStr, "%s"))
		for i := range len(argsSprintf) {
			argsSprintf[i] = absPath + OD()
		}
		RunExec(fmt.Sprintf(cmdStr, argsSprintf...))
	} else {
		RunExec(cmdStr + " " + absPath + OD())
	}
}

func NUndoneToLog() {
	LogOut(fmt.Sprintf("NUndone: %v", NUndone))
}

func castString(val interface{}) string {
	return fmt.Sprintf("%v", val)
}

func castInt(val interface{}) int {
	switch v := val.(type) {
	case string:
		valStr, ok := strconv.Atoi(v)
		if ok == nil {
			return (valStr)
		} else {
			panic("Got non-castable string.")
		}
	case int32:
		return int(v)
	case float32:
		return int(v)
	case float64:
		return int(v)
	default:
		panic(fmt.Sprintf("Type not recognized for %v", v))
	}
}

func castFloat(val interface{}) float64 {
	switch v := val.(type) {
	case string:
		valStr, ok := strconv.ParseFloat(v, 64)
		if ok == nil {
			return valStr
		} else {
			panic("Got non-castable string.")
		}
	case int32:
		return float64(v)
	case float32:
		return float64(v)
	case float64:
		return v
	default:
		panic(fmt.Sprintf("Type not recognized for %v", v))
	}
}

func EraseOD() {
	dir := OD()
	d, err := os.Open(dir)
	if err != nil {
		util.PanicErr(err)
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		util.PanicErr(err)
	}
	for _, name := range names {
		if name != "log.txt" && name != "gui" && name != "table.txt" {
			err = os.RemoveAll(filepath.Join(dir, name))
			if err != nil {
				util.PanicErr(err)
			}
		}
	}
}

func ConcStr(strs ...string) string {
	var resultStr string
	for _, str := range strs {
		resultStr += str
	}

	return resultStr
}

func CheckIfFileExists(path string) bool {
	if _, err := os.Stat(path); err == nil {
		return true
	} else if errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		panic("State of file " + path + " not known")
	}
}

func CheckIfFileExistsOD(filename string) bool {
	return CheckIfFileExists(OD() + filename)
}

// Returns a new new slice (3D array) with given number of components and size.
func NewSlice(ncomp, Nx, Ny, Nz int) *data.Slice {
	return data.NewSlice(ncomp, [3]int{Nx, Ny, Nz})
}

func NewVectorMask(Nx, Ny, Nz int) *data.Slice {
	return data.NewSlice(3, [3]int{Nx, Ny, Nz})
}

func NewScalarMask(Nx, Ny, Nz int) *data.Slice {
	return data.NewSlice(1, [3]int{Nx, Ny, Nz})
}

// Constructs a vector
func Vector(x, y, z float64) data.Vector {
	return data.Vector{x, y, z}
}

// Test if have lies within want +/- maxError,
// and print suited message.
func Expect(msg string, have, want, maxError float64) {
	if math.IsNaN(have) || math.IsNaN(want) || math.Abs(have-want) > maxError {
		LogOut(msg, ":", " have: ", have, " want: ", want, "±", maxError)
		Close()
		os.Exit(1)
	} else {
		LogOut(msg, ":", have, "OK")
	}
	// note: we also check "want" for NaN in case "have" and "want" are switched.
}

func ExpectV(msg string, have, want data.Vector, maxErr float64) {
	for c := 0; c < 3; c++ {
		Expect(fmt.Sprintf("%v[%v]", msg, c), have[c], want[c], maxErr)
	}
}

// Append msg to file. Used to write aggregated output of many simulations in one file.
func Fprintln(filename string, msg ...interface{}) {
	if !path.IsAbs(filename) {
		filename = OD() + filename
	}
	httpfs.Touch(filename)
	err := httpfs.Append(filename, []byte(fmt.Sprintln(myFmt(msg)...)))
	util.FatalErr(err)
}

func LoadFileDSliceMyDir(fname string) *data.Slice {
	return LoadFileDSlice(OD() + fname)
}

// Read a magnetization state from .dump file.
func LoadFileDSlice(fname string) *data.Slice {
	in, err := httpfs.Open(fname)
	util.FatalErr(err)
	var s *data.Slice
	if path.Ext(fname) == ".dump" {
		s, _, err = dump.Read(in)
	} else {
		s, _, err = oommf.Read(in)
	}
	util.FatalErr(err)
	return s
}

func LoadFileMeta(fname string) (*data.Slice, data.Meta) {
	in, err := httpfs.Open(fname)
	if err != nil {
		panic(err)
	}
	var d *data.Slice
	var s data.Meta
	if path.Ext(fname) == ".dump" {
		d, s, err = dump.Read(in)
	} else {
		d, s, err = oommf.Read(in)
	}
	if err != nil {
		panic(err)
	}
	return d, s
}

func LoadFileTime(fname string) float64 {
	in, err := httpfs.Open(fname)
	util.FatalErr(err)
	var s data.Meta
	if path.Ext(fname) == ".dump" {
		_, s, err = dump.Read(in)
	} else {
		_, s, err = oommf.Read(in)
	}
	util.FatalErr(err)
	return s.Time
}

func LoadFileAsConfig(fname string) Config {
	var d, meta = LoadFileMeta(fname)
	var vec = d.Vectors()
	var xCellSize, yCellSize, zCellSize = meta.CellSize[0], meta.CellSize[1], meta.CellSize[2]
	var size = M.Mesh().Size()
	var xSize, ySize, zSize = float64(size[0]), float64(size[1]), float64(size[2])
	return func(x, y, z float64) data.Vector {
		var xIndex, yIndex, zIndex = int((x + xSize*xCellSize/2) / xCellSize), int((y + ySize*yCellSize/2) / yCellSize), int((z + zSize*zCellSize/2) / zCellSize)
		return data.New_dataVector(float64(vec[0][zIndex][yIndex][xIndex]), float64(vec[1][zIndex][yIndex][xIndex]), float64(vec[2][zIndex][yIndex][xIndex]))
	}
}

func LoadFileWithoutMem(q Quantity, fname string) {
	if NameOf(q) == "normStrain" {
		loadNormStrain = true
		loadNormStrainPath = fname
	} else if NameOf(q) == "shearStrain" {
		loadShearStrain = true
		loadShearStrainPath = fname
	} else if NameOf(q) == "normStress" {
		loadNormStress = true
		loadNormStressPath = fname
	} else if NameOf(q) == "shearStress" {
		loadShearStress = true
		loadShearStressPath = fname
	} else {
		util.AssertMsg(true, "Loading file for "+NameOf(q)+" not yet supported.")
	}

}

func SetQuantityWithoutMemToConfig(q Quantity, cfg Config) {
	if NameOf(q) == "normStrain" {
		loadNormStrainConfig = true
		normStrainConfig = cfg
	} else if NameOf(q) == "shearStrain" {
		loadShearStrainConfig = true
		shearStrainConfig = cfg
	} else if NameOf(q) == "normStress" {
		loadNormStressConfig = true
		normStressConfig = cfg
	} else if NameOf(q) == "shearStress" {
		loadShearStressConfig = true
		shearStressConfig = cfg
	} else {
		util.AssertMsg(true, "Loading config for "+NameOf(q)+" not yet supported.")
	}

}

func SetInShape(dst *data.Slice, region Shape, conf Config) {
	checkMesh()

	if region == nil {
		region = universe
	}
	host := dst.HostCopy()
	h := host.Vectors()
	n := dst.Size()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				r := Index2Coord(ix, iy, iz)
				x, y, z := r[X], r[Y], r[Z]
				if region(x, y, z) { // inside
					u := conf(x, y, z)
					h[X][iz][iy][ix] = float32(u[X])
					h[Y][iz][iy][ix] = float32(u[Y])
					h[Z][iz][iy][ix] = float32(u[Z])
				}
			}
		}
	}
	SetArray(dst, host)
}

func GetGradMx(dst *data.Slice) {
	cuda.Grad(dst, M.Buffer(), M.Mesh())
}

func GetGradMy(dst *data.Slice) {
	cuda.Grad2(dst, M.Buffer(), M.Mesh())
}

func GetGradMz(dst *data.Slice) {
	cuda.Grad3(dst, M.Buffer(), M.Mesh())
}

// Download a quantity to host,
// or just return its data when already on host.
func Download(q Quantity) *data.Slice {
	// TODO: optimize for Buffer()
	buf := ValueOf(q)
	defer cuda.Recycle(buf)
	if buf.CPUAccess() {
		return buf
	} else {
		return buf.HostCopy()
	}
}

// print with special formatting for some known types
func myprint(msg ...interface{}) {
	LogOut(myFmt(msg)...)
}

// mumax specific formatting (Slice -> average, etc).
func myFmt(msg []interface{}) []interface{} {
	for i, m := range msg {
		if e, ok := m.(*float64); ok {
			msg[i] = *e
		}
		// Tabledata: print average
		if m, ok := m.(Quantity); ok {
			str := fmt.Sprint(AverageOf(m))
			msg[i] = str[1 : len(str)-1] // remove [ ]
			continue
		}
	}
	return msg
}

// converts cell index to coordinate, internal coordinates
func Index2Coord(ix, iy, iz int) data.Vector {
	m := Mesh()
	n := m.Size()
	c := m.CellSize()
	x := c[X]*(float64(ix)-0.5*float64(n[X]-1)) - TotalShift
	y := c[Y]*(float64(iy)-0.5*float64(n[Y]-1)) - TotalYShift
	z := c[Z] * (float64(iz) - 0.5*float64(n[Z]-1))
	return data.Vector{x, y, z}
}

func sign(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}

// returns a/b, or 0 when b == 0
func safediv(a, b float32) float32 {
	if b == 0 {
		return 0
	} else {
		return a / b
	}
}

// dst = a/b, unless b == 0
func paramDiv(dst, a, b [][NREGION]float32) {
	util.Assert(len(dst) == 1 && len(a) == 1 && len(b) == 1)
	for i := 0; i < NREGION; i++ { // not regions.maxreg
		dst[0][i] = safediv(a[0][i], b[0][i])
	}
}

// shortcut for slicing unaddressable_vector()[:]
func slice(v [3]float64) []float64 {
	return v[:]
}

func unslice(v []float64) [3]float64 {
	util.Assert(len(v) == 3)
	return [3]float64{v[0], v[1], v[2]}
}

func assureGPU(s *data.Slice) *data.Slice {
	if s.GPUAccess() {
		return s
	} else {
		return cuda.GPUCopy(s)
	}
}

type caseIndep []string

func (s *caseIndep) Len() int           { return len(*s) }
func (s *caseIndep) Less(i, j int) bool { return strings.ToLower((*s)[i]) < strings.ToLower((*s)[j]) }
func (s *caseIndep) Swap(i, j int)      { (*s)[i], (*s)[j] = (*s)[j], (*s)[i] }

func sortNoCase(s []string) {
	i := caseIndep(s)
	sort.Sort(&i)
}

func checkNaN1(x float64) {
	if math.IsNaN(x) {
		panic("NaN")
	}
}

// trim trailing newlines
func rmln(a string) string {
	for strings.HasSuffix(a, "\n") {
		a = a[:len(a)-1]
	}
	return a
}

const (
	X = 0
	Y = 1
	Z = 2
)

const (
	SCALAR = 1
	VECTOR = 3
)
