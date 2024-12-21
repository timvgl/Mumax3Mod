package engine

import (
	"fmt"
	"path"
	"reflect"
	"strings"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"github.com/mumax/3/dump"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("Save", Save, "Save space-dependent quantity once, with auto filename")
	DeclFunc("SaveAs", SaveAs, "Save space-dependent quantity with custom filename")
	DeclFunc("SaveAsAt", SaveAsAt, "")

	DeclLValue("FilenameFormat", &fformat{}, "printf formatting string for output filenames.")
	DeclLValue("OutputFormat", &oformat{}, "Format for data files: OVF1_TEXT, OVF1_BINARY, OVF2_TEXT or OVF2_BINARY")

	DeclROnly("OVF1_BINARY", OVF1_BINARY, "OutputFormat = OVF1_BINARY sets binary OVF1 output")
	DeclROnly("OVF2_BINARY", OVF2_BINARY, "OutputFormat = OVF2_BINARY sets binary OVF2 output")
	DeclROnly("OVF1_TEXT", OVF1_TEXT, "OutputFormat = OVF1_TEXT sets text OVF1 output")
	DeclROnly("OVF2_TEXT", OVF2_TEXT, "OutputFormat = OVF2_TEXT sets text OVF2 output")
	DeclROnly("DUMP", DUMP, "OutputFormat = DUMP sets text DUMP output")
	DeclFunc("Snapshot", Snapshot, "Save image of quantity")
	DeclFunc("SnapshotAs", SnapshotAs, "Save image of quantity with custom filename")
	DeclVar("SnapshotFormat", &SnapshotFormat, "Image format for snapshots: jpg, png or gif.")
}

var (
	FilenameFormat = "%s%06d"    // formatting string for auto filenames.
	SnapshotFormat = "jpg"       // user-settable snapshot format
	outputFormat   = OVF2_BINARY // user-settable output format
)

type fformat struct{}

func (*fformat) Eval() interface{}      { return FilenameFormat }
func (*fformat) SetValue(v interface{}) { drainOutput(); FilenameFormat = v.(string) }
func (*fformat) Type() reflect.Type     { return reflect.TypeOf("") }

type oformat struct{}

func (*oformat) Eval() interface{}      { return outputFormat }
func (*oformat) SetValue(v interface{}) { drainOutput(); outputFormat = v.(OutputFormat) }
func (*oformat) Type() reflect.Type     { return reflect.TypeOf(OutputFormat(OVF2_BINARY)) }

// Save once, with auto file name
func Save(q Quantity) {
	qname := NameOf(q)
	fname := autoFname(NameOf(q), outputFormat, autonum[qname])
	SaveAs(q, fname)
	autonum[qname]++
}

func SavePrefix(q Quantity, prefix string) {
	qname := NameOf(q)
	fname := autoFnamePrefix(prefix, NameOf(q), outputFormat, autonumPrefix[qname])
	SaveAs(q, fname)
	autonumPrefix[qname]++
}

func SaveAsOverwrite(q Quantity, name string) {
	fname := autoFname(name, outputFormat, autonumAs[name])
	SaveAs(q, fname)
	autonumAs[name]++
}

func SaveAsOverwritePrefix(q Quantity, prefix, name string) {
	fname := autoFnamePrefix(prefix, name, outputFormat, autonumPrefixAs[name])
	SaveAs(q, fname)
	autonumPrefixAs[name]++
}

// Save under given file name (transparent async I/O).
func SaveAs(q Quantity, fname string) {

	if !strings.HasPrefix(fname, OD()) {
		fname = OD() + fname // don't clean, turns http:// in http:/
	}

	if path.Ext(fname) == "" {
		fname += ("." + StringFromOutputFormat[outputFormat])
	}
	buffer := ValueOf(q) // TODO: check and optimize for Buffer()
	defer cuda.Recycle(buffer)
	info := data.Meta{Time: Time, Name: NameOf(q), Unit: UnitOf(q), CellSize: MeshOf(q).CellSize()}
	data := buffer.HostCopy() // must be copy (async io)
	if s, ok := q.(interface {
		Axis() ([3]int, [3]float64, [3]float64, []string)
		FFTOutputSize() [3]int
	}); ok {
		NxNyNz, startK, endK, transformedAxis := s.Axis()
		queOutput(func() {
			saveAsFFT_sync(fname, data, info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, true)
		})
	} else if s, ok := q.(interface {
		Axis() ([3]int, [3]float64, [3]float64, []string)
	}); ok {
		NxNyNz, startK, endK, transformedAxis := s.Axis()
		queOutput(func() {
			saveAsFFT_sync(fname, data, info, outputFormat, NxNyNz, startK, endK, transformedAxis, false, true)
		})
	} else {
		queOutput(func() { saveAs_sync(fname, data, info, outputFormat) })
	}
}

func SaveAsAt(q Quantity, fname, dir string) {

	if !strings.HasPrefix(fname, dir) {
		fname = dir + fname // don't clean, turns http:// in http:/
	}

	if path.Ext(fname) == "" {
		fname += ("." + StringFromOutputFormat[outputFormat])
	}
	buffer := ValueOf(q) // TODO: check and optimize for Buffer()
	defer cuda.Recycle(buffer)
	info := data.Meta{Time: Time, Name: NameOf(q), Unit: UnitOf(q), CellSize: MeshOf(q).CellSize()}
	data := buffer.HostCopy() // must be copy (async io)
	if s, ok := q.(interface {
		Axis() ([3]int, [3]float64, [3]float64, []string)
		FFTOutputSize() [3]int
	}); ok {
		NxNyNz, startK, endK, transformedAxis := s.Axis()
		queOutput(func() {
			saveAsFFT_sync(fname, data, info, outputFormat, NxNyNz, startK, endK, transformedAxis, true, true)
		})
	} else if s, ok := q.(interface {
		Axis() ([3]int, [3]float64, [3]float64, []string)
	}); ok {
		NxNyNz, startK, endK, transformedAxis := s.Axis()
		queOutput(func() {
			saveAsFFT_sync(fname, data, info, outputFormat, NxNyNz, startK, endK, transformedAxis, false, true)
		})
	} else {
		queOutput(func() { saveAs_sync(fname, data, info, outputFormat) })
	}
}

// Save image once, with auto file name
func Snapshot(q Quantity) {
	qname := NameOf(q)
	if s, ok := q.(interface {
		Real() *fftOperation3DReal
		Imag() *fftOperation3DImag
	}); ok {
		fnameReal := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, qname+"_real", autonumSnapshots[qname])
		size, _, _, _ := s.Real().Axis()
		buf := cuda.Buffer(s.Real().nComp, size)
		cuda.Zero(buf)
		defer cuda.Recycle(buf)
		s.Real().EvalTo(buf)
		dataReal := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameReal, dataReal) })

		fnameImag := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, qname+"_imag", autonumSnapshots[qname])
		cuda.Zero(buf)
		s.Imag().EvalTo(buf)
		dataImag := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameImag, dataImag) })
	} else {
		s := ValueOf(q)
		defer cuda.Recycle(s)
		data := s.HostCopy() // must be copy (asyncio)
		fname := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, qname, autonumSnapshots[qname])
		queOutput(func() { snapshot_sync(fname, data) })
	}
	autonumSnapshots[qname]++
}

func SnapshotPrefix(q Quantity, prefix string) {
	qname := NameOf(q)

	if s, ok := q.(interface {
		Real() *fftOperation3DReal
		Imag() *fftOperation3DImag
	}); ok {
		fnameReal := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, prefix+"_"+qname+"_real", autonumSnapshotsPrefix[qname])
		size, _, _, _ := s.Real().Axis()
		buf := cuda.Buffer(s.Real().nComp, size)
		cuda.Zero(buf)
		defer cuda.Recycle(buf)
		s.Real().EvalTo(buf)
		dataReal := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameReal, dataReal) })

		fnameImag := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, prefix+"_"+qname+"_imag", autonumSnapshotsPrefix[qname])
		cuda.Zero(buf)
		s.Imag().EvalTo(buf)
		dataImag := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameImag, dataImag) })
	} else {
		s := ValueOf(q)
		defer cuda.Recycle(s)
		data := s.HostCopy() // must be copy (asyncio)
		fname := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, prefix+"_"+qname, autonumSnapshotsPrefix[qname])
		queOutput(func() { snapshot_sync(fname, data) })
	}
	autonumSnapshotsPrefix[qname]++
}

func SnapshotAsOverwrite(q Quantity, name string) {
	if s, ok := q.(interface {
		Real() *fftOperation3DReal
		Imag() *fftOperation3DImag
	}); ok {
		fnameReal := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, name+"_real", autonumSnapshotsAs[name])
		size, _, _, _ := s.Real().Axis()
		buf := cuda.Buffer(s.Real().nComp, size)
		cuda.Zero(buf)
		defer cuda.Recycle(buf)
		s.Real().EvalTo(buf)
		dataReal := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameReal, dataReal) })

		fnameImag := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, name+"_imag", autonumSnapshotsAs[name])
		cuda.Zero(buf)
		s.Imag().EvalTo(buf)
		dataImag := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameImag, dataImag) })
	} else {
		s := ValueOf(q)
		defer cuda.Recycle(s)
		data := s.HostCopy() // must be copy (asyncio)
		fname := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, name, autonumSnapshotsAs[name])
		queOutput(func() { snapshot_sync(fname, data) })
	}
	autonumSnapshotsAs[name]++
}

func SnapshotAsOverwritePrefix(q Quantity, prefix, name string) {
	if s, ok := q.(interface {
		Real() *fftOperation3DReal
		Imag() *fftOperation3DImag
	}); ok {
		fnameReal := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, prefix+"_"+name+"_real", autonumSnapshotsPrefixAs[name])
		size, _, _, _ := s.Real().Axis()
		buf := cuda.Buffer(s.Real().nComp, size)
		cuda.Zero(buf)
		defer cuda.Recycle(buf)
		s.Real().EvalTo(buf)
		dataReal := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameReal, dataReal) })

		fnameImag := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, prefix+"_"+name+"_imag", autonumSnapshotsPrefixAs[name])
		cuda.Zero(buf)
		s.Imag().EvalTo(buf)
		dataImag := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameImag, dataImag) })
	} else {
		s := ValueOf(q)
		defer cuda.Recycle(s)
		data := s.HostCopy() // must be copy (asyncio)
		fname := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, prefix+"_"+name, autonumSnapshotsPrefixAs[name])
		queOutput(func() { snapshot_sync(fname, data) })
	}
	autonumSnapshotsPrefixAs[name]++
}

func SnapshotAs(q Quantity, fname string) {
	if s, ok := q.(interface {
		Real() *fftOperation3DReal
		Imag() *fftOperation3DImag
	}); ok {
		fnameReal := fmt.Sprintf(OD() + fname + "_real" + "." + SnapshotFormat)
		size, _, _, _ := s.Real().Axis()
		buf := cuda.Buffer(s.Real().nComp, size)
		cuda.Zero(buf)
		defer cuda.Recycle(buf)
		s.Real().EvalTo(buf)
		dataReal := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameReal, dataReal) })

		fnameImag := fmt.Sprintf(OD() + fname + "_imag" + "." + SnapshotFormat)
		cuda.Zero(buf)
		s.Imag().EvalTo(buf)
		dataImag := buf.HostCopy()
		queOutput(func() { snapshot_sync(fnameImag, dataImag) })
	} else {
		s := ValueOf(q)
		defer cuda.Recycle(s)
		data := s.HostCopy() // must be copy (asyncio)
		fname := fmt.Sprintf(OD() + fname + "." + SnapshotFormat)
		queOutput(func() { snapshot_sync(fname, data) })
	}
}

// synchronous snapshot
func snapshot_sync(fname string, output *data.Slice) {
	f, err := httpfs.Create(fname)
	util.FatalErr(err)
	defer f.Close()
	draw.RenderFormat(f, output, "auto", "auto", arrowSize, path.Ext(fname))
}

// synchronous save
func saveAs_sync(fname string, s *data.Slice, info data.Meta, format OutputFormat) {
	f, err := httpfs.Create(fname)
	util.FatalErr(err)
	defer f.Close()

	switch format {
	case OVF1_TEXT:
		oommf.WriteOVF1(f, s, info, "text")
	case OVF1_BINARY:
		oommf.WriteOVF1(f, s, info, "binary 4")
	case OVF2_TEXT:
		oommf.WriteOVF2(f, s, info, "text")
	case OVF2_BINARY:
		oommf.WriteOVF2(f, s, info, "binary 4")
	case DUMP:
		dump.Write(f, s, info)
	default:
		panic("invalid output format")
	}

}

func saveAsFFT_sync(fname string, s *data.Slice, info data.Meta, format OutputFormat, NxNyNz [3]int, startK, endK [3]float64, transformedAxes []string, complex bool, timeSpace bool) {
	f, err := httpfs.Create(fname)
	util.FatalErr(err)
	defer f.Close()

	switch format {
	case OVF2_TEXT:
		oommf.WriteOVF2FFT(f, s, info, "text", NxNyNz, startK, endK, transformedAxes, timeSpace)
	case OVF2_BINARY:
		if complex {
			oommf.WriteOVF2FFT(f, s, info, "binary 4+4", NxNyNz, startK, endK, transformedAxes, timeSpace)
		} else {
			oommf.WriteOVF2FFT(f, s, info, "binary 4", NxNyNz, startK, endK, transformedAxes, timeSpace)
		}
	case DUMP:
		dump.Write(f, s, info)
	default:
		panic("Invalid output format. OVF1 not supported for complex data.")
	}

}

type OutputFormat int

const (
	OVF1_TEXT OutputFormat = iota + 1
	OVF1_BINARY
	OVF2_TEXT
	OVF2_BINARY
	DUMP
)

var (
	StringFromOutputFormat = map[OutputFormat]string{
		OVF1_TEXT:   "ovf",
		OVF1_BINARY: "ovf",
		OVF2_TEXT:   "ovf",
		OVF2_BINARY: "ovf",
		DUMP:        "dump"}
)
