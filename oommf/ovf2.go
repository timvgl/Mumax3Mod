package oommf

import (
	"fmt"
	"io"
	"log"
	"slices"
	"strings"
	"unsafe"

	"github.com/mumax/3/data"
)

func WriteOVF2(out io.Writer, q *data.Slice, meta data.Meta, dataformat string) {
	writeOVF2Header(out, q, meta)
	writeOVF2Data(out, q, dataformat)
	hdr(out, "End", "Segment")
}

func WriteOVF2FFT(out io.Writer, q *data.Slice, meta data.Meta, dataformat string, NxNyNz [3]int, startK, endK [3]float64, transformedAxes []string) {
	writeOVF2HeaderFFT(out, q, meta, NxNyNz, startK, endK, transformedAxes)
	writeOVF2Data(out, q, dataformat)
	hdr(out, "End", "Segment")
}

func writeOVF2Header(out io.Writer, q *data.Slice, meta data.Meta) {
	gridsize := q.Size()
	cellsize := meta.CellSize

	fmt.Fprintln(out, "# OOMMF OVF 2.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")

	hdr(out, "Title", meta.Name)
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")

	hdr(out, "xmin", 0)
	hdr(out, "ymin", 0)
	hdr(out, "zmin", 0)

	hdr(out, "xmax", cellsize[X]*float64(gridsize[X]))
	hdr(out, "ymax", cellsize[Y]*float64(gridsize[Y]))
	hdr(out, "zmax", cellsize[Z]*float64(gridsize[Z]))

	name := meta.Name
	var labels []interface{}
	if q.NComp() == 1 {
		labels = []interface{}{name}
	} else {
		for i := 0; i < q.NComp(); i++ {
			labels = append(labels, name+"_"+string('x'+i))
		}
	}
	hdr(out, "valuedim", q.NComp())
	hdr(out, "valuelabels", labels...) // TODO
	unit := meta.Unit
	if unit == "" {
		unit = "1"
	}
	if q.NComp() == 1 {
		hdr(out, "valueunits", unit)
	} else {
		hdr(out, "valueunits", unit, unit, unit)
	}

	// We don't really have stages
	//fmt.Fprintln(out, "# Desc: Stage simulation time: ", meta.TimeStep, " s") // TODO
	hdr(out, "Desc", "Total simulation time: ", meta.Time, " s")

	hdr(out, "xbase", cellsize[X]/2)
	hdr(out, "ybase", cellsize[Y]/2)
	hdr(out, "zbase", cellsize[Z]/2)
	hdr(out, "xnodes", gridsize[X])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[Z])
	hdr(out, "xstepsize", cellsize[X])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[Z])
	hdr(out, "End", "Header")
}

func writeOVF2HeaderFFT(out io.Writer, q *data.Slice, meta data.Meta, NxNyNz [3]int, startK, endK [3]float64, transformedAxes []string) {
	gridsize := NxNyNz
	cellsize := meta.CellSize

	fmt.Fprintln(out, "# OOMMF OVF 2.0")
	hdr(out, "Segment count", "1")
	hdr(out, "Begin", "Segment")
	hdr(out, "Begin", "Header")

	hdr(out, "Title", meta.Name)
	hdr(out, "meshtype", "rectangular")
	hdr(out, "meshunit", "m")
	if slices.Contains(transformedAxes, "x") {
		hdr(out, "k_xmin", startK[0])
	} else {
		hdr(out, "xmin", startK[0])
	}
	if slices.Contains(transformedAxes, "y") {
		hdr(out, "k_ymin", startK[1])
	} else {
		hdr(out, "ymin", startK[1])
	}
	if slices.Contains(transformedAxes, "z") {
		hdr(out, "k_zmin", startK[2])
	} else {
		hdr(out, "zmin", startK[2])
	}

	if slices.Contains(transformedAxes, "x") {
		hdr(out, "k_xmax", endK[0])
	} else {
		hdr(out, "xmax", endK[0])
	}
	if slices.Contains(transformedAxes, "y") {
		hdr(out, "k_ymax", endK[1])
	} else {
		hdr(out, "ymax", endK[1])
	}
	if slices.Contains(transformedAxes, "z") {
		hdr(out, "k_zmax", endK[2])
	} else {
		hdr(out, "zmax", endK[2])
	}

	name := meta.Name
	var labels []interface{}
	if q.NComp() == 1 {
		labels = []interface{}{name}
	} else {
		for i := 0; i < q.NComp(); i++ {
			labels = append(labels, name+"_"+string('x'+i))
		}
	}
	hdr(out, "valuedim", q.NComp())
	hdr(out, "valuelabels", labels...) // TODO
	unit := meta.Unit
	if unit == "" {
		unit = "1"
	}
	if q.NComp() == 1 {
		hdr(out, "valueunits", unit)
	} else {
		hdr(out, "valueunits", unit, unit, unit)
	}

	// We don't really have stages
	//fmt.Fprintln(out, "# Desc: Stage simulation time: ", meta.TimeStep, " s") // TODO
	hdr(out, "Desc", "Total simulation time: ", meta.Time, " s")

	hdr(out, "xbase", cellsize[X])
	hdr(out, "ybase", cellsize[Y])
	hdr(out, "zbase", cellsize[Z])
	hdr(out, "xnodes", gridsize[X])
	hdr(out, "ynodes", gridsize[Y])
	hdr(out, "znodes", gridsize[Z])
	hdr(out, "xstepsize", cellsize[X])
	hdr(out, "ystepsize", cellsize[Y])
	hdr(out, "zstepsize", cellsize[Z])
	hdr(out, "End", "Header")
}

func writeOVF2Data(out io.Writer, q *data.Slice, dataformat string) {
	canonicalFormat := ""
	switch strings.ToLower(dataformat) {
	case "text":
		canonicalFormat = "Text"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVFText(out, q)
	case "binary", "binary 4":
		canonicalFormat = "Binary 4"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVF2DataBinary4(out, q)
	case "binary 4+4":
		canonicalFormat = "Binary 4+4"
		hdr(out, "Begin", "Data "+canonicalFormat)
		writeOVF2DataBinary4(out, q)
	default:
		log.Fatalf("Illegal OMF data format: %v. Options are: Text, Binary 4, Binary 4+4", dataformat)
	}
	hdr(out, "End", "Data "+canonicalFormat)
}

func writeOVF2DataBinary4(out io.Writer, array *data.Slice) {

	//w.count(w.out.Write((*(*[1<<31 - 1]byte)(unsafe.Pointer(&list[0])))[0 : 4*len(list)])) // (shortcut)

	data := array.Tensors()
	size := array.Size()

	var bytes []byte

	// OOMMF requires this number to be first to check the format
	var controlnumber float32 = OVF_CONTROL_NUMBER_4
	bytes = (*[4]byte)(unsafe.Pointer(&controlnumber))[:]
	out.Write(bytes)

	ncomp := array.NComp()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					bytes = (*[4]byte)(unsafe.Pointer(&data[c][iz][iy][ix]))[:]
					out.Write(bytes)
				}
			}
		}
	}
}

func readOVF2DataBinary4(in io.Reader, array *data.Slice) {
	size := array.Size()
	data := array.Tensors()

	// OOMMF requires this number to be first to check the format
	controlnumber := readFloat32(in)
	if controlnumber != OVF_CONTROL_NUMBER_4 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}

	ncomp := array.NComp()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz][iy][ix] = readFloat32(in)
				}
			}
		}
	}
}

// fully read buf, panic on error
func readFull(in io.Reader, buf []byte) {
	_, err := io.ReadFull(in, buf)
	if err != nil {
		panic(err)
	}
	return
}

// read float32 in machine endianess, panic on error
func readFloat32(in io.Reader) float32 {
	var bytes4 [4]byte
	bytes := bytes4[:]
	readFull(in, bytes)
	return *((*float32)(unsafe.Pointer(&bytes4)))
}

// read float64 in machine endianess, panic on error
func readFloat64(in io.Reader) float64 {
	var bytes8 [8]byte
	bytes := bytes8[:]
	readFull(in, bytes)
	return *((*float64)(unsafe.Pointer(&bytes8)))
}

func readOVF2DataBinary8(in io.Reader, array *data.Slice) {
	size := array.Size()
	data := array.Tensors()

	// OOMMF requires this number to be first to check the format
	controlnumber := readFloat64(in)
	if controlnumber != OVF_CONTROL_NUMBER_8 {
		panic("invalid OVF2 control number: " + fmt.Sprint(controlnumber))
	}

	ncomp := array.NComp()
	for iz := 0; iz < size[Z]; iz++ {
		for iy := 0; iy < size[Y]; iy++ {
			for ix := 0; ix < size[X]; ix++ {
				for c := 0; c < ncomp; c++ {
					data[c][iz][iy][ix] = float32(readFloat64(in))
				}
			}
		}
	}
}
