package cuda

import (
	"fmt"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func CopyPart(dst, src *data.Slice,
	xStart_src, xEnd_src,
	yStart_src, yEnd_src,
	zStart_src, zEnd_src,
	fStart_src, fEnd_src,
	xStart_dst, yStart_dst,
	zStart_dst, fStart_dst int) {
	util.AssertMsg(dst.NComp() == src.NComp(), fmt.Sprintf("slice copy: illegal sizes: dst: %vx%v, src: %vx%v", dst.NComp(), dst.Len(), src.NComp(), src.Len()))
	srcSize := src.Size()
	dstSize := dst.Size()
	cfg := make1DConf(prod([3]int{xEnd_src - xStart_src, yEnd_src - yStart_src, zEnd_src - zStart_src}) * (fEnd_src - fStart_src))
	for c := range dst.NComp() {
		k_CopyPartKernel_async(dst.DevPtr(c), src.DevPtr(c),
			xStart_src, yStart_src, zStart_src, fStart_src,
			xEnd_src-xStart_src, yEnd_src-yStart_src, zEnd_src-zStart_src, fEnd_src-fStart_src,
			xStart_dst, yStart_dst, zStart_dst, fStart_dst,
			srcSize[X], srcSize[Y], srcSize[Z],
			dstSize[X], dstSize[Y], dstSize[Z], cfg)
	}
}
