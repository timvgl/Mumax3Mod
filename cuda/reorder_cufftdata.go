package cuda

import "github.com/mumax/3/data"

func ReorderCufftData(output, input *data.Slice, mesh *data.Mesh) {
	NxNyNz := mesh.Size()
	cfg := make3DConf(NxNyNz)
	if input.NComp() == 3 {
		for c := range 3 {
			k_fftshift3D_partial_async(output.DevPtr(c), input.DevPtr(c), NxNyNz[0], NxNyNz[1], NxNyNz[2], cfg)
		}
	}
}
