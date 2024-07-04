#include <stdint.h>
#include <stdio.h>
#include "stencil.h"
#include "amul.h"

// dst[i] = grad(a[3])
extern "C" __global__ void
pointwise_grad(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
                      float* __restrict__  a,
                      float rcsx, float rcsy, float rcsz,
					  int Nx, int Ny, int Nz, 
                      uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;


    if (ix >= Nx || iy >= Ny || iz >= Nz)
    {
    	return;
    }

	int I = idx(ix, iy, iz);
    int i_;                                       // neighbor index

    float ac = a[I];               				  //center

	// get neighbour cells into x direction
	float ax_m1 = 0.0f;     // -1
	i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor if inside grid, keep 0 otherwise
	if (ix-1 >= 0 || PBCx)
	{
		ax_m1 = a[i_];
	}
    float ax_p1 = 0.0f;     // +1
	i_ = idx(lclampx(ix+1), iy, iz);                 // load neighbor if inside grid, keep 0 otherwise
	if (ix+1 >= 0 || PBCx)
	{
		ax_p1 = a[i_];
	}

    // get neighbour cells into y direction
	float ay_m1 = 0.0f;     // -1
	i_ = idx(ix, lclampy(iy-1), iz);                 // load neighbor if inside grid, keep 0 otherwise
	if (iy-1 >= 0 || PBCy)
	{
		ay_m1 = a[i_];
	}
    float ay_p1 = 0.0f;     // +1
	i_ = idx(ix, lclampy(iy+1), iz);                 // load neighbor if inside grid, keep 0 otherwise
	if (iy+1 >= 0 || PBCy)
	{
		ay_p1 = a[i_];
	}

    // get neighbour cells into z direction
	float az_m1 = 0.0f;     // -1
	i_ = idx(ix, iy, lclampz(iz-1));                 // load neighbor if inside grid, keep 0 otherwise
	if (iz-1 >= 0 || PBCz)
	{
		az_m1 = a[i_];
	}
    float az_p1 = 0.0f;     // +1
	i_ = idx(ix, iy, lclampz(iz+1));                 // load neighbor if inside grid, keep 0 otherwise
	if (iz+1 >= 0 || PBCz)
	{
		az_p1 = a[i_];
	}

	float half = 1.0f/2.0f;

	if (ax_m1 != 0.0f && ax_p1 != 0.0f) {
		dstx[I] = half * rcsx * (2.0f * ac - ax_m1 - ax_p1);
	} else if ((ax_m1 != 0.0f || ax_p1 != 0.0f) && Nx > 1) {
		dstx[I] = rcsx * (ac - ax_m1 - ax_p1);
	} else {
		dstx[I] = 0.0f;
	}

	if (ay_m1 != 0.0f && ay_p1 != 0.0f) {
		dsty[I] = half * rcsy * (2.0f * ac - ay_m1 - ay_p1);
	} else if ((ay_m1 != 0.0f || ay_p1 != 0.0f) && Ny > 1) {
		dsty[I] = rcsy * (ac - ay_m1 - ay_p1);
	} else {
		dsty[I] = 0.0f;
	}

	if (az_m1 != 0.0f && az_p1 != 0.0f) {
		dstz[I] = half * rcsz * (2.0f * ac - az_m1 - az_p1);
	} else if ((az_m1 != 0.0f || az_p1 != 0.0f) && Nz > 1) {
		dstz[I] = rcsz * (ac - az_m1 - az_p1);
	} else {
		dstz[I] = 0.0f;
	}
}

