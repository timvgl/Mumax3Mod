#include <stdint.h>
#include <stdio.h>
#include "float3.h"
#include "stencil.h"

// Calculate force from strain
extern "C" __global__ void
getelasticforce(float* __restrict__  fx, float* __restrict__  fy, float* __restrict__  fz,
					  float* __restrict__ sxx, float* __restrict__ syy, float* __restrict__ szz,
                 	  float* __restrict__ sxy, float* __restrict__ syz, float* __restrict__ sxz,
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
	float3 norm = make_float3(sxx[I], syy[I], szz[I]); // +0
	float3 shear = make_float3(sxy[I], syz[I], sxz[I]); // +0
    int i_;                                       // neighbor index


	// get neighbour cells into x direction
	float3 x_norm_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
	i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor norm strain if inside grid, keep 0 otherwise
	if (ix-1 >= 0 || PBCx)
	{
		x_norm_m1 = make_float3(sxx[i_], syy[i_], szz[i_]);
	}

	float3 x_norm_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
	i_ = idx(lclampx(ix+1), iy, iz);                 // load neighbor norm strain if inside grid, keep 0 otherwise
	if (ix+1 >= 0 || PBCx)
	{
		x_norm_p1 = make_float3(sxx[i_], syy[i_], szz[i_]);
	}

	float3 x_shear_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
	i_ = idx(lclampx(ix-1), iy, iz);                 // load neighbor shear strain if inside grid, keep 0 otherwise
	if (ix-1 >= 0 || PBCx)
	{
		x_shear_m1 = make_float3(sxy[i_], syz[i_], sxz[i_]);
	}

	float3 x_shear_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
	i_ = idx(lclampx(ix+1), iy, iz);                 // load neighbor shear strain if inside grid, keep 0 otherwise
	if (ix+1 >= 0 || PBCx)
	{
		x_shear_p1 = make_float3(sxy[i_], syz[i_], sxz[i_]);
	}


	// get neighbour cells into y direction
	float3 y_norm_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
	i_ = idx(ix, lclampx(iy-1), iz);                 // load neighbor norm strain if inside grid, keep 0 otherwise
	if (iy-1 >= 0 || PBCy)
	{
		y_norm_m1 = make_float3(sxx[i_], syy[i_], szz[i_]);
	}

	float3 y_norm_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
	i_ = idx(ix, lclampx(iy+1), iz);                 // load neighbor norm strain if inside grid, keep 0 otherwise
	if (iy+1 >= 0 || PBCy)
	{
		y_norm_p1 = make_float3(sxx[i_], syy[i_], szz[i_]);
	}

	float3 y_shear_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
	i_ = idx(ix, lclampx(iy-1), iz);                 // load neighbor shear strain if inside grid, keep 0 otherwise
	if (iy-1 >= 0 || PBCy)
	{
		y_shear_m1 = make_float3(sxy[i_], syz[i_], sxz[i_]);
	}

	float3 y_shear_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
	i_ = idx(ix, lclampx(iy+1), iz);                 // load neighbor shear strain if inside grid, keep 0 otherwise
	if (iy+1 >= 0 || PBCy)
	{
		y_shear_p1 = make_float3(sxy[i_], syz[i_], sxz[i_]);
	}

	
	// get neighbour cells into z direction
	float3 z_norm_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
	i_ = idx(ix, iy, lclampx(iz-1));                 // load neighbor norm strain if inside grid, keep 0 otherwise
	if (iz-1 >= 0 || PBCz)
	{
		z_norm_m1 = make_float3(sxx[i_], syy[i_], szz[i_]);
	}

	float3 z_norm_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
	i_ = idx(ix, iy, lclampx(iz+1));                // load neighbor norm strain if inside grid, keep 0 otherwise
	if (iz+1 >= 0 || PBCz)
	{
		z_norm_p1 = make_float3(sxx[i_], syy[i_], szz[i_]);
	}

	float3 z_shear_m1 = make_float3(0.0f, 0.0f, 0.0f);     // -1
	i_ = idx(ix, iy, lclampx(iz-1));                // load neighbor shear strain if inside grid, keep 0 otherwise
	if (iz-1 >= 0 || PBCz)
	{
		z_shear_m1 = make_float3(sxy[i_], syz[i_], sxz[i_]);
	}

	float3 z_shear_p1 = make_float3(0.0f, 0.0f, 0.0f);     // +1
	i_ = idx(ix, iy, lclampx(iz+1));                // load neighbor shear strain if inside grid, keep 0 otherwise
	if (iz+1 >= 0 || PBCz)
	{
		z_shear_p1 = make_float3(sxy[i_], syz[i_], sxz[i_]);
	}


	float3 deltaNormX = make_float3(0.0f, 0.0f, 0.0f); 
	float3 deltaNormY = make_float3(0.0f, 0.0f, 0.0f);
	float3 deltaNormZ = make_float3(0.0f, 0.0f, 0.0f); 

	deltaNormX = 2*norm -x_norm_m1 -x_norm_p1;
	deltaNormY = 2*norm -y_norm_m1 -y_norm_p1;
	deltaNormZ = 2*norm -z_norm_m1 -z_norm_p1;

	float3 deltaShearX = make_float3(0.0f, 0.0f, 0.0f); 
	float3 deltaShearY = make_float3(0.0f, 0.0f, 0.0f);
	float3 deltaShearZ = make_float3(0.0f, 0.0f, 0.0f); 

	deltaShearX = 2.0f*shear -x_shear_m1 -x_shear_p1;
	deltaShearY = 2.0f*shear -y_shear_m1 -y_shear_p1;
	deltaShearZ = 2.0f*shear -z_shear_m1 -z_shear_p1;

	float half = 1.0f/2.0f;

	fx[I] = half * deltaNormX.x * rcsx + half * deltaShearX.x * rcsy + half * deltaShearX.z * rcsz;
	fy[I] = half * deltaNormY.y * rcsy + half * deltaShearX.x * rcsx + half * deltaShearZ.y * rcsz;
	fz[I] = half * deltaNormZ.z * rcsz + half * deltaShearY.y * rcsy + half * deltaShearX.z * rcsx;
}
