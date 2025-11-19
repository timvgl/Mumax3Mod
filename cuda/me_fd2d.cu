// cuda/me_fd2d.cu
#include "stencil.h"
#include "amul.h"
#include "float3.h"
#include <stdint.h>

// 2D (x-z plane). Ny may be 1. Same mode semantics as in 3D.
extern "C" __global__ void me_fd2d(
    float* vx, float* vy, float* vz,
    float* sxx, float* syy, float* szz,
    float* sxy, float* syz, float* szx,
    float* mx, float* my, float* mz,
    float* bx_, float bx_mul,
    float* by_, float by_mul,
    float* bz_, float bz_mul,
    float* etaxx1_, float etaxx1_mul,
    float* etaxx2_, float etaxx2_mul,
    float* etaxx3_, float etaxx3_mul,
    float* etayy1_, float etayy1_mul,
    float* etayy2_, float etayy2_mul,
    float* etayy3_, float etayy3_mul,
    float* etazz1_, float etazz1_mul,
    float* etazz2_, float etazz2_mul,
    float* etazz3_, float etazz3_mul,
    float* muxy_, float muxy_mul,
    float* muyz_, float muyz_mul,
    float* muzx_, float muzx_mul,
    float* C11_, float C11_mul,
    float* C12_, float C12_mul,
    float* C13_, float C13_mul,
    float* C33_, float C33_mul,
    float* C44_, float C44_mul,
    float* B1_,  float B1_mul,
    float* B2_,  float B2_mul,
    unsigned char* air_c,
    unsigned char* air_xy,
    unsigned char* air_yz,
    unsigned char* air_zx,
    unsigned char* air_vx,
    unsigned char* air_vy,
    unsigned char* air_vz,
    float dx, float dy, float dz,
    float dt,
    int Nx, int Ny, int Nz,
    uint8_t PBC,
    int mode)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    int iy = 0; // xz-plane
    if ((!PBCx && (ix < 0 || ix >= Nx)) ||
        (!PBCz && (iz < 0 || iz >= Nz))) return;

    int i = hclampx(lclampx(ix));
    int j = 0;
    int k = hclampz(lclampz(iz));

    auto Id = [&](int x,int z){ return ((z*Ny)+j)*Nx + x; };
    int I0 = Id(i,k);

    // magnetoelastic center stress from B1,B2,m
    float B1 = amul(B1_, B1_mul, I0);
    float B2 = amul(B2_, B2_mul, I0);
    float mx0 = mx ? mx[I0] : 0.f, my0 = my ? my[I0] : 0.f, mz0 = mz ? mz[I0] : 0.f;
    float sv_xx = B1 * mx0*mx0;
    float sv_zz = B1 * mz0*mz0;
    float sv_xz = 2.f * B2 * (mx0*mz0);

    if (mode == 2) {
        if (air_c && air_c[I0]) { if (sxx) sxx[I0]=0.f; if (syy) syy[I0]=0.f; if (szz) szz[I0]=0.f; }
        if (air_zx && air_zx[I0]) { if (szx) szx[I0]=0.f; } // τ_xz = τ_zx; we use szx slot
        if (air_vx && air_vx[I0]) { if (vx)  vx[I0]=0.f; }
        if (air_vz && air_vz[I0]) { if (vz)  vz[I0]=0.f; }
        return;
    }

    int ip = hclampx(i+1), im = lclampx(i-1);
    int kp = hclampz(k+1), km = lclampz(k-1);
    int Iip = Id(ip,k), Iim = Id(im,k), Ikp = Id(i,kp), Ikm = Id(i,km);

    if (mode == 0) {
        // velocities
        if (vx) {
            float dtxx_dx = ((sxx ? sxx[Iip] : 0.f) - (sxx ? sxx[I0] : 0.f)) / dx;
            float dszx_dz = ((szx ? szx[Ikp] : 0.f) - (szx ? szx[Ikm] : 0.f)) / (2.f*dz);
            float bx = amul(bx_, bx_mul, I0);
            if (!(air_vx && air_vx[I0])) vx[I0] += dt * bx * ((dtxx_dx - sv_xx/dx) + dszx_dz);
        }
        if (vz) {
            float dszx_dx = ((szx ? szx[Iip] : 0.f) - (szx ? szx[Iim] : 0.f)) / (2.f*dx);
            float dtzz_dz = ((szz ? szz[Ikp] : 0.f) - (szz ? szz[I0] : 0.f)) / dz;
            float bz = amul(bz_, bz_mul, I0);
            if (!(air_vz && air_vz[I0])) vz[I0] += dt * bz * (dszx_dx + (dtzz_dz - sv_zz/dz));
        }
        return;
    }

    if (mode == 1) {
        // stress update (normal at centers, τ_xz on zx slot)
        float dvx_dx = (vx ? (vx[I0] - vx[Iim])/dx : 0.f);
        float dvz_dz = (vz ? (vz[I0] - vz[Ikm])/dz : 0.f);

        if (sxx && !(air_c && air_c[I0])) {
            float e1 = amul(etaxx1_, etaxx1_mul, I0);
            float e2 = amul(etaxx2_, etaxx2_mul, I0);
            float e3 = amul(etaxx3_, etaxx3_mul, I0);
            sxx[I0] += dt * (e1*dvx_dx + e2*0.f + e3*dvz_dz);
        }
        if (szz && !(air_c && air_c[I0])) {
            float e1 = amul(etazz1_, etazz1_mul, I0);
            float e2 = amul(etazz2_, etazz2_mul, I0);
            float e3 = amul(etazz3_, etazz3_mul, I0);
            szz[I0] += dt * (e1*dvx_dx + e2*0.f + e3*dvz_dz);
        }
        if (szx && !(air_zx && air_zx[I0])) {
            float mu = amul(muzx_, muzx_mul, I0);
            float dvz_dx = (vz ? (vz[Iip]-vz[Iim])/(2.f*dx) : 0.f);
            float dvx_dz = (vx ? (vx[Ikp]-vx[Ikm])/(2.f*dz) : 0.f);
            szx[I0] += dt * mu * (dvz_dx + dvx_dz);
        }
        return;
    }
}
