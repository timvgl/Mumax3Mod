// cuda/me_coeffs.cu
#include "stencil.h"
#include "amul.h"
#include <stdint.h>
#include "me_coeffs.h"

// harmonic mean helper
__device__ inline float hmean2(float a, float b) {
    const float eps=1e-30f;
    return (2.f*a*b) / (a + b + eps);
}

// 4-cell harmonic mean for shear (Cao&Chen Eq.(5))
__device__ inline float hmean4(float a, float b, float c, float d) {
    const float eps=1e-30f;
    float ia = 1.f/(a+eps), ib=1.f/(b+eps), ic=1.f/(c+eps), id=1.f/(d+eps);
    float s = ia+ib+ic+id;
    return 4.f/(s + eps);
}

// ---------------------------------------------------------------------------------
// Build PM and jump-aware coefficients (3D).
// Inputs:
//   cat: category per cell center, see me_coeffs.h (0=interior,...,255=air)
//   rho_,lambda_,mu_: per-cell or uniform via *_mul
// Outputs (all pre-allocated):
//   bx,by,bz  : harmonic means of 1/rho on faces, modified per category (Eq.22–25)
//   eta**     : normal stress coefficients (Eq.21 + PM mods Eq.24/25)
//   mu~xy/yz/zx: shear coefficients on faces (Eq.21 with harmonic μ, Eq.5)
//   air_* masks to zero fields above free surface (1=air)
// ---------------------------------------------------------------------------------
extern "C" __global__ void me_build_coeffs3d(
    unsigned char* cat,          // in
    float* rho_, float rho_mul,  // in
    float* lam_, float lam_mul,  // in
    float* mu_,  float mu_mul,   // in
    // out: reciprocal densities on faces (modified)
    float* bx, float* by, float* bz,
    // out: eta tensors at centers
    float* exx1, float* exx2, float* exx3,
    float* eyy1, float* eyy2, float* eyy3,
    float* ezz1, float* ezz2, float* ezz3,
    // out: shear μ~ on faces
    float* muxy, float* muyz, float* muzx,
    // out: air masks per storage
    unsigned char* air_c, unsigned char* air_xy, unsigned char* air_yz, unsigned char* air_zx,
    unsigned char* air_vx, unsigned char* air_vy, unsigned char* air_vz,
    int Nx, int Ny, int Nz, uint8_t PBC)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if ((!PBCx && (ix<0 || ix>=Nx)) || (!PBCy && (iy<0 || iy>=Ny)) || (!PBCz && (iz<0 || iz>=Nz))) return;

    int i = hclampx(lclampx(ix));
    int j = hclampy(lclampy(iy));
    int k = hclampz(lclampz(iz));
    int I0 = ((k*Ny)+j)*Nx + i;

    unsigned char c = cat[I0];

    // material parameters at centers
    float rho = amul(rho_, rho_mul, I0);
    float lam = amul(lam_, lam_mul, I0);
    float mu  = amul(mu_,  mu_mul,  I0);

    // neighbors for face means
    int ip=((i+1<Nx)?i+1:(PBCx?0:i)), jp=((j+1<Ny)?j+1:(PBCy?0:j)), kp=((k+1<Nz)?k+1:(PBCz?0:k));
    int Iip=((k*Ny)+j)*Nx + ip, Ijp=((k*Ny)+jp)*Nx + i, Ikp=((kp*Ny)+j)*Nx + i;

    float rho_ip = amul(rho_, rho_mul, Iip);
    float rho_jp = amul(rho_, rho_mul, Ijp);
    float rho_kp = amul(rho_, rho_mul, Ikp);

    float mu_ip  = amul(mu_,  mu_mul,  Iip);
    float mu_jp  = amul(mu_,  mu_mul,  Ijp);
    float mu_kp  = amul(mu_,  mu_mul,  Ikp);

    float lam_ip = amul(lam_, lam_mul, Iip);
    float lam_jp = amul(lam_, lam_mul, Ijp);
    float lam_kp = amul(lam_, lam_mul, Ikp);

    // --- (1) interior defaults (Eq.23)
    float bx0 = hmean2(1.f/rho, 1.f/rho_ip);
    float by0 = hmean2(1.f/rho, 1.f/rho_jp);
    float bz0 = hmean2(1.f/rho, 1.f/rho_kp);

    float exx1_ = lam + 2.f*mu, exx2_ = lam, exx3_ = lam;
    float eyy1_ = lam, eyy2_ = lam + 2.f*mu, eyy3_ = lam;
    float ezz1_ = lam, ezz2_ = lam, ezz3_ = lam + 2.f*mu;

    // shear μ~ via Eq.(5) harmonic mean of the four surrounding μ (xy,yz,zx faces)
    // XY face around (i+1/2,j+1/2,k): cells (i,j,k), (i+1,j,k), (i,j+1,k), (i+1,j+1,k)
    int Ixy_ipjp = ((k*Ny)+( (j+1<Ny)? j+1: (PBCy?0:j) ))*Nx + ((i+1<Nx)? i+1: (PBCx?0:i));
    float mu_xy = hmean4(mu, mu_ip, mu_jp, amul(mu_,mu_mul,Ixy_ipjp));

    // YZ face around (i,j+1/2,k+1/2)
    int Iyz_jpkp = (( (k+1<Nz)?k+1:(PBCz?0:k) )*Ny)+((j+1<Ny)?j+1:(PBCy?0:j));
    Iyz_jpkp = Iyz_jpkp*Nx + i;
    float mu_yz = hmean4(mu, amul(mu_,mu_mul,Ijp), amul(mu_,mu_mul,Ikp), amul(mu_,mu_mul, Iyz_jpkp));

    // ZX face around (i+1/2,j,k+1/2)
    int Izx_ipkp = (( (k+1<Nz)?k+1:(PBCz?0:k) )*Ny + j)*Nx + ((i+1<Nx)?i+1:(PBCx?0:i));
    float mu_zx = hmean4(mu, mu_ip, amul(mu_,mu_mul,Ikp), amul(mu_,mu_mul, Izx_ipkp));

    // --- (2) PM modifications at discrete free surface categories (Cao&Chen Tab.2, Eqs.(24),(25))
    // We only modify the affected entries; categories not listed keep defaults (Eq.23).
    switch(c){
        case PM_H: { // horizontal top free surface
            // densities at top faces halved → reciprocal doubled for velocity faces crossing H
            by0 *= 0.5f; // ρy unchanged on H per Eq.(24): (b~y = 0.5 ρy^{-1}?) Eq.(24) uses example for VR; H in Tab.2 → ρx*0.5, ρy*0.5, ρz*1
            bx0 *= 0.5f;
            // τ_zz=0; implement via coefficients for normals
            ezz1_ = ezz2_ = ezz3_ = 0.f;
            // shear parallel to surface modified (see Tab.2): halve μ~ on faces crossing into air
            mu_yz *= 0.5f; mu_zx *= 0.5f;
        } break;
        case PM_VL: case PM_VR: case PM_VF: case PM_VB: {
            // Vertical segments: double b~ normal to surface per Eq.(24) example and Tab.2;
            // rotate-then-apply reduces finally to toggling components (Cao&Chen Sec. "generalized expression").
            // We handle uniformly by scaling the two face-recips crossing air by 2 and zeroing the mapped normal stress.
            // Use simple symmetric treatment that matches Tab.2 patterns.
            if (c==PM_VL || c==PM_VR){ // normal in x
                bx0 *= 2.f;
                // τ_xx & τ_zz mixed; for vertical pieces set τ_xx≈0 on surface cell:
                exx1_ = exx2_ = exx3_ = 0.f;
                mu_zx *= 0.5f; mu_xy *= 0.5f;
            } else { // VF/VB → normal in y
                by0 *= 2.f;
                eyy1_ = eyy2_ = eyy3_ = 0.f;
                mu_xy *= 0.5f; mu_yz *= 0.5f;
            }
        } break;
        case PM_CORNER_INNER:
        case PM_CORNER_OUTER: {
            // corners: both face-recips doubled; all normal τ set to 0; all shear μ~ halved (Eq.25)
            bx0 *= 2.f; by0 *= 2.f; bz0 *= 2.f;
            exx1_=exx2_=exx3_=0.f; eyy1_=eyy2_=eyy3_=0.f; ezz1_=ezz2_=ezz3_=0.f;
            mu_xy *= 0.5f; mu_yz *= 0.5f; mu_zx *= 0.5f;
        } break;
        case PM_AIR: {
            // above topo: set all media to zero
            bx0=by0=bz0=0.f; exx1_=exx2_=exx3_=0.f; eyy1_=eyy2_=eyy3_=0.f; ezz1_=ezz2_=ezz3_=0.f;
            mu_xy=mu_yz=mu_zx=0.f;
        } break;
        default: break; // interior
    }

    // write outputs
    bx[I0]=bx0; by[I0]=by0; bz[I0]=bz0;
    exx1[I0]=exx1_; exx2[I0]=exx2_; exx3[I0]=exx3_;
    eyy1[I0]=eyy1_; eyy2[I0]=eyy2_; eyy3[I0]=eyy3_;
    ezz1[I0]=ezz1_; ezz2[I0]=ezz2_; ezz3[I0]=ezz3_;
    muxy[I0]=mu_xy; muyz[I0]=mu_yz; muzx[I0]=mu_zx;

    // air masks per storage (1=must be zeroed by mode=2 pass)
    unsigned char a = (c==PM_AIR)?1:0;
    if (air_c)  air_c[I0]=a;
    if (air_xy) air_xy[I0]=a;
    if (air_yz) air_yz[I0]=a;
    if (air_zx) air_zx[I0]=a;
    if (air_vx) air_vx[I0]=a;
    if (air_vy) air_vy[I0]=a;
    if (air_vz) air_vz[I0]=a;
}

// 2D variant (xz-plane). Same semantics, Ny may be 1.
extern "C" __global__ void me_build_coeffs2d(
    unsigned char* cat,
    float* rho_, float rho_mul,
    float* lam_, float lam_mul,
    float* mu_,  float mu_mul,
    float* bx, float* by, float* bz,
    float* exx1, float* exx2, float* exx3,
    float* eyy1, float* eyy2, float* eyy3,
    float* ezz1, float* ezz2, float* ezz3,
    float* muxy, float* muyz, float* muzx,
    unsigned char* air_c, unsigned char* air_xy, unsigned char* air_yz, unsigned char* air_zx,
    unsigned char* air_vx, unsigned char* air_vy, unsigned char* air_vz,
    int Nx, int Ny, int Nz, uint8_t PBC)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int iz = blockIdx.z*blockDim.z + threadIdx.z;
    int i = hclampx(lclampx(ix));
    int k = hclampy(lclampy(iy));
    int j = hclampz(lclampz(iz));
    int j = 0;
    if ((!PBCx && (ix<0 || ix>=Nx)) || (!PBCy && (iy<0 || iy>=Ny)) || !PBCz && (iz < 0 || iz>= Nz)) return;
    int I0 = idx(i,k, );

    unsigned char c = cat[I0];
    float rho = amul(rho_, rho_mul, I0);
    float lam = amul(lam_, lam_mul, I0);
    float mu0 = amul(mu_,  mu_mul,  I0);

    int ip=(i+1<Nx)?i+1:(PBCx?0:i);
    int kp=(k+1<Nz)?k+1:(PBCz?0:k);
    int Iip = Id(ip,k), Ikp = Id(i,kp);

    float bx0 = hmean2(1.f/rho, 1.f/amul(rho_,rho_mul,Iip));
    float bz0 = hmean2(1.f/rho, 1.f/amul(rho_,rho_mul,Ikp));

    float exx1_=lam+2.f*mu0, exx2_=lam, exx3_=lam;
    float ezz1_=lam, ezz2_=lam, ezz3_=lam+2.f*mu0;

    // μ~_zx (shear) around (i+1/2,k+1/2)
    float mu_ip = amul(mu_,mu_mul,Iip), mu_kp = amul(mu_,mu_mul,Ikp);
    int Iipkp = Id(ip,kp);
    float mu_zx = hmean4(mu0, mu_ip, mu_kp, amul(mu_,mu_mul,Iipkp));

    // PM mods (subset of 3D)
    switch(c){
        case PM_H:
            bx0 *= 0.5f;        // halve for faces crossing into air along x at the top layer
            ezz1_=ezz2_=ezz3_=0.f;
            mu_zx *= 0.5f;
            break;
        case PM_VL: case PM_VR:
            bx0 *= 2.f;
            exx1_=exx2_=exx3_=0.f;
            mu_zx *= 0.5f;
            break;
        case PM_CORNER_INNER:
        case PM_CORNER_OUTER:
            bx0*=2.f; bz0*=2.f;
            exx1_=exx2_=exx3_=0.f; ezz1_=ezz2_=ezz3_=0.f;
            mu_zx*=0.5f;
            break;
        case PM_AIR:
            bx0=bz0=0.f; exx1_=exx2_=exx3_=0.f; ezz1_=ezz2_=ezz3_=0.f; mu_zx=0.f;
            break;
        default: break;
    }

    // y-quantities unused in xz-plane
    float by0 = 0.f; float dummy=0.f;

    bx[I0]=bx0; by[I0]=by0; bz[I0]=bz0;
    exx1[I0]=exx1_; exx2[I0]=exx2_; exx3[I0]=exx3_;
    eyy1[I0]=0.f;   eyy2[I0]=0.f;   eyy3[I0]=0.f;
    ezz1[I0]=ezz1_; ezz2[I0]=ezz2_; ezz3[I0]=ezz3_;
    muxy[I0]=0.f;   muyz[I0]=0.f;   muzx[I0]=mu_zx;

    unsigned char a=(c==PM_AIR)?1:0;
    if (air_c)  air_c[I0]=a;
    if (air_xy) air_xy[I0]=a;
    if (air_yz) air_yz[I0]=a;
    if (air_zx) air_zx[I0]=a;
    if (air_vx) air_vx[I0]=a;
    if (air_vy) air_vy[I0]=a;
    if (air_vz) air_vz[I0]=a;
}
