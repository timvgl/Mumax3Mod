#pragma once
#include "stencil.h"
__device__ __forceinline__ int ID3(int ix,int iy,int iz, int Nx,int Ny,int Nz, uint8_t PBC){
    int cx = hclampx(lclampx(ix));
    int cy = hclampy(lclampy(iy));
    int cz = hclampz(lclampz(iz));
    return idx(cx,cy,cz);
}
__device__ __forceinline__
float d1_central_5pt(const float* u, int I_m2,int I_m1,int I_p1,int I_p2, float h){
    float um2=u[I_m2], um1=u[I_m1], up1=u[I_p1], up2=u[I_p2];
    return (-up2 + 8.f*up1 - 8.f*um1 + um2) / (12.f*h);
}
// Einseitig links (i=0)
__device__ __forceinline__
float d1_one_sided_left_4(const float* u, int I0,int I1,int I2,int I3,int I4, float h){
    return (-25.f*u[I0] + 48.f*u[I1] - 36.f*u[I2] + 16.f*u[I3] - 3.f*u[I4]) / (12.f*h);
}
// Einseitig rechts (i=N-1)
__device__ __forceinline__
float d1_one_sided_right_4(const float* u, int I0,int I1,int I2,int I3,int I4, float h){
    return ( 25.f*u[I0] - 48.f*u[I1] + 36.f*u[I2] - 16.f*u[I3] + 3.f*u[I4]) / (12.f*h);
}

// ========== 1D-Ableitung mit PBC/Clamping + Randfallbehandlung ==============
// axis: 0=x, 1=y, 2=z
// HookLeft/HookRight: optional (0 -> ignorieren)
__device__ __forceinline__
float deriv_axis(
    const float* __restrict__ u,
    int i,int j,int k,
    int Nx,int Ny,int Nz,
    uint8_t PBC,
    float h, int axis,
    const float* __restrict__ HookLeft,
    const float* __restrict__ HookRight)
{
    const bool pbcAxis = (axis==0 ? PBCx : (axis==1 ? PBCy : PBCz));
    if (pbcAxis) {
        if (axis==0) return d1_central_5pt(u, ID3(i-2,j,k,Nx,Ny,Nz,PBC), ID3(i-1,j,k,Nx,Ny,Nz,PBC),
                                            ID3(i+1,j,k,Nx,Ny,Nz,PBC), ID3(i+2,j,k,Nx,Ny,Nz,PBC), h);
        if (axis==1) return d1_central_5pt(u, ID3(i,j-2,k,Nx,Ny,Nz,PBC), ID3(i,j-1,k,Nx,Ny,Nz,PBC),
                                            ID3(i,j+1,k,Nx,Ny,Nz,PBC), ID3(i,j+2,k,Nx,Ny,Nz,PBC), h);
        return d1_central_5pt(u, ID3(i,j,k-2,Nx,Ny,Nz,PBC), ID3(i,j,k-1,Nx,Ny,Nz,PBC),
                                ID3(i,j,k+1,Nx,Ny,Nz,PBC), ID3(i,j,k+2,Nx,Ny,Nz,PBC), h);
    }
    // inside sample (≥2 from edge) -> central 5-point
    if ((axis==0 && i>=2 && i<=Nx-3) ||
        (axis==1 && j>=2 && j<=Ny-3) ||
        (axis==2 && k>=2 && k<=Nz-3)) {
        if(axis==0) return d1_central_5pt(u, ID3(i-2,j,k,Nx,Ny,Nz,PBC), ID3(i-1,j,k,Nx,Ny,Nz,PBC),
                                             ID3(i+1,j,k,Nx,Ny,Nz,PBC), ID3(i+2,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_central_5pt(u, ID3(i,j-2,k,Nx,Ny,Nz,PBC), ID3(i,j-1,k,Nx,Ny,Nz,PBC),
                                             ID3(i,j+1,k,Nx,Ny,Nz,PBC), ID3(i,j+2,k,Nx,Ny,Nz,PBC), h);
        return d1_central_5pt(u, ID3(i,j,k-2,Nx,Ny,Nz,PBC), ID3(i,j,k-1,Nx,Ny,Nz,PBC),
                                 ID3(i,j,k+1,Nx,Ny,Nz,PBC), ID3(i,j,k+2,Nx,Ny,Nz,PBC), h);
    }
    // edge case: at the boundary of index being zero in at least one of the three directions
    if((axis==0 && i==0)||(axis==1 && j==0)||(axis==2 && k==0)){
        if(HookLeft) return HookLeft[idx(i,j,k)];
        if(axis==0) return d1_one_sided_left_4 (u, ID3(0,j,k,Nx,Ny,Nz,PBC), ID3(1,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(2,j,k,Nx,Ny,Nz,PBC), ID3(3,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(4,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_one_sided_left_4 (u, ID3(i,0,k,Nx,Ny,Nz,PBC), ID3(i,1,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,2,k,Nx,Ny,Nz,PBC), ID3(i,3,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,4,k,Nx,Ny,Nz,PBC), h);
        return             d1_one_sided_left_4 (u, ID3(i,j,0,Nx,Ny,Nz,PBC), ID3(i,j,1,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,2,Nx,Ny,Nz,PBC), ID3(i,j,3,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,4,Nx,Ny,Nz,PBC), h);
    }
    if((axis==0 && i==1)||(axis==1 && j==1)||(axis==2 && k==1)){
        if(axis==0){
            // symmetrisch zu i==Nx-2
            float u0=u[ID3(0,j,k,Nx,Ny,Nz,PBC)], u1=u[ID3(1,j,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(2,j,k,Nx,Ny,Nz,PBC)], u3=u[ID3(3,j,k,Nx,Ny,Nz,PBC)], u4=u[ID3(4,j,k,Nx,Ny,Nz,PBC)];
            return (-3.f*u0 -10.f*u1 +18.f*u2 -6.f*u3 +1.f*u4)/(12.f*h);
        }
        if(axis==1){
            float u0=u[ID3(i,0,k,Nx,Ny,Nz,PBC)], u1=u[ID3(i,1,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(i,2,k,Nx,Ny,Nz,PBC)], u3=u[ID3(i,3,k,Nx,Ny,Nz,PBC)], u4=u[ID3(i,4,k,Nx,Ny,Nz,PBC)];
            return (-3.f*u0 -10.f*u1 +18.f*u2 -6.f*u3 +1.f*u4)/(12.f*h);
        }
        // axis==2: bei Nz==1 sowieso thin → hier nie hinein
    }
    // edge case: one cell before the boundary of index being max in at least one of the three directions
    if((axis==0 && i==Nx-2)||(axis==1 && j==Ny-2)||(axis==2 && k==Nz-2)){
        if(axis==0){
            float u0=u[ID3(Nx-1,j,k,Nx,Ny,Nz,PBC)], u1=u[ID3(Nx-2,j,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(Nx-3,j,k,Nx,Ny,Nz,PBC)], u3=u[ID3(Nx-4,j,k,Nx,Ny,Nz,PBC)], u4=u[ID3(Nx-5,j,k,Nx,Ny,Nz,PBC)];
            return (3.f*u0 +10.f*u1 -18.f*u2 +6.f*u3 -1.f*u4)/(12.f*h);
        }
        if(axis==1){
            float u0=u[ID3(i,Ny-1,k,Nx,Ny,Nz,PBC)], u1=u[ID3(i,Ny-2,k,Nx,Ny,Nz,PBC)];
            float u2=u[ID3(i,Ny-3,k,Nx,Ny,Nz,PBC)], u3=u[ID3(i,Ny-4,k,Nx,Ny,Nz,PBC)], u4=u[ID3(i,Ny-5,k,Nx,Ny,Nz,PBC)];
            return (3.f*u0 +10.f*u1 -18.f*u2 +6.f*u3 -1.f*u4)/(12.f*h);
        }
        float u0=u[ID3(i,j,Nz-1,Nx,Ny,Nz,PBC)], u1=u[ID3(i,j,Nz-2,Nx,Ny,Nz,PBC)];
        float u2=u[ID3(i,j,Nz-3,Nx,Ny,Nz,PBC)], u3=u[ID3(i,j,Nz-4,Nx,Ny,Nz,PBC)], u4=u[ID3(i,j,Nz-5,Nx,Ny,Nz,PBC)];
        return (3.f*u0 +10.f*u1 -18.f*u2 +6.f*u3 -1.f*u4)/(12.f*h);
    }
    // edge case: at the boundary of index being max in at least one of the three directions
    if((axis==0 && i==Nx-1)||(axis==1 && j==Ny-1)||(axis==2 && k==Nz-1)){
        if(HookRight) return HookRight[idx(i,j,k)];
        if(axis==0) return d1_one_sided_right_4(u, ID3(Nx-1,j,k,Nx,Ny,Nz,PBC), ID3(Nx-2,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(Nx-3,j,k,Nx,Ny,Nz,PBC), ID3(Nx-4,j,k,Nx,Ny,Nz,PBC),
                                                   ID3(Nx-5,j,k,Nx,Ny,Nz,PBC), h);
        if(axis==1) return d1_one_sided_right_4(u, ID3(i,Ny-1,k,Nx,Ny,Nz,PBC), ID3(i,Ny-2,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,Ny-3,k,Nx,Ny,Nz,PBC), ID3(i,Ny-4,k,Nx,Ny,Nz,PBC),
                                                   ID3(i,Ny-5,k,Nx,Ny,Nz,PBC), h);
        return          d1_one_sided_right_4(u, ID3(i,j,Nz-1,Nx,Ny,Nz,PBC), ID3(i,j,Nz-2,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,Nz-3,Nx,Ny,Nz,PBC), ID3(i,j,Nz-4,Nx,Ny,Nz,PBC),
                                                   ID3(i,j,Nz-5,Nx,Ny,Nz,PBC), h);
    }
    printf("Error in deriv_axis: should never reach here!\n");
    // Fallback (2. Ordnung)
    if(axis==0) return (u[ID3(i+1,j,k,Nx,Ny,Nz,PBC)] - u[ID3(i-1,j,k,Nx,Ny,Nz,PBC)])/(2.f*h);
    if(axis==1) return (u[ID3(i,j+1,k,Nx,Ny,Nz,PBC)] - u[ID3(i,j-1,k,Nx,Ny,Nz,PBC)])/(2.f*h);
    return          (u[ID3(i,j,k+1,Nx,Ny,Nz,PBC)] - u[ID3(i,j,k-1,Nx,Ny,Nz,PBC)])/(2.f*h);
}