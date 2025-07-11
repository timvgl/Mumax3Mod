#include "stencil.h"
#include <cuComplex.h>

// Berechnet exp(z) für komplexe z = z.x + i*z.y.
__device__ __forceinline__ cuDoubleComplex my_cexp(cuDoubleComplex z) {
    cuDoubleComplex res;
    double t = expf(z.x);
    sincos(z.y, &res.y, &res.x); // berechnet cos(z.y) -> res.x und sin(z.y) -> res.y
    res.x *= t;
    res.y *= t;
    return res;
}

// Kernel zur inkrementellen NUDFT-Update.
// Parameter:
//  - sum: 4D-Array (Raum: (Nx/2, Ny, Nz), Frequenz: Nf) – gespeicherte Summe, komplex (interleaved: Real, Imag)
//  - src: 3D-Array mit neuen Daten, komplex (interleaved), Dimensionen: (Nx/2, Ny, Nz)
//  - Nx: Länge der x-Achse in Floats (also 2*(Anzahl komplexer Elemente in x))
//  - Ny, Nz: Anzahl der Elemente in y und z (in komplexen Elementen)
//  - Nf: Anzahl der Frequenzindizes, für die die NUDFT evaluiert wird
//  - minF, dF: Parameter zur Berechnung der Frequenz (f = minF + dF * fi)
//  - t: Zeitpunkt bzw. Zeitparameter des aktuellen Datenblocks
extern "C" __global__ void
FFT_Step_MEM_Complex(
    float* __restrict__ sum,   // 4D Summe (in interleaved Complex)
    float* __restrict__ src,   // 3D neue Daten (in interleaved Complex)
    int Nx, int Ny, int Nz,     // Nx: Länge in Floats, also x enthält Nx/2 komplexe Werte
    int Nf,
    float minF, float dF, float t)
{
    // Berechne die Anzahl komplexer Elemente in x.
    const double pi = 3.14159265358979323846;
    int Nx_c = Nx / 2;
    
    // Berechne die 3D-Raumkoordinaten (für das 3D-Array src)
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // x als Index in komplexen Elementen
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= Nx_c || y >= Ny || z >= Nz)
        return;
    
    // Berechne den linearen Index für src (3D-Array) – interleaved komplex:
    // Hier liefert das Makro idx(2*x, y, z) den Index für den Realteil,
    // da Nx in idx() die Länge in Floats ist (also 2*Nx_c).
    int srcIndex = idx(2 * x, y, z);  // Realteil an srcIndex, Imag an srcIndex+1
    cuDoubleComplex newData = make_cuDoubleComplex((double)src[srcIndex], (double)src[srcIndex + 1]);
    // Aktualisiere für jeden Frequenzindex die Summe:
    for (int fi = 0; fi < Nf; fi++) {
        // Berechne den Phasenwinkel für diese Frequenz:
        double phase = -2.0 * pi * ((double)minF + (double)dF * (double)fi) * (double)t;
        cuDoubleComplex expVal = my_cexp(make_cuDoubleComplex(0.0, phase));
        
        // Komplexe Multiplikation: Beitrag = newData * expVal
        cuDoubleComplex contribution;
        contribution.x = newData.x * expVal.x - newData.y * expVal.y;
        contribution.y = newData.x * expVal.y + newData.y * expVal.x;
        
        // Berechne den 4D-Index für die Summe:
        // Das 4D-Array hat Dimensionen: (Nx/2, Ny, Nz, Nf) in komplexen Elementen,
        // wobei die erste Dimension (x) in Floats als 2*(Nx/2) gespeichert wird.
        // Wir verwenden idx4D, wobei wir als x-Wert wieder 2*x verwenden, um den Realteil zu adressieren.
        int sumIndex = idx4D(2 * x, y, z, fi);  // Realteil
        // Akkumulieren: addiere den Beitrag zur vorhandenen Summe.
        sum[sumIndex]     += (float)contribution.x;
        sum[sumIndex + 1] += (float)contribution.y;
    }
}
