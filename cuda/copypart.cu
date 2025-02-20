extern "C" __global__ void CopyPartKernel(
    float* dst,   // Quellarray im Device
    float* src,   // Zielarray im Device
    // Startindizes im src-Array:
    int xStart_src, int yStart_src, int zStart_src, int fStart_src,
    // Anzahl der zu kopierenden Elemente pro Dimension:
    int xCount, int yCount, int zCount, int fCount,
    // Startindizes im dst-Array:
    int xStart_dst, int yStart_dst, int zStart_dst, int fStart_dst,
    // Gesamtdimensionen des src-Arrays (nur x, y, z – f wird implizit über das Produkt berechnet):
    int src_dim_x, int src_dim_y, int src_dim_z,
    // Gesamtdimensionen des dst-Arrays:
    int dst_dim_x, int dst_dim_y, int dst_dim_z)
{
    // Gesamtzahl der zu kopierenden Elemente (über alle Dimensionen)
    int total = fCount * zCount * yCount * xCount;
    int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    
    // Jeder Thread kopiert (über einen Grid-stride Loop) mehrere Elemente
    if (idx < total) {
        // Bestimme die 4D-Koordinaten (relativ zum kopierten Block)
        int temp = idx;
        int x = temp % xCount;        // x-Index im Block
        temp /= xCount;
        int y = temp % yCount;        // y-Index im Block
        temp /= yCount;
        int z = temp % zCount;        // z-Index im Block
        temp /= zCount;
        int f = temp;                 // f-Index im Block

        // Berechne den Offset im Quellarray
        int src_index = ((fStart_src + f) * src_dim_z * src_dim_y * src_dim_x) +
                        ((zStart_src + z) * src_dim_y * src_dim_x) +
                        ((yStart_src + y) * src_dim_x) +
                        (xStart_src + x);
        
        // Berechne den Offset im Zielarray
        int dst_index = ((fStart_dst + f) * dst_dim_z * dst_dim_y * dst_dim_x) +
                        ((zStart_dst + z) * dst_dim_y * dst_dim_x) +
                        ((yStart_dst + y) * dst_dim_x) +
                        (xStart_dst + x);

        // Kopiere das Element
        dst[dst_index] = src[src_index];
    }
}