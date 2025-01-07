/******************************************************************************
 * addCompressedArraysKernel
 *
 * Merges two compressed float arrays (arr1, arr2) in 5-byte-run format:
 *   [marker][4-byte runLength or floatBits]
 *
 * Produces sum in a new compressed array "outArr".
 *   - Single-thread approach (blockDim.x=1, gridDim.x=1).
 *   - Writes final size to *d_outSize.
 *
 * Markers:
 *   0x00 => zero run, next 4 bytes = runLength
 *   0xFF => one non-zero float, next 4 bytes = float bits
 *
 * Summation rules:
 *   zero + zero = zero
 *   zero + non-zero = that non-zero
 *   non-zero + zero = that non-zero
 *   non-zero + non-zero = sum of the two floats
 ******************************************************************************/

#include <cuda_runtime.h>
#include <stdint.h>

// Helper for reading a run from compressed data
// Returns: (marker, runLength or floatVal), plus how many elements remain in that run
__device__ __inline__
void parseRun(const uint8_t* arr, size_t arrSize, size_t pos,
              uint8_t &marker, float &floatVal, unsigned int &runCount)
{
    // Out-of-bounds or invalid => set defaults
    if (pos + 5 > arrSize) {
        marker = 0x00;
        runCount = 0;
        floatVal = 0.0f;
        return;
    }

    marker = arr[pos];
    if (marker == 0x00) {
        // Zero run => next 4 bytes = run length
        runCount = (arr[pos+1]) 
                 | (arr[pos+2] << 8)
                 | (arr[pos+3] << 16)
                 | (arr[pos+4] << 24);
        floatVal = 0.0f; // zero
    }
    else if (marker == 0xFF) {
        // Non-zero => next 4 bytes = float bits
        union {
            uint8_t b[4];
            float   f;
        } u;
        u.b[0] = arr[pos+1];
        u.b[1] = arr[pos+2];
        u.b[2] = arr[pos+3];
        u.b[3] = arr[pos+4];
        floatVal = u.f;
        runCount = 1;  // exactly 1 float
    }
    else {
        // Unknown => treat as zero run of length 0
        marker = 0x00;
        runCount = 0;
        floatVal = 0.0f;
    }
}

// Writes a 5-byte run to outArr at outPos
//   If marker=0x00 => 4 bytes after are runLength
//   If marker=0xFF => 4 bytes after are float bits
__device__ __inline__
void writeRun(uint8_t* outArr, size_t outSize, size_t outPos,
              uint8_t marker, float floatVal, unsigned int runCount)
{
    if (outPos + 5 > outSize) {
        // Minimal check, production code should handle errors
        return;
    }
    outArr[outPos] = marker;

    if (marker == 0x00) {
        // zero run => write runCount
        outArr[outPos+1] = (uint8_t)( runCount        & 0xFF);
        outArr[outPos+2] = (uint8_t)((runCount >> 8 ) & 0xFF);
        outArr[outPos+3] = (uint8_t)((runCount >>16 ) & 0xFF);
        outArr[outPos+4] = (uint8_t)((runCount >>24 ) & 0xFF);
    } 
    else {
        // non-zero => write float bits
        union {
            float   f;
            uint8_t b[4];
        } u;
        u.f = floatVal;
        outArr[outPos+1] = u.b[0];
        outArr[outPos+2] = u.b[1];
        outArr[outPos+3] = u.b[2];
        outArr[outPos+4] = u.b[3];
    }
}

/**
 * addCompressedArraysKernel:
 *   - Single-thread kernel that merges two compressed arrays (arr1, arr2),
 *     each in 5-byte-run format, into an output array "outArr".
 *
 * arr1, arr2:    pointers to device memory of compressed floats
 * len1, len2:    total byte sizes of arr1, arr2
 * outArr:        device pointer for the merged, compressed output
 * outArrSize:    capacity (in bytes) of outArr
 * d_outSize:     pointer to a device-size_t, will store the final used size
 *
 * Usage: 
 *   - Launch with <<<1,1>>> (1 block, 1 thread) for clarity.
 *   - After kernel, read d_outSize to know how many bytes of outArr are valid.
 */
extern "C" __global__
void Madd2CompressedArraysKernel(uint8_t* __restrict__ outArr, size_t outArrSize,
                                 uint8_t* __restrict__ arr1, size_t len1,
                                 uint8_t* __restrict__ arr2, size_t len2,
                                 float fac1, float fac2,
                                 size_t* d_outSize)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return; // single-thread approach
    }

    // We'll iterate through arr1, arr2 in parallel, run by run
    size_t pos1 = 0; // current byte pointer in arr1
    size_t pos2 = 0; // current byte pointer in arr2
    size_t outPos = 0; // current write pointer in outArr

    // Current run's leftover count
    unsigned int runCount1 = 0;
    unsigned int runCount2 = 0;
    float        val1      = 0.0f;
    float        val2      = 0.0f;
    uint8_t      marker1   = 0x00;
    uint8_t      marker2   = 0x00;

    // We'll parse initial runs
    if (pos1 < len1) {
        parseRun(arr1, len1, pos1, marker1, val1, runCount1);
    }
    if (pos2 < len2) {
        parseRun(arr2, len2, pos2, marker2, val2, runCount2);
    }

    // We'll keep track of how many bytes of each run we've "consumed".
    // For zero runs, that count is runCount. For non-zero runs, that count is 1.
    // Once a run is fully consumed, move pos forward by 5 bytes and parse next run.
    while ( (pos1 < len1) || (pos2 < len2) || (runCount1 > 0) || (runCount2 > 0) )
    {
        // If runCount is 0, parse a new run from that array
        if (runCount1 == 0 && pos1 < len1) {
            pos1 += 5; // skip the 5 bytes we previously read
            if (pos1 < len1) {
                parseRun(arr1, len1, pos1, marker1, val1, runCount1);
            } else {
                marker1 = 0x00;
                val1 = 0.0f;
                runCount1 = 0;
            }
        }
        if (runCount2 == 0 && pos2 < len2) {
            pos2 += 5;
            if (pos2 < len2) {
                parseRun(arr2, len2, pos2, marker2, val2, runCount2);
            } else {
                marker2 = 0x00;
                val2 = 0.0f;
                runCount2 = 0;
            }
        }

        // If both runs are exhausted & no more data => break
        if (runCount1 == 0 && runCount2 == 0 && pos1 >= len1 && pos2 >= len2) {
            break;
        }

        // Determine how many elements we can combine from these runs
        // For zero-run, runCount means "how many zeros remain"
        // For a non-zero run, runCount=1 means "one float"
        unsigned int n1 = (runCount1 > 0) ? 1 : 0; 
        if (marker1 == 0x00) n1 = runCount1; // zero run => can combine up to runCount1
        unsigned int n2 = (runCount2 > 0) ? 1 : 0;
        if (marker2 == 0x00) n2 = runCount2;

        // We combine min(n1, n2) elements
        unsigned int n = (n1 < n2) ? n1 : n2;
        if (n == 0) {
            // If either array is out of data, but the other isn't, we handle that scenario
            // We'll treat the missing array as zero => just produce from the other array
            if (runCount1 > 0 && runCount2 == 0) {
                // We have data in arr1, none in arr2 => effectively arr2=0
                n = n1; 
            } else if (runCount2 > 0 && runCount1 == 0) {
                n = n2;
            } else {
                // Nothing to do
                break;
            }
        }

        // We'll produce "n" elements of the sum
        //  - If we are in zero-run vs zero-run => sum is zero-run
        //  - If zero-run vs non-zero => sum is that non-zero
        //  - If both non-zero => sum is (val1 + val2)
        // Because we produce "n" identical results, we can form a single run in the output
        float sumVal = 0.0f;
        uint8_t outMarker = 0x00;
        unsigned int outCount = n;

        // Case: zero + zero => zero
        //       zero + non-zero => that float
        //       non-zero + zero => that float
        //       non-zero + non-zero => sum
        bool isZero1 = (marker1 == 0x00);
        bool isZero2 = (marker2 == 0x00);

        if (!isZero1 && !isZero2) {
            // both non-zero => sum
            sumVal = val1*fac1 + val2*fac2;
            outMarker = 0xFF; 
            outCount = n;   // typically n=1 in non-zero case
        }
        else if (!isZero1 && isZero2) {
            // val1 + 0 => val1
            sumVal = val1*fac1;
            outMarker = 0xFF;
            outCount = n; // if n>1, means multiple zeros from arr2 combining with the single float arr1 => tricky, but let's allow
        }
        else if (isZero1 && !isZero2) {
            // 0 + val2 => val2
            sumVal = val2*fac2;
            outMarker = 0xFF;
            outCount = n;
        }
        else {
            // zero + zero => zero
            sumVal = 0.0f;
            outMarker = 0x00;
            outCount = n; // if both are zero runs, we can produce zero-run length n
        }

        // Write the run to the output
        // We'll use atomicAdd on outPos if we suspect multiple threads, 
        // but here it's single-thread, so we can just proceed.
        // For demonstration, we skip concurrency concerns.
        if (outPos + 5 <= outArrSize) {
            writeRun(outArr, outArrSize, outPos, outMarker, sumVal, outCount);
            outPos += 5;
        }

        // Reduce runCount1, runCount2 by n
        // If non-zero run => runCount was 1 => subtract 1 or 0
        // If zero run => subtract n from runCount
        if (marker1 == 0xFF) {
            if (runCount1 > 0) runCount1 -= 1; // always 1
        } else {
            if (runCount1 >= n) runCount1 -= n;
            else runCount1 = 0;
        }
        if (marker2 == 0xFF) {
            if (runCount2 > 0) runCount2 -= 1;
        } else {
            if (runCount2 >= n) runCount2 -= n;
            else runCount2 = 0;
        }
    }

    // Possibly: we can merge adjacent runs in outArr if they are both zero 
    // or combine consecutive non-zero runs. That is an optional "post-process."

    // Write final used size
    *d_outSize = outPos;
}
