#!/bin/bash
# split_cuda_functions.sh
#
# This script splits the file cuda_functions.cu into separate kernel files.
# Each block is assumed to be separated by a line that exactly matches:
#    #################
#
# In each block the kernel declaration is expected to be on a line like:
#    extern "C" __global__ void absGovaluate(...
#
# The script extracts the function name, removes the trailing "Govaluate",
# and renames the block to: <function>_function.cu inside the directory "cuda_functions_split".

set -e

INPUT="cuda_functions.cu"

# Create output directory if it does not exist

# Use csplit to split the input file into chunks using the delimiter.
# The regex '^#################$' matches a line that consists solely of 20 '#' characters.
# The -z option removes empty output files.
csplit -z -f chunk_ "$INPUT" '/^#################$/' '{*}'

# Iterate over each generated chunk file
for chunk in chunk_*; do
    # Remove the delimiter line if it is the first line of the chunk
    if head -n 1 "$chunk" | grep -q '^#################$'; then
        sed -i '1d' "$chunk"
    fi

    # Extract the kernel function name.
    # We expect a line like:
    #   extern "C" __global__ void absGovaluate(...
    # The following grep uses Perl regex (-P) with \K to output only the function name.
    fname=$(grep -oP 'extern "C" __global__ void\s+\K[^ (]+' "$chunk" | head -n 1)
    
    # If no function name was found, remove the chunk and skip it.
    if [ -z "$fname" ]; then
        echo "No function name found in $chunk, skipping."
        rm -f "$chunk"
        continue
    fi

    # Remove the trailing "Govaluate" if present.
    fname=$(echo "$fname" | sed 's/Govaluate$//')
    
    # Build the output filename.
    new_filename="${fname}_function.f.cu"
    
    # Move (rename) the chunk to the new filename.
    mv "$chunk" "$new_filename"
    echo "Created file: $new_filename"
done
