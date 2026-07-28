#!/bin/bash
export PATH=/usr/local/go/bin:$PATH
cd "$(dirname "$0")"
go build -o /tmp/mumax3vet ./cmd/mumax3 || { echo BUILD_FAIL; exit 1; }
cd /home/tvo/mumax_mod
for f in sanity_check_NP.mx3 \
         validation/A_static_stack/stack.mx3 \
         validation/B_bulk_shear/bulk_shear.mx3 \
         validation/C_rayleigh_bc/rayleigh_bc.mx3 \
         validation/D_rsaw_thinfilm/rsaw.mx3 \
         validation/E_shsaw_thickfilm/shsaw.mx3 \
         validation/F_nistripes/nistripes.mx3; do
    echo "=== vet $f"
    /tmp/mumax3vet -vet "$f" 2>&1 | tail -5
done
echo VET_DONE
