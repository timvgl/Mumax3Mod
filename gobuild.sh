#!/bin/bash
export PATH=/usr/local/go/bin:$PATH
cd "$(dirname "$0")"
go build ./cuda/ ./engine/ 2>&1 | head -60
echo "EXIT: ${PIPESTATUS[0]}"
