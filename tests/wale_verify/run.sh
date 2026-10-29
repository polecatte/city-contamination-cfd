#!/usr/bin/env bash
# OpenLB's stock WALE (collisionLES.h, copied verbatim into wale_check.cpp) vs Nicoud & Ducros
# (1999) in numpy and the urban_les.h formula, on a set of velocity-gradient tensors.
set -e; cd "$(dirname "$0")"
g++ -O2 -o wale_check wale_check.cpp
python3 ref.py
# urban_les.h's sqrt form of the WALE powers vs pow, in double and float
g++ -O2 -std=c++17 -o pow_sqrt_check pow_sqrt_check.cpp && ./pow_sqrt_check
