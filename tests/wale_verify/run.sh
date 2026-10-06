#!/usr/bin/env bash
# OpenLB's stock WALE (collisionLES.h, copied verbatim into wale_check.cpp) vs Nicoud & Ducros
# (1999) in numpy and the urban_les.h formula, on a set of velocity-gradient tensors.
set -e; cd "$(dirname "$0")"
g++ -O2 -o wale_check wale_check.cpp
python3 ref.py
