#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12

pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu124"
pip install -r "$ROOT/requirements.txt"

rm -rf "$ROOT/submodules/diff-surfel-rasterization/build" "$ROOT/submodules/diff-surfel-rasterization"/*.egg-info
pip install --no-build-isolation --force-reinstall "$ROOT/submodules/diff-surfel-rasterization"

rm -rf "$ROOT/submodules/simple-knn/build" "$ROOT/submodules/simple-knn"/*.egg-info
pip install --no-build-isolation --force-reinstall "$ROOT/submodules/simple-knn"

rm -rf "$ROOT/submodules/cubemapencoder/build" "$ROOT/submodules/cubemapencoder"/*.egg-info
pip install --no-build-isolation --force-reinstall -e "$ROOT/submodules/cubemapencoder"

echo "Done."
