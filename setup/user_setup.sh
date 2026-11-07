#!/bin/bash
# Installs the Python dependencies, nvdiffrast and BART (v0.6.00, into ./libs).
set -e

pip install ninja
# nvdiffrast is compiled for these CUDA architectures; adjust to your GPUs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;9.0+PTX}"
pip install --no-binary :all: git+https://github.com/NVlabs/nvdiffrast --no-deps --no-build-isolation
python -c "import nvdiffrast.torch as dr; ctx=dr.RasterizeCudaContext()" # first time compilation takes some time

pip install -r ./setup/requirements.txt

mkdir -p ./libs
(cd ./libs && wget https://github.com/mrirecon/bart/archive/refs/tags/v0.6.00.tar.gz && tar -xf v0.6.00.tar.gz && cd bart-0.6.00 && make)
echo "BART installed to $(pwd)/libs/bart-0.6.00 (found automatically; set TOOLBOX_PATH only to use a different BART build)"
