#!/bin/bash

# curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
# bash Miniforge3-$(uname)-$(uname -m).sh

# source /home/$USER/mambaforge/etc/profile.d/conda.sh
# source /home/$USER/mambaforge/etc/profile.d/mamba.sh

micromamba create --name ct_viewer python=3.12

eval "$(micromamba shell hook --shell bash)"

sleep 0.5

micromamba activate ct_viewer

micromamba info

micromamba install -y colorcet h5py numpy pandas numba scipy quaternionic matplotlib openpyxl xlsxwriter nibabel

nvcc_version="$(nvcc --version | grep 'release' | cut -d ' ' -f 5 | cut -d ',' -f 1)"

micromamba install -y cupy cuda-version=$nvcc_version

pip install dearpygui python-gdcm

# Install ct_viewer

cd /home/pboyle/Dropbox/Code/Python/medical_physics/ct_viewer/dist/

latest_whl_file="$(find ./ -type f -name '*.whl' -printf '%T@ %p\n' | sort -n -r | head -1 | cut -d' ' -f2-)"

pip install --no-deps --no-build-isolation $latest_whl_file