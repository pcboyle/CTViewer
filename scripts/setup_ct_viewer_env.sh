#!/bin/bash

"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

sleep 0.5

eval "$(micromamba shell hook --shell bash)"

sleep 0.5

micromamba create --name ct_viewer python=3.13

micromamba activate ct_viewer

micromamba info

micromamba install -y colorcet h5py numpy pandas numba scipy quaternionic matplotlib openpyxl xlsxwriter nibabel

nvcc_version="$(nvcc --version | grep 'release' | cut -d ' ' -f 5 | cut -d ',' -f 1)"

micromamba install -y cupy cuda-version=$nvcc_version

pip install dearpygui python-gdcm

# Install ct_viewer

cd /home/pboyle/Dropbox/Code/Python/medical_physics/CTViewer/dist/

latest_whl_file="$(ls -t ./*.whl | head -n 1)"

pip install --no-deps --no-build-isolation $latest_whl_file

cd $pwd