# # Load Conda and Mamba into the session
# $conda_path = "$HOME\mambaforge\etc\profile.d\conda.ps1"
# $mamba_path = "$HOME\mambaforge\etc\profile.d\mamba.ps1"

# if (Test-Path $conda_path) {
#     . $conda_path
# }
# if (Test-Path $mamba_path) {
#     . $mamba_path
# }

# Create and activate the Conda environment
& micromamba create -n ct_viewer python=3.12 -y
& micromamba activate ct_viewer

# Install required Python packages
& micromamba install -y colorcet h5py numpy pandas numba scipy quaternionic matplotlib openpyxl xlsxwriter nibabel

# Detect the CUDA version
$nvcc_version = & nvcc --version | Select-String -Pattern "release" | ForEach-Object {
    ($_ -split ",")[1] -replace ".*release ", ""
}

# Install CuPy with the correct CUDA version
& micromamba install -y cupy "cuda-version=$nvcc_version"

# Install additional Python packages with pip
& pip install dearpygui python-gdcm

# Install ct_viewer
$ct_viewer_dir = "$env:DROPBOX\Code\Python\medical_physics\ct_viewer\dist"
Set-Location $ct_viewer_dir

$latest_whl_file = Get-ChildItem -Path . -Filter *.whl | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($latest_whl_file) {
    & pip install --no-deps --no-build-isolation $latest_whl_file.FullName
} else {
    Write-Output "No .whl file found in $ct_viewer_dir"
}