#!/bin/bash

echo "Updating ct_viewer module for $USER."

source /home/$USER/mambaforge/etc/profile.d/conda.sh
source /home/$USER/mambaforge/etc/profile.d/mamba.sh

mamba activate ct_viewer

active_environment="$(mamba info | grep 'active environment' | cut -d ':' -f 2 | xargs)"

echo "Updating ct_viewer module in the $active_environment environment."

cd /home/pboyle/Dropbox/Code/Python/medical_physics/ct_viewer/dist/

latest_whl_file="$(find ./ -type f -iname '*.whl' | sort | tail -1)"

latest_version="$(echo $latest_whl_file | cut -d '-' -f 2)"

installed_version="$(mamba list ^ct --json | jq -r '.[0].version')"

if [ $latest_version == $installed_version ]
then
    echo "Installed version $installed_version matches latest version $latest_version. Exiting script."

else
    echo "Updating ct_viewer from $installed_version to $latest_version."

    pip install --no-deps --no-build-isolation $latest_whl_file
fi