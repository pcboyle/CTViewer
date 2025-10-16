#!/bin/bash

eval "$(micromamba shell hook --shell bash)"

sleep 0.5

echo "Updating ct_viewer module for $USER."

micromamba activate ct_viewer

active_environment="$(micromamba info | grep 'active environment' | cut -d ':' -f 2 | xargs)"

echo "Updating ct_viewer module in the $active_environment environment."

cd /home/pboyle/Dropbox/Code/Python/medical_physics/CTViewer/dist/

latest_whl_file="$(ls -t ./*.whl | head -n 1)"
# latest_whl_file="$(find ./ -type f -iname '*.whl' | sort | tail -1)"

latest_version="$(echo $latest_whl_file | cut -d '-' -f 2)"

installed_version="$(micromamba list ^ct --json | jq -r '.[0].version')"

if [ $latest_version == $installed_version ]
then
    echo "Installed version $installed_version matches latest version $latest_version. Exiting script."

else
    echo "Updating ct_viewer from $installed_version to $latest_version."

    pip install --no-deps --no-build-isolation $latest_whl_file
fi