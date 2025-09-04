# Updating ct_viewer module for the current user
Write-Host "Updating ct_viewer module for $env:USERNAME."

# Source the required Conda and Mamba scripts
# In PowerShell, this would require invoking the appropriate script to activate the environment
# Assuming conda is added to the path
& "micromamba" "activate" "ct_viewer"

# Get the active environment
$active_environment = (& "micromamba" "info" | Select-String 'active').Line.Split(":")[1].Split(" ")[1].Trim()

Write-Host "Updating ct_viewer module in the $active_environment environment."

# Change to the directory containing the .whl files

if (Test-Path "$env:DROPBOX"){
    Set-Location "$env:DROPBOX\Code\Python\medical_physics\CTViewer\dist"
}

else{
    Set-Location "$env:USERPROFILE\Dropbox\Code\Python\medical_physics\CTViewer\dist"
}

# Find the latest .whl file
$latest_whl_file = Get-ChildItem -Filter *.whl | Sort-Object LastWriteTime | Select-Object -Last 1

# Extract the latest version from the .whl file name
$latest_version = ($latest_whl_file.Name -split '-')[1]

# Get the currently installed version of ct_viewer
$json_contents = (micromamba list ^ct --json | ConvertFrom-Json)

if ($json_contents.count -gt 0)
{
    $installed_version = $json_contents[0].version
}
else {
    $installed_version = '0.0.0.0'
}

# Compare the versions
if ($latest_version -eq $installed_version) 
{
    Write-Host "Installed version $installed_version matches the latest version $latest_version. Exiting script."
} 
else 
{
    Write-Host "Updating ct_viewer from $installed_version to $latest_version."

    # Install the latest version using pip
    pip install --no-deps --no-build-isolation $latest_whl_file.FullName
}