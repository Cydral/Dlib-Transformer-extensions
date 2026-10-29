# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
<#
    Builds Albireo for Windows with Visual Studio 2022: the engines of the devices asked
    for, each in its own build tree, then the service, the tray, the processor engines and
    the installer, which carries all the engines.

        powershell -ExecutionPolicy Bypass -File build_windows.ps1 [-OpenVINO] [-Cuda]
            [-CudaVersion 12.9] [-Cudnn <folder of cuDNN>] [-CudaArchitectures <list>]
            [-Tests] [-Jobs <n>]

    Requirements: Visual Studio 2022 with the C++ workload, CMake 3.21 or later.  -OpenVINO
    needs the runtime of OpenVINO (python -m pip install openvino, or the archive with
    INTEL_OPENVINO_DIR set); -Cuda needs the CUDA toolkit 12.8 or later and cuDNN 9.

    -CudaVersion picks one of several installed toolkits, and -Cudnn the folder of one cuDNN
    installation (C:\Program Files\NVIDIA\CUDNN\v9.11 for instance).  Without
    -CudaArchitectures, the CUDA engine is built for the cards from the GeForce GTX 900
    series on.  The GTX 900 and 10 series (Maxwell, Pascal) need CUDA 12.8 or 12.9 with
    cuDNN 9.11 or earlier: CUDA 13 and cuDNN 9.12 start at Turing (GTX 16 and RTX series).
    -Jobs bounds the number of compilations run at once, for a machine with little memory.

    The build trees are created at the root of the repository: build-albireo,
    build-albireo-ov and build-albireo-cuda.  The installer is build-albireo\bin\albireo-setup.exe.
#>
param(
    [switch]$OpenVINO,
    [switch]$Cuda,
    [string]$CudaVersion = "",
    [string]$Cudnn = "",
    [string]$CudaArchitectures = "",
    [switch]$Tests,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$root = (Resolve-Path (Join-Path $source "..\..\..\..")).Path
$generator = @("-G", "Visual Studio 17 2022", "-A", "x64")
$testing = if ($Tests) { "ON" } else { "OFF" }
$parallel = if ($Jobs -gt 0) { @("--parallel", "$Jobs") } else { @("--parallel") }

function Invoke-Step([string]$what, [string[]]$arguments) {
    Write-Host "== $what"
    & cmake @arguments
    if ($LASTEXITCODE -ne 0) { throw "$what failed" }
}

# A file of a cuDNN installation, in the folder of the major version of the toolkit when the
# installation carries several (include\12.9 and include\13.0 for instance).
function Find-CudnnFile([string]$under, [string]$name, [string]$major) {
    $found = @(Get-ChildItem -Path $under -Recurse -Filter $name -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending)
    if ($major) {
        $pattern = '\\' + $major + '\.[0-9]+(\\|$)'
        $matching = @($found | Where-Object { $_.DirectoryName -match $pattern })
        if ($matching.Count -gt 0) { $found = $matching }
    }
    if ($found.Count -eq 0) { throw "no $name under $under" }
    return $found[0].FullName.Replace('\', '/')
}

$extra = @()
if ($OpenVINO) {
    $tree = Join-Path $root "build-albireo-ov"
    Invoke-Step "configuring the OpenVINO engine" (@("-S", $source, "-B", $tree) + $generator +
        @("-DALBIREO_PROGRAMS=OFF", "-DALBIREO_ENGINE_OPENVINO=ON", "-DALBIREO_TESTS=$testing"))
    Invoke-Step "building the OpenVINO engine" (@("--build", $tree, "--config", "Release") + $parallel)
    $extra += (Join-Path $tree "bin\engines")
}
if ($Cuda) {
    $tree = Join-Path $root "build-albireo-cuda"
    $options = @("-DALBIREO_PROGRAMS=OFF", "-DALBIREO_ENGINE_CUDA=ON", "-DALBIREO_TESTS=$testing")
    $toolset = @()
    $major = ""
    if ($CudaVersion) {
        # The CUDA integration of Visual Studio and the toolkit CMake searches, of that version.
        $variable = "CUDA_PATH_V" + $CudaVersion.Replace('.', '_')
        $toolkit = [Environment]::GetEnvironmentVariable($variable)
        if (-not $toolkit) { throw "CUDA $CudaVersion is not installed ($variable is not set)" }
        $toolset = @("-T", "cuda=$CudaVersion")
        $options += "-DCUDAToolkit_ROOT=$($toolkit.Replace('\', '/'))"
        $major = $CudaVersion.Split('.')[0]
    }
    elseif ($env:CUDA_PATH -match 'v([0-9]+)\.') {
        $major = $Matches[1]
    }
    if ($Cudnn) {
        $header = Find-CudnnFile $Cudnn "cudnn.h" $major
        $options += "-DCUDNN_INCLUDE_PATH=$(Split-Path $header -Parent)".Replace('\', '/')
        $options += "-DCUDNN_LIBRARY_PATH=$(Find-CudnnFile $Cudnn 'cudnn.lib' $major)"
    }
    if ($CudaArchitectures) { $options += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" }
    Invoke-Step "configuring the CUDA engine" (@("-S", $source, "-B", $tree) + $generator + $toolset + $options)
    Invoke-Step "building the CUDA engine" (@("--build", $tree, "--config", "Release") + $parallel)
    $extra += (Join-Path $tree "bin\engines")
}

$main = Join-Path $root "build-albireo"
$dirs = ($extra | ForEach-Object { $_.Replace('\', '/') }) -join ';'
Invoke-Step "configuring the programs" (@("-S", $source, "-B", $main) + $generator +
    @("-DALBIREO_EXTRA_ENGINE_DIRS=$dirs", "-DALBIREO_TESTS=$testing"))
Invoke-Step "building the programs and the installer" (@("--build", $main, "--config", "Release") + $parallel)

$setup = Join-Path $main "bin\albireo-setup.exe"
$hash = (Get-FileHash -Algorithm SHA256 $setup).Hash.ToLower()
$size = (Get-Item $setup).Length
Write-Host "== $setup"
Write-Host "   $size bytes, SHA-256 $hash"
