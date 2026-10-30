# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
<#
    Builds Albireo for Windows with Visual Studio 2022 or its Build Tools: the engines of
    the devices, each in its own build tree, then the service, the tray, the processor
    engines and the installer, which carries all the engines.

        powershell -ExecutionPolicy Bypass -File build_windows.ps1 [-NoOpenVINO] [-Cuda]
            [-CudaVersion 12.9] [-Cudnn <folder of cuDNN>] [-CudaArchitectures <list>]
            [-NoBuiltinModels] [-ModelCache <folder>] [-Tests] [-Jobs <n>]

    Requirements: Visual Studio 2022, or the Build Tools for Visual Studio 2022, with the
    C++ workload, and CMake 3.21 or later (the C++ CMake tools of the workload, found
    without the PATH).  The OpenVINO engines, one per family of processor kernels, are
    built by default, so that every installation can use the NPU and the graphics
    processor of Intel: they need the runtime of OpenVINO (python -m pip install openvino,
    or the archive with INTEL_OPENVINO_DIR set).  -NoOpenVINO leaves them out, and
    -OpenVINO, which earlier versions needed, is still accepted.  -Cuda needs the CUDA
    toolkit 12.8 or later and cuDNN 9; with the Build Tools, the CUDA integration of the
    toolkit installer goes to the Build Tools when Visual Studio is absent.

    -CudaVersion picks one of several installed toolkits, and -Cudnn the folder of one cuDNN
    installation (C:\Program Files\NVIDIA\CUDNN\v9.11 for instance).  Without
    -CudaArchitectures, the CUDA engine is built for the cards from the GeForce GTX 900
    series on.  The GTX 900 and 10 series (Maxwell, Pascal) need CUDA 12.8 or 12.9 with
    cuDNN 9.11 or earlier: CUDA 13 and cuDNN 9.12 start at Turing (GTX 16 and RTX series).
    -Jobs bounds the number of compilations run at once, for a machine with little memory.

    The models that come with Albireo (resources/builtin_models.json) are downloaded once
    into the cache, %LOCALAPPDATA%\Albireo\build-cache or -ModelCache, about 4.2 GB with
    their converted files, then checked, converted and appended to the complete installer.
    A file already on this computer, in the models folder of an installation of Albireo or
    in $HOME\models, joins the cache instead, when its SHA-256 matches.  Behind a proxy, the
    downloads take the one of HTTPS_PROXY.  -NoBuiltinModels leaves the models out: the two
    installers are then the same program.

    The build trees are created at the root of the repository: build-albireo,
    build-albireo-ov and build-albireo-cuda.  A tree configured by a Visual Studio
    installation removed since (Visual Studio Community replaced by the Build Tools, for
    instance) is created again.  The installers are build-albireo\bin\albireo-setup.exe,
    complete with its models, and build-albireo\bin\albireo-setup-online.exe, the one of
    the updates, which installs the programs and downloads the models.
#>
param(
    [switch]$NoOpenVINO,
    [switch]$OpenVINO,
    [switch]$Cuda,
    [string]$CudaVersion = "",
    [string]$Cudnn = "",
    [string]$CudaArchitectures = "",
    [switch]$NoBuiltinModels,
    [string]$ModelCache = "",
    [switch]$Tests,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$root = (Resolve-Path (Join-Path $source "..\..\..\..")).Path
$generator = @("-G", "Visual Studio 17 2022", "-A", "x64")
$testing = if ($Tests) { "ON" } else { "OFF" }
$parallel = if ($Jobs -gt 0) { @("--parallel", "$Jobs") } else { @("--parallel") }

# CMake: the one of the PATH, else the one of the C++ workload of Visual Studio or of its
# Build Tools.
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installations = @(& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath)
        foreach ($vs in $installations) {
            $candidate = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
            if (-not $cmake -and (Test-Path $candidate)) { $cmake = $candidate }
        }
    }
}
if (-not $cmake) {
    throw "CMake was not found: add the C++ CMake tools to the C++ workload, or install CMake 3.21 or later"
}

function Invoke-Step([string]$what, [string[]]$arguments) {
    Write-Host "== $what"
    & $cmake @arguments
    if ($LASTEXITCODE -ne 0) { throw "$what failed" }
}

# A build tree keeps the Visual Studio installation that configured it, which CMake then
# requires: a tree of an installation removed since is created again.
function Reset-StaleTree([string]$tree) {
    $cache = Join-Path $tree "CMakeCache.txt"
    if (-not (Test-Path $cache)) { return }
    $line = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR_INSTANCE:[A-Z]+=(.+)$' | Select-Object -First 1
    if (-not $line) { return }
    $instance = $line.Matches[0].Groups[1].Value.Trim()
    if ($instance -and -not (Test-Path $instance)) {
        Write-Host "== $tree was configured by $instance, which is no longer installed: creating it again"
        Remove-Item -Recurse -Force $tree
    }
}

# The runtime of OpenVINO, where dlib looks for it (dlib/cmake_utils/find_openvino.cmake).
function Test-OpenVINO {
    if ($env:INTEL_OPENVINO_DIR -or $env:OpenVINO_DIR) { return $true }
    foreach ($name in @("python", "python3")) {
        $python = (Get-Command $name -ErrorAction SilentlyContinue).Source
        if (-not $python) { continue }
        # The alias of the Microsoft Store, without Python, writes its message to the errors.
        try {
            $probe = "import importlib.util; print(1 if importlib.util.find_spec('openvino') else 0)"
            $found = & $python -c $probe 2>$null
            if ($LASTEXITCODE -eq 0 -and "$found".Trim() -eq "1") { return $true }
        }
        catch {
            continue
        }
    }
    return $false
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
if (-not $NoOpenVINO) {
    if (-not (Test-OpenVINO)) {
        throw ("the runtime of OpenVINO was not found: python -m pip install openvino, or the archive of " +
            "OpenVINO with INTEL_OPENVINO_DIR set; -NoOpenVINO builds without the engines of the NPU")
    }
    $tree = Join-Path $root "build-albireo-ov"
    Reset-StaleTree $tree
    Invoke-Step "configuring the OpenVINO engine" (@("-S", $source, "-B", $tree) + $generator +
        @("-DALBIREO_PROGRAMS=OFF", "-DALBIREO_ENGINE_OPENVINO=ON", "-DALBIREO_TESTS=$testing"))
    Invoke-Step "building the OpenVINO engine" (@("--build", $tree, "--config", "Release") + $parallel)
    $extra += (Join-Path $tree "bin\engines")
}
if ($Cuda) {
    $tree = Join-Path $root "build-albireo-cuda"
    Reset-StaleTree $tree
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

# The files of the models that come with Albireo already on this computer join the cache,
# by a hard link on the same volume, else by a copy, instead of being downloaded again.
function Import-BuiltinSources([string]$cache, [string[]]$folders) {
    $manifest = Get-Content -Raw -Path (Join-Path $source "resources\builtin_models.json") | ConvertFrom-Json
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    foreach ($m in $manifest.models) {
        $target = Join-Path $cache ($m.from.sha256 + ".gguf")
        if (Test-Path $target) { continue }
        $found = $null
        foreach ($folder in $folders) {
            if ($found -or -not (Test-Path $folder)) { continue }
            foreach ($f in @(Get-ChildItem -Path $folder -Filter *.gguf -File -ErrorAction SilentlyContinue)) {
                if ($found -or $f.Length -ne [long]$m.from.size) { continue }
                if ((Get-FileHash -Algorithm SHA256 $f.FullName).Hash.ToLower() -eq $m.from.sha256) { $found = $f }
            }
        }
        if (-not $found) { continue }
        Write-Host "== $($m.id): $($found.FullName) joins the cache of the models"
        try { New-Item -ItemType HardLink -Path $target -Target $found.FullName -ErrorAction Stop | Out-Null }
        catch { Copy-Item -Path $found.FullName -Destination $target }
    }
}

$main = Join-Path $root "build-albireo"
Reset-StaleTree $main
if (-not $NoBuiltinModels) {
    $cache = if ($ModelCache) { $ModelCache } else { Join-Path $env:LOCALAPPDATA "Albireo\build-cache" }
    Import-BuiltinSources $cache @((Join-Path $env:ProgramData "Albireo\models"), (Join-Path $HOME "models"))
}
$dirs = ($extra | ForEach-Object { $_.Replace('\', '/') }) -join ';'
$builtin = if ($NoBuiltinModels) { "OFF" } else { "ON" }
$options = @("-DALBIREO_EXTRA_ENGINE_DIRS=$dirs", "-DALBIREO_TESTS=$testing", "-DALBIREO_BUILTIN_MODELS=$builtin")
if ($ModelCache) { $options += "-DALBIREO_MODEL_CACHE=$($ModelCache.Replace('\', '/'))" }
Invoke-Step "configuring the programs" (@("-S", $source, "-B", $main) + $generator + $options)
Invoke-Step "building the programs, the models and the installers" (@("--build", $main, "--config", "Release") +
    $parallel)

foreach ($name in @("albireo-setup.exe", "albireo-setup-online.exe")) {
    $setup = Join-Path $main "bin\$name"
    $hash = (Get-FileHash -Algorithm SHA256 $setup).Hash.ToLower()
    $size = (Get-Item $setup).Length
    Write-Host "== $setup"
    Write-Host "   $size bytes, SHA-256 $hash"
}
