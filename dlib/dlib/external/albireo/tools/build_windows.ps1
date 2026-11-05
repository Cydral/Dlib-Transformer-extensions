# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
<#
    Builds Albireo for Windows with Visual Studio 2022 or its Build Tools: the engines of
    the devices, each in its own build tree, then the service, the tray, the processor
    engines and the installer, which carries all the engines.

        powershell -ExecutionPolicy Bypass -File build_windows.ps1 [-NoInstall] [-NoOpenVINO] [-Cuda | -NoCuda]
            [-CudaVersion 12.9] [-Cudnn <folder of cuDNN>] [-CudaCache <folder>] [-CudaArchitectures <list>]
            [-NoBuiltinModels] [-ModelCache <folder>] [-Tests] [-Jobs <n>]

    The first stage, "checking the prerequisites", checks what the build needs, installs what
    is missing and prints a table of what it found and installed: Visual Studio 2022 or its
    Build Tools with the C++ workload (the x64 tools, a Windows SDK, the redistributable files of
    the C++ runtime) and the C++ CMake tools, CMake 3.21 or later, Python 3.10 or later, the
    runtime of OpenVINO, CUDA and cuDNN, and the free space of the volumes of the build trees and
    of the caches, where it stops when one is short.  Without an installation of Visual Studio
    2022, the Build Tools come from https://aka.ms/vs/17/release/vs_BuildTools.exe, whose
    signature is checked; the components an installation lacks come from its installer
    (setup.exe modify).  Both need the rights of an administrator, which Windows asks for once
    the script has said so, and after a new installation of Visual Studio the script stops, to be
    run again from a new Developer PowerShell for VS 2022.  Python comes from winget for this
    user, else from the installer of python.org (3.12.10, its SHA-256 checked), and OpenVINO from
    pip for this user (openvino>=2025.0,<2027); neither needs the rights of an administrator.  An
    installation that fails stops the build, with the place of its log.  -NoInstall installs
    nothing; the table then gives the command that installs each missing item, for a computer
    where an administrator installs them, and the build stops when one is missing.  The
    downloads take the proxy of HTTPS_PROXY, else the one of the system, which the script gives
    to CMake and pip through HTTPS_PROXY, and Windows checks the servers with its certificate
    store, so that a proxy of the company that inspects TLS is accepted.

    The OpenVINO engines, one per family of processor kernels, are built by default, so that
    every installation can use the NPU and the graphics processor of Intel; the runtime of
    OpenVINO comes from pip, or from its archive with INTEL_OPENVINO_DIR set.  -NoOpenVINO
    leaves them out, and -OpenVINO, which earlier versions needed, is still accepted.

    The CUDA engine is built on every computer, and no NVIDIA card is needed for it.  An
    installed CUDA toolkit 12.8 or later, with its integration with Visual Studio and cuDNN 9,
    is taken first.  Without them, the build takes CUDA 12.9.1 and cuDNN 9.11.0 from the
    redistributable archives of NVIDIA, downloaded once into %LOCALAPPDATA%\Albireo\build-cache\cuda
    or -CudaCache (1.6 GB, 4.2 GB on disk with their extracted files), checked against the
    SHA-256 of their manifests and laid out there as a toolkit (cmake/cuda_redist.cmake), which
    needs neither an installation nor the rights of an administrator.  When a download or an
    extraction fails, the build goes on without the CUDA engine and says why; -Cuda makes it an
    error, and -NoCuda leaves the engine out.  With the Build Tools, the CUDA integration of the
    toolkit installer goes to the Build Tools when Visual Studio is absent.  cuDNN is found in
    C:\Program Files\NVIDIA\CUDNN, where its installer puts it, or in the folder -Cudnn names
    (an extracted archive).  Of several versions there, the latest before 9.12 that serves the
    toolkit is taken, else the latest.  When the cuDNN found is 9.12 or later, which leaves out
    the GeForce GTX 900 and 10 series, a CUDA 12 toolkit receives cuDNN 9.11.0 from the
    redistributable archives of NVIDIA instead, downloaded once into the same cache (0.6 GB,
    1.4 GB on disk with its extracted files), unless -Cudnn or -CudaArchitectures is given.

    -CudaVersion picks one of several installed toolkits; the archives serve when it is absent
    or names their version, 12.9.  -Cudnn names the folder of one cuDNN installation for an
    installed toolkit (C:\Program Files\NVIDIA\CUDNN\v9.11 for instance).  Without
    -CudaArchitectures, the CUDA engine is built for the cards from the GeForce GTX 900
    series on, whatever an earlier build of the tree was given.  The GTX 900 and 10 series
    (Maxwell, Pascal) need CUDA 12.8 or 12.9 with cuDNN 9.11 or earlier: CUDA 13 and cuDNN
    9.12 start at Turing (GTX 16 and RTX series), which a warning repeats at the end.
    -Jobs bounds the number of compilations run at once, for a machine with little memory.

    The models that come with Albireo (resources/builtin_models.json) are downloaded once
    into the cache, %LOCALAPPDATA%\Albireo\build-cache or -ModelCache, about 6.8 GB with
    their converted files, then checked, converted and appended to the complete installer.
    A file already on this computer, in the models folder of an installation of Albireo or
    in $HOME\models, joins the cache instead, when its SHA-256 matches.  -NoBuiltinModels
    leaves the models out: the two installers are then the same program.

    The build trees are created at the root of the repository: build-albireo,
    build-albireo-ov and build-albireo-cuda.  A tree configured by a Visual Studio
    installation removed since (Visual Studio Community replaced by the Build Tools, for
    instance), or the CUDA tree configured for another CUDA integration (an installed toolkit,
    then the archives), is created again.  The bin\engines folder of a device tree is emptied before
    its build, which places the engines and the libraries they load there again, so that the
    installer carries the libraries of the current runtimes only; the payload of the
    installer is then packed again.  The installers are build-albireo\bin\albireo-setup.exe,
    complete with its models, and build-albireo\bin\albireo-setup-online.exe, the one of
    the updates, which installs the programs and downloads the models.
#>
param(
    [switch]$NoInstall,
    [switch]$NoOpenVINO,
    [switch]$OpenVINO,
    [switch]$Cuda,
    [switch]$NoCuda,
    [string]$CudaVersion = "",
    [string]$Cudnn = "",
    [string]$CudaCache = "",
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
# CMake, which the check of the prerequisites finds.
$cmake = $null

function Invoke-Step([string]$what, [string[]]$arguments) {
    Write-Host "== $what"
    & $cmake @arguments
    if ($LASTEXITCODE -ne 0) { throw "$what failed" }
}

# A build tree keeps the Visual Studio installation that configured it, which CMake then
# requires: a tree of an installation removed since is created again.  It also keeps its
# toolset, which CMake reuses when none is given and refuses to change: given $toolset ("" for
# none), a tree configured with another one is created again.
function Reset-StaleTree([string]$tree, $toolset = $null) {
    $cache = Join-Path $tree "CMakeCache.txt"
    if (-not (Test-Path $cache)) { return }
    $line = Select-String -Path $cache -Encoding UTF8 -Pattern '^CMAKE_GENERATOR_TOOLSET:[A-Z]+=(.*)$' |
        Select-Object -First 1
    $previous = if ($line) { $line.Matches[0].Groups[1].Value.Trim() } else { $null }
    if ($null -ne $toolset -and $null -ne $previous -and $previous -cne $toolset) {
        Write-Host "== $tree was configured with the toolset `"$previous`": creating it again"
        Remove-Item -Recurse -Force $tree
        return
    }
    $line = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR_INSTANCE:[A-Z]+=(.+)$' | Select-Object -First 1
    if (-not $line) { return }
    $instance = $line.Matches[0].Groups[1].Value.Trim()
    if ($instance -and -not (Test-Path $instance)) {
        Write-Host "== $tree was configured by $instance, which is no longer installed: creating it again"
        Remove-Item -Recurse -Force $tree
    }
}

# The engines folder of a device tree only receives files: the libraries of an earlier version
# of the CUDA runtime, for instance, would stay there and join the installer.  Its build places
# the engines and the libraries they load there again.
function Clear-Engines([string]$tree) {
    $engines = Join-Path $tree "bin\engines"
    if (-not (Test-Path -LiteralPath $engines)) { return }
    try {
        Get-ChildItem -LiteralPath $engines -Force | Remove-Item -Recurse -Force
    }
    catch {
        throw "$engines cannot be emptied, a program may still use its files: $($_.Exception.Message)"
    }
}

# The runtime of OpenVINO, where dlib looks for it (dlib/cmake_utils/find_openvino.cmake): the
# archive of INTEL_OPENVINO_DIR or OpenVINO_DIR, else the package of the interpreter given.  Its
# version, "archive" for an archive, or "" when it is absent.
function Get-OpenVINO([string]$python) {
    if ($env:INTEL_OPENVINO_DIR -or $env:OpenVINO_DIR) { return "archive" }
    if (-not $python) { return "" }
    # An interpreter without the package writes a traceback to the errors.
    try {
        $found = & $python -c "import importlib.metadata as m; print(m.version('openvino'))" 2>$null
        if ($LASTEXITCODE -eq 0) { return "$found".Trim() }
    }
    catch {
        return ""
    }
    return ""
}

# Whether a version of the package of OpenVINO is one the build supports, 2025 to 2026.
$openvinoRange = "openvino>=2025.0,<2027"
function Test-OpenVINOVersion([string]$version) {
    return ($version -match '^([0-9]+)\.') -and [int]$Matches[1] -ge 2025 -and [int]$Matches[1] -lt 2027
}

# A file of a cuDNN installation, in the folder of the major version of the toolkit when the
# installation carries several (include\12.9 and include\13.0 for instance).  With -Strict,
# only that folder serves.
function Find-CudnnFile([string]$under, [string]$name, [string]$major, [switch]$Strict) {
    $found = @(Get-ChildItem -Path $under -Recurse -Filter $name -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending)
    if ($major) {
        $pattern = '\\' + $major + '\.[0-9]+(\\|$)'
        $matching = @($found | Where-Object { $_.DirectoryName -match $pattern })
        if ($matching.Count -gt 0 -or $Strict) { $found = $matching }
    }
    if ($found.Count -eq 0) { throw "no $name under $under" }
    return $found[0].FullName.Replace('\', '/')
}

# The folders of the versions of cuDNN under a folder of its installer (v9.11, v9.14), the
# latest first; with a toolkit before CUDA 13, those before 9.12 come first, since cuDNN 9.12
# left the GeForce GTX 900 and 10 series (Maxwell, Pascal).
function Get-CudnnVersionFolders([string]$under, [string]$major) {
    $versions = @(Get-ChildItem -LiteralPath $under -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -match '^v?([0-9]+\.[0-9]+)(\.[0-9]+)*$') {
            [pscustomobject]@{ Path = $_.FullName; Version = [version]$Matches[1] }
        }
    } | Sort-Object Version -Descending)
    if (-not $major -or [int]$major -lt 13) {
        $before = @($versions | Where-Object { $_.Version -lt [version]"9.12" })
        $after = @($versions | Where-Object { $_.Version -ge [version]"9.12" })
        $versions = $before + $after
    }
    return @($versions | ForEach-Object { $_.Path })
}

# The CUDA toolkit of -CudaVersion, else that of CUDA_PATH: its folder when its compiler is there.
function Find-CudaToolkit {
    $toolkit = $env:CUDA_PATH
    if ($CudaVersion) {
        $toolkit = [Environment]::GetEnvironmentVariable("CUDA_PATH_V" + $CudaVersion.Replace('.', '_'))
    }
    if ($toolkit -and (Test-Path (Join-Path $toolkit "bin\nvcc.exe"))) { return $toolkit }
    return $null
}

# The integration of the toolkit with Visual Studio or its Build Tools, which the CUDA projects of
# CMake need: the files "CUDA <version>.props" among the build customizations of MSBuild.
function Test-CudaIntegration {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $pattern = if ($CudaVersion) { "CUDA $CudaVersion.props" } else { "CUDA *.props" }
    $installations = @(& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath)
    foreach ($vs in $installations) {
        $folder = Join-Path $vs "MSBuild\Microsoft\VC\v170\BuildCustomizations"
        if (@(Get-ChildItem -Path $folder -Filter $pattern -ErrorAction SilentlyContinue).Count -gt 0) { return $true }
    }
    return $false
}

# The folder of cuDNN: that of -Cudnn, else the one of its installer, else the toolkit itself,
# the first that holds its header and its library.  The folders of its versions come first
# (Get-CudnnVersionFolders), each one with the files of the major version of the toolkit.
function Find-Cudnn([string]$toolkit, [string]$major) {
    $folders = @((Join-Path $env:ProgramFiles "NVIDIA\CUDNN"), $toolkit)
    if ($Cudnn) { $folders = @($Cudnn) }
    $versions = @()
    foreach ($folder in $folders) {
        if ($folder -and (Test-Path -LiteralPath $folder)) { $versions += @(Get-CudnnVersionFolders $folder $major) }
    }
    foreach ($folder in $versions) {
        try {
            Find-CudnnFile $folder "cudnn.h" $major -Strict | Out-Null
            Find-CudnnFile $folder "cudnn.lib" $major -Strict | Out-Null
            return $folder
        }
        catch {
            continue
        }
    }
    foreach ($folder in $folders) {
        if (-not $folder -or -not (Test-Path $folder)) { continue }
        try {
            Find-CudnnFile $folder "cudnn.h" $major | Out-Null
            Find-CudnnFile $folder "cudnn.lib" $major | Out-Null
            return $folder
        }
        catch {
            continue
        }
    }
    return $null
}

# The oldest compute capability of the CUDA engine of a tree, as ten times its major version
# plus its minor version, which its configuration records; 0 when unknown.
function Get-CudaMinimum([string]$tree) {
    $cache = Join-Path $tree "CMakeCache.txt"
    if (-not (Test-Path -LiteralPath $cache)) { return 0 }
    $line = Select-String -LiteralPath $cache -Pattern '^ALBIREO_CUDA_MIN_CC_BUILT:INTERNAL=([0-9]+)$' |
        Select-Object -First 1
    if (-not $line) { return 0 }
    return [int]$line.Matches[0].Groups[1].Value
}

# The warning given when the CUDA engine leaves out the GeForce GTX 900 and 10 series.
function Write-CudaWarning([int]$minimum) {
    $cc = "$([math]::Floor($minimum / 10)).$($minimum % 10)"
    Write-Warning "THE CUDA ENGINE OF THIS BUILD STARTS AT COMPUTE CAPABILITY $cc."
    Write-Warning ("It leaves out the GeForce GTX 900 and 10 series (Maxwell 5.x, Pascal 6.1), which the " +
        "release supports.")
    Write-Warning ("They need CUDA 12.8 or 12.9 with cuDNN 9.11 or earlier, and the default architectures " +
        "(no -CudaArchitectures): -CudaVersion 12.9 -Cudnn `"C:\Program Files\NVIDIA\CUDNN\v9.11`" for instance, " +
        "or -CudaVersion 12.9 alone without CUDA 12.9 installed, which takes the redistributable archives of NVIDIA.")
}

# The CUDA toolkit and cuDNN of the redistributable archives of NVIDIA, downloaded once into the
# cache and laid out there by cmake/cuda_redist.cmake: their folders and the bytes downloaded by
# this run, or $null when a download, a hash or an extraction fails, which CMake explains.
function Get-CudaRedist([string]$cache) {
    Write-Host "== CUDA $redistCuda and cuDNN $redistCudnn from the redistributable archives of NVIDIA, in $cache"
    $printed = @{}
    & $cmake "-DCACHE=$($cache.Replace('\', '/'))" "-DCUDA_VERSION=$redistCuda" "-DCUDNN_VERSION=$redistCudnn" `
        -P (Join-Path $source "cmake\cuda_redist.cmake") | ForEach-Object {
        Write-Host $_
        if ($_ -match '^-- CUDA_REDIST_([A-Z]+)=(.+)$') { $printed[$Matches[1]] = $Matches[2].Trim() }
    }
    if ($LASTEXITCODE -ne 0 -or -not $printed["TOOLKIT"] -or -not $printed["CUDNN"]) { return $null }
    # The folders the script names, under the cache given to it: CMake writes UTF-8 into a pipe,
    # which Windows PowerShell reads in the code page of the console.
    return [pscustomobject]@{
        Toolkit = Join-Path $cache (Split-Path -Leaf $printed["TOOLKIT"])
        Cudnn = Join-Path $cache (Split-Path -Leaf $printed["CUDNN"])
        Downloaded = [long]$printed["DOWNLOADED"]
    }
}

# The version of cuDNN in a folder, from its cudnn_version.h, or $null when the header does not give it.
function Get-CudnnVersion([string]$folder, [string]$major) {
    try {
        $header = Find-CudnnFile $folder "cudnn_version.h" $major
    }
    catch {
        return $null
    }
    $text = Get-Content -Raw -LiteralPath $header
    if ($text -match '#define\s+CUDNN_MAJOR\s+([0-9]+)') { $first = $Matches[1] } else { return $null }
    if ($text -match '#define\s+CUDNN_MINOR\s+([0-9]+)') { return [version]"$first.$($Matches[1])" }
    return $null
}

# cuDNN of the redistributable archives of NVIDIA alone, for the major version of the toolkit of
# the archives, laid out by cmake/cuda_redist.cmake beside an installed toolkit: its folder and the
# bytes downloaded by this run, or $null when a download, a hash or an extraction fails.
function Get-CudnnRedist([string]$cache) {
    Write-Host "== cuDNN $redistCudnn from the redistributable archives of NVIDIA, in $cache"
    $printed = @{}
    & $cmake "-DCACHE=$($cache.Replace('\', '/'))" "-DCUDA_VERSION=$redistCuda" "-DCUDNN_VERSION=$redistCudnn" `
        "-DCUDNN_ONLY=ON" -P (Join-Path $source "cmake\cuda_redist.cmake") | ForEach-Object {
        Write-Host $_
        if ($_ -match '^-- CUDA_REDIST_([A-Z]+)=(.+)$') { $printed[$Matches[1]] = $Matches[2].Trim() }
    }
    if ($LASTEXITCODE -ne 0 -or -not $printed["CUDNN"]) { return $null }
    return [pscustomobject]@{
        Cudnn = Join-Path $cache (Split-Path -Leaf $printed["CUDNN"])
        Downloaded = [long]$printed["DOWNLOADED"]
    }
}

# The table of the prerequisites, one line per item with its state: found, installed, missing
# (with the command that installs it), short (the free space of a volume), skipped (not checked).
$prerequisites = @()
function Add-Prerequisite([string]$item, [string]$state, [string]$detail) {
    $script:prerequisites += [pscustomobject]@{ Item = $item; State = $state; Detail = $detail }
}
function Write-Prerequisites {
    Write-Host ($script:prerequisites | Format-Table -Property Item, State, Detail -AutoSize -Wrap |
        Out-String -Width 220).TrimEnd()
}

# A download through the proxy of HTTPS_PROXY, else the one of the system, with the account of
# the user for a proxy that authenticates it; a failure stops the build with the item, the
# address and the error.
function Save-Download([string]$item, [string]$url, [string]$file) {
    Write-Host "== ${item}: downloading $url"
    $request = @{ Uri = $url; OutFile = $file; UseBasicParsing = $true }
    if ($env:HTTPS_PROXY) {
        $request.Proxy = $env:HTTPS_PROXY
        $request.ProxyUseDefaultCredentials = $true
    }
    # The progress bar of Windows PowerShell slows the transfer down.
    $progress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        Invoke-WebRequest @request
    }
    catch {
        throw "${item}: the download of $url failed: $($_.Exception.Message)"
    }
    finally {
        $ProgressPreference = $progress
    }
}

# Runs an installer with the rights of an administrator, which Windows asks for after this
# script has said so; a refusal, a failure or a restart that Windows needs stops the build, with
# the place of the logs of the installer.
function Invoke-Elevated([string]$item, [string]$file, [string[]]$arguments, [string]$logs) {
    Write-Host "== ${item}: the installation needs the rights of an administrator, which Windows asks for now"
    try {
        $process = Start-Process -FilePath $file -ArgumentList $arguments -Verb RunAs -Wait -PassThru `
            -WorkingDirectory $env:TEMP
    }
    catch {
        throw "${item}: the installation did not start: $($_.Exception.Message)"
    }
    if ($process.ExitCode -eq 3010) {
        throw "${item}: installed; restart Windows, which completes the installation, then run this script again"
    }
    if ($process.ExitCode -ne 0) {
        throw "${item}: the installation failed with the code $($process.ExitCode); logs: $logs"
    }
}

# The components of Visual Studio 2022 or of its Build Tools that the build needs: the C++ tools
# for x64, the C++ CMake tools, the redistributable files of the C++ runtime, which the device
# engines carry, and a Windows SDK, the first of the list when one is installed.
$vsComponents = @("Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
    "Microsoft.VisualStudio.Component.VC.CMake.Project", "Microsoft.VisualStudio.Component.VC.Redist.14.Latest")
$vsSdks = @("Microsoft.VisualStudio.Component.Windows11SDK.26100",
    "Microsoft.VisualStudio.Component.Windows11SDK.22621", "Microsoft.VisualStudio.Component.Windows11SDK.22000",
    "Microsoft.VisualStudio.Component.Windows10SDK.20348", "Microsoft.VisualStudio.Component.Windows10SDK.19041",
    "Microsoft.VisualStudio.Component.Windows10SDK.18362")

# The installations of Visual Studio 2022 and of its Build Tools, each with the components of the
# list and the Windows SDK it lacks.
function Get-VisualStudio([string[]]$components) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return @() }
    $range = @("-products", "*", "-version", "[17.0,18.0)")
    $found = @()
    foreach ($installation in @(& $vswhere @range -property installationPath)) {
        $lacks = @()
        foreach ($component in $components) {
            if (@(& $vswhere @range -requires $component -property installationPath) -notcontains $installation) {
                $lacks += $component
            }
        }
        if (@(& $vswhere @range -requires $vsSdks -requiresAny -property installationPath) -notcontains $installation) {
            $lacks += $vsSdks[0]
        }
        $found += [pscustomobject]@{ Path = $installation; Lacks = $lacks }
    }
    return $found
}

# CMake 3.21 or later: the one of the PATH, else the C++ CMake tools of an installation of Visual
# Studio; its path and its version, or $null.
function Find-CMake([object[]]$installations) {
    $candidates = @((Get-Command cmake -All -CommandType Application -ErrorAction SilentlyContinue).Source)
    foreach ($vs in $installations) {
        $candidates += Join-Path $vs.Path "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    }
    foreach ($candidate in $candidates) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate)) { continue }
        try {
            $text = "$(& $candidate --version 2>$null | Select-Object -First 1)"
        }
        catch {
            continue
        }
        if ($text -match '([0-9]+\.[0-9]+(\.[0-9]+)?)' -and [version]$Matches[1] -ge [version]"3.21") {
            return [pscustomobject]@{ Path = $candidate; Version = $Matches[1] }
        }
    }
    return $null
}

# Python 3.10 or later that runs: the interpreters the PATH names (python, python3, the launcher
# py), then the installations of python.org and winget for the user and for the machine.  The
# alias of the Microsoft Store, without Python, writes its message to the errors and is passed
# over.  The path of the interpreter itself, its version and whether it is a virtual environment,
# or $null.
function Find-Python {
    $candidates = @()
    foreach ($name in @("python", "python3", "py")) {
        $candidates += @((Get-Command $name -All -CommandType Application -ErrorAction SilentlyContinue).Source)
    }
    foreach ($pattern in @((Join-Path $env:LOCALAPPDATA "Programs\Python\Python3*\python.exe"),
            (Join-Path $env:ProgramFiles "Python3*\python.exe"))) {
        $candidates += @(Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending |
            ForEach-Object { $_.FullName })
    }
    $probe = "import sys; print('%d.%d.%d' % sys.version_info[:3]); print(sys.executable); " +
        "print(int(sys.prefix != sys.base_prefix))"
    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        try {
            $out = @(& $candidate -c $probe 2>$null)
            if ($LASTEXITCODE -eq 0 -and $out.Count -ge 3 -and [version]"$($out[0])".Trim() -ge [version]"3.10") {
                return [pscustomobject]@{
                    Path = "$($out[1])".Trim()
                    Version = "$($out[0])".Trim()
                    Virtual = "$($out[2])".Trim() -eq "1"
                }
            }
        }
        catch {
            continue
        }
    }
    return $null
}

# The PATH of the session, read again from the settings of the machine and of the user after an
# installation that changed them.
function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
        [Environment]::GetEnvironmentVariable("Path", "User")
}

# The bytes of the files of a folder, with those of its subfolders when asked; 0 when it does not
# exist.
function Get-FolderBytes([string]$folder, [switch]$Recurse) {
    if (-not (Test-Path -LiteralPath $folder)) { return [long]0 }
    $files = Get-ChildItem -LiteralPath $folder -File -Force -Recurse:$Recurse -ErrorAction SilentlyContinue
    return [long](($files | Measure-Object -Property Length -Sum).Sum)
}

# The root of the volume of a folder, which may not exist yet.
function Get-Volume([string]$folder) {
    $full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($folder)
    return [System.IO.Path]::GetPathRoot($full)
}

# The space the build adds to the volume of a folder: an estimate in GB, less what the folder holds.
$needs = @{}
function Add-Space([string]$folder, [double]$gigabytes, [long]$present) {
    $volume = Get-Volume $folder
    if (-not $script:needs.ContainsKey($volume)) { $script:needs[$volume] = [double]0 }
    $script:needs[$volume] += [math]::Max([double]0, $gigabytes * 1GB - $present)
}

# The prerequisites, each checked and installed when it is missing, unless -NoInstall.
Write-Host "== checking the prerequisites"
if ($Cuda -and $NoCuda) { throw "-Cuda and -NoCuda exclude each other" }

# The downloads take the proxy of HTTPS_PROXY, else the one of the system, with the account of
# the user for a proxy that authenticates it, and TLS 1.2; Windows checks the servers with its
# certificate store, which holds the certificate of a proxy of the company that inspects TLS.
# CMake and pip read HTTPS_PROXY alone, which receives the proxy of the system.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor
    [Net.SecurityProtocolType]::Tls12
$systemProxy = [System.Net.WebRequest]::GetSystemWebProxy()
$systemProxy.Credentials = [System.Net.CredentialCache]::DefaultNetworkCredentials
[System.Net.WebRequest]::DefaultWebProxy = $systemProxy
if (-not $env:HTTPS_PROXY) {
    $probe = [Uri]"https://developer.download.nvidia.com/"
    if (-not $systemProxy.IsBypassed($probe)) {
        $env:HTTPS_PROXY = $systemProxy.GetProxy($probe).GetLeftPart([UriPartial]::Authority)
        Write-Host "== downloads through the proxy of the system, $env:HTTPS_PROXY"
    }
}

# Visual Studio 2022 or its Build Tools: the Build Tools when no installation is there, else the
# components the installation lacks, both installed with the rights of an administrator.
$vsSetup = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\setup.exe"
$vsBootstrapper = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
$vsLogs = "the files dd_bootstrapper_*, dd_client_* and dd_setup_* of $env:TEMP"
$vsAdd = @("--add", "Microsoft.VisualStudio.Workload.VCTools")
foreach ($component in $vsComponents + $vsSdks[0]) { $vsAdd += @("--add", $component) }
# A CMake 3.21 or later on the PATH serves without the C++ CMake tools of Visual Studio.
$vsNeeded = $vsComponents
if (Find-CMake @()) {
    $vsNeeded = @($vsComponents | Where-Object { $_ -ne "Microsoft.VisualStudio.Component.VC.CMake.Project" })
}
$installations = @(Get-VisualStudio $vsNeeded)
$complete = @($installations | Where-Object { $_.Lacks.Count -eq 0 })
$vs = $null
if ($complete.Count -gt 0) { $vs = $complete[0] }
elseif ($installations.Count -gt 0) { $vs = $installations[0] }
if (-not $vs) {
    $command = "vs_BuildTools.exe $($vsAdd -join ' ') --passive --wait --norestart"
    if ($NoInstall) {
        Add-Prerequisite "Visual Studio 2022" "missing" "as an administrator, with $vsBootstrapper, $command"
    }
    else {
        $file = Join-Path $env:TEMP "vs_BuildTools.exe"
        Save-Download "Visual Studio Build Tools 2022" $vsBootstrapper $file
        $signature = Get-AuthenticodeSignature -FilePath $file
        $signer = "$($signature.SignerCertificate.Subject)"
        if ($signature.Status -ne "Valid" -or $signer -notmatch "O=Microsoft Corporation") {
            throw "Visual Studio Build Tools 2022: $file carries no valid signature of Microsoft ($($signature.Status))"
        }
        $arguments = $vsAdd + @("--passive", "--wait", "--norestart")
        Invoke-Elevated "Visual Studio Build Tools 2022" $file $arguments $vsLogs
        Add-Prerequisite "Visual Studio 2022" "installed" "Build Tools, $command"
        Write-Prerequisites
        throw ("Visual Studio Build Tools 2022 is installed; run this script again from a new Developer PowerShell " +
            "for VS 2022, whose environment knows it")
    }
}
elseif ($vs.Lacks.Count -gt 0) {
    $add = @($vs.Lacks | ForEach-Object { "--add"; $_ })
    $command = "`"$vsSetup`" modify --installPath `"$($vs.Path)`" $($add -join ' ') --passive --norestart"
    if ($NoInstall) {
        Add-Prerequisite "Visual Studio 2022" "missing" "as an administrator, $command"
    }
    else {
        Invoke-Elevated "Visual Studio 2022, $($vs.Lacks -join ', ')" $vsSetup `
            (@("modify", "--installPath", "`"$($vs.Path)`"") + $add + @("--passive", "--norestart")) $vsLogs
        $vsPath = $vs.Path
        $vs = @(Get-VisualStudio $vsNeeded | Where-Object { $_.Path -eq $vsPath }) | Select-Object -First 1
        if (-not $vs -or $vs.Lacks.Count -gt 0) {
            throw "Visual Studio 2022: $vsPath still lacks components after their installation; logs: $vsLogs"
        }
        Add-Prerequisite "Visual Studio 2022" "installed" "$vsPath, $($add -join ' ')"
    }
}
else {
    Add-Prerequisite "Visual Studio 2022" "found" $vs.Path
}

# CMake 3.21 or later: the one of the PATH, else the C++ CMake tools of Visual Studio.
$found = Find-CMake @(Get-VisualStudio $vsNeeded)
if ($found) {
    $cmake = $found.Path
    Add-Prerequisite "CMake 3.21 or later" "found" "$($found.Version), $cmake"
}
else {
    Add-Prerequisite "CMake 3.21 or later" "missing" "the C++ CMake tools of Visual Studio 2022, installed with it"
}

# Python 3.10 or later, for this user: winget, else the installer of python.org, whose SHA-256 is
# checked; neither needs the rights of an administrator.
$pythonVersion = "3.12.10"
$pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe"
$pythonSha256 = "67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb"
$winget = (Get-Command winget -ErrorAction SilentlyContinue).Source
$wingetArguments = @("install", "--id", "Python.Python.3.12", "-e", "--scope", "user", "--silent",
    "--accept-package-agreements", "--accept-source-agreements")
$python = Find-Python
if ($python) {
    Add-Prerequisite "Python 3.10 or later" "found" "$($python.Version), $($python.Path)"
}
elseif ($NoInstall) {
    $command = "python-$pythonVersion-amd64.exe /quiet InstallAllUsers=0 PrependPath=1, from $pythonUrl"
    if ($winget) { $command = "winget $($wingetArguments -join ' ')" }
    Add-Prerequisite "Python 3.10 or later" "missing" $command
}
else {
    if ($winget) {
        $log = Join-Path $env:TEMP "albireo-python-winget.log"
        Write-Host "== Python 3.12: winget $($wingetArguments -join ' ')"
        & $winget @wingetArguments --log $log
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "winget ended with the code $LASTEXITCODE (log $log); the installer of python.org follows"
        }
        Update-SessionPath
        $python = Find-Python
    }
    if (-not $python) {
        $file = Join-Path $env:TEMP "python-$pythonVersion-amd64.exe"
        Save-Download "Python $pythonVersion" $pythonUrl $file
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLower() -ne $pythonSha256) {
            throw "Python ${pythonVersion}: $file does not have the SHA-256 of the installer of python.org"
        }
        $log = Join-Path $env:TEMP "albireo-python-install.log"
        Write-Host "== Python ${pythonVersion}: installing it for this user"
        $process = Start-Process -FilePath $file -Wait -PassThru `
            -ArgumentList @("/quiet", "InstallAllUsers=0", "PrependPath=1", "/log", "`"$log`"")
        if ($process.ExitCode -ne 0) {
            throw "Python ${pythonVersion}: the installation failed with the code $($process.ExitCode); log: $log"
        }
        Update-SessionPath
        $python = Find-Python
    }
    if (-not $python) {
        throw "Python: no interpreter 3.10 or later answers after the installation; log: $log"
    }
    Add-Prerequisite "Python 3.10 or later" "installed" "$($python.Version), $($python.Path)"
}

# The runtime of OpenVINO, for the engines of the NPU and of the graphics processor of Intel: the
# package of pip for this user, of the versions the build supports.  pip 24.2 and later check the
# servers with the certificate store of Windows, and those from 22.2 on when asked.
if ($NoOpenVINO) {
    Add-Prerequisite "OpenVINO runtime" "skipped" "-NoOpenVINO"
}
else {
    $interpreter = if ($python) { $python.Path } else { "" }
    $version = Get-OpenVINO $interpreter
    $pipArguments = @("-m", "pip", "install")
    if (-not $python -or -not $python.Virtual) { $pipArguments += "--user" }
    $shown = if ($interpreter) { $interpreter } else { "python" }
    $command = "`"$shown`" $($pipArguments -join ' ') `"$openvinoRange`""
    if ($version -eq "archive") {
        # The folder the variable names, which may also be the package of an interpreter.
        $named = "OpenVINO_DIR $env:OpenVINO_DIR"
        if ($env:INTEL_OPENVINO_DIR) { $named = "INTEL_OPENVINO_DIR $env:INTEL_OPENVINO_DIR" }
        Add-Prerequisite "OpenVINO runtime" "found" $named
    }
    elseif (Test-OpenVINOVersion $version) {
        Add-Prerequisite "OpenVINO runtime" "found" "$version, package of $interpreter"
    }
    elseif ($NoInstall -or -not $python) {
        Add-Prerequisite "OpenVINO runtime" "missing" $command
    }
    else {
        $log = Join-Path $env:TEMP "albireo-pip-openvino.log"
        $pipArguments += @("--disable-pip-version-check", "--log", $log)
        try {
            $pip = "$(& $interpreter -m pip --version 2>$null)"
        }
        catch {
            $pip = ""
        }
        if ($pip -match '^pip ([0-9]+\.[0-9]+)' -and [version]$Matches[1] -ge [version]"22.2" -and
                [version]$Matches[1] -lt [version]"24.2") {
            $pipArguments += "--use-feature=truststore"
        }
        Write-Host "== OpenVINO runtime: $command"
        & $interpreter @pipArguments $openvinoRange
        if ($LASTEXITCODE -ne 0) {
            throw "OpenVINO runtime: pip failed with the code $LASTEXITCODE; log: $log"
        }
        $version = Get-OpenVINO $interpreter
        if (-not (Test-OpenVINOVersion $version)) {
            throw "OpenVINO runtime: the package of $interpreter is missing after its installation; log: $log"
        }
        Add-Prerequisite "OpenVINO runtime" "installed" "$version, package of $interpreter"
    }
}

# CUDA and cuDNN: the toolkit and cuDNN installed on this computer, else those of the
# redistributable archives of NVIDIA, CUDA 12.9 and cuDNN 9.11, which serve the GeForce GTX 900
# and 10 series, laid out in the cache once the free space is checked; a card is not needed.
$redistCuda = "12.9.1"
$redistCudnn = "9.11.0"
# The cache as an absolute path, which the toolset of Visual Studio names.
$cudaCacheFolder = Join-Path $env:LOCALAPPDATA "Albireo\build-cache\cuda"
if ($CudaCache) {
    $cudaCacheFolder = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CudaCache)
}
$redist = $null
$useRedist = $false
$useRedistCudnn = $false
$toolkit = $null
$cudnnFolder = $null
$major = ""
$buildCuda = $false
$cudaMinimum = 0
$cudaMissing = @()
if ($NoCuda) {
    Add-Prerequisite "CUDA and cuDNN" "skipped" "-NoCuda"
}
else {
    $toolkit = Find-CudaToolkit
    if ($CudaVersion) { $major = $CudaVersion.Split('.')[0] }
    elseif ($toolkit -and ($toolkit -match 'v([0-9]+)\.')) { $major = $Matches[1] }
    if (-not $toolkit) {
        if ($CudaVersion) { $cudaMissing += "the CUDA toolkit $CudaVersion" }
        else { $cudaMissing += "the CUDA toolkit 12.8 or later (CUDA_PATH)" }
    }
    else {
        if (-not (Test-CudaIntegration)) {
            $cudaMissing += ("the integration of CUDA with Visual Studio (Visual Studio Integration in the " +
                "toolkit installer)")
        }
        $cudnnFolder = Find-Cudnn $toolkit $major
        if (-not $cudnnFolder) { $cudaMissing += "cuDNN 9 (its installer, or -Cudnn with the folder of its archive)" }
    }
    if ($cudaMissing.Count -eq 0) {
        $buildCuda = $true
        Add-Prerequisite "CUDA and cuDNN" "found" "toolkit $toolkit, cuDNN $cudnnFolder"
        # cuDNN 9.12 and later leave out the GeForce GTX 900 and 10 series, which CUDA 12 still
        # compiles for: the archives of NVIDIA give cuDNN 9.11 to the installed toolkit.
        $installedCudnn = Get-CudnnVersion $cudnnFolder $major
        if (-not $Cudnn -and -not $CudaArchitectures -and $major -eq "12" -and $installedCudnn -and
            $installedCudnn -ge [version]"9.12") {
            $useRedistCudnn = $true
        }
    }
    elseif (-not $CudaVersion -or "$redistCuda.".StartsWith("$CudaVersion.")) {
        $useRedist = $true
    }
    elseif ($Cuda) {
        throw ("the CUDA engine needs " + ($cudaMissing -join ", "))
    }
    else {
        Write-Warning ("no CUDA engine in this build, this computer lacks " + ($cudaMissing -join ", ") +
            ". The toolkit installs without an NVIDIA card, and the redistributable archives of NVIDIA serve " +
            "without -CudaVersion or with -CudaVersion 12.9; -NoCuda leaves the engine out without this message.")
        Add-Prerequisite "CUDA and cuDNN" "missing" ($cudaMissing -join ", ")
    }
}

# The free space of each volume against what the build adds to it, estimated in GB, less what
# each folder holds already: the tree of the programs and of the installers (3.5, and 3.5 more for
# the models in the complete installer, 3.5 more again for their copies in the tree when their
# cache lies on another volume), the trees of OpenVINO (1) and of CUDA (2), the models that come
# with Albireo in their cache (6.8) and the archives of NVIDIA laid out (4.2), or cuDNN alone (1.5).
$main = Join-Path $root "build-albireo"
$modelCacheFolder = if ($ModelCache) { $ModelCache } else { Join-Path $env:LOCALAPPDATA "Albireo\build-cache" }
$programs = 3.5
if (-not $NoBuiltinModels) {
    $programs += 3.5
    if ((Get-Volume $main) -ne (Get-Volume $modelCacheFolder)) { $programs += 3.5 }
    Add-Space $modelCacheFolder 6.8 (Get-FolderBytes $modelCacheFolder)
}
Add-Space $main $programs (Get-FolderBytes $main -Recurse)
if (-not $NoOpenVINO) {
    $tree = Join-Path $root "build-albireo-ov"
    Add-Space $tree 1 (Get-FolderBytes $tree -Recurse)
}
if ($buildCuda -or $useRedist) {
    $tree = Join-Path $root "build-albireo-cuda"
    Add-Space $tree 2 (Get-FolderBytes $tree -Recurse)
}
if ($useRedist) { Add-Space $cudaCacheFolder 4.2 (Get-FolderBytes $cudaCacheFolder -Recurse) }
if ($useRedistCudnn) { Add-Space $cudaCacheFolder 1.5 (Get-FolderBytes $cudaCacheFolder -Recurse) }
$short = @()
foreach ($volume in @($needs.Keys | Sort-Object)) {
    try {
        $free = ([System.IO.DriveInfo]::new($volume)).AvailableFreeSpace
    }
    catch {
        Add-Prerequisite "Free space on $volume" "skipped" "not measured: $($_.Exception.Message)"
        continue
    }
    $text = "{0:N1} GB free, {1:N1} GB needed" -f ($free / 1GB), ($needs[$volume] / 1GB)
    if ($free -lt $needs[$volume]) {
        Add-Prerequisite "Free space on $volume" "short" $text
        $short += $volume
    }
    else {
        Add-Prerequisite "Free space on $volume" "found" $text
    }
}

# The archives of NVIDIA, laid out when the free space allows it; -NoInstall takes those already there.
if ($useRedist) {
    $cudnnStamp = "cudnn-$redistCudnn-cuda$($redistCuda.Split('.')[0])\albireo-redist.stamp"
    $laidOut = (Test-Path -LiteralPath (Join-Path $cudaCacheFolder "cuda-$redistCuda\albireo-redist.stamp")) -and
        (Test-Path -LiteralPath (Join-Path $cudaCacheFolder $cudnnStamp))
    $command = ("`"$(if ($cmake) { $cmake } else { 'cmake' })`" `"-DCACHE=$cudaCacheFolder`" " +
        "-DCUDA_VERSION=$redistCuda -DCUDNN_VERSION=$redistCudnn -P `"$(Join-Path $source 'cmake\cuda_redist.cmake')`"")
    if (($NoInstall -and -not $laidOut) -or -not $cmake) {
        Add-Prerequisite "CUDA and cuDNN" "missing" ("the archives of NVIDIA, for " + ($cudaMissing -join ", ") +
            ": $command")
    }
    elseif ($short.Count -gt 0) {
        Add-Prerequisite "CUDA and cuDNN" "missing" "the archives of NVIDIA, laid out once the free space is there"
    }
    else {
        $redist = Get-CudaRedist $cudaCacheFolder
        if ($redist) {
            $toolkit = $redist.Toolkit
            $cudnnFolder = $redist.Cudnn
            $major = $redistCuda.Split('.')[0]
            $buildCuda = $true
            $state = "found"
            if ($redist.Downloaded -gt 0) {
                $state = "installed"
                Write-Host ("== CUDA archives: $([math]::Round($redist.Downloaded / 1MB)) MB downloaded into " +
                    "$cudaCacheFolder, which the next builds reuse")
            }
            $what = "archives of NVIDIA, CUDA $redistCuda and cuDNN $redistCudnn in $cudaCacheFolder"
            Add-Prerequisite "CUDA and cuDNN" $state $what
        }
        elseif ($Cuda) {
            throw ("the CUDA engine needs " + ($cudaMissing -join ", ") + ", or the redistributable archives of " +
                "NVIDIA, which could not be laid out in $cudaCacheFolder (see the messages above)")
        }
        else {
            Write-Warning ("no CUDA engine in this build, this computer lacks " + ($cudaMissing -join ", ") +
                ", and the redistributable archives of NVIDIA could not be laid out in $cudaCacheFolder (see the " +
                "messages above); -NoCuda leaves the engine out without this message.")
            Add-Prerequisite "CUDA and cuDNN" "missing" "the archives of NVIDIA, not laid out (see the messages above)"
        }
    }
}

# cuDNN 9.11 for the installed toolkit, laid out when the free space allows it; otherwise the
# engine keeps the cuDNN installed, and the warning of the end names the cards it leaves out.
if ($useRedistCudnn) {
    $item = "cuDNN for GTX 900 and 10"
    $cudnnStamp = "cudnn-$redistCudnn-cuda$major\albireo-redist.stamp"
    $laidOut = Test-Path -LiteralPath (Join-Path $cudaCacheFolder $cudnnStamp)
    $command = ("`"$(if ($cmake) { $cmake } else { 'cmake' })`" `"-DCACHE=$cudaCacheFolder`" " +
        "-DCUDA_VERSION=$redistCuda -DCUDNN_VERSION=$redistCudnn -DCUDNN_ONLY=ON " +
        "-P `"$(Join-Path $source 'cmake\cuda_redist.cmake')`"")
    if (($NoInstall -and -not $laidOut) -or -not $cmake) {
        Add-Prerequisite $item "skipped" "cuDNN $installedCudnn kept; the archive of cuDNN ${redistCudnn}: $command"
    }
    elseif ($short.Count -gt 0) {
        Add-Prerequisite $item "skipped" "cuDNN $installedCudnn kept, the free space lacks for cuDNN $redistCudnn"
    }
    else {
        $cudnnRedist = Get-CudnnRedist $cudaCacheFolder
        if ($cudnnRedist) {
            $cudnnFolder = $cudnnRedist.Cudnn
            $state = if ($cudnnRedist.Downloaded -gt 0) { "installed" } else { "found" }
            Add-Prerequisite $item $state ("cuDNN $redistCudnn of the archives of NVIDIA in $cudaCacheFolder, " +
                "in place of cuDNN $installedCudnn")
        }
        else {
            Write-Warning ("cuDNN $redistCudnn could not be laid out in $cudaCacheFolder (see the messages above): " +
                "the CUDA engine keeps cuDNN $installedCudnn")
            Add-Prerequisite $item "skipped" "cuDNN $installedCudnn kept (see the messages above)"
        }
    }
}

Write-Prerequisites
if ($short.Count -gt 0) {
    throw ("the free space is short on " + ($short -join ", ") + " (see the table): free some space there, or " +
        "place the caches on another volume with -ModelCache and -CudaCache")
}
if ($NoInstall -and @($prerequisites | Where-Object { $_.State -eq "missing" }).Count -gt 0) {
    throw "prerequisites are missing (see the table): install them with the commands given, then run this script again"
}
if (-not $cmake) { throw "CMake 3.21 or later was not found (see the table)" }

$extra = @()
if (-not $NoOpenVINO) {
    $tree = Join-Path $root "build-albireo-ov"
    Reset-StaleTree $tree
    # The interpreter of the check of the prerequisites, through which dlib finds the package of OpenVINO.
    $options = @("-DALBIREO_PROGRAMS=OFF", "-DALBIREO_ENGINE_OPENVINO=ON", "-DALBIREO_TESTS=$testing")
    if ($python) { $options += "-DPython3_EXECUTABLE=$($python.Path.Replace('\', '/'))" }
    Invoke-Step "configuring the OpenVINO engine" (@("-S", $source, "-B", $tree) + $generator + $options)
    Clear-Engines $tree
    Invoke-Step "building the OpenVINO engine" (@("--build", $tree, "--config", "Release") + $parallel)
    $extra += (Join-Path $tree "bin\engines")
}
if ($buildCuda) {
    $tree = Join-Path $root "build-albireo-cuda"
    # The CUDA integration of Visual Studio: the one of the archives, in the folder of their toolkit
    # (a standalone toolkit), else the one of the version asked for.
    $toolsetName = ""
    if ($redist) { $toolsetName = "cuda=$toolkit" }
    elseif ($CudaVersion) { $toolsetName = "cuda=$CudaVersion" }
    $toolset = @()
    if ($toolsetName) { $toolset = @("-T", $toolsetName) }
    Reset-StaleTree $tree $toolsetName
    $options = @("-DALBIREO_PROGRAMS=OFF", "-DALBIREO_ENGINE_CUDA=ON", "-DALBIREO_TESTS=$testing",
        "-DCUDAToolkit_ROOT=$($toolkit.Replace('\', '/'))")
    $header = Find-CudnnFile $cudnnFolder "cudnn.h" $major
    $options += "-DCUDNN_INCLUDE_PATH=$(Split-Path $header -Parent)".Replace('\', '/')
    $options += "-DCUDNN_LIBRARY_PATH=$(Find-CudnnFile $cudnnFolder 'cudnn.lib' $major)"
    $origin = if ($redist) { " (NVIDIA redistributable archives)" } else { "" }
    Write-Host "== CUDA engine: toolkit $toolkit$origin, cuDNN $(Split-Path $header -Parent)"
    # The architectures of each build: those given, else the default list of the toolkit, never
    # the list an earlier configuration of the tree kept.
    if ($CudaArchitectures) { $options += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" }
    else { $options += "-UCMAKE_CUDA_ARCHITECTURES" }
    Invoke-Step "configuring the CUDA engine" (@("-S", $source, "-B", $tree) + $generator + $toolset + $options)
    $cudaMinimum = Get-CudaMinimum $tree
    if ($cudaMinimum -gt 61) { Write-CudaWarning $cudaMinimum }
    Clear-Engines $tree
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
        catch {
            # A copy takes the name of the cache once complete: a copy cut short leaves no file of full name.
            $partial = "$target.partial"
            Copy-Item -LiteralPath $found.FullName -Destination $partial -Force
            Move-Item -LiteralPath $partial -Destination $target
        }
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
# The payload of the installer depends on the programs of this tree only: deleted, it is packed
# again with the engines the device trees have just built.
$payload = Join-Path $main "payload"
if ($extra.Count -gt 0 -and (Test-Path -LiteralPath $payload)) {
    Remove-Item -LiteralPath $payload -Recurse -Force
}
Invoke-Step "building the programs, the models and the installers" (@("--build", $main, "--config", "Release") +
    $parallel)

# The models of a complete installer that would reach the 4 GiB Windows runs lie beside it, in albireo-setup.models.
foreach ($name in @("albireo-setup.exe", "albireo-setup.models", "albireo-setup-online.exe")) {
    $setup = Join-Path $main "bin\$name"
    if (-not (Test-Path -LiteralPath $setup)) { continue }
    $hash = (Get-FileHash -Algorithm SHA256 $setup).Hash.ToLower()
    $size = (Get-Item $setup).Length
    Write-Host "== $setup"
    Write-Host "   $size bytes, SHA-256 $hash"
}
if ($cudaMinimum -gt 61) { Write-CudaWarning $cudaMinimum }
