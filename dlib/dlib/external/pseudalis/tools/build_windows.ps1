# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
<#
    Builds Pseudalis for Windows with Visual Studio 2022 or its Build Tools: the service, its
    two engines (every x64 processor, and the processors with AVX2), the icon, the commands
    of the Explorer and the installer, which carries them with the web application and the
    licenses.

        powershell -ExecutionPolicy Bypass -File build_windows.ps1 [-Tests] [-Install] [-Jobs <n>] [-AnyFolder]

    Requirements: Visual Studio 2022, or the Build Tools for Visual Studio 2022, with the
    C++ workload, and CMake 3.21 or later (the C++ CMake tools of the workload, found
    without the PATH).  Nothing else: dlib, Mbed TLS, cpp-httplib, SQLite and zlib come with
    the repository, and the drives need no driver.

    The update feed is checked with the public key of resources\feed-public-key.pem, which
    the publisher makes once on Linux (tools/publish_release.py --create-key) and commits:
    without it, the build goes on and the programs install no update.

    The programs link the runtime of the compiler statically: the build stops when a program, an
    engine or the installer imports another library than those of Windows (tools\check_imports.cpp),
    so that the installer carries every file the programs need and no library of a third party.

    -Tests also builds and runs the tests of the engine and of the service.  -Install then
    runs the installer built and waits for it, which updates the installation in place; a
    failure of a step stops the script before it.  -Jobs bounds the number of compilations
    run at once, for a machine with little memory.  The build tree is build-pseudalis at the
    root of the repository; the installer is build-pseudalis\bin\pseudalis-setup.exe, whose
    path under WSL the script prints for the publication (tools/publish_release.py).

    The production chain works in one copy of the repository on Windows,
    %USERPROFILE%\work\Dlib-Transformer-extensions, whose installer the publication reads: the
    script stops when it runs from another copy, unless -AnyFolder is given for a build that is
    not to be published.
#>
param(
    [switch]$Tests,
    [switch]$Install,
    [int]$Jobs = 0,
    [switch]$AnyFolder
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$source = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$root = (Resolve-Path (Join-Path $source "..\..\..\..")).Path
$chain = Join-Path $env:USERPROFILE "work\Dlib-Transformer-extensions"
if (-not $AnyFolder -and
    [System.IO.Path]::GetFullPath($root).TrimEnd('\') -ne [System.IO.Path]::GetFullPath($chain).TrimEnd('\')) {
    throw "the production chain builds in $chain, not in $root`: run the script of that copy, or give -AnyFolder"
}
$tree = Join-Path $root "build-pseudalis"
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
function Reset-StaleTree([string]$folder) {
    $cache = Join-Path $folder "CMakeCache.txt"
    if (-not (Test-Path $cache)) { return }
    $line = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR_INSTANCE:[A-Z]+=(.+)$' | Select-Object -First 1
    if (-not $line) { return }
    $instance = $line.Matches[0].Groups[1].Value.Trim()
    if ($instance -and -not (Test-Path $instance)) {
        Write-Host "== $folder was configured by $instance, which is no longer installed: creating it again"
        Remove-Item -Recurse -Force $folder
    }
}

function Get-Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$publicKey = Join-Path $source "resources\feed-public-key.pem"
if (-not (Test-Path -LiteralPath $publicKey)) {
    Write-Warning ("no $publicKey`: the programs will install no update. The publisher makes it once with " +
        "pseudalis-sign keygen <private key> $publicKey, then commits it")
}

Reset-StaleTree $tree
Invoke-Step "configuring $tree" @("-S", $source, "-B", $tree, "-G", "Visual Studio 17 2022", "-A", "x64",
    "-DPSEUDALIS_TESTS=$testing")
Invoke-Step "building Pseudalis" (@("--build", $tree, "--config", "Release") + $parallel)

if ($Tests) {
    foreach ($test in @("test_engine.exe", "test_host.exe")) {
        $program = Join-Path $tree "bin\$test"
        Write-Host "== $test"
        & $program
        if ($LASTEXITCODE -ne 0) { throw "$test failed" }
    }
}

$setup = Join-Path $tree "bin\pseudalis-setup.exe"
if (-not (Test-Path -LiteralPath $setup)) { throw "the build made no $setup" }
$size = (Get-Item -LiteralPath $setup).Length
Write-Host ("== {0}: {1:N1} MB, SHA-256 {2}" -f $setup, ($size / 1MB), (Get-Sha256 $setup))
# The installer as WSL reads it, for the publication.
if ($setup -match '^([A-Za-z]):\\(.*)$') {
    Write-Host ("== under WSL: /mnt/{0}/{1}" -f $Matches[1].ToLowerInvariant(), ($Matches[2] -replace '\\', '/'))
}

if ($Install) {
    Write-Host "== installing $setup"
    $process = Start-Process -FilePath $setup -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "the installer ended with the code $($process.ExitCode)" }
}
