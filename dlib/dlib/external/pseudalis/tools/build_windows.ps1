# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
<#
    Builds Pseudalis for Windows with Visual Studio 2022 or its Build Tools: the service, its
    two engines (every x64 processor, and the processors with AVX2), the icon, the commands
    of the Explorer and the installer, which carries them with the web application, the
    licenses and the installer of WinFsp.

        powershell -ExecutionPolicy Bypass -File build_windows.ps1 [-Tests] [-Jobs <n>]

    Requirements: Visual Studio 2022, or the Build Tools for Visual Studio 2022, with the
    C++ workload, and CMake 3.21 or later (the C++ CMake tools of the workload, found
    without the PATH).  Nothing else: dlib, Mbed TLS, cpp-httplib, SQLite and zlib come with
    the repository, and the files of WinFsp 2.1 (its installer, the two headers the drive
    compiles with, its license) are downloaded once from its repository and its release
    into build-pseudalis\winfsp, each checked against its SHA-256.  Behind a proxy, the
    downloads take the one of HTTPS_PROXY.

    The update feed is checked with the public key of resources\feed-public-key.pem, which
    the publisher makes once with pseudalis-sign keygen and commits: without it, the build
    goes on and the programs install no update.

    -Tests also builds and runs the tests of the engine and of the service.  -Jobs bounds
    the number of compilations run at once, for a machine with little memory.  The build
    tree is build-pseudalis at the root of the repository; the installer is
    build-pseudalis\bin\pseudalis-setup.exe.
#>
param(
    [switch]$Tests,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$source = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$root = (Resolve-Path (Join-Path $source "..\..\..\..")).Path
$tree = Join-Path $root "build-pseudalis"
$testing = if ($Tests) { "ON" } else { "OFF" }
$parallel = if ($Jobs -gt 0) { @("--parallel", "$Jobs") } else { @("--parallel") }

# The files of WinFsp 2.1 (build 25156), with their SHA-256.
$winfspVersion = "2.1.25156"
$winfspFiles = @(
    @{ Url = "https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi"; Path = "winfsp.msi";
       Sha256 = "073a70e00f77423e34bed98b86e600def93393ba5822204fac57a29324db9f7a" },
    @{ Url = "https://raw.githubusercontent.com/winfsp/winfsp/v2.1/inc/winfsp/winfsp.h"; Path = "inc\winfsp\winfsp.h";
       Sha256 = "163332ebe058dd6e49d038edefef273454912c2ec44e8430a18ed73112e5678c" },
    @{ Url = "https://raw.githubusercontent.com/winfsp/winfsp/v2.1/inc/winfsp/fsctl.h"; Path = "inc\winfsp\fsctl.h";
       Sha256 = "7a83a69cdeb155751032eedbbb6875234b1e0824be7cf2d7ba23c1ef44944651" },
    @{ Url = "https://raw.githubusercontent.com/winfsp/winfsp/v2.1/License.txt"; Path = "License.txt";
       Sha256 = "68ae2fa9d8e6110ebdc9a236beecf118e279f8a26530daf09c9e682c231f7879" }
)

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

# Downloads a file once: a file in place with the expected SHA-256 stays, another is replaced.
function Get-CheckedFile([string]$url, [string]$path, [string]$sha256) {
    if ((Test-Path -LiteralPath $path) -and (Get-Sha256 $path) -eq $sha256) { return }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
    $partial = "$path.partial"
    Write-Host "== downloading $url"
    $parameters = @{ Uri = $url; OutFile = $partial; UseBasicParsing = $true }
    if ($env:HTTPS_PROXY) { $parameters.Proxy = $env:HTTPS_PROXY }
    Invoke-WebRequest @parameters
    $found = Get-Sha256 $partial
    if ($found -ne $sha256) {
        Remove-Item -LiteralPath $partial -Force
        throw "$url has the SHA-256 $found, not $sha256"
    }
    Move-Item -LiteralPath $partial -Destination $path -Force
}

# Windows PowerShell 5.1 may offer TLS 1.0 alone, which GitHub refuses.
$protocols = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
[Net.ServicePointManager]::SecurityProtocol = $protocols
$winfsp = Join-Path $tree "winfsp"
foreach ($f in $winfspFiles) {
    Get-CheckedFile $f.Url (Join-Path $winfsp $f.Path) $f.Sha256
}
Write-Host "== WinFsp $winfspVersion in $winfsp"

$publicKey = Join-Path $source "resources\feed-public-key.pem"
if (-not (Test-Path -LiteralPath $publicKey)) {
    Write-Warning ("no $publicKey`: the programs will install no update. The publisher makes it once with " +
        "pseudalis-sign keygen <private key> $publicKey, then commits it")
}

Reset-StaleTree $tree
$winfspInclude = (Join-Path $winfsp "inc").Replace('\', '/')
$winfspMsi = (Join-Path $winfsp "winfsp.msi").Replace('\', '/')
$winfspLicense = (Join-Path $winfsp "License.txt").Replace('\', '/')
Invoke-Step "configuring $tree" @("-S", $source, "-B", $tree, "-G", "Visual Studio 17 2022", "-A", "x64",
    "-DPSEUDALIS_TESTS=$testing", "-DPSEUDALIS_WINFSP_INCLUDE=$winfspInclude", "-DPSEUDALIS_WINFSP_MSI=$winfspMsi",
    "-DPSEUDALIS_WINFSP_LICENSE=$winfspLicense")
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
