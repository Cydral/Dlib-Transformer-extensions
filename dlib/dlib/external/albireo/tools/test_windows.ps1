# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
<#
    Tests a Windows build of Albireo and prints a summary, one line per step.

        powershell -ExecutionPolicy Bypass -File test_windows.ps1 [-Model <model.gguf>]
            [-EmbeddingModel <model.gguf>] [-Device CPU|GPU|NPU] [-NoService]

    Run from the root of the repository after build_windows.ps1 -Tests.  Defaults:
    $HOME\models\Qwen3-0.6B-Q4_K_M.gguf and $HOME\models\Qwen3-Embedding-0.6B-Q8_0.gguf; the
    steps whose model or build tree is missing are skipped.

    Steps: the quantized kernels; the loading of damaged model files and of a file of more
    than 2 GiB; a question to each processor engine; with the OpenVINO tree, test_offload on
    the device of -Device (CPU by default, which every machine has) and a question to the
    OpenVINO engine; with the CUDA tree, when the machine has a card it supports, a question to
    the CUDA engine, then test_device_ops, test_offload on the card and its check of the attention
    of the card on a tiny model; the embedding model;
    the models that come with Albireo, prepared by the build, and a question to their language
    model, these engine steps with the PATH of the service, System32 and the Windows folder
    only; a sentence read aloud by the program of the speech synthesis; then the installed
    service, unless -NoService, once it has started: its state, its models, its engines, each
    described within its time limit, with the devices they found, the models that came with
    it, a question and the vectors of two texts through the OpenAI interface, and a sentence
    read aloud through its speech route.
    The exit code is the number of failed steps.
#>
param(
    [string]$Model = (Join-Path $HOME "models\Qwen3-0.6B-Q4_K_M.gguf"),
    [string]$EmbeddingModel = (Join-Path $HOME "models\Qwen3-Embedding-0.6B-Q8_0.gguf"),
    [string]$Device = "CPU",
    [switch]$NoService
)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
$main = Join-Path $root "build-albireo"
$openvino = Join-Path $root "build-albireo-ov"
$cuda = Join-Path $root "build-albireo-cuda"
$question = "What is the capital of France? Answer in one sentence."
$logs = Join-Path ([System.IO.Path]::GetTempPath()) "albireo-tests"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$results = New-Object System.Collections.ArrayList

function Add-Result([string]$step, [string]$result, [string]$detail, [double]$seconds) {
    $line = $detail
    if ($line.Length -gt 70) { $line = $line.Substring(0, 70) }
    [void]$results.Add([pscustomobject]@{ Step = $step; Result = $result; Seconds = [math]::Round($seconds, 1);
        Detail = $line })
}

# Runs a program, then checks that its output holds a pattern.
function Invoke-Check([string]$step, [string]$pattern, [string]$program, [string[]]$arguments) {
    if (-not (Test-Path $program)) {
        Add-Result $step "SKIP" "no $program" 0
        return $null
    }
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & $program @arguments 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $watch.Stop()
    $text = $output -join "`n"
    $found = @($output | Where-Object { $_ -match $pattern })
    if ($code -eq 0 -and $found.Count -gt 0) {
        Add-Result $step "PASS" $found[0] $watch.Elapsed.TotalSeconds
    }
    else {
        $log = Join-Path $logs (($step -replace '[^A-Za-z0-9]+', '-') + ".log")
        Set-Content -Path $log -Value $text -Encoding UTF8
        Add-Result $step "FAIL" "exit $code, output in $log" $watch.Elapsed.TotalSeconds
    }
    return $text
}

$haveModel = Test-Path $Model
$chat = Join-Path $main "tests\engine_chat.exe"
# The engine steps run engine_chat.exe, which build_windows.ps1 -Tests builds.
$haveChat = Test-Path $chat
$noChat = "no $chat (build_windows.ps1 -Tests)"

# The kernels of each family of processors, skipped by the processors without their instructions.
foreach ($family in @("", "_avxvnni")) {
    $test = Join-Path $main "tests\test_qkernels$family.exe"
    $name = "quantized kernels" + $family.Replace("_", " ")
    if (-not (Test-Path $test)) {
        if (-not $family) { Add-Result $name "SKIP" "no $test (ALBIREO_TESTS=ON)" 0 }
        continue
    }
    $out = (& $test 2>&1 | ForEach-Object { "$_" }) -join "`n"
    if ($out -match "skipped, the processor lacks ([^(]*)") { Add-Result $name "SKIP" "lacks $($Matches[1])" 0 }
    else { Invoke-Check $name "all checks passed" $test @() | Out-Null }
}

# The loading of damaged model files and of a model file of more than 2 GiB, mapped and streamed.
Invoke-Check "model files" "all checks passed" (Join-Path $main "tests\test_model_file.exe") @($logs) | Out-Null

# The attention of the steps split among the threads, against one task per key/value head, with keys and values
# in half precision and in 8 bits, on a tiny model written in the folder of the logs.
$runtimeTest = Join-Path $main "tests\test_runtime.exe"
if (Test-Path $runtimeTest) {
    Invoke-Check "attention" "all checks passed" $runtimeTest @("synthetic", $logs) | Out-Null
}
else { Add-Result "attention" "SKIP" "no $runtimeTest (build_windows.ps1 -Tests)" 0 }

# The engine steps run with the PATH of a service, System32 and the Windows folder only: the
# libraries an engine loads by name must come from the folders of Albireo, not from the PATH of
# this computer.
$savedPath = $env:PATH
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
try {
    foreach ($variant in @("cpu-avxvnni", "cpu-avx2", "cpu")) {
        $engine = Join-Path $main "bin\engines\albireo-engine-$variant.dll"
        if (-not $haveModel) { Add-Result "engine $variant" "SKIP" "no $Model" 0; continue }
        if (-not (Test-Path $engine)) { Add-Result "engine $variant" "SKIP" "no $engine" 0; continue }
        if (-not $haveChat) { Add-Result "engine $variant" "SKIP" $noChat 0; continue }
        $description = (& $chat $engine --describe 2>&1 | ForEach-Object { "$_" }) -join "`n"
        if ($LASTEXITCODE -eq 3) {
            $reason = if ($description -match "lacks ([^(]*)") { "lacks $($Matches[1])" } else { "not usable here" }
            Add-Result "engine $variant" "SKIP" $reason 0
            continue
        }
        Invoke-Check "engine $variant" "Paris" $chat @($engine, $Model, "--reasoning", "off", "--max", "40",
            $question) | Out-Null
    }

    $ovEngine = Join-Path $openvino "bin\engines\albireo-engine-openvino.dll"
    if ((Test-Path $ovEngine) -and $haveModel) {
        $env:DLIB_NPU_DEVICE = $Device
        Invoke-Check "offload $Device" "all checks passed" (Join-Path $openvino "tests\test_offload.exe") @($Model,
            $Device) | Out-Null
        Remove-Item Env:DLIB_NPU_DEVICE
        # One OpenVINO engine per family of processor kernels, skipped by the processors that lack them.
        foreach ($variant in @("openvino-avxvnni", "openvino")) {
            $engine = Join-Path $openvino "bin\engines\albireo-engine-$variant.dll"
            if (-not (Test-Path $engine)) { Add-Result "engine $variant" "SKIP" "no $engine" 0; continue }
            if (-not $haveChat) { Add-Result "engine $variant" "SKIP" $noChat 0; continue }
            $description = (& $chat $engine --describe 2>&1 | ForEach-Object { "$_" }) -join "`n"
            if ($LASTEXITCODE -eq 3 -and $description -match "lacks ([^(]*)") {
                Add-Result "engine $variant" "SKIP" "lacks $($Matches[1])" 0
                continue
            }
            $arguments = @($engine, $Model, "--reasoning", "off", "--max", "40")
            if ($Device -ne "CPU") { $arguments += @("--device", $Device) }
            Invoke-Check "engine $variant" "Paris" $chat ($arguments + $question) | Out-Null
        }
    }
    else {
        Add-Result "engine openvino" "SKIP" "no OpenVINO tree or no model" 0
    }

    $cudaEngine = Join-Path $cuda "bin\engines\albireo-engine-cuda.dll"
    if ((Test-Path $cudaEngine) -and $haveModel -and -not $haveChat) {
        Add-Result "engine cuda" "SKIP" $noChat 0
    }
    elseif ((Test-Path $cudaEngine) -and $haveModel) {
        # A machine without a supported card: the engine says so in its description.
        $description = & $chat $cudaEngine $Model --max 1 "Hello" 2>&1 | ForEach-Object { "$_" } |
            Where-Object { $_ -match '^\{' } | Select-Object -First 1
        if ("$description" -match '"usable": *false') {
            $reason = ""
            if ("$description" -match '"reason": *"([^"]*)"') { $reason = $Matches[1] }
            Add-Result "engine cuda" "SKIP" "unusable here: $reason" 0
        }
        else {
            Invoke-Check "engine cuda" "Paris" $chat @($cudaEngine, $Model, "--reasoning", "off", "--max", "40",
                $question) | Out-Null
            # The operations of dlib on the weights in their GGUF formats and of the attention, then the
            # products and the layers of the card against the processor, and the attention of the card
            # on a tiny model written in the folder of the logs.
            Invoke-Check "cuda operations" "all checks passed" (Join-Path $cuda "tests\test_device_ops.exe") @() |
                Out-Null
            Invoke-Check "offload cuda" "all checks passed" (Join-Path $cuda "tests\test_offload.exe") @($Model) |
                Out-Null
            Invoke-Check "attention cuda" "all checks passed" (Join-Path $cuda "tests\test_offload.exe") @(
                "synthetic", $logs) | Out-Null
        }
    }
    else {
        Add-Result "engine cuda" "SKIP" "no CUDA tree or no model" 0
    }

    $avx2 = Join-Path $main "bin\engines\albireo-engine-cpu-avx2.dll"
    if (Test-Path $EmbeddingModel) {
        Invoke-Check "embedding model" "cosine" $chat @($avx2, $EmbeddingModel, "--parallel",
            "The service runs on this computer.", "Where does the service run?",
            "The quarterly revenue grew.") | Out-Null
    }
    else {
        Add-Result "embedding model" "SKIP" "no $EmbeddingModel" 0
    }

    # The models that come with Albireo: their manifest in the programs of the build, the complete
    # installer that carries them, or the file of its models beside it when both would reach the
    # 4 GiB Windows runs (albireo-setup.models), and a question to their language model.
    $builtinDir = Join-Path $main "bin\models"
    $manifest = Join-Path $builtinDir "builtin.json"
    if (Test-Path $manifest) {
        $builtin = Get-Content -Raw -Path $manifest | ConvertFrom-Json
        $ids = @($builtin.models | ForEach-Object { $_.id })
        $complete = Join-Path $main "bin\albireo-setup.exe"
        $online = Join-Path $main "bin\albireo-setup-online.exe"
        $beside = Join-Path $main "bin\albireo-setup.models"
        $bytes = 0
        foreach ($m in $builtin.models) {
            $bytes += [int64]$m.size
            if ($m.projector) { $bytes += [int64]$m.projector.size }
        }
        $result = "FAIL"
        $detail = $ids -join ", "
        if ((Test-Path $complete) -and (Test-Path $online)) {
            $carried = (Get-Item $complete).Length - (Get-Item $online).Length
            if (Test-Path $beside) {
                $carried = (Get-Item $beside).Length
                $detail += " (beside the installer, in albireo-setup.models)"
            }
            if ($carried -gt $bytes) { $result = "PASS" }
        }
        Add-Result "builtin models" $result $detail 0
        $language = @($builtin.models | Where-Object { -not $_.embedding } | Select-Object -First 1)
        if ($language.Count -gt 0) {
            Invoke-Check "builtin chat" "Paris" $chat @($avx2, (Join-Path $builtinDir $language[0].file), "--reasoning",
                "off", "--max", "40", $question) | Out-Null
        }
    }
    else {
        Add-Result "builtin models" "SKIP" "no $manifest (a build with -NoBuiltinModels)" 0
    }
}
finally {
    $env:PATH = $savedPath
}

# The bytes of a RIFF WAVE file, and the seconds of its samples of 16 bits at 16 kHz.
function Test-Wave([byte[]]$bytes) {
    return $bytes.Length -gt 44 -and [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4) -eq "RIFF"
}
$sentence = "Bonjour, Albireo lit ses r" + [char]0x00E9 + "ponses " + [char]0x00E0 + " voix haute."

# The program of the speech synthesis alone: a sentence read aloud in the folder of the voices.
$speaker = Join-Path $main "bin\albireo-speech.exe"
$voices = Join-Path $main "bin\speech"
if ((Test-Path $speaker) -and (Test-Path $voices)) {
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $speaker
        $info.Arguments = "fr-FR 1"
        $info.WorkingDirectory = $voices
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardInput = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = [System.Diagnostics.Process]::Start($info)
        $errors = $process.StandardError.ReadToEndAsync()
        $text = [System.Text.Encoding]::UTF8.GetBytes($sentence)
        $process.StandardInput.BaseStream.Write($text, 0, $text.Length)
        $process.StandardInput.Close()
        $sound = New-Object System.IO.MemoryStream
        $process.StandardOutput.BaseStream.CopyTo($sound)
        $process.WaitForExit()
        $bytes = $sound.ToArray()
        if ($process.ExitCode -eq 0 -and (Test-Wave $bytes)) {
            Add-Result "speech program" "PASS" ("RIFF, {0:N1} s of sound" -f (($bytes.Length - 44) / 32000)) `
                $watch.Elapsed.TotalSeconds
        }
        else {
            Add-Result "speech program" "FAIL" "exit $($process.ExitCode): $($errors.Result)" `
                $watch.Elapsed.TotalSeconds
        }
    }
    catch {
        Add-Result "speech program" "FAIL" "$($_.Exception.Message)" $watch.Elapsed.TotalSeconds
    }
}
else {
    Add-Result "speech program" "SKIP" "no $speaker or no $voices" 0
}

if (-not $NoService) {
    $service = Get-Service -Name Albireo -ErrorAction SilentlyContinue
    if (-not $service) {
        Add-Result "service" "SKIP" "Albireo is not installed (albireo-setup.exe)" 0
    }
    else {
        $port = 8443
        $settings = Join-Path $env:ProgramData "Albireo\settings.json"
        if (Test-Path $settings) {
            $saved = Get-Content -Raw -Path $settings | ConvertFrom-Json
            if ($saved.port) { $port = [int]$saved.port }
        }
        $base = "https://localhost:$port"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        # A service that starts answers with the state "starting" until its models are loaded,
        # which a slow computer makes long.
        $deadline = (Get-Date).AddMinutes(10)
        do {
            try {
                $status = Invoke-RestMethod -Uri "$base/albireo/v1/status" -TimeoutSec 10
                if ($status.state -ne "starting") { break }
            }
            catch {
                $status = $null
            }
            Start-Sleep -Seconds 5
        } while ((Get-Date) -lt $deadline)
        try {
            $status = Invoke-RestMethod -Uri "$base/albireo/v1/status" -TimeoutSec 10
            $active = ($status.active.PSObject.Properties | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ", "
            Add-Result "service status" "PASS" "$($service.Status), engine $($status.engine.variant), $active" `
                $watch.Elapsed.TotalSeconds
            $models = Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 10
            $names = @($models.data | ForEach-Object { $_.id })
            Add-Result "service models" "PASS" ($names -join ", ") 0
            # The engines of the service, each described within its time limit, with their devices.
            $engines = Invoke-RestMethod -Uri "$base/albireo/v1/engines" -TimeoutSec 30
            $stuck = @($engines.engines | Where-Object { $_.reason_code -in @("probe_timeout", "library_failed") } |
                ForEach-Object { $_.variant })
            $devices = @($engines.engines | Where-Object { $_.usable } | ForEach-Object {
                $engine = $_
                $found = @($engine.devices | Where-Object { $_.type -ne "cpu" } | ForEach-Object { $_.name })
                if ($found.Count -gt 0) { "$($engine.variant): $($found -join ' + ')" } })
            if ($stuck.Count -gt 0) { Add-Result "service engines" "FAIL" ("not described: " + ($stuck -join ", ")) 0 }
            else { Add-Result "service engines" "PASS" ("current $($engines.current); " + ($devices -join "; ")) 0 }
            $listed = Invoke-RestMethod -Uri "$base/albireo/v1/models?lang=en" -TimeoutSec 30
            $came = @($listed.models | Where-Object { $_.builtin } | ForEach-Object { $_.id }) +
                @($listed.embedding_models | Where-Object { $_.source -eq "builtin" } | ForEach-Object { $_.id })
            if ($came.Count -gt 0) { Add-Result "service builtin" "PASS" ($came -join ", ") 0 }
            else { Add-Result "service builtin" "SKIP" "installed without the models of the complete installer" 0 }
            if ($names -contains "albireo-general") {
                $watch.Restart()
                $body = @{ model = "albireo-general"; reasoning = $false; max_tokens = 40;
                    messages = @(@{ role = "user"; content = $question }) } | ConvertTo-Json -Depth 5
                $answer = Invoke-RestMethod -Method Post -Uri "$base/v1/chat/completions" -Body $body `
                    -ContentType "application/json" -TimeoutSec 300
                $text = "$($answer.choices[0].message.content)"
                $result = "FAIL"
                if ($text -match "Paris") { $result = "PASS" }
                Add-Result "service chat" $result $text $watch.Elapsed.TotalSeconds
            }
            else {
                Add-Result "service chat" "SKIP" "no general model active" 0
            }
            if ($names -contains "albireo-embedding") {
                $watch.Restart()
                $body = @{ model = "albireo-embedding"; input = @("The service runs on this computer.",
                    "Where does the service run?") } | ConvertTo-Json -Depth 5
                $vectors = Invoke-RestMethod -Method Post -Uri "$base/v1/embeddings" -Body $body `
                    -ContentType "application/json" -TimeoutSec 300
                $count = @($vectors.data).Count
                $width = @($vectors.data[0].embedding).Count
                $result = "FAIL"
                if ($count -eq 2 -and $width -gt 0) { $result = "PASS" }
                Add-Result "service embeddings" $result "$count vectors of $width dimensions" `
                    $watch.Elapsed.TotalSeconds
            }
            else {
                Add-Result "service embeddings" "SKIP" "no embedding model active" 0
            }
            # The voices of the speech synthesis, then a sentence read aloud by the program the service starts.
            $watch.Restart()
            $listed = Invoke-RestMethod -Uri "$base/albireo/v1/speech/voices" -TimeoutSec 10
            $ids = @($listed.voices | ForEach-Object { $_.id })
            if ($ids.Count -eq 0) {
                Add-Result "service speech" "FAIL" "no voice installed" 0
            }
            else {
                $body = [System.Text.Encoding]::UTF8.GetBytes((@{ input = $sentence; voice = "fr-FR" } |
                    ConvertTo-Json))
                $wav = Join-Path $logs "service-speech.wav"
                Invoke-WebRequest -Method Post -Uri "$base/v1/audio/speech" -Body $body -UseBasicParsing `
                    -ContentType "application/json; charset=utf-8" -OutFile $wav -TimeoutSec 60
                $bytes = [System.IO.File]::ReadAllBytes($wav)
                $result = "FAIL"
                if (Test-Wave $bytes) { $result = "PASS" }
                Add-Result "service speech" $result ("{0} voices, {1:N1} s of sound" -f $ids.Count,
                    (($bytes.Length - 44) / 32000)) $watch.Elapsed.TotalSeconds
            }
        }
        catch {
            Add-Result "service" "FAIL" "$($_.Exception.Message)" $watch.Elapsed.TotalSeconds
        }
    }
}

$results | Format-Table -AutoSize | Out-String -Width 160 | Write-Host
$failed = @($results | Where-Object { $_.Result -eq "FAIL" }).Count
Write-Host "failed steps: $failed"
exit $failed
