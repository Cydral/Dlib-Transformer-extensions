# Albireo

Albireo runs language models on the computer where it is installed. It is a Windows 11
program built on dlib and Dlib-Transformer-extensions: an inference engine for quantized GGUF
models, a Windows service that serves it on a local HTTPS address with an interface compatible
with the OpenAI one, a web application served by that service, an icon in the notification area
to manage it, and an installer. The conversations, the documents and the models stay on the
machine, and nothing asks for an account.

Albireo is published by Cydral Technology and was created by Aldric Pierrain. Its name is that
of a double star of the constellation Cygnus, one gold and one blue, which its icon draws.

## Contents

- [What it does](#what-it-does)
- [How it is built](#how-it-is-built)
- [Source tree](#source-tree)
- [Building on Windows](#building-on-windows)
- [Building on Linux](#building-on-linux)
- [Tests](#tests)
- [Installation](#installation)
- [Settings](#settings)
- [Interfaces](#interfaces)
- [Models](#models)
- [Publishing a version](#publishing-a-version)
- [Security](#security)
- [Data folders](#data-folders)

## What it does

- **Chat** in the web application, with the discussions kept in the browser (IndexedDB), an
  incognito mode that keeps them in memory only, a search over the discussions, and the reasoning
  of the models that reason shown while they think, then folded once the answer starts.
- **Projects**: discussions grouped with instructions and documents (text, Office, PDF) that the
  model receives as context, whole when they fit in a third of its context, by passages chosen
  with BM25 otherwise.
- **Three functions**, general, code and security, each served by a model of the catalog, chosen
  in the Models window or by the service when a model arrives.
- **Agent mode**: the model calls tools that run in the page (calculator, JavaScript in a
  sandboxed frame, search in the documents of the project), up to eight steps per answer.
- **Several conversations at once**: the requests of every client are computed together, one
  forward pass serving the next token of each, and the keys and values of a finished request stay
  cached so that the next turn of a conversation starts where the previous one ended.
- **An interface for other programs**: `/v1/chat/completions`, `/v1/completions` and
  `/v1/models` follow the OpenAI interface, streaming included, with its documentation as a
  Swagger page at `/docs/`.
- **Engines for each kind of processor**: portable, AVX2, Intel NPU and integrated GPU through
  OpenVINO, NVIDIA GPU through CUDA, behind a single C interface. The service measures them on the
  machine and runs the fastest, and the icon menu can force the NPU, the GPU or both while a
  processor engine runs.
- **Models from a signed catalog**, downloaded in parallel parts, checked against their hashes,
  or imported from a disk or a memory card for a machine without network.
- **Updates** announced by a signed feed, downloaded in the background and installed when the
  service restarts.
- **Five languages** in every window: English, French, German, Italian and Spanish, the language
  of Windows by default.

## How it is built

### Engines

An engine is a library (`engines\albireo-engine-<variant>.dll`) that exports the functions of
[`common/engine_api.h`](common/engine_api.h): describe, create, load and unload a model, submit a
request whose tokens come back as events, cancel it, and answer queries (statistics, tokens,
models). Every variant is built from the same sources:

| Variant    | Products with the weights                                   | Built with                        |
|------------|-------------------------------------------------------------|-----------------------------------|
| `cpu`      | quantized kernels, every x86-64 processor                   | the main build tree               |
| `cpu-avx2` | quantized kernels, AVX2, FMA and F16C                       | the main build tree               |
| `openvino` | NPU and integrated GPU of Intel processors, AVX2 kernels   | `ALBIREO_ENGINE_OPENVINO=ON`      |
| `cuda`     | NVIDIA graphics processors, AVX2 kernels                    | `ALBIREO_ENGINE_CUDA=ON`          |

The forward pass ([`engine/cpu_runtime.cpp`](engine/cpu_runtime.cpp)) keeps the weights in the
memory mapping of the GGUF file, in their quantized form, and multiplies them with kernels that
quantize the activations to 8 bits per block, so that a generated token reads each weight once.
The keys and values live in half precision in pages of 256 positions, one sequence per request.
The scheduler ([`engine/engine.cpp`](engine/engine.cpp)) builds each pass from the next token of
every request being generated and a slice of the prompts being read.

The engines of the devices add a `matmul_offload` ([`engine/offload.h`](engine/offload.h)) that
receives each product with the way to compute it on the processor
([`engine/tensor_backend.cpp`](engine/tensor_backend.cpp)). The product goes to the device through
the tensors of dlib, with a copy of the weight matrix in single precision kept while the copies of
all the models fit in a budget:

- with OpenVINO, the routing of the NPU module of dlib times the processor kernels against the NPU
  and the GPU, shape by shape, and keeps the faster;
- with CUDA, the engine makes these measures itself, on two matrices of each shape, and checks its
  choice at growing intervals; the extended device memory of dlib (V²RAM) lets the copies exceed
  the memory of the card, the blocks that do not fit moving to the memory of the host or to a file
  and coming back ahead of their use.

The products of a single token often stay on the processor, which serves them from the compact
weights; the products of the prompts go to the device.

### Service

`albireo-service.exe` runs as the Windows service `Albireo` (LocalSystem, delayed automatic start)
and, for development, in a console with `--console`. It:

- describes each engine library in a child process (`--probe-engine`), so that a library that
  cannot load leaves the service running, and measures them the same way (`--bench-engine`);
- starts the chosen engine and the active models, and serves `https://localhost:<port>/` with
  cpp-httplib and Mbed TLS, with a certificate issued by its own authority, which the installer adds
  to the trusted roots of the machine;
- keeps its settings in `settings.json`, the models, the downloads and the usage in an SQLite
  database, and its log in `logs\service.log`;
- restarts itself when a setting requires it, in a new process once the engine of a device has
  run, since that library stays loaded with the threads of its runtime.

### Tray, installer and web application

`albireo-tray.exe` places the icon in the notification area and opens the windows of the models,
of the computer (processor, engine, extended memory with its usage, models folder, port, language,
proxy) and of the product. `albireo-setup.exe` carries the programs as a ZIP payload, installs them
with a wizard or silently, registers the service and the entry of Apps & Features, and removes them.
The web application ([`web/`](web)) is plain ES modules with its libraries in
[`web/vendor`](web/vendor), served by the service with a content security policy that allows no
other origin.

## Source tree

| Folder                   | Content                                                                      |
|--------------------------|------------------------------------------------------------------------------|
| [`engine/`](engine)      | the inference engine: runtime, kernels, scheduler, chat formats, devices      |
| [`host/`](host)          | the service: server, settings, store, models, updates, certificates           |
| [`common/`](common)      | the C interface of the engines, JSON, logs, platform and processor details    |
| [`windows/`](windows)    | the tray, the installer, their windows and their texts in five languages      |
| [`web/`](web)            | the web application, the Swagger page and the OpenAPI description            |
| [`resources/`](resources)| icons, images, license, version resources                                     |
| [`cmake/`](cmake)        | MinGW-w64 toolchain, payload of the installer, runtime libraries of devices   |
| [`tests/`](tests)        | kernel, runtime, tokenizer and device tests, engine driver, benchmark          |
| [`tools/`](tools)        | build script, catalog, feed, signatures, model parts, images, web libraries   |

## Building on Windows

Requirements: Visual Studio 2022 with the C++ workload and CMake 3.21 or later. The OpenVINO
engine needs the runtime of OpenVINO (`python -m pip install openvino`, or the archive with
`INTEL_OPENVINO_DIR` set); the CUDA engine needs the CUDA toolkit 12.8 or later and cuDNN 9.

dlib is configured for one device per build tree, so the engines of the devices are built in their
own trees and join the installer through `ALBIREO_EXTRA_ENGINE_DIRS`. The script
[`tools/build_windows.ps1`](tools/build_windows.ps1) runs the whole sequence from the root of the
repository:

```
powershell -ExecutionPolicy Bypass -File dlib\dlib\external\albireo\tools\build_windows.ps1 -OpenVINO -Cuda
```

`-Tests` adds the test programs, `-Jobs <n>` bounds the compilations run at once on a machine with
little memory, `-CudaVersion 12.9` picks one of several installed CUDA toolkits, `-Cudnn <folder>`
one cuDNN installation (`C:\Program Files\NVIDIA\CUDNN\v9.11` for instance), and
`-CudaArchitectures` replaces the list of architectures of the CUDA engine. The script amounts to:

```
cmake -S dlib/dlib/external/albireo -B build-albireo-ov -G "Visual Studio 17 2022" -A x64 -DALBIREO_PROGRAMS=OFF -DALBIREO_ENGINE_OPENVINO=ON
cmake --build build-albireo-ov --config Release
cmake -S dlib/dlib/external/albireo -B build-albireo-cuda -G "Visual Studio 17 2022" -A x64 -DALBIREO_PROGRAMS=OFF -DALBIREO_ENGINE_CUDA=ON
cmake --build build-albireo-cuda --config Release
cmake -S dlib/dlib/external/albireo -B build-albireo -G "Visual Studio 17 2022" -A x64 "-DALBIREO_EXTRA_ENGINE_DIRS=build-albireo-ov/bin/engines;build-albireo-cuda/bin/engines"
cmake --build build-albireo --config Release
```

The installer is `build-albireo\bin\albireo-setup.exe`. The device trees copy beside their engine
the libraries it loads (OpenVINO with the plugins of the NPU and of the GPU, or the runtime of CUDA,
cuBLAS and the parts of cuDNN the engine uses, with the Visual C++ runtime) and their licenses,
which the installer places in its `licenses` folder; the same libraries go beside
`tests\test_offload.exe`, which then runs from its folder. The random numbers and the solvers of
CUDA, which dlib links and the engine never calls, are loaded on first use only and are left out of
the installation.

The CUDA engine carries the machine code of the cards of desktops and laptops from the GeForce GTX
900 series on, and the intermediate code of the newest architecture, which the driver compiles for
the cards that follow:

| Toolkit         | Architectures (`CMAKE_CUDA_ARCHITECTURES`)                             | Cards                         |
|-----------------|------------------------------------------------------------------------|-------------------------------|
| CUDA 12.8, 12.9 | `50-real 61-real 70-real 75-real 86-real 89-real 120-real 120-virtual` | GTX 900, 10, 16, RTX 20 to 50 |
| CUDA 13         | `75-real 86-real 89-real 120-real 120-virtual`                         | GTX 16, RTX 20 to 50          |

cuDNN 9.12 and later run on Turing and later cards only (GTX 16 and RTX series). A build for the
GTX 900 and 10 series therefore takes CUDA 12.8 or 12.9 with cuDNN 9.11 or earlier, the version
copied beside the engine. The configuration prints the toolkit, the cuDNN version and the oldest
compute capability of the result, which the engine also checks at its start: a card below it is
reported as unusable, with its compute capability, and the other engines serve the models.

| CMake option              | Default | Effect                                                            |
|---------------------------|---------|-------------------------------------------------------------------|
| `ALBIREO_PROGRAMS`        | ON      | service, tray, installer, processor engines                       |
| `ALBIREO_ENGINE_OPENVINO` | OFF     | OpenVINO engine (dlib with `DLIB_USE_NPU`)                        |
| `ALBIREO_ENGINE_CUDA`     | OFF     | CUDA engine (dlib with `DLIB_USE_CUDA`)                           |
| `ALBIREO_ENGINE_DLIB`     | OFF     | engine whose products go through the host code of dlib (tests)     |
| `ALBIREO_TESTS`           | ON      | test programs, in `<build>/tests`                                 |
| `ALBIREO_EXTRA_ENGINE_DIRS` |       | folders of engines built in other trees, added to the installer   |
| `ALBIREO_COPY_RUNTIME`    | ON on Windows | copy of the runtime libraries beside the device engines      |
| `ALBIREO_RUNTIME_LICENSES`|         | more license files for the `licenses` folder                      |

The programs link their C and C++ runtime statically and depend on system libraries only; the
OpenVINO engine uses the shared runtime of OpenVINO's own libraries, installed beside it.

## Building on Linux

The service, the processor engines and the tests build on Linux, which serves for development
(`--console` runs the service in the terminal, `ALBIREO_HOME` names its data folder):

```
cmake -S dlib/dlib/external/albireo -B build-albireo
cmake --build build-albireo -j
ALBIREO_HOME=$HOME/albireo-data build-albireo/bin/albireo-service --console
```

The Windows programs can also be cross-compiled from Linux with MinGW-w64 (POSIX threads), for
the processor engines:

```
cmake -S dlib/dlib/external/albireo -B build-albireo-win -DCMAKE_TOOLCHAIN_FILE=dlib/dlib/external/albireo/cmake/mingw-w64.cmake
cmake --build build-albireo-win -j
```

## Tests

| Program          | Checks                                                                               |
|------------------|--------------------------------------------------------------------------------------|
| `test_qkernels`  | every quantized kernel against its portable version and a reference in floats        |
| `test_runtime`   | the logits of a model against `tests/reference_forward.py`, and the same results whatever the cut of the batches |
| `test_tokenizer` | the tokenizer of a GGUF file against reference token sequences                     |
| `test_offload`   | the products of a device against the processor kernels, the small budget path and the measured routing |
| `engine_chat`    | drives an engine library through its C interface: chat, tools, reasoning, parallel requests |
| `bench_runtime`  | tokens per second of the prompt and of the generation                               |

`test_offload model.gguf CPU` validates the OpenVINO engine on a machine without NPU, with the CPU
plugin of OpenVINO; `ALBIREO_ENGINE_DLIB=ON` builds the same test on the host code of dlib.

## Installation

The wizard asks for the language, the license, the folders of the programs and of the models, the
options (start with Windows, desktop shortcut, trust of the local certificate, download of the
default model, port), then installs, registers the service and starts the tray. Over an installed
version, its fields start from the options in place, and the programs keep their folder. The
command line covers the other cases; the options it does not give keep the values of the
installation in place:

```
albireo-setup /silent [/dir <folder>] [/models <folder>] [/port <port>]
albireo-setup /uninstall [/silent] [/purge]
albireo-setup /models-dir <folder>
albireo-setup /service start|stop|restart
albireo-setup /lang en|fr|de|it|es
```

A `models` folder beside the installer, on a memory card for instance, is imported at the first
start of the service, which installs a machine without network with its models.

## Settings

`settings.json`, in the data folder, holds the settings; the tray changes them through the
administration interface, and the service restarts when a change requires it.

| Key             | Default | Meaning                                                                   |
|-----------------|---------|---------------------------------------------------------------------------|
| `port`          | 8443    | port of `https://localhost`                                               |
| `models_dir`    | `""`    | folder of the models, `<data>\models` when empty                          |
| `language`      | `auto`  | `auto`, `en`, `fr`, `de`, `it`, `es`                                      |
| `engine`        | `auto`  | `auto` (fastest measured, else the first usable of cuda, cpu-avx2, cpu), `best` (measured first), or a variant |
| `accelerator`   | `auto`  | `none`, `npu`, `gpu`, `both`: the device of the engines that have several |
| `v2ram`         | true    | extended device memory of the CUDA engine                                  |
| `v2ram_max_mb`  | 0       | size of its file in megabytes, 0 for automatic                            |
| `threads`       | 0       | threads of the engine, 0 for the physical cores                           |
| `context`       | 0       | context of the models, 0 for the trained one up to 32768                  |
| `max_sequences` | 4       | requests computed together per model                                      |
| `kv_mb`         | 0       | memory of the keys and values, 0 for automatic                            |
| `check_updates` | true    | checks the feed at each start and every six hours                          |
| `default_model` | true    | downloads the default model of the catalog at the first start             |
| `proxy`         | `""`    | `""` for the proxy of the system (`HTTPS_PROXY`, then WinHTTP), `none`, or `http://[user:password@]host:port` |
| `active`        |         | the model of each function: `general`, `code`, `security`                  |
| `update_feed`, `catalog_url` | | addresses of the signed feed and catalog                       |

## Interfaces

The interface compatible with OpenAI takes no key: the service listens on the loopback interface
only and refuses the requests whose `Host` or `Origin` names another site. The `model` field takes
the identifier of a loaded model, a function (`general`, `code`, `security`, or `albireo-code` and
its siblings), or `auto`.

| Route                                    | Use                                                       |
|------------------------------------------|-----------------------------------------------------------|
| `GET /v1/models`                         | loaded models and the aliases of the functions            |
| `POST /v1/chat/completions`              | chat, streaming, tools, `reasoning_content`               |
| `POST /v1/completions`                   | completion of a prompt                                    |
| `GET /albireo/v1/status`                 | host, engine, statistics, models, downloads, updates      |
| `GET /albireo/v1/models?lang=fr`         | catalog and installed models                              |
| `GET /albireo/v1/engines`                | engine libraries, options and measures                    |
| `POST /albireo/v1/count`                 | tokens of a conversation for a model                      |
| `/docs/`                                 | Swagger page of the interface                             |

The administration routes (`/albireo/v1/settings`, `/albireo/v1/models/<id>/download`,
`/activate`, `/deactivate`, `/verify`, `/models/import`, `/models/upload`, `/updates/check`,
`/updates/apply`, `/restart`, `/engines/benchmark`, `/logs`) require the header
`X-Albireo-Admin` with the token of `admin.token`, which only the administrators, the service and
the interactive users of the machine can read.

## Models

The catalog ([`dist/albireo/catalog.json`](../../../../dist/albireo/catalog.json)) lists models of
the architectures the engine supports (llama, mistral, mistral3, granite, granitemoe, qwen2, qwen3,
qwen3moe, mixtral), with their functions, their descriptions in the five languages, their license,
and the address, size and SHA-256 of each file. [`tools/make_catalog.py`](tools/make_catalog.py)
writes it from the Hugging Face hub, the addresses pinned to a commit so that a file never changes
under its hash. The service downloads the default model at its first start, unless the installation
said otherwise, and a model that arrives takes the functions its entry names that no model holds.

A model can also be imported from any GGUF file: the Models window sends it to the service, or
the service imports the files placed in `<data>\import`.

## Publishing a version

The feed and the catalog live in [`dist/albireo`](../../../../dist/albireo) of the repository, each
with a detached signature (ECDSA P-256 over its exact bytes) that the programs check with the public
key built into [`host/crypto.cpp`](host/crypto.cpp). The private key stays with the publisher.

```
albireo-sign keygen albireo-feed-signing.key albireo-feed-signing.pub
python3 dlib/dlib/external/albireo/tools/make_catalog.py dist/albireo/catalog.json
albireo-sign sign albireo-feed-signing.key dist/albireo/catalog.json
python3 dlib/dlib/external/albireo/tools/make_feed.py dist/albireo/feed.json 1.0.1 albireo-setup.exe https://github.com/Cydral/Dlib-Transformer-extensions/releases/download/albireo-v1.0.1/albireo-setup-1.0.1.exe notes.json
albireo-sign sign albireo-feed-signing.key dist/albireo/feed.json
```

`keygen` is needed only to replace the key pair; its public key then goes into `host/crypto.cpp`,
and the programs built afterwards accept only what the new private key signs. The installer of a
version is attached to a GitHub release, whose assets accept files up to 2 GB; a model published
outside the hub is cut into parts with [`tools/publish_model.py`](tools/publish_model.py), whose
output is the `files` of its catalog entry. [`dist/albireo/README.md`](../../../../dist/albireo/README.md)
gives the steps of a publication.

The version of the programs is in [`common/version.h`](common/version.h); a feed announcing a
higher version makes the services download the installer, check its size and its SHA-256, and run
it in update mode at their next start.

## Security

- The service listens on `127.0.0.1` only, refuses the requests whose `Host` is not local
  (421) and those whose `Origin` is another site (403), takes JSON bodies only, of 64 MB at most,
  and answers with `nosniff`, `no-referrer`, a content security policy and frame protections.
- The administration routes compare the token in constant time; the token and the private key of
  the certificate are files with restricted rights, and the data folder grants writing to the
  administrators and the service only.
- The feed, the catalog and every downloaded file are checked against signatures and hashes;
  imports read only GGUF files from local folders.
- The JavaScript the agent mode runs executes in a sandboxed frame of opaque origin, in a worker
  stopped after a time limit.
- The engine libraries are loaded from the `engines` folder of the installation, with their
  dependencies searched in that folder first.

## Data folders

| Path (Windows)                         | Content                                                      |
|----------------------------------------|--------------------------------------------------------------|
| `C:\Program Files\Albireo`             | programs, engines, web application, licenses                 |
| `C:\ProgramData\Albireo\settings.json` | settings                                                     |
| `C:\ProgramData\Albireo\albireo.db`    | installed models, downloads, usage, catalog                   |
| `C:\ProgramData\Albireo\models`        | models (or the folder chosen at installation)                |
| `C:\ProgramData\Albireo\import`        | files imported at the next start                             |
| `C:\ProgramData\Albireo\certificates`  | local authority and certificate of the server                |
| `C:\ProgramData\Albireo\engine-cache`  | compiled graphs of OpenVINO, file of the extended memory     |
| `C:\ProgramData\Albireo\logs`          | logs of the service and of the installer                     |

`ALBIREO_HOME` replaces the data folder, on Windows as on Linux.
