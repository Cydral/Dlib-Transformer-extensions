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
- [Computers without network](#computers-without-network)
- [Publishing a version](#publishing-a-version)
- [Security](#security)
- [Data folders](#data-folders)

## What it does

- **Chat** in the web application, with the discussions kept in the browser (IndexedDB), an
  incognito mode that keeps them in memory only, a search over the discussions, and the reasoning
  of the models that reason shown while they think, then folded once the answer starts.
- **Documents** attached to a message or to a project: Word, Excel, PowerPoint, OpenDocument,
  PDF, RTF, HTML, text, CSV, Markdown and source code. The browser extracts the text of every page
  of a document, of every sheet of a workbook in the order of its tabs, and of every slide of a
  presentation in the order of the show. Each file appears as a thumbnail with its format, which
  can be removed before sending, and the edition of a question keeps, removes or adds its files.
- **Projects**: discussions grouped with instructions and documents that form their shared
  context, read whole when they fit, searched otherwise by meaning and by words, the search and
  its index running in the browser (see [Documents and context](#documents-and-context)).
- **Three functions**, general, code and security, each served by a language model of the
  catalog, chosen in the Models window or by the service when a model arrives, and an embedding
  model for the search in documents, which the service installs in every case.
- **Agent mode**: the model calls tools that run in the page (calculator, JavaScript in a
  sandboxed frame, outline, reading and search of the documents), up to eight steps per answer.
- **Several conversations at once**: the requests of every client are computed together, one
  forward pass serving the next token of each, and the keys and values of a finished request stay
  cached so that the next turn of a conversation starts where the previous one ended.
- **An interface for other programs**: `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings` and `/v1/models` follow the OpenAI interface, streaming included, with its
  documentation as a Swagger page at `/docs/`.
- **Engines for each kind of processor**: portable, AVX2, Intel NPU and integrated GPU through
  OpenVINO, NVIDIA GPU through CUDA, behind a single C interface. The service measures them on the
  machine and runs the fastest, and the icon menu can force the NPU, the GPU or both while a
  processor engine runs.
- **Models from a signed catalog**, downloaded in parallel parts, checked against their hashes,
  or imported from a disk or a memory card.
- **Updates** announced by a signed feed, attached to the GitHub releases of the repository,
  downloaded in the background and installed when the service restarts.
- **An offline mode** for the computers without network: the service opens no connection, and
  the catalog, the models and the updates come from an offline bundle brought on a disk.
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

### Documents and context

The text of a document is extracted in the browser ([`web/js/documents.js`](web/js/documents.js)):
the Office and OpenDocument formats are ZIP archives of XML, read with the decompression of the
browser, and PDF goes through pdf.js. The text marks each page, slide and sheet with a line that
starts with `## `, which names the passages of the search and lets the tools of the agent read a
document part by part. Nothing leaves the browser but the text placed in the messages.

The search ([`web/js/rag.js`](web/js/rag.js)) cuts the documents into passages of 1,200
characters overlapping by 240, named after their document and their part. The embedding model of
the service computes the vector of each passage once, through `/v1/embeddings`, and the passages
are kept with their vectors in the database of the browser (IndexedDB), or in memory for an
incognito discussion. A question is compared with the passages by meaning (cosine of the vectors)
and by words (BM25); the two rankings are fused (reciprocal rank fusion), the passages chosen so
that they do not repeat each other (maximal marginal relevance), and the consecutive passages of
a document merged into one excerpt. The engine computes the vectors of every client in the same
passes and keeps these passes short while a conversation is answered, so that indexing documents
never delays an answer.

The prompt of a question is built for the cache of the engine, which keeps the keys and values of
each conversation and computes again only the tokens that differ from what it already read
([`web/js/app.js`](web/js/app.js)):

| Part of the prompt | Content | Share of the context |
|---|---|---|
| System message | date, instructions, documents of the project read whole | 30% at most, else the project is searched |
| Attached document | read whole | 25% at most, else indexed and searched |
| Excerpts of a question | the passages found for it, kept with the message once sent | 20% at most |
| Exchanges | the conversation | the rest; the oldest leave in one step, down to 60% of the room |

The beginning of the prompt therefore stays the same from one question to the next: each question
costs its own tokens only, the conversations of a project share the reading of its documents
(the engine copies the cached prefix from one sequence to another), and the cut of the oldest
exchanges happens once every several questions. The service serves several sessions at once,
each with its own context: the requests of every session are computed together, one forward pass
serving the next token of each.

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
| [`tools/`](tools)        | build and test scripts, catalog, feed, publication, offline bundle, images    |

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
`tests\test_offload.exe`, with the CPU plugin of OpenVINO for a machine without NPU or Intel
graphics, and the test then runs from its folder. The random numbers and the solvers of
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
| `engine_chat`    | drives an engine library through its C interface: chat, tools, reasoning, parallel requests, embeddings compared with a reference |
| `bench_runtime`  | tokens per second of the prompt and of the generation                               |

`test_offload model.gguf CPU` validates the OpenVINO engine on a machine without NPU, with the CPU
plugin of OpenVINO; `ALBIREO_ENGINE_DLIB=ON` builds the same test on the host code of dlib.

Two scripts run these programs and the service, then print one line per step with its result:

```
bash dlib/dlib/external/albireo/tools/test_linux.sh
powershell -ExecutionPolicy Bypass -File dlib\dlib\external\albireo\tools\test_windows.ps1
```

They take `~/models/Qwen3-0.6B-Q4_K_M.gguf` and `~/models/Qwen3-Embedding-0.6B-Q8_0.gguf` by
default and skip the steps whose model or build tree is missing. The Linux script runs a service
offline in a data folder of its own, imports the model and asks a question through the OpenAI
interface; the Windows script asks the installed service a question and the vectors of two texts,
and tests the OpenVINO and CUDA engines when their trees exist (`-Device GPU` or `-Device NPU`
for `test_offload`, CPU by default). Their exit code is the number of failed steps.

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

The installer started from an offline bundle (see [Computers without network](#computers-without-network))
leaves its catalog and its models in the import folder, which the service reads at its first start.
The option "This computer has no Internet access" of the wizard, or `/offline`, sets the service
offline from the start.

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
| `default_model` | true    | downloads the default language model of the catalog at the first start    |
| `offline`       | false   | no connection: catalog, models and updates come from imports              |
| `proxy`         | `""`    | `""` for the proxy of the system (`HTTPS_PROXY`, then WinHTTP), `none`, or `http://[user:password@]host:port` |
| `active`        |         | the model of each function: `general`, `code`, `security`, `embedding`     |
| `update_feed`, `catalog_url` | | addresses of the signed feed and catalog                       |

## Interfaces

The interface compatible with OpenAI takes no key: the service listens on the loopback interface
only and refuses the requests whose `Host` or `Origin` names another site. The `model` field takes
the identifier of a loaded model, a function (`general`, `code`, `security`, or `albireo-code` and
its siblings), or `auto`; the `model` of `/v1/embeddings` takes `albireo-embedding`.

| Route                                    | Use                                                       |
|------------------------------------------|-----------------------------------------------------------|
| `GET /v1/models`                         | loaded models and the aliases of the functions            |
| `POST /v1/chat/completions`              | chat, streaming, tools, `reasoning_content`               |
| `POST /v1/completions`                   | completion of a prompt                                    |
| `POST /v1/embeddings`                    | vectors of texts, float or base64, `dimensions` to shorten them |
| `GET /albireo/v1/status`                 | host, engine, statistics, models, downloads, updates      |
| `GET /albireo/v1/models?lang=fr`         | catalog and installed models                              |
| `GET /albireo/v1/engines`                | engine libraries, options and measures                    |
| `POST /albireo/v1/count`                 | tokens of a conversation for a model                      |
| `/docs/`                                 | Swagger page of the interface                             |

The administration routes (`/albireo/v1/settings`, `/albireo/v1/models/<id>/download`,
`/activate`, `/deactivate`, `/verify`, `/import`, `/models/upload`, `/updates/check`,
`/updates/apply`, `/restart`, `/engines/benchmark`, `/logs`) require the header
`X-Albireo-Admin` with the token of `admin.token`, which only the administrators, the service and
the interactive users of the machine can read.

## Models

The catalog ([`dist/albireo/catalog.json`](../../../../dist/albireo/catalog.json)) lists models of
the architectures the engine supports (llama, mistral, mistral3, granite, granitemoe, qwen2, qwen3,
qwen3moe, mixtral), with their functions, their descriptions in the five languages, their license,
and the address, size and SHA-256 of each file. [`tools/make_catalog.py`](tools/make_catalog.py)
writes it from the Hugging Face hub, the addresses pinned to a commit so that a file never changes
under its hash.

| Model                  | Functions       | Size    | License                                  |
|------------------------|-----------------|---------|------------------------------------------|
| Qwen3 4B (default)     | general         | 2.50 GB | Apache-2.0                               |
| Qwen3 Embedding 0.6B (default) | embedding | 0.64 GB | Apache-2.0                             |
| Qwen3 1.7B             | general         | 1.83 GB | Apache-2.0                               |
| Ministral 3 3B         | general         | 2.15 GB | Apache-2.0                               |
| Llama 3.2 3B           | general         | 2.02 GB | Llama 3.2 Community License, Built with Llama |
| Qwen3 8B               | general         | 5.03 GB | Apache-2.0                               |
| Qwen3 30B A3B          | general         | 18.56 GB | Apache-2.0                              |
| Qwen2.5 Coder 1.5B     | code            | 1.12 GB | Apache-2.0                               |
| Qwen2.5 Coder 7B       | code            | 4.68 GB | Apache-2.0                               |
| Granite 3.3 2B         | general, code   | 1.55 GB | Apache-2.0                               |
| Granite 3.3 8B         | general, code   | 4.94 GB | Apache-2.0                               |
| Foundation-Sec 1.1 8B  | security        | 4.92 GB | Llama 3.1 Community License              |

The service downloads the default language model at its first start, unless the installation said
otherwise, and the default embedding model in every case, since the search in documents relies on
it. A language model that arrives takes the functions its entry names that no model holds; an
embedding model serves the `embedding` function alone. The engine computes an embedding model as
it computes a language model, then takes the hidden state of the last token of each text,
normalized, as its vector (`pooling_type` 3 of the GGUF file); its keys and values are given back
as soon as each text is read.

A model can also be imported from any GGUF file: the Models window sends it to the service, or
the service imports the files placed in `<data>\import`.

## Computers without network

The `offline` setting, set by the installer (option "This computer has no Internet access",
`/offline`) or by the window of the computer, makes the service open no connection: the catalog, the updates and the downloads are
refused with the reason, and the menu of the icon greys "Check for updates". What the service
needs then comes from an offline bundle, prepared on a computer with network:

```
python3 dlib/dlib/external/albireo/tools/make_offline_bundle.py <folder> [--models qwen3-4b,qwen3-embedding-0.6b] [--from ~/models]
```

The bundle holds the signed catalog and the signed update feed of `dist/albireo`, the installer the
feed announces, a `models` folder with the models chosen (the two default models when `--models`
is absent) and a `README.txt`. Each file is checked against the hashes of the signed files when it
enters the bundle, and again by the service when it installs it. The bundle is used in one of
three ways:

- `albireo-setup-<version>.exe` started from the bundle installs Albireo with its models;
- the Import button of the Models window, given the folder of the bundle, installs the catalog,
  the models and the update of an installed computer;
- the content of the bundle copied into `C:\ProgramData\Albireo\import` is installed by the service
  within a minute; a file that cannot be installed is renamed with the suffix `.rejected`.

A catalog older than the one in place is never installed, and an update is installed at the next
start of the service.

On a computer with network, every request of the service runs on a thread of its own and is
bounded in time: 10 seconds to resolve the name of the server and to connect to it, including the
TLS handshake, then 20 seconds without data for the catalog and the feed. The requests of the users
never wait for the network. When the service stops or goes offline, the transfers in progress stop
at once and no new connection opens; a connection that httplib is opening ends within its bounds,
10 seconds for the connection and its TLS handshake and, behind a proxy, the answer of the proxy,
while the service tells the Windows service manager that its stop goes on.

## Publishing a version

The feed and the catalog live in [`dist/albireo`](../../../../dist/albireo) of the repository, each
with a detached signature (ECDSA P-256 over its exact bytes) that the programs check with the public
key built into [`host/crypto.cpp`](host/crypto.cpp). The private key stays with the publisher. The
programs travel in the installer, which carries the service, the tray, every engine with the
runtime libraries of its device, and the web application; the installer of each version is a file
of the GitHub release `albireo-v<version>`.

[`tools/publish_release.py`](tools/publish_release.py) publishes a version from its installer: it
checks that the installer carries the version of [`common/version.h`](common/version.h), creates
the release, attaches the installer, writes the feed with its address, size and SHA-256, signs it
and checks the signature, then prints the commands that push the feed:

```
python3 dlib/dlib/external/albireo/tools/publish_release.py build-albireo/bin/albireo-setup.exe --key ~/keys/albireo-feed-signing.key --notes notes.json
```

The token of GitHub comes from `GITHUB_TOKEN`, `GH_TOKEN` or `gh auth token`. A model published
outside the hub, a model converted for the engine for instance, is cut into parts of less than
2 GiB and attached to the release `albireo-models` by [`tools/publish_model.py`](tools/publish_model.py),
whose output is the entry of the catalog, with the architecture, the parameters and the context
read from the file:

```
python3 dlib/dlib/external/albireo/tools/publish_model.py model.gguf my-model parts --release albireo-models --gzip
```

[`dist/albireo/README.md`](../../../../dist/albireo/README.md) gives every step, the catalog
included. A feed announcing a higher version makes the services download the installer, check its
size and its SHA-256, and run it in update mode at their next start.

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
| `C:\ProgramData\Albireo\import`        | offline bundles and models, installed within a minute        |
| `C:\ProgramData\Albireo\certificates`  | local authority and certificate of the server                |
| `C:\ProgramData\Albireo\engine-cache`  | compiled graphs of OpenVINO, file of the extended memory     |
| `C:\ProgramData\Albireo\logs`          | logs of the service and of the installer                     |

`ALBIREO_HOME` replaces the data folder, on Windows as on Linux.
