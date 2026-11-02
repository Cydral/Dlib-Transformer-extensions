# Pseudalis

Pseudalis protects the personal data of documents on a Windows computer, on that computer alone.
A service anonymizes the files it receives through a drive, through commands of the Explorer, through a
web application and through a local interface, and keeps their originals encrypted, so that each one
comes back on demand from any copy of its anonymized file. It is built on
[`dlib/privacy`](../../privacy.h) and on the recognizers trained for the example
[`slm_document_anonymizer_ex`](../../../examples/slm_document_anonymizer_ex.cpp), which its engines carry.

Pseudalis is written by Aldric Pierrain, Copyright (C) 2026 Cydral Technology, and distributed under the
Boost Software License 1.0 ([`resources/license.txt`](resources/license.txt)). Its drive relies on
WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos,
[github.com/winfsp/winfsp](https://github.com/winfsp/winfsp), which its installer carries unchanged and
installs when it is missing.

## What it does

Each personal value of a document (names, addresses, telephone numbers, e-mail addresses, identifiers,
bank details, dates and the rest of the types of `dlib/privacy`) is replaced by a tag such as
`[[PERSON:Xoydoeh Nuggwdfsorn#EKR]]`, which carries the value encrypted with FF1 over AES-256 and a
check. The key of the computer alone restores the document, or any text derived from it. Word, Excel,
PowerPoint, PDF, text, CSV, Markdown, HTML, XML, JSON and YAML files keep their layout; a PDF keeps its
pages, its text drawn again in place.

### The drive

The installer gives the drive a letter chosen among the free ones, `A:` when it is free. The drive
works for every session of the computer as any drive does, from the Explorer and from the prompt
(`dir`, `copy`, `xcopy`, `move`, `del`, `ren`, `md`, `rd /s`, which replaces `deltree`, `type`, and
any editor of text files), with one difference: a file written there is anonymized when its last
handle closes, under its name, and only its anonymized version can ever be read back.

- A document keeps its name, anonymized; its original goes to the encrypted vault of the service.
- A picture becomes `<name>.txt`, its marker `[Image@<id> : <description>]`, or `[Image@<id>]` when no
  description is available, and the picture goes to the vault.
- A ZIP, TAR or TAR.GZ archive keeps its name; its documents are anonymized, its pictures become text
  files, and its other entries are left out, which its report says.
- A file of another kind is refused when it is created (access denied). A file whose content cannot be
  read, a damaged document for instance, keeps its previous version, or is removed when it had none, and
  the icon tells which one and why.
- While a file is written, then protected, the other programs cannot open its content (sharing
  violation), a few seconds for most documents.
- Deleting a file, a folder or the whole tree removes them from the drive and from its database at once.
  The original of a file deleted or replaced by a new version stays restorable from the copies of the
  file for 30 days (the setting `removed_days`), then leaves the vault with the pictures no other file
  shows.

### The commands of the Explorer

A right click on a file Pseudalis processes shows, under Show more options on Windows 11, a menu
Pseudalis with four commands, which `pseudalis.exe` carries out through the service:

| Command | Effect |
|---|---|
| Anonymize | the file is replaced by its anonymized version; a picture by its text file |
| Anonymize as a copy | `name (anonymized).ext` is written beside the file, ` (2)` added when the name is taken |
| Restore the original | the original takes the place of the anonymized file |
| Restore as a copy | `name (restored).ext` is written beside the file |

The suffixes follow the language of the installation. The original of an anonymized file is found by
the SHA-256 of its content, and by the NTFS object identifier of the file, which the command records
with its place; a file edited since its anonymization gets its values back from its tags. A file of the
drive is anonymized already: Restore asks where to save its original, outside the drive. The same
program runs from a prompt, `pseudalis protect [--copy] <file>...` and
`pseudalis restore [--copy] <file>...`, its results printed there (`start /wait pseudalis ...` waits
for them).

### The web application and the interface

The icon opens the web application of the service, `https://localhost:8444`: files dropped there are
anonymized or restored and downloaded back, texts are anonymized, restored or analyzed, and, opened
from the icon, it lists the originals kept (download, forget), exports and imports the key, and changes
the settings. The interface it uses is described with Swagger UI at `https://localhost:8444/docs/`
(Interface documentation in the menu of the icon), without account: the operations of administration
take the token of the data folder in the header `X-Pseudalis-Admin`, which the pages opened by the
icon receive in the fragment of their address and use by themselves.

| Route | Purpose |
|---|---|
| `GET /v1/health`, `GET /v1/info` | the state of the service, of its engine, of the drive and of Albireo |
| `POST /v1/anonymize?name=&record=&format=` | anonymizes the file of the body; the answer is the file, or JSON with `format=json` |
| `POST /v1/restore?name=&object_id=&format=` | restores the file of the body: its original, its picture or its tags |
| `GET /v1/lookup?sha256=&object_id=` | whether a file is an anonymized file of this computer |
| `POST /v1/records/{id}/location` | records where an anonymized file was written |
| `POST /v1/text/anonymize`, `/restore`, `/detect` | the same protection on a text, `{"text": ...}` |
| `GET /v1/drive`, `GET /v1/events?since=` | the drive, the files it protected or refused |
| `GET /v1/albireo` | whether Albireo describes the pictures |
| `GET /v1/records`, `GET /v1/records/{id}/original`, `DELETE /v1/records/{id}` | the originals kept (token) |
| `GET /v1/settings`, `PUT /v1/settings`, `POST /v1/restart` | the settings and the restart (token) |
| `GET /v1/updates`, `POST /v1/updates/check`, `POST /v1/updates/apply` | the updates |
| `POST /v1/key/export`, `POST /v1/key/import` | the key, protected by a passphrase (token) |
| `GET /v1/logs` | the last lines of the log (token) |

### The icon

The icon in the notification area follows the service: its tooltip gives the version, the drive and
the number of originals kept, and it notifies the files the drive anonymized or refused, the results of
the commands of the Explorer, the updates and a service that stops answering. Its menu opens Pseudalis
(the web application) and the drive, checks for updates and installs them, restarts the service,
opens the documentation of the interface and the About window.

### Pictures and Albireo

When Albireo, the local inference service, is installed on the same computer and one of its models reads
images, Pseudalis finds it through its interface (its data folder, its port, its local authority) and
asks it to describe each picture, in the language of the installation. The description is anonymized in
turn before it is written: a picture alone becomes `<name>.txt` holding `[Image@<id> : <description>]`,
and a picture inside a document gives the same text in its place (a run of Word, a text box of
PowerPoint at the place of the picture, the caption of a gray box in a PDF). The identifier is the start
of the SHA-256 of the picture, recorded in the database with the picture in the vault, so that
restoring the text file gives the picture back. The setting `albireo` turns the descriptions off.

### Languages

The installer, the icon, the commands of the Explorer and the web application speak English, French,
German, Italian and Spanish, English by default; the installer takes the language of Windows, and the
setting `language` changes it afterwards.

## Programs and data

| File | Role |
|---|---|
| `pseudalis-service.exe` | the service (LocalSystem): the engine, the drive, the vault, the database, the interface on `127.0.0.1` |
| `pseudalis-engine.dll`, `pseudalis-engine-avx2.dll` | the engine, for every x64 processor and for those with AVX2, chosen at the start of the service |
| `pseudalis-tray.exe` | the icon, started at the sign in of each user |
| `pseudalis.exe` | the commands of the Explorer |
| `pseudalis-setup.exe` | the installer, kept in the folder of the programs as the uninstaller |
| `web\` | the web application and the documentation of the interface |
| `winfsp\winfsp.msi`, `licenses\` | the installer of WinFsp and the licenses of the components |

The data lie in `%ProgramData%\Pseudalis`, or in the folder chosen at the installation, recorded in
`HKLM\Software\Pseudalis`: `settings.json`, `admin.token`, `keys\master.key`, `vault\`, `db\pseudalis.db`,
`certificates\`, `logs\` and `updates\`.

## Security

- The master key, 32 random bytes, is encrypted with DPAPI for the account of the service (LocalSystem)
  in `keys\master.key`, whose permissions admit SYSTEM and the administrators alone. HKDF-SHA-256
  derives from it the key of the pseudonymization and the two keys of the vault.
- Each original is kept in the vault encrypted with AES-256-CTR and authenticated with HMAC-SHA-256; the
  database holds the names, the sizes, the hashes and the places of the files, never their content.
- The key can be exported, protected by a passphrase of 12 characters or more (PBKDF2-SHA-256 with
  600 000 iterations, AES-256-CTR and HMAC-SHA-256), and imported into a new installation that keeps no
  original yet. Without its key, an anonymized document cannot be restored anywhere: the export is the
  backup of the installation.
- The interface listens on the loopback address only, over HTTPS with a certificate of the local
  authority "Pseudalis Local Authority", restricted by name constraints to the local names and
  addresses, whose key is never written and whose server key is encrypted with DPAPI. Requests whose
  `Host` is not local, and requests of another origin, are refused, so that no web page reads or changes
  anything through the browser.
- The model of Pseudalis is the workstation of one user. Every program of the users of the computer
  reaches the interface, and the token of administration lies in the data folder, readable by them: a
  user of the computer can restore any anonymized file. On a computer shared by several people, each one
  needs an installation of their own, on a computer of their own.
- The drive gives its files to the interactive users, SYSTEM and the administrators.

## Choices made from the specification

- A file system cannot tell a move out of the drive from a copy followed by a deletion: Windows copies
  the file, then deletes it. The drive therefore never gives back an original, which would otherwise
  reach any program that reads it; the originals come back through Restore, from the Explorer, the web
  application or the interface, from any copy of the anonymized file. For the same reason, the original
  of a file that leaves the drive stays in the vault for `removed_days` days, which the records of the
  web application show, then goes, or at once when it is forgotten there: gone with its file, it would
  leave the copy restorable from its tags alone, without its pictures and the parts the protection
  removed.
- A file of a kind Pseudalis does not process is refused on the drive at its creation, rather than
  kept unprotected; the temporary files of Office are among them, so that documents are edited outside
  the drive and copied onto it.
- A picture becomes a text file of the same name with `.txt` added, which the Explorer and the prompt
  show as text; its picture comes back through Restore.
- On Windows 11, the commands of the Explorer lie under Show more options: the first menu of Windows 11
  only takes the commands of packaged applications.
- The service listens on port 8444, beside Albireo on 8443, with the same stack: dlib, cpp-httplib and
  Mbed TLS.
- WinFsp stays installed when Pseudalis is uninstalled, since other programs may use it.

## Installing

`pseudalis-setup.exe` installs the programs, WinFsp when it is missing or older than the version it
carries (2.1), the service, the icon, the commands of the Explorer, the shortcuts and the entry of the
installed applications of Windows; started again over an installation, it updates it and keeps the
data, the key, the drive and the options. The wizard asks for the language, the license, the folders,
the letter of the drive and the options. Without window:

```
pseudalis-setup /silent [/dir <folder>] [/data-dir <folder>] [/drive <letter>|none] [/port <port>] [/no-context-menu]
pseudalis-setup /uninstall [/silent] [/keep-data]
pseudalis-setup /service start|stop|restart
```

The uninstallation removes the programs and, unless the key and the originals are kept for a later
installation (`/keep-data`, or the option of the wizard), the data: the documents anonymized with that
key can then no longer be restored.

## Building

### Windows

With Visual Studio 2022 or its Build Tools (C++ workload, CMake 3.21 or later), from the root of the
repository:

```
powershell -ExecutionPolicy Bypass -File dlib\dlib\external\pseudalis\tools\build_windows.ps1
```

The script downloads once the files of WinFsp 2.1 (its installer, the two headers the drive compiles
with, its license), each checked against its SHA-256, into `build-pseudalis\winfsp`, then builds every
program into `build-pseudalis\bin`; the installer, `build-pseudalis\bin\pseudalis-setup.exe`, carries
them all, with the runtime linked statically. The engines compile the models of the header
[`examples/slm_document_anonymizer_ex.h`](../../../examples/slm_document_anonymizer_ex.h)
(`PSEUDALIS_MODELS_HEADER`). `-Tests` also builds and runs the tests.

### Linux

The service builds on Linux with its engine inside and without drive, for the tests of the engine, of
the host and of the interface:

```
bash dlib/dlib/external/pseudalis/tools/test_linux.sh
```

### Cross compilation

```
cmake -S dlib/dlib/external/pseudalis -B build-pseudalis-win -DCMAKE_TOOLCHAIN_FILE=dlib/dlib/external/pseudalis/cmake/mingw-w64.cmake -DPSEUDALIS_WINFSP_INCLUDE=<inc of WinFsp> -DPSEUDALIS_WINFSP_MSI=<winfsp.msi>
cmake --build build-pseudalis-win
```

## Publishing a version

The update feed lies in [`dist/pseudalis`](../../../../dist/pseudalis) with its signature (ECDSA P-256),
which the programs check with the public key of `resources/feed-public-key.pem`, built into them. The
publisher makes the pair once, the private key outside the repository:

```
build-pseudalis-linux/bin/pseudalis-sign keygen ~/keys/pseudalis-feed-signing.key dlib/dlib/external/pseudalis/resources/feed-public-key.pem
```

The published files lie in the folder `dist` of the public link of pCloud, whose password the service
carries masked, read at the configuration from `keys/pseudalis-public-link.txt` of the home folder, else
from `keys/albireo-public-link.txt` (Albireo publishes its own files in the same link, under names of
their own). A version is published from its installer, then the local copy of the link is filled and
checked:

```
python3 dlib/dlib/external/pseudalis/tools/publish_release.py build-pseudalis/bin/pseudalis-setup.exe --key ~/keys/pseudalis-feed-signing.key --notes notes.json
python dlib\dlib\external\pseudalis\tools\fill_public_link.py dist <copy of the link>\dist --installer build-pseudalis\bin\pseudalis-setup.exe
python dlib\dlib\external\pseudalis\tools\fill_public_link.py check
```

`publish_release.py` checks that the installer carries the version of
[`common/version.h`](common/version.h), writes the feed with the name of the installer in the link,
`pseudalis-setup-<version>.exe`, its size and its SHA-256, signs it, checks the signature and prints the
commands that push it; `fill_public_link.py dist` copies the installer, then the signature and the feed
as `pseudalis-feed.json` and `pseudalis-feed.json.sig`; `check` lists what the link still lacks. The
services read the feed of the link first, then the one of GitHub, check its signature, download the
installer in the background, check its size and its SHA-256, and install it at the next start of the
service, or at once from the menu of the icon.

## Limitations

- The recognizers read French and English first; a document in another language keeps more values in
  clear. A protected document should be read again before it is shared.
- A scanned page without text layer is kept as a gray box: no OCR is performed.
- The drive takes files of 256 MB at most by default (`max_file_mb`).
- The protection is a pseudonymization in the sense of the GDPR: the key restores the data, which stay
  personal data for its holder.
