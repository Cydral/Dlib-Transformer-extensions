# Albireo distribution

From version 1.0.25, every installed copy of Albireo reads four signed files on the website of
Cydral Technology, in the folder of the versions,
`https://cydral-technology.eu/files/albireo/dist/`, where the chain of the website publishes them
beside the installers:

| File                | Read by                                   | Content                                             |
|---------------------|-------------------------------------------|-----------------------------------------------------|
| `feed.json`         | the service, at each start and every 6 h  | the latest version and its installer, named beside it |
| `feed.json.sig`     | the service                               | signature of `feed.json`                            |
| `catalog.json`      | the service and the Models window         | the models offered, their files and their hashes    |
| `catalog.json.sig`  | the service                               | signature of `catalog.json`                         |

This folder keeps the catalog, which `tools/make_catalog.py` writes from the list of the models,
and its signature, which `albireo-sign` makes. The feed is written and signed at each publication,
in the folder of the chain of the website, and this folder keeps none: the tests of the updates
read the feed of version 1.0.24, the last one written here, in
`dlib/dlib/external/albireo/tests/data`.

The services read the feed, the catalog and the installer of an update in the folder of the
versions alone, which asks for no account and no password: a file the website does not serve makes
the operation fail with a message that names its address. They check each signature (ECDSA P-256
over the exact bytes of the file) with the public key built into the programs, and ignore a file
whose signature fails. A change of one byte of a JSON file therefore needs a new signature, and
`.gitattributes` keeps the files of this folder byte for byte on every checkout: Git for Windows
would otherwise give them CRLF ends of line (`core.autocrlf`), which their signatures do not cover.

The installers lie beside these files on the website, as `albireo-setup-online-<version>.exe`, the
installer of the updates, and `albireo-setup-complete-<version>.exe`, with the models that come
with Albireo, which lie beside it in `albireo-setup-complete-<version>.models` when both would
reach the 4 GiB Windows runs. The files of the models are published by their authors or their
converters on the Hugging Face hub, at the addresses the catalog gives, the only place the services
download them from; the website holds no copy of them.

## The chain of the website

The chain of the website, a folder of its own outside the repository (`~/work/cydral-website` under
WSL), is the only one that carries files to the hosting. It publishes the files of its folder
`releases/files` under `files/` of the website, with one layout for every product: the folder of
the publication of Albireo is `~/work/cydral-website/releases/files/albireo/dist`, which
`tools/site.sh setup` of the website makes with the others. `publish_release.py` fills that folder
and reads no setting of the website.

The folders of the user differ from one computer to the other and the rest of the paths is the
same: the tools below read the folder of the user from the system, of Linux for the folder of the
website and the private key (`~/work/cydral-website`, `~/keys/albireo-feed-signing.key`), of
Windows for the installers of the build
(`%USERPROFILE%\work\Dlib-Transformer-extensions\build-albireo\bin`). The same commands therefore
publish from each computer.

## Publishing a version

1. In `dlib/dlib/external/albireo`, set the version in `common/version.h` and in `project()` of
   `CMakeLists.txt`, run `tools/make_openapi.py`, which writes it in the description of the
   interface, the configuration comparing these places, and write the release notes of the version
   by language in `notes/<version>.json`,
   `{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}`; commit and push the sources.
2. Build the installers on Windows, in `%USERPROFILE%\work\Dlib-Transformer-extensions` (see the
   README of Albireo, section Building on Windows): `build-albireo\bin\albireo-setup-online.exe`,
   which the feed announces, and `albireo-setup.exe`, complete with the models that come with
   Albireo.
3. Under WSL, in a copy of the repository up to date, build `albireo-sign`, then publish:

```
cmake -S dlib/dlib/external/albireo -B build-albireo
cmake --build build-albireo --target albireo-sign
python3 dlib/dlib/external/albireo/tools/publish_release.py --dry-run
python3 dlib/dlib/external/albireo/tools/publish_release.py
```

`publish_release.py` checks that both installers carry the version of `version.h`, that the private
key signs what the key built into the programs verifies, that the catalog of this folder bears its
signature, that neither the folder of the publication nor the website holds the feed of a later
version or a catalog published after the one of this folder, and that the disk of the folder has
room for the files. A website that cannot be read from the computer, or that answers with something
else than a feed and a catalog, as a portal or a filter of a network does, stops the publication:
the version it serves is then unknown.

The program then fills the folder of the publication, where a file already there as it is stays:
the installers under their published names, with `albireo-setup-complete-<version>.models` when the
models of the complete installer lie beside it, then `catalog.json` and `catalog.json.sig` of this
folder, each first copied under a name that starts with a dot, which no feed names and the chain
leaves aside, and compared with its source; and `feed.json`, with the name of the installer of the
updates, its size, its SHA-256 and the release notes, with `feed.json.sig`, its signature, checked.
The files take their names once all are ready, the installers first and the feed last, and the
installers of the other versions leave the folder. A failure before the first name is taken leaves
the folder as it was; a failure after it is told with what the folder then holds, and the same
command completes the version.

The program last runs `tools/site.sh status` and `tools/site.sh publish`, from the folder of the
website and on the terminal of the publisher, where the chain asks for the password of the hosting
when it needs it, and reads the website until it serves the four signed files of the folder of the
publication byte for byte and the installers at their sizes, with the SHA-256 the list of the files
of the website gives for each of them. It opens no session with the hosting, and it neither reads,
keeps nor passes on its password.

The chain of the website uploads the installers first, then the feed, so that the website never
announces an installer it does not hold yet.

| Option         | Effect                                                                              |
|----------------|-------------------------------------------------------------------------------------|
| `--dry-run`    | makes the checks, prints what would be done, writes nothing and publishes nothing    |
| `--deposit`    | writes the files of the folder of the publication; the publisher runs `tools/site.sh publish` from the folder of the website, with the files of another product for instance, then `publish_release.py --check`; the only way on a computer that cannot read the website, which stops the publication otherwise |
| `--check`      | reads what the website serves, for the folder of the publication as it is, and checks the signatures of its feed and of its catalog when `albireo-sign` is found |
| `--catalog`    | publishes the catalog of this folder between two versions (see below)                |
| `--notes`      | another file of release notes than `notes/<version>.json` of the sources; without either, the notes the feed of the folder of the publication already gives the version are kept, and a version without notes reads "Albireo <version>" |
| `--complete`   | another complete installer than `albireo-setup.exe` beside the installer of the updates |
| `--wait`       | the seconds given to the website to serve the files, 300 by default                  |
| `--key`, `--site`, `--sign` | another private key, folder of the website, `albireo-sign`              |
| `--any-folder` | takes installers named on the command line outside the copy of the repository of step 2 |

`make_feed.py` writes the feed alone, with the name of the installer beside it on the website.

On the other computer of the publisher, `git pull` brings the sources, and `tools/site.sh take
online`, from the folder of the website, the files the hosting serves, the feed and the installers
among them: that computer then publishes the website in its turn. `tools/site.sh deploy --prune`
removes the installers of the earlier versions from the hosting.

The services see the new feed within six hours, or at once with "Check for updates" in the menu
of the icon, download the installer in the background, check its size and its SHA-256, and install
it when the service restarts, or at once with "Install version <version> and restart". The update
keeps the models that came with an earlier complete installer. The versions up to 1.0.24 read
their feed in a storage space that no longer receives it: they are updated once by hand, with an
installer of the website. When that space does not answer, they read the catalog of this folder on
GitHub, where it therefore stays.

## Updating the catalog

The list of models is in `dlib/dlib/external/albireo/tools/make_catalog.py`: identifiers,
functions (general, code, security, embedding), publishers, inputs (text, image), descriptions in
the five languages, licenses, and the file of each model on the Hugging Face hub, with the projector
of a model that reads images. The script asks the hub for the commit, the size and the hash of each
file and writes the catalog, which is then signed, committed and published:

```
python3 dlib/dlib/external/albireo/tools/make_catalog.py dist/albireo/catalog.json
build-albireo/bin/albireo-sign sign ~/keys/albireo-feed-signing.key dist/albireo/catalog.json
build-albireo/bin/albireo-sign verify dist/albireo/catalog.json
git add dist/albireo/catalog.json dist/albireo/catalog.json.sig
git commit -m "Revise the catalog of Albireo"
git push
python3 dlib/dlib/external/albireo/tools/publish_release.py --catalog
```

`publish_release.py --catalog` publishes the catalog between two versions: it checks both
signatures, puts the catalog and its signature in the folder of the publication, beside the feed of
the version, which stays as it is, runs the chain of the website as for a version, and reads the
website until it serves the new catalog. The feed of the folder must be the one the website serves,
byte for byte, beside the installer it names: on a computer that did not publish the version,
`tools/site.sh take online` brings them first. The program tells a catalog that Git does not hold
yet. The publication of a version carries the catalog of this folder with it, and needs no such
step.

The catalog names one default language model, which the service downloads at its first start,
and one default embedding model, which it downloads in every case for the search in documents.
Its `updated` member is the time of publication: a service never replaces its catalog with an
older one, and `publish_release.py` refuses to publish a catalog older than the one the folder of
the publication or the website holds.

The hub is the only source of the files of the models. A file, a revision or a repository that
leaves it can no longer be downloaded: the service ends the download with the code
`download_unavailable`, once a second answer of the hub confirms the first, names the model, its
address and the status of the answer, and the model stays out of the functions until a catalog
names another address for it; the copies already installed keep working. `make_catalog.py --check`
asks the hub for each address of the catalog of this folder, with its size and its SHA-256, without
downloading the files, and lists the models to revise, apart from those it could not check; it ends
with 1 when a model is to revise, else with 2 when one could not be checked:

```
python3 dlib/dlib/external/albireo/tools/make_catalog.py --check
```

A model to revise takes another repository or another file in the list of `make_catalog.py`, or
leaves the list; the catalog is then written, signed and published again as above.

## Offline bundle

A computer without network receives the catalog, the models and the updates on a disk:

```
python3 dlib/dlib/external/albireo/tools/make_offline_bundle.py <folder> [--models <id,id>] [--from ~/models]
```

The bundle holds the signed files the website serves, or those of the folder `--dist` names, the
folder of the publication for instance; the installer the feed announces, taken from `--installer`,
else from the folder of `--dist`, else downloaded from the folder of the versions on the website;
and the models chosen, the two default models by default, each taken from the folders of `--from`
or else downloaded from its address of the catalog. Its `README.txt` tells how to install it. The
program checks the signatures with `albireo-sign` (`--sign`, else `build-albireo/bin`, else the
`PATH`) and stops when it does not find it, unless `--no-verify` leaves that check to the service.

## The signing key

`albireo-sign` is built with the service (`build-albireo/bin`). The private key signs the feed
and the catalog; it stays with the publisher, out of the repository, by default in
`~/keys/albireo-feed-signing.key` on each computer that publishes. A new key pair is made with
`albireo-sign keygen <private key> <public key>`; its public key replaces `feed_public_key_pem` in
`dlib/dlib/external/albireo/host/crypto.cpp`, and only the programs built afterwards accept what the
new key signs.
