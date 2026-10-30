# Albireo distribution

This folder is read by every installed copy of Albireo:

| File                | Read by                                   | Content                                             |
|---------------------|-------------------------------------------|-----------------------------------------------------|
| `feed.json`         | the service, at each start and every 6 h  | the latest version and the address of its installer |
| `feed.json.sig`     | the service                               | signature of `feed.json`                            |
| `catalog.json`      | the service and the Models window         | the models offered, their files and their hashes    |
| `catalog.json.sig`  | the service                               | signature of `catalog.json`                         |

The services fetch these files from
`https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/albireo/`, check
each signature (ECDSA P-256 over the exact bytes of the file) with the public key built into the
programs, and ignore a file whose signature fails. A change of one byte of a JSON file therefore
needs a new signature.

The installers are attached to the GitHub releases of the repository, named `albireo-v<version>`,
and the models published outside the Hugging Face hub to the release `albireo-models`; a release
accepts files of less than 2 GiB, and the repository itself keeps only these small files.

## Publishing a version

1. Set the version in `dlib/dlib/external/albireo/common/version.h`, commit and push the sources.
2. Build the installers on Windows (see the README of Albireo, section Building on Windows):
   `albireo-setup-online.exe`, which the feed announces, and `albireo-setup.exe`, complete with
   the models that come with Albireo.
3. Write the release notes as `notes.json`, `{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}`;
   without this file, the notes the feed already holds for the same version are kept.
4. Publish the installers and sign the feed, then push the feed:

```
python3 dlib/dlib/external/albireo/tools/publish_release.py build-albireo/bin/albireo-setup-online.exe --complete build-albireo/bin/albireo-setup.exe --key <private key> --notes notes.json
git add dist/albireo/feed.json dist/albireo/feed.json.sig
git commit -m "Albireo <version>"
git push
```

`publish_release.py` checks that the installer carries the version of `version.h`, creates the
release `albireo-v<version>` when it does not exist, attaches the installer as
`albireo-setup-<version>.exe` (replacing a file of that name whose content differs), and the
complete installer as `albireo-setup-complete-<version>.exe` when it fits 2 GiB, reads the whole
installer back from its public address and stops when the address serves another file, then writes
the new `feed.json` beside the published one, signs it with `albireo-sign`, checks the signature and
puts the feed and its signature in place together. Its token of GitHub comes from `GITHUB_TOKEN`, `GH_TOKEN`
or `gh auth token`, with the right to write the contents of the repository; `--dry-run` checks
everything without publishing. `make_feed.py` writes the feed alone, for an installer published
by other means.

The services see the new feed within six hours, or at once with "Check for updates" in the menu
of the icon, download the installer in the background, check its size and its SHA-256, and install
it when the service restarts, or at once with "Install version <version> and restart". The update
keeps the models that came with an earlier complete installer.

## Updating the catalog

The list of models is in `dlib/dlib/external/albireo/tools/make_catalog.py`: identifiers,
functions (general, code, security, embedding), descriptions in the five languages, licenses, and
the file of each model on the Hugging Face hub. The script asks the hub for the commit, the size and
the hash of each file and writes the catalog:

```
python3 dlib/dlib/external/albireo/tools/make_catalog.py dist/albireo/catalog.json
albireo-sign sign <private key> dist/albireo/catalog.json
albireo-sign verify dist/albireo/catalog.json
```

The catalog names one default language model, which the service downloads at its first start,
and one default embedding model, which it downloads in every case for the search in documents.
Its `updated` member is the time of publication: a service never replaces its catalog with an
older one.

A model published outside the hub, a model converted for the engine for instance, is cut into
parts and attached to the release `albireo-models` by `tools/publish_model.py`, which prints its
entry: architecture, parameters, context, `size`, `sha256` and `files`. The entry is added to the
list of `make_catalog.py`, completed with the name, the functions and the descriptions, then the
catalog is written and signed again:

```
python3 dlib/dlib/external/albireo/tools/publish_model.py model.gguf my-model parts --release albireo-models --gzip
```

## Offline bundle

A computer without network receives the catalog, the models and the updates on a disk:

```
python3 dlib/dlib/external/albireo/tools/make_offline_bundle.py <folder> [--models <id,id>] [--from ~/models]
```

The bundle holds the signed files of this folder, the installer the feed announces and the models
chosen, the two default models by default; its `README.txt` tells how to install it. The program
checks the signatures with `albireo-sign` (`--sign`, else `build-albireo/bin`, else the `PATH`) and
stops when it does not find it, unless `--no-verify` leaves that check to the service.

## The signing key

`albireo-sign` is built with the service (`build-albireo/bin`). The private key signs the feed
and the catalog; it stays with the publisher, out of the repository. A new key pair is made with
`albireo-sign keygen <private key> <public key>`; its public key replaces `feed_public_key_pem` in
`dlib/dlib/external/albireo/host/crypto.cpp`, and only the programs built afterwards accept what the
new key signs.
