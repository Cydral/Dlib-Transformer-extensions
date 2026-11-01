# Albireo distribution

This folder is read by every installed copy of Albireo:

| File                | Read by                                   | Content                                             |
|---------------------|-------------------------------------------|-----------------------------------------------------|
| `feed.json`         | the service, at each start and every 6 h  | the latest version and its installer, named in the public link |
| `feed.json.sig`     | the service                               | signature of `feed.json`                            |
| `catalog.json`      | the service and the Models window         | the models offered, their files and their hashes    |
| `catalog.json.sig`  | the service                               | signature of `catalog.json`                         |

The services fetch these files from the folder `dist` of the public link of Albireo first, a
public folder of pCloud read only, protected by a password built into the service
(`dlib/dlib/external/albireo/host/public_link.h`), then from
`https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/albireo/` when the
link does not answer or holds no file whose signature verifies. They check each signature (ECDSA
P-256 over the exact bytes of the file) with the public key built into the programs, and ignore a
file whose signature fails. A change of one byte of a JSON file therefore needs a new signature.

The installers lie in the folder `dist` of the public link, as `albireo-setup-online-<version>.exe`,
the installer of the updates, and `albireo-setup-complete-<version>.exe`, with the models that come
with Albireo, which lie beside it in `albireo-setup-complete-<version>.models` when both would reach
the 4 GiB Windows runs; the files of the models lie in its folder `models`, each at
`<identifier>/<file name of its official address>`, and the services read them there before their
official addresses, the hashes of the catalog checking both. The models published outside the
Hugging Face hub are attached to the GitHub release `albireo-models`, and the repository itself
keeps only the small files of this folder.

## Publishing a version

1. Set the version in `dlib/dlib/external/albireo/common/version.h`, commit and push the sources.
2. Build the installers on Windows (see the README of Albireo, section Building on Windows):
   `albireo-setup-online.exe`, which the feed announces, and `albireo-setup.exe`, complete with
   the models that come with Albireo.
3. Write the release notes as `notes.json`, `{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}`;
   without this file, the notes the feed already holds for the same version are kept.
4. Sign the feed that names the installers in the public link, then push the feed:

```
python3 dlib/dlib/external/albireo/tools/publish_release.py build-albireo/bin/albireo-setup-online.exe --public-link --complete build-albireo/bin/albireo-setup.exe --key <private key> --notes notes.json
git add dist/albireo/feed.json dist/albireo/feed.json.sig
git commit -m "Albireo <version>"
git push
```

5. Fill the local copy of the public link, the folder the storage service synchronizes, then check
   the link once it is synchronized; the password of the link comes from
   `keys/albireo-public-link.txt` of the folder of the user, or from `ALBIREO_PUBLIC_LINK_PASSWORD`:

```
python dlib/dlib/external/albireo/tools/fill_public_link.py dist <copy of the link>/dist --online build-albireo/bin/albireo-setup-online.exe --complete build-albireo/bin/albireo-setup.exe
python dlib/dlib/external/albireo/tools/fill_public_link.py models <copy of the link>/models --from <folder of models>
python dlib/dlib/external/albireo/tools/fill_public_link.py check
```

`publish_release.py --public-link` checks that the installer carries the version of `version.h`,
writes the feed with the name of the installer in the link, its size and its SHA-256, signs it,
checks the signature and puts the feed and its signature in place together; nothing goes to GitHub.
`fill_public_link.py dist` copies the signed files of this folder, checked, and both installers under
their names, with `albireo-setup-complete-<version>.models` when the models of the complete installer
lie beside it (Windows runs no program of 4 GiB or more), the feed last, so that the link never
announces an installer it does not hold yet;
`models` copies each file of the catalog from the folders of `--from` when a file has its size and
SHA-256, else downloads it from its official address, and checks it before it takes its name;
`check` lists what the link lacks.

Without `--public-link`, `publish_release.py` publishes on GitHub: it creates the
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
functions (general, code, security, embedding), publishers, inputs (text, image), descriptions in
the five languages, licenses, and the file of each model on the Hugging Face hub, with the projector
of a model that reads images. The script asks the hub for the commit, the size and the hash of each
file and writes the catalog:

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
