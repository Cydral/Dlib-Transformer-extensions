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
whose assets accept files up to 2 GB: the repository itself keeps only these small files.

## Publishing a version

1. Set the version in `dlib/dlib/external/albireo/common/version.h`.
2. Build the installer on Windows (see the README of Albireo, section Building on Windows).
3. Create the release `albireo-v<version>` on GitHub and attach `albireo-setup-<version>.exe`.
4. Write the release notes as `notes.json`, `{"en": "...", "fr": "...", "de": "...", "it": "...", "es": "..."}`;
   without this file, `make_feed.py` keeps the notes the feed already holds for the same version.
5. Write and sign the feed, then push it:

```
python3 dlib/dlib/external/albireo/tools/make_feed.py dist/albireo/feed.json <version> <path of albireo-setup.exe> https://github.com/Cydral/Dlib-Transformer-extensions/releases/download/albireo-v<version>/albireo-setup-<version>.exe notes.json
albireo-sign sign <private key> dist/albireo/feed.json
albireo-sign verify dist/albireo/feed.json
git add dist/albireo/feed.json dist/albireo/feed.json.sig
git commit -m "Albireo <version>"
git push
```

The services see the new feed within six hours, or at once with "Check for updates" in the menu
of the icon, download the installer in the background, check its size and its SHA-256, and install
it when the service restarts.

## Updating the catalog

The list of models is in `dlib/dlib/external/albireo/tools/make_catalog.py`: identifiers,
functions (general, code, security), descriptions in the five languages, licenses, and the file of
each model on the Hugging Face hub. The script asks the hub for the commit, the size and the hash of
each file and writes the catalog:

```
python3 dlib/dlib/external/albireo/tools/make_catalog.py dist/albireo/catalog.json
albireo-sign sign <private key> dist/albireo/catalog.json
albireo-sign verify dist/albireo/catalog.json
```

Exactly one model carries `"default": true`: the service downloads it at its first start.

A model published elsewhere, as parts attached to a release for instance, is cut with
`tools/publish_model.py`, which prints the `size`, `sha256` and `files` of its entry; the entry is
then added to the list of `make_catalog.py` with these fields in place of `hub`.

## The signing key

`albireo-sign` is built with the service (`build-albireo/bin`). The private key signs the feed
and the catalog; it stays with the publisher, out of the repository. A new key pair is made with
`albireo-sign keygen <private key> <public key>`; its public key replaces `feed_public_key_pem` in
`dlib/dlib/external/albireo/host/crypto.cpp`, and only the programs built afterwards accept what the
new key signs.
