# Pseudalis distribution

This folder is read by every installed copy of Pseudalis:

| File            | Read by                                          | Content                                         |
|-----------------|--------------------------------------------------|-------------------------------------------------|
| `feed.json`     | the service, 5 minutes after its start, then daily | the latest version and its installer          |
| `feed.json.sig` | the service                                      | signature of `feed.json`                        |

The services fetch the feed from the folder `dist` of the public link of the published files
first, a public folder of pCloud read only, protected by a password built into the service
(`dlib/dlib/external/pseudalis/host/public_link.h`), where it is named `pseudalis-feed.json` with
its signature `pseudalis-feed.json.sig`, then from
`https://raw.githubusercontent.com/Cydral/Dlib-Transformer-extensions/main/dist/pseudalis/feed.json`
when the link does not answer or holds no feed whose signature verifies. They check the signature
(ECDSA P-256 over the exact bytes of the file) with the public key built into the programs
(`resources/feed-public-key.pem` of Pseudalis), and ignore a feed whose signature fails. A change of
one byte of the feed therefore needs a new signature, and `.gitattributes` keeps these files byte
for byte on every checkout: Git for Windows would otherwise give them CRLF ends of line
(`core.autocrlf`), which their signatures do not cover.

The installer lies in the folder `dist` of the public link as `pseudalis-setup-<version>.exe`; the
feed gives its size and its SHA-256, which the services check before they install it. Albireo
publishes its own files in the same folder, under names of their own.

A version is published in three steps (`dlib/dlib/external/pseudalis/README.md`, Publishing a
version):

1. `tools/publish_release.py` writes and signs the feed of the version of `common/version.h`
   from its installer, with the private key of the publisher, which never enters the repository;
2. the feed and its signature are committed and pushed;
3. `tools/fill_public_link.py dist` copies the installer, then the signature and the feed, into
   the local copy of the public link, which the storage service synchronizes, and
   `tools/fill_public_link.py check` lists what the link still lacks.
