# Pseudalis distribution

This folder holds the published version of Pseudalis Community, which anyone downloads from
GitHub, and which every installed copy reads for its updates:

| File | Read by | Content |
|---|---|---|
| `pseudalis-setup-<version>.exe` | the users, and the services that update themselves | the installer of the version |
| `feed.json` | the service, 5 minutes after its start, then daily | the version and its installer |
| `feed.json.sig` | the service | signature of `feed.json` |

The services read the feed and its signature in this folder on GitHub, check the signature
(ECDSA P-256 over the exact bytes of the file) with the public key built into the programs
(`resources/feed-public-key.pem` of Pseudalis), and ignore a feed whose signature fails. When the
feed announces a later version, they download the installer it names beside it, check its size
and its SHA-256 against the feed, and install it at their next start. A change of one byte of the
feed therefore needs a new signature, and `.gitattributes` keeps these files byte for byte on
every checkout: Git for Windows would otherwise give the feed CRLF ends of line
(`core.autocrlf`), which its signature does not cover.

The folder holds one installer, the one the feed names; the installers of the earlier versions
stay in the history of the repository. GitHub refuses the files of 100 MB and more: the
installer of Pseudalis stays far under that size.

A version is published once its sources are pushed and its installer built on Windows
(`dlib/dlib/external/pseudalis/README.md`, Publishing a version): on Linux,
`tools/publish_release.py` copies the installer here, writes the feed of the version of
`common/version.h` and signs it with the private key of the publisher, which never enters the
repository, then the folder is committed and pushed.
