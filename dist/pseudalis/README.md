# Pseudalis distribution

The versions of Pseudalis Community are published on the website of Cydral Technology, where anyone
downloads the installer and where every installed copy reads its updates:

| Address | Read by | Content |
|---|---|---|
| `https://cydral-technology.eu/files/pseudalis/dist/pseudalis-setup-<version>.exe` | the users, from the downloads page, and the services that update themselves | the installer of the version |
| `https://cydral-technology.eu/files/pseudalis/dist/feed.json` | the service, 5 minutes after its start, then daily | the version and its installer |
| `https://cydral-technology.eu/files/pseudalis/dist/feed.json.sig` | the service | signature of `feed.json` |

The repository holds the sources alone: nothing of a publication is written in it. A publication writes
these three files in the chain of the website, a folder of its own on the system of the publisher
(`~/work/cydral-website`), whose folder `releases/files` has the layout of `files/` of the website. The
website gives the files of each product the same layout, `files/<product>/dist/`: the three files go into
`releases/files/pseudalis/dist` of the chain, which publishes them from there and is the only one to carry
files to the hosting. The versions up to 1.0.11 were published in this folder of the repository, whose
history keeps their installers.

The services check the signature of the feed (ECDSA P-256 over the exact bytes of the file) with the
public key built into the programs (`resources/feed-public-key.pem` of Pseudalis), and ignore a feed
whose signature fails. When the feed announces a later version, they download the installer it names
beside it, check its size and its SHA-256 against the feed, and install it at their next start. A change
of one byte of the feed therefore needs a new signature: the feed is written with LF ends of line and
published as it is.

A version is published once its sources are pushed and its installer built on Windows
(`dlib/dlib/external/pseudalis/README.md`, Publishing a version): on Linux,
`tools/publish_release.py` copies the installer into that folder, writes the feed of the version of
`common/version.h`, signs it with the private key of the publisher, which never enters the repository,
runs the chain of the website (`tools/site.sh status`, then `tools/site.sh publish`), and waits until the
website serves the feed, its signature and the installer as they were written.
