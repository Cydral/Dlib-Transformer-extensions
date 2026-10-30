# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
Creates the releases of a GitHub repository and attaches files to them, through the REST
interface of GitHub, for publish_release.py (the installers of Albireo) and publish_model.py
(the models published outside the Hugging Face hub).

The token is read from GITHUB_TOKEN, else GH_TOKEN, else from the GitHub command line
(gh auth token); it needs the right to write the contents of the repository.  GITHUB_API_URL
replaces the address of the interface, for GitHub Enterprise.  The files go through the
proxy of HTTPS_PROXY when it is set.

A release holds files of less than 2 GiB each.  A file attached under a name already taken
is left as it is when GitHub gives for the earlier one the same size and SHA-256, and
replaces it otherwise, a file without SHA-256 included.
"""

import hashlib
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MAX_ASSET = (2 << 30) - 1
DEFAULT_REPOSITORY = "Cydral/Dlib-Transformer-extensions"


def api_base():
    return os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")


def find_token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError("no GitHub token: set GITHUB_TOKEN, or sign in with the GitHub command line (gh auth login)")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_url(repository, tag, name):
    """The public address of the file name of the release tag."""
    return "https://github.com/%s/releases/download/%s/%s" % (repository, urllib.parse.quote(tag),
                                                             urllib.parse.quote(name))


class _progress_reader:
    """Reads a file for an upload and prints how much of it has gone."""

    def __init__(self, f, size, label):
        self.f = f
        self.size = size
        self.label = label
        self.sent = 0
        self.shown = 0.0

    def read(self, n=-1):
        data = self.f.read(n)
        if not data:
            return data
        self.sent += len(data)
        now = time.monotonic()
        if now - self.shown >= 2 or self.sent == self.size:
            self.shown = now
            print("\r  %s: %d of %d MB" % (self.label, self.sent >> 20, self.size >> 20), end="", file=sys.stderr)
            if self.sent == self.size:
                print(file=sys.stderr)
        return data


class repository:
    """The releases of one repository, owner/name."""

    def __init__(self, name=DEFAULT_REPOSITORY, token=None):
        self.name = name
        self.token = token or find_token()

    def _request(self, method, url, body=None, data=None, headers=None, timeout=120):
        h = {
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + self.token,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "albireo-publisher",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            h["Content-Type"] = "application/json"
        h.update(headers or {})
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read()
                return r.status, json.loads(text) if text else None
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                message = json.loads(text).get("message", text)
            except ValueError:
                message = text
            return e.code, {"message": message}

    def _api(self, method, path, body=None, expect=(200,)):
        status, j = self._request(method, api_base() + path, body=body)
        if status not in expect:
            raise RuntimeError("%s %s: %d %s" % (method, path, status, (j or {}).get("message", "")))
        return j

    def release(self, tag):
        """The release of tag, or None."""
        status, j = self._request("GET", "%s/repos/%s/releases/tags/%s" % (api_base(), self.name,
                                                                           urllib.parse.quote(tag)))
        if status == 404:
            return None
        if status != 200:
            raise RuntimeError("release %s: %d %s" % (tag, status, (j or {}).get("message", "")))
        return j

    def ensure_release(self, tag, title, notes, latest=True):
        """The release of tag, created at the head of the default branch when missing."""
        r = self.release(tag)
        if r is not None:
            return r
        body = {"tag_name": tag, "name": title, "body": notes, "draft": False, "prerelease": False,
                "make_latest": "true" if latest else "false"}
        r = self._api("POST", "/repos/%s/releases" % self.name, body=body, expect=(201,))
        print("release %s created: %s" % (tag, r.get("html_url", "")), file=sys.stderr)
        return r

    def assets(self, release):
        return self._api("GET", "/repos/%s/releases/%d/assets?per_page=100" % (self.name, release["id"]))

    def is_private(self):
        """Whether the repository is private, its files then downloaded by signed-in users only."""
        return bool((self._api("GET", "/repos/%s" % self.name) or {}).get("private"))

    def attach(self, release, path, name=None, sha256=None, attempts=3):
        """Attaches the file path to release under name and returns its asset.  A file of the
        same name, size and SHA-256 stays; one of another content, or whose SHA-256 GitHub does
        not give, is replaced."""
        name = name or os.path.basename(path)
        size = os.path.getsize(path)
        if size > MAX_ASSET:
            raise RuntimeError("%s holds %d bytes, more than the 2 GiB a release file may hold" % (path, size))
        digest = "sha256:" + (sha256 or sha256_of(path))
        upload = release["upload_url"].split("{", 1)[0]
        error = None
        for attempt in range(attempts):
            for a in self.assets(release):
                if a["name"] != name:
                    continue
                if a.get("state") == "uploaded" and a.get("size") == size and a.get("digest") == digest:
                    print("  %s: already attached" % name, file=sys.stderr)
                    return a
                self._api("DELETE", "/repos/%s/releases/assets/%d" % (self.name, a["id"]), expect=(204,))
            with open(path, "rb") as f:
                status, j = self._request("POST", upload + "?name=" + urllib.parse.quote(name),
                                          data=_progress_reader(f, size, name),
                                          headers={"Content-Type": "application/octet-stream",
                                                   "Content-Length": str(size)},
                                          timeout=600)
            # A server that computes no SHA-256 gives none, and the size then checks the upload.
            if status == 201 and j.get("size") == size and j.get("digest") in (digest, None):
                return j
            error = "%d %s" % (status, (j or {}).get("message", "the size or the digest differs"))
            print("  %s: attempt %d failed, %s" % (name, attempt + 1, error), file=sys.stderr)
        raise RuntimeError("%s could not be attached: %s" % (name, error))


def read_back(url, path, sha256=None, timeout=120):
    """Whether the file served at url, a public address, is the file path: the whole download
    is compared with its size and its SHA-256.  Raises OSError when the address cannot be read
    to its end."""
    size = os.path.getsize(path)
    expected = sha256 or sha256_of(path)
    label = os.path.basename(path)
    h = hashlib.sha256()
    got = 0
    shown = 0.0
    req = urllib.request.Request(url, headers={"User-Agent": "albireo-publisher"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            # An announced size other than the one of the file is another file; the same one
            # tells a transfer cut before its end.
            length = r.headers.get("Content-Length", "").strip()
            if length.isdigit() and int(length) != size:
                return False
            while got <= size:
                block = r.read(1 << 20)
                if not block:
                    break
                got += len(block)
                h.update(block)
                now = time.monotonic()
                if now - shown >= 2:
                    shown = now
                    print("\r  %s: %d of %d MB read back" % (label, got >> 20, size >> 20), end="", file=sys.stderr)
    except http.client.HTTPException as e:
        raise OSError("the transfer stopped after %d bytes (%r)" % (got, e))
    print("\r  %s: %d of %d MB read back" % (label, got >> 20, size >> 20), file=sys.stderr)
    if length.isdigit() and got < size:
        raise OSError("the transfer stopped after %d of %d bytes" % (got, size))
    return got == size and h.hexdigest() == expected


def verify_public(repo, url, path, sha256=None):
    """Reads back the whole file served at url, the public address of the file path attached
    to a release of repo.  Raises RuntimeError when the address serves another file, or when
    it cannot be read and the repository is not known to be private.  Returns a warning when
    the repository is private, whose files only signed-in users download, and None when the
    address serves the file."""
    try:
        same = read_back(url, path, sha256)
    except OSError as e:
        failure = e
    else:
        if not same:
            raise RuntimeError("%s serves another file than %s" % (url, path))
        return None
    try:
        private = repo.is_private()
    except (OSError, RuntimeError, ValueError) as e:
        raise RuntimeError("%s cannot be read (%s), and whether %s is private cannot be told (%s)"
                           % (url, failure, repo.name, e))
    if not private:
        raise RuntimeError("%s cannot be read (%s)" % (url, failure))
    return ("%s cannot be read without signing in (%s): %s is a private repository, whose files the services "
            "download only once it is public" % (url, failure, repo.name))
