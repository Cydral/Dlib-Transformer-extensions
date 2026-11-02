# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
"""
The public link of the published files of Albireo, a public folder of pCloud read through its
API, as the service reads it (host/public_link.h), the only source of its downloads: its folder
dist holds the update feed, the catalog, their signatures and the installers, its folder models
the files of the models of the catalog, each at models/<identifier>/<file name of its official
address>.

The password of the link comes from the variable ALBIREO_PUBLIC_LINK_PASSWORD, else from the
file keys/albireo-public-link.txt of the home folder (%USERPROFILE% on Windows), the file the
configuration of the service reads (ALBIREO_PUBLIC_LINK_PASSWORD_FILE).

No message names the link, its code, its password, an address of its API or of a content
server, or a host of pCloud: hidden() replaces them in the messages of the tools, as the service
does in its own (common/log.h).
"""

import json
import os
import posixpath
import re
import urllib.parse
import urllib.request

DEFAULT_LINK = "https://u.pcloud.link/publink/show?code=kZeNhSJZpcMygkcCvGXmq1MTvkCfBuSsx2GV"
SPACE = "the publication space of Albireo"

APIS = {
    "u.pcloud.link": "https://api.pcloud.com",
    "my.pcloud.com": "https://api.pcloud.com",
    "e.pcloud.link": "https://eapi.pcloud.com",
    "e.pcloud.com": "https://eapi.pcloud.com",
}


def password_file():
    home = os.environ.get("USERPROFILE") if os.name == "nt" else None
    return os.path.join(home or os.path.expanduser("~"), "keys", "albireo-public-link.txt")


def read_password():
    """The password of the link, without the byte order mark and the ends of line of its file;
    an empty text when neither the variable nor the file gives one."""
    given = os.environ.get("ALBIREO_PUBLIC_LINK_PASSWORD")
    if given is not None:
        return given
    try:
        with open(password_file(), "rb") as f:
            data = f.read()
    except OSError:
        return ""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.rstrip(b"\r\n").decode("utf-8")


def api_of(link):
    """The origin of the API of link and its code; a loopback address is its own API, for the tests."""
    m = re.fullmatch(r"(https://[A-Za-z.]+|http://127\.0\.0\.1:\d{1,5})/publink/show\?code=([A-Za-z0-9]{8,64})(&.*)?",
                     link)
    if not m:
        raise ValueError("the link given is not a public link")
    origin = m.group(1)
    host = origin.split("://", 1)[1].lower()
    if origin.startswith("https://"):
        if host not in APIS:
            raise ValueError("the link given is not a public link")
        return APIS[host], m.group(2)
    return origin, m.group(2)


# The codes and the passwords of the links read, the code of the default link first, which hidden() replaces as well.
_secrets = {api_of(DEFAULT_LINK)[1]}


def hidden(message):
    """The message with each word that names the publication space replaced by SPACE: an address or a
    host of pCloud (a word holding "pcloud" or "publink", whatever their case), or a code or a password
    of a link read.  A word ends at a space, a quote, a bracket, a comma or a semicolon, and the
    punctuation that ends a sentence after it stays."""
    def replace(m):
        word = m.group(0)
        tail = ""
        while word and word[-1] in ".:!?":
            word, tail = word[:-1], word[-1] + tail
        folded = word.lower()
        if word and ("pcloud" in folded or "publink" in folded or any(x in word for x in _secrets)):
            return SPACE + tail
        return m.group(0)
    return re.sub(r"[^\s\"'`<>()\[\]{},;|\\]+", replace, str(message))


def without_query(message):
    """The message with the query of each address it quotes left out, the punctuation after the
    address kept: "<address>: status 404"."""
    def strip(m):
        token, tail = m.group(0), ""
        while token and token[-1] in ":,;)":
            token, tail = token[:-1], token[-1] + tail
        return token.split("?", 1)[0] + tail
    return re.sub(r"\w+://[^\s'\"]+", strip, message)


def model_path(identifier, url):
    """The path in the link of the file of the model identifier published at url."""
    name = posixpath.basename(urllib.parse.urlparse(url).path)
    if not name or "/" in identifier:
        raise ValueError("no file name in %s" % url)
    return "models/%s/%s" % (identifier, name)


class Link:
    """The listing and the addresses of a public link, read through the API of pCloud."""

    def __init__(self, link=DEFAULT_LINK, password=None):
        self.api, self.code = api_of(link)
        self.password = read_password() if password is None else password
        self._files = None
        _secrets.add(self.code)
        if len(self.password) >= 6:
            _secrets.add(self.password)

    def _call(self, method, **params):
        params["code"] = self.code
        if self.password:
            params["linkpassword"] = self.password
        url = "%s/%s?%s" % (self.api, method, urllib.parse.urlencode(params))
        request = urllib.request.Request(url, headers={"User-Agent": "albireo-tools"})
        try:
            with urllib.request.urlopen(request, timeout=30) as r:
                answer = json.load(r)
        except (OSError, ValueError) as e:
            raise RuntimeError("the public link cannot be reached: %s" % hidden(without_query(str(e))))
        if not isinstance(answer, dict):
            raise RuntimeError("the public link gives an answer that is not JSON")
        if answer.get("result") != 0:
            raise RuntimeError('the public link answers "%s" (%s)' % (hidden(answer.get("error", ""))[:80],
                                                                      hidden(answer.get("result"))))
        return answer

    def files(self):
        """{path: (identifier, size)} of the files of the folder of the link, its subfolders walked."""
        if self._files is None:
            found = {}

            def walk(folder, prefix, depth):
                for c in folder.get("contents", []):
                    name = c.get("name", "")
                    if not name or "/" in name or name in (".", ".."):
                        continue
                    if c.get("isfolder"):
                        if depth < 8:
                            walk(c, prefix + name + "/", depth + 1)
                    elif c.get("fileid"):
                        found[prefix + name] = (c["fileid"], c.get("size", 0))

            metadata = self._call("showpublink").get("metadata")
            if not isinstance(metadata, dict):
                raise RuntimeError("the public link gives a listing that is not the one of its API")
            walk(metadata, "", 0)
            self._files = found
        return self._files

    def address(self, path):
        """The address of the file path of the link on a content server, valid for a few hours."""
        entry = self.files().get(path)
        if not entry:
            raise RuntimeError("the publication space of Albireo does not hold %s yet" % path)
        answer = self._call("getpublinkdownload", fileid=entry[0], forcedownload=1)
        hosts = answer.get("hosts") or []
        where = answer.get("path", "")
        if not hosts or not where.startswith("/") or not re.fullmatch(r"[A-Za-z0-9.:-]+", hosts[0]):
            raise RuntimeError("the public link gives no address for %s" % path)
        scheme = "https://" if self.api.startswith("https://") else "http://"
        return scheme + hosts[0] + urllib.parse.quote(where, safe="/%:@!$&'()*+,;=-._~")
