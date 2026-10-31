#!/bin/bash
# Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
# License: Boost Software License   See LICENSE.txt for the full license.
#
# Replaces the third-party libraries of the web application with the given versions of
# their npm packages, and writes the versions installed into web/vendor/components.json:
#
#   tools/update_web_vendor.sh <marked> <dompurify> <highlight.js> <katex> <pdfjs-dist> <swagger-ui-dist>
#   tools/update_web_vendor.sh 18.1.1 3.4.17 11.12.0 0.19.0 6.4.299 5.33.2
#
# The web application runs without network: the local service serves every library it uses
# from web/vendor and web/docs, each folder with the license of its component.  The packages
# are fetched with npm into a temporary folder; the files the application serves are copied
# with their licenses, and the references to source maps, which the application does not
# ship, are removed.
#
# web/vendor/components.json is the list of these components, which the page of the licenses
# of the application shows: name, version, npm package, license, the files of its text, the
# folder of the files and the source.  A component added here is added to that list by hand,
# with its role in the five languages of web/js/i18n.js (component_<id>).
#
# PDF.js comes from the legacy build of pdfjs-dist, which runs in Edge and Chrome from
# version 125 on.  Its modern build needs version 147, and a computer kept off the network
# may hold an older browser.  The CMaps of pdfjs-dist give the text of the CJK fonts that a
# PDF names without embedding them.  The standard fonts and the WebAssembly image decoders
# of pdfjs-dist only draw pages, which the application does not do, and are left out.
#
# After an update, test the application (Markdown, code, formulas, PDF documents and the
# documentation of the interface), then review the changes of components.json.

set -euo pipefail
if [ $# -ne 6 ]; then
   sed -n '4,30p' "$0" | sed 's/^# \{0,1\}//'
   exit 2
fi

here="$(cd "$(dirname "$0")/.." && pwd)"
web="$here/web"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cd "$work"
npm init -y > /dev/null
npm install --no-audit --no-fund --ignore-scripts "marked@$1" "dompurify@$2" "@highlightjs/cdn-assets@$3" \
   "katex@$4" "pdfjs-dist@$5" "swagger-ui-dist@$6" > /dev/null
m="$work/node_modules"

copy() { # source destination
   mkdir -p "$(dirname "$2")"
   cp "$1" "$2"
}

copy "$m/marked/lib/marked.esm.js" "$web/vendor/marked/marked.esm.js"
copy "$m/marked/LICENSE" "$web/vendor/marked/LICENSE"
copy "$m/dompurify/dist/purify.es.mjs" "$web/vendor/dompurify/purify.es.mjs"
copy "$m/dompurify/LICENSE" "$web/vendor/dompurify/LICENSE"
copy "$m/@highlightjs/cdn-assets/highlight.min.js" "$web/vendor/highlight/highlight.min.js"
copy "$m/@highlightjs/cdn-assets/styles/github.min.css" "$web/vendor/highlight/github.min.css"
copy "$m/@highlightjs/cdn-assets/styles/github-dark.min.css" "$web/vendor/highlight/github-dark.min.css"
copy "$m/@highlightjs/cdn-assets/LICENSE" "$web/vendor/highlight/LICENSE"
copy "$m/katex/dist/katex.min.js" "$web/vendor/katex/katex.min.js"
copy "$m/katex/dist/katex.min.css" "$web/vendor/katex/katex.min.css"
copy "$m/katex/LICENSE" "$web/vendor/katex/LICENSE"
rm -rf "$web/vendor/katex/fonts"
mkdir -p "$web/vendor/katex/fonts"
cp "$m"/katex/dist/fonts/*.woff2 "$web/vendor/katex/fonts/"
# PDF.js comes from its legacy build, which runs in Edge and Chrome from version 125 on.  The modern
# build needs version 147 or later, and a computer kept off the network may hold an older browser.
copy "$m/pdfjs-dist/legacy/build/pdf.min.mjs" "$web/vendor/pdfjs/pdf.min.mjs"
copy "$m/pdfjs-dist/legacy/build/pdf.worker.min.mjs" "$web/vendor/pdfjs/pdf.worker.min.mjs"
copy "$m/pdfjs-dist/LICENSE" "$web/vendor/pdfjs/LICENSE"
# The CMaps of the CJK fonts that a PDF names without embedding them, which its text needs.  The
# standard fonts and the WebAssembly decoders of images of PDF.js only draw pages.
rm -rf "$web/vendor/pdfjs/cmaps"
mkdir -p "$web/vendor/pdfjs/cmaps"
cp "$m"/pdfjs-dist/cmaps/*.bcmap "$m/pdfjs-dist/cmaps/LICENSE" "$web/vendor/pdfjs/cmaps/"
copy "$m/swagger-ui-dist/swagger-ui-bundle.js" "$web/docs/swagger-ui-bundle.js"
copy "$m/swagger-ui-dist/swagger-ui.css" "$web/docs/swagger-ui.css"
copy "$m/swagger-ui-dist/LICENSE" "$web/docs/LICENSE"
copy "$m/swagger-ui-dist/NOTICE" "$web/docs/NOTICE"

# The source maps are not shipped: their references would only produce failed requests.
for f in "$web"/vendor/*/*.js "$web"/vendor/*/*.mjs "$web"/vendor/*/*.css "$web"/docs/swagger-ui-bundle.js \
   "$web"/docs/swagger-ui.css; do
   sed -i -E -e '/^\/\/# sourceMappingURL=/d' -e 's|/\*# sourceMappingURL=[^*]*\*/||g' "$f"
done

# The versions installed, those of the packages npm resolved, in the list of the components.
node - "$web/vendor/components.json" "$m" <<'EOF'
const fs = require('fs');
const [file, modules] = process.argv.slice(2);
const list = JSON.parse(fs.readFileSync(file, 'utf8'));
for (const c of list.components)
   c.version = JSON.parse(fs.readFileSync(modules + '/' + c.package + '/package.json', 'utf8')).version;
fs.writeFileSync(file, JSON.stringify(list, null, 2) + '\n');
EOF

echo "Updated: marked $1, DOMPurify $2, highlight.js $3, KaTeX $4, PDF.js $5, Swagger UI $6"
