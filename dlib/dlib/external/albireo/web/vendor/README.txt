Third-party components of the Albireo web application
=====================================================

The web application runs without network: every library it uses is served by the
local service from this folder.  Each folder keeps the license of its component.

   marked        18.1.1    MIT License                      vendor/marked
   DOMPurify     3.4.17    Apache License 2.0 or MPL 2.0     vendor/dompurify
   highlight.js  11.12.0   BSD 3-Clause License              vendor/highlight
   KaTeX         0.19.0    MIT License                      vendor/katex
   PDF.js        6.4.299   Apache License 2.0                vendor/pdfjs
   Swagger UI    5.33.2    Apache License 2.0                docs/

The files come from the npm packages of the same names (@highlightjs/cdn-assets for
highlight.js, pdfjs-dist for PDF.js, swagger-ui-dist for Swagger UI).  To update them,
run tools/update_web_vendor.sh with the versions to install, then test the application.
