cpp-httplib bundled with dlib
=============================

Version   : 0.63.0
Origin    : https://github.com/yhirose/cpp-httplib, tag v0.63.0
            (commit 5bcfea7b90575c80a3e54a6967b11b39afa60d5c)
SHA-256   : 9021001a719768893a8a6e4051c56107690e8b8e49b41e5f903534fa19ae9afb (httplib.h as
            released, before the changes listed below)
License   : MIT, see LICENSE in this folder

How to use it
-------------

A program includes <dlib/external/httplib/httplib.h> and, with CMake, links the
dlib::httplib target:

   target_link_libraries(my_program dlib::dlib dlib::httplib)

The target carries everything the header needs, so that every program of the
repository configures it the same way:

   - HTTPS through the Mbed TLS copy in dlib/external/mbedtls (CPPHTTPLIB_MBEDTLS_SUPPORT),
     unless DLIB_HTTPLIB_TLS is OFF, in which case the programs speak HTTP only;
   - the socket and certificate libraries of Windows (ws2_32, crypt32), and
     _WIN32_WINNT=0x0A00, since the header supports Windows 10 and later only;
   - the threads library elsewhere.

In a project build, Mbed TLS is compiled only when a program links dlib::httplib.  Clients
verify servers against the certificates of the system: the Windows certificate stores, the
macOS keychain, or the usual CA bundle paths of Linux.

The other options of the header (CPPHTTPLIB_*) keep their upstream defaults and may be
defined by a program before the include.

Changes made to the released header
-----------------------------------

The changes are marked in the source by comments starting with "dlib:", so that a newer
release can be compared with this copy and each change carried over or dropped.

1. Static files on Windows.  detail::canonicalize_path() resolved the base directory of a
   mount point and the requested file with _fullpath(), which works on the narrow,
   code-page form of a path and does not follow junctions or symbolic links.  A junction
   placed under a mounted directory could therefore make the server deliver a file from
   outside it.  The function now opens the path and asks Windows for its final name
   (GetFinalPathNameByHandleW), in UTF-16, and the check that the file lies inside the base
   directory compares those final names.

2. Header lines ended by a bare LF.  read_headers() skipped such a line and went on parsing
   the next ones.  An intermediary that reads the bare LF as a line end sees one header more
   than the server does, which is the ground of request smuggling.  The line now makes the
   request invalid (400), unless CPPHTTPLIB_ALLOW_LF_AS_LINE_TERMINATOR is defined, in which
   case the bare LF is accepted as a line end as before.

3. Transfer codings of requests.  A request framed by "Transfer-Encoding: gzip, chunked"
   was read as chunked and reached its handler with a body still compressed, presented as
   the content, since the library decodes no transfer coding but chunked.  Such a request
   is now answered 501, as RFC 9112 6.1 prescribes for a coding the server does not
   understand, and a request listing chunked more than once is answered 400.  The checks
   run in Server::process_request() through detail::check_request_transfer_codings().

Tests
-----

test_httplib in the dtest suite starts servers and clients on the loopback interface and
checks routing, parameters, bodies of every encoding, streaming, server-sent events,
multipart forms, ranges, static files, limits and malformed requests, keep-alive and
concurrency, and TLS 1.2 and 1.3 with certificate and host name verification.

The test suite of the release (test/test.cc), built against this copy with the dlib::httplib
target, passes but for RequestSmugglingTest.NonFinalChunkedTransferEncodingRejected, which
expects the "gzip, chunked" requests that change 3 refuses to be served.  Its tests that
reach servers on the Internet, or need an IPv6 loopback, depend on the machine.

Updating
--------

Replace httplib.h with a newer release, carry over the changes listed above if the release
does not already contain them, update the version, origin and hash above, and run the dtest
suite.
