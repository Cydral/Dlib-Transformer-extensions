SQLite amalgamation bundled with dlib
=====================================

Version   : 3.54.1, source id 2026-10-16 14:45:20
            555e07ef33f69a0957d24d8e555b702464a131d6f1607f4c3c1e4551e8e6ca76
Origin    : https://www.sqlite.org/2026/sqlite-amalgamation-3540100.zip
SHA3-256  : 92281c86e2957f9b9305890f9ac04204caa7b677cc868dc326d942689df28e30 (archive,
            as published on https://www.sqlite.org/download.html)
Files     : sqlite3.c, sqlite3.h and sqlite3ext.h, kept byte for byte as released.
License   : SQLite is in the public domain, see https://www.sqlite.org/copyright.html

How dlib uses this copy
-----------------------

dlib/sqlite.h wraps the C interface of SQLite.  With CMake, DLIB_LINK_WITH_SQLITE3 is ON
by default and dlib compiles sqlite3.c into its own library, so a program that uses
dlib/sqlite.h links nothing else and runs on a machine where SQLite is not installed.
The folder of this file is added to the public include directories of the dlib target,
which is how #include <sqlite3.h> resolves in the build tree and after an install.

Setting DLIB_USE_SYSTEM_SQLITE3=ON links the SQLite library found on the system
instead, and falls back to this copy when none is found.

The copy is compiled with the following definitions, chosen among the options the SQLite
documentation recommends, none of which changes the SQL a program can run:

   SQLITE_THREADSAFE=1               serialized mode, the database handles of dlib may be
                                     used from several threads
   SQLITE_DQS=0                      double-quoted strings are identifiers only, never
                                     literals
   SQLITE_DEFAULT_MEMSTATUS=0        no global memory statistics, a lock less per
                                     allocation
   SQLITE_DEFAULT_WAL_SYNCHRONOUS=1  NORMAL synchronization when a database uses WAL
   SQLITE_LIKE_DOESNT_MATCH_BLOBS    LIKE and GLOB never match a BLOB operand
   SQLITE_TEMP_STORE=2               temporary tables and indices kept in memory unless a
                                     PRAGMA asks otherwise
   SQLITE_SECURE_DELETE              deleted content is overwritten with zeros
   SQLITE_ENABLE_API_ARMOR           misuse of the C interface is detected and reported
   SQLITE_ENABLE_FTS5                full-text search
   SQLITE_ENABLE_MATH_FUNCTIONS      the built-in mathematical SQL functions
   SQLITE_OMIT_LOAD_EXTENSION        no loading of extensions from shared libraries

Builds made without CMake compile sqlite3.c as a C file beside dlib/all/source.cpp, with
the folder of this file on the include path and the definitions above.

Updating
--------

Replace the three files with those of a newer amalgamation, update the version, the
origin and the hash above, and run the dtest suite: test_sqlite exercises dlib/sqlite.h
and dlib/sqlite/sqlite_tools.h against this copy.
