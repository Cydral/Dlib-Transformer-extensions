// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_ENGINE_API_H_
#define PSEUDALIS_COMMON_ENGINE_API_H_

/*
    The interface of the engine libraries: pseudalis-engine.dll, built for every x64
    processor, and pseudalis-engine-avx2.dll, built for the processors with AVX2 and FMA,
    which the service loads when the processor and Windows offer them (engine_loader.h).
    Both export the same C functions, so that the service never depends on the compiler or
    on the options of the engine.

    An engine holds the models of the detector and the key of the pseudonymization; it
    processes several files at once, each call on the thread of its caller.

        psd_process(engine, request, data, size, picture, lookup, context, &result, &output, &error)

    request is a JSON object:
        "operation"     "anonymize", "restore", "anonymize_text", "restore_text", "detect",
                        "anonymize_name", "restore_name" or "extract"
        "name"          the name of the file, whose extension tells its kind when its
                        content does not
        "entry"         "extract": the path of an entry in the archive given as data
        "tags"          "anonymize" and "anonymize_text": false writes each value of the text
                        as its substitute, without its tag (true by default)
        "substitutions" [{"substitute", "tag"}]: the substitutes of the text of a file or of
                        a text protected without tags, which "restore" and "restore_text" give
                        back and "anonymize" leaves as they are
    result receives a JSON object: "name" (the name of the output, protected or restored:
    "photo.png.txt" for a picture), "kind" ("txt", "csv", "xml", "docx", "xlsx", "pptx",
    "pdf", "image", "zip", "tar", "tar.gz" or "text"), "report" ({"values": {type: count},
    "removed": [...], "notes": [...], "dropped": [entries of an archive left out],
    "tags_restored", "tags_rejected", "substitutes", "substitutes_restored", "pictures_removed"
    (the picture markers that "restore" removed, which no restoration gives back into a
    document)}), "pictures"
    (the identifiers of the pictures found) and, for "anonymize" and "anonymize_name",
    "substitutions" ([{"substitute", "tag"}]: the values written without their tags in the
    names and the properties, which the caller records for "restore", and, "text": true, in
    the text without tags, once for each place); "substitutions" too for "anonymize_text"
    without tags; "spans" for "detect".  output receives the bytes of the file written.  "anonymize_name" and
    "restore_name" take a name alone as data, a folder for instance, and give the new name
    as output and as "name".
    "anonymize" of an archive also gives "entries", each entry written: [{"original" (its
    path in the archive read), "name" (its path in the archive written), "sha256" (of the
    bytes written), "kind"}], the entries of an archive inside the archive after it, their
    paths after its own and '!' ("dossier/notes.tar.gz!a.txt").  The caller knows by them a
    file extracted from a protected archive.  "extract" gives as output the bytes of the
    entry "entry" of the archive given as data, read as "anonymize" read it: an "original"
    of "entries", whose original archive the caller kept.

    lookup, when not null, receives each candidate substitute that "restore" and
    "restore_name" meet in the names and the properties, and returns 1 with the tags
    recorded for it in *tags, one per line, the most likely first (a buffer the engine
    releases with psd_free(), allocated with psd_alloc()), or 0 when it knows none.
    "anonymize" and "anonymize_name" ask it too, so that a substitute written before, in a
    name or a property edited since, is not replaced again.

    picture, when not null, receives each picture found: in a document (Word, PowerPoint,
    PDF), in an archive, or the file itself, with its identifier (16 hexadecimal digits of a
    digest of the picture keyed by the key of the engine), its SHA-256 and its media type, and
    returns 1 with its description in clear in *description (a buffer the engine releases with
    psd_free(), allocated with psd_alloc()), or 0 without description.  The engine protects the
    description with the detector, with what the document taught it besides, and writes
    "[Image@<identifier> : <description>]" in the place of the picture, or "[Image@<identifier>]"
    without description.

    Each function returns 1 on success, 0 on failure with the reason in error (UTF-8).
*/

#include <stddef.h>

#if defined(_WIN32)
#if defined(PSEUDALIS_ENGINE_BUILD)
#define PSD_API __declspec(dllexport)
#else
#define PSD_API
#endif
#define PSD_CALL __cdecl
#else
#define PSD_API __attribute__((visibility("default")))
#define PSD_CALL
#endif

#define PSEUDALIS_ENGINE_API_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

typedef struct psd_buffer
{
    char* data;
    size_t size;
} psd_buffer;

typedef struct psd_picture
{
    const char* bytes;
    size_t size;
    const char* id;
    const char* sha256;
    const char* media_type;
    const char* location;
} psd_picture;

typedef int (PSD_CALL* psd_picture_fn)(void* context, const psd_picture* picture, psd_buffer* description);
typedef int (PSD_CALL* psd_lookup_fn)(void* context, const char* candidate, size_t size, psd_buffer* tags);

typedef struct psd_engine psd_engine;

PSD_API int PSD_CALL psd_api_version(void);
// "avx2" or "baseline".
PSD_API const char* PSD_CALL psd_variant(void);
/*
    The version and the build of the engine, "1.0.0+20261103170000-0123abcd": the service loads
    only the engine of its own build, which the installer placed beside it.
*/
PSD_API const char* PSD_CALL psd_build(void);
// The models of the engine: {"tagger": source, "span_model": source, "size": "tiny"}.
PSD_API int PSD_CALL psd_models(psd_buffer* description);

/*
    Creates an engine with a key of 32 bytes, and options as a JSON object: "date_order"
    ("dmy" or "mdy"), "context" (the context of the pseudonymization, "default").
*/
PSD_API psd_engine* PSD_CALL psd_create(const char* key, size_t key_size, const char* options, psd_buffer* error);
PSD_API void PSD_CALL psd_destroy(psd_engine* engine);
PSD_API int PSD_CALL psd_process(psd_engine* engine, const char* request, const char* data, size_t size,
    psd_picture_fn picture, psd_lookup_fn lookup, void* context, psd_buffer* result, psd_buffer* output,
    psd_buffer* error);
// The fingerprint of the key, as the reports give it.
PSD_API int PSD_CALL psd_key_fingerprint(psd_engine* engine, psd_buffer* fingerprint);
/*
    Replaces the imposed terms of the engine (common/imposed_terms.h): terms is a JSON list
    [{"term", "type", "case_sensitive"}] of size bytes, which the calls that start afterwards
    read before anything else of their texts.  0, with the reason in error, for a list the
    service does not take, the terms of the engine then unchanged.
*/
PSD_API int PSD_CALL psd_set_terms(psd_engine* engine, const char* terms, size_t size, psd_buffer* error);

PSD_API char* PSD_CALL psd_alloc(size_t size);
PSD_API void PSD_CALL psd_free(psd_buffer* buffer);

#ifdef __cplusplus
}
#endif

#endif // PSEUDALIS_COMMON_ENGINE_API_H_
