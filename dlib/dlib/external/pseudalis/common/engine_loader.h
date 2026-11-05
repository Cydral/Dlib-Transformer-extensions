// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_ENGINE_LOADER_H_
#define PSEUDALIS_COMMON_ENGINE_LOADER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine_api.h"
#include "json.h"

namespace pseudalis
{
    // Whether the processor and the system run AVX2 and FMA code: the bits of CPUID and the
    // AVX state that the system saves (XGETBV).
    bool cpu_has_avx2();

    class engine_library
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The engine library of this computer, loaded at run time: on Windows
                pseudalis-engine-avx2.dll beside the program when the processor runs AVX2,
                else pseudalis-engine.dll; elsewhere the engine compiled into the program
                (PSEUDALIS_ENGINE_STATIC) or libpseudalis-engine.so beside it.  The library is
                loaded by its full path, the libraries it imports from its folder and System32
                alone, and is taken only when it is of the version and of the build of the
                program (psd_build()): a library of the same name that another installation
                or another build left is refused.  The library
                stays loaded until the process ends, whatever becomes of the object: the pool
                of threads its networks run on, the one of dlib, would be joined by its
                unloading, which Windows runs under its loader lock that the ending threads
                need, so that the unloading would wait forever and the service with it.
        !*/
    public:
        engine_library() {}
        ~engine_library();
        engine_library(const engine_library&) = delete;
        engine_library& operator=(const engine_library&) = delete;

        // Loads the library of folder; force names a variant, "avx2" or "baseline", or is empty.
        bool load(const std::string& folder, const std::string& force, std::string& error);
        bool loaded() const { return create != nullptr; }
        const std::string& path() const { return path_; }

        int (PSD_CALL* api_version)(void) = nullptr;
        const char* (PSD_CALL* variant)(void) = nullptr;
        const char* (PSD_CALL* build)(void) = nullptr;
        int (PSD_CALL* models)(psd_buffer*) = nullptr;
        psd_engine* (PSD_CALL* create)(const char*, size_t, const char*, psd_buffer*) = nullptr;
        void (PSD_CALL* destroy)(psd_engine*) = nullptr;
        int (PSD_CALL* process)(psd_engine*, const char*, const char*, size_t, psd_picture_fn, psd_lookup_fn, void*,
            psd_buffer*, psd_buffer*, psd_buffer*) = nullptr;
        int (PSD_CALL* key_fingerprint)(psd_engine*, psd_buffer*) = nullptr;
        int (PSD_CALL* set_terms)(psd_engine*, const char*, size_t, psd_buffer*) = nullptr;
        char* (PSD_CALL* alloc)(size_t) = nullptr;
        void (PSD_CALL* free_buffer)(psd_buffer*) = nullptr;

    private:
        void* handle_ = nullptr;
        std::string path_;
    };

    struct picture_request
    {
        std::string bytes, id, sha256, media_type, location;
    };

    class engine_client
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An engine of a loaded library, with its key: the calls of the service.
                process() may be called from several threads at once.
        !*/
    public:
        engine_client(const std::shared_ptr<engine_library>& lib, const std::string& key, const json& options,
            std::string& error);
        ~engine_client();
        engine_client(const engine_client&) = delete;
        engine_client& operator=(const engine_client&) = delete;

        bool ok() const { return e_ != nullptr; }
        /*!
            ensures
                - Runs request (engine_api.h) on data; result receives the result document
                  and output the bytes written.  pictures, when given, receives each
                  picture and returns its description, or an empty string; lookup, when
                  given, receives each candidate substitute of a restoration and returns
                  the tags recorded for it.
                - Returns false with the reason in error.
        !*/
        bool process(const json& request, const std::string& data, json& result, std::string& output,
            const std::function<std::string(const picture_request&)>& pictures, std::string& error,
            const std::function<std::vector<std::string>(const std::string&)>& lookup =
                std::function<std::vector<std::string>(const std::string&)>()) const;
        std::string key_fingerprint() const;
        // Replaces the imposed terms of the engine ([{"term", "type", "case_sensitive"}]).
        bool set_terms(const json& terms, std::string& error) const;
        std::string variant() const;
        // The version and the build of the engine (psd_build()).
        std::string build() const;
        json models() const;

    private:
        std::shared_ptr<engine_library> lib_;
        psd_engine* e_ = nullptr;
    };
}

#endif // PSEUDALIS_COMMON_ENGINE_LOADER_H_
