// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_ENGINE_LOADER_H_
#define ALBIREO_COMMON_ENGINE_LOADER_H_

#include <string>

#include "engine_api.h"
#include "json.h"

namespace albireo
{

    class engine_library
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An inference engine library loaded at run time, with the functions of
                engine_api.h resolved.  The host keeps one for the engine it selected;
                the library is released when the object is destroyed, after the engine
                created from it.
        !*/
    public:
        engine_library() {}
        ~engine_library();
        engine_library(const engine_library&) = delete;
        engine_library& operator=(const engine_library&) = delete;

        // Loads the library at path (UTF-8); false with the reason in error otherwise.
        bool load(const std::string& path, std::string& error);
        void unload();
        bool loaded() const { return handle_ != nullptr; }
        const std::string& path() const { return path_; }

        // The functions of the library.
        uint32_t (ALBIREO_ENGINE_CALL* abi_version)(void) = nullptr;
        int (ALBIREO_ENGINE_CALL* describe)(char*, size_t, size_t*) = nullptr;
        albireo_engine* (ALBIREO_ENGINE_CALL* create)(
            const char*, albireo_log_callback, void*, char*, size_t) = nullptr;
        void (ALBIREO_ENGINE_CALL* destroy)(albireo_engine*) = nullptr;
        int (ALBIREO_ENGINE_CALL* load_model)(
            albireo_engine*, const char*, const char*, const char*, char*, size_t) = nullptr;
        int (ALBIREO_ENGINE_CALL* unload_model)(albireo_engine*, const char*) = nullptr;
        int (ALBIREO_ENGINE_CALL* query)(albireo_engine*, const char*, char*, size_t, size_t*) = nullptr;
        int (ALBIREO_ENGINE_CALL* submit)(
            albireo_engine*, const char*, albireo_event_callback, void*, uint64_t*, char*, size_t) = nullptr;
        int (ALBIREO_ENGINE_CALL* cancel)(albireo_engine*, uint64_t) = nullptr;

        // describe() as a document, or null.
        json description() const;

        // query() with a buffer grown as needed.
        int run_query(albireo_engine* e, const json& request, json& answer) const;

    private:
        void* handle_ = nullptr;
        std::string path_;
    };

    // File name of an engine library of the given variant on this platform:
    // "albireo-engine-cpu-avx2.dll", "libalbireo-engine-cpu-avx2.so".
    std::string engine_file_name(const std::string& variant);

}

#endif // ALBIREO_COMMON_ENGINE_LOADER_H_
