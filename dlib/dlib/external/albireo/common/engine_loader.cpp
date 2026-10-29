// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "engine_loader.h"

#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace albireo
{

    namespace
    {
#if defined(_WIN32)
        std::wstring widen(const std::string& s)
        {
            if (s.empty()) return std::wstring();
            const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
            std::wstring w(static_cast<size_t>(n), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], n);
            return w;
        }
#endif

        void* symbol(void* handle, const char* name)
        {
#if defined(_WIN32)
            return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
            return dlsym(handle, name);
#endif
        }

        template <typename F>
        bool resolve(void* handle, const char* name, F& f)
        {
            f = reinterpret_cast<F>(symbol(handle, name));
            return f != nullptr;
        }
    }

    engine_library::~engine_library()
    {
        unload();
    }

    bool engine_library::load(const std::string& path, std::string& error)
    {
        unload();
#if defined(_WIN32)
        // The dependencies of the engine are looked up next to it first, a search that needs
        // the full path of the library.
        std::wstring file = widen(path);
        const DWORD size = GetFullPathNameW(file.c_str(), 0, nullptr, nullptr);
        if (size > 0)
        {
            std::wstring full(size, L'\0');
            const DWORD length = GetFullPathNameW(file.c_str(), size, &full[0], nullptr);
            if (length > 0 && length < size)
            {
                full.resize(length);
                file = full;
            }
        }
        HMODULE h = LoadLibraryExW(file.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!h)
        {
            error = "cannot load " + path + " (error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        handle_ = h;
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle_)
        {
            const char* e = dlerror();
            error = "cannot load " + path + ": " + (e ? e : "unknown error");
            return false;
        }
#endif
        path_ = path;
        const bool ok = resolve(handle_, "albireo_engine_abi_version", abi_version)
            && resolve(handle_, "albireo_engine_describe", describe)
            && resolve(handle_, "albireo_engine_create", create)
            && resolve(handle_, "albireo_engine_destroy", destroy)
            && resolve(handle_, "albireo_engine_load_model", load_model)
            && resolve(handle_, "albireo_engine_unload_model", unload_model)
            && resolve(handle_, "albireo_engine_query", query)
            && resolve(handle_, "albireo_engine_submit", submit)
            && resolve(handle_, "albireo_engine_cancel", cancel);
        if (!ok)
        {
            error = path + " is not an Albireo engine (missing functions)";
            unload();
            return false;
        }
        if (abi_version() != ALBIREO_ENGINE_ABI_VERSION)
        {
            error = path + " implements the engine interface " + std::to_string(abi_version()) + ", this program needs "
                + std::to_string(ALBIREO_ENGINE_ABI_VERSION);
            unload();
            return false;
        }
        return true;
    }

    void engine_library::unload()
    {
        if (!handle_) return;
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
        handle_ = nullptr;
        path_.clear();
        abi_version = nullptr;
        describe = nullptr;
        create = nullptr;
        destroy = nullptr;
        load_model = nullptr;
        unload_model = nullptr;
        query = nullptr;
        submit = nullptr;
        cancel = nullptr;
    }

    json engine_library::description() const
    {
        if (!describe) return json();
        size_t needed = 0;
        std::vector<char> buf(4096);
        int rc = describe(buf.data(), buf.size(), &needed);
        if (rc == ALBIREO_E_BUFFER_TOO_SMALL)
        {
            buf.resize(needed);
            rc = describe(buf.data(), buf.size(), &needed);
        }
        if (rc != ALBIREO_OK) return json();
        return json::parse_or_null(buf.data());
    }

    int engine_library::run_query(albireo_engine* e, const json& request, json& answer) const
    {
        const std::string text = request.dump();
        std::vector<char> buf(16384);
        size_t needed = 0;
        int rc = query(e, text.c_str(), buf.data(), buf.size(), &needed);
        if (rc == ALBIREO_E_BUFFER_TOO_SMALL)
        {
            buf.resize(needed);
            rc = query(e, text.c_str(), buf.data(), buf.size(), &needed);
        }
        answer = json::parse_or_null(buf.data());
        return rc;
    }

    std::string engine_file_name(const std::string& variant)
    {
#if defined(_WIN32)
        return "albireo-engine-" + variant + ".dll";
#elif defined(__APPLE__)
        return "libalbireo-engine-" + variant + ".dylib";
#else
        return "libalbireo-engine-" + variant + ".so";
#endif
    }

}
