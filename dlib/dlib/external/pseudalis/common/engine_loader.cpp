// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "engine_loader.h"

#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#include <dlfcn.h>
#endif
#if !defined(_MSC_VER) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

#include "build_stamp.h"
#include "platform.h"
#include "version.h"

namespace pseudalis
{
    namespace
    {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
        void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4])
        {
#if defined(_MSC_VER)
            int regs[4];
            __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(sub));
            for (int i = 0; i < 4; ++i) r[i] = static_cast<uint32_t>(regs[i]);
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        }

        // Read without the intrinsic, which GCC only allows in code compiled for XSAVE.
        uint64_t xgetbv0()
        {
#if defined(_MSC_VER)
            return _xgetbv(0);
#else
            uint32_t eax, edx;
            __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
            return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
        }
#endif

        struct callback_context
        {
            const std::function<std::string(const picture_request&)>* pictures;
            const std::function<std::vector<std::string>(const std::string&)>* lookup;
            const engine_library* lib;
        };

        int PSD_CALL on_lookup(void* context, const char* candidate, size_t size, psd_buffer* tags)
        {
            const callback_context* c = static_cast<const callback_context*>(context);
            tags->data = nullptr;
            tags->size = 0;
            try
            {
                std::string all;
                for (const std::string& t : (*c->lookup)(std::string(candidate, size)))
                    if (!t.empty() && t.find('\n') == std::string::npos) all += (all.empty() ? "" : "\n") + t;
                if (all.empty()) return 0;
                tags->data = c->lib->alloc(all.size());
                if (!tags->data) return 0;
                std::memcpy(tags->data, all.data(), all.size());
                tags->size = all.size();
                return 1;
            }
            catch (...)
            {
                return 0;
            }
        }

        int PSD_CALL on_picture(void* context, const psd_picture* p, psd_buffer* description)
        {
            const callback_context* c = static_cast<const callback_context*>(context);
            description->data = nullptr;
            description->size = 0;
            try
            {
                picture_request r;
                r.bytes.assign(p->bytes, p->size);
                r.id = p->id;
                r.sha256 = p->sha256;
                r.media_type = p->media_type;
                r.location = p->location;
                const std::string text = (*c->pictures)(r);
                if (text.empty()) return 0;
                description->data = c->lib->alloc(text.size());
                if (!description->data) return 0;
                std::memcpy(description->data, text.data(), text.size());
                description->size = text.size();
                return 1;
            }
            catch (...)
            {
                return 0;
            }
        }
    }


    bool cpu_has_avx2()
    {
#if defined(_M_X64) || defined(__x86_64__)
        uint32_t r[4];
        cpuid(0, 0, r);
        if (r[0] < 7) return false;
        cpuid(1, 0, r);
        const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1;
        if (!osxsave || !avx || !fma) return false;
        // The system saves the SSE and AVX registers.
        if ((xgetbv0() & 6) != 6) return false;
        cpuid(7, 0, r);
        return (r[1] >> 5) & 1;
#else
        return false;
#endif
    }

#if defined(_WIN32) && !defined(PSEUDALIS_ENGINE_STATIC)
    namespace
    {
        // The build an engine library holds in its resources, read without running any of its code.
        std::string resource_build(const std::string& path)
        {
            HMODULE m = LoadLibraryExW(widen(path).c_str(), nullptr,
                                       LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
            if (!m) return std::string();
            std::string out;
            HRSRC r = FindResourceW(m, L"PSEUDALIS_BUILD", MAKEINTRESOURCEW(10));
            HGLOBAL g = r ? LoadResource(m, r) : nullptr;
            const char* data = g ? static_cast<const char*>(LockResource(g)) : nullptr;
            const DWORD size = r ? SizeofResource(m, r) : 0;
            if (data && size > 0 && size < 256) out.assign(data, size);
            FreeLibrary(m);
            while (!out.empty() && out.back() == '\0') out.pop_back();
            return out;
        }
    }
#endif

    engine_library::~engine_library()
    {
        // The library is never unloaded (see engine_loader.h): a restart of the service
        // within its process loads it again, which only counts one more reference.
    }

    bool engine_library::load(const std::string& folder, const std::string& force, std::string& error)
    {
#if defined(PSEUDALIS_ENGINE_STATIC)
        (void)folder;
        (void)force;
        (void)error;
        api_version = psd_api_version;
        variant = psd_variant;
        build = psd_build;
        models = psd_models;
        create = psd_create;
        destroy = psd_destroy;
        process = psd_process;
        key_fingerprint = psd_key_fingerprint;
        set_terms = psd_set_terms;
        set_columns = psd_set_columns;
        alloc = psd_alloc;
        free_buffer = psd_free;
        path_ = "(built in)";
        return true;
#else
        const bool avx2 = force == "avx2" || (force != "baseline" && cpu_has_avx2());
#if defined(_WIN32)
        const char* names[] = { "pseudalis-engine-avx2.dll", "pseudalis-engine.dll" };
#else
        const char* names[] = { "libpseudalis-engine-avx2.so", "libpseudalis-engine.so" };
#endif
        for (int k = avx2 ? 0 : 1; k < 2 && !handle_; ++k)
        {
            const std::string path = join_path(folder, names[k]);
            if (!file_exists(path)) continue;
#if defined(_WIN32)
            // The build first, in the resources: a library of another build never runs here.
            const std::string stamp = resource_build(path);
            if (stamp != PSEUDALIS_BUILD_STAMP)
            {
                error = path + " is the engine of another build of Pseudalis (" + (stamp.empty() ? "none" : stamp) +
                        ", this program " + PSEUDALIS_BUILD_STAMP + ")";
                return false;
            }
            handle_ = LoadLibraryExW(widen(path).c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
            handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            if (handle_) path_ = path;
        }
        if (!handle_)
        {
            error = std::string("no engine library in ") + folder;
            return false;
        }
        auto get = [this](const char* name) -> void* {
#if defined(_WIN32)
            return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
            return dlsym(handle_, name);
#endif
        };
#define PSD_RESOLVE(member, name) member = reinterpret_cast<decltype(member)>(get(name))
        PSD_RESOLVE(api_version, "psd_api_version");
        PSD_RESOLVE(variant, "psd_variant");
        PSD_RESOLVE(build, "psd_build");
        PSD_RESOLVE(models, "psd_models");
        PSD_RESOLVE(create, "psd_create");
        PSD_RESOLVE(destroy, "psd_destroy");
        PSD_RESOLVE(process, "psd_process");
        PSD_RESOLVE(key_fingerprint, "psd_key_fingerprint");
        PSD_RESOLVE(set_terms, "psd_set_terms");
        PSD_RESOLVE(set_columns, "psd_set_columns");
        PSD_RESOLVE(alloc, "psd_alloc");
        PSD_RESOLVE(free_buffer, "psd_free");
#undef PSD_RESOLVE
        if (!api_version || !variant || !build || !models || !create || !destroy || !process || !key_fingerprint ||
            !set_terms || !set_columns || !alloc || !free_buffer || api_version() != PSEUDALIS_ENGINE_API_VERSION)
        {
            error = path_ + " is not an engine of this version of Pseudalis";
            create = nullptr;
            return false;
        }
        // The engine of this build alone, which the installer placed beside the program.
        const std::string own = PSEUDALIS_VERSION "+" PSEUDALIS_BUILD_STAMP;
        const char* theirs = build();
        if (!theirs || own != theirs)
        {
            error = path_ + " is the engine of another build of Pseudalis (" + (theirs ? theirs : "none") +
                    ", this program " + own + ")";
            create = nullptr;
            return false;
        }
        return true;
#endif
    }

    engine_client::engine_client(const std::shared_ptr<engine_library>& lib, const std::string& key,
                                 const json& options, std::string& error)
        : lib_(lib)
    {
        if (!lib_ || !lib_->loaded())
        {
            error = "no engine library";
            return;
        }
        psd_buffer err = { nullptr, 0 };
        e_ = lib_->create(key.data(), key.size(), options.dump().c_str(), &err);
        if (!e_) error = err.data ? std::string(err.data, err.size) : "the engine cannot start";
        lib_->free_buffer(&err);
    }

    engine_client::~engine_client()
    {
        if (e_) lib_->destroy(e_);
    }

    bool engine_client::process(const json& request, const std::string& data, json& result, std::string& output,
                                const std::function<std::string(const picture_request&)>& pictures,
                                std::string& error,
                                const std::function<std::vector<std::string>(const std::string&)>& lookup) const
    {
        if (!e_)
        {
            error = "no engine";
            return false;
        }
        callback_context c = { &pictures, &lookup, lib_.get() };
        psd_buffer r = { nullptr, 0 }, o = { nullptr, 0 }, err = { nullptr, 0 };
        const int ok = lib_->process(e_, request.dump().c_str(), data.data(), data.size(),
                                     pictures ? on_picture : nullptr, lookup ? on_lookup : nullptr, &c, &r, &o, &err);
        if (ok == 1)
        {
            result = json::parse_or_null(std::string(r.data ? r.data : "", r.size));
            output.assign(o.data ? o.data : "", o.size);
        }
        else
        {
            error = err.data ? std::string(err.data, err.size) : "the engine failed";
        }
        lib_->free_buffer(&r);
        lib_->free_buffer(&o);
        lib_->free_buffer(&err);
        return ok == 1;
    }

    std::string engine_client::key_fingerprint() const
    {
        if (!e_) return std::string();
        psd_buffer b = { nullptr, 0 };
        std::string out;
        if (lib_->key_fingerprint(e_, &b) == 1 && b.data) out.assign(b.data, b.size);
        lib_->free_buffer(&b);
        return out;
    }

    bool engine_client::set_terms(const json& terms, std::string& error) const
    {
        if (!e_ || !lib_->set_terms)
        {
            error = "no engine";
            return false;
        }
        const std::string text = terms.dump();
        psd_buffer err = { nullptr, 0 };
        const int ok = lib_->set_terms(e_, text.data(), text.size(), &err);
        if (ok != 1) error = err.data ? std::string(err.data, err.size) : std::string("the terms were refused");
        lib_->free_buffer(&err);
        return ok == 1;
    }

    bool engine_client::set_columns(const json& columns, std::string& error) const
    {
        if (!e_ || !lib_->set_columns)
        {
            error = "no engine";
            return false;
        }
        const std::string text = columns.dump();
        psd_buffer err = { nullptr, 0 };
        const int ok = lib_->set_columns(e_, text.data(), text.size(), &err);
        if (ok != 1) error = err.data ? std::string(err.data, err.size) : std::string("the columns were refused");
        lib_->free_buffer(&err);
        return ok == 1;
    }

    std::string engine_client::variant() const
    {
        return lib_ && lib_->variant ? lib_->variant() : std::string();
    }

    std::string engine_client::build() const
    {
        const char* b = lib_ && lib_->build ? lib_->build() : nullptr;
        return b ? b : std::string();
    }

    json engine_client::models() const
    {
        psd_buffer b = { nullptr, 0 };
        json j;
        if (lib_ && lib_->models && lib_->models(&b) == 1 && b.data)
            j = json::parse_or_null(std::string(b.data, b.size));
        if (lib_ && b.data) lib_->free_buffer(&b);
        return j;
    }
}
