// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

/*
    The C interface of common/engine_api.h over albireo::engine.  No exception crosses
    it: each one becomes a return code and a message.
*/

#ifndef ALBIREO_ENGINE_BUILD
#define ALBIREO_ENGINE_BUILD
#endif

#include "../common/engine_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#ifdef ALBIREO_TENSOR_BACKEND
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
#endif

#include "engine.h"

struct albireo_engine
{
    std::unique_ptr<albireo::engine> impl;
};

namespace
{
    void copy_message(const std::string& message, char* buffer, size_t capacity)
    {
        if (!buffer || capacity == 0) return;
        // Truncated on a character boundary, so that the message stays valid UTF-8.
        size_t n = std::min(message.size(), capacity - 1);
        while (n > 0 && n < message.size() && (static_cast<unsigned char>(message[n]) & 0xC0) == 0x80) --n;
        std::memcpy(buffer, message.data(), n);
        buffer[n] = '\0';
    }

    int write_json(const std::string& text, char* json, size_t capacity, size_t* needed)
    {
        if (needed) *needed = text.size() + 1;
        if (!json || capacity < text.size() + 1) return ALBIREO_E_BUFFER_TOO_SMALL;
        std::memcpy(json, text.data(), text.size());
        json[text.size()] = '\0';
        return ALBIREO_OK;
    }

    int code_of(const std::exception& e)
    {
        if (dynamic_cast<const std::invalid_argument*>(&e)) return ALBIREO_E_INVALID_ARGUMENT;
        if (dynamic_cast<const std::bad_alloc*>(&e)) return ALBIREO_E_OUT_OF_MEMORY;
        return ALBIREO_E_INTERNAL;
    }

#ifdef ALBIREO_TENSOR_BACKEND
    /*
        Keeps this library in the process until it ends, whatever the program releases.
        The runtimes of the devices start threads and register handlers that run at the end
        of the process, which must find the code of the library where it was.
    */
    void pin_library()
    {
#if defined(_WIN32)
        HMODULE self = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            reinterpret_cast<LPCWSTR>(&pin_library), &self);
#else
        Dl_info info;
        if (dladdr(reinterpret_cast<void*>(&pin_library), &info) && info.dli_fname)
            dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
#endif
    }
#endif

    albireo::json parse_or_throw(const char* text)
    {
        if (!text || !*text) return albireo::json::object();
        albireo::json j;
        std::string error;
        if (!albireo::json::parse(text, j, &error)) throw std::invalid_argument("invalid JSON: " + error);
        return j;
    }
}

extern "C"
{

ALBIREO_ENGINE_EXPORT uint32_t ALBIREO_ENGINE_CALL albireo_engine_abi_version(void)
{
    return ALBIREO_ENGINE_ABI_VERSION;
}

ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_describe(char* json, size_t capacity, size_t* needed)
{
    try
    {
        return write_json(albireo::engine::describe().dump(), json, capacity, needed);
    }
    catch (...)
    {
        return ALBIREO_E_INTERNAL;
    }
}

ALBIREO_ENGINE_EXPORT albireo_engine* ALBIREO_ENGINE_CALL albireo_engine_create(const char* config_json,
    albireo_log_callback log, void* log_user, char* error, size_t error_capacity)
{
    try
    {
#ifdef ALBIREO_TENSOR_BACKEND
        pin_library();
#endif
        std::unique_ptr<albireo_engine> e(new albireo_engine());
        albireo::log_sink sink;
        if (log)
            sink = [log, log_user](int level, const std::string& message) { log(log_user, level, message.c_str()); };
        e->impl.reset(new albireo::engine(parse_or_throw(config_json), sink));
        return e.release();
    }
    catch (const std::exception& ex)
    {
        copy_message(ex.what(), error, error_capacity);
    }
    catch (...)
    {
        copy_message("unknown error", error, error_capacity);
    }
    return nullptr;
}

ALBIREO_ENGINE_EXPORT void ALBIREO_ENGINE_CALL albireo_engine_destroy(albireo_engine* engine)
{
    try
    {
        delete engine;
    }
    catch (...)
    {
    }
}

ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_load_model(albireo_engine* engine,
    const char* model_id, const char* path, const char* options_json, char* error, size_t error_capacity)
{
    if (!engine || !model_id || !path) return ALBIREO_E_INVALID_ARGUMENT;
    try
    {
        engine->impl->load_model(model_id, path, parse_or_throw(options_json));
        return ALBIREO_OK;
    }
    catch (const std::exception& ex)
    {
        copy_message(ex.what(), error, error_capacity);
        return code_of(ex);
    }
    catch (...)
    {
        copy_message("unknown error", error, error_capacity);
        return ALBIREO_E_INTERNAL;
    }
}

ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_unload_model(albireo_engine* engine, const char* model_id)
{
    if (!engine || !model_id) return ALBIREO_E_INVALID_ARGUMENT;
    try
    {
        engine->impl->unload_model(model_id);
        return ALBIREO_OK;
    }
    catch (...)
    {
        return ALBIREO_E_INTERNAL;
    }
}

ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_query(albireo_engine* engine,
    const char* request_json, char* json, size_t capacity, size_t* needed)
{
    if (!engine) return ALBIREO_E_INVALID_ARGUMENT;
    try
    {
        std::string text;
        int code = ALBIREO_OK;
        try
        {
            text = engine->impl->query(parse_or_throw(request_json)).dump();
        }
        catch (const std::exception& ex)
        {
            code = code_of(ex);
            albireo::json e = albireo::json::object();
            e["error"] = ex.what();
            text = e.dump();
        }
        const int w = write_json(text, json, capacity, needed);
        return w != ALBIREO_OK ? w : code;
    }
    catch (...)
    {
        // Out of memory while the answer or the error was written.
        return ALBIREO_E_INTERNAL;
    }
}

ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_submit(albireo_engine* engine,
    const char* request_json, albireo_event_callback callback, void* user, uint64_t* request_id,
    char* error, size_t error_capacity)
{
    if (!engine || !callback) return ALBIREO_E_INVALID_ARGUMENT;
    try
    {
        const uint64_t id = engine->impl->submit(parse_or_throw(request_json),
            [callback, user](int kind, const std::string& data) { callback(user, kind, data.data(), data.size()); });
        if (request_id) *request_id = id;
        return ALBIREO_OK;
    }
    catch (const std::exception& ex)
    {
        copy_message(ex.what(), error, error_capacity);
        const std::string what = ex.what();
        if (what.find("beyond the context") != std::string::npos) return ALBIREO_E_CONTEXT_OVERFLOW;
        return code_of(ex);
    }
    catch (...)
    {
        copy_message("unknown error", error, error_capacity);
        return ALBIREO_E_INTERNAL;
    }
}

ALBIREO_ENGINE_EXPORT int ALBIREO_ENGINE_CALL albireo_engine_cancel(albireo_engine* engine, uint64_t request_id)
{
    if (!engine) return ALBIREO_E_INVALID_ARGUMENT;
    try
    {
        engine->impl->cancel(request_id);
        return ALBIREO_OK;
    }
    catch (...)
    {
        return ALBIREO_E_INTERNAL;
    }
}

}
