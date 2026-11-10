// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The C interface of the engine libraries (common/engine_api.h).

#ifndef PSEUDALIS_ENGINE_BUILD
#define PSEUDALIS_ENGINE_BUILD
#endif
#include "../common/engine_api.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "build_stamp.h"
#include "../common/version.h"
#include "engine.h"

struct psd_engine
{
    std::unique_ptr<pseudalis::engine> e;
};

namespace
{
    bool put(psd_buffer* b, const std::string& s)
    {
        if (!b) return true;
        b->data = nullptr;
        b->size = 0;
        b->data = static_cast<char*>(std::malloc(s.size() + 1));
        if (!b->data) return false;
        std::memcpy(b->data, s.data(), s.size());
        b->data[s.size()] = '\0';
        b->size = s.size();
        return true;
    }

    void clear(psd_buffer* b)
    {
        if (b)
        {
            b->data = nullptr;
            b->size = 0;
        }
    }

    pseudalis::json substitutions_of(const std::vector<pseudalis::substitution>& made)
    {
        pseudalis::json list = pseudalis::json::array();
        for (const pseudalis::substitution& s : made)
        {
            pseudalis::json one = pseudalis::json::object();
            one["substitute"] = s.substitute;
            one["tag"] = s.tag;
            if (s.text) one["text"] = true;
            list.push_back(one);
        }
        return list;
    }

    // The substitutes of the text a request gives: [{"substitute", "tag"}].
    std::vector<pseudalis::substitution> substitutions_in(const pseudalis::json& list)
    {
        std::vector<pseudalis::substitution> out;
        for (const pseudalis::json& one : list.elements())
        {
            pseudalis::substitution s;
            s.substitute = one["substitute"].str();
            s.tag = one["tag"].str();
            s.text = true;
            if (!s.substitute.empty() && !s.tag.empty()) out.push_back(s);
        }
        return out;
    }

    pseudalis::json result_of(const pseudalis::engine_result& r)
    {
        pseudalis::json j = pseudalis::json::object();
        j["name"] = r.name;
        j["kind"] = r.kind;
        j["report"] = r.report;
        pseudalis::json ids = pseudalis::json::array();
        for (const std::string& id : r.pictures) ids.push_back(id);
        j["pictures"] = ids;
        j["substitutions"] = substitutions_of(r.substitutions);
        if (!r.entries.empty())
        {
            pseudalis::json entries = pseudalis::json::array();
            for (const pseudalis::entry_written& w : r.entries)
            {
                pseudalis::json e = pseudalis::json::object();
                e["original"] = w.original;
                e["name"] = w.name;
                e["sha256"] = w.sha256;
                e["kind"] = w.kind;
                entries.push_back(e);
            }
            j["entries"] = entries;
        }
        return j;
    }
}

extern "C" {

PSD_API int PSD_CALL psd_api_version(void)
{
    return PSEUDALIS_ENGINE_API_VERSION;
}

PSD_API const char* PSD_CALL psd_variant(void)
{
#if defined(__AVX2__)
    return "avx2";
#else
    return "baseline";
#endif
}

PSD_API const char* PSD_CALL psd_build(void)
{
    return PSEUDALIS_VERSION "+" PSEUDALIS_BUILD_STAMP;
}

PSD_API int PSD_CALL psd_models(psd_buffer* description)
{
    clear(description);
    try
    {
        return put(description, pseudalis::engine::models().dump()) ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

PSD_API psd_engine* PSD_CALL psd_create(const char* key, size_t key_size, const char* options, psd_buffer* error)
{
    clear(error);
    try
    {
        pseudalis::engine_options o;
        pseudalis::json j;
        if (options && *options && !pseudalis::json::parse(options, j))
        {
            put(error, "the options of the engine are not JSON");
            return nullptr;
        }
        o.date_order = j["date_order"].str("dmy");
        o.context = j["context"].str("default");
        std::unique_ptr<psd_engine> e(new psd_engine);
        e->e.reset(new pseudalis::engine(std::string(key ? key : "", key ? key_size : 0), o));
        return e.release();
    }
    catch (const std::exception& x)
    {
        put(error, x.what());
    }
    catch (...)
    {
        put(error, "unexpected error");
    }
    return nullptr;
}

PSD_API void PSD_CALL psd_destroy(psd_engine* engine)
{
    delete engine;
}

PSD_API int PSD_CALL psd_process(psd_engine* engine, const char* request, const char* data, size_t size,
    psd_picture_fn picture, psd_lookup_fn lookup, void* context, psd_buffer* result, psd_buffer* output,
    psd_buffer* error)
{
    clear(result);
    clear(output);
    clear(error);
    if (!engine || !engine->e)
    {
        put(error, "no engine");
        return 0;
    }
    try
    {
        pseudalis::json req;
        if (!request || !pseudalis::json::parse(request, req) || !req.is_object())
        {
            put(error, "the request is not a JSON object");
            return 0;
        }
        const std::string op = req["operation"].str();
        const std::string name = req["name"].str("document");
        const std::string bytes(data ? data : "", data ? size : 0);
        pseudalis::picture_handler handler;
        if (picture)
            handler = [picture, context](const pseudalis::picture_found& f) {
                psd_picture p;
                p.bytes = f.bytes->data();
                p.size = f.bytes->size();
                p.id = f.id.c_str();
                p.sha256 = f.sha256.c_str();
                p.media_type = f.media_type.c_str();
                p.location = f.location.c_str();
                psd_buffer d = { nullptr, 0 };
                std::string text;
                if (picture(context, &p, &d) == 1 && d.data) text.assign(d.data, d.size);
                psd_free(&d);
                return text;
            };
        pseudalis::substitute_lookup tags_of;
        if (lookup)
            tags_of = [lookup, context](const std::string& candidate) {
                std::vector<std::string> tags;
                psd_buffer b = { nullptr, 0 };
                if (lookup(context, candidate.data(), candidate.size(), &b) == 1 && b.data)
                {
                    const std::string all(b.data, b.size);
                    std::size_t start = 0;
                    while (start < all.size())
                    {
                        std::size_t end = all.find('\n', start);
                        if (end == std::string::npos) end = all.size();
                        if (end > start) tags.push_back(all.substr(start, end - start));
                        start = end + 1;
                    }
                }
                psd_free(&b);
                return tags;
            };
        pseudalis::json res = pseudalis::json::object();
        std::string out;
        // Without tags, the values of the text take substitutes; the substitutes of the text known.
        const bool tags = req["tags"].as_bool(true);
        const std::vector<pseudalis::substitution> known = substitutions_in(req["substitutions"]);
        if (op == "anonymize" || op == "restore")
        {
            pseudalis::engine_result r = op == "anonymize"
                ? engine->e->anonymize(bytes, name, handler, tags_of, tags, known)
                : engine->e->restore(bytes, name, tags_of, known, req["ascii_encoding"].str() == "windows-1252");
            res = result_of(r);
            out = std::move(r.bytes);
        }
        else if (op == "anonymize_name" || op == "restore_name")
        {
            pseudalis::json report;
            std::vector<pseudalis::substitution> made;
            out = op == "anonymize_name" ? engine->e->anonymize_name(bytes, tags_of, made, report)
                                         : engine->e->restore_name(bytes, tags_of, report);
            res["name"] = out;
            res["kind"] = "name";
            res["report"] = report;
            res["pictures"] = pseudalis::json::array();
            res["substitutions"] = substitutions_of(made);
        }
        else if (op == "anonymize_text" || op == "restore_text")
        {
            pseudalis::json report;
            std::vector<pseudalis::substitution> made;
            out = op == "anonymize_text" ? engine->e->anonymize_text(bytes, report, tags, &made)
                                         : engine->e->restore_text(bytes, report, known);
            res["name"] = name;
            res["kind"] = "text";
            res["report"] = report;
            res["pictures"] = pseudalis::json::array();
            // The places of the tags or substitutes written, or of the values restored, for the colouring.
            if (op == "anonymize_text")
            {
                res["tags"] = engine->e->tag_spans(out, made);
                if (!tags) res["substitutions"] = substitutions_of(made);
            }
            else
            {
                res["restored"] = engine->e->restored_spans(bytes, out, known);
            }
        }
        else if (op == "detect")
        {
            res["spans"] = engine->e->detect(bytes);
        }
        else if (op == "extract")
        {
            const std::string entry = req["entry"].str();
            out = engine->e->extract(bytes, name, entry);
            const std::size_t cut = entry.find_last_of("!/");
            res["name"] = cut == std::string::npos ? entry : entry.substr(cut + 1);
            res["kind"] = "entry";
            res["report"] = pseudalis::json::object();
            res["pictures"] = pseudalis::json::array();
        }
        else
        {
            put(error, "unknown operation: " + op);
            return 0;
        }
        if (!put(result, res.dump()) || !put(output, out))
        {
            psd_free(result);
            psd_free(output);
            put(error, "out of memory");
            return 0;
        }
        return 1;
    }
    catch (const std::bad_alloc&)
    {
        put(error, "out of memory");
    }
    catch (const std::exception& x)
    {
        put(error, x.what());
    }
    catch (...)
    {
        put(error, "unexpected error");
    }
    return 0;
}

PSD_API int PSD_CALL psd_key_fingerprint(psd_engine* engine, psd_buffer* fingerprint)
{
    clear(fingerprint);
    if (!engine || !engine->e) return 0;
    // No exception crosses the interface of the library.
    try
    {
        return put(fingerprint, engine->e->key_fingerprint()) ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

PSD_API int PSD_CALL psd_set_terms(psd_engine* engine, const char* terms, size_t size, psd_buffer* error)
{
    clear(error);
    if (!engine || !engine->e)
    {
        put(error, "no engine");
        return 0;
    }
    try
    {
        pseudalis::json list;
        if (!pseudalis::json::parse(std::string(terms ? terms : "", terms ? size : 0), list))
        {
            put(error, "the terms are not JSON");
            return 0;
        }
        std::vector<pseudalis::imposed_term> parsed;
        std::string message, code;
        long long index = -1;
        if (!pseudalis::parse_imposed_terms(list, parsed, message, code, index))
        {
            put(error, message);
            return 0;
        }
        engine->e->set_terms(parsed);
        return 1;
    }
    catch (const std::exception& x)
    {
        put(error, x.what());
    }
    catch (...)
    {
        put(error, "unexpected error");
    }
    return 0;
}

PSD_API int PSD_CALL psd_set_columns(psd_engine* engine, const char* columns, size_t size, psd_buffer* error)
{
    clear(error);
    if (!engine || !engine->e)
    {
        put(error, "no engine");
        return 0;
    }
    try
    {
        pseudalis::json list;
        if (!pseudalis::json::parse(std::string(columns ? columns : "", columns ? size : 0), list))
        {
            put(error, "the columns are not JSON");
            return 0;
        }
        std::vector<pseudalis::imposed_column> parsed;
        std::string message, code;
        long long index = -1;
        if (!pseudalis::parse_imposed_columns(list, parsed, message, code, index))
        {
            put(error, message);
            return 0;
        }
        engine->e->set_columns(parsed);
        return 1;
    }
    catch (const std::exception& x)
    {
        put(error, x.what());
    }
    catch (...)
    {
        put(error, "unexpected error");
    }
    return 0;
}

PSD_API char* PSD_CALL psd_alloc(size_t size)
{
    return static_cast<char*>(std::malloc(size + 1));
}

PSD_API void PSD_CALL psd_free(psd_buffer* buffer)
{
    if (!buffer) return;
    std::free(buffer->data);
    buffer->data = nullptr;
    buffer->size = 0;
}

}
