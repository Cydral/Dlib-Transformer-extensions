// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "engine.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <dlib/base64.h>
#include <dlib/privacy.h>
#include <dlib/privacy/pii_recognizers.h>

#include "../common/file_kinds.h"
#include "archives.h"

/*
    The models compiled into the engine: the header the embed command of the example
    slm_document_anonymizer_ex writes (examples/slm_document_anonymizer_ex.h), whose path
    the build gives.
*/
#ifndef PSEUDALIS_MODELS_HEADER
#define PSEUDALIS_MODELS_HEADER "slm_document_anonymizer_ex.h"
#endif
#include PSEUDALIS_MODELS_HEADER

namespace pseudalis
{

    namespace
    {
        using namespace dlib;

        std::string lower(std::string s)
        {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        }

        std::string base_name(const std::string& path)
        {
            const size_t slash = path.find_last_of("/\\");
            return slash == std::string::npos ? path : path.substr(slash + 1);
        }

        std::string sha256_hex(const std::string& data)
        {
            const std::array<unsigned char, 32> d = sha256_digest(data);
            return to_hex(d.data(), d.size());
        }

        std::string decode_model(const std::string& text)
        {
            base64 coder;
            std::istringstream in(text);
            std::ostringstream packed;
            coder.decode(in, packed);
            return zlib_decompress(packed.str());
        }

        struct model_set
        {
            std::shared_ptr<const pii_sequence_tagger> tagger;
            std::shared_ptr<const pii_span_recognizer> span;
        };

        // The models compiled into the engine, read once for every engine of the process.
        const model_set& shared_models()
        {
            static const model_set m = []() {
                model_set r;
                const std::string tagger_text = anonymizer_models::tagger();
                if (!tagger_text.empty())
                {
                    auto t = std::make_shared<pii_sequence_tagger>();
                    std::istringstream in(decode_model(tagger_text));
                    deserialize(*t, in);
                    r.tagger = t;
                }
                const std::string span_text = anonymizer_models::span_model();
                if (!span_text.empty())
                {
                    std::istringstream in(decode_model(span_text));
                    r.span = std::make_shared<pii_span_recognizer>(in);
                }
                return r;
            }();
            return m;
        }

        json report_json(const document_report& r)
        {
            json j = json::object();
            json values = json::object();
            for (const auto& e : r.entities) values[e.first] = static_cast<unsigned long long>(e.second);
            j["values"] = values;
            j["total"] = static_cast<unsigned long long>(r.total_entities());
            json removed = json::array();
            for (const std::string& s : r.removed) removed.push_back(s);
            j["removed"] = removed;
            json notes = json::array();
            for (const std::string& s : r.notes) notes.push_back(s);
            j["notes"] = notes;
            j["tags_restored"] = static_cast<unsigned long long>(r.tags_restored);
            j["tags_rejected"] = static_cast<unsigned long long>(r.tags_rejected);
            j["dates_shifted"] = static_cast<unsigned long long>(r.dates_shifted);
            j["dropped"] = json::array();
            return j;
        }

        // Adds the counts and the lists of part, an entry of an archive, to whole.
        void merge_report(json& whole, const json& part, const std::string& entry)
        {
            for (const auto& kv : part["values"].items())
                whole["values"][kv.first] =
                    static_cast<unsigned long long>(whole["values"][kv.first].as_int(0) + kv.second.as_int(0));
            for (const char* key : { "total", "tags_restored", "tags_rejected", "dates_shifted" })
                whole[key] = static_cast<unsigned long long>(whole[key].as_int(0) + part[key].as_int(0));
            for (const char* key : { "removed", "notes", "dropped" })
                for (const json& v : part[key].elements())
                    whole[key].push_back(entry.empty() ? v.str() : entry + ": " + v.str());
        }

        // The name of an entry not taken yet among names: "a.png.txt", then "a.png (2).txt"...
        std::string free_name(const std::string& name, std::set<std::string>& names)
        {
            std::string candidate = name;
            const size_t dot = name.rfind('.');
            for (int k = 2; names.count(lower(candidate)); ++k)
                candidate = dot == std::string::npos
                    ? name + " (" + std::to_string(k) + ")"
                    : name.substr(0, dot) + " (" + std::to_string(k) + ")" + name.substr(dot);
            names.insert(lower(candidate));
            return candidate;
        }
    }

// ----------------------------------------------------------------------------------------

    struct engine::impl
    {
        std::unique_ptr<pseudonymizer> p;
        pii_detector_options dopt;
        model_set m;

        void attach(pii_detector& d) const
        {
            attach_pii_recognizers(d, m.tagger, m.span, 0.5, 0.9);
        }

        // The protected form of a description: one line, without brackets, at most 1000 bytes.
        std::string protect_description(const std::string& description) const
        {
            std::string line;
            for (char c : description)
            {
                if (c == '\r' || c == '\n' || c == '\t') c = ' ';
                if (c == '[') c = '(';
                if (c == ']') c = ')';
                if (c == ' ' && (line.empty() || line.back() == ' ')) continue;
                line.push_back(c);
            }
            while (!line.empty() && line.back() == ' ') line.pop_back();
            if (line.size() > 1000)
            {
                size_t cut = 1000;
                while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) --cut;
                line = line.substr(0, cut) + "...";
            }
            if (line.empty()) return line;
            pii_detector d(dopt);
            attach(d);
            document_anonymizer a(*p, d);
            document_report r;
            return a.anonymize_text(line, r);
        }

        std::string marker_of(const std::string& bytes, const std::string& media_type, const std::string& location,
                              const picture_handler& handler, std::vector<std::string>& ids) const
        {
            picture_found f;
            f.sha256 = sha256_hex(bytes);
            f.id = f.sha256.substr(0, 16);
            f.media_type = media_type;
            f.location = location;
            f.bytes = &bytes;
            if (std::find(ids.begin(), ids.end(), f.id) == ids.end()) ids.push_back(f.id);
            return engine::picture_marker(f.id, protect_description(handler(f)));
        }

        engine_result document(const std::string& bytes, const std::string& name, document_format f,
                               const picture_handler& handler, bool layout, bool restoring) const
        {
            engine_result out;
            pii_detector d(dopt);
            if (!restoring) attach(d);
            document_anonymizer a(*p, d);
            a.set_pdf_layout(layout);
            std::vector<std::string> ids;
            if (handler && !restoring)
                a.set_image_describer([&](const document_image& img) {
                    return marker_of(img.bytes, img.media_type, base_name(name) + ": " + img.location, handler, ids);
                });
            document_report report;
            out.bytes = restoring ? a.restore(bytes, f, report, base_name(name))
                                  : a.anonymize(bytes, f, report, base_name(name));
            out.name = name;
            out.kind = document_format_name(f);
            out.report = report_json(report);
            out.pictures = ids;
            return out;
        }

        engine_result picture(const std::string& bytes, const std::string& name, const picture_handler& handler) const
        {
            engine_result out;
            std::vector<std::string> ids;
            const picture_handler none = [](const picture_found&) { return std::string(); };
            out.bytes = marker_of(bytes, image_media_type(bytes), base_name(name), handler ? handler : none, ids)
                + "\r\n";
            out.name = name + ".txt";
            out.kind = "image";
            out.report = report_json(document_report());
            out.pictures = ids;
            return out;
        }

        engine_result any(const std::string& bytes, const std::string& name, const picture_handler& handler,
                          bool layout, bool restoring, int depth) const
        {
            const file_kind k = engine::kind_of(bytes, name);
            if (k == file_kind::picture)
            {
                if (!restoring) return picture(bytes, name, handler);
                engine_result out;
                out.bytes = bytes;
                out.name = name;
                out.kind = "image";
                out.report = report_json(document_report());
                return out;
            }
            if (k == file_kind::archive) return archive(bytes, name, handler, layout, restoring, depth);
            if (k == file_kind::document)
                return document(bytes, name, detect_document_format(bytes, base_name(name)), handler, layout,
                    restoring);
            throw std::runtime_error(base_name(name) + ": this kind of file is not supported");
        }

        engine_result archive(const std::string& bytes, const std::string& name, const picture_handler& handler,
                              bool layout, bool restoring, int depth) const
        {
            if (depth > 3) throw std::runtime_error(base_name(name) + ": archives nested too deep");
            archive_kind ak = archive_kind_of(bytes);
            if (ak == archive_kind::none) ak = archive_kind_of_name(name);
            std::vector<archive_entry> entries = read_archive(ak, bytes);
            engine_result out;
            out.name = name;
            out.kind = archive_kind_name(ak);
            out.report = report_json(document_report());
            std::set<std::string> names;
            for (const archive_entry& e : entries) names.insert(lower(e.name));
            std::vector<archive_entry> kept;
            for (archive_entry& e : entries)
            {
                if (e.directory)
                {
                    kept.push_back(std::move(e));
                    continue;
                }
                if (!e.regular)
                {
                    out.report["dropped"].push_back(e.name);
                    continue;
                }
                const file_kind k = engine::kind_of(e.data, e.name);
                if (k == file_kind::unsupported)
                {
                    out.report["dropped"].push_back(e.name);
                    continue;
                }
                try
                {
                    engine_result r = any(e.data, e.name, handler, layout, restoring, depth + 1);
                    if (r.name != e.name)
                    {
                        names.erase(lower(e.name));
                        e.name = free_name(r.name, names);
                    }
                    e.data = std::move(r.bytes);
                    merge_report(out.report, r.report, e.name);
                    for (const std::string& id : r.pictures)
                        if (std::find(out.pictures.begin(), out.pictures.end(), id) == out.pictures.end())
                            out.pictures.push_back(id);
                    kept.push_back(std::move(e));
                }
                catch (std::exception& x)
                {
                    // An entry that cannot be read could hold personal data: it is left out.
                    if (restoring)
                    {
                        kept.push_back(std::move(e));
                        continue;
                    }
                    out.report["dropped"].push_back(e.name);
                    out.report["notes"].push_back(e.name + ": " + x.what());
                }
            }
            out.bytes = write_archive(ak, kept);
            return out;
        }
    };

// ----------------------------------------------------------------------------------------

    engine::engine(const std::string& key, const engine_options& options) : p_(std::make_shared<impl>())
    {
        if (key.size() != 32) throw std::runtime_error("the key of the engine must have 32 bytes");
        pseudonymizer_options popt;
        popt.context = options.context.empty() ? "default" : options.context;
        popt.dates = date_policy::shift;
        popt.date_order = options.date_order == "mdy" ? numeric_date_order::month_day_year
                                                      : numeric_date_order::day_month_year;
        p_->p.reset(new pseudonymizer(pseudonymization_key::from_bytes(key), popt));
        p_->dopt.date_order = popt.date_order;
        try
        {
            p_->m = shared_models();
        }
        catch (std::exception& e)
        {
            throw std::runtime_error(std::string("the models of the engine cannot be read: ") + e.what());
        }
    }

    file_kind engine::kind_of_name(const std::string& name)
    {
        switch (pseudalis::kind_of_name(name))
        {
            case name_kind::document: return file_kind::document;
            case name_kind::picture: return file_kind::picture;
            case name_kind::archive: return file_kind::archive;
            default: return file_kind::unsupported;
        }
    }

    file_kind engine::kind_of(const std::string& bytes, const std::string& name)
    {
        if (!image_media_type(bytes).empty()) return file_kind::picture;
        const document_format f = detect_document_format(bytes, base_name(name));
        if (f == document_format::docx || f == document_format::xlsx || f == document_format::pptx)
            return file_kind::document;
        if (archive_kind_of(bytes) != archive_kind::none) return file_kind::archive;
        if (f == document_format::pdf || f == document_format::text || f == document_format::delimited ||
            f == document_format::markup)
            return file_kind::document;
        return file_kind::unsupported;
    }

    std::string engine::picture_marker(const std::string& id, const std::string& description)
    {
        return description.empty() ? "[Image@" + id + "]" : "[Image@" + id + " : " + description + "]";
    }

    engine_result engine::anonymize(const std::string& bytes, const std::string& name, const picture_handler& pictures,
                                    bool pdf_layout) const
    {
        return p_->any(bytes, name, pictures, pdf_layout, false, 0);
    }

    engine_result engine::restore(const std::string& bytes, const std::string& name) const
    {
        return p_->any(bytes, name, picture_handler(), true, true, 0);
    }

    std::string engine::anonymize_text(const std::string& text, json& report) const
    {
        pii_detector d(p_->dopt);
        p_->attach(d);
        document_anonymizer a(*p_->p, d);
        document_report r;
        const std::string out = a.anonymize_text(text, r);
        report = report_json(r);
        return out;
    }

    std::string engine::restore_text(const std::string& text, json& report) const
    {
        pii_detector d(p_->dopt);
        document_anonymizer a(*p_->p, d);
        document_report r;
        const std::string out = a.restore(text, document_format::text, r);
        report = report_json(r);
        return out;
    }

    json engine::detect(const std::string& text) const
    {
        pii_detector d(p_->dopt);
        p_->attach(d);
        d.begin_document();
        d.learn(text);
        json spans = json::array();
        for (const pii_entity& e : d.detect(text))
        {
            json s = json::object();
            s["begin"] = static_cast<unsigned long long>(e.begin);
            s["end"] = static_cast<unsigned long long>(e.end);
            s["type"] = pii_type_keyword(e.type);
            s["confidence"] = e.confidence;
            s["text"] = text.substr(e.begin, e.end - e.begin);
            spans.push_back(s);
        }
        return spans;
    }

    std::string engine::key_fingerprint() const
    {
        return p_->p->key_fingerprint();
    }

    json engine::models()
    {
        json j = json::object();
        j["tagger"] = anonymizer_models::tagger_source();
        j["span_model"] = anonymizer_models::span_model_source();
        j["size"] = anonymizer_models::span_model_size();
        return j;
    }

}
