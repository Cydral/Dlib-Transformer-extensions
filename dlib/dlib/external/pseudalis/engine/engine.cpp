// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "engine.h"

#include <algorithm>
#include <cctype>
#include <cstring>
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

        /*
            A name and its extension, ".docx" or ".tar.gz", which keeps its letters: the extension
            of a kind Pseudalis processes, since the end of a folder or of an e-mail address
            ("jean.dupont@example.fr") is no extension.
        */
        void split_extension(const std::string& base, std::string& stem, std::string& ext)
        {
            ext = pseudalis::kind_of_name(base) == name_kind::unsupported ? std::string() : name_extension(base);
            if (ext.size() >= base.size()) ext.clear();
            ext = base.substr(base.size() - ext.size());
            stem = base.substr(0, base.size() - ext.size());
        }

        // A substitute with the characters Windows refuses in a name replaced by '_'.
        std::string windows_safe(std::string s)
        {
            for (char& c : s)
                if (static_cast<unsigned char>(c) < 0x20 || std::strchr("<>:\"/\\|?*", c) != nullptr) c = '_';
            return s;
        }

        // The number of UTF-16 units of a UTF-8 text, which bounds a name of Windows.
        std::size_t utf16_units(const std::string& s)
        {
            std::size_t n = 0;
            for (std::size_t i = 0; i < s.size(); ++i)
            {
                const unsigned char c = static_cast<unsigned char>(s[i]);
                if ((c & 0xC0) != 0x80) n += c >= 0xF0 ? 2 : 1;
            }
            return n;
        }

        /*
            A protected name that Windows takes: a stem that names a device (CON, NUL, COM1...)
            gets '_', and a name past 255 UTF-16 units loses the end of its stem, its extension
            kept.  The original name of a file, which the caller records, gives it back whole.
        */
        std::string windows_name(std::string stem, const std::string& ext)
        {
            std::string head = lower(stem.substr(0, stem.find('.')));
            while (!head.empty() && head.back() == ' ') head.pop_back();
            const bool device = head == "con" || head == "prn" || head == "aux" || head == "nul" ||
                (head.size() == 4 && (head.compare(0, 3, "com") == 0 || head.compare(0, 3, "lpt") == 0) &&
                 head[3] >= '1' && head[3] <= '9');
            if (device) stem.insert(stem.find('.') == std::string::npos ? stem.size() : stem.find('.'), "_");
            while (!stem.empty() && utf16_units(stem + ext) > 255)
            {
                stem.pop_back();
                while (!stem.empty() && (static_cast<unsigned char>(stem.back()) & 0xC0) == 0x80) stem.pop_back();
                if (!stem.empty() && static_cast<unsigned char>(stem.back()) >= 0xC0) stem.pop_back();
            }
            while (!stem.empty() && (stem.back() == ' ' || stem.back() == '.')) stem.pop_back();
            return stem.empty() ? "_" + ext : stem + ext;
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
            j["substitutes"] = static_cast<unsigned long long>(r.substitutions.size());
            j["substitutes_restored"] = static_cast<unsigned long long>(r.substitutes_restored);
            j["dropped"] = json::array();
            return j;
        }

        // Adds the counts and the lists of part, an entry of an archive, to whole.
        void merge_report(json& whole, const json& part, const std::string& entry)
        {
            for (const auto& kv : part["values"].items())
                whole["values"][kv.first] =
                    static_cast<unsigned long long>(whole["values"][kv.first].as_int(0) + kv.second.as_int(0));
            for (const char* key : { "total", "tags_restored", "tags_rejected", "dates_shifted", "substitutes",
                                     "substitutes_restored" })
                whole[key] = static_cast<unsigned long long>(whole[key].as_int(0) + part[key].as_int(0));
            for (const char* key : { "removed", "notes", "dropped" })
                for (const json& v : part[key].elements())
                    whole[key].push_back(entry.empty() ? v.str() : entry + ": " + v.str());
        }

        /*
            The name of an entry not taken yet among names: "a.png.txt", then "a.png (2).txt"...
            next remembers, for each name, the number to try first, so that an archive that
            repeats a name thousands of times is not walked again for each of them.
        */
        std::string free_name(const std::string& name, std::set<std::string>& names, std::map<std::string, int>& next)
        {
            std::string candidate = name;
            const size_t dot = name.rfind('.');
            int& k = next[lower(name)];
            if (k < 2) k = 2;
            for (; names.count(lower(candidate)); ++k)
                candidate = dot == std::string::npos
                    ? name + " (" + std::to_string(k) + ")"
                    : name.substr(0, dot) + " (" + std::to_string(k) + ")" + name.substr(dot);
            names.insert(lower(candidate));
            return candidate;
        }

        /*
            The bytes that the archives and the Office documents of one file may decompress, all
            levels together: 64 times the file, from 16 MB to 2 GB.  The memory and the time of
            the processing grow with them: a small file that holds hundreds of megabytes of text
            is refused rather than read for hours.
        */
        uint64_t decompression_budget(size_t file_size)
        {
            const uint64_t wanted = static_cast<uint64_t>(file_size) * 64;
            return std::min<uint64_t>(2048ull << 20, std::max<uint64_t>(16ull << 20, wanted));
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

        /*
            The base name of a file, its values replaced by substitutes that Windows takes, its
            extension kept: a, with the detector of the document it names, reads it.
        */
        std::string protect_base(document_anonymizer& a, document_report& r, const std::string& base,
                                 std::vector<substitution>& made) const
        {
            std::string stem, ext;
            split_extension(base, stem, ext);
            const std::size_t first = r.substitutions.size();
            std::string out = a.anonymize_name(stem, r);
            for (std::size_t k = first; k < r.substitutions.size(); ++k)
            {
                document_substitution& s = r.substitutions[k];
                const std::string safe = windows_safe(s.substitute);
                if (safe != s.substitute)
                {
                    const std::size_t at = out.find(s.substitute);
                    if (at != std::string::npos) out.replace(at, s.substitute.size(), safe);
                    s.substitute = safe;
                }
                made.push_back(substitution{ s.substitute, s.tag });
            }
            return out == stem ? base : windows_name(out, ext);
        }

        std::string restore_base(document_anonymizer& a, document_report& r, const std::string& base) const
        {
            std::string stem, ext;
            split_extension(base, stem, ext);
            return a.restore_name(stem, r) + ext;
        }

        // The resolver of the document anonymizer, from the lookup of the caller.
        static pseudonymizer::substitute_resolver resolver_of(const substitute_lookup& lookup)
        {
            if (!lookup) return pseudonymizer::substitute_resolver();
            return [lookup](const std::string& candidate) { return lookup(candidate); };
        }

        // The protected name of a file read alone: a picture, an archive, a folder.
        std::string protect_alone(const std::string& base, const substitute_lookup& lookup,
                                  std::vector<substitution>& made, document_report& r) const
        {
            pii_detector d(dopt);
            attach(d);
            document_anonymizer a(*p, d);
            a.set_substitute_resolver(resolver_of(lookup));
            return protect_base(a, r, base, made);
        }

        std::string restore_alone(const std::string& base, const substitute_lookup& lookup, document_report& r) const
        {
            pii_detector d(dopt);
            document_anonymizer a(*p, d);
            a.set_substitute_resolver(resolver_of(lookup));
            return restore_base(a, r, base);
        }

        static std::string folder_of(const std::string& name)
        {
            return name.substr(0, name.size() - base_name(name).size());
        }

        engine_result document(const std::string& bytes, const std::string& name, document_format f,
                               const picture_handler& handler, bool restoring, const substitute_lookup& lookup,
                               uint64_t& budget) const
        {
            engine_result out;
            pii_detector d(dopt);
            if (!restoring) attach(d);
            // A PDF keeps its pages: its text is replaced in place (the default of the anonymizer).
            document_anonymizer a(*p, d);
            // The properties of the document take substitutes, which a window of properties shows as they are.
            a.set_untagged_properties(true);
            a.set_substitute_resolver(resolver_of(lookup));
            /*
                A part of an Office document takes about 26 bytes of memory for each of its bytes
                while it is read: each part stays within 32 times the document, from 32 MB.
            */
            zip_limits limits;
            limits.max_total_size = std::min<uint64_t>(limits.max_total_size, budget);
            limits.max_entry_size = std::min<uint64_t>(std::min<uint64_t>(limits.max_entry_size, budget),
                std::max<uint64_t>(32ull << 20, static_cast<uint64_t>(bytes.size()) * 32));
            a.set_package_limits(limits);
            std::vector<std::string> ids;
            if (handler && !restoring)
                a.set_image_describer([&](const document_image& img) {
                    return marker_of(img.bytes, img.media_type, base_name(name) + ": " + img.location, handler, ids);
                });
            document_report report;
            out.bytes = restoring ? a.restore(bytes, f, report, base_name(name))
                                  : a.anonymize(bytes, f, report, base_name(name));
            // The name is read with what the document taught the detector.
            std::vector<substitution> named;
            out.name = folder_of(name) + (restoring ? restore_base(a, report, base_name(name))
                                                    : protect_base(a, report, base_name(name), named));
            out.kind = document_format_name(f);
            out.report = report_json(report);
            out.pictures = ids;
            // The substitutes of the properties and of the name, as written.
            for (const document_substitution& s : report.substitutions)
                out.substitutions.push_back(substitution{ s.substitute, s.tag });
            return out;
        }

        engine_result picture(const std::string& bytes, const std::string& name, const picture_handler& handler,
                              const substitute_lookup& lookup) const
        {
            engine_result out;
            std::vector<std::string> ids;
            const picture_handler none = [](const picture_found&) { return std::string(); };
            out.bytes = marker_of(bytes, image_media_type(bytes), base_name(name), handler ? handler : none, ids)
                + "\r\n";
            document_report report;
            out.name = folder_of(name) + protect_alone(base_name(name), lookup, out.substitutions, report) + ".txt";
            out.kind = "image";
            out.report = report_json(report);
            out.pictures = ids;
            return out;
        }

        engine_result any(const std::string& bytes, const std::string& name, const picture_handler& handler,
                          bool restoring, const substitute_lookup& lookup, int depth, uint64_t& budget) const
        {
            const file_kind k = engine::kind_of(bytes, name);
            if (k == file_kind::picture)
            {
                if (!restoring) return picture(bytes, name, handler, lookup);
                engine_result out;
                document_report report;
                out.bytes = bytes;
                out.name = folder_of(name) + restore_alone(base_name(name), lookup, report);
                out.kind = "image";
                out.report = report_json(report);
                return out;
            }
            if (k == file_kind::archive) return archive(bytes, name, handler, restoring, lookup, depth, budget);
            if (k == file_kind::document)
                return document(bytes, name, detect_document_format(bytes, base_name(name)), handler, restoring,
                                lookup, budget);
            throw std::runtime_error(base_name(name) + ": this kind of file is not supported");
        }

        /*
            The folders of the path of an entry, each one protected, or restored, once for the
            whole archive: "Clients/Jean Dupont/" gives the same folder to all its entries.
        */
        std::string entry_folders(const std::string& folders, bool restoring, const substitute_lookup& lookup,
                                  std::map<std::string, std::string>& done, engine_result& out) const
        {
            std::string result;
            std::size_t start = 0;
            while (start < folders.size())
            {
                std::size_t slash = folders.find('/', start);
                if (slash == std::string::npos) slash = folders.size();
                const std::string part = folders.substr(start, slash - start);
                if (!part.empty() && part != "." && part != "..")
                {
                    auto it = done.find(part);
                    if (it == done.end())
                    {
                        document_report r;
                        std::string changed;
                        if (restoring)
                        {
                            pii_detector d(dopt);
                            document_anonymizer a(*p, d);
                            a.set_substitute_resolver(resolver_of(lookup));
                            changed = a.restore_name(part, r);
                        }
                        else
                        {
                            pii_detector d(dopt);
                            attach(d);
                            document_anonymizer a(*p, d);
                            a.set_substitute_resolver(resolver_of(lookup));
                            // A folder has no extension: its whole name is read.
                            const std::size_t first = r.substitutions.size();
                            changed = a.anonymize_name(part, r);
                            for (std::size_t k = first; k < r.substitutions.size(); ++k)
                            {
                                document_substitution& s = r.substitutions[k];
                                const std::string safe = windows_safe(s.substitute);
                                if (safe != s.substitute)
                                {
                                    const std::size_t at = changed.find(s.substitute);
                                    if (at != std::string::npos) changed.replace(at, s.substitute.size(), safe);
                                    s.substitute = safe;
                                }
                                out.substitutions.push_back(substitution{ s.substitute, s.tag });
                            }
                            if (changed != part) changed = windows_name(changed, std::string());
                        }
                        merge_report(out.report, report_json(r), std::string());
                        it = done.insert(std::make_pair(part, changed)).first;
                    }
                    result += it->second;
                }
                else
                {
                    result += part;
                }
                if (slash < folders.size()) result += '/';
                start = slash + 1;
            }
            return result;
        }

        engine_result archive(const std::string& bytes, const std::string& name, const picture_handler& handler,
                              bool restoring, const substitute_lookup& lookup, int depth, uint64_t& budget) const
        {
            if (depth > 3) throw std::runtime_error(base_name(name) + ": archives nested too deep");
            archive_kind ak = archive_kind_of(bytes);
            if (ak == archive_kind::none) ak = archive_kind_of_name(name);
            // The entries come out of what is left of the budget of the file.
            archive_limits limits;
            limits.max_total_size = std::min<uint64_t>(limits.max_total_size, budget);
            limits.max_entry_size = std::min<uint64_t>(limits.max_entry_size, budget);
            std::vector<archive_entry> entries = read_archive(ak, bytes, limits);
            uint64_t read = 0;
            for (const archive_entry& e : entries) read += e.data.size();
            budget -= std::min(budget, read);
            engine_result out;
            out.kind = archive_kind_name(ak);
            out.report = report_json(document_report());
            {
                document_report r;
                out.name = folder_of(name) + (restoring ? restore_alone(base_name(name), lookup, r)
                                                        : protect_alone(base_name(name), lookup, out.substitutions, r));
                merge_report(out.report, report_json(r), std::string());
            }
            std::set<std::string> names;
            std::map<std::string, int> next_number;
            std::set<std::string> pictures;
            std::map<std::string, std::string> folders;
            for (const archive_entry& e : entries) names.insert(lower(e.name));
            std::vector<archive_entry> kept;
            for (archive_entry& e : entries)
            {
                if (e.directory)
                {
                    const std::string renamed = entry_folders(e.name, restoring, lookup, folders, out);
                    if (renamed != e.name)
                    {
                        names.erase(lower(e.name));
                        e.name = renamed;
                        names.insert(lower(e.name));
                    }
                    kept.push_back(std::move(e));
                    continue;
                }
                // An entry left out gives its memory back at once.
                if (!e.regular || engine::kind_of(e.data, e.name) == file_kind::unsupported)
                {
                    out.report["dropped"].push_back(e.name);
                    std::string().swap(e.data);
                    continue;
                }
                try
                {
                    engine_result r = any(e.data, e.name, handler, restoring, lookup, depth + 1, budget);
                    const std::string renamed =
                        entry_folders(folder_of(e.name), restoring, lookup, folders, out) + base_name(r.name);
                    if (renamed != e.name)
                    {
                        names.erase(lower(e.name));
                        e.name = free_name(renamed, names, next_number);
                    }
                    e.data = std::move(r.bytes);
                    merge_report(out.report, r.report, e.name);
                    for (const std::string& id : r.pictures)
                        if (pictures.insert(id).second) out.pictures.push_back(id);
                    for (substitution& s : r.substitutions) out.substitutions.push_back(std::move(s));
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
                    std::string().swap(e.data);
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

    engine_result engine::anonymize(const std::string& bytes, const std::string& name,
                                    const picture_handler& pictures, const substitute_lookup& lookup) const
    {
        uint64_t budget = decompression_budget(bytes.size());
        return p_->any(bytes, name, pictures, false, lookup, 0, budget);
    }

    engine_result engine::restore(const std::string& bytes, const std::string& name,
                                  const substitute_lookup& lookup) const
    {
        uint64_t budget = decompression_budget(bytes.size());
        return p_->any(bytes, name, picture_handler(), true, lookup, 0, budget);
    }

    std::string engine::anonymize_name(const std::string& name, const substitute_lookup& lookup,
                                       std::vector<substitution>& made, json& report) const
    {
        document_report r;
        const std::string out = impl::folder_of(name) + p_->protect_alone(base_name(name), lookup, made, r);
        report = report_json(r);
        return out;
    }

    std::string engine::restore_name(const std::string& name, const substitute_lookup& lookup, json& report) const
    {
        document_report r;
        const std::string out = impl::folder_of(name) + p_->restore_alone(base_name(name), lookup, r);
        report = report_json(r);
        return out;
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
