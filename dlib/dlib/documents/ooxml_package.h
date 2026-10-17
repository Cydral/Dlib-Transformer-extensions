// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OOXML_PACKAGE_H_
#define DLIB_OOXML_PACKAGE_H_

#include "ooxml_package_abstract.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "../zip/zip_archive.h"
#include "xml_scanner.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class ooxml_kind
    {
        unknown,
        wordprocessing,
        spreadsheet,
        presentation
    };

    struct ooxml_relationship
    {
        std::string id;
        std::string type;
        std::string target;
        bool external = false;
    };

// ----------------------------------------------------------------------------------------

    class ooxml_package
    {
    public:
        explicit ooxml_package(const std::string& zip_bytes, const zip_limits& limits = zip_limits())
        {
            const zip_archive_reader zip(zip_bytes, limits);
            for (std::size_t i = 0; i < zip.entries().size(); ++i)
            {
                const std::string& stored = zip.entries()[i].name;
                if (!stored.empty() && stored.back() == '/')
                    continue;
                const std::string name = strip_slash(stored);
                // Part names are compared without case; two members naming one part are refused.
                if (parts_.count(key(name)))
                    throw error("ooxml_package: two members name the part " + name);
                parts_[key(name)] = stored_part{ name, zip.read(i) };
                order_.push_back(key(name));
            }
            const auto it = parts_.find(key("[Content_Types].xml"));
            if (it == parts_.end())
                throw error("ooxml_package: [Content_Types].xml is missing");
            it->second.content = xml_to_utf8(it->second.content);
            parse_content_types(it->second.content);
            // XML parts in UTF-16 are converted, so that every reader sees UTF-8.
            for (auto& p : parts_)
            {
                const std::string ct = content_type(p.second.name);
                if (ct.find("xml") != std::string::npos || ends_with(p.first, ".xml") || ends_with(p.first, ".rels"))
                    p.second.content = xml_to_utf8(p.second.content);
            }
        }

        ooxml_kind kind() const
        {
            const std::string main = main_part();
            if (main.empty()) return ooxml_kind::unknown;
            const std::string ct = content_type(main);
            if (ct.find("wordprocessingml") != std::string::npos || ct.find("ms-word") != std::string::npos)
                return ooxml_kind::wordprocessing;
            if (ct.find("spreadsheetml") != std::string::npos || ct.find("ms-excel") != std::string::npos)
                return ooxml_kind::spreadsheet;
            if (ct.find("presentationml") != std::string::npos || ct.find("ms-powerpoint") != std::string::npos)
                return ooxml_kind::presentation;
            return ooxml_kind::unknown;
        }

        std::string main_part() const
        {
            for (const ooxml_relationship& r : relationships(""))
            {
                if (!r.external && ends_with(r.type, "/officeDocument"))
                    return target_part("", r.target);
            }
            return std::string();
        }

        std::vector<std::string> part_names() const
        {
            std::vector<std::string> names;
            for (const std::string& k : order_)
            {
                const auto it = parts_.find(k);
                if (it != parts_.end()) names.push_back(it->second.name);
            }
            return names;
        }

        bool has_part(const std::string& name) const { return parts_.count(key(name)) != 0; }

        const std::string& part(const std::string& name) const
        {
            const auto it = parts_.find(key(name));
            if (it == parts_.end())
                throw error("ooxml_package: no part named " + name);
            return it->second.content;
        }

        void set_part(const std::string& name, const std::string& content)
        {
            const std::string k = key(name);
            const auto it = parts_.find(k);
            if (it != parts_.end())
            {
                it->second.content = content;
                return;
            }
            parts_[k] = stored_part{ name, content };
            if (std::find(order_.begin(), order_.end(), k) == order_.end())
                order_.push_back(k);
        }

        void remove_part(const std::string& name)
        {
            const std::string k = key(name);
            parts_.erase(k);
            overrides_.erase("/" + k);
            order_.erase(std::remove(order_.begin(), order_.end(), k), order_.end());
        }

        std::string content_type(const std::string& name) const
        {
            const auto o = overrides_.find("/" + key(name));
            if (o != overrides_.end()) return o->second;
            const std::size_t dot = name.rfind('.');
            if (dot != std::string::npos && name.find('/', dot) == std::string::npos)
            {
                const auto d = defaults_.find(key(name.substr(dot + 1)));
                if (d != defaults_.end()) return d->second;
            }
            return std::string();
        }

        void set_override(const std::string& name, const std::string& type)
        {
            overrides_["/" + key(name)] = type;
        }

        static std::string rels_name(const std::string& source)
        {
            if (source.empty()) return "_rels/.rels";
            const std::size_t slash = source.rfind('/');
            if (slash == std::string::npos) return "_rels/" + source + ".rels";
            return source.substr(0, slash) + "/_rels/" + source.substr(slash + 1) + ".rels";
        }

        std::vector<ooxml_relationship> relationships(const std::string& source) const
        {
            std::vector<ooxml_relationship> rels;
            const auto it = parts_.find(key(rels_name(source)));
            if (it == parts_.end()) return rels;
            const std::string& xml = it->second.content;
            for (const xml_token& t : xml_tokenize(xml))
            {
                if ((t.kind != xml_token::start_tag && t.kind != xml_token::empty_tag) ||
                    xml_local_name(t.name) != "Relationship")
                    continue;
                ooxml_relationship r;
                for (const xml_attribute& a : t.attributes)
                {
                    const std::string v = xml_unescape(xml.substr(a.value_begin, a.value_end - a.value_begin));
                    if (a.name == "Id") r.id = v;
                    else if (a.name == "Type") r.type = v;
                    else if (a.name == "Target") r.target = v;
                    else if (a.name == "TargetMode") r.external = (v == "External");
                }
                rels.push_back(r);
            }
            return rels;
        }

        static std::string resolve_target(const std::string& source, const std::string& target)
        {
            const std::string out = fold_path(source, target);
            // Percent-encoded names ("my%20image.png") map to the stored member name.
            std::string decoded;
            for (std::size_t i = 0; i < out.size(); ++i)
            {
                if (out[i] == '%' && i + 2 < out.size() && std::isxdigit(static_cast<unsigned char>(out[i + 1])) &&
                    std::isxdigit(static_cast<unsigned char>(out[i + 2])))
                {
                    decoded.push_back(static_cast<char>(std::stoi(out.substr(i + 1, 2), nullptr, 16)));
                    i += 2;
                }
                else
                {
                    decoded.push_back(out[i]);
                }
            }
            return decoded;
        }

        std::string target_part(const std::string& source, const std::string& target) const
        {
            const std::string decoded = resolve_target(source, target);
            auto it = parts_.find(key(decoded));
            if (it != parts_.end()) return it->second.name;
            it = parts_.find(key(fold_path(source, target)));
            if (it != parts_.end()) return it->second.name;
            return decoded;
        }

        std::vector<std::string> collect_garbage()
        {
            std::set<std::string> reachable;
            std::vector<std::string> todo(1, std::string());
            while (!todo.empty())
            {
                const std::string src = todo.back();
                todo.pop_back();
                if (!src.empty())
                {
                    if (reachable.count(key(src))) continue;
                    reachable.insert(key(src));
                }
                for (const ooxml_relationship& r : relationships(src))
                {
                    if (r.external) continue;
                    const std::string target = target_part(src, r.target);
                    if (has_part(target) && !reachable.count(key(target)))
                        todo.push_back(target);
                }
            }
            std::vector<std::string> removed;
            for (const std::string& name : part_names())
            {
                const std::string k = key(name);
                if (k == "[content_types].xml" || k == "_rels/.rels") continue;
                if (reachable.count(k)) continue;
                if (is_rels(name) && reachable.count(key(rels_source(name)))) continue;
                removed.push_back(name);
            }
            for (const std::string& name : removed)
                remove_part(name);
            // Overrides for parts that no longer exist.
            for (auto it = overrides_.begin(); it != overrides_.end();)
            {
                if (!parts_.count(it->first.substr(1))) it = overrides_.erase(it);
                else ++it;
            }
            return removed;
        }

        std::string save(int level = 6) const
        {
            zip_archive_writer w(level);
            w.add("[Content_Types].xml", build_content_types());
            for (const std::string& name : part_names())
            {
                if (key(name) == "[content_types].xml") continue;
                w.add(name, part(name));
            }
            return w.finish();
        }

        static bool is_rels(const std::string& name)
        {
            const std::string k = key(name);
            return ends_with(k, ".rels") && (k.compare(0, 6, "_rels/") == 0 || k.find("/_rels/") != std::string::npos);
        }

        static std::string rels_source(const std::string& rels)
        {
            const std::string k = key(rels);
            if (k == "_rels/.rels") return std::string();
            const std::size_t pos = k.rfind("_rels/");
            if (pos == std::string::npos || rels.size() < pos + 6 + 5) return std::string();
            const std::string folder = rels.substr(0, pos);
            const std::string file = rels.substr(pos + 6, rels.size() - pos - 6 - 5);
            return folder + file;
        }

        static bool ends_with(const std::string& s, const std::string& suffix)
        {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

    private:
        struct stored_part
        {
            std::string name;       // as stored, without the leading slash
            std::string content;
        };

        // Lookup key of a part name: ASCII letters in lower case, as OPC compares them.
        static std::string key(const std::string& name)
        {
            std::string k = name;
            for (char& c : k) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            return k;
        }

        static std::string strip_slash(const std::string& name)
        {
            return (!name.empty() && name[0] == '/') ? name.substr(1) : name;
        }

        // The target resolved against the folder of source, with "." and ".." folded.
        static std::string fold_path(const std::string& source, const std::string& target)
        {
            std::string path;
            if (!target.empty() && target[0] == '/')
            {
                path = target.substr(1);
            }
            else
            {
                const std::size_t slash = source.rfind('/');
                path = (slash == std::string::npos ? std::string() : source.substr(0, slash + 1)) + target;
            }
            std::vector<std::string> segs;
            std::size_t start = 0;
            while (start <= path.size())
            {
                std::size_t slash = path.find('/', start);
                if (slash == std::string::npos) slash = path.size();
                const std::string seg = path.substr(start, slash - start);
                if (seg == "..") { if (!segs.empty()) segs.pop_back(); }
                else if (!seg.empty() && seg != ".") segs.push_back(seg);
                start = slash + 1;
            }
            std::string out;
            for (const std::string& seg : segs)
            {
                if (!out.empty()) out += "/";
                out += seg;
            }
            return out;
        }

        void parse_content_types(const std::string& xml)
        {
            for (const xml_token& t : xml_tokenize(xml))
            {
                if (t.kind != xml_token::start_tag && t.kind != xml_token::empty_tag) continue;
                const std::string local = xml_local_name(t.name);
                std::string ext, part_name, type;
                for (const xml_attribute& a : t.attributes)
                {
                    const std::string v = xml_unescape(xml.substr(a.value_begin, a.value_end - a.value_begin));
                    if (a.name == "Extension") ext = v;
                    else if (a.name == "PartName") part_name = v;
                    else if (a.name == "ContentType") type = v;
                }
                if (local == "Default" && !ext.empty())
                {
                    defaults_[key(ext)] = type;
                    default_order_.push_back(key(ext));
                }
                else if (local == "Override" && !part_name.empty())
                {
                    overrides_["/" + key(strip_slash(part_name))] = type;
                }
            }
        }

        std::string build_content_types() const
        {
            std::string x = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\r\n"
                            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">";
            std::set<std::string> done;
            for (const std::string& ext : default_order_)
            {
                if (done.count(ext)) continue;
                done.insert(ext);
                x += "<Default Extension=\"" + xml_escape_attribute(ext) + "\" ContentType=\"" +
                     xml_escape_attribute(defaults_.at(ext)) + "\"/>";
            }
            for (const auto& o : overrides_)
            {
                const auto it = parts_.find(o.first.substr(1));
                if (it == parts_.end()) continue;
                x += "<Override PartName=\"" + xml_escape_attribute("/" + it->second.name) + "\" ContentType=\"" +
                     xml_escape_attribute(o.second) + "\"/>";
            }
            x += "</Types>";
            return x;
        }

        std::map<std::string, stored_part> parts_;      // by key()
        std::vector<std::string> order_;                // keys, in archive order
        std::map<std::string, std::string> defaults_;   // by lower case extension
        std::vector<std::string> default_order_;
        std::map<std::string, std::string> overrides_;  // by "/" + key()
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OOXML_PACKAGE_H_
