// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OOXML_ANONYMIZER_H_
#define DLIB_OOXML_ANONYMIZER_H_

#include "ooxml_anonymizer_abstract.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "../crc32.h"
#include "../zip/deflate.h"
#include "../documents/xml_scanner.h"
#include "../documents/ooxml_package.h"
#include "pii_detector.h"
#include "pseudonymizer.h"
#include "document_report.h"
#include "date_formats.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        struct ooxml_ns
        {
            static bool w(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/wordprocessingml/2006/main" ||
                    u == "http://purl.oclc.org/ooxml/wordprocessingml/main";
            }
            static bool a(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/drawingml/2006/main" ||
                    u == "http://purl.oclc.org/ooxml/drawingml/main";
            }
            static bool p(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/presentationml/2006/main" ||
                    u == "http://purl.oclc.org/ooxml/presentationml/main";
            }
            static bool s(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/spreadsheetml/2006/main" ||
                    u == "http://purl.oclc.org/ooxml/spreadsheetml/main";
            }
            static bool mc(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/markup-compatibility/2006";
            }
            static bool r(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/officeDocument/2006/relationships" ||
                    u == "http://purl.oclc.org/ooxml/officeDocument/relationships";
            }
            static bool core(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/package/2006/metadata/core-properties";
            }
            static bool dc(const std::string& u) { return u == "http://purl.org/dc/elements/1.1/"; }
            static bool dcterms(const std::string& u) { return u == "http://purl.org/dc/terms/"; }
            static bool extended(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" ||
                    u == "http://purl.oclc.org/ooxml/officeDocument/extendedProperties";
            }
            static bool custom(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" ||
                    u == "http://purl.oclc.org/ooxml/officeDocument/customProperties";
            }
            static bool vt(const std::string& u)
            {
                return u == "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes" ||
                    u == "http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes";
            }
            static bool pc2018(const std::string& u)
            {
                return u == "http://schemas.microsoft.com/office/powerpoint/2018/8/main";
            }
            static bool tc2018(const std::string& u)
            {
                return u == "http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments";
            }
        };

        inline std::string png_placeholder()
        {
            auto chunk = [](const std::string& type, const std::string& data) {
                std::string c;
                const std::uint32_t len = static_cast<std::uint32_t>(data.size());
                for (int i = 3; i >= 0; --i) c.push_back(static_cast<char>((len >> (8 * i)) & 0xFF));
                c += type + data;
                dlib::crc32 crc;
                crc.add(type + data);
                const std::uint32_t v = static_cast<std::uint32_t>(crc.get_checksum());
                for (int i = 3; i >= 0; --i) c.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
                return c;
            };
            const std::string ihdr = std::string("\0\0\0\x01\0\0\0\x01\x08\x06\0\0\0", 13);
            return std::string("\x89PNG\r\n\x1a\n", 8) + chunk("IHDR", ihdr) +
                   chunk("IDAT", zlib_compress(std::string(5, '\0'))) + chunk("IEND", "");
        }

        inline std::string image_placeholder(const std::string& name)
        {
            std::string ext = name.substr(name.rfind('.') + 1);
            for (char& c : ext) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            if (ext == "gif")
                return std::string("GIF89a\x01\0\x01\0\x80\0\0\0\0\0\xff\xff\xff\x21\xf9\x04\x01\0\0\0\0\x2c\0\0\0\0"
                                   "\x01\0\x01\0\0\x02\x02\x44\x01\0\x3b",
                                   43);
            if (ext == "jpg" || ext == "jpeg" || ext == "jpe")
                return from_hex("ffd8ffe000104a46494600010100000100010000ffdb004300100b0c0e0c0a100e0d0e1211101318281a18"
                                "1616183123251d"
                                "283a333d3c3933383740485c4e404457453738506d51575f626768673e4d71797064785c656763ffdb0043"
                                "011112121815182f"
                                "1a1a2f63423842636363636363636363636363636363636363636363636363636363636363636363636363"
                                "6363636363636363"
                                "636363636363ffc00011080001000103012200021101031101ffc400150001010000000000000000000000"
                                "0000000007ffc400"
                                "14100100000000000000000000000000000000ffc40014010100000000000000000000000000000000ffc4"
                                "0014110100000000"
                                "000000000000000000000000ffda000c03010002110311003f00a0000fffd9");
            if (ext == "bmp" || ext == "dib")
                return std::string(
                    "BM\x3a\0\0\0\0\0\0\0\x36\0\0\0\x28\0\0\0\x01\0\0\0\x01\0\0\0\x01\0\x18\0\0\0\0\0\x04\0\0\0"
                    "\x13\x0b\0\0\x13\x0b\0\0\0\0\0\0\0\0\0\0\xff\xff\xff\0",
                    58);
            if (ext == "svg")
                return "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1\" height=\"1\"/>";
            return png_placeholder();
        }

        inline bool is_media_part(const std::string& name)
        {
            const std::size_t dot = name.rfind('.');
            if (dot == std::string::npos) return false;
            std::string ext = name.substr(dot + 1);
            for (char& c : ext) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            static const std::set<std::string> exts = { "png", "jpg", "jpeg", "jpe", "gif", "bmp", "dib", "tif", "tiff",
                "emf", "wmf", "emz", "wmz", "svg", "wdp", "jxr", "hdp", "ico", "pict", "pct", "webp", "heic", "avif" };
            return exts.count(ext) != 0;
        }

        inline bool removable_relationship(const std::string& type)
        {
            static const char* suffixes[] = { "/image",
                                              "/chart",
                                              "/diagramData",
                                              "/diagramLayout",
                                              "/diagramQuickStyle",
                                              "/diagramColors",
                                              "/diagramDrawing",
                                              "/oleObject",
                                              "/package",
                                              "/video",
                                              "/audio",
                                              "/media",
                                              "/hdphoto",
                                              "/drawing",
                                              "/chartUserShapes",
                                              "/control",
                                              "/activeXControlBinary",
                                              "/vbaProject",
                                              "/printerSettings",
                                              "/thumbnail",
                                              "/model3d",
                                              "/slideUpdateInfo",
                                              "/ctrlProp",
                                              "/attachedToolbars",
                                              "/legacyDocTextInfo",
                                              "/oleObjectStorage",
                                              "/vbaProjectSignature",
                                              "/digital-signature/origin",
                                              "/digital-signature/signature",
                                              "/aFChunk",
                                              "/subDocument",
                                              "/chartEx" };
            for (const char* s : suffixes)
                if (ooxml_package::ends_with(type, s)) return true;
            return false;
        }

        // Shifts the YYYY-MM-DD at the start of an ISO 8601 timestamp, keeping the rest.
        inline bool shift_timestamp(std::string& value, long days)
        {
            if (value.size() < 10 || value[4] != '-' || value[7] != '-') return false;
            for (int k : { 0, 1, 2, 3, 5, 6, 8, 9 })
                if (value[static_cast<std::size_t>(k)] < '0' || value[static_cast<std::size_t>(k)] > '9') return false;
            const int y = std::stoi(value.substr(0, 4)), m = std::stoi(value.substr(5, 2)),
                      d = std::stoi(value.substr(8, 2));
            if (!is_valid_civil_date(y, m, d)) return false;
            const civil_date c = civil_from_days(days_from_civil(y, m, d) + days);
            date_format iso;
            value = format_date(c, iso) + value.substr(10);
            return true;
        }

        /* Adds days to the integer part of a spreadsheet serial written in decimal, keeping
           the fractional digits exactly. */
        inline bool shift_serial_text(std::string& value, long days)
        {
            if (value.empty()) return false;
            const std::size_t dot = value.find('.');
            const std::string ip = value.substr(0, dot);
            if (ip.empty() || ip.size() > 9) return false;
            for (char c : ip) if (c < '0' || c > '9') return false;
            if (dot != std::string::npos)
                for (std::size_t k = dot + 1; k < value.size(); ++k) if (value[k] < '0' || value[k] > '9') return false;
            const long v = std::stol(ip) + days;
            if (v < 1) return false;
            value = std::to_string(v) + (dot == std::string::npos ? std::string() : value.substr(dot));
            return true;
        }

        // A non-negative decimal integer of at most nine digits at the start of s, or -1.
        inline long small_number(const std::string& s, std::size_t from = 0)
        {
            long v = 0;
            std::size_t k = from;
            for (; k < s.size() && k < from + 9 && s[k] >= '0' && s[k] <= '9'; ++k)
                v = v * 10 + (s[k] - '0');
            return k == from ? -1 : v;
        }

        // Column of a cell reference ("C12" gives 2); -1 when it has no column or more than three letters.
        inline int column_index(const std::string& ref)
        {
            int col = 0;
            std::size_t k = 0;
            for (; k < ref.size(); ++k)
            {
                const char c = ref[k];
                if (c >= 'A' && c <= 'Z') col = col * 26 + (c - 'A' + 1);
                else if (c >= 'a' && c <= 'z') col = col * 26 + (c - 'a' + 1);
                else break;
                if (k >= 3) return -1;
            }
            return col - 1;
        }

        inline int row_index(const std::string& ref)
        {
            std::size_t k = 0;
            while (k < ref.size() && ((ref[k] >= 'A' && ref[k] <= 'Z') || (ref[k] >= 'a' && ref[k] <= 'z'))) ++k;
            return static_cast<int>(std::max(0L, small_number(ref, k)));
        }

        /* Parts read as XML: content types of the XML family, and .xml parts without one.
           Binary parts whose type merely names the openxmlformats family are not. */
        inline bool xml_part(const std::string& name, const std::string& ct)
        {
            if (ooxml_package::ends_with(name, ".vml") || ooxml_package::ends_with(name, ".VML")) return false;
            if (ct == "application/xml" || ct == "text/xml" || ooxml_package::ends_with(ct, "+xml")) return true;
            if (!ct.empty()) return false;
            std::string lower = name;
            for (char& c : lower) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            return ooxml_package::ends_with(lower, ".xml");
        }

        // Content types of parts that are dropped with any relationship that targets them.
        inline bool removable_content_type(const std::string& ct)
        {
            return ct == "application/inkml+xml" || ct == "application/vnd.ms-office.chartex+xml" ||
                   ct == "application/vnd.ms-office.inkml+xml";
        }

        /* Number formats that display a date: the built-in identifiers and custom codes
           containing day or year fields outside quoted text and bracketed sections. */
        inline bool is_date_format(int id, const std::string& code)
        {
            if ((id >= 14 && id <= 17) || id == 22 || (id >= 27 && id <= 36) || (id >= 50 && id <= 58))
                return true;
            if (code.empty()) return false;
            bool quote = false, bracket = false;
            for (std::size_t i = 0; i < code.size(); ++i)
            {
                const char c = code[i];
                if (c == '\\') { ++i; continue; }
                if (c == '"') { quote = !quote; continue; }
                if (quote) continue;
                if (c == '[') { bracket = true; continue; }
                if (c == ']') { bracket = false; continue; }
                if (bracket) continue;
                if (c == 'd' || c == 'D' || c == 'y' || c == 'Y') return true;
            }
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    class ooxml_anonymizer
    {
    public:
        ooxml_anonymizer(
            const pseudonymizer& p,
            pii_detector* detector,
            document_report& report
        ) : p_(p), detector_(detector), report_(report) {}

        std::string anonymize(const std::string& package_bytes)
        {
            if (!detector_)
                throw error("ooxml_anonymizer::anonymize(): a detector is required");
            ooxml_package pkg(package_bytes);
            restoring_ = false;
            run(pkg);
            return pkg.save();
        }

        std::string restore(const std::string& package_bytes)
        {
            ooxml_package pkg(package_bytes);
            restoring_ = true;
            run(pkg);
            return pkg.save();
        }

    private:
        typedef privacy_impl::ooxml_ns ns;

        struct text_node
        {
            std::size_t raw_begin, raw_end;     // raw content range, or attribute value range
            std::size_t tag_end;                // end of the start tag, for xml:space insertion
            std::string text;
            bool attribute;
            bool wml_text;                      // w:t, which needs xml:space="preserve" for edge spaces
            bool has_preserve;
            bool changed;
            std::string new_text;
            int kind = 0;                       // 0 text, 1 spreadsheet formula, 2 sheet name, 3 sheet reference
        };

        struct piece
        {
            int node;                           // -1 for a virtual character
            std::size_t ub, ue;                 // range in the unit text
            std::size_t xml_begin, xml_end;     // element to delete when a virtual piece is replaced
        };

        struct unit
        {
            std::string text;
            std::vector<piece> pieces;
            detection_context ctx;
            int table = -1, row = -1, col = -1;
            bool main_stream = true;
            int shared_index = -1;
            std::vector<std::size_t> cuts;      // container boundaries no value may cross
        };

        struct cell_info
        {
            int row, col;
            std::string type;
            int style;
            bool formula;
            std::size_t c_begin, c_end;         // whole <c> element
            std::size_t v_begin, v_end;         // raw <v> content
            bool has_v;
            std::string v_text;
            int inline_unit;
            std::vector<std::pair<std::string, std::string>> attrs;  // raw attributes of <c>
        };

        struct part_state
        {
            std::string name;
            std::string xml;
            std::vector<xml_token> tokens;
            std::vector<text_node> nodes;
            std::vector<unit> units;
            xml_edit_list edits;
            std::vector<std::pair<std::size_t, std::size_t>> removed;
            std::string role;                   // wml, pml, sst, sheet, comments, generic...
            std::vector<cell_info> cells;
            std::map<int, pii_type> column_types;
            int header_row = -1;
            std::set<std::string> removed_ids;  // shape identifiers of removed slide elements
        };

    // ------------------------------------------------------------------------------------

        void run(ooxml_package& pkg)
        {
            report_.format = pkg.kind() == ooxml_kind::wordprocessing ? "docx" :
                             pkg.kind() == ooxml_kind::spreadsheet ? "xlsx" :
                             pkg.kind() == ooxml_kind::presentation ? "pptx" : "ooxml";
            markers_ = read_markers(pkg);
            dates_marked_ = markers_.count("dlib.privacy") != 0;
            // Dates held in typed fields cannot carry a tag: they are shifted unless kept.
            shift_dates_ = restoring_ ? dates_marked_ : (p_.get_options().dates != date_policy::keep);
            shift_ = restoring_ ? -p_.date_shift_days() : p_.date_shift_days();
            sheet_names_.clear();
            if (restoring_) read_sheet_markers();

            load_spreadsheet_tables(pkg);

            std::vector<part_state> parts;
            for (const std::string& name : pkg.part_names())
            {
                if (name == "[Content_Types].xml" || ooxml_package::is_rels(name)) continue;
                if (!privacy_impl::xml_part(name, pkg.content_type(name))) continue;
                part_state st;
                st.name = name;
                st.xml = pkg.part(name);
                try
                {
                    st.tokens = xml_tokenize(st.xml);
                }
                catch (xml_error& e)
                {
                    // A part that cannot be read could hold personal data: the package is refused.
                    throw error("ooxml_anonymizer: part " + name + " is not well-formed XML (" + e.info +
                                "); nothing was written");
                }
                analyze_part(st);
                std::vector<xml_token>().swap(st.tokens);
                parts.push_back(std::move(st));
            }

            assign_shared_string_types(parts);
            assign_table_contexts(parts);

            if (!restoring_)
            {
                detector_->begin_document();
                for (const part_state& st : parts)
                    for (const unit& u : st.units)
                        if (!u.text.empty()) detector_->learn(u.text, u.ctx);
                rename_sheets(parts);
            }

            for (part_state& st : parts)
            {
                for (unit& u : st.units)
                    transform_unit(st, u);
                transform_special_nodes(st);
                finish_part(st);
                if (st.role == "sheet")
                    transform_cells(st);
                pkg.set_part(st.name, st.edits.apply(st.xml));
            }

            process_relationships(pkg);
            if (!restoring_)
            {
                for (const std::string& r : pkg.collect_garbage())
                    report_.removed.push_back(r);
                for (const std::string& name : pkg.part_names())
                {
                    if (privacy_impl::is_media_part(name))
                    {
                        pkg.set_part(name, privacy_impl::image_placeholder(name));
                        report_.removed.push_back(name + " (content replaced by a blank image, still referenced)");
                    }
                }
                std::map<std::string, std::string> markers;
                if (shift_dates_ && report_.dates_shifted > 0) markers["dlib.privacy"] = "dates-shifted";
                int k = 0;
                for (const auto& r : sheet_names_)
                    markers["dlib.privacy.sheet" + std::to_string(++k)] = r.second + "=" + sheet_tags_[r.first];
                if (!markers.empty()) write_markers(pkg, markers);
            }
            else if (!markers_.empty())
            {
                clear_markers(pkg);
            }
        }

    // ------------------------------------------------------------------------------------
    // Classification and unit extraction.

        std::vector<std::size_t> match_ends(const std::vector<xml_token>& tokens)
        {
            std::vector<std::size_t> match(tokens.size(), std::string::npos);
            std::vector<std::size_t> stack;
            for (std::size_t i = 0; i < tokens.size(); ++i)
            {
                if (tokens[i].kind == xml_token::start_tag) stack.push_back(i);
                else if (tokens[i].kind == xml_token::end_tag && !stack.empty())
                {
                    match[stack.back()] = i;
                    stack.pop_back();
                }
                else if (tokens[i].kind == xml_token::empty_tag) match[i] = i;
            }
            return match;
        }

        static bool has_image_marker(const part_state& st, std::size_t from, std::size_t to)
        {
            for (std::size_t k = from; k <= to && k < st.tokens.size(); ++k)
            {
                const xml_token& t = st.tokens[k];
                if (t.kind != xml_token::start_tag && t.kind != xml_token::empty_tag) continue;
                const std::string l = xml_local_name(t.name);
                if (l == "blip" || l == "imagedata" || l == "OLEObject" || l == "chart" || l == "relIds" ||
                    l == "binData" || l == "contentPart" || l == "videoFile" || l == "audioFile" ||
                    l == "wavAudioFile" || l == "quickTimeFile" || l == "oleObj" || l == "webVideoPr" ||
                    (l == "pic" && xml_prefix(t.name) == "pic"))
                    return true;
                if (l == "fill" || l == "shape" || l == "rect")
                    for (const xml_attribute& a : t.attributes)
                        if (xml_local_name(a.name) == "id" && xml_prefix(a.name) == "r") return true;
            }
            return false;
        }

        void analyze_part(part_state& st)
        {
            const std::vector<std::size_t> match = match_ends(st.tokens);
            xml_namespace_tracker nst;
            std::string root_ns, root_local;
            for (const xml_token& t : st.tokens)
            {
                if (t.kind == xml_token::start_tag || t.kind == xml_token::empty_tag)
                {
                    nst.enter(st.xml, t);
                    root_ns = nst.resolve(t.name);
                    root_local = xml_local_name(t.name);
                    break;
                }
            }

            if (ns::w(root_ns)) st.role = "wml";
            else if (ns::p(root_ns) &&
                     (root_local == "sld" || root_local == "sldLayout" || root_local == "sldMaster" ||
                      root_local == "notes" || root_local == "notesMaster" || root_local == "handoutMaster"))
                st.role = "pml";
            else if (ns::p(root_ns) && root_local == "cmLst") st.role = "pml_comments";
            else if (ns::p(root_ns) && root_local == "cmAuthorLst") st.role = "attributes";
            else if (ns::p(root_ns) && root_local == "tagLst") st.role = "attributes";
            else if (ns::pc2018(root_ns)) st.role = "pml";
            else if (ns::s(root_ns) && root_local == "sst") st.role = "sst";
            else if (ns::s(root_ns) && (root_local == "worksheet" || root_local == "dialogsheet")) st.role = "sheet";
            else if (ns::s(root_ns) && root_local == "comments") st.role = "xlsx_comments";
            else if (ns::s(root_ns) && (root_local == "pivotCacheDefinition" || root_local == "pivotCacheRecords" ||
                                        root_local == "pivotTableDefinition"))
                st.role = "attributes";
            else if (ns::s(root_ns) && root_local == "connections") st.role = "attributes";
            else if (ns::s(root_ns) && root_local == "workbook") st.role = "workbook";
            else if (ns::s(root_ns) && root_local == "chartsheet") st.role = "chartsheet";
            else if (ns::s(root_ns) && (root_local == "styleSheet" || root_local == "calcChain")) st.role = "skip";
            // External links, revision logs and user lists, and any other workbook part.
            else if (ns::s(root_ns)) st.role = "generic";
            else if (ns::tc2018(root_ns)) st.role = "threaded";
            else if (root_local == "personList") st.role = "attributes";
            else if (ns::core(root_ns)) st.role = "core";
            else if (ns::extended(root_ns)) st.role = "app";
            else if (ns::custom(root_ns)) st.role = "custom";
            else if (ns::a(root_ns) && root_local == "theme") st.role = "skip";
            else if (root_local == "layoutDef" || root_local == "styleDef" || root_local == "colorsDef") st.role = "skip";
            else if (ns::p(root_ns)) st.role = "attributes";
            // Charts, diagram data and drawings that outlive the removals are read as text.
            else st.role = "generic";

            if (st.role == "skip") return;
            walk_part(st, match);
        }

        static void add_piece(unit& u, int node, const std::string& text,
                              std::size_t xml_begin = std::string::npos, std::size_t xml_end = std::string::npos)
        {
            piece pc;
            pc.node = node;
            pc.ub = u.text.size();
            u.text += text;
            pc.ue = u.text.size();
            pc.xml_begin = xml_begin;
            pc.xml_end = xml_end;
            u.pieces.push_back(pc);
        }

        int add_text_node(part_state& st, const xml_token& start, std::size_t start_index, std::size_t end_index,
                          bool wml_text)
        {
            text_node n;
            n.attribute = false;
            n.wml_text = wml_text;
            n.changed = false;
            n.tag_end = start.end;
            n.has_preserve = false;
            for (const xml_attribute& a : start.attributes)
                if (a.name == "xml:space") n.has_preserve = true;
            n.raw_begin = start.end;
            n.raw_end = (end_index != std::string::npos && end_index < st.tokens.size()) ? st.tokens[end_index].begin
                                                                                         : start.end;
            // A text element holds character data, possibly split around comments; take it whole.
            std::string raw;
            for (std::size_t k = start_index + 1; k < end_index && k < st.tokens.size(); ++k)
            {
                const xml_token& t = st.tokens[k];
                if (t.kind == xml_token::text)
                    raw += xml_unescape(st.xml.substr(t.content_begin, t.content_end - t.content_begin));
                else if (t.kind == xml_token::cdata)
                    raw += st.xml.substr(t.content_begin, t.content_end - t.content_begin);
            }
            n.text = raw;
            st.nodes.push_back(n);
            return static_cast<int>(st.nodes.size() - 1);
        }

        int add_attribute_node(part_state& st, const xml_attribute& a, int kind = 0)
        {
            text_node n;
            n.kind = kind;
            n.attribute = true;
            n.wml_text = false;
            n.changed = false;
            n.has_preserve = true;
            n.tag_end = std::string::npos;
            n.raw_begin = a.value_begin;
            n.raw_end = a.value_end;
            n.text = xml_unescape(st.xml.substr(a.value_begin, a.value_end - a.value_begin));
            st.nodes.push_back(n);
            return static_cast<int>(st.nodes.size() - 1);
        }

        void attribute_unit(part_state& st, const xml_attribute& a, pii_type forced)
        {
            const int node = add_attribute_node(st, a);
            if (st.nodes[static_cast<std::size_t>(node)].text.empty()) return;
            unit u;
            u.ctx.forced = forced;
            add_piece(u, node, st.nodes[static_cast<std::size_t>(node)].text);
            st.units.push_back(u);
        }

        void date_attribute(part_state& st, const xml_attribute& a)
        {
            if (!shift_dates_) return;
            std::string v = xml_unescape(st.xml.substr(a.value_begin, a.value_end - a.value_begin));
            if (privacy_impl::shift_timestamp(v, shift_))
            {
                st.edits.replace(a.value_begin, a.value_end, xml_escape_attribute(v));
                ++report_.dates_shifted;
            }
        }

        static pii_type identity_type(const std::string& v)
        {
            return v.find('@') != std::string::npos ? pii_type::email : pii_type::identifier;
        }

        void handle_attributes(part_state& st, const xml_token& t, const std::string& element_ns,
                               const std::string& local)
        {
            for (const xml_attribute& a : t.attributes)
            {
                const std::string an = xml_local_name(a.name);
                if (a.name.compare(0, 5, "xmlns") == 0) continue;
                const std::string v = st.xml.substr(a.value_begin, a.value_end - a.value_begin);
                if (an == "author" || (an == "initials" && local != "rPr"))
                    attribute_unit(st, a, pii_type::person);
                else if ((an == "date" && ns::w(element_ns)) || (an == "dt" && local == "cm") ||
                         (an == "created" && local == "cm") || (an == "dT" && local == "threadedComment") ||
                         (an == "fullDate" && local == "date" && ns::w(element_ns)) ||
                         (an == "dateTime" && (local == "header" || local == "userInfo")))
                    date_attribute(st, a);
                else if ((an == "userName" && (local == "fileSharing" || local == "header")) ||
                         (an == "name" && local == "userInfo"))
                    attribute_unit(st, a, pii_type::person);
                else if (ns::w(element_ns) &&
                         (((an == "displayText" || an == "value") && local == "listItem") ||
                          (an == "val" && (local == "default" || local == "listEntry" || local == "helpText" ||
                                           local == "statusText" || local == "connectString" || local == "query" ||
                                           local == "udl"))))
                    attribute_unit(st, a, pii_type::none);
                else if (an == "val" && (local == "filter" || local == "customFilter" || local == "sheetName"))
                    attribute_unit(st, a, pii_type::none);
                else if ((an == "n" && local == "item") || (an == "name" && (local == "dataField" || local == "section" ||
                                                                            local == "custShow")))
                    attribute_unit(st, a, pii_type::none);
                else if (an == "name" && local == "sheet" && st.role == "workbook")
                    add_attribute_node(st, a, 2);
                else if ((an == "location" && local == "hyperlink" && st.role == "sheet") ||
                         (an == "sheet" && local == "worksheetSource"))
                    add_attribute_node(st, a, 3);
                else if (an == "name" && (local == "cmAuthor" || (local == "author" && ns::pc2018(element_ns))))
                    attribute_unit(st, a, pii_type::person);
                else if (an == "displayName" && local == "person")
                    attribute_unit(st, a, pii_type::person);
                else if (an == "userId" && (local == "person" || local == "author" || local == "presenceInfo"))
                    attribute_unit(st, a, identity_type(xml_unescape(v)));
                else if ((an == "descr" || an == "title") && (local == "docPr" || local == "cNvPr"))
                    attribute_unit(st, a, pii_type::none);
                else if ((an == "instr" && local == "fldSimple") ||
                         (an == "val" && (local == "docVar" || local == "tag")) ||
                         (an == "tooltip" && local == "hyperlink") || (an == "display" && local == "hyperlink"))
                    attribute_unit(st, a, pii_type::none);
                else if (an == "connection" && local == "dbPr")
                    attribute_unit(st, a, pii_type::none);
                else if (an == "v" && local == "s" && st.role == "attributes")
                    attribute_unit(st, a, pivot_field_type_);
            }
        }

        // Elements that carry pictures, charts, diagrams, ink, embedded objects or imported content.
        bool removal_target(const part_state& st, const std::string& e_ns, const std::string& local) const
        {
            if (st.role == "wml")
                return (ns::w(e_ns) &&
                        (local == "drawing" || local == "pict" || local == "object" || local == "binData" ||
                         local == "numPicBullet" || local == "lvlPicBulletId" || local == "altChunk")) ||
                    (ns::mc(e_ns) && local == "AlternateContent");
            if (st.role == "pml")
                return (ns::p(e_ns) && (local == "pic" || local == "graphicFrame" || local == "contentPart")) ||
                       (ns::mc(e_ns) && local == "AlternateContent");
            if (st.role == "sheet")
                return ns::s(e_ns) && (local == "drawing" || local == "picture" || local == "legacyDrawingHF" ||
                                       local == "oleObjects" || local == "controls" || local == "drawingHF");
            // A chart sheet keeps the drawing that holds its chart, which is read as text.
            if (st.role == "chartsheet")
                return ns::s(e_ns) && (local == "picture" || local == "legacyDrawingHF" || local == "drawingHF");
            return false;
        }

        // Sorted, non-overlapping ranges: whether [b, e) lies inside one of them.
        static bool inside_removed(const part_state& st, std::size_t b, std::size_t e)
        {
            auto it = std::upper_bound(st.removed.begin(), st.removed.end(), std::make_pair(b, std::string::npos));
            if (it == st.removed.begin()) return false;
            --it;
            return b >= it->first && e <= it->second;
        }

        int formula_node(part_state& st, const xml_token& start, std::size_t start_index, std::size_t end_index)
        {
            if (end_index == std::string::npos) return -1;
            const int node = add_text_node(st, start, start_index, end_index, false);
            st.nodes[static_cast<std::size_t>(node)].kind = 1;
            return node;
        }

        void walk_part(part_state& st, const std::vector<std::size_t>& match)
        {
            xml_namespace_tracker nst;
            struct frame { std::string ns, local; };
            std::vector<frame> stack;
            struct para { int current = -1, deleted = -1, field = -1; int table = -1, row = -1, col = -1; };
            std::vector<para> paras;
            struct table_state { int id, row, col; };
            std::vector<table_state> tables;
            int next_table = 0;
            int si_unit = -1, phonetic_unit = -1, next_si = 0;
            int comment_unit = -1;
            bool in_phonetic = false;
            bool in_text_element = false;

            auto unit_for = [&](para& p, int stream) -> unit& {
                int& idx = stream == 0 ? p.current : (stream == 1 ? p.deleted : p.field);
                if (idx < 0)
                {
                    unit u;
                    u.table = p.table; u.row = p.row; u.col = p.col;
                    u.main_stream = stream == 0;
                    st.units.push_back(u);
                    idx = static_cast<int>(st.units.size() - 1);
                }
                return st.units[static_cast<std::size_t>(idx)];
            };
            // A boundary of a hyperlink, field, revision or content control inside the paragraph.
            auto cut_here = [&](para& p) {
                for (int idx : { p.current, p.deleted, p.field })
                    if (idx >= 0)
                    {
                        unit& u = st.units[static_cast<std::size_t>(idx)];
                        if (u.cuts.empty() || u.cuts.back() != u.text.size()) u.cuts.push_back(u.text.size());
                    }
            };
            // The end of an element, its own end tag included, for empty and start tags alike.
            auto element_end = [&](std::size_t k) {
                return (match[k] != std::string::npos) ? st.tokens[match[k]].end : st.tokens[k].end;
            };

            for (std::size_t i = 0; i < st.tokens.size(); ++i)
            {
                const xml_token& t = st.tokens[i];
                if (t.kind == xml_token::start_tag || t.kind == xml_token::empty_tag)
                {
                    nst.enter(st.xml, t);
                    const std::string e_ns = nst.resolve(t.name);
                    const std::string local = xml_local_name(t.name);

                    if (!restoring_ && removal_target(st, e_ns, local))
                    {
                        const std::size_t end_tok = match[i];
                        bool remove = true;
                        if (end_tok != std::string::npos && st.role != "sheet" && (local == "AlternateContent" ||
                            local == "graphicFrame" || local == "pict" || local == "drawing"))
                            remove = has_image_marker(st, i, end_tok);
                        if (remove && end_tok != std::string::npos)
                        {
                            // Shape identifiers, so that animations aimed at them go too.
                            if (st.role == "pml")
                                for (std::size_t k = i; k <= end_tok; ++k)
                                    if (xml_local_name(st.tokens[k].name) == "cNvPr")
                                        for (const xml_attribute& a : st.tokens[k].attributes)
                                            if (a.name == "id")
                                                st.removed_ids.insert(
                                                    st.xml.substr(a.value_begin, a.value_end - a.value_begin));
                            st.edits.replace(t.begin, st.tokens[end_tok].end, "");
                            st.removed.push_back(std::make_pair(t.begin, st.tokens[end_tok].end));
                            report_.removed.push_back(st.name + ": " + local);
                            nst.leave();
                            i = end_tok;
                            continue;
                        }
                    }
                    if (!restoring_ && st.role == "pml" && ns::p(e_ns) && local == "timing" &&
                        match[i] != std::string::npos && !st.removed_ids.empty())
                    {
                        bool aimed = false;
                        for (std::size_t k = i; k <= match[i] && !aimed; ++k)
                            for (const xml_attribute& a : st.tokens[k].attributes)
                                if (xml_local_name(a.name) == "spid" &&
                                    st.removed_ids.count(st.xml.substr(a.value_begin, a.value_end - a.value_begin)))
                                    aimed = true;
                        if (aimed)
                        {
                            st.edits.replace(t.begin, st.tokens[match[i]].end, "");
                            st.removed.push_back(std::make_pair(t.begin, st.tokens[match[i]].end));
                            report_.removed.push_back(st.name + ": animations of removed elements");
                            nst.leave();
                            i = match[i];
                            continue;
                        }
                    }
                    if (!restoring_ && st.role == "workbook" && local == "absPath")
                    {
                        // The folder the workbook was last saved in.
                        st.edits.replace(t.begin, element_end(i), "");
                        st.removed.push_back(std::make_pair(t.begin, element_end(i)));
                        report_.removed.push_back(st.name + ": absPath");
                        if (t.kind == xml_token::start_tag && match[i] != std::string::npos) i = match[i];
                        nst.leave();
                        continue;
                    }
                    if (!restoring_ && st.role == "pml" && ns::a(e_ns) && (local == "blipFill" || local == "buBlip"))
                    {
                        const std::size_t end_tok = match[i];
                        if (end_tok != std::string::npos)
                        {
                            st.edits.replace(t.begin, st.tokens[end_tok].end,
                                             local == "blipFill" ? "<a:noFill/>" : "<a:buNone/>");
                            st.removed.push_back(std::make_pair(t.begin, st.tokens[end_tok].end));
                            report_.removed.push_back(st.name + ": " + local);
                            nst.leave();
                            i = end_tok;
                            continue;
                        }
                    }
                    if (!restoring_ && (st.role == "sheet" || st.role == "chartsheet") && ns::s(e_ns) &&
                        local == "pageSetup")
                    {
                        // The printer settings relationship is removed, so is the reference to it.
                        for (const xml_attribute& a : t.attributes)
                        {
                            if (xml_local_name(a.name) == "id" && ns::r(nst.resolve(a.name, true)))
                            {
                                std::size_t b = a.value_begin;
                                while (b > t.begin && !impl::xml_space(st.xml[b - 1])) --b;
                                if (b > t.begin) st.edits.replace(b - 1, a.value_end + 1, "");
                            }
                        }
                    }

                    handle_attributes(st, t, e_ns, local);

                    if (st.role == "wml")
                    {
                        if (ns::w(e_ns) && local == "tbl" && t.kind == xml_token::start_tag)
                            tables.push_back(table_state{ next_table++, -1, -1 });
                        else if (ns::w(e_ns) && local == "tr" && !tables.empty())
                        {
                            ++tables.back().row;
                            tables.back().col = -1;
                        }
                        else if (ns::w(e_ns) && local == "tc" && !tables.empty()) ++tables.back().col;
                        if (ns::w(e_ns) && local == "p" && t.kind == xml_token::start_tag)
                        {
                            para p;
                            if (!tables.empty())
                            {
                                p.table = tables.back().id;
                                p.row = tables.back().row;
                                p.col = tables.back().col;
                            }
                            paras.push_back(p);
                        }
                        const bool in_run = !stack.empty() && ns::w(stack.back().ns) && stack.back().local == "r";
                        if (ns::w(e_ns) && !paras.empty() &&
                            (((local == "hyperlink" || local == "fldSimple" || local == "ins" || local == "del" ||
                               local == "moveFrom" || local == "moveTo" || local == "sdt" || local == "smartTag" ||
                               local == "customXml" || local == "bdo" || local == "dir") &&
                              t.kind == xml_token::start_tag) ||
                             local == "fldChar"))
                            cut_here(paras.back());
                        if (ns::w(e_ns) &&
                            (local == "t" || local == "delText" || local == "instrText" || local == "delInstrText") &&
                            t.kind == xml_token::start_tag && !paras.empty())
                        {
                            const int stream = local == "t" ? 0 : (local == "delText" ? 1 : 2);
                            const int node = add_text_node(st, t, i, match[i], local == "t");
                            add_piece(unit_for(paras.back(), stream), node,
                                      st.nodes[static_cast<std::size_t>(node)].text);
                        }
                        else if (in_run && ns::w(e_ns) && !paras.empty())
                        {
                            // Breaks, also written with an end tag.
                            if (local == "tab") add_piece(unit_for(paras.back(), 0), -1, "\t", t.begin, element_end(i));
                            else if (local == "br" || local == "cr")
                                add_piece(unit_for(paras.back(), 0), -1, "\n", t.begin, element_end(i));
                            else if (local == "noBreakHyphen")
                                add_piece(unit_for(paras.back(), 0), -1, "-", t.begin, element_end(i));
                            else if (local == "sym") add_piece(unit_for(paras.back(), 0), -1, " ");
                        }
                    }
                    else if (st.role == "pml")
                    {
                        if (ns::a(e_ns) && local == "tbl" && t.kind == xml_token::start_tag)
                            tables.push_back(table_state{ next_table++, -1, -1 });
                        else if (ns::a(e_ns) && local == "tr" && !tables.empty())
                        {
                            ++tables.back().row;
                            tables.back().col = -1;
                        }
                        else if (ns::a(e_ns) && local == "tc" && !tables.empty()) ++tables.back().col;
                        if (ns::a(e_ns) && local == "p" && t.kind == xml_token::start_tag)
                        {
                            para p;
                            if (!tables.empty())
                            {
                                p.table = tables.back().id;
                                p.row = tables.back().row;
                                p.col = tables.back().col;
                            }
                            paras.push_back(p);
                        }
                        if (ns::a(e_ns) && local == "fld" && t.kind == xml_token::start_tag && !paras.empty())
                            cut_here(paras.back());
                        if (ns::a(e_ns) && local == "t" && t.kind == xml_token::start_tag && !paras.empty())
                        {
                            const int node = add_text_node(st, t, i, match[i], false);
                            add_piece(unit_for(paras.back(), 0), node,
                                      st.nodes[static_cast<std::size_t>(node)].text);
                        }
                        // A line break, written <a:br/> or <a:br><a:rPr/></a:br>.
                        else if (ns::a(e_ns) && local == "br" && !paras.empty())
                            add_piece(unit_for(paras.back(), 0), -1, "\n", t.begin, element_end(i));
                    }
                    else if (st.role == "pml_comments")
                    {
                        if (ns::p(e_ns) && local == "text" && t.kind == xml_token::start_tag)
                            text_element_unit(st, t, i, match[i], pii_type::none);
                    }
                    else if (st.role == "sst")
                    {
                        if (ns::s(e_ns) && local == "si")
                        {
                            if (t.kind == xml_token::start_tag)
                            {
                                unit u;
                                u.shared_index = next_si;
                                st.units.push_back(u);
                                si_unit = static_cast<int>(st.units.size() - 1);
                                phonetic_unit = -1;
                            }
                            ++next_si;
                        }
                        else if (ns::s(e_ns) && local == "rPh" && t.kind == xml_token::start_tag)
                            in_phonetic = true;
                        else if (ns::s(e_ns) && local == "t" && t.kind == xml_token::start_tag && si_unit >= 0)
                        {
                            const int node = add_text_node(st, t, i, match[i], false);
                            if (in_phonetic)
                            {
                                if (phonetic_unit < 0)
                                {
                                    st.units.push_back(unit());
                                    phonetic_unit = static_cast<int>(st.units.size() - 1);
                                }
                                add_piece(st.units[static_cast<std::size_t>(phonetic_unit)], node,
                                          st.nodes[static_cast<std::size_t>(node)].text);
                            }
                            else
                            {
                                add_piece(st.units[static_cast<std::size_t>(si_unit)], node,
                                          st.nodes[static_cast<std::size_t>(node)].text);
                            }
                        }
                    }
                    else if (st.role == "sheet" || st.role == "chartsheet")
                    {
                        if (ns::s(e_ns) && local == "c" && t.kind == xml_token::start_tag)
                            read_cell(st, t, i, match[i]);
                        else if (ns::s(e_ns) &&
                                 (local == "oddHeader" || local == "oddFooter" || local == "evenHeader" ||
                                  local == "evenFooter" || local == "firstHeader" || local == "firstFooter") &&
                                 t.kind == xml_token::start_tag)
                            text_element_unit(st, t, i, match[i], pii_type::none);
                        // Cell formulas, validation lists, conditional formats and their extensions.
                        else if ((local == "f" || local == "formula" || local == "formula1" || local == "formula2") &&
                                 t.kind == xml_token::start_tag)
                            formula_node(st, t, i, match[i]);
                    }
                    else if (st.role == "workbook")
                    {
                        if (ns::s(e_ns) && local == "definedName" && t.kind == xml_token::start_tag)
                            formula_node(st, t, i, match[i]);
                    }
                    else if (st.role == "xlsx_comments")
                    {
                        if (ns::s(e_ns) && local == "author" && t.kind == xml_token::start_tag)
                            text_element_unit(st, t, i, match[i], pii_type::person);
                        else if (ns::s(e_ns) && local == "comment" && t.kind == xml_token::start_tag)
                        {
                            st.units.push_back(unit());
                            comment_unit = static_cast<int>(st.units.size() - 1);
                        }
                        else if (ns::s(e_ns) && local == "rPh" && t.kind == xml_token::start_tag)
                            in_phonetic = true;
                        else if (ns::s(e_ns) && local == "t" && t.kind == xml_token::start_tag && comment_unit >= 0 &&
                                 !in_phonetic)
                        {
                            const int node = add_text_node(st, t, i, match[i], false);
                            add_piece(st.units[static_cast<std::size_t>(comment_unit)], node,
                                      st.nodes[static_cast<std::size_t>(node)].text);
                        }
                    }
                    else if (st.role == "threaded")
                    {
                        if (local == "text" && t.kind == xml_token::start_tag)
                            text_element_unit(st, t, i, match[i], pii_type::none);
                    }
                    else if (st.role == "core")
                    {
                        if (t.kind == xml_token::start_tag)
                        {
                            if ((ns::dc(e_ns) && local == "creator") || (ns::core(e_ns) && local == "lastModifiedBy"))
                                text_element_unit(st, t, i, match[i], pii_type::person);
                            else if ((ns::dc(e_ns) &&
                                      (local == "title" || local == "subject" || local == "description")) ||
                                     (ns::core(e_ns) &&
                                      (local == "keywords" || local == "category" || local == "contentStatus")))
                                text_element_unit(st, t, i, match[i], pii_type::none);
                            else if ((ns::dcterms(e_ns) && (local == "created" || local == "modified")) ||
                                     (ns::core(e_ns) && local == "lastPrinted"))
                                date_element(st, i, match[i]);
                        }
                    }
                    else if (st.role == "app")
                    {
                        if (t.kind == xml_token::start_tag)
                        {
                            if (local == "Company") text_element_unit(st, t, i, match[i], pii_type::organization);
                            else if (local == "Manager") text_element_unit(st, t, i, match[i], pii_type::person);
                            else if (local == "Template" || local == "HyperlinkBase" ||
                                     (ns::vt(e_ns) && (local == "lpstr" || local == "lpwstr" || local == "bstr")))
                                text_element_unit(st, t, i, match[i], pii_type::none);
                        }
                    }
                    else if (st.role == "custom")
                    {
                        if (local == "property" && t.kind == xml_token::start_tag)
                        {
                            bool marker = false;
                            for (const xml_attribute& a : t.attributes)
                                if (a.name == "name" && a.value_end - a.value_begin >= 12 &&
                                    st.xml.compare(a.value_begin, 12, "dlib.privacy") == 0)
                                    marker = true;
                            if (marker && match[i] != std::string::npos) { nst.leave(); i = match[i]; continue; }
                        }
                        if (ns::vt(e_ns) && (local == "lpwstr" || local == "lpstr" || local == "bstr") &&
                            t.kind == xml_token::start_tag)
                            text_element_unit(st, t, i, match[i], pii_type::none);
                    }
                    else if (st.role == "attributes")
                    {
                        if (local == "cacheField")
                        {
                            pivot_field_type_ = pii_type::none;
                            for (const xml_attribute& a : t.attributes)
                                if (a.name == "name" && detector_)
                                    pivot_field_type_ = detector_->label_type(
                                        xml_unescape(st.xml.substr(a.value_begin, a.value_end - a.value_begin)));
                        }
                    }
                    else if (st.role == "generic")
                    {
                        in_text_element = true;
                    }

                    if (t.kind == xml_token::empty_tag)
                        nst.leave();
                    else
                        stack.push_back(frame{ e_ns, local });
                }
                else if (t.kind == xml_token::end_tag)
                {
                    if (!stack.empty())
                    {
                        const frame f = stack.back();
                        stack.pop_back();
                        if (st.role == "wml")
                        {
                            if (ns::w(f.ns) && f.local == "p" && !paras.empty()) paras.pop_back();
                            else if (ns::w(f.ns) && f.local == "tbl" && !tables.empty()) tables.pop_back();
                            else if (ns::w(f.ns) && !paras.empty() &&
                                     (f.local == "hyperlink" || f.local == "fldSimple" || f.local == "ins" ||
                                      f.local == "del" || f.local == "moveFrom" || f.local == "moveTo" ||
                                      f.local == "sdt" || f.local == "smartTag" || f.local == "customXml" ||
                                      f.local == "bdo" || f.local == "dir"))
                                cut_here(paras.back());
                        }
                        else if (st.role == "pml")
                        {
                            if (ns::a(f.ns) && f.local == "p" && !paras.empty()) paras.pop_back();
                            else if (ns::a(f.ns) && f.local == "tbl" && !tables.empty()) tables.pop_back();
                            else if (ns::a(f.ns) && f.local == "fld" && !paras.empty()) cut_here(paras.back());
                        }
                        else if (st.role == "sst" || st.role == "xlsx_comments")
                        {
                            if (f.local == "si") si_unit = -1;
                            if (f.local == "rPh") in_phonetic = false;
                            if (f.local == "comment") comment_unit = -1;
                        }
                        else if (st.role == "attributes" && f.local == "cacheField")
                        {
                            pivot_field_type_ = pii_type::none;
                        }
                    }
                    nst.leave();
                }
                else if (t.kind == xml_token::text && st.role == "generic" && in_text_element)
                {
                    const std::string raw = st.xml.substr(t.content_begin, t.content_end - t.content_begin);
                    if (raw.find_first_not_of(" \t\r\n") != std::string::npos && i > 0 && i + 1 < st.tokens.size() &&
                        st.tokens[i - 1].kind == xml_token::start_tag && st.tokens[i + 1].kind == xml_token::end_tag)
                    {
                        // Chart references and external defined names are formulas.
                        const std::string l = xml_local_name(st.tokens[i - 1].name);
                        if (l == "f" || l == "definedName") formula_node(st, st.tokens[i - 1], i - 1, i + 1);
                        else text_element_unit(st, st.tokens[i - 1], i - 1, i + 1, pii_type::none);
                    }
                }
            }
        }

        void text_element_unit(part_state& st, const xml_token& start, std::size_t start_index, std::size_t end_index,
                               pii_type forced)
        {
            if (end_index == std::string::npos) return;
            const int node = add_text_node(st, start, start_index, end_index, false);
            if (st.nodes[static_cast<std::size_t>(node)].text.empty()) return;
            unit u;
            u.ctx.forced = forced;
            add_piece(u, node, st.nodes[static_cast<std::size_t>(node)].text);
            st.units.push_back(u);
        }

        void date_element(part_state& st, std::size_t start_index, std::size_t end_index)
        {
            if (!shift_dates_ || end_index == std::string::npos || end_index != start_index + 2) return;
            const xml_token& t = st.tokens[start_index + 1];
            if (t.kind != xml_token::text) return;
            std::string v = st.xml.substr(t.content_begin, t.content_end - t.content_begin);
            if (privacy_impl::shift_timestamp(v, shift_))
            {
                st.edits.replace(t.content_begin, t.content_end, v);
                ++report_.dates_shifted;
            }
        }

    // ------------------------------------------------------------------------------------
    // Spreadsheets: shared strings, styles, cells and column typing.

        void load_spreadsheet_tables(ooxml_package& pkg)
        {
            shared_texts_.clear();
            date_styles_.clear();
            if (pkg.kind() != ooxml_kind::spreadsheet) return;
            const std::string wb = pkg.main_part();
            for (const ooxml_relationship& r : pkg.relationships(wb))
            {
                const std::string target = pkg.target_part(wb, r.target);
                if (!pkg.has_part(target)) continue;
                if (ooxml_package::ends_with(r.type, "/sharedStrings")) read_shared_texts(pkg.part(target));
                else if (ooxml_package::ends_with(r.type, "/styles")) read_date_styles(pkg.part(target));
            }
        }

        void read_shared_texts(const std::string& xml)
        {
            const std::vector<xml_token> tokens = xml_tokenize(xml);
            bool in_si = false, in_t = false, in_rph = false;
            std::string cur;
            for (const xml_token& t : tokens)
            {
                const std::string l = xml_local_name(t.name);
                if (t.kind == xml_token::start_tag && l == "si") { in_si = true; cur.clear(); }
                else if (t.kind == xml_token::end_tag && l == "si") { in_si = false; shared_texts_.push_back(cur); }
                else if (t.kind == xml_token::empty_tag && l == "si") shared_texts_.push_back("");
                else if (t.kind == xml_token::start_tag && l == "rPh") in_rph = true;
                else if (t.kind == xml_token::end_tag && l == "rPh") in_rph = false;
                else if (t.kind == xml_token::start_tag && l == "t") in_t = true;
                else if (t.kind == xml_token::end_tag && l == "t") in_t = false;
                else if (t.kind == xml_token::text && in_si && in_t && !in_rph)
                    cur += xml_unescape(xml.substr(t.content_begin, t.content_end - t.content_begin));
            }
        }

        void read_date_styles(const std::string& xml)
        {
            const std::vector<xml_token> tokens = xml_tokenize(xml);
            std::map<int, std::string> custom;
            bool in_xfs = false;
            int index = 0;
            for (const xml_token& t : tokens)
            {
                const std::string l = xml_local_name(t.name);
                auto attr = [&](const char* name) {
                    for (const xml_attribute& a : t.attributes)
                        if (a.name == name) return xml_unescape(xml.substr(a.value_begin, a.value_end - a.value_begin));
                    return std::string();
                };
                if ((t.kind == xml_token::start_tag || t.kind == xml_token::empty_tag) && l == "numFmt")
                {
                    const long id = privacy_impl::small_number(attr("numFmtId"));
                    if (id >= 0) custom[static_cast<int>(id)] = attr("formatCode");
                }
                else if (t.kind == xml_token::start_tag && l == "cellXfs") { in_xfs = true; index = 0; }
                else if (t.kind == xml_token::end_tag && l == "cellXfs") in_xfs = false;
                else if (in_xfs && (t.kind == xml_token::start_tag || t.kind == xml_token::empty_tag) && l == "xf")
                {
                    const int id = static_cast<int>(privacy_impl::small_number(attr("numFmtId")));
                    const auto it = custom.find(id);
                    if (privacy_impl::is_date_format(id, it == custom.end() ? std::string() : it->second))
                        date_styles_.insert(index);
                    ++index;
                }
            }
        }

        void read_cell(part_state& st, const xml_token& t, std::size_t start_index, std::size_t end_index)
        {
            if (end_index == std::string::npos) return;
            cell_info c;
            c.type = "n";
            c.style = 0;
            c.formula = false;
            c.has_v = false;
            c.inline_unit = -1;
            c.c_begin = t.begin;
            c.c_end = st.tokens[end_index].end;
            std::string ref;
            for (const xml_attribute& a : t.attributes)
            {
                const std::string v = st.xml.substr(a.value_begin, a.value_end - a.value_begin);
                c.attrs.push_back(std::make_pair(a.name, v));
                if (a.name == "r") ref = v;
                else if (a.name == "t") c.type = v;
                else if (a.name == "s") c.style = static_cast<int>(privacy_impl::small_number(v));
            }
            c.row = privacy_impl::row_index(ref);
            c.col = privacy_impl::column_index(ref);
            bool in_is = false;
            int inline_unit = -1;
            for (std::size_t k = start_index + 1; k < end_index; ++k)
            {
                const xml_token& x = st.tokens[k];
                const std::string l = xml_local_name(x.name);
                if (x.kind == xml_token::start_tag && l == "f") c.formula = true;
                else if (x.kind == xml_token::empty_tag && l == "f") c.formula = true;
                else if (x.kind == xml_token::start_tag && l == "is") in_is = true;
                else if (x.kind == xml_token::end_tag && l == "is") in_is = false;
                else if (x.kind == xml_token::start_tag && l == "v" && k + 1 < end_index)
                {
                    c.has_v = true;
                    if (st.tokens[k + 1].kind == xml_token::text)
                    {
                        c.v_begin = st.tokens[k + 1].content_begin;
                        c.v_end = st.tokens[k + 1].content_end;
                        c.v_text = xml_unescape(st.xml.substr(c.v_begin, c.v_end - c.v_begin));
                    }
                    else
                    {
                        c.v_begin = c.v_end = x.end;
                    }
                    if (c.type == "str")
                    {
                        std::size_t end_v = k + 1;
                        while (end_v < end_index &&
                               !(st.tokens[end_v].kind == xml_token::end_tag &&
                                 xml_local_name(st.tokens[end_v].name) == "v"))
                            ++end_v;
                        const int node = add_text_node(st, x, k, end_v, false);
                        unit u;
                        add_piece(u, node, st.nodes[static_cast<std::size_t>(node)].text);
                        st.units.push_back(u);
                        c.inline_unit = static_cast<int>(st.units.size() - 1);
                    }
                }
                else if (x.kind == xml_token::start_tag && l == "t" && in_is)
                {
                    std::size_t end_t = k + 1;
                    while (
                        end_t < end_index &&
                        !(st.tokens[end_t].kind == xml_token::end_tag && xml_local_name(st.tokens[end_t].name) == "t"))
                        ++end_t;
                    const int node = add_text_node(st, x, k, end_t, false);
                    if (inline_unit < 0)
                    {
                        st.units.push_back(unit());
                        inline_unit = static_cast<int>(st.units.size() - 1);
                    }
                    add_piece(st.units[static_cast<std::size_t>(inline_unit)], node,
                              st.nodes[static_cast<std::size_t>(node)].text);
                    c.inline_unit = inline_unit;
                }
            }
            st.cells.push_back(c);
        }

        void assign_shared_string_types(std::vector<part_state>& parts)
        {
            if (restoring_ || !detector_) return;
            shared_types_.assign(shared_texts_.size(), pii_type::none);
            for (part_state& st : parts)
            {
                if (st.role != "sheet") continue;
                // Header row: the first of the top rows holding at least two field labels.
                std::map<int, std::map<int, std::string>> rows;
                for (const cell_info& c : st.cells)
                {
                    if (c.row > 12) continue;
                    std::string text;
                    if (c.type == "s" && c.has_v)
                    {
                        const std::size_t idx = static_cast<std::size_t>(privacy_impl::small_number(c.v_text));
                        if (idx < shared_texts_.size()) text = shared_texts_[idx];
                    }
                    else if (c.inline_unit >= 0)
                    {
                        text = st.units[static_cast<std::size_t>(c.inline_unit)].text;
                    }
                    if (!text.empty()) rows[c.row][c.col] = text;
                }
                for (const auto& r : rows)
                {
                    std::map<int, pii_type> types;
                    for (const auto& cell : r.second)
                    {
                        const pii_type t = detector_->label_type(cell.second);
                        if (t != pii_type::none) types[cell.first] = t;
                    }
                    if (types.size() >= 2 || (types.size() == 1 && r.second.size() == 1))
                    {
                        st.header_row = r.first;
                        st.column_types = types;
                        break;
                    }
                }
                if (st.header_row < 0) continue;
                for (const cell_info& c : st.cells)
                {
                    if (c.row <= st.header_row) continue;
                    const auto it = st.column_types.find(c.col);
                    if (it == st.column_types.end()) continue;
                    if (c.type == "s" && c.has_v)
                    {
                        const std::size_t idx = static_cast<std::size_t>(privacy_impl::small_number(c.v_text));
                        if (idx < shared_types_.size() && shared_types_[idx] == pii_type::none)
                            shared_types_[idx] = it->second;
                    }
                    else if (c.inline_unit >= 0)
                    {
                        st.units[static_cast<std::size_t>(c.inline_unit)].ctx.forced = it->second;
                    }
                }
            }
            for (part_state& st : parts)
            {
                if (st.role != "sst") continue;
                for (unit& u : st.units)
                {
                    if (u.shared_index < 0) continue;
                    const std::size_t idx = static_cast<std::size_t>(u.shared_index);
                    if (idx < shared_types_.size() && shared_types_[idx] != pii_type::none &&
                        forceable(shared_types_[idx], u.text))
                        u.ctx.forced = shared_types_[idx];
                }
            }
        }

        static bool forceable(pii_type t, const std::string& text)
        {
            int alnum = 0;
            for (char c : text)
                if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    static_cast<unsigned char>(c) >= 0x80)
                    ++alnum;
            if (alnum < 2) return false;
            return t != pii_type::url && t != pii_type::none;
        }

        void assign_table_contexts(std::vector<part_state>& parts)
        {
            if (restoring_ || !detector_) return;
            for (part_state& st : parts)
            {
                if (st.role != "wml" && st.role != "pml") continue;
                std::map<std::pair<int, std::pair<int, int>>, std::string> cell_text;
                for (const unit& u : st.units)
                    if (u.table >= 0 && u.main_stream)
                        cell_text[std::make_pair(u.table, std::make_pair(u.row, u.col))] += u.text + " ";
                for (unit& u : st.units)
                {
                    if (u.table < 0 || !u.main_stream || u.ctx.forced != pii_type::none) continue;
                    pii_type label = pii_type::none;
                    if (u.col >= 1)
                    {
                        const auto left = cell_text.find(std::make_pair(u.table, std::make_pair(u.row, u.col - 1)));
                        if (left != cell_text.end()) label = detector_->label_type(left->second);
                    }
                    if (label == pii_type::none && u.row >= 1)
                    {
                        const auto head = cell_text.find(std::make_pair(u.table, std::make_pair(0, u.col)));
                        if (head != cell_text.end()) label = detector_->label_type(head->second);
                    }
                    if (label != pii_type::none && detector_->label_type(u.text) == pii_type::none)
                        u.ctx.label = label;
                }
            }
        }

        void transform_cells(part_state& st)
        {
            const long shift = shift_;
            for (const cell_info& c : st.cells)
            {
                if (inside_removed(st, c.c_begin, c.c_end)) continue;
                const bool numeric = c.type == "n";
                // Identifiers stored as numbers in typed columns become inline string tags.
                if (!restoring_ && numeric && !c.formula && c.has_v && st.header_row >= 0 && c.row > st.header_row)
                {
                    const auto it = st.column_types.find(c.col);
                    if (it != st.column_types.end() &&
                        (it->second == pii_type::phone || it->second == pii_type::nir ||
                         it->second == pii_type::identifier || it->second == pii_type::siret ||
                         it->second == pii_type::iban || it->second == pii_type::payment_card) &&
                        !c.v_text.empty() && c.v_text.find_first_not_of("0123456789") == std::string::npos)
                    {
                        const std::string tag = p_.tag(it->second, c.v_text);
                        report_.count(it->second);
                        std::string cell = "<c";
                        for (const auto& a : c.attrs)
                            if (a.first != "t") cell += " " + a.first + "=\"" + a.second + "\"";
                        cell += " t=\"inlineStr\"><is><t>" + xml_escape_text(tag) + "</t></is></c>";
                        st.edits.replace(c.c_begin, c.c_end, cell);
                        continue;
                    }
                }
                if (!shift_dates_ || !c.has_v) continue;
                if (numeric && date_styles_.count(c.style))
                {
                    std::string v = c.v_text;
                    if (privacy_impl::shift_serial_text(v, shift))
                    {
                        st.edits.replace(c.v_begin, c.v_end, v);
                        ++report_.dates_shifted;
                    }
                }
                else if (c.type == "d")
                {
                    std::string v = c.v_text;
                    if (privacy_impl::shift_timestamp(v, shift))
                    {
                        st.edits.replace(c.v_begin, c.v_end, v);
                        ++report_.dates_shifted;
                    }
                }
            }
        }

    // ------------------------------------------------------------------------------------
    // Text transformation and write-back.

        void transform_unit(part_state& st, unit& u)
        {
            if (u.text.empty()) return;
            // Text inside removed elements goes with them.
            bool kept = false;
            for (const piece& pc : u.pieces)
                if (pc.node >= 0)
                {
                    const text_node& n = st.nodes[static_cast<std::size_t>(pc.node)];
                    if (!inside_removed(st, n.raw_begin, n.raw_end)) kept = true;
                }
            if (!kept) return;
            ++report_.text_units;
            struct span { std::size_t b, e; std::string rep; };
            std::vector<span> edits;
            if (restoring_)
            {
                pseudonymizer::restore_report rr;
                for (const auto& o : p_.find_tags(u.text, &rr))
                    edits.push_back(span{ o.begin, o.end, o.original });
                report_.tags_restored += rr.restored;
                report_.tags_rejected += rr.rejected;
            }
            else
            {
                for (const pii_entity& e : detector_->detect(u.text, u.ctx))
                {
                    if (p_.keeps_in_clear(e.type)) continue;
                    report_.count(e.type);
                    /* A value is cut where a hyperlink, field, revision or content control
                       begins or ends, so that each part keeps its text in its own container. */
                    std::size_t b = e.begin;
                    for (std::size_t k = 0; k <= u.cuts.size(); ++k)
                    {
                        const std::size_t cut = k < u.cuts.size() ? u.cuts[k] : e.end;
                        if (cut <= b || (k < u.cuts.size() && cut >= e.end)) continue;
                        std::size_t pb = b, pe = cut;
                        while (pb < pe && u.text[pb] == ' ') ++pb;
                        while (pe > pb && u.text[pe - 1] == ' ') --pe;
                        if (pe > pb) edits.push_back(span{ pb, pe, p_.tag(e.type, u.text.substr(pb, pe - pb)) });
                        b = cut;
                        if (cut >= e.end) break;
                    }
                }
            }
            if (edits.empty()) return;

            /* Each piece keeps its characters outside the edits; the replacement of an edit
               goes to the real node holding its first character. */
            std::vector<std::string> out(u.pieces.size());
            std::vector<bool> touched(u.pieces.size(), false);
            std::size_t ei = 0;
            std::vector<bool> placed(edits.size(), false);
            for (std::size_t k = 0; k < u.pieces.size(); ++k)
            {
                const piece& pc = u.pieces[k];
                std::size_t pos = pc.ub;
                while (pos < pc.ue)
                {
                    while (ei < edits.size() && edits[ei].e <= pos) ++ei;
                    if (ei < edits.size() && edits[ei].b <= pos)
                    {
                        if (!placed[ei])
                        {
                            std::size_t target = k;
                            while (target < u.pieces.size() && u.pieces[target].node < 0) ++target;
                            if (target == u.pieces.size())
                            {
                                target = k;
                                while (target > 0 && u.pieces[target].node < 0) --target;
                            }
                            out[target] += edits[ei].rep;
                            touched[target] = true;
                            placed[ei] = true;
                        }
                        touched[k] = true;
                        pos = std::min(edits[ei].e, pc.ue);
                        continue;
                    }
                    const std::size_t stop = (ei < edits.size()) ? std::min(edits[ei].b, pc.ue) : pc.ue;
                    out[k] += u.text.substr(pos, stop - pos);
                    pos = stop;
                }
            }
            for (std::size_t k = 0; k < u.pieces.size(); ++k)
            {
                const piece& pc = u.pieces[k];
                if (!touched[k]) continue;
                if (pc.node < 0)
                {
                    if (out[k] != u.text.substr(pc.ub, pc.ue - pc.ub) && pc.xml_begin != std::string::npos)
                        st.edits.replace(pc.xml_begin, pc.xml_end, "");
                    continue;
                }
                text_node& n = st.nodes[static_cast<std::size_t>(pc.node)];
                n.changed = true;
                n.new_text = out[k];
            }
        }

    // ------------------------------------------------------------------------------------
    // Formulas and sheet names.

        static bool sheet_name_char(char c)
        {
            return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' ||
                   static_cast<unsigned char>(c) >= 0x80;
        }

        static std::string fold_ascii(std::string s)
        {
            for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
            return s;
        }

        // The new name of a sheet for the current operation, or the name itself.
        std::string renamed_sheet(const std::string& name) const
        {
            for (const auto& r : sheet_names_)
                if (fold_ascii(r.first) == fold_ascii(name)) return r.second;
            return name;
        }

        // A sheet reference before '!': one name, or a range of sheets "First:Last".
        std::string renamed_reference(const std::string& ref) const
        {
            if (!ref.empty() && ref[0] == '[') return ref;      // another workbook
            const std::size_t colon = ref.find(':');
            if (colon == std::string::npos) return renamed_sheet(ref);
            return renamed_sheet(ref.substr(0, colon)) + ":" + renamed_sheet(ref.substr(colon + 1));
        }

        static std::string quoted_sheet(const std::string& name)
        {
            bool plain = !name.empty() && !(name[0] >= '0' && name[0] <= '9');
            std::size_t letters = 0;
            while (letters < name.size() && ((name[letters] >= 'A' && name[letters] <= 'Z') ||
                                              (name[letters] >= 'a' && name[letters] <= 'z')))
                ++letters;
            // Names that read as a cell reference ("AB12") are quoted too.
            if (letters <= 3 && letters < name.size() && name.find_first_not_of("0123456789", letters) == std::string::npos)
                plain = false;
            for (char c : name) if (!sheet_name_char(c)) plain = false;
            if (plain) return name;
            std::string q = "'";
            for (char c : name) { if (c == '\'') q += "''"; else q.push_back(c); }
            return q + "'";
        }

        /* A spreadsheet formula with its string literals anonymized or restored and its
           references to renamed sheets rewritten. */
        std::string transform_formula(const std::string& f)
        {
            std::string out;
            std::size_t i = 0;
            while (i < f.size())
            {
                const char c = f[i];
                if (c == '"' || c == '\'')
                {
                    std::size_t j = i + 1;
                    std::string inner;
                    while (j < f.size())
                    {
                        if (f[j] == c)
                        {
                            if (j + 1 < f.size() && f[j + 1] == c) { inner.push_back(c); j += 2; continue; }
                            break;
                        }
                        inner.push_back(f[j++]);
                    }
                    const bool closed = j < f.size();
                    if (c == '\'' && closed && j + 1 < f.size() && f[j + 1] == '!')
                    {
                        out += quoted_sheet(renamed_reference(inner)) + "!";
                        i = j + 2;
                        continue;
                    }
                    std::string value = inner;
                    if (c == '"')
                    {
                        if (restoring_)
                        {
                            pseudonymizer::restore_report rr;
                            value = p_.restore(inner, &rr);
                            report_.tags_restored += rr.restored;
                            report_.tags_rejected += rr.rejected;
                        }
                        else
                        {
                            std::vector<pii_entity> found;
                            for (const pii_entity& e : detector_->detect(inner))
                                if (!p_.keeps_in_clear(e.type)) { found.push_back(e); report_.count(e.type); }
                            value = p_.apply(inner, found);
                        }
                    }
                    out.push_back(c);
                    for (char v : value) { if (v == c) out.push_back(c); out.push_back(v); }
                    if (closed) out.push_back(c);
                    i = closed ? j + 1 : j;
                    continue;
                }
                if (sheet_name_char(c) && (i == 0 || !sheet_name_char(f[i - 1])))
                {
                    std::size_t j = i;
                    while (j < f.size() && sheet_name_char(f[j])) ++j;
                    if (j < f.size() && f[j] == '!')
                    {
                        out += quoted_sheet(renamed_reference(f.substr(i, j - i))) + "!";
                        i = j + 1;
                        continue;
                    }
                    out.append(f, i, j - i);
                    i = j;
                    continue;
                }
                out.push_back(c);
                ++i;
            }
            return out;
        }

        void transform_special_nodes(part_state& st)
        {
            for (text_node& n : st.nodes)
            {
                if (n.kind == 0 || n.text.empty()) continue;
                const std::string v = n.kind == 2 ? renamed_sheet(n.text) : transform_formula(n.text);
                if (v != n.text)
                {
                    n.changed = true;
                    n.new_text = v;
                }
            }
        }

        /* Sheets whose name holds personal data are renamed "SheetN"; the original name is
           kept as a tag in the custom properties, for restoration. */
        void rename_sheets(const std::vector<part_state>& parts)
        {
            std::vector<std::string> names;
            for (const part_state& st : parts)
                if (st.role == "workbook")
                    for (const text_node& n : st.nodes)
                        if (n.kind == 2 && !n.text.empty()) names.push_back(n.text);
            std::set<std::string> taken;
            for (const std::string& n : names) taken.insert(fold_ascii(n));
            int index = 0;
            for (const std::string& n : names)
            {
                ++index;
                const std::vector<pii_entity> found = detector_->detect(n);
                pii_type type = pii_type::none;
                for (const pii_entity& e : found)
                    if (!p_.keeps_in_clear(e.type)) { type = e.type; break; }
                if (type == pii_type::none || sheet_names_.count(n)) continue;
                std::string fresh = "Sheet" + std::to_string(index);
                for (int k = 2; taken.count(fold_ascii(fresh)); ++k)
                    fresh = "Sheet" + std::to_string(index) + "_" + std::to_string(k);
                taken.insert(fold_ascii(fresh));
                sheet_names_[n] = fresh;
                sheet_tags_[n] = p_.tag(type, n);
                report_.count(type);
                report_.note("sheet " + std::to_string(index) + " renamed " + fresh + ": its name holds personal data");
            }
        }

        // For restoration: the sheet names recorded by anonymize().
        void read_sheet_markers()
        {
            for (const auto& m : markers_)
            {
                if (m.first.compare(0, 18, "dlib.privacy.sheet") != 0) continue;
                const std::size_t eq = m.second.find('=');
                if (eq == std::string::npos) continue;
                pseudonymizer::restore_report rr;
                const std::string tag = m.second.substr(eq + 1);
                const std::string original = p_.restore(tag, &rr);
                if (rr.restored == 1 && original != tag) sheet_names_[m.second.substr(0, eq)] = original;
                else ++report_.tags_rejected;
            }
        }

        void finish_part(part_state& st)
        {
            for (const text_node& n : st.nodes)
            {
                if (!n.changed) continue;
                if (inside_removed(st, n.raw_begin, n.raw_end)) continue;
                if (n.attribute)
                {
                    st.edits.replace(n.raw_begin, n.raw_end, xml_escape_attribute(n.new_text));
                    continue;
                }
                st.edits.replace(n.raw_begin, n.raw_end, xml_escape_text(n.new_text));
                const bool edge_space = !n.new_text.empty() && (n.new_text.front() == ' ' || n.new_text.back() == ' ');
                if (n.wml_text && edge_space && !n.has_preserve && n.tag_end > 0 && n.tag_end != std::string::npos)
                    st.edits.insert(n.tag_end - 1, " xml:space=\"preserve\"");
            }
        }

        void process_relationships(ooxml_package& pkg)
        {
            for (const std::string& name : pkg.part_names())
            {
                if (!ooxml_package::is_rels(name)) continue;
                const std::string source = ooxml_package::rels_source(name);
                const std::string& xml = pkg.part(name);
                // Attribute values of the source part, among which relationship identifiers are referenced.
                std::set<std::string> referenced_ids;
                if (!source.empty() && pkg.has_part(source) &&
                    privacy_impl::xml_part(source, pkg.content_type(source)))
                {
                    const std::string& source_xml = pkg.part(source);
                    try
                    {
                        for (const xml_token& t : xml_tokenize(source_xml))
                            for (const xml_attribute& a : t.attributes)
                                if (a.value_end - a.value_begin <= 64)
                                    referenced_ids.insert(source_xml.substr(a.value_begin, a.value_end - a.value_begin));
                    }
                    catch (xml_error&) {}
                }
                std::vector<xml_token> tokens;
                try { tokens = xml_tokenize(xml); }
                catch (xml_error& e)
                {
                    throw error("ooxml_anonymizer: part " + name + " is not well-formed XML (" + e.info +
                                "); nothing was written");
                }
                xml_edit_list edits;
                for (const xml_token& t : tokens)
                {
                    if ((t.kind != xml_token::start_tag && t.kind != xml_token::empty_tag) ||
                        xml_local_name(t.name) != "Relationship")
                        continue;
                    std::string id, type;
                    bool external = false;
                    const xml_attribute* target = nullptr;
                    for (const xml_attribute& a : t.attributes)
                    {
                        const std::string v = xml_unescape(xml.substr(a.value_begin, a.value_end - a.value_begin));
                        if (a.name == "Id") id = v;
                        else if (a.name == "Type") type = v;
                        else if (a.name == "TargetMode") external = v == "External";
                        else if (a.name == "Target") target = &a;
                    }
                    if (!target) continue;
                    const std::string value =
                        xml_unescape(xml.substr(target->value_begin, target->value_end - target->value_begin));
                    if (!restoring_)
                    {
                        const bool referenced = referenced_ids.count(id) != 0;
                        const std::string target_ct =
                            external ? std::string() : pkg.content_type(pkg.target_part(source, value));
                        if ((privacy_impl::removable_relationship(type) &&
                             (!referenced || ooxml_package::ends_with(type, "/thumbnail") ||
                              ooxml_package::ends_with(type, "/printerSettings") ||
                              ooxml_package::ends_with(type, "/vbaProject") ||
                              ooxml_package::ends_with(type, "/aFChunk"))) ||
                            privacy_impl::removable_content_type(target_ct))
                        {
                            edits.replace(t.begin, t.end, "");
                            continue;
                        }
                        if (external)
                        {
                            std::vector<pii_entity> ents;
                            for (const pii_entity& e : detector_->detect(value))
                                if (!p_.keeps_in_clear(e.type)) ents.push_back(e);
                            if (ents.empty() && value.compare(0, 7, "mailto:") != 0 &&
                                value.find("://") == std::string::npos)
                            {
                                // A local file path or UNC share: the whole target is an identifier.
                                pii_entity e;
                                e.begin = 0; e.end = value.size(); e.type = pii_type::identifier; e.confidence = 0.5;
                                ents.push_back(e);
                            }
                            if (!ents.empty())
                            {
                                std::string out;
                                std::size_t pos = 0;
                                for (const pii_entity& e : ents)
                                {
                                    out += value.substr(pos, e.begin - pos);
                                    out += percent_encode_tag(p_.tag(e.type, value.substr(e.begin, e.end - e.begin)));
                                    pos = e.end;
                                    report_.count(e.type);
                                }
                                out += value.substr(pos);
                                edits.replace(target->value_begin, target->value_end, xml_escape_attribute(out));
                            }
                        }
                    }
                    else if (external)
                    {
                        pseudonymizer::restore_report rr;
                        const std::string back = p_.restore(value, &rr);
                        if (back != value)
                            edits.replace(target->value_begin, target->value_end, xml_escape_attribute(back));
                        report_.tags_restored += rr.restored;
                        report_.tags_rejected += rr.rejected;
                    }
                }
                if (!edits.empty())
                    pkg.set_part(name, edits.apply(xml));
            }
        }

    // ------------------------------------------------------------------------------------
    // Markers in the custom properties: "dlib.privacy" when dates were shifted,
    // "dlib.privacy.sheetN" for each renamed sheet, "dlib.privacy.created" when the custom
    // properties part was created for them.

        static const char* custom_rel_type()
        {
            return "http://schemas.openxmlformats.org/officeDocument/2006/relationships/custom-properties";
        }

        static const char* custom_part_type()
        {
            return "application/vnd.openxmlformats-officedocument.custom-properties+xml";
        }

        static std::string custom_part(const ooxml_package& pkg)
        {
            for (const ooxml_relationship& r : pkg.relationships(""))
                if (ooxml_package::ends_with(r.type, "/custom-properties") && !r.external)
                    return pkg.target_part("", r.target);
            return std::string();
        }

        // The marker properties, by name, with their text value.
        static std::map<std::string, std::string> read_markers(const ooxml_package& pkg)
        {
            std::map<std::string, std::string> out;
            const std::string part = custom_part(pkg);
            if (part.empty() || !pkg.has_part(part)) return out;
            const std::string& xml = pkg.part(part);
            std::vector<xml_token> tokens;
            try { tokens = xml_tokenize(xml); }
            catch (xml_error&) { return out; }
            std::string current;
            bool in_value = false;
            for (const xml_token& t : tokens)
            {
                const std::string l = xml_local_name(t.name);
                if (t.kind == xml_token::start_tag && l == "property")
                {
                    current.clear();
                    for (const xml_attribute& a : t.attributes)
                        if (a.name == "name") current = xml_unescape(xml.substr(a.value_begin, a.value_end - a.value_begin));
                    if (current.compare(0, 12, "dlib.privacy") != 0) current.clear();
                    else out[current];
                }
                else if (t.kind == xml_token::end_tag && l == "property") current.clear();
                else if (t.kind == xml_token::start_tag && !current.empty()) in_value = true;
                else if (t.kind == xml_token::end_tag) in_value = false;
                else if (t.kind == xml_token::text && in_value && !current.empty())
                    out[current] += xml_unescape(xml.substr(t.content_begin, t.content_end - t.content_begin));
            }
            return out;
        }

        static std::string marker_property(int pid, const std::string& name, const std::string& value)
        {
            return "<property fmtid=\"{D5CDD505-2E9C-101B-9397-08002B2CF9AE}\" pid=\"" + std::to_string(pid) +
                   "\" name=\"" + xml_escape_attribute(name) +
                   "\"><vt:lpwstr xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">" +
                   xml_escape_text(value) + "</vt:lpwstr></property>";
        }

        void write_markers(ooxml_package& pkg, std::map<std::string, std::string> markers)
        {
            std::string part = custom_part(pkg);
            if (part.empty() || !pkg.has_part(part))
            {
                part = "docProps/custom.xml";
                markers["dlib.privacy.created"] = "1";
                std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\r\n"
                                  "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/"
                                  "custom-properties\">";
                int pid = 1;
                for (const auto& m : markers) xml += marker_property(++pid, m.first, m.second);
                xml += "</Properties>";
                pkg.set_part(part, xml);
                pkg.set_override(part, custom_part_type());
                std::string rels = pkg.has_part("_rels/.rels") ? pkg.part("_rels/.rels") : std::string();
                const std::size_t close = rels.rfind("</Relationships>");
                if (close != std::string::npos)
                {
                    rels.insert(close, "<Relationship Id=\"rIdDlibPrivacy\" Type=\"" + std::string(custom_rel_type()) +
                                       "\" Target=\"docProps/custom.xml\"/>");
                    pkg.set_part("_rels/.rels", rels);
                }
                return;
            }
            std::string xml = pkg.part(part);
            std::vector<xml_token> tokens;
            try { tokens = xml_tokenize(xml); }
            catch (xml_error&) { return; }
            int pid = 1;
            const xml_token* root = nullptr;
            const xml_token* root_end = nullptr;
            for (const xml_token& t : tokens)
            {
                if ((t.kind == xml_token::start_tag || t.kind == xml_token::empty_tag) && !root) root = &t;
                if (t.kind == xml_token::end_tag) root_end = &t;
                for (const xml_attribute& a : t.attributes)
                    if (a.name == "pid")
                        pid = std::max<int>(pid, static_cast<int>(privacy_impl::small_number(
                                                     xml.substr(a.value_begin, a.value_end - a.value_begin))));
            }
            if (!root) return;
            std::string added;
            for (const auto& m : markers) added += marker_property(++pid, m.first, m.second);
            if (root->kind == xml_token::empty_tag)
            {
                // <Properties .../> becomes <Properties ...>markers</Properties>.
                std::size_t close = root->end - 2;
                while (close > root->begin && impl::xml_space(xml[close - 1])) --close;
                xml = xml.substr(0, close) + ">" + added + "</" + root->name + ">" + xml.substr(root->end);
            }
            else if (root_end)
            {
                xml.insert(root_end->begin, added);
            }
            pkg.set_part(part, xml);
        }

        void clear_markers(ooxml_package& pkg)
        {
            const std::string part = custom_part(pkg);
            if (part.empty() || !pkg.has_part(part)) return;
            if (markers_.count("dlib.privacy.created"))
            {
                pkg.remove_part(part);
                std::string rels = pkg.part("_rels/.rels");
                const std::size_t r = rels.find("Id=\"rIdDlibPrivacy\"");
                if (r != std::string::npos)
                {
                    const std::size_t b = rels.rfind('<', r);
                    const std::size_t e = rels.find("/>", r);
                    if (b != std::string::npos && e != std::string::npos)
                        rels.erase(b, e + 2 - b);
                    pkg.set_part("_rels/.rels", rels);
                }
                return;
            }
            const std::string& xml = pkg.part(part);
            std::vector<xml_token> tokens;
            try { tokens = xml_tokenize(xml); }
            catch (xml_error&) { return; }
            const std::vector<std::size_t> match = match_ends(tokens);
            xml_edit_list edits;
            for (std::size_t i = 0; i < tokens.size(); ++i)
            {
                const xml_token& t = tokens[i];
                if (t.kind != xml_token::start_tag || xml_local_name(t.name) != "property" || match[i] == std::string::npos)
                    continue;
                for (const xml_attribute& a : t.attributes)
                    if (a.name == "name" && xml.compare(a.value_begin, 12, "dlib.privacy") == 0)
                        edits.replace(t.begin, tokens[match[i]].end, "");
            }
            pkg.set_part(part, edits.apply(xml));
        }

        const pseudonymizer& p_;
        pii_detector* detector_;
        document_report& report_;
        bool restoring_ = false;
        bool shift_dates_ = false;
        bool dates_marked_ = false;
        long shift_ = 0;
        std::map<std::string, std::string> markers_;
        std::map<std::string, std::string> sheet_names_;    // sheet renamings of the current operation
        std::map<std::string, std::string> sheet_tags_;     // anonymize(): the tag of each renamed sheet
        pii_type pivot_field_type_ = pii_type::none;
        std::vector<std::string> shared_texts_;
        std::vector<pii_type> shared_types_;
        std::set<int> date_styles_;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OOXML_ANONYMIZER_H_
