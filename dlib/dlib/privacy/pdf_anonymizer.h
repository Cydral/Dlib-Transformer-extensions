// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PDF_ANONYMIZER_H_
#define DLIB_PDF_ANONYMIZER_H_

#include "pdf_anonymizer_abstract.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "../numeric_constants.h"
#include "../documents/pdf_text.h"
#include "../documents/pdf_rewrite.h"
#include "pii_detector.h"
#include "pseudonymizer.h"
#include "document_report.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pdf_anonymizer
    {
    public:
        pdf_anonymizer(
            const pseudonymizer& p,
            pii_detector* detector,
            document_report& report,
            bool keep_layout = true
        ) : p_(p), detector_(detector), report_(report), keep_layout_(keep_layout) {}

        void set_image_describer(const document_image_describer& describer) { describer_ = describer; }

        void set_untagged_properties(bool untagged) { untagged_properties_ = untagged; }
        void set_substitute_resolver(const pseudonymizer::substitute_resolver& resolve) { resolve_ = resolve; }

        void set_untagged_text(bool untagged) { untagged_text_ = untagged; }
        void set_text_substitute_resolver(const pseudonymizer::substitute_resolver& resolve)
        {
            text_resolve_ = resolve;
        }

        void set_picture_text_finder(const document_picture_text_finder& find) { pictures_ = find; }

        std::string anonymize(const std::string& pdf_bytes)
        {
            if (!detector_)
                throw error("pdf_anonymizer::anonymize(): a detector is required");
            return run(pdf_bytes, false);
        }

        std::string restore(const std::string& pdf_bytes)
        {
            return run(pdf_bytes, true);
        }

    private:
        struct piece
        {
            std::size_t seg;                    // index in the page layout
            std::size_t ub, ue;                 // range in the unit text
            std::size_t sb;                     // where the range starts in the text of the segment
        };

        struct unit
        {
            std::size_t page = 0;
            std::string text;
            std::vector<piece> pieces;
            detection_context ctx;
        };

        // Position of a segment in the frame of a baseline direction.
        static double along(const pdf_text_segment& s, double angle)
        {
            const double r = angle * pi / 180.0;
            return s.x * std::cos(r) + s.y * std::sin(r);
        }

        static double across(const pdf_text_segment& s, double angle)
        {
            const double r = angle * pi / 180.0;
            return -s.x * std::sin(r) + s.y * std::cos(r);
        }

        static std::size_t code_points(const std::string& s, std::size_t from = 0, std::size_t to = std::string::npos)
        {
            std::size_t n = 0;
            for (std::size_t i = from; i < std::min(to, s.size()); ++i)
                if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) ++n;
            return n;
        }

        static bool ends_with_hyphen(const std::string& s)
        {
            if (s.size() < 2) return false;
            const unsigned char last = static_cast<unsigned char>(s.back());
            const bool letter_before = std::isalpha(static_cast<unsigned char>(s[s.size() - 2])) ||
                (static_cast<unsigned char>(s[s.size() - 2]) & 0x80);
            if (last == '-') return letter_before;
            // U+00AD soft hyphen and U+2010 hyphen.
            return (s.size() >= 3 && s.compare(s.size() - 2, 2, "\xC2\xAD") == 0) ||
                   (s.size() >= 4 && s.compare(s.size() - 3, 3, "\xE2\x80\x90") == 0);
        }

        static bool spaced(const pdf_text_segment& prev, const pdf_text_segment& next)
        {
            return (!prev.text.empty() && prev.text.back() == ' ') || (!next.text.empty() && next.text[0] == ' ');
        }

        static bool short_label_cell(const std::string& s)
        {
            int words = 0;
            bool in_word = false;
            for (char c : s)
            {
                if (std::isdigit(static_cast<unsigned char>(c)) || c == '@') return false;
                const bool space = c == ' ' || c == '\t';
                if (!space && !in_word) ++words;
                in_word = !space;
            }
            return words >= 1 && words <= 5;
        }

    // ------------------------------------------------------------------------------------

        std::string run(const std::string& bytes, bool restoring)
        {
            restoring_ = restoring;
            report_.format = "pdf";
            pdf_text_document doc = read_pdf_text(bytes);
            /* A text-only reconstruction stays one; any other document keeps its pages when asked,
               and is restored keeping them. */
            layout_mode_ = !doc.produced_by_dlib && (restoring || keep_layout_ || doc.rewritten_by_dlib);
            replacements_.clear();
            std::vector<std::vector<pdf_text_segment>> layout = layout_pdf_text(doc);

            for (const std::string& n : doc.notes)
                report_.note("PDF " + n);
            if (doc.undecodable_glyphs > 0)
                report_.note(std::to_string(doc.undecodable_glyphs) +
                             " glyphs have no Unicode mapping in their font; they were not copied (written as '?')");

            std::vector<unit> units;
            for (std::size_t p = 0; p < layout.size(); ++p)
                build_units(p, layout[p], units);
            report_.text_units += units.size();

            if (!restoring_)
            {
                detector_->begin_document();
                for (const unit& u : units)
                    if (!u.text.empty()) detector_->learn(u.text, u.ctx);
            }
            for (const unit& u : units)
                transform_unit(u);
            bool info_changed = false;
            const std::vector<std::pair<std::string, std::string>> info = properties(doc, info_changed);

            if (!layout_mode_)
            {
                // The edits of a segment are applied from its end, so that their offsets hold.
                std::vector<pdf_text_replacement> edits = replacements_;
                std::stable_sort(edits.begin(), edits.end(),
                                 [](const pdf_text_replacement& a, const pdf_text_replacement& b) {
                                     if (a.page != b.page) return a.page < b.page;
                                     if (a.segment != b.segment) return a.segment < b.segment;
                                     return a.begin > b.begin;
                                 });
                for (const pdf_text_replacement& r : edits)
                {
                    std::string& t = layout[r.page][r.segment].text;
                    if (r.begin <= r.end && r.end <= t.size()) t.replace(r.begin, r.end - r.begin, r.text);
                }
            }

            if (layout_mode_)
            {
                // A document that holds no tag and was not protected here is given back as it is.
                if (restoring_ && !doc.rewritten_by_dlib && replacements_.empty() && !info_changed)
                {
                    report_.note("no tag found: the document is returned unchanged");
                    return bytes;
                }
                pdf_rewrite_report rr;
                const std::string out = rewrite_pdf_text(bytes, replacements_, &rr,
                                                         restoring_ ? document_image_describer() : describer_, info);
                if (!restoring_)
                {
                    if (rr.images > 0)
                        report_.removed.push_back("images (" + std::to_string(rr.images) + ", replaced by gray boxes)");
                    if (rr.described_images > 0)
                        report_.note(std::to_string(rr.described_images) +
                                     " gray box(es) received the text of the image describer");
                    if (rr.annotations > 0)
                        report_.removed.push_back("annotations (" + std::to_string(rr.annotations) + ")");
                    removed_features(doc);
                    report_.removed.push_back("embedded fonts (text drawn with standard fonts of the same style)");
                    if (rr.drawn_annotations > 0)
                        report_.note(std::to_string(rr.drawn_annotations) +
                                     " form field, free text or stamp appearance(s) drawn into their page");
                    for (std::size_t p = 0; p < doc.pages.size(); ++p)
                        if (doc.pages[p].spans.empty() && doc.pages[p].images > 0)
                            report_.note("page " + std::to_string(p + 1) +
                                         " has no text layer (scanned page?): its pictures became gray boxes, no "
                                         "OCR is performed");
                    if (rr.scanned_pages > 0)
                        report_.note(std::to_string(rr.scanned_pages) +
                                     " scanned page(s): their recognized text is drawn visible over gray boxes");
                }
                if (rr.unencodable > 0)
                    report_.note(std::to_string(rr.unencodable) +
                                 " characters outside the Latin repertoire of the standard PDF fonts were written as '?'");
                return out;
            }

            if (!restoring_)
            {
                std::size_t images = 0, annotations = 0;
                for (std::size_t p = 0; p < doc.pages.size(); ++p)
                {
                    images += doc.pages[p].images;
                    annotations += doc.pages[p].annotations;
                    if (doc.pages[p].spans.empty() && doc.pages[p].images > 0)
                        report_.note("page " + std::to_string(p + 1) +
                                     " has no text layer (scanned page?): its content was removed with the images, no "
                                     "OCR is performed");
                }
                if (images > 0) report_.removed.push_back("images (" + std::to_string(images) + ")");
                if (annotations > 0) report_.removed.push_back("annotations (" + std::to_string(annotations) + ")");
                removed_features(doc);
                report_.removed.push_back("fonts, colors and vector graphics (text-only reconstruction)");
            }

            std::size_t unencodable = 0;
            const std::string out = write_pdf_text(doc, layout, &unencodable, info);
            if (unencodable > 0)
                report_.note(std::to_string(unencodable) +
                             " characters outside the Latin repertoire of the standard PDF fonts were written as '?'");
            return out;
        }

    // ------------------------------------------------------------------------------------
    // Document information.

        // The entries of the document information that untagged properties keep, with their substitutes.
        static bool kept_property(const std::string& key)
        {
            return key == "Title" || key == "Author" || key == "Subject" || key == "Keywords" || key == "Creator";
        }

        /* The document information written with the PDF: none unless the properties are untagged;
           then the kept entries with their values replaced by substitutes, or given back. */
        std::vector<std::pair<std::string, std::string>> properties(const pdf_text_document& doc, bool& changed)
        {
            std::vector<std::pair<std::string, std::string>> out;
            changed = false;
            if (!untagged_properties_) return out;
            for (const auto& kv : doc.info)
            {
                if (!kept_property(kv.first) || kv.second.empty()) continue;
                ++report_.text_units;
                std::string v = kv.second;
                if (restoring_)
                {
                    pseudonymizer::restore_report rr, sr;
                    v = p_.restore(v, &rr);
                    report_.tags_restored += rr.restored;
                    report_.tags_rejected += rr.rejected;
                    v = p_.restore_substitutes(v, resolve_, &sr);
                    report_.substitutes_restored += sr.restored;
                }
                else
                {
                    detection_context ctx;
                    if (kv.first == "Author") ctx.forced = pii_type::person;
                    // The substitutes written before, a document protected and then edited, stay as they are.
                    const std::vector<pseudonymizer::substitute_occurrence> known = p_.find_substitutes(v, resolve_);
                    std::vector<pii_entity> found;
                    for (const pii_entity& e : detector_->detect(v, ctx))
                    {
                        if (p_.keeps_in_clear(e.type)) continue;
                        bool written = false;
                        for (const auto& o : known)
                            if (e.begin < o.end && o.begin < e.end) written = true;
                        if (written) continue;
                        pii_entity f = e;
                        while (f.begin < f.end && v[f.begin] == ' ') ++f.begin;
                        while (f.end > f.begin && v[f.end - 1] == ' ') --f.end;
                        if (f.end <= f.begin) continue;
                        found.push_back(f);
                        report_.count(e.type);
                    }
                    std::vector<pseudonymizer::substitution> made;
                    v = p_.apply_substitutes(v, found, &made);
                    for (const pseudonymizer::substitution& s : made)
                        report_.substitutions.push_back(document_substitution{ s.substitute, s.tag });
                }
                if (v != kv.second) changed = true;
                out.push_back(std::make_pair(kv.first, v));
            }
            return out;
        }

        // What the protected PDF leaves out; the document information it keeps is not listed.
        void removed_features(const pdf_text_document& doc)
        {
            for (const std::string& f : doc.features)
            {
                if (!untagged_properties_ || f.compare(0, 20, "document information") != 0)
                {
                    report_.removed.push_back(f);
                    continue;
                }
                // "document information (Author, CreationDate, Producer)": the keys not kept.
                const std::size_t open = f.find('('), close = f.rfind(')');
                std::string dropped;
                if (open != std::string::npos && close != std::string::npos && close > open)
                {
                    std::istringstream keys(f.substr(open + 1, close - open - 1));
                    std::string key;
                    while (std::getline(keys, key, ','))
                    {
                        while (!key.empty() && key[0] == ' ') key.erase(0, 1);
                        if (!key.empty() && !kept_property(key)) dropped += (dropped.empty() ? "" : ", ") + key;
                    }
                }
                if (!dropped.empty()) report_.removed.push_back("document information (" + dropped + ")");
            }
        }

    // ------------------------------------------------------------------------------------
    // Text units.

        void build_units(std::size_t page, const std::vector<pdf_text_segment>& segs, std::vector<unit>& units) const
        {
            std::vector<std::vector<std::size_t>> lines;
            for (std::size_t i = 0; i < segs.size(); ++i)
            {
                if (lines.empty() || segs[lines.back().front()].line != segs[i].line)
                    lines.push_back(std::vector<std::size_t>());
                lines.back().push_back(i);
            }

            struct column { double b, e; pii_type type; };
            std::vector<column> header;
            double last_row_v = 0, last_row_fs = 0;

            /* The end of a word cut by its hyphen in a cell of a table, which the next line
               holds under the cell: "fatou.ndiaye@arvelis-" then "services.fr".  The writer of
               the file may have drawn it with the text of the cell on its left, in one segment:
               "Ouessant Logistique SARL logistique.fr".  The part of a segment taken by such a
               cell is [taken_b, taken_e). */
            const std::size_t npos = std::string::npos;
            std::vector<std::size_t> taken_b(segs.size(), npos), taken_e(segs.size(), 0);
            struct taken_part { std::size_t seg, b, e; };
            auto continuation = [&](std::size_t k, std::size_t si) -> taken_part {
                const taken_part none = { npos, 0, 0 };
                const pdf_text_segment& s = segs[si];
                if (k + 1 >= lines.size() || !ends_with_hyphen(s.text)) return none;
                const double a = s.angle, fs = std::max(1.0, s.font_size);
                const double x = along(s, a);
                for (std::size_t c : lines[k + 1])
                {
                    const pdf_text_segment& n = segs[c];
                    if (taken_b[c] != npos || std::abs(n.angle - a) > 0.5 || n.text.empty()) continue;
                    const double dv = across(s, a) - across(n, a);
                    if (dv < 0.5 * fs || dv > 2.5 * fs) continue;
                    const double nx = along(n, a);
                    std::size_t b = npos;
                    if (std::abs(x - nx) <= std::max(1.5, 0.6 * fs))
                    {
                        b = 0;
                    }
                    else if (nx < x - fs && nx + n.width > x + fs)
                    {
                        /* The word starting the closest to the cell, its place estimated from
                           the number of characters before it.  The segment joins texts closer
                           than two sizes of the font, which bounds the error of the estimate. */
                        const double per = n.width / static_cast<double>(std::max<std::size_t>(1, code_points(n.text)));
                        double best = 2.0 * fs;
                        std::size_t cps = 0;
                        for (std::size_t i = 0; i < n.text.size(); ++i)
                        {
                            const unsigned char ch = static_cast<unsigned char>(n.text[i]);
                            if (i > 0 && n.text[i - 1] == ' ' && ch != ' ')
                            {
                                const double d = std::abs(nx + per * static_cast<double>(cps) - x);
                                if (d < best)
                                {
                                    best = d;
                                    b = i;
                                }
                            }
                            if ((ch & 0xC0) != 0x80) ++cps;
                        }
                    }
                    if (b == npos) continue;
                    const unsigned char c0 = static_cast<unsigned char>(n.text[b]);
                    if (!std::islower(c0) && !std::isdigit(c0)) continue;
                    std::size_t e = n.text.size();
                    if (b > 0)
                    {
                        e = n.text.find(' ', b);
                        if (e == npos) e = n.text.size();
                    }
                    return taken_part{ c, b, e };
                }
                return none;
            };

            // Appends to a unit the parts of a segment that no cell above took.
            auto add_own = [&](unit& u, std::size_t si) {
                const std::string& t = segs[si].text;
                const std::size_t b = taken_b[si] == npos ? t.size() : taken_b[si];
                const std::size_t e = taken_b[si] == npos ? t.size() : taken_e[si];
                if (b > 0)
                {
                    u.pieces.push_back(piece{ si, u.text.size(), u.text.size() + b, 0 });
                    u.text.append(t, 0, b);
                }
                if (e < t.size())
                {
                    if (!u.text.empty()) u.text += "\n";
                    u.pieces.push_back(piece{ si, u.text.size(), u.text.size() + (t.size() - e), e });
                    u.text.append(t, e, npos);
                }
            };

            std::size_t k = 0;
            while (k < lines.size())
            {
                const std::vector<std::size_t>& line = lines[k];
                if (line.size() == 1 && taken_b[line[0]] != npos)
                {
                    unit u;
                    u.page = page;
                    add_own(u, line[0]);
                    if (!u.pieces.empty()) units.push_back(u);
                    ++k;
                    continue;
                }
                if (line.size() == 1)
                {
                    // A chain of single segment lines: paragraph, address block, list item.
                    std::vector<std::size_t> chain(1, line[0]);
                    while (k + 1 < lines.size() && lines[k + 1].size() == 1 && taken_b[lines[k + 1][0]] == npos &&
                           continues(segs[chain.back()], segs[lines[k + 1][0]], chain.size() == 1))
                    {
                        chain.push_back(lines[k + 1][0]);
                        ++k;
                    }
                    if (chain.size() > 1 ||
                        std::abs(across(segs[chain[0]], segs[chain[0]].angle) - last_row_v) > 3 * last_row_fs)
                        header.clear();
                    units.push_back(chain_unit(page, segs, chain));
                    ++k;
                    continue;
                }

                // A row of cells.
                const pdf_text_segment& first = segs[line[0]];
                const double v = across(first, first.angle);
                const double fs = std::max(1.0, first.font_size);
                if (!header.empty() && std::abs(v - last_row_v) > 3 * std::max(fs, last_row_fs))
                    header.clear();

                /* A header row is made of labels: every cell, or most of them when the row
                   is wide. "Ville | Saint-Malo" holds a label and its value. */
                std::vector<pii_type> label(line.size(), pii_type::none);
                std::size_t labels = 0;
                bool all_short = true;
                if (!restoring_ && detector_)
                {
                    for (std::size_t j = 0; j < line.size(); ++j)
                    {
                        label[j] = detector_->label_type(segs[line[j]].text);
                        if (label[j] != pii_type::none) ++labels;
                        if (!short_label_cell(segs[line[j]].text)) all_short = false;
                    }
                }
                const bool is_header = all_short && labels > 0 &&
                    (labels == line.size() || (line.size() >= 3 && labels >= 2 && 10 * labels >= 6 * line.size()));

                for (std::size_t j = 0; j < line.size(); ++j)
                {
                    unit u;
                    u.page = page;
                    add_own(u, line[j]);
                    if (u.pieces.empty()) continue;
                    if (!restoring_ && taken_b[line[j]] == npos)
                    {
                        const taken_part t = continuation(k, line[j]);
                        if (t.seg != npos)
                        {
                            const std::size_t b = u.text.size();
                            u.text.append(segs[t.seg].text, t.b, t.e - t.b);
                            u.pieces.push_back(piece{ t.seg, b, u.text.size(), t.b });
                            taken_b[t.seg] = t.b;
                            taken_e[t.seg] = t.e;
                        }
                    }
                    if (!restoring_ && !is_header)
                    {
                        if (j > 0 && label[j - 1] != pii_type::none)
                        {
                            u.ctx.label = label[j - 1];
                        }
                        else if (!header.empty())
                        {
                            // The header cell overlapping this cell the most.
                            const pdf_text_segment& s = segs[line[j]];
                            const double b = along(s, s.angle), e = b + std::max(s.width, 1.0);
                            double best = 0;
                            for (const column& c : header)
                            {
                                const double overlap = std::min(e, c.e) - std::max(b, c.b);
                                if (overlap > best) { best = overlap; u.ctx.label = c.type; }
                            }
                        }
                    }
                    units.push_back(u);
                }

                if (is_header)
                {
                    header.clear();
                    for (std::size_t j = 0; j < line.size(); ++j)
                    {
                        const pdf_text_segment& s = segs[line[j]];
                        const double b = along(s, s.angle);
                        header.push_back(column{ b, b + std::max(s.width, 1.0), label[j] });
                    }
                }
                last_row_v = v;
                last_row_fs = fs;
                ++k;
            }
        }

        /* Whether next continues the block of prev: same direction and size, regular line
           spacing, aligned left edges (or a first-line or hanging indent), or centered lines. */
        static bool continues(const pdf_text_segment& prev, const pdf_text_segment& next, bool prev_is_first)
        {
            if (std::abs(prev.angle - next.angle) > 0.5) return false;
            const double fs = std::max(1.0, std::max(prev.font_size, next.font_size));
            if (std::abs(prev.font_size - next.font_size) > 0.25 * fs) return false;
            const double a = prev.angle;
            const double dv = across(prev, a) - across(next, a);
            if (dv < 0.5 * fs || dv > 2.5 * fs) return false;
            const double up = along(prev, a), un = along(next, a);
            if (std::abs(up - un) <= std::max(1.5, 0.6 * fs)) return true;
            if (prev_is_first && std::abs(up - un) <= 4 * fs) return true;
            const double cp = up + prev.width / 2, cn = un + next.width / 2;
            return std::abs(cp - cn) <= fs && prev.width < 40 * fs;
        }

        unit chain_unit(std::size_t page, const std::vector<pdf_text_segment>& segs,
                        const std::vector<std::size_t>& chain) const
        {
            unit u;
            u.page = page;
            const double a = segs[chain[0]].angle;
            double max_end = 0;
            for (std::size_t i : chain)
                max_end = std::max(max_end, along(segs[i], a) + segs[i].width);
            for (std::size_t c = 0; c < chain.size(); ++c)
            {
                const pdf_text_segment& s = segs[chain[c]];
                if (c > 0)
                {
                    const pdf_text_segment& prev = segs[chain[c - 1]];
                    const bool hyphen = ends_with_hyphen(prev.text);
                    if (restoring_)
                    {
                        // Tags may wrap at any space of a line.
                        u.text += (hyphen || spaced(prev, s)) ? "" : " ";
                    }
                    else
                    {
                        /* A line wraps when the first word of the next line would not
                           have fitted at its end. */
                        const double fs = std::max(1.0, prev.font_size);
                        std::size_t word_end = s.text.find(' ');
                        if (word_end == std::string::npos) word_end = s.text.size();
                        const std::size_t total = std::max<std::size_t>(1, code_points(s.text));
                        const double first_word =
                            s.width * static_cast<double>(code_points(s.text, 0, word_end)) / total;
                        const double prev_end = along(prev, a) + prev.width;
                        const bool full = prev.width >= 10 * fs && prev_end + 0.28 * fs + first_word > max_end - 0.5;
                        /* A word cut by its hyphen at the end of a line of a narrow cell goes on
                           in the next one: "jean.dupont@exemple-" then "conseil.fr". */
                        const unsigned char next0 = s.text.empty() ? 0 : static_cast<unsigned char>(s.text[0]);
                        const bool continued = hyphen && prev.text.find(' ') == std::string::npos &&
                                               (std::islower(next0) || std::isdigit(next0));
                        if (full || continued) u.text += (hyphen || spaced(prev, s)) ? "" : " ";
                        else u.text += "\n";
                    }
                }
                const std::size_t b = u.text.size();
                u.text += s.text;
                u.pieces.push_back(piece{ chain[c], b, u.text.size(), 0 });
            }
            return u;
        }

    // ------------------------------------------------------------------------------------

        // The protected form of a value of the text: its tag, or its substitute listed in the report.
        std::string text_value(pii_type type, const std::string& value)
        {
            if (!untagged_text_) return p_.tag(type, value);
            document_substitution s;
            s.substitute = p_.substitute(type, value, &s.tag);
            s.text = true;
            report_.substitutions.push_back(s);
            return s.substitute;
        }

        void transform_unit(const unit& u)
        {
            if (u.text.empty()) return;
            struct span { std::size_t b, e; std::string rep; };
            std::vector<span> edits;
            if (restoring_)
            {
                // The texts written in the place of pictures leave, with the tags they hold.
                std::vector<std::pair<std::size_t, std::size_t>> pictures;
                if (pictures_) pictures = pictures_(u.text);
                auto in_picture = [&](std::size_t b, std::size_t e) {
                    for (const auto& r : pictures)
                        if (b < r.second && r.first < e) return true;
                    return false;
                };
                for (const auto& r : pictures)
                    if (r.first < r.second && r.second <= u.text.size())
                    {
                        edits.push_back(span{ r.first, r.second, std::string() });
                        ++report_.pictures_removed;
                    }
                pseudonymizer::restore_report rr;
                std::size_t inside = 0;
                for (const auto& o : p_.find_tags(u.text, &rr))
                {
                    if (in_picture(o.begin, o.end)) ++inside;
                    else edits.push_back(span{ o.begin, o.end, o.original });
                }
                report_.tags_restored += rr.restored - std::min(inside, rr.restored);
                report_.tags_rejected += rr.rejected;
                // The substitutes of an untagged text, outside its tags and the texts of its pictures.
                if (text_resolve_)
                    for (const auto& o : p_.find_substitutes(u.text, text_resolve_))
                    {
                        bool covered = false;
                        for (const span& s : edits)
                            if (o.begin < s.e && s.b < o.end) covered = true;
                        if (covered) continue;
                        edits.push_back(span{ o.begin, o.end, o.original });
                        ++report_.substitutes_restored;
                    }
                std::sort(edits.begin(), edits.end(), [](const span& a, const span& b) { return a.b < b.b; });
            }
            else
            {
                // The substitutes that the text resolver knows, an untagged document protected again, stay.
                std::vector<pseudonymizer::substitute_occurrence> known;
                if (text_resolve_) known = p_.find_substitutes(u.text, text_resolve_);
                std::vector<pii_entity> found = detector_->detect(u.text, u.ctx);
                if (untagged_text_)
                {
                    /* Without tags, the tags of a text protected before with them take substitutes, those
                       that one piece of the text holds: a tag cut at a line end would become two values. */
                    std::vector<pii_entity> with = p_.with_tag_entities(u.text, std::move(found));
                    found.clear();
                    for (const pii_entity& e : with)
                    {
                        pii_type type = pii_type::none;
                        std::string value;
                        bool whole = u.text.compare(e.begin, 2, "[[") != 0 ||
                                     !p_.decode(u.text.substr(e.begin, e.end - e.begin), type, value);
                        for (const piece& pc : u.pieces)
                            if (pc.ub <= e.begin && e.end <= pc.ue) whole = true;
                        if (whole) found.push_back(e);
                    }
                }
                for (const pii_entity& e : found)
                {
                    bool written = false;
                    for (const auto& o : known)
                        if (e.begin < o.end && o.begin < e.end) written = true;
                    if (written) continue;
                    if (p_.keeps_in_clear(e.type)) continue;
                    /* A value read across lines is cut at each line end, so that every
                       line keeps its part of it and restoration puts it back in place.
                       The report counts the tags written, one for each part. */
                    std::size_t b = e.begin;
                    for (std::size_t k = 0; k <= u.pieces.size(); ++k)
                    {
                        const std::size_t cut = k < u.pieces.size() ? u.pieces[k].ue : e.end;
                        if (cut <= b) continue;
                        const std::size_t end = std::min(cut, e.end);
                        std::size_t pb = b, pe = end;
                        while (pb < pe && (u.text[pb] == ' ' || u.text[pb] == '\n')) ++pb;
                        while (pe > pb && (u.text[pe - 1] == ' ' || u.text[pe - 1] == '\n')) --pe;
                        if (pe > pb)
                        {
                            edits.push_back(span{ pb, pe, text_value(e.type, u.text.substr(pb, pe - pb)) });
                            report_.count(e.type);
                        }
                        if (end >= e.end) break;
                        b = k + 1 < u.pieces.size() ? u.pieces[k + 1].ub : end;
                        if (b >= e.end) break;
                    }
                }
            }

            /* Each edit replaces the characters of the segments it covers: its text goes to
               the first one (drawn over its extent when the pages are kept), the others lose
               theirs. */
            for (const span& ed : edits)
            {
                bool first = true;
                for (const piece& pc : u.pieces)
                {
                    const std::size_t b = std::max(ed.b, pc.ub), e = std::min(ed.e, pc.ue);
                    if (b >= e) continue;
                    pdf_text_replacement r;
                    r.page = u.page;
                    r.segment = pc.seg;
                    r.begin = pc.sb + (b - pc.ub);
                    r.end = pc.sb + (e - pc.ub);
                    if (first) r.text = ed.rep;
                    first = false;
                    replacements_.push_back(r);
                }
            }
        }

        const pseudonymizer& p_;
        pii_detector* detector_;
        document_report& report_;
        bool keep_layout_ = true;
        document_image_describer describer_;
        bool untagged_properties_ = false;
        pseudonymizer::substitute_resolver resolve_;
        bool untagged_text_ = false;
        pseudonymizer::substitute_resolver text_resolve_;
        document_picture_text_finder pictures_;
        bool restoring_ = false;
        bool layout_mode_ = false;
        std::vector<pdf_text_replacement> replacements_;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PDF_ANONYMIZER_H_
