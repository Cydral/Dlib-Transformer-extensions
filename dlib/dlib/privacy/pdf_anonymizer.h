// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PDF_ANONYMIZER_H_
#define DLIB_PDF_ANONYMIZER_H_

#include "pdf_anonymizer_abstract.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include "../numeric_constants.h"
#include "../documents/pdf_text.h"
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
            document_report& report
        ) : p_(p), detector_(detector), report_(report) {}

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
            for (unit& u : units)
                transform_unit(u, layout[u.page]);

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
                for (const std::string& f : doc.features)
                    report_.removed.push_back(f);
                report_.removed.push_back("fonts, colors and vector graphics (text-only reconstruction)");
            }

            std::size_t unencodable = 0;
            const std::string out = write_pdf_text(doc, layout, &unencodable);
            if (unencodable > 0)
                report_.note(std::to_string(unencodable) +
                             " characters outside the Latin repertoire of the standard PDF fonts were written as '?'");
            return out;
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

            std::size_t k = 0;
            while (k < lines.size())
            {
                const std::vector<std::size_t>& line = lines[k];
                if (line.size() == 1)
                {
                    // A chain of single segment lines: paragraph, address block, list item.
                    std::vector<std::size_t> chain(1, line[0]);
                    while (k + 1 < lines.size() && lines[k + 1].size() == 1 &&
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
                    u.text = segs[line[j]].text;
                    u.pieces.push_back(piece{ line[j], 0, u.text.size() });
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
                        if (full) u.text += (hyphen || spaced(prev, s)) ? "" : " ";
                        else u.text += "\n";
                    }
                }
                const std::size_t b = u.text.size();
                u.text += s.text;
                u.pieces.push_back(piece{ chain[c], b, u.text.size() });
            }
            return u;
        }

    // ------------------------------------------------------------------------------------

        void transform_unit(unit& u, std::vector<pdf_text_segment>& segs)
        {
            if (u.text.empty()) return;
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
                            edits.push_back(span{ pb, pe, p_.tag(e.type, u.text.substr(pb, pe - pb)) });
                            report_.count(e.type);
                        }
                        if (end >= e.end) break;
                        b = k + 1 < u.pieces.size() ? u.pieces[k + 1].ub : end;
                        if (b >= e.end) break;
                    }
                }
            }
            if (edits.empty()) return;

            /* Each segment keeps its characters outside the edits; the replacement of an
               edit goes to the segment holding its first character, or to the next one
               when the edit starts on a line separator. */
            std::vector<std::string> out(u.pieces.size());
            std::vector<bool> touched(u.pieces.size(), false);
            std::vector<bool> placed(edits.size(), false);
            std::size_t ei = 0;
            for (std::size_t k = 0; k < u.pieces.size(); ++k)
            {
                const piece& pc = u.pieces[k];
                while (ei < edits.size() && edits[ei].e <= pc.ub)
                {
                    if (!placed[ei]) { out[k] += edits[ei].rep; touched[k] = true; placed[ei] = true; }
                    ++ei;
                }
                std::size_t pos = pc.ub;
                while (pos < pc.ue)
                {
                    while (ei < edits.size() && edits[ei].e <= pos) ++ei;
                    if (ei < edits.size() && edits[ei].b <= pos)
                    {
                        if (!placed[ei]) { out[k] += edits[ei].rep; placed[ei] = true; }
                        touched[k] = true;
                        pos = std::min(edits[ei].e, pc.ue);
                        continue;
                    }
                    const std::size_t stop = (ei < edits.size()) ? std::min(edits[ei].b, pc.ue) : pc.ue;
                    out[k] += u.text.substr(pos, stop - pos);
                    pos = stop;
                }
            }
            for (std::size_t i = 0; i < edits.size(); ++i)
            {
                if (placed[i] || u.pieces.empty()) continue;
                out.back() += edits[i].rep;
                touched.back() = true;
            }
            for (std::size_t k = 0; k < u.pieces.size(); ++k)
                if (touched[k]) segs[u.pieces[k].seg].text = out[k];
        }

        const pseudonymizer& p_;
        pii_detector* detector_;
        document_report& report_;
        bool restoring_ = false;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PDF_ANONYMIZER_H_
