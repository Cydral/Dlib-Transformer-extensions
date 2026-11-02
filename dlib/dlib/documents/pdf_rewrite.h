// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PDF_REWRITE_H_
#define DLIB_PDF_REWRITE_H_

#include "pdf_rewrite_abstract.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "document_image.h"
#include "pdf_text.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pdf_text_replacement
    {
        std::size_t page = 0;
        std::size_t segment = 0;
        std::size_t begin = 0, end = 0;
        std::string text;
    };

    struct pdf_rewrite_report
    {
        std::size_t images = 0;
        std::size_t annotations = 0;
        std::size_t drawn_annotations = 0;
        std::size_t unencodable = 0;
        std::size_t described_images = 0;
        std::size_t scanned_pages = 0;
    };

// ----------------------------------------------------------------------------------------

    namespace pdf_impl
    {
        // Numbers of a rewritten file: integers as such, others with six decimals at most.
        inline std::string format_exact(double v)
        {
            if (!std::isfinite(v)) v = 0;
            v = std::max(-1e15, std::min(1e15, v));
            if (std::abs(v - std::round(v)) < 1e-9) return std::to_string(static_cast<long long>(std::llround(v)));
            long long m = std::llround(v * 1000000);
            std::string s;
            if (m < 0) { s.push_back('-'); m = -m; }
            s += std::to_string(m / 1000000) + ".";
            std::string frac = std::to_string(m % 1000000);
            frac = std::string(6 - frac.size(), '0') + frac;
            while (!frac.empty() && frac.back() == '0') frac.pop_back();
            s += frac;
            return s;
        }

        inline std::string pdf_name(const std::string& n)
        {
            std::string out = "/";
            for (unsigned char c : n)
            {
                if (c > 0x20 && c < 0x7F && c != '#' && c != '/' && c != '(' && c != ')' && c != '<' && c != '>' &&
                    c != '[' && c != ']' && c != '{' && c != '}' && c != '%')
                    out.push_back(static_cast<char>(c));
                else
                {
                    char buf[4];
                    std::snprintf(buf, sizeof(buf), "#%02X", c);
                    out += buf;
                }
            }
            return out;
        }

        inline std::string pdf_hex_string(const std::string& bytes)
        {
            static const char* digits = "0123456789ABCDEF";
            std::string out = "<";
            for (unsigned char c : bytes)
            {
                out.push_back(digits[c >> 4]);
                out.push_back(digits[c & 15]);
            }
            return out + ">";
        }

        inline std::string pdf_matrix_operands(const pdf_matrix& m)
        {
            return format_exact(m.a) + " " + format_exact(m.b) + " " + format_exact(m.c) + " " + format_exact(m.d) +
                   " " + format_exact(m.e) + " " + format_exact(m.f);
        }

        // Keys of dictionaries that carry metadata, structure or alternate texts, left out of a rewrite.
        inline bool dropped_key(const std::string& k)
        {
            return k == "Metadata" || k == "PieceInfo" || k == "StructParent" || k == "StructParents" || k == "OPI" ||
                   k == "AF" || k == "ActualText" || k == "Alt" || k == "E" || k == "LastModified";
        }

        /* The rewrite of one document: its pages and the objects they use are written again,
           the text of every page drawn with standard fonts. */
        class pdf_rewriter
        {
        public:
            pdf_rewriter(const std::string& bytes, const std::vector<pdf_text_replacement>& replacements,
                         pdf_rewrite_report& report, const document_image_describer& describer)
                : doc_(bytes), report_(report)
            {
                text_ = read_document(doc_, &rec_, &leaves_);
                layout_ = layout_pages(text_, &sources_);
                index_runs();
                place_replacements(replacements);
                find_scanned_pages();
                plan_operations();
                if (describer) plan_captions(describer);
            }

            std::string write()
            {
                const object* root = doc_.resolve_key(doc_.trailer, "Root");
                const int catalog = reserve();
                const int pages = reserve();
                font_base_ = static_cast<int>(bodies_.size()) + 1;
                for (int k = 0; k < 12; ++k)
                {
                    std::string widths;
                    for (int c = 32; c < 256; ++c)
                        widths += format_exact(standard_width(k, static_cast<unsigned char>(c))) + (c < 255 ? " " : "");
                    add("<< /Type /Font /Subtype /Type1 /BaseFont /" + std::string(standard_font_name(k)) +
                        " /Encoding /WinAnsiEncoding /FirstChar 32 /LastChar 255 /Widths [" + widths + "] >>");
                }
                if (!extra_chars_.empty()) write_extended_font();

                std::vector<int> kids;
                for (std::size_t p = 0; p < leaves_.size(); ++p)
                    kids.push_back(write_page(p, pages));
                drain();

                std::string k;
                for (int id : kids) k += std::to_string(id) + " 0 R ";
                set(pages, "<< /Type /Pages /Kids [" + k + "] /Count " + std::to_string(kids.size()) + " >>");
                std::string cat = "<< /Type /Catalog /Pages " + std::to_string(pages) + " 0 R /DlibPrivacyLayout true";
                if (root && root->is(object::dict_k))
                {
                    for (const char* key : { "Lang", "ViewerPreferences", "OCProperties", "PageLayout" })
                        if (const object* v = root->get(key))
                            cat += " /" + std::string(key) + " " + value(*v, 0);
                }
                set(catalog, cat + " >>");
                drain();
                return file(catalog);
            }

        private:
            struct span_edit
            {
                std::size_t b, e;       // byte range in the span text
                std::string text;       // drawn instead; empty when another span draws it
                double extent;          // width of the drawing in text space, or < 0 for its own range
                std::size_t id;         // the replacement
            };

            typedef std::pair<long long, std::size_t> op_key;

            // ------------------------------------------------------------------------------

            void index_runs()
            {
                runs_.resize(text_.pages.size());
                for (std::size_t p = 0; p < text_.pages.size(); ++p)
                    runs_[p].assign(text_.pages[p].spans.size(), nullptr);
                for (const text_operation& op : rec_.operations)
                    for (const glyph_run& r : op.runs)
                        if (op.page < runs_.size() && r.span < runs_[op.page].size() && !runs_[op.page][r.span])
                            runs_[op.page][r.span] = &r;
            }

            // Advance in text space at byte q of the run, at the start of its glyph or at the end of the glyph before.
            static double advance_at(const glyph_run& r, std::size_t q, bool end)
            {
                if (r.offsets.size() < 2) return 0;
                const std::size_t n = r.offsets.size() - 1;
                if (end)
                {
                    if (q == 0) return r.advances[0];
                    std::size_t g = static_cast<std::size_t>(
                        std::upper_bound(r.offsets.begin(), r.offsets.end(), q - 1) - r.offsets.begin());
                    g = std::min(n, std::max<std::size_t>(g, 1));
                    return r.advances[g];
                }
                std::size_t g = static_cast<std::size_t>(
                    std::upper_bound(r.offsets.begin(), r.offsets.end(), q) - r.offsets.begin());
                g = g == 0 ? 0 : std::min(n, g - 1);
                return r.advances[g];
            }

            // Position along the baseline of the span, in default user space, of byte q of its text.
            double user_position(std::size_t page, std::size_t span, std::size_t q, bool end) const
            {
                const pdf_text_span& s = text_.pages[page].spans[span];
                const double rad = s.angle * pi / 180.0;
                const double origin = s.x * std::cos(rad) + s.y * std::sin(rad);
                const glyph_run* r = runs_[page][span];
                if (r && !r->advances.empty() && std::abs(r->advances.back()) > 1e-9)
                    return origin + advance_at(*r, q, end) * s.width / r->advances.back();
                const std::size_t n = std::max<std::size_t>(1, s.text.size());
                return origin + s.width * static_cast<double>(std::min(q, n)) / n;
            }

            void place_replacements(const std::vector<pdf_text_replacement>& replacements)
            {
                edits_.resize(text_.pages.size());
                for (std::size_t p = 0; p < text_.pages.size(); ++p)
                    edits_[p].resize(text_.pages[p].spans.size());
                for (const pdf_text_replacement& r : replacements)
                {
                    if (r.page >= sources_.size() || r.segment >= sources_[r.page].segments.size() || r.end <= r.begin)
                        continue;
                    const std::vector<pdf_text_span>& spans = text_.pages[r.page].spans;
                    struct hit { std::size_t span, b, e; };
                    std::vector<hit> hits;
                    for (const auto& part : sources_[r.page].segments[r.segment])
                    {
                        const std::size_t o = part.second, len = spans[part.first].text.size();
                        const std::size_t lo = std::max(r.begin, o), hi = std::min(r.end, o + len);
                        if (lo < hi) hits.push_back(hit{ part.first, lo - o, hi - o });
                    }
                    if (hits.empty()) continue;
                    double extent = -1;
                    if (hits.size() > 1)
                    {
                        const hit& f = hits.front();
                        const hit& l = hits.back();
                        const double width = user_position(r.page, l.span, l.e, true) -
                                             user_position(r.page, f.span, f.b, false);
                        const glyph_run* run = runs_[r.page][f.span];
                        const pdf_text_span& s = spans[f.span];
                        if (run && !run->advances.empty() && std::abs(s.width) > 1e-9)
                            extent = width * run->advances.back() / s.width;
                        else
                            extent = width;
                    }
                    const std::size_t id = parts_.size();
                    parts_.push_back(std::vector<std::size_t>());
                    for (std::size_t k = 0; k < hits.size(); ++k)
                    {
                        edits_[r.page][hits[k].span].push_back(span_edit{
                            hits[k].b, hits[k].e, k == 0 ? r.text : std::string(), k == 0 ? extent : -1, id });
                        parts_[id].push_back(hits[k].span);
                    }
                }
                /* A span drawn again over another one takes its edits, as copies of their own:
                   "fake bold" draws a name twice. */
                for (std::size_t p = 0; p < sources_.size(); ++p)
                {
                    std::map<std::size_t, std::size_t> copies;
                    for (const auto& d : sources_[p].duplicates)
                    {
                        if (d.first >= edits_[p].size() || d.second >= edits_[p].size() || !edits_[p][d.first].empty())
                            continue;
                        for (span_edit ed : edits_[p][d.second])
                        {
                            if (!copies.count(ed.id))
                            {
                                copies[ed.id] = parts_.size();
                                parts_.push_back(std::vector<std::size_t>());
                            }
                            ed.id = copies[ed.id];
                            parts_[ed.id].push_back(d.first);
                            edits_[p][d.first].push_back(ed);
                        }
                    }
                }
                for (auto& page : edits_)
                    for (auto& span : page)
                        std::sort(span.begin(), span.end(),
                                  [](const span_edit& a, const span_edit& b) { return a.b < b.b; });
            }

            /* A page whose text is all invisible over pictures is a scan with its recognized text:
               once the pictures are replaced, its text is drawn visible. */
            void find_scanned_pages()
            {
                scanned_.assign(text_.pages.size(), false);
                std::vector<int> state(text_.pages.size(), 0);  // 1 invisible text only, 2 visible text seen
                for (const text_operation& op : rec_.operations)
                    for (const glyph_run& r : op.runs)
                        if (op.page < state.size())
                            state[op.page] = (r.render == 3 || r.render == 7) && state[op.page] != 2 ? 1 : 2;
                for (std::size_t p = 0; p < scanned_.size(); ++p)
                    if (state[p] == 1 && text_.pages[p].images > 0)
                    {
                        scanned_[p] = true;
                        ++report_.scanned_pages;
                    }
            }

            // ------------------------------------------------------------------------------
            // Drawing text with the standard fonts.

            struct encoded_run
            {
                int font;               // 0 to 11, or 12 for the font of Latin Extended-A letters
                std::string bytes;
                double width;           // thousandths of an em
            };

            void collect_extra_chars(const std::string& text)
            {
                for (std::size_t i = 0; i < text.size();)
                {
                    const std::uint32_t cp = utf8_get(text, i);
                    if (unicode_to_winansi(cp) < 0 && !writer_substitute(cp) && latin_extended_a_name(cp) &&
                        !extra_codes_.count(cp) && extra_chars_.size() < 223)
                    {
                        extra_codes_[cp] = static_cast<int>(33 + extra_chars_.size());
                        extra_chars_.push_back(cp);
                    }
                }
            }

            std::vector<encoded_run> encode(const std::string& text, int k)
            {
                std::vector<encoded_run> runs;
                auto put = [&](int f, unsigned char b, double w) {
                    if (runs.empty() || runs.back().font != f) runs.push_back(encoded_run{ f, std::string(), 0 });
                    runs.back().bytes.push_back(static_cast<char>(b));
                    runs.back().width += w;
                };
                for (std::size_t i = 0; i < text.size();)
                {
                    const std::uint32_t cp = utf8_get(text, i);
                    if (cp == '\r' || cp == '\n') { put(k, ' ', standard_width(k, ' ')); continue; }
                    const int b = unicode_to_winansi(cp);
                    if (b >= 0) { put(k, static_cast<unsigned char>(b), standard_width(k, static_cast<unsigned char>(b))); continue; }
                    if (const char* sub = writer_substitute(cp))
                    {
                        for (const char* q = sub; *q; ++q)
                            put(k, static_cast<unsigned char>(*q), standard_width(k, static_cast<unsigned char>(*q)));
                        continue;
                    }
                    const auto e = extra_codes_.find(cp);
                    if (e != extra_codes_.end())
                    {
                        put(12, static_cast<unsigned char>(e->second), helvetica_width(cp));
                        continue;
                    }
                    put(k, '?', standard_width(k, '?'));
                    ++report_.unencodable;
                }
                return runs;
            }

            /* The operators that draw text in the place of a run of the original: starting at
               advance a0 of the run, over width in text space, with standard font k. */
            std::string draw(const glyph_run& r, const std::string& text, double a0, double width, int k,
                             bool visible)
            {
                if (text.empty() || std::abs(r.size) < 1e-6) return std::string();
                const std::vector<encoded_run> runs = encode(text, k);
                const double size = r.size * (r.em > 1e-6 ? r.em : 1);
                double natural = 0;
                for (const encoded_run& e : runs) natural += e.width;
                natural = natural / 1000.0 * size;
                if (std::abs(natural) < 1e-9 || std::abs(width) < 1e-9) return std::string();
                const double tz = std::max(1.0, std::min(2000.0, 100.0 * width / natural));
                const pdf_matrix m = pdf_matrix(1, 0, 0, 1, a0, 0) * r.tm;
                std::string out = std::string(visible ? " 0 Tr" : "") + " " + format_exact(tz) + " Tz " +
                                  format_exact(r.ts) + " Ts 0 Tc 0 Tw " + pdf_matrix_operands(m) + " Tm";
                for (const encoded_run& e : runs)
                    out += " /DlibF" + std::to_string(e.font) + " " + format_exact(size) + " Tf " +
                           pdf_hex_string(e.bytes) + " Tj";
                return out;
            }

            // Natural width in text space of text drawn with standard font k for run r, horizontal scaling 1.
            double natural_width(const glyph_run& r, const std::string& text, int k)
            {
                const std::size_t bad = report_.unencodable;
                double w = 0;
                for (const encoded_run& e : encode(text, k)) w += e.width;
                report_.unencodable = bad;
                return w / 1000.0 * r.size * (r.em > 1e-6 ? r.em : 1);
            }

            /* Free room after a span along its line, in default user space: up to the next span of
               the line, or to the edge of the page for upright text, at most twice the span. */
            double room_after(std::size_t page, std::size_t span)
            {
                if (room_.size() != text_.pages.size()) room_.resize(text_.pages.size());
                std::vector<double>& room = room_[page];
                if (room.empty())
                {
                    const std::vector<pdf_text_span>& spans = text_.pages[page].spans;
                    room.assign(spans.size(), 0);
                    std::map<std::size_t, std::vector<std::size_t>> lines;
                    for (std::size_t g = 0; g < sources_[page].segments.size(); ++g)
                        for (const auto& part : sources_[page].segments[g])
                            lines[layout_[page][g].line].push_back(part.first);
                    // The right edge of the upright text of the page, where a line may run to.
                    double right = text_.pages[page].x0;
                    for (std::size_t i = 0; i < spans.size(); ++i)
                        if (std::abs(spans[i].angle) < 0.5)
                            right = std::max(right, user_position(page, i, spans[i].text.size(), true));
                    for (const auto& l : lines)
                    {
                        for (std::size_t i : l.second)
                        {
                            const pdf_text_span& s = spans[i];
                            const double end = user_position(page, i, s.text.size(), true);
                            double next = 1e18;
                            for (std::size_t j : l.second)
                            {
                                const double b = user_position(page, j, 0, false);
                                if (j != i && b > end - 0.5) next = std::min(next, b);
                            }
                            const double fs = std::max(1.0, s.font_size);
                            if (next > 1e17) next = std::abs(s.angle) < 0.5 ? right : end;
                            room[i] = std::max(0.0, std::min(next - end - 0.3 * fs, 2 * std::abs(s.width)));
                        }
                    }
                    // Text drawn again over a span has its room, so that both are drawn alike.
                    for (const auto& d : sources_[page].duplicates)
                        if (d.first < room.size() && d.second < room.size()) room[d.first] = room[d.second];
                }
                return span < room.size() ? room[span] : 0;
            }

            /* The drawing of a text operator with the edits of its spans. Text without edits is
               drawn in place, each run condensed or expanded to its width. Around edits, the runs
               of the operator are cut into groups at the gaps wider than an em (columns, tabs):
               the items of a group, pieces of text and the gaps between them, are drawn one after
               the other with one horizontal scaling, at most 100 %, so that a value longer than
               the text it replaces takes room from the rest of its group, and from the free room
               after it when the scaling would fall under 80 %. A value that continues in another
               operator keeps the width of its extent, and the part of a value drawn by another
               operator keeps its place. */
            std::string draw_operation(const std::vector<const text_operation*>& chain)
            {
                std::string out;
                if (chain.empty() || chain.front()->runs.empty()) return out;
                const std::size_t page = chain.front()->page;
                const bool visible = scanned_[page];
                const glyph_run& first = chain.front()->runs.front();
                const int k = standard_font_of(first.style);
                std::vector<const glyph_run*> all_runs;
                for (const text_operation* op : chain)
                    for (const glyph_run& r : op->runs) all_runs.push_back(&r);
                std::set<std::size_t> spans_here;
                for (const glyph_run* r : all_runs) spans_here.insert(r->span);
                auto inside = [&](std::size_t id) {
                    for (std::size_t sp : parts_[id]) if (!spans_here.count(sp)) return false;
                    return true;
                };
                // Where the text space of run r starts along the baseline of the first run.
                const double norm = first.tm.a * first.tm.a + first.tm.b * first.tm.b;
                auto shift = [&](const glyph_run& r) {
                    if (norm < 1e-12) return 0.0;
                    return ((r.tm.e - first.tm.e) * first.tm.a + (r.tm.f - first.tm.f) * first.tm.b) / norm;
                };
                struct item
                {
                    std::string text;
                    double a0, extent, width;
                    bool scaled, edited;
                    const glyph_run* run;   // for the text drawn in place
                    std::size_t span;
                };
                std::vector<item> items;
                std::map<std::size_t, std::size_t> drawn_here;      // replacement -> its item
                for (const glyph_run* rp : all_runs)
                {
                    const glyph_run& r = *rp;
                    const std::string& text = text_.pages[page].spans[r.span].text;
                    const double d = shift(r);
                    std::size_t q = 0;
                    auto plain = [&](std::size_t b, std::size_t e) {
                        if (e <= b) return;
                        const double a0 = advance_at(r, b, false);
                        items.push_back(item{ text.substr(b, e - b), d + a0, advance_at(r, e, true) - a0, 0, true, false,
                                              &r, r.span });
                    };
                    for (const span_edit& ed : edits_[page][r.span])
                    {
                        if (ed.b < q) continue;
                        plain(q, ed.b);
                        const double a0 = advance_at(r, ed.b, false);
                        const double own = advance_at(r, ed.e, true) - a0;
                        if (!ed.text.empty())
                        {
                            if (inside(ed.id))
                            {
                                drawn_here[ed.id] = items.size();
                                items.push_back(item{ ed.text, d + a0, own, 0, true, true, &r, r.span });
                            }
                            else
                                items.push_back(item{ ed.text, d + a0, own, ed.extent >= 0 ? ed.extent : own, false, true,
                                                      &r, r.span });
                        }
                        else if (!inside(ed.id))
                        {
                            items.push_back(item{ std::string(), d + a0, own, own, false, true, &r, r.span });
                        }
                        else if (drawn_here.count(ed.id))
                        {
                            // The rest of a value drawn by an earlier run of the operator: its item reaches here.
                            item& it = items[drawn_here[ed.id]];
                            it.extent = std::max(it.extent, d + a0 + own - it.a0);
                        }
                        q = ed.e;
                    }
                    plain(q, text.size());
                }
                if (items.empty()) return out;

                const double em = std::max(1e-6, std::abs(first.size * (first.em > 1e-6 ? first.em : 1)));
                std::size_t g0 = 0;
                while (g0 < items.size())
                {
                    std::size_t g1 = g0 + 1;
                    while (g1 < items.size() && items[g1].a0 - (items[g1 - 1].a0 + items[g1 - 1].extent) <= em)
                        ++g1;
                    bool edited = false;
                    for (std::size_t i = g0; i < g1; ++i) edited = edited || items[i].edited;
                    if (!edited)
                    {
                        // Text without edits, each piece in place.
                        for (std::size_t i = g0; i < g1; ++i)
                        {
                            const item& it = items[i];
                            const double local = it.a0 - shift(*it.run);
                            out += draw(*it.run, it.text, local, it.extent, standard_font_of(it.run->style), visible);
                        }
                        g0 = g1;
                        continue;
                    }
                    // The pieces of the group with the gaps between them.
                    std::vector<item> line;
                    for (std::size_t i = g0; i < g1; ++i)
                    {
                        if (i > g0)
                        {
                            const double gap = items[i].a0 - (items[i - 1].a0 + items[i - 1].extent);
                            if (std::abs(gap) > 1e-9)
                                line.push_back(item{ std::string(), 0, gap, gap, true, false, nullptr, 0 });
                        }
                        line.push_back(items[i]);
                    }
                    const double begin = items[g0].a0, total = items[g1 - 1].a0 + items[g1 - 1].extent - begin;
                    double fixed = 0, natural = 0;
                    for (item& it : line)
                    {
                        if (!it.scaled) fixed += it.width;
                        else if (!it.text.empty())
                            natural += (it.width = natural_width(*it.run, it.text, standard_font_of(it.run->style)));
                        else natural += it.width;
                    }
                    double scale = 1;
                    if (natural > 1e-9 && total - fixed > 1e-9)
                    {
                        scale = std::min(1.0, (total - fixed) / natural);
                        if (scale < 0.8)
                        {
                            double room = 0;
                            if (g1 < items.size())
                                room = std::max(0.0, items[g1].a0 - (begin + total) - 0.3 * em);
                            else
                            {
                                const pdf_text_span& s = text_.pages[page].spans[items[g1 - 1].span];
                                const glyph_run& last = *items[g1 - 1].run;
                                room = room_after(page, items[g1 - 1].span);
                                if (std::abs(s.width) > 1e-9 && !last.advances.empty())
                                    room *= std::abs(last.advances.back() / s.width);
                            }
                            scale = std::max(scale, std::min(0.8, (total - fixed + room) / natural));
                        }
                    }
                    /* One TJ operator when every item scales and one font draws the group, so that a
                       restoration reads the group as one line again. */
                    bool single = std::abs(first.size) > 1e-6;
                    std::vector<std::string> bytes(line.size());
                    for (std::size_t i = 0; i < line.size() && single; ++i)
                    {
                        if (!line[i].scaled) { single = false; break; }
                        if (line[i].text.empty()) continue;
                        if (standard_font_of(line[i].run->style) != k ||
                            std::abs(line[i].run->size * line[i].run->em - first.size * first.em) > 1e-6)
                        {
                            single = false;
                            break;
                        }
                        const std::size_t bad = report_.unencodable;
                        for (const encoded_run& e : encode(line[i].text, k))
                        {
                            if (e.font != k) single = false;
                            bytes[i] += e.bytes;
                        }
                        report_.unencodable = bad;
                    }
                    if (single)
                    {
                        const double size = first.size * (first.em > 1e-6 ? first.em : 1);
                        std::string array;
                        for (std::size_t i = 0; i < line.size(); ++i)
                        {
                            if (!line[i].text.empty()) array += pdf_hex_string(bytes[i]);
                            else array += " " + format_exact(-line[i].width * 1000.0 / size) + " ";
                        }
                        // The characters of the group are counted once, as draw() would.
                        for (const item& it : line) if (!it.text.empty()) encode(it.text, k);
                        const pdf_matrix m = pdf_matrix(1, 0, 0, 1, begin, 0) * first.tm;
                        out += std::string(visible ? " 0 Tr" : "") + " " + format_exact(100 * scale) + " Tz " +
                               format_exact(first.ts) + " Ts 0 Tc 0 Tw " + pdf_matrix_operands(m) + " Tm /DlibF" +
                               std::to_string(k) + " " + format_exact(size) + " Tf [" + array + "] TJ";
                        g0 = g1;
                        continue;
                    }
                    double x = begin;
                    for (const item& it : line)
                    {
                        const double w = it.scaled ? it.width * scale : it.width;
                        if (!it.text.empty())
                        {
                            // The item in its own font, placed along the baseline of the first run.
                            glyph_run r = *it.run;
                            r.tm = first.tm;
                            out += draw(r, it.text, x, w, standard_font_of(r.style), visible);
                        }
                        x += w;
                    }
                    g0 = g1;
                }
                return out;
            }

            // The edited text of a span, for the plans and the spans drawn without operator.
            std::string edited_text(std::size_t page, std::size_t span) const
            {
                const std::string& text = text_.pages[page].spans[span].text;
                std::string out;
                std::size_t q = 0;
                for (const span_edit& ed : edits_[page][span])
                {
                    if (ed.b < q) continue;
                    out += text.substr(q, ed.b - q) + ed.text;
                    q = ed.e;
                }
                return out + text.substr(std::min(q, text.size()));
            }

            // The line of the layout of each span, or -1.
            long long line_of(std::size_t page, std::size_t span)
            {
                if (lines_.size() != text_.pages.size())
                {
                    lines_.resize(text_.pages.size());
                    for (std::size_t p = 0; p < text_.pages.size(); ++p)
                    {
                        lines_[p].assign(text_.pages[p].spans.size(), -1);
                        for (std::size_t g = 0; g < sources_[p].segments.size(); ++g)
                            for (const auto& part : sources_[p].segments[g])
                                lines_[p][part.first] = static_cast<long long>(layout_[p][g].line);
                    }
                }
                return span < lines_[page].size() ? lines_[page][span] : -1;
            }

            /* Whether operator b draws the rest of the line of operator a, in the same stream: the
               same line of the layout, the same orientation and size, and its text starts where the
               text of a ends, within an em. */
            bool continues_line(const text_operation& a, const text_operation& b)
            {
                if (a.stream != b.stream || a.page != b.page || a.runs.empty() || b.runs.empty()) return false;
                const glyph_run& x = a.runs.back();
                const glyph_run& y = b.runs.front();
                const long long line = line_of(a.page, x.span);
                if (line < 0 || line != line_of(b.page, y.span)) return false;
                if (std::abs(x.tm.a - y.tm.a) > 1e-6 || std::abs(x.tm.b - y.tm.b) > 1e-6 ||
                    std::abs(x.tm.c - y.tm.c) > 1e-6 || std::abs(x.tm.d - y.tm.d) > 1e-6)
                    return false;
                const double em = std::abs(x.size * x.em);
                if (em < 1e-6 || std::abs(em - std::abs(y.size * y.em)) > 0.01 * em) return false;
                const double norm = x.tm.a * x.tm.a + x.tm.b * x.tm.b;
                if (norm < 1e-12 || x.advances.empty()) return false;
                const double end = x.advances.back();
                const double start = ((y.tm.e - x.tm.e) * x.tm.a + (y.tm.f - x.tm.f) * x.tm.b) / norm;
                return start - end >= -0.5 * em && start - end <= em;
            }

            void plan_operations()
            {
                // Every text that will be drawn, for the characters of the extended font.
                for (std::size_t p = 0; p < text_.pages.size(); ++p)
                    for (std::size_t i = 0; i < text_.pages[p].spans.size(); ++i)
                        collect_extra_chars(edited_text(p, i));
                /* An operator of a form drawn several times is planned from the use with the most
                   edits: each use reads the same text. */
                std::map<op_key, std::pair<std::size_t, std::size_t>> best;    // key -> (edited bytes, operation)
                for (std::size_t i = 0; i < rec_.operations.size(); ++i)
                {
                    const text_operation& op = rec_.operations[i];
                    std::size_t edited = 0;
                    for (const glyph_run& r : op.runs)
                        for (const span_edit& ed : edits_[op.page][r.span]) edited += ed.e - ed.b + 1;
                    const op_key key(op.stream, op.begin);
                    const auto it = best.find(key);
                    if (it == best.end() || edited > it->second.first) best[key] = std::make_pair(edited, i);
                }
                for (const auto& b : best)
                {
                    if (b.first.first < 0) continue;
                    plans_[b.first.first][b.first.second] =
                        draw_operation(std::vector<const text_operation*>(1, &rec_.operations[b.second.second]));
                }
                /* The operators of a page that draw one line one after the other are laid out
                   together when one of them is edited, so that a value takes room from the rest of
                   its line: "Dr Yannick Le Floc'h" in bold, then ", médecin du travail". The first
                   operator draws the line, the others nothing. */
                std::vector<const text_operation*> chain;
                auto flush = [&]() {
                    if (chain.empty()) return;
                    bool edited = false;
                    for (const text_operation* op : chain)
                        for (const glyph_run& r : op->runs)
                            if (!edits_[op->page][r.span].empty()) edited = true;
                    if (edited && chain.size() > 1)
                    {
                        plans_[chain.front()->stream][chain.front()->begin] = draw_operation(chain);
                        for (std::size_t i = 1; i < chain.size(); ++i) plans_[chain[i]->stream][chain[i]->begin] = "";
                    }
                    else
                        for (const text_operation* op : chain)
                            plans_[op->stream][op->begin] = draw_operation(std::vector<const text_operation*>(1, op));
                    chain.clear();
                };
                for (const text_operation& op : rec_.operations)
                {
                    if (op.stream >= 0) continue;
                    if (!chain.empty() && !continues_line(*chain.back(), op)) flush();
                    chain.push_back(&op);
                }
                flush();
            }

            // ------------------------------------------------------------------------------
            // The texts written in the boxes of the pictures.

            /* The picture of an image object as a file: a JPEG as it is stored, the other
               pictures of 8 bits per component in gray, RGB or CMYK, indexed or not, written as
               PNG; empty for the others (JPEG 2000, JBIG2, CCITT, masks) and for pictures larger
               than 25 million pixels. */
            document_image picture_of(int number, std::size_t page)
            {
                document_image img;
                img.location = "page " + std::to_string(page + 1);
                const object& o = doc_.get_object(number);
                if (!o.is(object::stream_k) || !o.buffer) return img;
                std::vector<std::string> filters;
                if (const object* f = doc_.resolve_key(o, "Filter"))
                {
                    if (f->is(object::name_k)) filters.push_back(f->str);
                    else if (f->is(object::array_k))
                        for (const object& e : f->arr)
                        {
                            const object& r = doc_.resolve(e);
                            if (!r.is(object::name_k)) return img;
                            filters.push_back(r.str);
                        }
                }
                const object* mask = doc_.resolve_key(o, "ImageMask");
                if (mask && mask->is(object::bool_k) && mask->b) return img;
                if (filters.size() == 1 && (filters[0] == "DCTDecode" || filters[0] == "DCT"))
                {
                    img.bytes = o.buffer->substr(o.data_begin, o.data_end - o.data_begin);
                    if (image_media_type(img.bytes) == "image/jpeg") img.media_type = "image/jpeg";
                    else img.bytes.clear();
                    return img;
                }
                for (const std::string& n : filters)
                    if (n == "DCTDecode" || n == "DCT" || n == "JPXDecode" || n == "JBIG2Decode" ||
                        n == "CCITTFaxDecode" || n == "CCF")
                        return img;
                auto number_of = [this, &o](const char* key) {
                    const object* v = doc_.resolve_key(o, key);
                    return v && v->is(object::num_k) ? static_cast<long>(v->num) : 0L;
                };
                const long w = number_of("Width"), h = number_of("Height");
                if (w <= 0 || h <= 0 || static_cast<double>(w) * h > 25e6 || number_of("BitsPerComponent") != 8)
                    return img;
                /*
                    The components of the color space, and the palette of an indexed one.  The base of
                    an indexed space is a plain space: depth bounds the spaces that name themselves.
                */
                int n = 0;
                std::string palette;
                int palette_n = 0;
                std::function<int(const object&, int)> components = [&](const object& cs, int depth) -> int {
                    if (depth > 2) return 0;
                    const object& c = doc_.resolve(cs);
                    if (c.is(object::name_k))
                    {
                        if (c.str == "DeviceGray" || c.str == "G" || c.str == "CalGray") return 1;
                        if (c.str == "DeviceRGB" || c.str == "RGB" || c.str == "CalRGB") return 3;
                        if (c.str == "DeviceCMYK" || c.str == "CMYK") return 4;
                        return 0;
                    }
                    if (!c.is(object::array_k) || c.arr.empty()) return 0;
                    const object& family = doc_.resolve(c.arr[0]);
                    if (!family.is(object::name_k)) return 0;
                    if (family.str == "CalGray") return 1;
                    if (family.str == "CalRGB" || family.str == "Lab") return family.str == "Lab" ? 0 : 3;
                    if (family.str == "ICCBased" && c.arr.size() > 1)
                    {
                        const object& icc = doc_.resolve(c.arr[1]);
                        const object* nn = doc_.resolve_key(icc, "N");
                        return nn && nn->is(object::num_k) ? static_cast<int>(nn->num) : 0;
                    }
                    if ((family.str == "Indexed" || family.str == "I") && c.arr.size() > 3 && palette.empty())
                    {
                        palette_n = components(c.arr[1], depth + 1);
                        if (palette_n < 0) palette_n = 0;
                        const object& table = doc_.resolve(c.arr[3]);
                        palette = table.is(object::stream_k) ? doc_.stream_data(table) :
                                  table.is(object::str_k) ? table.str : std::string();
                        return palette_n > 0 && !palette.empty() ? -1 : 0;
                    }
                    return 0;
                };
                if (const object* cs = doc_.resolve_key(o, "ColorSpace")) n = components(*cs, 0);
                if (n == 0 || n == 2 || n > 4) return img;
                const std::string data = doc_.stream_data(o);
                const std::size_t count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
                const std::size_t in = n < 0 ? 1 : static_cast<std::size_t>(n);
                if (data.size() < count * in) return img;
                const int out_n = (n < 0 ? palette_n : n) == 1 ? 1 : 3;
                std::string pixels(count * static_cast<std::size_t>(out_n), '\0');
                auto put = [&](std::size_t k, const unsigned char* c, int cn) {
                    unsigned char rgb[3];
                    if (cn == 1) { pixels[k] = static_cast<char>(c[0]); return; }
                    if (cn == 3) { rgb[0] = c[0]; rgb[1] = c[1]; rgb[2] = c[2]; }
                    else
                    {
                        for (int q = 0; q < 3; ++q)
                            rgb[q] = static_cast<unsigned char>((255 - c[q]) * (255 - c[3]) / 255);
                    }
                    for (int q = 0; q < 3; ++q) pixels[k * 3 + static_cast<std::size_t>(q)] = static_cast<char>(rgb[q]);
                };
                const unsigned char* d = reinterpret_cast<const unsigned char*>(data.data());
                const unsigned char* pal = reinterpret_cast<const unsigned char*>(palette.data());
                for (std::size_t k = 0; k < count; ++k)
                {
                    if (n > 0)
                    {
                        put(k, d + k * in, n);
                        continue;
                    }
                    const std::size_t at = static_cast<std::size_t>(d[k]) * static_cast<std::size_t>(palette_n);
                    unsigned char none[4] = { 0, 0, 0, 0 };
                    put(k, at + static_cast<std::size_t>(palette_n) <= palette.size() ? pal + at : none, palette_n);
                }
                img.bytes = encode_png(pixels, w, h, out_n);
                if (!img.bytes.empty()) img.media_type = "image/png";
                return img;
            }

            // Width of text drawn with Helvetica at size.
            double helvetica_text_width(const std::string& text, double size)
            {
                const std::size_t bad = report_.unencodable;
                double w = 0;
                for (const encoded_run& e : encode(text, 0)) w += e.width;
                report_.unencodable = bad;
                return w / 1000.0 * size;
            }

            /* The lines of text within width at size: cut between words, and inside a word
               wider than a line. */
            std::vector<std::string> wrap_text(const std::string& text, double width, double size)
            {
                std::vector<std::string> lines;
                std::string line;
                std::size_t i = 0;
                while (i < text.size())
                {
                    while (i < text.size() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t'))
                        ++i;
                    if (i >= text.size()) break;
                    std::size_t j = i;
                    while (j < text.size() && text[j] != ' ' && text[j] != '\n' && text[j] != '\r' && text[j] != '\t')
                        ++j;
                    std::string word = text.substr(i, j - i);
                    i = j;
                    const std::string joined = line.empty() ? word : line + " " + word;
                    if (helvetica_text_width(joined, size) <= width)
                    {
                        line = joined;
                        continue;
                    }
                    if (!line.empty()) lines.push_back(line);
                    line.clear();
                    // A word wider than the line goes over several, cut between characters.
                    std::string piece;
                    for (std::size_t q = 0; q < word.size();)
                    {
                        const std::size_t from = q;
                        utf8_get(word, q);
                        const std::string c = word.substr(from, q - from);
                        if (!piece.empty() && helvetica_text_width(piece + c, size) > width)
                        {
                            lines.push_back(piece);
                            piece.clear();
                        }
                        piece += c;
                    }
                    line = piece;
                }
                if (!line.empty()) lines.push_back(line);
                return lines;
            }

            /* The operators that write text over the box of a picture drawn with matrix ctm,
               after its Do operator: centered lines of Helvetica as large as fit, from 10 points
               down to 3, the last line shortened with "..." when even 3 points do not fit. */
            std::string caption(const std::string& text, const pdf_matrix& ctm)
            {
                const double w = std::sqrt(ctm.a * ctm.a + ctm.b * ctm.b);
                const double h = std::sqrt(ctm.c * ctm.c + ctm.d * ctm.d);
                if (w < 24 || h < 10) return std::string();
                const double margin = std::min(4.0, std::min(w, h) / 10);
                const double room_w = w - 2 * margin, room_h = h - 2 * margin;
                double size = 10;
                std::vector<std::string> lines;
                for (; size >= 3; size -= 0.5)
                {
                    lines = wrap_text(text, room_w, size);
                    if (lines.size() * size * 1.2 <= room_h) break;
                }
                if (size < 3)
                {
                    size = 3;
                    lines = wrap_text(text, room_w, size);
                    const std::size_t keep = std::max<std::size_t>(1, static_cast<std::size_t>(room_h / (size * 1.2)));
                    if (lines.size() > keep)
                    {
                        lines.resize(keep);
                        std::string& last = lines.back();
                        while (!last.empty() && helvetica_text_width(last + "...", size) > room_w)
                        {
                            std::size_t cut = last.size() - 1;
                            while (cut > 0 && (static_cast<unsigned char>(last[cut]) & 0xC0) == 0x80) --cut;
                            last.erase(cut);
                        }
                        last += "...";
                    }
                }
                if (lines.empty()) return std::string();
                // Units of the page inside the box, upright even when the picture is flipped.
                const bool flipped = ctm.a * ctm.d - ctm.b * ctm.c < 0;
                std::string out = "q " + format_exact(1 / w) + " 0 0 " + format_exact((flipped ? -1 : 1) / h) +
                                  " 0 " + (flipped ? "1" : "0") + " cm BT 0 Tr 100 Tz 0 Ts 0 Tc 0 Tw 0.25 g";
                const double lead = size * 1.2;
                double y = h / 2 + lead * static_cast<double>(lines.size()) / 2 - size;
                for (const std::string& l : lines)
                {
                    const double x = std::max(margin, (w - helvetica_text_width(l, size)) / 2);
                    out += " 1 0 0 1 " + format_exact(x) + " " + format_exact(y) + " Tm";
                    for (const encoded_run& e : encode(l, 0))
                        out += " /DlibF" + std::to_string(e.font) + " " + format_exact(size) + " Tf " +
                               pdf_hex_string(e.bytes) + " Tj";
                    y -= lead;
                }
                return out + " ET Q\n";
            }

            // The texts of the describer, each written over the gray box of its picture.
            void plan_captions(const document_image_describer& describer)
            {
                std::map<int, std::string> texts;
                for (const image_draw& d : rec_.images)
                {
                    if (d.object <= 0 || captions_[d.stream].count(d.at)) continue;
                    auto t = texts.find(d.object);
                    if (t == texts.end())
                    {
                        const document_image img = picture_of(d.object, d.page);
                        t = texts.insert(std::make_pair(d.object, img.bytes.empty() ? std::string() :
                                                                                     describer(img))).first;
                        collect_extra_chars(t->second);
                    }
                    if (t->second.empty()) continue;
                    const std::string ops = caption(t->second, d.ctm);
                    if (ops.empty()) continue;
                    captions_[d.stream][d.at] = ops;
                    ++report_.described_images;
                }
            }

            // ------------------------------------------------------------------------------
            // Content streams.

            /* A content stream written again: text operators replaced by their plan (dropped
               without one), font selections dropped, inline pictures replaced by a gray pixel,
               alternate texts of marked content left out, comments removed. */
            std::string rewrite_content(const std::string& content, long long stream)
            {
                static const std::map<std::size_t, std::string> no_plans;
                const auto pit = plans_.find(stream);
                const std::map<std::size_t, std::string>& plans = pit == plans_.end() ? no_plans : pit->second;
                const auto cit = captions_.find(stream);
                const std::map<std::size_t, std::string>& captions = cit == captions_.end() ? no_plans : cit->second;
                std::string out;
                lexer lx(content);
                std::vector<object> operands;
                std::vector<std::string> raw;
                std::size_t op_begin = 0, guard = 0;
                while (!lx.at_end() && ++guard < 50000000)
                {
                    const std::size_t t0 = lx.pos();
                    object o;
                    try { o = lx.read(false); }
                    catch (pdf_error&) { break; }
                    const std::string token = content.substr(t0, lx.pos() - t0);
                    if (!o.is(object::op_k))
                    {
                        operands.push_back(o);
                        raw.push_back(token);
                        if (operands.size() > 4096) { operands.clear(); raw.clear(); }
                        continue;
                    }
                    const std::string& op = o.str;
                    if (op == "Tj" || op == "TJ" || op == "'" || op == "\"")
                    {
                        const auto it = plans.find(op_begin);
                        if (it != plans.end()) out += it->second + "\n";
                    }
                    else if (op == "Tf")
                    {
                    }
                    else if (op == "BI")
                    {
                        skip_inline_image(lx, content);
                        out += "BI /W 1 /H 1 /BPC 8 /CS /G ID \xE6 EI\n";
                        ++report_.images;
                    }
                    else if (op == "BDC" && operands.size() >= 2 && operands[1].is(object::dict_k))
                    {
                        out += value(operands[0], 0) + " " + value(operands[1], 0) + " BDC\n";
                    }
                    else if (op == "Do" && captions.count(op_begin))
                    {
                        for (const std::string& r : raw) out += r + " ";
                        out += "Do\n" + captions.at(op_begin);
                    }
                    else
                    {
                        for (const std::string& r : raw) out += r + " ";
                        out += op + "\n";
                    }
                    operands.clear();
                    raw.clear();
                    op_begin = lx.pos();
                }
                return out;
            }

            // An inline picture, its dictionary then its data up to EI between white space.
            static void skip_inline_image(lexer& lx, const std::string& content)
            {
                while (!lx.at_end())
                {
                    object o;
                    try { o = lx.read(false); }
                    catch (pdf_error&) { return; }
                    if (o.is(object::op_k) && o.str == "ID") break;
                }
                std::size_t p = lx.pos() + 1;
                while (p + 2 <= content.size())
                {
                    if (content[p] == 'E' && content[p + 1] == 'I' && is_ws(content[p - 1]) &&
                        (p + 2 == content.size() || is_ws(content[p + 2])))
                    {
                        lx.seek(p + 2);
                        return;
                    }
                    ++p;
                }
                lx.seek(content.size());
            }

            // ------------------------------------------------------------------------------
            // Objects.

            int reserve()
            {
                bodies_.push_back(std::string());
                return static_cast<int>(bodies_.size());
            }

            int add(const std::string& body)
            {
                bodies_.push_back(body);
                return static_cast<int>(bodies_.size());
            }

            void set(int n, const std::string& body) { bodies_[static_cast<std::size_t>(n - 1)] = body; }

            int copy(int old)
            {
                const auto it = renumber_.find(old);
                if (it != renumber_.end()) return it->second;
                const int n = reserve();
                renumber_[old] = n;
                queue_.push_back(std::make_pair(old, n));
                return n;
            }

            static std::string stream_body(const std::string& dict_entries, const std::string& data, bool compress)
            {
                const std::string packed = compress ? zlib_compress(data) : data;
                return "<<" + dict_entries + (compress ? " /Filter /FlateDecode" : "") + " /Length " +
                       std::to_string(packed.size()) + " >>\nstream\n" + packed + "\nendstream";
            }

            // A direct value; the objects it refers to are copied.
            std::string value(const object& o, int depth)
            {
                if (depth > 64) return "null";
                switch (o.kind)
                {
                    case object::null_k: return "null";
                    case object::bool_k: return o.b ? "true" : "false";
                    case object::num_k: return format_exact(o.num);
                    case object::str_k: return pdf_hex_string(o.str);
                    case object::name_k: return pdf_name(o.str);
                    case object::ref_k:
                    {
                        const object& target = doc_.resolve(o);
                        if (target.is(object::null_k)) return "null";
                        // A font is never copied: the text is drawn with the standard fonts.
                        if (target.is(object::dict_k))
                            if (const object* t = target.get("Type"))
                                if (t->is(object::name_k) && t->str == "Font") return "null";
                        return std::to_string(copy(o.ref)) + " 0 R";
                    }
                    case object::array_k:
                    {
                        std::string s = "[";
                        for (const object& e : o.arr) s += value(e, depth + 1) + " ";
                        return s + "]";
                    }
                    case object::dict_k:
                    {
                        std::string s = "<<";
                        for (const auto& kv : o.dict)
                            if (!dropped_key(kv.first)) s += " " + pdf_name(kv.first) + " " + value(kv.second, depth + 1);
                        return s + " >>";
                    }
                    default: return "null";
                }
            }

            std::string fonts_entry() const
            {
                std::string s = " /Font <<";
                for (int k = 0; k < 12; ++k)
                    s += " /DlibF" + std::to_string(k) + " " + std::to_string(font_base_ + k) + " 0 R";
                if (!extra_chars_.empty()) s += " /DlibF12 " + std::to_string(font_base_ + 12) + " 0 R";
                return s + " >>";
            }

            /* Resources of a rewritten stream: the standard fonts in the place of the original
               ones, the font of graphics states left out, more pictures when given. */
            std::string resources(const object* res, const std::string& more_xobjects)
            {
                std::string s = "<<";
                std::string xobjects = more_xobjects;
                if (res && res->is(object::dict_k))
                {
                    for (const auto& kv : res->dict)
                    {
                        if (kv.first == "Font") continue;
                        if (kv.first == "XObject")
                        {
                            const object& x = doc_.resolve(kv.second);
                            if (x.is(object::dict_k))
                                for (const auto& e : x.dict) xobjects += " " + pdf_name(e.first) + " " + value(e.second, 1);
                            continue;
                        }
                        if (kv.first == "ExtGState")
                        {
                            const object& g = doc_.resolve(kv.second);
                            if (!g.is(object::dict_k)) continue;
                            s += " /ExtGState <<";
                            for (const auto& e : g.dict)
                            {
                                const object& gs = doc_.resolve(e.second);
                                if (!gs.is(object::dict_k)) continue;
                                std::string d = "<<";
                                for (const auto& f : gs.dict)
                                    if (f.first != "Font" && !dropped_key(f.first))
                                        d += " " + pdf_name(f.first) + " " + value(f.second, 2);
                                s += " " + pdf_name(e.first) + " " + d + " >>";
                            }
                            s += " >>";
                            continue;
                        }
                        s += " " + pdf_name(kv.first) + " " + value(kv.second, 1);
                    }
                }
                if (!xobjects.empty()) s += " /XObject <<" + xobjects + " >>";
                return s + fonts_entry() + " >>";
            }

            // An object of the original, written again; streams of drawing are rewritten.
            void write_copy(int old, int n)
            {
                const object& o = doc_.get_object(old);
                if (o.is(object::stream_k))
                {
                    const object* subtype = doc_.resolve_key(o, "Subtype");
                    const std::string st = subtype && subtype->is(object::name_k) ? subtype->str : "";
                    const object* pattern = doc_.resolve_key(o, "PatternType");
                    if (st == "Image")
                    {
                        ++report_.images;
                        set(n, stream_body(" /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray "
                                           "/BitsPerComponent 8", "\xE6", false));
                        return;
                    }
                    if (st == "Form" || (pattern && pattern->is(object::num_k) && pattern->num == 1))
                    {
                        std::string d;
                        for (const auto& kv : o.dict)
                        {
                            if (kv.first == "Length" || kv.first == "Filter" || kv.first == "DecodeParms" ||
                                kv.first == "Resources" || dropped_key(kv.first))
                                continue;
                            d += " " + pdf_name(kv.first) + " " + value(kv.second, 1);
                        }
                        if (o.get("Resources")) d += " /Resources " + resources(doc_.resolve_key(o, "Resources"), "");
                        set(n, stream_body(d, rewrite_content(doc_.stream_data(o), old), true));
                        return;
                    }
                    if (const object* t = doc_.resolve_key(o, "Type"))
                        if (t->is(object::name_k) && t->str == "Metadata")
                        {
                            set(n, stream_body("", "", false));
                            return;
                        }
                    // Other streams (functions, color profiles, shadings) as they are.
                    std::string d;
                    for (const auto& kv : o.dict)
                    {
                        if (kv.first == "Length" || dropped_key(kv.first)) continue;
                        d += " " + pdf_name(kv.first) + " " + value(kv.second, 1);
                    }
                    set(n, stream_body(d, o.buffer->substr(o.data_begin, o.data_end - o.data_begin), false));
                    return;
                }
                set(n, value(o, 0));
            }

            void drain()
            {
                while (next_ < queue_.size())
                {
                    const std::pair<int, int> q = queue_[next_++];
                    write_copy(q.first, q.second);
                }
            }

            void write_extended_font()
            {
                std::string diff = "33";
                for (std::uint32_t cp : extra_chars_) diff += std::string(" /") + latin_extended_a_name(cp);
                std::string cmap =
                    "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
                    "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
                    "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
                    "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n";
                for (std::size_t i = 0; i < extra_chars_.size(); i += 100)
                {
                    const std::size_t n = std::min<std::size_t>(100, extra_chars_.size() - i);
                    cmap += std::to_string(n) + " beginbfchar\n";
                    for (std::size_t k = i; k < i + n; ++k)
                    {
                        char buf[32];
                        std::snprintf(buf, sizeof(buf), "<%02X> <%04X>\n", static_cast<unsigned>(33 + k),
                                      static_cast<unsigned>(extra_chars_[k]));
                        cmap += buf;
                    }
                    cmap += "endbfchar\n";
                }
                cmap += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n";
                const int font = reserve();
                const int to_unicode = add(stream_body("", cmap, true));
                std::string widths;
                for (std::uint32_t cp : extra_chars_) widths += format_exact(helvetica_width(cp)) + " ";
                set(font, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding << /Type /Encoding "
                          "/Differences [" + diff + "] >> /FirstChar 33 /LastChar " +
                          std::to_string(32 + extra_chars_.size()) + " /Widths [" + widths + "] /ToUnicode " +
                          std::to_string(to_unicode) + " 0 R >>");
            }

            static std::string box(document& doc, const object* b)
            {
                if (!b || !b->is(object::array_k) || b->arr.size() < 4) return std::string();
                std::string s = "[";
                for (std::size_t k = 0; k < 4; ++k) s += format_exact(doc.resolve(b->arr[k]).num) + " ";
                return s + "]";
            }

            int write_page(std::size_t p, int parent)
            {
                const page_leaf& leaf = leaves_[p];
                const object& page = *leaf.dict;
                std::string content = "q\n" + rewrite_content(page_content(doc_, page), -static_cast<long long>(p + 1)) +
                                      "Q\n";
                // The appearances of form fields, free texts and stamps, drawn with the page.
                std::string xobjects;
                int k = 0;
                for (const drawn_annotation& a : rec_.annotations)
                {
                    if (a.page != p) continue;
                    const std::string name = "DlibA" + std::to_string(k++);
                    xobjects += " /" + name + " " + std::to_string(copy(a.object)) + " 0 R";
                    content += "q " + pdf_matrix_operands(a.to_rect) + " cm /" + name + " Do Q\n";
                    ++report_.drawn_annotations;
                }
                if (const object* annots = doc_.resolve_key(page, "Annots"))
                    if (annots->is(object::array_k) && annots->arr.size() > static_cast<std::size_t>(k))
                        report_.annotations += annots->arr.size() - static_cast<std::size_t>(k);
                // Text without operator: the value of a form field without appearance.
                std::string free;
                const std::vector<pdf_text_span>& spans = text_.pages[p].spans;
                for (std::size_t i = 0; i < spans.size(); ++i)
                {
                    if (runs_[p][i]) continue;
                    const pdf_text_span& s = spans[i];
                    const std::string text = edited_text(p, i);
                    glyph_run r;
                    const double rad = s.angle * pi / 180.0;
                    r.size = s.font_size > 0.5 ? s.font_size : 10;
                    r.tm = pdf_matrix(std::cos(rad), std::sin(rad), -std::sin(rad), std::cos(rad), s.x, s.y);
                    free += draw(r, text, 0, std::max(s.width, 1.0), 0, false);
                }
                if (!free.empty()) content += "BT" + free + " ET\n";

                std::string d = "<< /Type /Page /Parent " + std::to_string(parent) + " 0 R";
                const std::string media = box(doc_, leaf.mediabox);
                d += " /MediaBox " + (media.empty() ? std::string("[0 0 612 792]") : media);
                const std::string crop = box(doc_, leaf.cropbox);
                if (!crop.empty()) d += " /CropBox " + crop;
                for (const char* key : { "BleedBox", "TrimBox", "ArtBox" })
                {
                    const std::string b = box(doc_, doc_.resolve_key(page, key));
                    if (!b.empty()) d += " /" + std::string(key) + " " + b;
                }
                if (leaf.rotate % 360 != 0) d += " /Rotate " + std::to_string(((leaf.rotate % 360) + 360) % 360);
                for (const char* key : { "UserUnit", "Group" })
                    if (const object* v = page.get(key)) d += " /" + std::string(key) + " " + value(*v, 1);
                d += " /Resources " + resources(leaf.resources, xobjects);
                const int contents = add(stream_body("", content, true));
                d += " /Contents " + std::to_string(contents) + " 0 R >>";
                return add(d);
            }

            std::string file(int catalog) const
            {
                std::string out = "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n";
                std::vector<std::size_t> offsets;
                for (std::size_t i = 0; i < bodies_.size(); ++i)
                {
                    offsets.push_back(out.size());
                    out += std::to_string(i + 1) + " 0 obj\n" + (bodies_[i].empty() ? "null" : bodies_[i]) +
                           "\nendobj\n";
                }
                const std::size_t xref = out.size();
                out += "xref\n0 " + std::to_string(bodies_.size() + 1) + "\n0000000000 65535 f \n";
                for (std::size_t off : offsets)
                {
                    char buf[32];
                    std::snprintf(buf, sizeof(buf), "%010lu 00000 n \n", static_cast<unsigned long>(off));
                    out += buf;
                }
                out += "trailer\n<< /Size " + std::to_string(bodies_.size() + 1) + " /Root " + std::to_string(catalog) +
                       " 0 R >>\nstartxref\n" + std::to_string(xref) + "\n%%EOF\n";
                return out;
            }

            document doc_;
            pdf_rewrite_report& report_;
            text_recording rec_;
            std::vector<page_leaf> leaves_;
            pdf_text_document text_;
            std::vector<std::vector<pdf_text_segment>> layout_;
            std::vector<page_layout_source> sources_;
            std::vector<std::vector<const glyph_run*>> runs_;
            std::vector<std::vector<std::vector<span_edit>>> edits_;
            std::vector<std::vector<std::size_t>> parts_;     // spans of each replacement
            std::vector<bool> scanned_;
            std::vector<std::vector<double>> room_;
            std::vector<std::vector<long long>> lines_;
            std::map<long long, std::map<std::size_t, std::string>> plans_;
            std::map<long long, std::map<std::size_t, std::string>> captions_;   // after the Do of a picture
            std::map<std::uint32_t, int> extra_codes_;
            std::vector<std::uint32_t> extra_chars_;
            std::vector<std::string> bodies_;
            std::map<int, int> renumber_;
            std::vector<std::pair<int, int>> queue_;
            std::size_t next_ = 0;
            int font_base_ = 0;
        };
    }

// ----------------------------------------------------------------------------------------

    inline std::string rewrite_pdf_text(
        const std::string& pdf_bytes,
        const std::vector<pdf_text_replacement>& replacements,
        pdf_rewrite_report* report = nullptr,
        const document_image_describer& describer = document_image_describer()
    )
    {
        pdf_rewrite_report local;
        pdf_impl::pdf_rewriter w(pdf_bytes, replacements, report ? *report : local, describer);
        return w.write();
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PDF_REWRITE_H_
