// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_DATASET_H_
#define DLIB_PII_DATASET_H_

#include "pii_dataset_abstract.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "../assert.h"
#include "../error.h"
#include "pii_types.h"
#include "unicode_text.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pii_example
    {
        std::string text;
        std::vector<pii_span> spans;
    };

    struct pii_score
    {
        std::string label;
        std::size_t true_positives = 0;
        std::size_t false_positives = 0;
        std::size_t false_negatives = 0;

        double precision() const;
        double recall() const;
        double f1() const;
    };

    struct pii_evaluation
    {
        std::vector<pii_score> labels;
        pii_score overall;

        std::string to_string() const;
    };

    enum class span_match
    {
        exact,
        overlap
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        inline bool utf8_boundary(const std::string& s, std::size_t i)
        {
            return i == 0 || i >= s.size() || (static_cast<unsigned char>(s[i]) & 0xC0) != 0x80;
        }

        inline std::string escape_dataset_text(const std::string& s)
        {
            std::string out;
            out.reserve(s.size() + 8);
            // A text starting with '#' would read as a comment line.
            if (!s.empty() && s[0] == '#') out += "\\";
            for (char c : s)
            {
                switch (c)
                {
                    case '\\': out += "\\\\"; break;
                    case '\t': out += "\\t"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    default: out.push_back(c);
                }
            }
            return out;
        }
    }

// ----------------------------------------------------------------------------------------

    inline pii_example parse_pii_example(
        const std::string& raw_line
    )
    {
        std::string line = raw_line;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t tab = line.find('\t');
        if (tab != std::string::npos && line.find('\t', tab + 1) != std::string::npos)
            throw error("dataset line: more than one tab, the tabs of the text must be written \\t");
        const std::string escaped = tab == std::string::npos ? line : line.substr(0, tab);
        const std::string spans = tab == std::string::npos ? std::string() : line.substr(tab + 1);

        pii_example ex;
        ex.text.reserve(escaped.size());
        for (std::size_t i = 0; i < escaped.size(); ++i)
        {
            const char c = escaped[i];
            if (c != '\\')
            {
                ex.text.push_back(c);
                continue;
            }
            if (i + 1 >= escaped.size()) throw error("dataset line: dangling backslash");
            const char e = escaped[++i];
            if (e == '\\') ex.text.push_back('\\');
            else if (e == 't') ex.text.push_back('\t');
            else if (e == 'n') ex.text.push_back('\n');
            else if (e == 'r') ex.text.push_back('\r');
            else if (e == '#' && i == 1) ex.text.push_back('#');
            else throw error(std::string("dataset line: unknown escape \\") + e);
        }

        std::size_t p = 0;
        while (p < spans.size())
        {
            std::size_t q = spans.find(',', p);
            if (q == std::string::npos) q = spans.size();
            const std::string item = spans.substr(p, q - p);
            p = q + 1;
            if (item.empty()) continue;
            const std::size_t c1 = item.find(':');
            const std::size_t c2 = c1 == std::string::npos ? std::string::npos : item.find(':', c1 + 1);
            if (c2 == std::string::npos || c2 + 1 >= item.size())
                throw error("dataset line: malformed span \"" + item + "\"");
            pii_span s;
            try
            {
                std::size_t used = 0;
                s.begin = std::stoul(item.substr(0, c1), &used);
                if (used != c1) throw std::invalid_argument("begin");
                s.end = std::stoul(item.substr(c1 + 1, c2 - c1 - 1), &used);
                if (used != c2 - c1 - 1) throw std::invalid_argument("end");
            }
            catch (std::exception&)
            {
                throw error("dataset line: malformed span \"" + item + "\"");
            }
            s.label = item.substr(c2 + 1);
            s.type = pii_type_from_label(s.label);
            s.confidence = 1;
            if (s.type == pii_type::none)
                throw error("dataset line: unknown label in \"" + item + "\"");
            if (s.end <= s.begin || s.end > ex.text.size())
                throw error("dataset line: span \"" + item + "\" outside the text");
            if (!privacy_impl::utf8_boundary(ex.text, s.begin) || !privacy_impl::utf8_boundary(ex.text, s.end))
                throw error("dataset line: span \"" + item + "\" cuts a UTF-8 character");
            if (ex.text.find_first_of("\t\n\r", s.begin) < s.end)
                throw error("dataset line: span \"" + item + "\" crosses a tab or a line break");
            ex.spans.push_back(s);
        }
        std::sort(ex.spans.begin(), ex.spans.end(),
                  [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
        for (std::size_t k = 1; k < ex.spans.size(); ++k)
            if (ex.spans[k].begin < ex.spans[k - 1].end)
                throw error("dataset line: overlapping spans");
        return ex;
    }

    inline std::string format_pii_example(
        const pii_example& example
    )
    {
        std::string line = privacy_impl::escape_dataset_text(example.text);
        line.push_back('\t');
        for (std::size_t k = 0; k < example.spans.size(); ++k)
        {
            const pii_span& s = example.spans[k];
            if (k) line.push_back(',');
            line += std::to_string(s.begin) + ":" + std::to_string(s.end) + ":" + s.label;
        }
        return line;
    }

    inline std::vector<pii_example> load_pii_dataset(
        const std::string& filename
    )
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in) throw error("cannot read " + filename);
        std::vector<pii_example> examples;
        std::string line;
        std::size_t number = 0;
        while (std::getline(in, line))
        {
            ++number;
            // A byte order mark may open the file.
            if (number == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
            if (line.empty() || line[0] == '#' || line == "\r") continue;
            try
            {
                examples.push_back(parse_pii_example(line));
            }
            catch (error& e)
            {
                throw error(filename + ":" + std::to_string(number) + ": " + e.what());
            }
        }
        return examples;
    }

    inline void save_pii_dataset(
        const std::vector<pii_example>& examples,
        const std::string& filename
    )
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out) throw error("cannot write " + filename);
        out << "# dlib pii dataset: text<TAB>begin:end:LABEL,... (byte offsets)\n";
        for (const pii_example& ex : examples)
            out << format_pii_example(ex) << '\n';
        if (!out) throw error("cannot write " + filename);
    }

// ----------------------------------------------------------------------------------------

    inline unsigned long pii_bio_class_count()
    {
        return 1 + 2 * static_cast<unsigned long>(pii_fine_labels().size());
    }

    inline unsigned long pii_bio_class(
        const std::string& label,
        bool inside
    )
    {
        const std::vector<std::string>& labels = pii_fine_labels();
        for (std::size_t k = 0; k < labels.size(); ++k)
            if (labels[k] == label)
                return static_cast<unsigned long>(1 + 2 * k + (inside ? 1 : 0));
        return 0;
    }

    inline std::string pii_bio_class_label(
        unsigned long cls
    )
    {
        if (cls == 0 || cls >= pii_bio_class_count()) return std::string();
        return pii_fine_labels()[(cls - 1) / 2];
    }

    inline bool pii_bio_class_inside(
        unsigned long cls
    )
    {
        return cls != 0 && cls < pii_bio_class_count() && (cls - 1) % 2 == 1;
    }

    inline bool pii_bio_transition_allowed(
        unsigned long previous,
        unsigned long cls
    )
    {
        if (!pii_bio_class_inside(cls)) return true;
        return previous != 0 && (previous - 1) / 2 == (cls - 1) / 2;
    }

    inline std::vector<unsigned long> pii_spans_to_bio(
        const std::vector<std::pair<std::size_t, std::size_t>>& units,
        const std::vector<pii_span>& spans,
        std::size_t* misaligned = nullptr
    )
    {
        std::vector<unsigned long> classes(units.size(), 0);
        std::vector<pii_span> sorted(spans);
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
        std::size_t u = 0;
        for (const pii_span& s : sorted)
        {
            const unsigned long b = pii_bio_class(s.label, false);
            if (b == 0) continue;
            while (u < units.size() && units[u].second <= s.begin) ++u;
            bool first = true, aligned = true;
            std::size_t k = u;
            if (k < units.size() && units[k].first < s.begin) aligned = false;
            for (; k < units.size() && units[k].first < s.end; ++k)
            {
                if (units[k].second > s.end) aligned = false;
                classes[k] = first ? b : b + 1;
                first = false;
            }
            if (first || units[k - 1].second != s.end) aligned = false;
            if (!aligned && misaligned) ++*misaligned;
        }
        return classes;
    }

    inline std::vector<pii_span> pii_bio_to_spans(
        const std::vector<std::pair<std::size_t, std::size_t>>& units,
        const std::vector<unsigned long>& classes,
        const std::vector<double>& confidences = std::vector<double>()
    )
    {
        DLIB_CASSERT(units.size() == classes.size());
        DLIB_CASSERT(confidences.empty() || confidences.size() == classes.size());
        std::vector<pii_span> spans;
        std::size_t k = 0;
        while (k < classes.size())
        {
            if (classes[k] == 0 || classes[k] >= pii_bio_class_count()) { ++k; continue; }
            const std::size_t group = (classes[k] - 1) / 2;
            std::size_t j = k + 1;
            double sum = confidences.empty() ? 1 : confidences[k];
            while (j < classes.size() && pii_bio_class_inside(classes[j]) && (classes[j] - 1) / 2 == group)
            {
                sum += confidences.empty() ? 1 : confidences[j];
                ++j;
            }
            pii_span s;
            s.begin = units[k].first;
            s.end = units[j - 1].second;
            s.label = pii_fine_labels()[group];
            s.type = pii_type_from_label(s.label);
            s.confidence = sum / static_cast<double>(j - k);
            spans.push_back(s);
            k = j;
        }
        return spans;
    }

// ----------------------------------------------------------------------------------------

    inline std::vector<pii_span> coarsen_pii_spans(
        const std::string& text,
        const std::vector<pii_span>& spans
    )
    {
        std::vector<pii_span> out;
        for (const pii_span& s : spans)
        {
            DLIB_ASSERT(s.begin <= s.end && s.end <= text.size(), "\t coarsen_pii_spans(): a span lies outside the text");
            pii_span c = s;
            c.type = s.type != pii_type::none ? s.type : pii_type_from_label(s.label);
            c.label = pii_type_keyword(c.type);
            if (!out.empty())
            {
                // At most two spaces or hyphens between the parts, as pii_detector merges them.
                pii_span& m = out.back();
                bool gap = m.type == c.type && (c.type == pii_type::person || c.type == pii_type::location) &&
                           c.begin >= m.end && c.begin - m.end <= 8 && c.begin <= text.size();
                std::size_t k = m.end, parts = 0;
                while (gap && k < c.begin)
                {
                    const char32_t ch = privacy_impl::utf8_next(text, k);
                    gap = ++parts <= 2 && (privacy_impl::is_space(ch) || privacy_impl::is_hyphen(ch)) &&
                          !privacy_impl::is_line_break(ch) && ch != '\t';
                }
                if (gap)
                {
                    m.end = c.end;
                    m.confidence = std::min(m.confidence, c.confidence);
                    continue;
                }
            }
            out.push_back(c);
        }
        return out;
    }

    inline double pii_score::precision() const
    {
        const std::size_t n = true_positives + false_positives;
        return n == 0 ? 1.0 : static_cast<double>(true_positives) / static_cast<double>(n);
    }

    inline double pii_score::recall() const
    {
        const std::size_t n = true_positives + false_negatives;
        return n == 0 ? 1.0 : static_cast<double>(true_positives) / static_cast<double>(n);
    }

    inline double pii_score::f1() const
    {
        const double p = precision(), r = recall();
        return p + r == 0 ? 0.0 : 2 * p * r / (p + r);
    }

    inline std::string pii_evaluation::to_string() const
    {
        std::ostringstream out;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%-16s %7s %7s %7s %9s %7s %7s\n", "label", "tp", "fp", "fn", "precision",
                      "recall", "f1");
        out << buf;
        auto line = [&](const pii_score& s) {
            std::snprintf(buf, sizeof(buf), "%-16s %7lu %7lu %7lu %9.4f %7.4f %7.4f\n", s.label.c_str(),
                          static_cast<unsigned long>(s.true_positives), static_cast<unsigned long>(s.false_positives),
                          static_cast<unsigned long>(s.false_negatives), s.precision(), s.recall(), s.f1());
            out << buf;
        };
        for (const pii_score& s : labels) line(s);
        line(overall);
        return out.str();
    }

    inline pii_evaluation evaluate_pii_spans(
        const std::vector<std::vector<pii_span>>& truth,
        const std::vector<std::vector<pii_span>>& predicted,
        span_match match = span_match::exact,
        const std::vector<std::string>& labels = std::vector<std::string>()
    )
    {
        DLIB_CASSERT(truth.size() == predicted.size());
        auto counted = [&](const pii_span& s) {
            return labels.empty() || std::find(labels.begin(), labels.end(), s.label) != labels.end();
        };
        std::map<std::string, pii_score> scores;
        for (const std::string& l : labels) scores[l].label = l;
        for (std::size_t i = 0; i < truth.size(); ++i)
        {
            std::vector<bool> used(truth[i].size(), false);
            for (const pii_span& p : predicted[i])
            {
                if (!counted(p)) continue;
                pii_score& s = scores[p.label];
                s.label = p.label;
                bool hit = false;
                for (std::size_t k = 0; k < truth[i].size() && !hit; ++k)
                {
                    const pii_span& t = truth[i][k];
                    if (used[k] || t.label != p.label) continue;
                    const bool same = match == span_match::exact ? (t.begin == p.begin && t.end == p.end)
                                                                 : (p.begin < t.end && t.begin < p.end);
                    if (same) { used[k] = true; hit = true; }
                }
                if (hit) ++s.true_positives;
                else ++s.false_positives;
            }
            for (std::size_t k = 0; k < truth[i].size(); ++k)
            {
                if (used[k] || !counted(truth[i][k])) continue;
                pii_score& s = scores[truth[i][k].label];
                s.label = truth[i][k].label;
                ++s.false_negatives;
            }
        }
        pii_evaluation ev;
        ev.overall.label = "all";
        for (const auto& kv : scores)
        {
            ev.labels.push_back(kv.second);
            ev.overall.true_positives += kv.second.true_positives;
            ev.overall.false_positives += kv.second.false_positives;
            ev.overall.false_negatives += kv.second.false_negatives;
        }
        return ev;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_DATASET_H_
