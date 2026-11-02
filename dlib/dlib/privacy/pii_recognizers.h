// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_RECOGNIZERS_H_
#define DLIB_PII_RECOGNIZERS_H_

#include "pii_recognizers_abstract.h"

#include <algorithm>
#include <istream>
#include <memory>
#include <string>
#include <vector>
#include "pii_detector.h"
#include "pii_sequence_tagger.h"
#include "pii_span_model.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pii_span_recognizer
    {
    public:
        pii_span_recognizer() = default;

        explicit pii_span_recognizer(std::istream& in)
        {
            size_ = serialized_size(in);
            if (size_ == "tiny")
            {
                tiny_ = std::make_shared<pii_span_model_tiny>();
                deserialize(*tiny_, in);
            }
            else if (size_ == "small")
            {
                small_ = std::make_shared<pii_span_model_small>();
                deserialize(*small_, in);
            }
            else
            {
                throw serialization_error("pii_span_recognizer: the stream holds no tiny or small span model");
            }
        }

        const std::string& size() const { return size_; }
        bool empty() const { return !tiny_ && !small_; }

        std::vector<pii_span> find(const std::string& text) const
        {
            if (tiny_) return tiny_->find(text);
            if (small_) return small_->find(text);
            return std::vector<pii_span>();
        }

        /* The size of the span model serialized at the start of in, "tiny", "small" or empty
           for another content; the stream is left where it was. */
        static std::string serialized_size(std::istream& in)
        {
            const std::streampos at = in.tellg();
            std::string name;
            long d = 0, l = 0, h = 0;
            try
            {
                deserialize(name, in);
                if (name == "pii_span_model")
                {
                    deserialize(d, in);
                    deserialize(l, in);
                    deserialize(h, in);
                }
            }
            catch (serialization_error&) {}
            in.clear();
            in.seekg(at);
            if (name != "pii_span_model") return std::string();
            if (d == 64 && l == 2 && h == 2) return "tiny";
            if (d == 128 && l == 4 && h == 4) return "small";
            return std::string();
        }

    private:
        std::string size_;
        std::shared_ptr<pii_span_model_tiny> tiny_;
        std::shared_ptr<pii_span_model_small> small_;
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        inline bool name_part(const pii_span& s)
        {
            return s.label == "GIVEN_NAME" || s.label == "FAMILY_NAME";
        }

        // True when s is a name part separated from a name part of spans by spaces or a hyphen only.
        inline bool next_to_name(const std::string& text, const pii_span& s, const std::vector<pii_span>& spans)
        {
            for (const pii_span& o : spans)
            {
                if (!name_part(o) || (o.begin < s.end && s.begin < o.end)) continue;
                const std::size_t from = o.end <= s.begin ? o.end : s.end;
                const std::size_t to = o.end <= s.begin ? s.begin : o.begin;
                if (to - from > 2) continue;
                bool gap = true;
                for (std::size_t k = from; k < to; ++k)
                    if (text[k] != ' ' && text[k] != '-') gap = false;
                if (gap) return true;
            }
            return false;
        }

        /* The spans of the model, where the tagger reads a name with more confidence. A span
           read as an organisation with a confidence below doubt is replaced by the name parts
           the tagger reads over its words, when they cover all of it: "Lucie Rey" alone on its
           line. A single word read as a town or an organisation is replaced by the given name
           the tagger reads there with a confidence of doubt at least, and close to that of the
           model or above, when the tagger also reads the family name that follows it: "Malo
           Cozic", "Nolwenn Le Floc'h". */
        inline std::vector<pii_span> prefer_names(const std::string& text, const std::vector<pii_span>& model_spans,
                                                  const std::vector<pii_span>& tagger_spans, double doubt)
        {
            std::vector<pii_span> out;
            for (const pii_span& m : model_spans)
            {
                std::vector<pii_span> parts;
                if (m.label == "ORGANIZATION" && m.confidence < doubt)
                    for (const pii_span& s : tagger_spans)
                        if (name_part(s) && s.begin < m.end && m.begin < s.end && s.confidence > m.confidence)
                            parts.push_back(s);
                bool covered = !parts.empty();
                for (const pii_span& s : parts)
                    if (s.begin < m.begin || s.end > m.end) covered = false;
                for (std::size_t k = m.begin; k < m.end && covered; ++k)
                {
                    if (text[k] == ' ' || text[k] == '-') continue;
                    bool inside = false;
                    for (const pii_span& s : parts)
                        if (s.begin <= k && k < s.end) inside = true;
                    covered = inside;
                }
                if (!covered && (m.label == "ORGANIZATION" || m.label == "CITY") && text.find(' ', m.begin) >= m.end)
                {
                    const pii_span* given = nullptr;
                    const pii_span* family = nullptr;
                    for (const pii_span& s : tagger_spans)
                    {
                        if (s.label == "GIVEN_NAME" && s.begin == m.begin && s.end == m.end && s.confidence >= doubt &&
                            s.confidence + 0.02 >= m.confidence)
                            given = &s;
                        if (s.label == "FAMILY_NAME" && s.begin > m.end && s.begin - m.end <= 2 &&
                            text.find_first_not_of(" -", m.end) == s.begin)
                            family = &s;
                    }
                    if (given && family)
                    {
                        out.push_back(*given);
                        bool free = true;
                        for (const pii_span& o : model_spans)
                            if (o.begin < family->end && family->begin < o.end) free = false;
                        if (free) out.push_back(*family);
                        continue;
                    }
                }
                if (covered) out.insert(out.end(), parts.begin(), parts.end());
                else out.push_back(m);
            }
            std::sort(out.begin(), out.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
            return out;
        }
    }

// ----------------------------------------------------------------------------------------

    inline std::vector<pii_span> combine_pii_recognizers(
        const std::string& text,
        const std::vector<pii_span>& model_spans,
        const std::vector<pii_span>& tagger_spans,
        double min_confidence = 0.5,
        double tagger_min_confidence = 0.9
    )
    {
        using namespace privacy_impl;
        std::vector<pii_span> kept, spans, pending;
        for (const pii_span& s : model_spans)
            if (s.confidence >= min_confidence) kept.push_back(s);
        kept = prefer_names(text, kept, tagger_spans, 0.9);
        spans = kept;
        for (const pii_span& s : tagger_spans)
        {
            bool free = true;
            for (const pii_span& m : kept)
                if (s.begin < m.end && m.begin < s.end) free = false;
            if (!free) continue;
            if (s.confidence >= tagger_min_confidence) spans.push_back(s);
            else if (s.confidence >= min_confidence && name_part(s)) pending.push_back(s);
        }
        for (bool grown = true; grown;)
        {
            grown = false;
            for (std::size_t k = 0; k < pending.size(); ++k)
            {
                if (!next_to_name(text, pending[k], spans)) continue;
                spans.push_back(pending[k]);
                pending.erase(pending.begin() + static_cast<long>(k));
                grown = true;
                break;
            }
        }
        std::sort(spans.begin(), spans.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
        return spans;
    }

    inline void attach_pii_recognizers(
        pii_detector& detector,
        std::shared_ptr<const pii_sequence_tagger> tagger,
        std::shared_ptr<const pii_span_recognizer> model,
        double min_confidence = 0.5,
        double tagger_min_confidence = 0.9
    )
    {
        if (model && model->empty()) model.reset();
        if (model && tagger)
        {
            detector.add_recognizer([model, tagger, min_confidence, tagger_min_confidence](const std::string& t) {
                return combine_pii_recognizers(t, model->find(t), tagger->find(t), min_confidence,
                                               tagger_min_confidence);
            }, 0);
        }
        else if (model)
            detector.add_recognizer([model](const std::string& t) { return model->find(t); }, min_confidence);
        else if (tagger)
            detector.add_recognizer([tagger](const std::string& t) { return tagger->find(t); }, min_confidence);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_RECOGNIZERS_H_
