// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_RECOGNIZERS_H_
#define DLIB_PII_RECOGNIZERS_H_

#include "pii_recognizers_abstract.h"

#include <algorithm>
#include <istream>
#include <memory>
#include <string>
#include <utility>
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

        inline bool spans_overlap(const pii_span& a, const pii_span& b) { return a.begin < b.end && b.begin < a.end; }
    }

// ----------------------------------------------------------------------------------------

    enum class pii_leading_recognizer
    {
        span_model,
        tagger
    };

    inline std::vector<pii_span> combine_pii_recognizers(
        const std::string& text,
        const std::vector<pii_span>& model_spans,
        const std::vector<pii_span>& tagger_spans,
        pii_leading_recognizer leading,
        double min_confidence = 0.5,
        double second_min_confidence = 0.9
    )
    {
        using namespace privacy_impl;
        const bool tagger_leads = leading == pii_leading_recognizer::tagger;
        const std::vector<pii_span>& second = tagger_leads ? model_spans : tagger_spans;
        /* What a span counts: its confidence, and a tenth more when its recognizer leads. A secret
           is no word: the span model, which reads bytes, leads on it beside any tagger, and the
           secrets it reads come before every other span, a secret left readable costing more than
           a value given another type. */
        const double lead = 0.1;
        std::vector<std::pair<double, const pii_span*>> candidates;
        for (const pii_span& s : model_spans)
        {
            const bool secret = s.label == "SECRET";
            if (s.confidence >= (secret || !tagger_leads ? min_confidence : second_min_confidence))
                candidates.emplace_back(s.confidence + (secret ? 1 : (tagger_leads ? 0 : lead)), &s);
        }
        const std::size_t from_model = candidates.size();
        /* A town is a word that its context alone tells from any other capitalized word, a heading
           or a product: below second_min_confidence, a leading tagger keeps one where the span
           model reads a town or a postal code too, even weakly. */
        auto town_confirmed = [&model_spans](const pii_span& s) {
            for (const pii_span& m : model_spans)
                if ((m.label == "CITY" || m.label == "POSTAL_CODE") && m.confidence >= 0.2 && spans_overlap(s, m))
                    return true;
            return false;
        };
        /* An account name written in lower-case letters alone has the look of any word of a sentence
           ("jdupont", "reseau"): below second_min_confidence, a leading tagger keeps one in running text
           where the span model reads an account name too, even weakly, and keeps one as any span where a
           layout writes a value. */
        auto plain_word = [&text](const pii_span& s) {
            if (s.begin >= s.end || s.end > text.size()) return false;
            for (std::size_t i = s.begin; i < s.end; ++i)
            {
                const unsigned char c = static_cast<unsigned char>(text[i]);
                if (!((c >= 'a' && c <= 'z') || c >= 0x80)) return false;
            }
            return true;
        };
        auto account_confirmed = [&model_spans](const pii_span& s) {
            for (const pii_span& m : model_spans)
                if (m.label == "USERNAME" && m.confidence >= 0.2 && spans_overlap(s, m)) return true;
            return false;
        };
        /* Where a layout writes a value: at the start of the text or of a line, after a tab, a colon,
           an equals sign, a semicolon or a vertical bar, spaces and an opening quote between, or right
           after a comma or a '>' ("Compte : jdupont", "id;jdupont;actif", "<td>jdupont"). */
        auto value_place = [&text](const pii_span& s) {
            std::size_t q = std::min(s.begin, text.size());
            if (q > 0 && (text[q - 1] == ',' || text[q - 1] == '>')) return true;
            if (q > 0 && (text[q - 1] == '"' || text[q - 1] == '\'')) --q;
            while (q > 0)
            {
                if (text[q - 1] == ' ') --q;
                else if (q >= 2 && text[q - 2] == '\xC2' && text[q - 1] == '\xA0') q -= 2;
                else break;
            }
            if (q == 0) return true;
            const char c = text[q - 1];
            return c == '\n' || c == '\r' || c == '\t' || c == ':' || c == '=' || c == ';' || c == '|';
        };
        for (const pii_span& s : tagger_spans)
        {
            const bool leads = tagger_leads && s.label != "SECRET";
            const bool sure =
                leads && (s.label != "CITY" || s.confidence >= second_min_confidence || town_confirmed(s)) &&
                (s.label != "USERNAME" || !plain_word(s) || s.confidence >= second_min_confidence ||
                 value_place(s) || account_confirmed(s));
            if (s.confidence >= (sure ? min_confidence : second_min_confidence))
                candidates.emplace_back(s.confidence + (leads ? lead : 0), &s);
        }
        // At equal counts, the spans of the leading recognizer come first.
        if (tagger_leads)
            std::rotate(candidates.begin(), candidates.begin() + static_cast<long>(from_model), candidates.end());
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const std::pair<double, const pii_span*>& a, const std::pair<double, const pii_span*>& b) {
                             return a.first > b.first;
                         });
        std::vector<pii_span> spans;
        auto is_free = [&spans](const pii_span& s) {
            for (const pii_span& o : spans)
                if (spans_overlap(s, o)) return false;
            return true;
        };
        for (const std::pair<double, const pii_span*>& c : candidates)
            if (is_free(*c.second)) spans.push_back(*c.second);
        // The less confident name parts of the second recognizer that complete a name already kept.
        std::vector<pii_span> pending;
        for (const pii_span& s : second)
            if (s.confidence >= min_confidence && s.confidence < second_min_confidence && name_part(s) && is_free(s))
                pending.push_back(s);
        for (bool grown = true; grown;)
        {
            grown = false;
            for (std::size_t k = 0; k < pending.size(); ++k)
            {
                if (!is_free(pending[k]) || !next_to_name(text, pending[k], spans)) continue;
                spans.push_back(pending[k]);
                pending.erase(pending.begin() + static_cast<long>(k));
                grown = true;
                break;
            }
        }
        std::sort(spans.begin(), spans.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
        return spans;
    }

    namespace privacy_impl
    {
        /* The parts of a name and the organisations that the span model reads below limit, down
           to 0.2, where no span kept lies: no value by themselves, they are the weak evidence
           that a capitalized word joined by a conjunction to a person found is another person
           ("Martin and Dupont"), which the detector reads under the label "WEAK". */
        inline void append_weak_names(
            const std::vector<pii_span>& model_spans,
            double limit,
            std::vector<pii_span>& out
        )
        {
            const std::size_t count = out.size();
            for (const pii_span& s : model_spans)
            {
                if (s.confidence >= limit || s.confidence < 0.2 || !(name_part(s) || s.label == "ORGANIZATION"))
                    continue;
                bool free = true;
                for (std::size_t k = 0; k < count && free; ++k)
                    if (spans_overlap(s, out[k])) free = false;
                if (!free) continue;
                pii_span w = s;
                w.label = "WEAK";
                w.type = pii_type::none;
                out.push_back(w);
            }
        }
    }

    inline pii_leading_recognizer default_leading_recognizer(const pii_span_recognizer& model)
    {
        return model.size() == "tiny" ? pii_leading_recognizer::tagger : pii_leading_recognizer::span_model;
    }

    inline void attach_pii_recognizers(
        pii_detector& detector,
        std::shared_ptr<const pii_sequence_tagger> tagger,
        std::shared_ptr<const pii_span_recognizer> model,
        pii_leading_recognizer leading,
        double min_confidence = 0.5,
        double second_min_confidence = 0.9
    )
    {
        if (model && model->empty()) model.reset();
        if (model && tagger)
        {
            // A span of the model that the combination keeps out below this confidence is weak evidence at most.
            const double weak = leading == pii_leading_recognizer::tagger ? second_min_confidence : min_confidence;
            detector.add_recognizer([model, tagger, leading, weak, min_confidence, second_min_confidence](
                                        const std::string& t) {
                const std::vector<pii_span> read = model->find(t);
                std::vector<pii_span> out = combine_pii_recognizers(t, read, tagger->find(t), leading, min_confidence,
                                                                    second_min_confidence);
                privacy_impl::append_weak_names(read, weak, out);
                return out;
            }, 0);
        }
        else if (model)
            detector.add_recognizer([model, min_confidence](const std::string& t) {
                const std::vector<pii_span> read = model->find(t);
                std::vector<pii_span> out;
                for (const pii_span& s : read)
                    if (s.confidence >= min_confidence) out.push_back(s);
                privacy_impl::append_weak_names(read, min_confidence, out);
                return out;
            }, 0);
        else if (tagger)
            detector.add_recognizer([tagger](const std::string& t) { return tagger->find(t); }, min_confidence);
    }

    inline void attach_pii_recognizers(
        pii_detector& detector,
        std::shared_ptr<const pii_sequence_tagger> tagger,
        std::shared_ptr<const pii_span_recognizer> model,
        double min_confidence = 0.5,
        double second_min_confidence = 0.9
    )
    {
        const pii_leading_recognizer leading =
            model && !model->empty() ? default_leading_recognizer(*model) : pii_leading_recognizer::tagger;
        attach_pii_recognizers(detector, std::move(tagger), std::move(model), leading, min_confidence,
                               second_min_confidence);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_RECOGNIZERS_H_
