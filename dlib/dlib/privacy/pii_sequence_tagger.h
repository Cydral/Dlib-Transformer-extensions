// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_SEQUENCE_TAGGER_H_
#define DLIB_PII_SEQUENCE_TAGGER_H_

#include "pii_sequence_tagger_abstract.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "../serialize.h"
#include "../svm/sequence_labeler.h"
#include "../svm/structural_sequence_labeling_trainer.h"
#include "pii_types.h"
#include "pii_dataset.h"
#include "pii_detector.h"
#include "unicode_text.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pii_token
    {
        std::size_t begin = 0;
        std::size_t end = 0;
    };

    inline std::vector<pii_token> tokenize_for_tagging(
        const std::string& text
    )
    {
        using namespace privacy_impl;
        const utf8_text t(text, true);
        std::vector<pii_token> tokens;
        std::size_t i = 0;
        const std::size_t n = t.size();
        /* Identifiers keep their underscores and inner periods (jean_dupont, olga.horvath,
           12.03.1985), and a handle keeps its at sign (@jdupont35); an at sign inside a
           word, as in an e-mail address, stays a token of its own. Words glued together
           are split where the case or the kind of character tells a new word starts:
           "DupontMarie", "DUPONTMarie", "75008Paris", and "J.Dupont" after an initial. */
        auto word = [&](char32_t c) { return is_word_char(c) || c == '_'; };
        auto starts_word = [&](std::size_t j) {
            const char32_t prev = t.cps[j - 1], c = t.cps[j], next = t.at(j + 1);
            if (is_lower(prev) && is_upper(c)) return true;
            if (is_upper(prev) && is_upper(c) && is_lower(next)) return true;
            return is_ascii_digit(prev) && is_upper(c) && is_lower(next);
        };
        while (i < n)
        {
            const char32_t c = t.cps[i];
            if (is_space(c)) { ++i; continue; }
            pii_token tok;
            tok.begin = t.offsets[i];
            const bool handle = c == '@' && i + 1 < n && word(t.cps[i + 1]) && (i == 0 || !word(t.cps[i - 1]));
            if (word(c) || handle)
            {
                std::size_t j = i + 1;
                while (j < n)
                {
                    if (word(t.cps[j]))
                    {
                        if (j > i + (handle ? 1 : 0) && starts_word(j)) break;
                        ++j;
                        continue;
                    }
                    // An inner period, except the one of an initial before a capital: "J.Dupont".
                    const bool initial = j == i + 1 && is_upper(t.cps[i]) && is_upper(t.at(j + 1));
                    if (t.cps[j] == '.' && j + 1 < n && word(t.cps[j + 1]) && word(t.cps[j - 1]) && !initial)
                    {
                        ++j;
                        continue;
                    }
                    break;
                }
                tok.end = t.offsets[j];
                i = j;
            }
            else
            {
                tok.end = t.offsets[i + 1];
                ++i;
            }
            tokens.push_back(tok);
        }
        return tokens;
    }

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        inline unsigned long long fnv1a(const std::string& s, unsigned long long h = 1469598103934665603ULL)
        {
            for (unsigned char c : s)
            {
                h ^= c;
                h *= 1099511628211ULL;
            }
            return h;
        }

        // The hash of a feature: its name, a separator byte that no name holds, its value.
        inline unsigned long long feature_hash(const char* name, const std::string& value)
        {
            unsigned long long h = fnv1a(name);
            h ^= 0x1F;
            h *= 1099511628211ULL;
            return fnv1a(value, h);
        }

        inline unsigned long long mix_label(unsigned long long f, unsigned long label)
        {
            unsigned long long h = f ^ (0x9E3779B97F4A7C15ULL * (label + 1));
            h ^= h >> 29;
            h *= 0xBF58476D1CE4E5B9ULL;
            h ^= h >> 32;
            return h;
        }

        inline std::string token_shape(const std::string& raw, bool collapse)
        {
            std::string shape;
            std::size_t i = 0;
            while (i < raw.size())
            {
                const char32_t c = utf8_next(raw, i);
                if (is_invisible_mark(c)) continue;
                char s;
                if (is_upper(c)) s = 'X';
                else if (is_lower(c)) s = 'x';
                else if (is_ascii_digit(c)) s = 'd';
                else if (c < 0x80) s = static_cast<char>(c);
                else s = '?';
                if (collapse && !shape.empty() && shape.back() == s) continue;
                if (!collapse && shape.size() >= 2 && shape[shape.size() - 1] == s && shape[shape.size() - 2] == s)
                    continue;
                shape.push_back(s);
            }
            return shape;
        }

        inline std::string length_bucket(std::size_t n)
        {
            if (n <= 1) return "1";
            if (n == 2) return "2";
            if (n <= 4) return "3-4";
            if (n <= 7) return "5-7";
            return "8+";
        }

        inline std::vector<const char*> lexicon_flags(const std::string& key)
        {
            std::vector<const char*> flags;
            if (first_names().count(key)) flags.push_back("first");
            if (ambiguous_first_names().count(key)) flags.push_back("ambiguous_first");
            if (person_titles().count(key)) flags.push_back("title");
            if (person_particles().count(key)) flags.push_back("particle");
            if (person_cues_before().count(key)) flags.push_back("cue_before");
            if (person_cues_after().count(key)) flags.push_back("cue_after");
            if (street_types().count(key)) flags.push_back("street");
            if (organization_forms().count(key)) flags.push_back("legal_form");
            if (organization_heads().count(key)) flags.push_back("org_head");
            if (location_triggers().count(key)) flags.push_back("place_trigger");
            if (common_words().count(key)) flags.push_back("common");
            if (acronym_words().count(key)) flags.push_back("acronym");
            return flags;
        }
    }

// ----------------------------------------------------------------------------------------

    class pii_tagger_features
    {
    public:
        struct unit
        {
            std::vector<std::uint32_t> features;    // hashed, before the conjunction with a label
            bool opens_segment = false;             // first token of a line or of a cell
        };
        typedef std::vector<unit> sequence_type;

        explicit pii_tagger_features(
            unsigned long hash_bits = 20
        ) : bits_(hash_bits)
        {
            DLIB_CASSERT(10 <= hash_bits && hash_bits <= 26);
        }

        unsigned long hash_bits() const { return bits_; }
        unsigned long num_labels() const { return pii_bio_class_count(); }
        unsigned long order() const { return 1; }

        unsigned long num_features() const
        {
            const unsigned long l = num_labels();
            return (1UL << bits_) + l * l + l;
        }

        unsigned long emission_index(std::uint32_t feature, unsigned long label) const
        {
            return static_cast<unsigned long>(privacy_impl::mix_label(feature, label) & ((1ULL << bits_) - 1));
        }

        template <typename EXP>
        bool reject_labeling(
            const sequence_type& x,
            const matrix_exp<EXP>& y,
            unsigned long position
        ) const
        {
            // A span never crosses a line break or a tab: values end with their cell.
            if (x[position].opens_segment && pii_bio_class_inside(y(0))) return true;
            const unsigned long previous = y.size() > 1 ? y(1) : 0;
            return !pii_bio_transition_allowed(previous, y(0));
        }

        template <typename feature_setter, typename EXP>
        void get_features(
            feature_setter& set_feature,
            const sequence_type& x,
            const matrix_exp<EXP>& y,
            unsigned long position
        ) const
        {
            const unsigned long label = y(0);
            for (std::uint32_t f : x[position].features)
                set_feature(emission_index(f, label));
            const unsigned long l = num_labels();
            const unsigned long base = 1UL << bits_;
            if (y.size() > 1)
                set_feature(base + y(1) * l + label);
            else
                set_feature(base + l * l + label);
        }

        sequence_type make_sequence(
            const std::string& text,
            const std::vector<pii_token>& tokens
        ) const
        {
            using namespace privacy_impl;
            const std::size_t n = tokens.size();
            std::vector<std::string> raw(n), key(n), shape(n), short_shape(n);
            std::vector<bool> line_start(n, false), line_end(n, false), after_tab(n, false);
            for (std::size_t i = 0; i < n; ++i)
            {
                raw[i] = text.substr(tokens[i].begin, tokens[i].end - tokens[i].begin);
                key[i] = fold_key(raw[i]);
                shape[i] = token_shape(raw[i], false);
                short_shape[i] = token_shape(raw[i], true);
                const std::size_t from = i == 0 ? 0 : tokens[i - 1].end;
                const std::string gap = text.substr(from, tokens[i].begin - from);
                if (i == 0 || gap.find('\n') != std::string::npos || gap.find('\r') != std::string::npos)
                    line_start[i] = true;
                if (gap.find('\t') != std::string::npos) after_tab[i] = true;
            }
            for (std::size_t i = 0; i < n; ++i)
                line_end[i] = i + 1 == n || line_start[i + 1];

            /* The field label a value follows: "Nom : DUPONT", "Ville\tRennes". The text
               of a segment up to a colon, or a whole segment before a tab, is looked up
               in the labels of the detector. */
            std::vector<pii_type> field(n, pii_type::none);
            {
                pii_type current = pii_type::none;
                bool open = false;      // a label just closed by its colon, value still to come
                std::size_t segment_start = 0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    if (i > 0 && (line_start[i] || after_tab[i]))
                    {
                        if (open)
                        {
                            // "Nom :" at the end of a line or a cell: the value follows.
                        }
                        else
                        {
                            // A segment made of a label alone ("Ville", "E-mail") types the next one.
                            const std::string seg = text.substr(tokens[segment_start].begin,
                                                                tokens[i - 1].end - tokens[segment_start].begin);
                            current = lookup_label(normalize_label(seg));
                            open = current != pii_type::none;
                        }
                        segment_start = i;
                    }
                    if (raw[i] == ":" && i > segment_start)
                    {
                        const std::string seg = text.substr(tokens[segment_start].begin,
                                                            tokens[i - 1].end - tokens[segment_start].begin);
                        const pii_type t = lookup_label(normalize_label(seg));
                        if (t != pii_type::none)
                        {
                            current = t;
                            open = true;
                            continue;
                        }
                    }
                    field[i] = current;
                    if (open && current != pii_type::none) open = false;
                    if (line_end[i] && !open) current = pii_type::none;
                }
            }

            sequence_type seq(n);
            for (std::size_t i = 0; i < n; ++i) seq[i].opens_segment = line_start[i] || after_tab[i];
            auto word = [&](long k) -> const std::string& {
                static const std::string bos = "<s>", eos = "</s>";
                if (k < 0) return bos;
                if (k >= static_cast<long>(n)) return eos;
                return key[static_cast<std::size_t>(k)];
            };
            auto sh = [&](long k) -> const std::string& {
                static const std::string none = "_";
                if (k < 0 || k >= static_cast<long>(n)) return none;
                return short_shape[static_cast<std::size_t>(k)];
            };
            for (std::size_t i = 0; i < n; ++i)
            {
                std::vector<std::uint32_t>& f = seq[i].features;
                // Both halves of the hash folded into the 32 bits kept for each feature.
                auto add = [&](const char* name, const std::string& value) {
                    const unsigned long long h = feature_hash(name, value);
                    f.push_back(static_cast<std::uint32_t>(h ^ (h >> 32)));
                };
                const long k = static_cast<long>(i);
                add("b", "");
                add("w", key[i]);
                add("s", shape[i]);
                add("ss", short_shape[i]);
                add("len", length_bucket(raw[i].size()));
                const std::string& w = key[i];
                for (std::size_t p = 1; p <= 4 && p < w.size(); ++p)
                {
                    add("p", w.substr(0, p));
                    add("x", w.substr(w.size() - p));
                }
                for (const char* flag : lexicon_flags(w)) add("lex", flag);
                for (long d : { -2L, -1L, 1L, 2L })
                {
                    const std::string tag = "w" + std::to_string(d);
                    add(tag.c_str(), word(k + d));
                    const std::string stag = "s" + std::to_string(d);
                    add(stag.c_str(), sh(k + d));
                }
                if (i > 0)
                    for (const char* flag : lexicon_flags(key[i - 1])) add("lex-1", flag);
                if (i + 1 < n)
                    for (const char* flag : lexicon_flags(key[i + 1])) add("lex+1", flag);
                add("bg-1", word(k - 1) + "|" + w);
                add("bg+1", w + "|" + word(k + 1));
                add("sh3", sh(k - 1) + "|" + short_shape[i] + "|" + sh(k + 1));
                if (line_start[i]) add("pos", "bol");
                if (line_end[i]) add("pos", "eol");
                if (after_tab[i]) add("pos", "tab");
                if (i > 0 && raw[i - 1] == ":") add("pos", "colon");
                if (raw[i].find('_') != std::string::npos) add("has", "underscore");
                if (raw[i].find('.') != std::string::npos && raw[i].size() > 2) add("has", "inner_period");
                if (raw[i][0] == '@') add("has", "at_sign");
                if (field[i] != pii_type::none) add("field", pii_type_keyword(field[i]));
                if (field[i] != pii_type::none) add("field_s", std::string(pii_type_keyword(field[i])) + short_shape[i]);
            }
            return seq;
        }

        friend void serialize(const pii_tagger_features& item, std::ostream& out)
        {
            serialize("pii_tagger_features", out);
            serialize(item.bits_, out);
        }

        friend void deserialize(pii_tagger_features& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "pii_tagger_features")
                throw serialization_error("Unexpected version found while deserializing dlib::pii_tagger_features.");
            unsigned long bits = 0;
            deserialize(bits, in);
            if (bits < 10 || bits > 26)
                throw serialization_error("Invalid hash size found while deserializing dlib::pii_tagger_features.");
            item.bits_ = bits;
        }

    private:
        unsigned long bits_;
    };

// ----------------------------------------------------------------------------------------

    struct pii_tagger_training_options
    {
        double c = 10;
        double epsilon = 0.05;
        unsigned long max_iterations = 200;
        unsigned long num_threads = 2;
        unsigned long hash_bits = 20;
        double entity_loss = 2;
        unsigned long max_cache_size = 0;
        bool verbose = false;
    };

    class pii_sequence_tagger
    {
    public:
        pii_sequence_tagger() = default;

        void train(
            const std::vector<pii_example>& examples,
            const pii_tagger_training_options& options = pii_tagger_training_options()
        )
        {
            DLIB_CASSERT(!examples.empty());
            DLIB_CASSERT(options.c > 0 && options.epsilon > 0 && options.max_iterations > 0 &&
                         options.num_threads > 0 && options.entity_loss >= 0);
            DLIB_CASSERT(10 <= options.hash_bits && options.hash_bits <= 26);
            // The object changes only once training has succeeded.
            const pii_tagger_features fe(options.hash_bits);
            std::vector<pii_tagger_features::sequence_type> x;
            std::vector<std::vector<unsigned long>> y;
            std::size_t misaligned = 0;
            for (const pii_example& ex : examples)
            {
                const std::vector<pii_token> tokens = tokenize_for_tagging(ex.text);
                if (tokens.empty()) continue;
                std::vector<std::pair<std::size_t, std::size_t>> units;
                for (const pii_token& t : tokens) units.push_back(std::make_pair(t.begin, t.end));
                y.push_back(pii_spans_to_bio(units, ex.spans, &misaligned));
                x.push_back(fe.make_sequence(ex.text, tokens));
            }
            DLIB_CASSERT(!x.empty(), "pii_sequence_tagger::train(): no token in the training examples");
            structural_sequence_labeling_trainer<pii_tagger_features> trainer(fe);
            trainer.set_c(options.c);
            trainer.set_epsilon(options.epsilon);
            trainer.set_max_iterations(options.max_iterations);
            trainer.set_num_threads(options.num_threads);
            trainer.set_max_cache_size(options.max_cache_size);
            for (unsigned long label = 1; label < fe.num_labels(); ++label)
                trainer.set_loss(label, options.entity_loss);
            if (options.verbose) trainer.be_verbose();
            labeler_ = trainer.train(x, y);
            misaligned_ = misaligned;
            trained_ = true;
        }

        bool is_trained() const { return trained_; }

        std::vector<pii_span> find(
            const std::string& text
        ) const
        {
            DLIB_CASSERT(is_trained(), "pii_sequence_tagger::find(): the tagger is not trained");
            const std::vector<pii_token> tokens = tokenize_for_tagging(text);
            if (tokens.empty()) return std::vector<pii_span>();
            // The extractor the labeler was trained with, so that both read the same features.
            const pii_tagger_features& fe = labeler_.get_feature_extractor();
            const pii_tagger_features::sequence_type x = fe.make_sequence(text, tokens);
            const std::vector<unsigned long> classes = labeler_(x);
            const matrix<double,0,1>& w = labeler_.get_weights();
            std::vector<double> confidence(classes.size(), 1.0);
            std::vector<std::pair<std::size_t, std::size_t>> units;
            for (std::size_t i = 0; i < tokens.size(); ++i)
            {
                units.push_back(std::make_pair(tokens[i].begin, tokens[i].end));
                if (classes[i] == 0) continue;
                double margin = 0;
                for (std::uint32_t f : x[i].features)
                    margin += w(fe.emission_index(f, classes[i])) - w(fe.emission_index(f, 0));
                confidence[i] = 1.0 / (1.0 + std::exp(-margin));
            }
            return pii_bio_to_spans(units, classes, confidence);
        }

        const pii_tagger_features& get_feature_extractor() const { return labeler_.get_feature_extractor(); }
        std::size_t misaligned_spans() const { return misaligned_; }

        /* The weights are written in single precision: the file is half as large, and a
           relative error of 1e-7 on each weight changes no decision of the labeler. */
        friend void serialize(const pii_sequence_tagger& item, std::ostream& out)
        {
            serialize("pii_sequence_tagger", out);
            serialize(item.trained_, out);
            if (!item.trained_) return;
            serialize(item.labeler_.get_feature_extractor(), out);
            const matrix<float,0,1> weights = matrix_cast<float>(item.labeler_.get_weights());
            serialize(weights, out);
        }

        friend void deserialize(pii_sequence_tagger& item, std::istream& in)
        {
            std::string name;
            deserialize(name, in);
            if (name != "pii_sequence_tagger")
                throw serialization_error("Unexpected version found while deserializing dlib::pii_sequence_tagger.");
            bool trained = false;
            deserialize(trained, in);
            sequence_labeler<pii_tagger_features> labeler;
            if (trained)
            {
                pii_tagger_features fe;
                deserialize(fe, in);
                matrix<float,0,1> weights;
                deserialize(weights, in);
                if (static_cast<unsigned long>(weights.size()) != fe.num_features())
                    throw serialization_error("The weights of a pii_sequence_tagger do not match its features.");
                labeler = sequence_labeler<pii_tagger_features>(matrix_cast<double>(weights), fe);
            }
            item.labeler_ = labeler;
            item.trained_ = trained;
            item.misaligned_ = 0;
        }

    private:
        bool trained_ = false;
        sequence_labeler<pii_tagger_features> labeler_;
        std::size_t misaligned_ = 0;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_SEQUENCE_TAGGER_H_
