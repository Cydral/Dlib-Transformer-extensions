// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_DETECTOR_H_
#define DLIB_PII_DETECTOR_H_

#include "pii_detector_abstract.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../error.h"
#include "pii_types.h"
#include "pii_lexicons.h"
#include "date_formats.h"
#include "unicode_text.h"
#include "id_checks.h"
#include "pseudonymizer.h"
#include "../crypto/sha256.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pii_detector_options
    {
        std::vector<pii_type> types = all_pii_types();
        numeric_date_order date_order = numeric_date_order::day_month_year;
        bool profile_urls_only = true;
        bool public_bodies = true;
    };

    struct detection_context
    {
        pii_type forced = pii_type::none;
        pii_type label = pii_type::none;
    };

    struct pii_pattern
    {
        pii_type type = pii_type::identifier;
        std::string regex;
        double confidence = 0.9;
        bool ignore_case = false;
        unsigned long group = 0;
        std::string check;
    };

    typedef std::function<std::vector<pii_span>(const std::string& text)> pii_recognizer;

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        struct pii_candidate
        {
            std::size_t b, e;       // code point indices
            pii_type type;
            int priority;
            double confidence;
            bool strong = false;    // usable to teach the document lexicon
            bool exact = false;     // boundaries of a known name or of a postal layout, matched whole
        };

        struct word_token
        {
            std::size_t b, e;       // code point indices
            std::string text;
            std::string key;
            int shape;              // 0 lower, 1 title, 2 upper, 3 camel, 4 uncased
        };

        inline int pii_priority(pii_type t)
        {
            switch (t)
            {
                case pii_type::email: return 100;
                case pii_type::url: return 95;
                case pii_type::iban: return 92;
                case pii_type::payment_card: return 92;
                case pii_type::nir: return 92;
                case pii_type::siret: return 88;
                case pii_type::ipv6: return 86;
                case pii_type::ipv4: return 85;
                case pii_type::mac_address: return 85;
                case pii_type::phone: return 80;
                case pii_type::geolocation: return 84;
                case pii_type::license_plate: return 78;
                case pii_type::date: return 75;
                case pii_type::identifier: return 70;
                case pii_type::address: return 62;
                case pii_type::location: return 60;
                case pii_type::organization: return 55;
                case pii_type::person: return 50;
                case pii_type::title: return 58;
                case pii_type::gender: return 72;
                case pii_type::nationality: return 72;
                case pii_type::none: break;
            }
            return 0;
        }

        inline bool is_hex_cp(char32_t c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        }

        inline bool is_ascii_alnum_cp(char32_t c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }

        inline bool has_vowel(const std::string& key)
        {
            return key.find_first_of("aeiouy") != std::string::npos;
        }

        inline std::string cp_substr(const utf8_text& t, const std::string& text, std::size_t b, std::size_t e)
        {
            return text.substr(t.offsets[b], t.offsets[e] - t.offsets[b]);
        }

        inline std::vector<word_token> tokenize_words(const utf8_text& t, const std::string& text)
        {
            (void)text;
            std::vector<word_token> words;
            const std::size_t n = t.size();
            std::size_t i = 0;
            while (i < n)
            {
                if (!is_letter(t.cps[i])) { ++i; continue; }
                std::size_t j = i;
                int upper = 0, lower = 0;
                bool first_upper = is_upper(t.cps[i]);
                bool lower_then_upper = false;
                char32_t prevc = 0;
                while (j < n && is_letter(t.cps[j]))
                {
                    if (is_upper(t.cps[j])) { ++upper; if (prevc && is_lower(prevc)) lower_then_upper = true; }
                    else if (is_lower(t.cps[j])) ++lower;
                    prevc = t.cps[j];
                    ++j;
                }
                word_token w;
                w.b = i;
                w.e = j;
                // From the code points, which leave out the invisible marks of the bytes.
                for (std::size_t k = i; k < j; ++k) utf8_append(w.text, t.cps[k]);
                w.key = fold_key(w.text);
                if (upper == 0 && lower == 0) w.shape = 4;
                else if (upper == 0) w.shape = 0;
                else if (lower == 0) w.shape = 2;
                else if (first_upper && upper == 1) w.shape = 1;
                else if (first_upper && lower_then_upper) w.shape = 3;
                else w.shape = 0;
                words.push_back(w);
                i = j;
            }
            return words;
        }

        inline bool same_line(const utf8_text& t, std::size_t a, std::size_t b)
        {
            for (std::size_t k = a; k < b; ++k)
                if (is_line_break(t.cps[k]) || t.cps[k] == '\t') return false;
            return true;
        }

        // A line in lower case around position pos: of its letters, at least eight, nearly all are small.
        inline bool lower_case_line(const utf8_text& t, std::size_t pos)
        {
            const std::size_t n = t.size();
            std::size_t b = pos, e = pos, letters = 0, upper = 0;
            while (b > 0 && !is_line_break(t.cps[b - 1]) && pos - b < 200) --b;
            while (e < n && !is_line_break(t.cps[e]) && e - pos < 200) ++e;
            for (std::size_t q = b; q < e; ++q)
                if (is_letter(t.cps[q]))
                {
                    ++letters;
                    if (is_upper(t.cps[q])) ++upper;
                }
            return letters >= 8 && upper * 20 <= letters;
        }

        /* The separator between two adjacent words of a name: a single or a doubled space, a
           hyphen, an apostrophe, or a period and a space after an initial. */
        inline bool name_gap(const utf8_text& t, std::size_t a, std::size_t b, bool after_initial)
        {
            if (b <= a) return false;
            const std::size_t len = b - a;
            auto space = [&](std::size_t i) { return t.cps[i] == ' ' || t.cps[i] == 0xA0; };
            if (len == 1)
            {
                const char32_t c = t.cps[a];
                return space(a) || is_hyphen(c) || is_apostrophe(c) || (after_initial && c == '.');
            }
            if (len == 2 && space(a) && space(a + 1))
                return true;
            if (len == 2 && after_initial)
                return t.cps[a] == '.' && (space(a + 1) || is_hyphen(t.cps[a + 1]));
            return false;
        }

        inline bool at_sentence_start(const utf8_text& t, std::size_t b)
        {
            std::size_t k = b;
            while (k > 0 && (t.cps[k - 1] == ' ' || t.cps[k - 1] == 0xA0 || t.cps[k - 1] == '\t'))
                --k;
            if (k == 0) return true;
            const char32_t c = t.cps[k - 1];
            return c == '.' || c == '!' || c == '?' || is_line_break(c) || c == 0x2022 || c == '-' ||
                   c == 0xAB || c == '"' || c == 0x201C || c == '(' || c == ':';
        }

        inline std::string normalize_label(const std::string& raw)
        {
            // "first_name" reads "first name".
            std::string key = fold_key(raw);
            for (char& c : key)
                if (c == '_') c = ' ';
            std::string cleaned;
            for (std::size_t i = 0; i < key.size(); ++i)
            {
                const unsigned char c = static_cast<unsigned char>(key[i]);
                if (c == 0xC2 && i + 1 < key.size() &&
                    (static_cast<unsigned char>(key[i + 1]) == 0xB0 || static_cast<unsigned char>(key[i + 1]) == 0xBA))
                {
                    ++i;
                    continue;
                }
                cleaned.push_back(key[i]);
            }
            while (!cleaned.empty() && (cleaned.back() == ':' || cleaned.back() == ' ' || cleaned.back() == '*' ||
                                        cleaned.back() == '.' || cleaned.back() == '-'))
                cleaned.pop_back();
            // "N° de dossier" reads "numero de dossier", "Phone No." reads "phone number".
            std::istringstream in(cleaned);
            std::string w, out;
            while (in >> w)
            {
                if (w == "n" || w == "no" || w == "num" || w == "num." || w == "nr" || w == "number")
                    w = out.empty() ? "numero" : "number";
                if (!out.empty()) out += ' ';
                out += w;
            }
            return out;
        }

        inline pii_type lookup_label(const std::string& normalized)
        {
            const auto& labels = field_labels();
            auto find = [&](const std::string& k) {
                const auto it = labels.find(k);
                return it == labels.end() ? pii_type::none : it->second;
            };
            if (normalized.empty() || normalized.size() > 48) return pii_type::none;
            pii_type t = find(normalized);
            if (t != pii_type::none) return t;
            if (normalized.compare(0, 7, "numero ") == 0)
            {
                std::string rest = normalized.substr(7);
                t = find("numero " + rest);
                if (t != pii_type::none) return t;
                if (rest.compare(0, 3, "de ") == 0) rest = rest.substr(3);
                if (rest.compare(0, 2, "d'") == 0) rest = rest.substr(2);
                t = find("numero de " + rest);
                if (t != pii_type::none) return t;
                t = find(rest);
                if (t != pii_type::none && t != pii_type::person) return t;
            }
            if (normalized.compare(0, 8, "adresse ") == 0 || normalized.compare(0, 6, "votre ") == 0)
                return find(normalized.substr(normalized.find(' ') + 1));
            return pii_type::none;
        }
    }

// ----------------------------------------------------------------------------------------

    class pii_detector
    {
    public:
        explicit pii_detector(const pii_detector_options& options = pii_detector_options())
            : options_(options), enabled_(32, false)
        {
            for (pii_type t : options_.types)
                if (static_cast<std::size_t>(t) < enabled_.size()) enabled_[static_cast<std::size_t>(t)] = true;
        }

        void add_person_name(const std::string& name)
        {
            const std::string key = privacy_impl::fold_key(name);
            if (key.empty()) return;
            user_names_full_.insert(key);
            add_name_words(name, user_name_words_);
        }

        void add_term(pii_type type, const std::string& term)
        {
            const std::vector<std::string> keys = term_keys(term);
            if (keys.empty() || type == pii_type::none) return;
            term_index_[keys.front()].push_back(terms_.size());
            terms_.push_back(std::make_pair(keys, type));
        }

        void add_allowed_term(const std::string& term)
        {
            const std::string key = privacy_impl::fold_key(term);
            if (!key.empty()) allowed_.insert(key);
        }

        void add_pattern(const pii_pattern& pattern)
        {
            if (pattern.type == pii_type::none)
                throw error("add_pattern(): the pattern \"" + pattern.regex + "\" has no type");
            if (!pattern.check.empty() && pattern.check != "luhn" && pattern.check != "iban")
                throw error("add_pattern(): unknown check \"" + pattern.check + "\", expected luhn or iban");
            compiled_pattern c;
            c.spec = pattern;
            try
            {
                auto flags = std::regex::ECMAScript | std::regex::optimize;
                if (pattern.ignore_case) flags |= std::regex::icase;
                c.re = std::make_shared<std::regex>(pattern.regex, flags);
            }
            catch (std::regex_error& e)
            {
                throw error("add_pattern(): invalid regular expression \"" + pattern.regex + "\": " + e.what());
            }
            if (pattern.group > c.re->mark_count())
                throw error("add_pattern(): \"" + pattern.regex + "\" has no group " + std::to_string(pattern.group));
            patterns_.push_back(c);
        }

        std::size_t pattern_count() const { return patterns_.size(); }

        void add_recognizer(const pii_recognizer& recognizer, double min_confidence = 0.5)
        {
            static std::atomic<std::size_t> next_id(1);
            recognizer_entry r;
            r.fn = recognizer;
            r.min_confidence = min_confidence;
            r.id = next_id++;
            recognizers_.push_back(r);
        }

        std::size_t recognizer_count() const { return recognizers_.size(); }

        void begin_document()
        {
            doc_name_words_.clear();
            doc_names_full_.clear();
            email_fragments_.clear();
            doc_terms_.clear();
            doc_term_set_.clear();
            doc_term_index_.clear();
            std::lock_guard<std::mutex> lock(cache_->m);
            cache_->spans.clear();
        }

        void learn(const std::string& text, const detection_context& context = detection_context())
        {
            std::vector<privacy_impl::pii_candidate> found;
            const privacy_impl::utf8_text t(text, true);
            analyze(text, t, context, found);
            for (const privacy_impl::pii_candidate& c : found)
            {
                const std::string value = privacy_impl::cp_substr(t, text, c.b, c.e);
                if (c.type == pii_type::person && c.strong)
                {
                    doc_names_full_.insert(privacy_impl::fold_key(value));
                    add_name_words(value, doc_name_words_);
                }
                else if ((c.type == pii_type::location || c.type == pii_type::organization) && c.confidence >= 0.8)
                {
                    /* Places and organisations established in context ("née à Saint-Malo",
                       "35000 Rennes", "Dupont Conseil SAS") are also replaced where they
                       appear bare, in a table cell or a later sentence. */
                    std::vector<std::string> keys = term_keys(value);
                    if (!keys.empty() && c.type == pii_type::location && is_digit_key(keys.front()))
                        keys.erase(keys.begin());
                    if (!keys.empty() && c.type == pii_type::organization)
                    {
                        while (!keys.empty() && privacy_impl::organization_forms().count(keys.back())) keys.pop_back();
                        while (!keys.empty() && privacy_impl::organization_forms().count(keys.front()))
                            keys.erase(keys.begin());
                    }
                    add_doc_term(keys, c.type);
                    add_doc_term(term_keys(value), c.type);
                }
                else if (c.type == pii_type::email)
                {
                    const std::string local = value.substr(0, value.find('@'));
                    std::string frag;
                    for (std::size_t k = 0; k <= local.size(); ++k)
                    {
                        const char ch = k < local.size() ? local[k] : '.';
                        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
                        {
                            frag.push_back(static_cast<char>(ch | 0x20));
                        }
                        else
                        {
                            if (frag.size() >= 3 && privacy_impl::common_words().count(frag) == 0)
                                email_fragments_.insert(frag);
                            frag.clear();
                        }
                    }
                }
            }
        }

        std::vector<pii_entity> detect(
            const std::string& text,
            const detection_context& context = detection_context()
        ) const
        {
            std::vector<privacy_impl::pii_candidate> found;
            const privacy_impl::utf8_text t(text, true);
            analyze(text, t, context, found);
            std::vector<pii_entity> out;
            for (const privacy_impl::pii_candidate& c : found)
            {
                pii_entity e;
                e.begin = t.offsets[c.b];
                e.end = t.offsets[c.e];
                e.type = c.type;
                e.confidence = c.confidence;
                out.push_back(e);
            }
            return out;
        }

        pii_type label_type(const std::string& text) const
        {
            return privacy_impl::lookup_label(privacy_impl::normalize_label(text));
        }

        std::size_t learned_name_count() const { return doc_names_full_.size(); }

    private:
        typedef privacy_impl::pii_candidate candidate;
        typedef privacy_impl::word_token word_token;
        typedef privacy_impl::utf8_text utf8_text;

        bool enabled(pii_type t) const
        {
            const std::size_t k = static_cast<std::size_t>(t);
            return k < enabled_.size() && enabled_[k];
        }

        /* The words of a text and its runs of digits, in order: the units a term is matched
           on, so that "EMP-4521" and "12345" are terms too. The runs of digits have the
           shape 4 (uncased). */
        static std::vector<word_token> term_tokens(const utf8_text& t, const std::vector<word_token>& words)
        {
            std::vector<word_token> tokens;
            tokens.reserve(words.size());
            std::size_t w = 0;
            for (std::size_t i = 0; i < t.size();)
            {
                if (!privacy_impl::is_ascii_digit(t.cps[i])) { ++i; continue; }
                std::size_t j = i;
                word_token d;
                while (j < t.size() && privacy_impl::is_ascii_digit(t.cps[j]))
                    d.key.push_back(static_cast<char>(t.cps[j++]));
                d.b = i;
                d.e = j;
                d.shape = 4;
                while (w < words.size() && words[w].b < i) tokens.push_back(words[w++]);
                tokens.push_back(d);
                i = j;
            }
            while (w < words.size()) tokens.push_back(words[w++]);
            return tokens;
        }

        static std::vector<std::string> term_keys(const std::string& term)
        {
            std::vector<std::string> keys;
            const utf8_text t(term, true);
            for (const word_token& w : term_tokens(t, privacy_impl::tokenize_words(t, term)))
                keys.push_back(w.key);
            return keys;
        }

        static bool is_digit_key(const std::string& key)
        {
            if (key.empty()) return false;
            for (char ch : key)
                if (ch < '0' || ch > '9') return false;
            return true;
        }

        void add_doc_term(const std::vector<std::string>& keys, pii_type type)
        {
            if (keys.empty()) return;
            // A single short or common word is too weak to be replaced everywhere.
            if (keys.size() == 1 &&
                (keys[0].size() < 4 || privacy_impl::common_words().count(keys[0]) || is_digit_key(keys[0])))
                return;
            std::string joined = std::string(pii_type_keyword(type)) + ":";
            for (const std::string& k : keys) joined += k + " ";
            if (!doc_term_set_.insert(joined).second) return;
            doc_term_index_[keys.front()].push_back(doc_terms_.size());
            doc_terms_.push_back(std::make_pair(keys, type));
        }

        static void add_name_words(const std::string& name, std::unordered_set<std::string>& words)
        {
            const utf8_text t(name, true);
            for (const word_token& w : privacy_impl::tokenize_words(t, name))
            {
                if (w.e - w.b < 2) continue;
                if (privacy_impl::person_particles().count(w.key) || privacy_impl::person_titles().count(w.key) ||
                    privacy_impl::common_words().count(w.key))
                    continue;
                words.insert(w.key);
            }
        }

        // True if code point b follows a street type and "du": "rue du ", "avenue du ".
        static bool names_street(const utf8_text& t, std::size_t b)
        {
            using namespace privacy_impl;
            auto word_before = [&](std::size_t& k) {
                while (k > 0 && t.cps[k - 1] == ' ')
                    --k;
                const std::size_t e = k;
                while (k > 0 && is_letter(t.cps[k - 1]))
                    --k;
                std::string w;
                for (std::size_t n = k; n < e; ++n)
                    utf8_append(w, to_lower(t.cps[n]));
                return fold_key(w);
            };
            std::size_t k = b;
            if (word_before(k) != "du")
                return false;
            return street_types().count(word_before(k)) != 0;
        }

        void add(std::vector<candidate>& out, std::size_t b, std::size_t e, pii_type type,
                 double confidence, int bonus = 0, bool strong = false) const
        {
            if (e <= b || !enabled(type)) return;
            candidate c;
            c.b = b;
            c.e = e;
            c.type = type;
            c.priority = privacy_impl::pii_priority(type) + bonus;
            c.confidence = confidence;
            c.strong = strong;
            out.push_back(c);
        }

    // ------------------------------------------------------------------------------------

        void analyze(
            const std::string& text,
            const utf8_text& t,
            const detection_context& context,
            std::vector<candidate>& result
        ) const
        {
            std::vector<candidate> c;
            const std::vector<word_token> words = privacy_impl::tokenize_words(t, text);

            // Existing tags are protected spans: nothing inside them is personal data.
            std::vector<std::pair<std::size_t, std::size_t>> masked = mask_tags(t);

            if (context.forced != pii_type::none)
            {
                std::size_t b = 0, e = t.size();
                while (b < e && privacy_impl::is_space(t.cps[b])) ++b;
                while (e > b && privacy_impl::is_space(t.cps[e - 1])) --e;
                // A column headed "Titre" or "Genre" holds a civility or a sex only when its
                // cell says so: closed vocabularies are checked like any labelled value.
                if (attribute_type(context.forced))
                    type_value(text, t, b, e, context.forced, c);
                else if (e > b && masked.empty())
                    add(c, b, e, context.forced, 1.0, 200, true);
            }

            scan_emails(t, c);
            // Web addresses that are not personal data are protected too: a date in their path stays.
            scan_urls(text, t, c, masked);
            std::sort(masked.begin(), masked.end());
            scan_ip(t, c);
            scan_mac(t, c);
            scan_iban(t, c);
            scan_digit_identifiers(t, c);
            scan_plates(t, c);
            scan_user_paths(t, c);
            scan_geo(t, c);
            scan_national_ids(t, c);
            scan_phone_cues(t, c);
            scan_handles(t, c);
            scan_old_plates(t, c);
            if (enabled(pii_type::date))
            {
                for (const date_match& m : find_dates(text, options_.date_order))
                {
                    const std::size_t b =
                        std::lower_bound(t.offsets.begin(), t.offsets.end(), m.begin) - t.offsets.begin();
                    const std::size_t e =
                        std::lower_bound(t.offsets.begin(), t.offsets.end(), m.end) - t.offsets.begin();
                    // "rue du 18 Juin 1940", "place du 8 Mai 1945": the date names the street.
                    if (names_street(t, b))
                        continue;
                    // "1-29-17 Gion, Tokyo": the numbers of a Japanese address before its district.
                    if (m.format.layout <= 1 && m.format.separator == 1 && m.format.two_digit_year &&
                        t.at(e) == ' ' && privacy_impl::is_upper(t.at(e + 1)))
                        continue;
                    add(c, b, e, pii_type::date, m.format.has_day() ? 0.95 : 0.9);
                }
            }
            scan_labels(text, t, words, context, c);
            scan_tables(text, t, c);
            scan_attributes(text, t, words, c);
            scan_addresses(t, words, c);
            scan_locations(t, words, c);
            scan_foreign_postcodes(text, t, words, c);
            scan_organizations(text, t, words, c);
            scan_known_organizations(text, t, words, c);
            scan_public_bodies(text, t, words, c);
            scan_english_bodies(t, words, c);
            scan_persons(text, t, words, c);
            scan_aliases(t, words, c);
            scan_terms(t, words, c);
            scan_cued_references(t, words, c);
            scan_year_references(t, c);
            scan_logins(t, words, c);
            scan_patterns(text, t, c);
            scan_recognizers(text, t, c);

            // Resolution: higher priority first, then longer, then earlier.
            std::sort(c.begin(), c.end(), [](const candidate& a, const candidate& b) {
                if (a.priority != b.priority) return a.priority > b.priority;
                if (a.e - a.b != b.e - b.b) return a.e - a.b > b.e - b.b;
                return a.b < b.b;
            });
            /* Accepted spans never overlap: they are kept, with the masked spans merged, in a
               map of disjoint ranges, where an overlap is found by looking at the range that
               starts just before the end of the candidate. */
            std::vector<candidate> accepted;
            std::map<std::size_t, std::size_t> taken;
            for (const auto& m : masked)
            {
                if (m.second <= m.first) continue;
                if (!taken.empty() && taken.rbegin()->second >= m.first)
                    taken.rbegin()->second = std::max(taken.rbegin()->second, m.second);
                else
                    taken[m.first] = m.second;
            }
            for (const candidate& x : c)
            {
                if (overlaps(taken, x.b, x.e)) continue;
                if (!allowed_.empty() &&
                    allowed_.count(privacy_impl::fold_key(privacy_impl::cp_substr(t, text, x.b, x.e))))
                    continue;
                accepted.push_back(x);
                taken[x.b] = x.e;
            }
            join_given_and_family_names(text, t, taken, accepted);
            attach_name_edges(t, words, taken, accepted);
            attach_department_codes(t, taken, accepted);
            attach_titles(t, words, taken, accepted);
            attach_attributes(text, t, taken, accepted);
            std::sort(accepted.begin(), accepted.end(),
                      [](const candidate& a, const candidate& b) { return a.b < b.b; });
            result.swap(accepted);
        }

    // ------------------------------------------------------------------------------------
    // Civilities, sexes and nationalities.

        static bool attribute_type(pii_type t)
        {
            return t == pii_type::title || t == pii_type::gender || t == pii_type::nationality;
        }

        /* Spans of the tags of pseudonymizer: "[[KEYWORD:body#CCC]]", with a known keyword, a
           body without brackets and a check of three Crockford characters. The closing
           marker is searched once for each run of text, which keeps the scan linear. */
        static std::vector<std::pair<std::size_t, std::size_t>> mask_tags(const utf8_text& t)
        {
            static const std::unordered_set<std::string> keywords = []() {
                std::unordered_set<std::string> k;
                for (pii_type type : all_pii_types()) k.insert(pii_type_keyword(type));
                return k;
            }();
            std::vector<std::pair<std::size_t, std::size_t>> masked;
            const std::size_t n = t.size();
            std::size_t close = 0;
            bool close_valid = false;
            for (std::size_t i = 0; i + 1 < n; ++i)
            {
                if (t.cps[i] != '[' || t.cps[i + 1] != '[') continue;
                if (!close_valid || close < i + 2)
                {
                    close = i + 2;
                    while (close + 1 < n && !(t.cps[close] == ']' && t.cps[close + 1] == ']')) ++close;
                    if (close + 1 >= n) close = n;
                    close_valid = true;
                }
                if (close >= n) break;
                std::size_t k = i + 2;
                std::string keyword;
                while (k < close && keyword.size() < 16 && ((t.cps[k] >= 'A' && t.cps[k] <= 'Z') ||
                                                             (t.cps[k] >= '0' && t.cps[k] <= '9')))
                    keyword.push_back(static_cast<char>(t.cps[k++]));
                if (k >= close || t.cps[k] != ':' || !keywords.count(keyword) || close < k + 6 ||
                    t.cps[close - 4] != '#')
                    continue;
                bool ok = true;
                for (std::size_t q = close - 3; q < close && ok; ++q)
                    ok = t.cps[q] < 0x80 && privacy_impl::crockford_value(static_cast<char>(t.cps[q])) >= 0;
                for (std::size_t q = k + 1; q + 4 < close && ok; ++q)
                    ok = t.cps[q] != '[' && t.cps[q] != ']';
                if (!ok) continue;
                masked.push_back(std::make_pair(i, close + 2));
                i = close + 1;
            }
            return masked;
        }

        // folded is a value folded with privacy_impl::fold_key().
        static bool attribute_value_ok(pii_type type, const std::string& folded)
        {
            using namespace privacy_impl;
            if (folded.empty()) return false;
            if (type == pii_type::title)
            {
                std::string k = folded;
                if (k.back() == '.') k.pop_back();
                return civility_words().count(k) != 0;
            }
            if (type == pii_type::gender)
            {
                std::string k;
                for (char c : folded)
                    if (c != ' ' && c != '-' && c != '.') k.push_back(c);
                return gender_values().count(k) != 0;
            }
            if (type == pii_type::nationality)
                return nationality_words().count(folded) != 0 || country_names().count(folded) != 0;
            return false;
        }

        // Whether [b, e) overlaps one of the disjoint ranges of taken, a map from begin to end.
        static bool overlaps(const std::map<std::size_t, std::size_t>& taken, std::size_t b, std::size_t e)
        {
            auto it = taken.lower_bound(e);
            if (it == taken.begin()) return false;
            --it;
            return it->second > b;
        }

        /* True if a civility right before b, or a name label before a colon, announces the value
           that starts at b: "Mme Le Floc'h", "Dr Le Floc'h", "Nom : LE FLOC'H", "Last name: Smith".
           A function ("le maire de Quimper") and other labels ("Expéditeur :") may announce a place
           or an organisation as well as a person. */
        static bool announced_person(const std::string& text, const utf8_text& t, std::size_t b)
        {
            using namespace privacy_impl;
            static const std::unordered_set<std::string> labels = make_phrase_set(
                "nom|prenom|prenoms|nom de famille|nom de naissance|nom d usage|nom marital|nom complet|"
                "nom et prenom|nom prenom|name|surname|last name|first name|given name|family name|full name");
            static const std::unordered_set<std::string> civilities = make_word_set(
                "m mr mrs ms mx mme mmes mlle mlles madame monsieur mademoiselle messieurs mesdames me maitre "
                "dr docteur pr professeur sir dame");
            auto blank = [&](std::size_t i) { return t.cps[i] == ' ' || t.cps[i] == 0xA0 || t.cps[i] == '\t'; };
            std::size_t k = b;
            while (k > 0 && blank(k - 1)) --k;
            bool colon = false;
            if (k > 0 && t.cps[k - 1] == ':')
            {
                colon = true;
                --k;
                while (k > 0 && blank(k - 1)) --k;
            }
            else if (k > 0 && t.cps[k - 1] == '.')
            {
                --k;
            }
            std::size_t s = k;
            while (s > 0 && k - s < 40 && !is_line_break(t.cps[s - 1]) && t.cps[s - 1] != '\t' &&
                   t.cps[s - 1] != ':' && t.cps[s - 1] != '|' && t.cps[s - 1] != ',' && t.cps[s - 1] != ';')
                --s;
            if (colon)
                return labels.count(normalize_label(cp_substr(t, text, s, k))) != 0;
            std::size_t w = k;
            while (w > s && is_letter(t.cps[w - 1])) --w;
            return w < k && civilities.count(fold_key(cp_substr(t, text, w, k))) != 0;
        }

        /* True if a word between b and e is a legal form, the head noun, the sector or the trade of
           a firm, or the type of a street: words that no person name holds. */
        static bool firm_or_street_words(const std::string& text, const utf8_text& t, std::size_t b, std::size_t e)
        {
            using namespace privacy_impl;
            for (std::size_t q = b; q < e;)
            {
                if (!is_letter(t.cps[q])) { ++q; continue; }
                std::size_t r = q;
                while (r < e && is_letter(t.cps[r])) ++r;
                const std::string key = fold_key(cp_substr(t, text, q, r));
                if (organization_forms().count(key) || organization_heads().count(key) ||
                    organization_sectors().count(key) || organization_trade_words().count(key) ||
                    street_types().count(key))
                    return true;
                q = r;
            }
            return false;
        }

        /* A first name read as a person and the family name after it read apart, as a person, an
           organisation, an address or a place, as a recognizer may read a name with a particle and
           an apostrophe: "Yann" and "Le Floc'h". The two make one person when the second opens
           with a capitalized particle, then holds one or two capitalized words, none of them a
           word of a firm or of a street, and is no organisation known by its own name. */
        static void join_given_and_family_names(const std::string& text, const utf8_text& t,
                                                std::map<std::size_t, std::size_t>& taken,
                                                std::vector<candidate>& accepted)
        {
            using namespace privacy_impl;
            std::vector<std::size_t> order(accepted.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(),
                      [&](std::size_t a, std::size_t b) { return accepted[a].b < accepted[b].b; });
            std::vector<bool> removed(accepted.size(), false);
            for (std::size_t i = 0; i + 1 < order.size(); ++i)
            {
                candidate& x = accepted[order[i]];
                candidate& y = accepted[order[i + 1]];
                if (removed[order[i]] || x.type != pii_type::person ||
                    !(y.type == pii_type::organization || y.type == pii_type::address ||
                      y.type == pii_type::location || y.type == pii_type::person) ||
                    y.b != x.e + 1 || !(t.cps[x.e] == ' ' || t.cps[x.e] == 0xA0))
                    continue;
                std::size_t w = x.e;
                while (w > x.b && is_letter(t.cps[w - 1])) --w;
                if (w == x.e || !first_names().count(fold_key(cp_substr(t, text, w, x.e))))
                    continue;
                // The words of the second span, separated by single spaces.
                std::vector<std::pair<std::size_t, std::size_t>> parts;
                bool ok = true;
                for (std::size_t q = y.b; q < y.e && ok;)
                {
                    std::size_t r = q;
                    while (r < y.e && (is_letter(t.cps[r]) ||
                                       ((is_apostrophe(t.cps[r]) || is_hyphen(t.cps[r])) && r > q &&
                                        r + 1 < t.size() && is_letter(t.cps[r + 1]))))
                        ++r;
                    if (r == q) ok = false;
                    parts.push_back(std::make_pair(q, r));
                    if (r < y.e && !(t.cps[r] == ' ' && r + 1 < y.e)) ok = false;
                    q = r + 1;
                }
                if (!ok || parts.size() < 2 || parts.size() > 3)
                    continue;
                const std::string particle = fold_key(cp_substr(t, text, parts[0].first, parts[0].second));
                if (!person_particles().count(particle) || !is_upper(t.cps[parts[0].first]) ||
                    known_organizations().count(fold_key(cp_substr(t, text, y.b, y.e))))
                    continue;
                for (std::size_t p = 1; p < parts.size() && ok; ++p)
                {
                    const std::string key = fold_key(cp_substr(t, text, parts[p].first, parts[p].second));
                    ok = is_upper(t.cps[parts[p].first]) && !common_words().count(key) && !function_words().count(key);
                }
                if (!ok || firm_or_street_words(text, t, y.b, y.e))
                    continue;
                taken.erase(y.b);
                taken[x.b] = y.e;
                x.e = y.e;
                x.confidence = std::min(x.confidence, y.confidence);
                removed[order[i + 1]] = true;
                ++i;
            }
            std::vector<candidate> kept;
            for (std::size_t i = 0; i < accepted.size(); ++i)
                if (!removed[i]) kept.push_back(accepted[i]);
            accepted.swap(kept);
        }

        /* The edges of a name that a rule or a recognizer leaves out. The end of a word after an
           apostrophe, as in the Breton "c'h" ("Le Floc'h", "Penmarc'h") or in "Prud'homme", goes
           with the name, place, address or organisation; an English possessive or contraction
           ("Martin's") and the elision of "que" ("jusqu'") stay out. A capitalized particle right
           before a person, in the case of the name, opens it: "LE FLOC'H Yann", "DE LA FONTAINE
           Jean", "avec Le Bihan"; "de", "du" and "des", which may be prepositions, only before
           another particle. */
        static void attach_name_edges(const utf8_text& t, const std::vector<word_token>& words,
                                      std::map<std::size_t, std::size_t>& taken,
                                      std::vector<candidate>& accepted)
        {
            using namespace privacy_impl;
            static const std::unordered_set<std::string> clitics = make_word_set("s t d m ll re ve");
            static const std::unordered_set<std::string> prepositions = make_word_set("de du des d");
            for (candidate& x : accepted)
            {
                if (!(x.type == pii_type::person || x.type == pii_type::organization ||
                      x.type == pii_type::location || x.type == pii_type::address))
                    continue;
                const std::size_t e = x.e;
                if (e >= x.b + 2 && is_apostrophe(t.at(e)) && is_letter(t.cps[e - 1]) && is_letter(t.cps[e - 2]) &&
                    is_letter(t.at(e + 1)) && !(to_lower(t.cps[e - 2]) == 'q' && to_lower(t.cps[e - 1]) == 'u'))
                {
                    const bool upper_word = is_upper(t.cps[e - 1]) && is_upper(t.cps[e - 2]);
                    std::size_t d = e + 1;
                    bool same_case = true;
                    std::string tail;
                    while (d < t.size() && is_letter(t.cps[d]))
                    {
                        if (is_upper(t.cps[d]) != upper_word) same_case = false;
                        utf8_append(tail, to_lower(t.cps[d]));
                        ++d;
                    }
                    if (same_case && !is_word_char(t.at(d)) && !clitics.count(tail) && !overlaps(taken, e, d))
                    {
                        taken[x.b] = d;
                        x.e = d;
                    }
                }
                if (x.type != pii_type::person)
                    continue;
                auto it = std::lower_bound(words.begin(), words.end(), x.b,
                                           [](const word_token& w, std::size_t p) { return w.b < p; });
                if (it == words.end() || it->b != x.b)
                    continue;
                std::size_t first = static_cast<std::size_t>(it - words.begin());
                const bool upper_head = it->shape == 2 && it->e - it->b >= 2;
                if (!(upper_head || it->shape == 1 || it->shape == 3))
                    continue;
                bool head_particle = person_particles().count(it->key) != 0;
                std::size_t b = x.b;
                for (std::size_t k = first; k > 0 && first - k < 2; --k)
                {
                    const word_token& p = words[k - 1];
                    if (p.e + 1 != words[k].b || !(t.cps[p.e] == ' ' || t.cps[p.e] == 0xA0) ||
                        !person_particles().count(p.key) || p.e - p.b > 4 ||
                        !(upper_head ? p.shape == 2 : p.shape == 1) ||
                        (prepositions.count(p.key) && !head_particle) ||
                        (p.b > 0 && (is_word_char(t.cps[p.b - 1]) || is_apostrophe(t.cps[p.b - 1]) ||
                                     is_hyphen(t.cps[p.b - 1]) || t.cps[p.b - 1] == '.')) ||
                        overlaps(taken, p.b, b))
                        break;
                    b = p.b;
                    head_particle = true;
                }
                if (b < x.b)
                {
                    taken.erase(x.b);
                    taken[b] = x.e;
                    x.b = b;
                }
            }
        }

        /* "Quimper (29)", "Quimper (2A)", "Quimper (29000)": the department or the postal code
           written in brackets after a town goes with it. */
        static void attach_department_codes(const utf8_text& t, std::map<std::size_t, std::size_t>& taken,
                                            std::vector<candidate>& accepted)
        {
            using namespace privacy_impl;
            for (candidate& x : accepted)
            {
                if (x.type != pii_type::location)
                    continue;
                std::size_t k = x.e;
                // A name learned with its code is found again up to its last word: "Quimper (29".
                while (k > x.b && x.e - k < 6 && (is_ascii_digit(t.cps[k - 1]) || t.cps[k - 1] == 'A' ||
                                                  t.cps[k - 1] == 'B'))
                    --k;
                if (k < x.e && k > x.b && t.cps[k - 1] == '(')
                    k = k - 1;
                else
                    k = x.e;
                if (t.at(k) == ' ')
                    ++k;
                if (t.at(k) != '(')
                    continue;
                std::size_t d = k + 1;
                std::string code;
                while (d < t.size() && d - k <= 6 && (is_ascii_digit(t.cps[d]) || t.cps[d] == 'A' || t.cps[d] == 'B'))
                    code.push_back(static_cast<char>(t.cps[d++]));
                if (t.at(d) != ')' || is_word_char(t.at(d + 1)))
                    continue;
                const bool digits = code.find_first_not_of("0123456789") == std::string::npos;
                const int value = digits && !code.empty() ? std::stoi(code) : -1;
                const bool department = (code.size() == 2 && digits && value >= 1 && value <= 95) ||
                                        code == "2A" || code == "2B" ||
                                        (code.size() == 3 && digits && value >= 971 && value <= 976);
                if (!department && !(code.size() == 5 && digits))
                    continue;
                if (overlaps(taken, x.e, d + 1))
                    continue;
                taken[x.b] = d + 1;
                x.e = d + 1;
            }
        }

        /* The length of a civility glued to the name after it by a lost space:
           "MonsieurVERDIER", "MmeLamia", "MrsAshley". 0 when the word starts with none. */
        static std::size_t glued_civility(const utf8_text& t, const word_token& w)
        {
            static const char* const forms[] = { "Mademoiselle", "Monsieur", "Madame", "MONSIEUR", "MADAME", "Mlle",
                                                 "Mme", "Mrs", "Mr", "Ms" };
            for (const char* f : forms)
            {
                const std::size_t len = std::char_traits<char>::length(f);
                if (w.e - w.b < len + 2) continue;
                bool same = true;
                for (std::size_t i = 0; i < len && same; ++i)
                    same = t.cps[w.b + i] == static_cast<char32_t>(f[i]);
                if (same && privacy_impl::is_upper(t.cps[w.b + len])) return len;
            }
            return 0;
        }

        /* A civility right before a name tells the sex of the person and is replaced as
           TITLE: "M. Dupont", "Mrs Smith", "M. et Mme Martin", "mr ethan woodwell", and
           "MmeLamia Chaoui" when a space was lost. A civility without a name ("Madame,
           Monsieur,", "Monsieur le Directeur") and professional titles (Dr, Me) stay
           readable. */
        void attach_titles(const utf8_text& t, const std::vector<word_token>& words,
                           std::map<std::size_t, std::size_t>& taken,
                           std::vector<candidate>& accepted) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::title) || words.empty()) return;
            auto free_span = [&](std::size_t b, std::size_t e) { return !overlaps(taken, b, e); };
            // The word ending at or before position p, with only spaces between, on the line.
            auto word_before = [&](std::size_t p, std::size_t& index) {
                auto it = std::lower_bound(words.begin(), words.end(), p,
                    [](const word_token& w, std::size_t pos) { return w.e <= pos; });
                if (it == words.begin()) return false;
                index = static_cast<std::size_t>(it - words.begin()) - 1;
                std::size_t end = words[index].e;
                if (t.at(end) == '.' && end < p) ++end;
                if (end > p || p - end > 2) return false;
                for (std::size_t g = end; g < p; ++g)
                    if (!(t.cps[g] == ' ' || t.cps[g] == 0xA0)) return false;
                return true;
            };
            auto civility_at = [&](std::size_t k, std::size_t& end) {
                const word_token& w = words[k];
                if (!civility_words().count(w.key)) return false;
                end = w.e;
                if (t.at(end) == '.') ++end;
                // A single M is an initial unless its period follows.
                return !(w.key == "m" && end == w.e);
            };
            auto title_candidate = [&](std::size_t b, std::size_t e) {
                candidate c;
                c.b = b;
                c.e = e;
                c.type = pii_type::title;
                c.priority = pii_priority(pii_type::title);
                c.confidence = 0.95;
                return c;
            };
            std::vector<candidate> titles;
            // "M. et Mme Martin": the civility before the conjunction goes with the name too.
            auto joined_civility = [&](std::size_t k) {
                std::size_t j = 0, end = 0;
                if (k > 0 && (words[k - 1].key == "et" || words[k - 1].key == "and") &&
                    word_before(words[k].b, j) && j == k - 1 && word_before(words[k - 1].b, j) &&
                    civility_at(j, end) && free_span(words[j].b, end))
                {
                    titles.push_back(title_candidate(words[j].b, end));
                    taken[words[j].b] = end;
                }
            };
            for (candidate& p : accepted)
            {
                if (p.type != pii_type::person) continue;
                // "MmeLamia Chaoui": the name starts inside the word of the civility, or with it.
                auto inside = std::upper_bound(words.begin(), words.end(), p.b,
                    [](std::size_t pos, const word_token& w) { return pos < w.b; });
                if (inside != words.begin())
                {
                    const word_token& w = *(inside - 1);
                    const std::size_t glued = glued_civility(t, w);
                    if (glued > 0 && w.e > p.b && (p.b == w.b || p.b == w.b + glued))
                    {
                        const std::size_t name = w.b + glued;
                        if (p.b == w.b)
                        {
                            taken.erase(p.b);
                            p.b = name;
                            taken[p.b] = p.e;
                        }
                        if (free_span(w.b, name))
                        {
                            titles.push_back(title_candidate(w.b, name));
                            taken[w.b] = name;
                            joined_civility(static_cast<std::size_t>(inside - words.begin()) - 1);
                        }
                        continue;
                    }
                }
                std::size_t k = 0, end = 0;
                if (!word_before(p.b, k) || !civility_at(k, end) || !free_span(words[k].b, end)) continue;
                /* After another title, M. is the initial of the name: "Ms M. Cook", "Monsieur
                   M. Denis", "Dr M. Martin". The name takes the initial, and the title before
                   it, if it is a civility, is the one replaced. */
                std::size_t j0 = 0, end0 = 0;
                if (words[k].key == "m" && word_before(words[k].b, j0) && person_titles().count(words[j0].key) &&
                    words[j0].shape != 0)
                {
                    taken.erase(p.b);
                    p.b = words[k].b;
                    taken[p.b] = p.e;
                    if (!civility_at(j0, end0) || !free_span(words[j0].b, end0)) continue;
                    k = j0;
                    end = end0;
                }
                titles.push_back(title_candidate(words[k].b, end));
                taken[words[k].b] = end;
                joined_civility(k);
            }
            accepted.insert(accepted.end(), titles.begin(), titles.end());
        }

        /* Values that describe a person found beside it: a nationality in apposition
           ("Louis Patel, American, currently...", "Mme Oliveida, francaise de 73 ans"), a
           reference between parentheses ("Matteo Lombardi (3323086)"), and the cells of a
           row that holds a name, where a sex, a nationality or a civility fills a cell of its
           own ("PEREIRA Awa – F – belge – agent", "Mme;CARRE;Stephanie;F"). A row has three
           cells at least, four for a civility in a row cut by commas; a sex is written in
           full or as M, F or H in capitals. */
        void attach_attributes(const std::string& text, const utf8_text& t,
                               std::map<std::size_t, std::size_t>& taken,
                               std::vector<candidate>& accepted) const
        {
            using namespace privacy_impl;
            const bool nationality = enabled(pii_type::nationality), gender = enabled(pii_type::gender),
                       title = enabled(pii_type::title);
            if (!nationality && !gender && !title && !enabled(pii_type::identifier)) return;
            const std::size_t n = t.size();
            std::vector<candidate> found;
            auto take = [&](std::size_t b, std::size_t e, pii_type type) {
                if (e <= b || overlaps(taken, b, e)) return;
                candidate c;
                c.b = b;
                c.e = e;
                c.type = type;
                c.priority = pii_priority(type);
                c.confidence = 0.85;
                found.push_back(c);
                taken[b] = e;
            };
            static const std::unordered_set<std::string> sexes = make_word_set("masculin feminin homme femme male female");
            auto cell_type = [&](std::size_t b, std::size_t e, bool titles) {
                const std::string key = fold_key(cp_substr(t, text, b, e));
                if (key.empty() || e - b > 40) return pii_type::none;
                if (nationality && nationality_words().count(key)) return pii_type::nationality;
                if (gender && (sexes.count(key) ||
                               (e - b == 1 && (t.cps[b] == 'M' || t.cps[b] == 'F' || t.cps[b] == 'H'))))
                    return pii_type::gender;
                if (titles && title && attribute_value_ok(pii_type::title, key)) return pii_type::title;
                return pii_type::none;
            };
            std::vector<std::size_t> lines;
            for (const candidate& p : accepted)
            {
                if (p.type != pii_type::person) continue;
                // ", American," after the name.
                if (nationality && p.e + 2 < n && t.cps[p.e] == ',' && t.cps[p.e + 1] == ' ')
                {
                    const std::size_t b = p.e + 2;
                    std::size_t e = b;
                    while (e < n && (is_letter(t.cps[e]) || (is_hyphen(t.cps[e]) && is_letter(t.at(e + 1))))) ++e;
                    const char32_t c = t.at(e);
                    if (e > b && (c == 0 || c == ',' || c == ' ' || c == '.' || c == ';' || c == ')' || is_line_break(c)) &&
                        nationality_words().count(fold_key(cp_substr(t, text, b, e))))
                        take(b, e, pii_type::nationality);
                }
                // "Monsieur Matteo Lombardi (3323086)", "Mr. Carter (2024-97993)": a reference after the name.
                std::size_t open = p.e;
                while (open < n && open < p.e + 2 && t.cps[open] == ' ') ++open;
                if (enabled(pii_type::identifier) && open > p.e && open + 2 < n && t.cps[open] == '(')
                {
                    const std::size_t b = open + 1;
                    std::size_t e = b, digits = 0;
                    while (e < n && e - b <= 20 && (is_ascii_alnum_cp(t.cps[e]) ||
                                                    ((t.cps[e] == '-' || t.cps[e] == '/') && e > b)))
                    {
                        if (is_ascii_digit(t.cps[e])) ++digits;
                        ++e;
                    }
                    if (e < n && t.cps[e] == ')' && e - b >= 5 && digits >= 4 && is_ascii_alnum_cp(t.cps[e - 1]))
                        take(b, e, pii_type::identifier);
                }
                std::size_t ls = p.b;
                while (ls > 0 && !is_line_break(t.cps[ls - 1])) --ls;
                lines.push_back(ls);
            }
            std::sort(lines.begin(), lines.end());
            lines.erase(std::unique(lines.begin(), lines.end()), lines.end());
            static const char32_t separators[] = { '|', '\t', ';', 0x2013, ',' };
            for (std::size_t ls : lines)
            {
                std::size_t le = ls;
                while (le < n && !is_line_break(t.cps[le])) ++le;
                for (char32_t sep : separators)
                {
                    // Cells cut at the separator; an en dash separates only between spaces.
                    std::vector<std::pair<std::size_t, std::size_t>> cells;
                    std::size_t cb = ls;
                    for (std::size_t q = ls; q <= le; ++q)
                    {
                        const bool cut = q == le || (t.cps[q] == sep && (sep != 0x2013 ||
                            (q > ls && t.cps[q - 1] == ' ' && t.at(q + 1) == ' ')));
                        if (!cut) continue;
                        std::size_t b = cb, e = q;
                        while (b < e && is_space(t.cps[b])) ++b;
                        while (e > b && is_space(t.cps[e - 1])) --e;
                        cells.push_back(std::make_pair(b, e));
                        cb = q + 1;
                    }
                    // The borders of "| a | b |" make empty cells.
                    if (sep == '|')
                    {
                        if (!cells.empty() && cells.front().first == cells.front().second) cells.erase(cells.begin());
                        if (!cells.empty() && cells.back().first == cells.back().second) cells.pop_back();
                    }
                    if (cells.size() < 3) continue;
                    const bool titles = sep != ',' || cells.size() >= 4;
                    for (const auto& c : cells)
                    {
                        if (c.first == c.second) continue;
                        const pii_type type = cell_type(c.first, c.second, titles);
                        if (type != pii_type::none) take(c.first, c.second, type);
                    }
                    break;
                }
            }
            accepted.insert(accepted.end(), found.begin(), found.end());
        }

        /* Nationalities and sexes written in the text after a cue: "de nationalite
           francaise", "ressortissant algerien", "passeport marocain", "a British citizen",
           "your Bulgarian identity card", "an 82 year old American", "francaise de 73 ans",
           "de sexe feminin". Labelled values ("Nationalite : francaise") go through
           type_value() like any other field. */
        void scan_attributes(const std::string& text, const utf8_text& t, const std::vector<word_token>& words,
                             std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            const bool nationality = enabled(pii_type::nationality), gender = enabled(pii_type::gender);
            if (!nationality && !gender) return;
            static const std::unordered_set<std::string> nationality_before = make_word_set(
                "nationalite nationalites citoyen citoyenne citoyens citoyennes ressortissant ressortissante "
                "ressortissants ressortissantes nationality citizenship passeport passeports identite");
            static const std::unordered_set<std::string> nationality_after = make_word_set(
                "national nationals citizen citizens nationality passport passports identity id age agee ages agees");
            static const std::unordered_set<std::string> gender_before = make_word_set("sexe sex");
            auto close = [&](std::size_t a, std::size_t b) {
                if (b < a || b - a > 3 || !same_line(t, a, b)) return false;
                for (std::size_t g = a; g < b; ++g)
                    if (!(t.cps[g] == ' ' || t.cps[g] == 0xA0 || t.cps[g] == ':')) return false;
                return true;
            };
            // "an 82 year old American", "a 32-year-old British engineer".
            auto age_before = [&](std::size_t k) {
                if (k < 2 || words[k - 1].key != "old" || !(words[k - 2].key == "year" || words[k - 2].key == "years"))
                    return false;
                std::size_t p = words[k - 2].b;
                while (p > 0 && (t.cps[p - 1] == ' ' || is_hyphen(t.cps[p - 1]))) --p;
                return p > 0 && is_ascii_digit(t.cps[p - 1]) && same_line(t, p, words[k].b) &&
                       close(words[k - 1].e, words[k].b);
            };
            // "francaise de 73 ans".
            auto age_after = [&](std::size_t a, std::size_t e) {
                if (a + 1 >= words.size() || words[a].key != "de" || words[a + 1].key != "ans" || !close(e, words[a].b))
                    return false;
                bool digit = false;
                for (std::size_t g = words[a].e; g < words[a + 1].b; ++g)
                {
                    if (is_ascii_digit(t.cps[g])) digit = true;
                    else if (t.cps[g] != ' ' && t.cps[g] != 0xA0) return false;
                }
                return digit;
            };
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                for (std::size_t len = 2; len >= 1; --len)
                {
                    if (k + len > words.size()) continue;
                    const std::size_t b = words[k].b, e = words[k + len - 1].e;
                    if (len == 2 && !(words[k + 1].b == words[k].e + 1 &&
                                      (t.cps[words[k].e] == ' ' || is_hyphen(t.cps[words[k].e]))))
                        continue;
                    const std::string key = len == 1 ? words[k].key : fold_key(cp_substr(t, text, b, e));
                    if (nationality && nationality_words().count(key))
                    {
                        const bool cued =
                            (k > 0 && nationality_before.count(words[k - 1].key) && close(words[k - 1].e, b)) ||
                            (k + len < words.size() && nationality_after.count(words[k + len].key) &&
                             close(e, words[k + len].b)) ||
                            age_before(k) || age_after(k + len, e);
                        if (cued)
                        {
                            add(out, b, e, pii_type::nationality, 0.9, 6);
                            k += len - 1;
                            break;
                        }
                    }
                    // "de sexe feminin", "sexe F.", "sex m".
                    if (gender && len == 1 && attribute_value_ok(pii_type::gender, key) &&
                        k > 0 && gender_before.count(words[k - 1].key) && close(words[k - 1].e, b))
                    {
                        add(out, b, e, pii_type::gender, 0.9, 6);
                        break;
                    }
                }
            }
        }

    // ------------------------------------------------------------------------------------
    // Public bodies.

        /* Administrations, courts, social security bodies and regulators are organisations
           too, and documents often must not say which one they deal with: acronyms in
           capitals ("la CAF", "HMRC"), names of their own ("France Travail", "Conseil
           d'Etat", "Home Office") and a head noun with what the body does and where
           ("Tribunal judiciaire de Rennes", "Ministere de l'Interieur", "Department for
           Education"). A head noun alone, or with generic adjectives only ("la direction
           generale"), stays readable. */
        void scan_public_bodies(const std::string& text, const utf8_text& t, const std::vector<word_token>& words,
                                std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!options_.public_bodies || !enabled(pii_type::organization)) return;
            const auto& acronyms = public_body_acronyms();
            const auto& names = public_body_names();
            const auto& heads = public_body_heads();
            const auto& qualifiers = public_body_words();
            static const std::unordered_set<std::string> connectors = make_word_set(
                "de du des d l la le les of for the and et pour sur en aux au");
            static const std::unordered_set<std::string> generic = make_word_set(
                "judiciaire administratif administrative correctionnel correctionnelle departemental "
                "departementale regional regionale municipal municipale general generale interdepartemental "
                "interdepartementale primaire nationale national nationaux superieur superieure "
                "interministeriel interministerielle territorial territoriale territoriaux publique publiques "
                "public publics");
            static const std::unordered_set<std::string> public_adjectives = make_word_set(
                "national nationale general generale regional regionale departemental departementale "
                "interministeriel interministerielle interdepartemental interdepartementale territorial "
                "territoriale superieur superieure europeen europeenne");
            static const std::unordered_set<std::string> branch_heads = make_word_set(
                "agence direction antenne bureau succursale");
            static const std::unordered_set<std::string> internal_heads = make_word_set(
                "conseil direction commission office department departement");
            // The last words of the English bodies that have local branches, and the words of their places.
            static const std::unordered_set<std::string> english_branch_words = make_word_set(
                "plus advice office agency authority service");
            static const std::unordered_set<std::string> compass_words = make_word_set(
                "north south east west little great upper lower new old");
            // Words that name the function of a council or a department, not the body itself.
            static const std::unordered_set<std::string> function_words = make_word_set(
                "scientifique scientifiques technique techniques financier financiere financiers financieres "
                "finances finance juridique juridiques medical medicale medicaux pedagogique consultatif "
                "consultative executif executive strategique administration surveillance travail avis ethique "
                "discipline gestion orientation coordination pilotage suivi direction communication achats "
                "informatique ressources humaines personnel recherche developpement formation qualite audit "
                "budget scientific technical advisory ethics management steering supervisory works staff");
            auto joined = [&](std::size_t a, std::size_t b) {
                if (b < a || b - a > 2 || !same_line(t, a, b)) return false;
                for (std::size_t g = a; g < b; ++g)
                {
                    const char32_t c = t.cps[g];
                    if (!(c == ' ' || c == 0xA0 || is_hyphen(c) || is_apostrophe(c) || c == '&')) return false;
                }
                return true;
            };
            // Inside a name of its own: "HM Courts & Tribunals Service", "Foreign, Commonwealth & Development Office".
            auto name_joined = [&](std::size_t a, std::size_t b) {
                if (b < a || b - a > 3 || !same_line(t, a, b)) return false;
                for (std::size_t g = a; g < b; ++g)
                {
                    const char32_t c = t.cps[g];
                    if (!(c == ' ' || c == 0xA0 || is_hyphen(c) || is_apostrophe(c) || c == '&' || c == ','))
                        return false;
                }
                return true;
            };
            /* The capitalized words of a place after "de", "d'" or "du": "CAF de Saint-Brieuc",
               "CPAM de L'Isle". */
            auto place_after = [&](std::size_t k, std::size_t& last) {
                if (k + 1 >= words.size() || !joined(words[k].e, words[k + 1].b)) return false;
                const std::string& c = words[k + 1].key;
                std::size_t j = k + 2;
                if (c == "d" && words[k + 1].e + 1 == (j < words.size() ? words[j].b : 0)) {}
                else if (!(c == "de" || c == "du")) return false;
                if (j >= words.size() || !joined(words[j - 1].e, words[j].b) ||
                    !(words[j].shape == 1 || words[j].shape == 2))
                    return false;
                last = j;
                // A common word in capitals ends the place: "CCAS DE LANESTER A RENDU HOMMAGE".
                while (last + 1 < words.size() && last < j + 5 && words[last + 1].b == words[last].e + 1 &&
                       (is_hyphen(t.cps[words[last].e]) || t.cps[words[last].e] == ' ' ||
                        (is_apostrophe(t.cps[words[last].e]) && words[last].e - words[last].b == 1)) &&
                       (words[last + 1].shape != 0 || location_kept_words().count(words[last + 1].key)) &&
                       !(words[last + 1].shape == 2 && common_words().count(words[last + 1].key) &&
                         !location_kept_words().count(words[last + 1].key)))
                    ++last;
                while (last > j && words[last].shape == 0) --last;
                return true;
            };
            static const std::unordered_set<std::string> number_words = make_word_set("number no numero num nr n");
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                const word_token& w = words[k];
                // "CAF", "HMRC", and "DGFiP" in its own case.
                if ((w.shape == 2 || w.shape == 3) && acronyms.count(w.key) && boundary_before(t, w.b) &&
                    boundary_after(t, w.e))
                {
                    // "NHS number", "AHV: 756.2846.9692.53": the label of a number.
                    if (k + 1 < words.size() && number_words.count(words[k + 1].key) && joined(w.e, words[k + 1].b))
                        continue;
                    {
                        std::size_t q = w.e;
                        if (q < t.size() && (t.cps[q] == ':' || t.cps[q] == '='))
                        {
                            ++q;
                            while (q < t.size() && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                            if (q < t.size() && is_ascii_digit(t.cps[q])) continue;
                        }
                    }
                    std::size_t last = k;
                    const bool with_place = place_after(k, last);
                    add(out, w.b, with_place ? words[last].e : w.e, pii_type::organization, 0.9, 10);
                    if (with_place) k = last;
                    continue;
                }
                if (w.shape == 1 || w.shape == 2)
                {
                    std::size_t best = 0;
                    for (std::size_t len = 1; len <= 12 && k + len <= words.size(); ++len)
                    {
                        const std::size_t j = k + len - 1;
                        if (len > 1 && !name_joined(words[j - 1].e, words[j].b)) break;
                        if (names.count(fold_key(cp_substr(t, text, w.b, words[j].e)))) best = len;
                    }
                    /* "numero de securite sociale", "Securite sociale : 1 85 05": the label of a number
                       names no body. */
                    if (best > 0)
                    {
                        std::size_t before = k;
                        if (before > 0 && (words[before - 1].key == "de" || words[before - 1].key == "d") &&
                            joined(words[before - 1].e, w.b))
                            --before;
                        const bool after_number = before > 0 && number_words.count(words[before - 1].key) &&
                                                  words[before].b - words[before - 1].e <= 3 &&
                                                  same_line(t, words[before - 1].e, words[before].b);
                        std::size_t q = words[k + best - 1].e;
                        while (q < t.size() && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                        bool before_number = false;
                        if (q < t.size() && t.cps[q] == ':')
                        {
                            ++q;
                            while (q < t.size() && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                            before_number = q < t.size() && is_ascii_digit(t.cps[q]);
                        }
                        if (after_number || before_number) best = 0;
                    }
                    if (best > 0)
                    {
                        std::size_t first = k, last = k + best - 1;
                        /* The branch of a national body: "Agence France Travail d'Arras", "the
                           Dallas Jobcentre Plus", "the East Stanley Jobcentre Plus". */
                        if (first > 0 && joined(words[first - 1].e, w.b) && words[first - 1].shape != 0 &&
                            branch_heads.count(words[first - 1].key))
                            --first;
                        else if (english_branch_words.count(words[last].key))
                        {
                            // In the case of the name: "OXFORD JOBCENTRE PLUS".
                            const int shape = w.shape;
                            std::size_t b = first, taken_words = 0;
                            while (b > 0 && taken_words < 3 && words[b - 1].shape == shape &&
                                   words[b - 1].e + 1 == words[b].b && t.cps[words[b - 1].e] == ' ' &&
                                   !person_titles().count(words[b - 1].key))
                            {
                                const std::string& key = words[b - 1].key;
                                const bool compass = compass_words.count(key) && taken_words > 0;
                                if (!compass && common_words().count(key)) break;
                                --b;
                                ++taken_words;
                            }
                            first = b;
                        }
                        std::size_t place_end = last;
                        if (place_after(last, place_end)) last = place_end;
                        // A name of the lexicon, matched whole: the recognizers keep to its boundaries.
                        const std::size_t count = out.size();
                        add(out, words[first].b, words[last].e, pii_type::organization, 0.9, 10);
                        if (out.size() > count) out.back().exact = true;
                        k = last;
                        continue;
                    }
                }
                if (!heads.count(w.key)) continue;
                // "la ville de Rennes" is a place, "la Ville de Rennes" the town council.
                if ((w.key == "ville" || w.key == "commune") && w.shape == 0) continue;
                std::size_t begin = w.b, end = w.e;
                // "Sous-prefecture de Brest"; "PONT-COURT-EN-COGLES" is a place.
                if (is_hyphen(t.before(w.b)))
                {
                    if (!(k > 0 && words[k - 1].key == "sous" && words[k - 1].e + 1 == w.b)) continue;
                    begin = words[k - 1].b;
                }
                if (is_hyphen(t.at(w.e))) continue;
                int named = 0, proper = 0, named_before_place = 0, domains = 0;
                std::size_t first_named = 0;
                bool company = false, place = false;
                const bool public_adjective = k + 1 < words.size() && public_adjectives.count(words[k + 1].key) &&
                                              joined(w.e, words[k + 1].b);
                for (std::size_t j = k + 1; j < words.size() && j <= k + 14 && joined(words[j - 1].e, words[j].b); ++j)
                {
                    const word_token& x = words[j];
                    // "Brigade de gendarmerie de Montauban".
                    if (w.key == "brigade" && x.key == "gendarmerie")
                    {
                        ++domains;
                        end = x.e;
                        continue;
                    }
                    // Another head starts another body: "Ministere de l'Interieur et Prefecture de ...".
                    if (heads.count(x.key)) break;
                    // A legal form names a company: "Dupont Conseil SAS" is read by scan_organizations().
                    if (organization_forms().count(x.key) && x.shape == 2) { company = true; break; }
                    // A civility starts a person, a common word in capitals a sentence in capitals.
                    if (person_titles().count(x.key) || civility_words().count(x.key)) break;
                    if (x.shape == 2 && common_words().count(x.key) && !qualifiers.count(x.key) &&
                        !connectors.count(x.key))
                        break;
                    if (x.key == "de" || x.key == "du" || x.key == "d" || x.key == "des") place = true;
                    if (x.shape == 1 || x.shape == 2)
                    {
                        if (connectors.count(x.key)) continue;
                        if (!qualifiers.count(x.key) || !generic.count(x.key))
                        {
                            ++named;
                            if (!place) ++named_before_place;
                            if (named == 1) first_named = j;
                            if (!function_words.count(x.key) && !generic.count(x.key)) ++proper;
                        }
                        end = x.e;
                    }
                    else if (qualifiers.count(x.key))
                    {
                        if (!generic.count(x.key)) ++domains;
                        end = x.e;
                    }
                    else if (!connectors.count(x.key))
                    {
                        break;
                    }
                }
                bool body;
                if (company)
                    body = false;
                // "l'agence de Lyon", "AGENCE DE VILLEBOEUF": a branch named by its place only.
                else if (branch_heads.count(w.key) && named_before_place == 0 && domains == 0 &&
                         (w.shape == 0 || !public_adjective))
                    body = false;
                /* A council, a direction or a commission may be internal to an organisation
                   ("Direction financiere", "Conseil scientifique", "Commission des finances"):
                   it is a public body when named by a place or a name of its own, or by a public
                   adjective and a domain ("Direction generale des finances publiques"). */
                else if (internal_heads.count(w.key))
                    body = proper > 0 || (public_adjective && domains > 0 && w.shape != 0);
                else
                    body = named > 0 || (domains > 0 && w.shape != 0);
                // "par l'intermediaire de l'agence Nerixa Atlantique": the name follows a lower case branch.
                if (body && w.shape == 0 && branch_heads.count(w.key) && named_before_place > 0 && first_named > k)
                    begin = words[first_named].b;
                // "the embassy of Spain", "le consulat d'Etats-Unis": a country names no diplomatic post.
                static const std::unordered_set<std::string> diplomatic_heads = make_word_set(
                    "ambassade consulat embassy consulate");
                if (body && diplomatic_heads.count(w.key) && first_named > k &&
                    country_names().count(fold_key(cp_substr(t, text, words[first_named].b, end))))
                    body = false;
                if (body)
                    add(out, begin, end, pii_type::organization, 0.85, 10);
            }
        }

        /* English public bodies and institutions named by their place: "Manchester City
           Council", "Leeds Crown Court", "East Ellham Magistrates' Court", "Ipswich Police
           Department", "Town of Vancouver", and the saints' names of hospitals and schools:
           "St Mary's Hospital", "St Paul's School". One capitalized word at least before the
           head is neither a common word nor a word of the head: "the Employment Tribunal"
           stays readable, and so do "19 Clover Court" and the hyphenated words of a place. */
        void scan_english_bodies(const utf8_text& t, const std::vector<word_token>& words,
                                 std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!options_.public_bodies || !enabled(pii_type::organization)) return;
            static const std::unordered_set<std::string> finals = make_word_set(
                "council court police constabulary authority tribunal");
            static const std::unordered_set<std::string> inner = make_word_set(
                "city county borough district crown magistrates magistrates' high family coroner's coroners police "
                "fire housing health parish town metropolitan regional state employment immigration asylum works "
                "first tier upper lower tax property social appeal appeals planning licensing");
            static const std::unordered_set<std::string> compass = make_word_set(
                "north south east west little great upper lower new old");
            static const std::unordered_set<std::string> saint_heads = make_word_set(
                "hospital hospice school college academy church chapel cathedral infirmary surgery clinic "
                "university hall");
            static const std::unordered_set<std::string> local_heads = make_word_set("town city borough county village");
            const std::size_t n = words.size();
            auto spaced = [&](std::size_t a, std::size_t b) {
                return b == a + 1 && (t.cps[a] == ' ' || t.cps[a] == 0xA0);
            };
            for (std::size_t k = 0; k < n; ++k)
            {
                const word_token& w = words[k];
                // "St Mary's Hospital", "St Thomas' Hospital", "Saint Paul's Primary School".
                if ((w.key == "st" || w.key == "saint") && w.shape == 1 && k + 2 < n && words[k + 1].shape == 1 &&
                    words[k + 1].b - (t.at(w.e) == '.' ? w.e + 1 : w.e) == 1)
                {
                    std::size_t j = k + 2;
                    const std::size_t e1 = words[k + 1].e;
                    if (is_apostrophe(t.at(e1)) && words[j].key == "s" && words[j].b == e1 + 1) ++j;
                    else if (!(is_apostrophe(t.at(e1)) && words[k + 1].key.back() == 's')) j = n;
                    std::size_t last = n;
                    for (std::size_t q = j; q < n && q < j + 2 && words[q].shape == 1 && words[q].b > words[q - 1].e &&
                                            words[q].b - words[q - 1].e <= 2; ++q)
                        if (saint_heads.count(words[q].key)) last = q;
                    if (last < n)
                    {
                        add(out, w.b, words[last].e, pii_type::organization, 0.85, 10);
                        k = last;
                        continue;
                    }
                }
                // "Town of Vancouver", "City of Chicago".
                if (local_heads.count(w.key) && w.shape == 1 && k + 2 < n && words[k + 1].key == "of" &&
                    spaced(w.e, words[k + 1].b) && spaced(words[k + 1].e, words[k + 2].b) &&
                    (words[k + 2].shape == 1 || words[k + 2].shape == 2) &&
                    !common_words().count(words[k + 2].key))
                {
                    std::size_t last = k + 2;
                    while (last + 1 < n && last < k + 4 && words[last + 1].shape == 1 &&
                           spaced(words[last].e, words[last + 1].b) && !common_words().count(words[last + 1].key))
                        ++last;
                    add(out, w.b, words[last].e, pii_type::organization, 0.85, 10);
                    k = last;
                    continue;
                }
                if (k == 0 || !finals.count(w.key) || w.shape == 0) continue;
                // "PONT-COURT-EN-COGLES" is a place.
                if (is_hyphen(t.before(w.b)) || is_hyphen(t.at(w.e))) continue;
                std::size_t first = k, named = 0;
                while (first > 0 && k - first < 5)
                {
                    const word_token& x = words[first - 1];
                    const std::size_t gap = words[first].b - x.e;
                    if (!(x.shape == 1 || x.shape == 2) || gap == 0 || gap > 2 || !same_line(t, x.e, words[first].b))
                        break;
                    bool spaces = true;
                    for (std::size_t g = x.e; g < words[first].b; ++g)
                        if (!(t.cps[g] == ' ' || is_apostrophe(t.cps[g]))) spaces = false;
                    if (!spaces || person_titles().count(x.key) || is_hyphen(t.before(x.b))) break;
                    if (!inner.count(x.key))
                    {
                        // "East Ellham": a compass word opens the name of a place.
                        if (compass.count(x.key) && named > 0 && x.shape == 1) { --first; break; }
                        if (common_words().count(x.key)) break;
                        ++named;
                    }
                    --first;
                }
                if (named == 0) continue;
                // "19 Clover Court" is a street.
                std::size_t p = words[first].b;
                while (p > 0 && (t.cps[p - 1] == ' ' || t.cps[p - 1] == 0xA0)) --p;
                if (p > 0 && is_ascii_digit(t.cps[p - 1])) continue;
                // "Police Department", "Police Station".
                static const std::unordered_set<std::string> tails = make_word_set("department station service force");
                std::size_t last = k;
                if (k + 1 < n && tails.count(words[k + 1].key) && words[k + 1].shape != 0 &&
                    words[k + 1].b == w.e + 1 && t.cps[w.e] == ' ')
                    last = k + 1;
                add(out, words[first].b, words[last].e, pii_type::organization, 0.85, 10);
                k = last;
            }
        }

        /* Organisations known by their own name: "Airbus", "Johnson & Johnson", "AG2R La
           Mondiale", "La Banque postale". The name starts with a capital; an article that opens
           it is followed by a capitalized word ("The Economist", while "the economist said" and
           "L'équipe" stay readable), and at the start of a sentence, where any word may take a
           capital, the words of a longer name are capitalized, its connectors excepted. A name
           read inside a longer one is left to the other scanners: a first name or a civility
           before it makes a person ("Mark Bloomberg", "M. Dassault"), and another capitalized
           word after it a product or a longer name ("Microsoft Teams", "Airbus A320"), unless
           that word is a legal form, a trade or a country, which then belongs to the name
           ("Airbus Group", "AXA France", "Equinor Finance SAS"). */
        void scan_known_organizations(const std::string& text, const utf8_text& t,
                                      const std::vector<word_token>& words, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::organization)) return;
            const auto& names = known_organizations();
            // The first word of each name, to skip the other words at once, and the longest name.
            struct name_index
            {
                std::unordered_set<std::string> first_words;
                std::size_t max_words = 1;
            };
            static const name_index index = [&names]() {
                name_index x;
                for (const std::string& name : names)
                {
                    const utf8_text nt(name, true);
                    const std::vector<word_token> nw = tokenize_words(nt, name);
                    if (nw.empty()) continue;
                    x.first_words.insert(nw.front().key);
                    x.max_words = std::max(x.max_words, nw.size());
                }
                return x;
            }();
            static const std::unordered_set<std::string> articles = make_word_set(
                "the le la les l el il lo der die das de o");
            static const std::unordered_set<std::string> connectors = make_word_set(
                "the le la les l el il lo der die das de du des d o of for and et pour sans contre a au aux en "
                "y e di del della dei do da dos van von zu");
            const std::size_t n = words.size();
            // Between two words of a name: spaces, hyphens, apostrophes, '&', '+' and digits ("G4S").
            auto name_gap = [&](std::size_t a, std::size_t b) {
                if (b <= a || b - a > 3 || !same_line(t, a, b)) return false;
                for (std::size_t g = a; g < b; ++g)
                {
                    const char32_t c = t.cps[g];
                    if (!(c == ' ' || c == 0xA0 || is_hyphen(c) || is_apostrophe(c) || c == '&' || c == '+' ||
                          is_ascii_digit(c)))
                        return false;
                }
                return true;
            };
            // The number that ends a name: "TF1", "N26", "France 24", "Racing 92".
            auto number_after = [&](std::size_t e) {
                std::size_t q = e;
                if (q < t.size() && t.cps[q] == ' ') ++q;
                std::size_t d = q;
                while (d < t.size() && is_ascii_digit(t.cps[d])) ++d;
                return d > q ? d : e;
            };
            // Not inside a longer token: an e-mail address, a domain, a path or a hyphenated word.
            auto open_before = [&](std::size_t b) {
                if (b == 0) return true;
                const char32_t c = t.cps[b - 1];
                return !(is_word_char(c) || c == '@' || c == '.' || c == '/' || c == '\\' || c == '_' || c == '#' ||
                         is_hyphen(c));
            };
            auto open_after = [&](std::size_t e) {
                if (e >= t.size()) return true;
                const char32_t c = t.cps[e];
                if (c == '.' || is_hyphen(c)) return !(e + 1 < t.size() && is_word_char(t.cps[e + 1]));
                return !(is_word_char(c) || c == '@' || c == '/' || c == '\\' || c == '_');
            };
            // "bioMerieux" opens with a small letter.
            auto capitalized = [&](const word_token& x) {
                if (x.shape == 1 || x.shape == 2 || x.shape == 3) return true;
                if (x.shape != 0) return false;
                for (std::size_t q = x.b; q < x.e; ++q)
                    if (is_upper(t.cps[q])) return true;
                return false;
            };
            auto spaced = [&](std::size_t a, std::size_t b) {
                return b == a + 1 && (t.cps[a] == ' ' || t.cps[a] == 0xA0);
            };
            for (std::size_t k = 0; k < n; ++k)
            {
                const word_token& w = words[k];
                const bool lower_line = !capitalized(w) && w.e - w.b >= 3 && lower_case_line(t, w.b);
                if (!index.first_words.count(w.key) || !(capitalized(w) || lower_line) || !open_before(w.b)) continue;
                std::size_t best = 0, best_end = 0;
                for (std::size_t len = 1; len <= index.max_words && k + len <= n; ++len)
                {
                    const std::size_t j = k + len - 1;
                    if (len > 1 && !name_gap(words[j - 1].e, words[j].b)) break;
                    const std::size_t e = words[j].e, d = number_after(e);
                    if (d > e && names.count(fold_key(cp_substr(t, text, w.b, d))))
                    {
                        best = len;
                        best_end = d;
                    }
                    else if (names.count(fold_key(cp_substr(t, text, w.b, e))))
                    {
                        best = len;
                        best_end = e;
                    }
                }
                if (best == 0 || !open_after(best_end)) continue;
                std::size_t last = k + best - 1, end = best_end;
                if (articles.count(w.key))
                {
                    if (best < 2 || !capitalized(words[k + 1])) continue;
                    // In a line of capitals, "L'EQUIPE" is the team.
                    auto capitals = [&](std::size_t q) { return words[q].shape == 2 && words[q].e - words[q].b > 1; };
                    const bool capitals_before = k > 0 && capitals(k - 1) && same_line(t, words[k - 1].e, w.b);
                    const bool capitals_after = last + 1 < n && capitals(last + 1) && same_line(t, end, words[last + 1].b);
                    if (w.shape == 2 && (capitals_before || capitals_after)) continue;
                }
                // "15 rue de la Poste": the name of a street.
                bool street = false;
                for (std::size_t q = k; q > 0 && k - q < 3 && !street; --q)
                    if (street_types().count(words[q - 1].key) && same_line(t, words[q - 1].e, w.b)) street = true;
                if (street) continue;
                if (best > 1 && !lower_line && at_sentence_start(t, w.b))
                {
                    bool all_capitalized = true;
                    for (std::size_t q = k + 1; q <= last; ++q)
                        if (!capitalized(words[q]) && !connectors.count(words[q].key)) all_capitalized = false;
                    if (!all_capitalized) continue;
                }
                std::size_t first = k;
                if (k > 0)
                {
                    const word_token& before = words[k - 1];
                    const bool close = spaced(before.e, w.b) ||
                                       (w.b == before.e + 2 && t.cps[before.e] == '.' && t.cps[before.e + 1] == ' ');
                    if (close && before.shape != 0 &&
                        (first_names().count(before.key) || person_titles().count(before.key) ||
                         civility_words().count(before.key)))
                        continue;
                    // "Groupe Equinor", "SAS Equinor": a head noun or a legal form opens the name.
                    if (spaced(before.e, w.b) && (before.shape == 1 || before.shape == 2) &&
                        (organization_heads().count(before.key) ||
                         (before.shape == 2 && organization_forms().count(before.key))))
                        first = k - 1;
                }
                bool longer = false;
                for (std::size_t q = last + 1; q < n && q <= last + 3; ++q)
                {
                    const word_token& x = words[q];
                    if (!spaced(end, x.b) || !capitalized(x)) break;
                    if (organization_forms().count(x.key) || organization_sectors().count(x.key) ||
                        country_names().count(x.key))
                    {
                        last = q;
                        end = x.e;
                        continue;
                    }
                    // A line in capitals: "SALARIE D'AIRBUS DEPUIS 2010".
                    if (x.shape == 2 && w.shape == 2 && (common_words().count(x.key) || function_words().count(x.key)))
                        break;
                    // "Turkcell Nous vous remercions": a sentence that lost its period ("Airbus A320" is a product).
                    if (x.e - x.b >= 2 && function_words().count(x.key)) break;
                    longer = true;
                    break;
                }
                if (longer) continue;
                // Below the addresses and the places that hold the same words.
                const std::size_t count = out.size();
                add(out, words[first].b, end, pii_type::organization, 0.9, 5);
                if (out.size() > count) out.back().exact = true;
                k = last;
            }
        }

        /* Nicknames, stage names and pen names announced by a cue: "Jean Dupont, dit
           « Jojo »", "Marie Martin, surnommée Mimi", "John Smith a.k.a. 'Smithy'", "sous le
           pseudonyme de Vernon Sullivan", "her stage name, Lady Gaga,". After "dit", "alias",
           "aka" and "known as", the cue follows the name of a person; "dit" and "dite" also
           follow a comma, unless the value ends the clause ("Jean Dupont dit Martin,"), so
           that reported speech ("Il a dit « bonjour »", "Jean Dupont dit « Je viens »") stays
           readable. The value is quoted, with one to four words, or one to three capitalized
           words; it holds a capital, and a word that is neither common nor a function word
           when the cue is "dit", "alias", "aka" or "known as". */
        void scan_aliases(const utf8_text& t, const std::vector<word_token>& words, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::person)) return;
            enum class cue_kind { said, alias, strong };
            struct cue
            {
                std::vector<std::string> keys;
                cue_kind kind;
                bool complement;   // "nom de scene de Stromae", "le pseudonyme d'Emile Ajar"
            };
            static const std::vector<cue> cues = {
                {{"dit"}, cue_kind::said, false}, {{"dite"}, cue_kind::said, false},
                {{"dits"}, cue_kind::said, false}, {{"dites"}, cue_kind::said, false},
                {{"alias"}, cue_kind::alias, false}, {{"aka"}, cue_kind::alias, false},
                {{"known", "as"}, cue_kind::alias, false},
                {{"connu", "sous", "le", "nom", "de"}, cue_kind::alias, false},
                {{"connue", "sous", "le", "nom", "de"}, cue_kind::alias, false},
                {{"surnomme"}, cue_kind::strong, false}, {{"surnommee"}, cue_kind::strong, false},
                {{"surnommes"}, cue_kind::strong, false}, {{"surnommees"}, cue_kind::strong, false},
                {{"surnom"}, cue_kind::strong, false}, {{"sobriquet"}, cue_kind::strong, false},
                {{"nicknamed"}, cue_kind::strong, false}, {{"nickname"}, cue_kind::strong, false},
                {{"pseudonyme"}, cue_kind::strong, true}, {{"pseudonym"}, cue_kind::strong, false},
                {{"nom", "de", "scene"}, cue_kind::strong, true}, {{"nom", "d", "artiste"}, cue_kind::strong, true},
                {{"nom", "de", "plume"}, cue_kind::strong, true}, {{"nom", "de", "guerre"}, cue_kind::strong, true},
                {{"stage", "name"}, cue_kind::strong, false}, {{"pen", "name"}, cue_kind::strong, false},
                {{"ring", "name"}, cue_kind::strong, false}, {{"artist", "name"}, cue_kind::strong, false},
                {{"professional", "name"}, cue_kind::strong, false}
            };
            // Words that answer or exclaim rather than name: "Jean Dupont dit Non."
            static const std::unordered_set<std::string> interjections = make_word_set(
                "non oui si ok okay stop attention adieu voila yes no well");
            static const std::unordered_set<std::string> modifiers = make_word_set("also better best plus mieux aussi");
            const std::size_t n = words.size(), size = t.size();
            auto space = [&](std::size_t i) { return i < size && (is_space(t.cps[i]) && !is_line_break(t.cps[i])); };
            auto common = [&](const word_token& x) {
                return common_words().count(x.key) || function_words().count(x.key) ||
                       greeting_words().count(x.key) || interjections.count(x.key);
            };
            auto capital = [&](const word_token& x) { return x.shape == 1 || x.shape == 2 || x.shape == 3; };
            /* The person the cue follows, ending at word p: two capitalized words or more, or a
               first name alone before a comma. */
            auto person_before = [&](std::size_t p, bool comma) {
                if (!capital(words[p])) return false;
                std::size_t run = 1, q = p;
                while (q > 0 && run < 4 && capital(words[q - 1]) &&
                       privacy_impl::name_gap(t, words[q - 1].e, words[q].b, words[q - 1].e - words[q - 1].b == 1))
                {
                    --q;
                    ++run;
                }
                if (run >= 2)
                {
                    int proper = 0;
                    for (std::size_t r = q; r <= p; ++r)
                        if (!common(words[r]) && !person_titles().count(words[r].key)) ++proper;
                    return proper >= 2 || first_names().count(words[q].key) != 0;
                }
                return comma && first_names().count(words[p].key) && !ambiguous_first_names().count(words[p].key);
            };
            for (std::size_t k = 0; k < n; ++k)
            {
                // The cue: one of the word sequences above, or "a.k.a.".
                const cue* found = nullptr;
                std::size_t cue_last = k, cue_end = words[k].e;
                static const cue dotted = {{"a", "k", "a"}, cue_kind::alias, false};
                if (words[k].key == "a" && k + 2 < n && words[k + 1].key == "k" && words[k + 2].key == "a" &&
                    t.at(words[k].e) == '.' && words[k + 1].b == words[k].e + 1 && t.at(words[k + 1].e) == '.' &&
                    words[k + 2].b == words[k + 1].e + 1)
                {
                    found = &dotted;
                    cue_last = k + 2;
                    cue_end = words[k + 2].e + (t.at(words[k + 2].e) == '.' ? 1 : 0);
                }
                else
                {
                    for (const cue& c : cues)
                    {
                        if (c.keys.front() != words[k].key || k + c.keys.size() > n) continue;
                        bool match = true;
                        for (std::size_t q = 1; q < c.keys.size() && match; ++q)
                        {
                            const std::size_t a = words[k + q - 1].e, b = words[k + q].b;
                            match = words[k + q].key == c.keys[q] && b > a && b - a <= 2 && same_line(t, a, b);
                        }
                        if (!match) continue;
                        found = &c;
                        cue_last = k + c.keys.size() - 1;
                        cue_end = words[cue_last].e;
                        break;
                    }
                }
                if (!found) continue;
                // "dit-il": an inverted verb.
                if (is_hyphen(t.at(cue_end))) continue;
                // What precedes the cue: a comma, and the person for the weaker cues.
                bool comma = false, person = false;
                {
                    std::size_t q = k;
                    while (q > 0 && modifiers.count(words[q - 1].key) && words[q - 1].e + 1 == words[q].b &&
                           t.cps[words[q - 1].e] == ' ')
                        --q;
                    std::size_t b = words[q].b;
                    while (b > 0 && space(b - 1)) --b;
                    if (b > 0 && t.cps[b - 1] == ',') { comma = true; --b; }
                    while (b > 0 && space(b - 1)) --b;
                    if (q > 0 && words[q - 1].e == b && b < words[q].b)
                        person = person_before(q - 1, comma);
                }
                if (found->kind != cue_kind::strong && !person) continue;
                // The value, after spaces, a colon or a comma, and "de" for the complements.
                std::size_t p = cue_end;
                while (space(p)) ++p;
                if (found->kind != cue_kind::said && p < size && (t.cps[p] == ':' || t.cps[p] == ','))
                {
                    ++p;
                    while (space(p)) ++p;
                }
                std::size_t next = cue_last + 1;
                while (next < n && words[next].b < p) ++next;
                if (found->complement && next < n && words[next].b == p &&
                    (words[next].key == "de" || words[next].key == "d") && words[next].shape == 0)
                {
                    p = words[next].e;
                    if (is_apostrophe(t.at(p))) ++p;
                    else while (space(p)) ++p;
                    ++next;
                }
                if (p >= size) continue;
                std::size_t vb = 0, ve = 0, first_word = next, last_word = next;
                bool quoted = false;
                const char32_t open = t.cps[p];
                if (open == 0xAB || open == '"' || open == 0x201C || open == 0x201E || open == 0x2039)
                {
                    // French guillemets, straight, English and German double quotes, single guillemets.
                    auto closes = [open](char32_t c) {
                        if (open == 0xAB) return c == 0xBB;
                        if (open == 0x2039) return c == 0x203A;
                        if (open == '"') return c == '"';
                        if (open == 0x201E) return c == 0x201C || c == 0x201D;
                        return c == 0x201D;
                    };
                    std::size_t q = p + 1;
                    while (q < size && q < p + 60 && !closes(t.cps[q]) && !is_line_break(t.cps[q])) ++q;
                    if (q >= size || !closes(t.cps[q])) continue;
                    vb = p + 1;
                    ve = q;
                    while (vb < ve && is_space(t.cps[vb])) ++vb;
                    while (ve > vb && is_space(t.cps[ve - 1])) --ve;
                    quoted = true;
                    while (first_word < n && words[first_word].b < vb) ++first_word;
                    last_word = first_word;
                    while (last_word < n && words[last_word].e <= ve) ++last_word;
                    if (last_word == first_word || last_word - first_word > 4) continue;
                    --last_word;
                }
                else
                {
                    // Up to three capitalized words; a capitalized article opens a nickname ("Le Pacha").
                    if (next >= n || words[next].b != p || !capital(words[next])) continue;
                    last_word = next;
                    while (last_word + 1 < n && last_word < next + 2 && capital(words[last_word + 1]) &&
                           privacy_impl::name_gap(t, words[last_word].e, words[last_word + 1].b, false))
                        ++last_word;
                    if (is_word_char(t.at(words[last_word].e))) continue;
                    vb = words[next].b;
                    ve = words[last_word].e;
                    first_word = next;
                }
                // A quoted nickname may be made of common words ("« le Grand »"), a bare one not.
                bool has_capital = false, has_proper = false;
                for (std::size_t q = first_word; q <= last_word; ++q)
                {
                    if (capital(words[q])) has_capital = true;
                    if (!common(words[q])) has_proper = true;
                }
                if (!has_capital || (!quoted && !has_proper)) continue;
                if (found->kind == cue_kind::said && !comma)
                {
                    // "Jean Dupont dit Martin," the value ends the clause.
                    std::size_t q = quoted ? ve + 1 : ve;
                    while (q < size && (t.cps[q] == ' ' || t.cps[q] == 0xA0 || t.cps[q] == 0xBB || t.cps[q] == '"' ||
                                        t.cps[q] == 0x201D))
                        ++q;
                    const char32_t c = t.at(q);
                    if (quoted || !(q >= size || c == ',' || c == '.' || c == ';' || c == ')' || is_line_break(c)))
                        continue;
                }
                if (ve <= vb) continue;
                add(out, vb, ve, pii_type::person, 0.9, 10, true);
                k = last_word;
            }
        }

    // ------------------------------------------------------------------------------------

        static bool boundary_before(const utf8_text& t, std::size_t i)
        {
            return i == 0 || !privacy_impl::is_word_char(t.cps[i - 1]);
        }

        static bool boundary_after(const utf8_text& t, std::size_t j)
        {
            return j >= t.size() || !privacy_impl::is_word_char(t.cps[j]);
        }

        void scan_emails(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            // Internationalized addresses have letters of any script before and after the '@'.
            auto local_char = [](char32_t c) {
                return is_ascii_alnum_cp(c) || is_letter(c) || c == '.' || c == '_' || c == '%' || c == '+' ||
                       c == '-' || c == '\'';
            };
            auto domain_char = [](char32_t c) { return is_ascii_alnum_cp(c) || is_letter(c) || c == '.' || c == '-'; };
            const std::size_t n = t.size();
            for (std::size_t i = 1; i + 1 < n; ++i)
            {
                if (t.cps[i] != '@') continue;
                std::size_t l = i;
                while (l > 0 && local_char(t.cps[l - 1])) --l;
                while (l < i && (t.cps[l] == '.' || t.cps[l] == '\'')) ++l;
                std::size_t r = i + 1;
                while (r < n && domain_char(t.cps[r])) ++r;
                while (r > i + 1 && (t.cps[r - 1] == '.' || t.cps[r - 1] == '-')) --r;
                if (l == i || r <= i + 3 || t.cps[i - 1] == '.') continue;
                bool dots = false, ok = true;
                std::size_t label_start = i + 1, tld_letters = 0;
                for (std::size_t k = i + 1; k <= r && ok; ++k)
                {
                    if (k == r || t.cps[k] == '.')
                    {
                        if (k == label_start) ok = false;
                        if (k < r) { dots = true; label_start = k + 1; }
                    }
                }
                if (!ok || !dots) continue;
                for (std::size_t k = label_start; k < r; ++k)
                    if ((t.cps[k] >= 'a' && t.cps[k] <= 'z') || (t.cps[k] >= 'A' && t.cps[k] <= 'Z')) ++tld_letters;
                if (tld_letters < 2 || tld_letters != r - label_start) continue;
                add(out, l, r, pii_type::email, 0.99);
            }
        }

        /* Web addresses. A profile page (a social network host followed by an account), an
           address holding an e-mail address, credentials or an IP address, and an address
           with a word of a known name or of a known e-mail address are personal data. The
           other addresses are added to protect: nothing inside them is replaced, so a link
           to a public page keeps working. */
        void scan_urls(const std::string& text, const utf8_text& t, std::vector<candidate>& out,
                       std::vector<std::pair<std::size_t, std::size_t>>& protect) const
        {
            using namespace privacy_impl;
            const std::size_t n = t.size();
            auto starts_ci = [&](std::size_t i, const char* s) {
                for (std::size_t k = 0; s[k]; ++k)
                    if (i + k >= n || to_lower(t.cps[i + k]) != static_cast<char32_t>(s[k])) return false;
                return true;
            };
            struct profile_host { const char* host; const char* path; };
            static const profile_host profile_hosts[] = {
                {"linkedin.com", "/in/"}, {"linkedin.com", "/pub/"}, {"facebook.com", "/"}, {"fb.com", "/"},
                {"twitter.com", "/"}, {"x.com", "/"}, {"instagram.com", "/"}, {"github.com", "/"},
                {"gitlab.com", "/"}, {"viadeo.com", "/"}, {"tiktok.com", "/@"}, {"youtube.com", "/@"},
                {"medium.com", "/@"}, {"threads.net", "/@"}, {"bsky.app", "/profile/"} };
            // Email candidates are found first: an address holding one is personal data.
            std::vector<std::size_t> emails;
            for (const candidate& x : out)
                if (x.type == pii_type::email) emails.push_back(x.b);
            std::sort(emails.begin(), emails.end());
            for (std::size_t i = 0; i < n; ++i)
            {
                if (!boundary_before(t, i)) continue;
                if (!(starts_ci(i, "http://") || starts_ci(i, "https://") || starts_ci(i, "www."))) continue;
                std::size_t j = i;
                int parens = 0;
                while (j < n)
                {
                    const char32_t c = t.cps[j];
                    if (is_space(c) || c == '<' || c == '>' || c == '"' || c == '`' || c == '|' || c == '{' ||
                        c == '}' || c == '[' || c == ']' || c == 0xAB || c == 0xBB)
                        break;
                    if (c == '(') ++parens;
                    if (c == ')') { if (parens == 0) break; --parens; }
                    ++j;
                }
                while (j > i &&
                       (t.cps[j - 1] == '.' || t.cps[j - 1] == ',' || t.cps[j - 1] == ';' || t.cps[j - 1] == ':' ||
                        t.cps[j - 1] == '!' || t.cps[j - 1] == '?' || t.cps[j - 1] == '\''))
                    --j;
                const std::string url = lowercase(cp_substr(t, text, i, j));
                // The host, after the scheme and any credentials, without its port.
                const std::size_t scheme = url.find("://");
                const std::size_t host_begin = scheme == std::string::npos ? 0 : scheme + 3;
                const std::size_t authority_end = std::min(url.find_first_of("/?#", host_begin), url.size());
                std::string host = url.substr(host_begin, authority_end - host_begin);
                const bool credentials = host.find('@') != std::string::npos;
                if (credentials) host = host.substr(host.rfind('@') + 1);
                if (!host.empty() && host[0] == '[') host = host.substr(0, host.find(']') + 1);
                else if (host.find(':') != std::string::npos) host = host.substr(0, host.find(':'));
                const std::string path = url.substr(authority_end);
                bool literal = !host.empty() && host[0] == '[';
                if (!literal && !host.empty())
                {
                    literal = true;
                    for (char ch : host) if (!((ch >= '0' && ch <= '9') || ch == '.')) literal = false;
                }
                bool flag = !options_.profile_urls_only || credentials || literal;
                for (const profile_host& h : profile_hosts)
                {
                    const std::string name = h.host;
                    const bool same_host = host == name ||
                        (host.size() > name.size() && host.compare(host.size() - name.size(), name.size(), name) == 0 &&
                         host[host.size() - name.size() - 1] == '.');
                    const std::size_t len = std::strlen(h.path);
                    if (same_host && path.size() > len && path.compare(0, len, h.path) == 0) flag = true;
                }
                const auto e = std::lower_bound(emails.begin(), emails.end(), i);
                if (e != emails.end() && *e < j) flag = true;
                if (!flag)
                {
                    std::string frag;
                    for (std::size_t k = 0; k <= url.size() && !flag; ++k)
                    {
                        const char ch = k < url.size() ? url[k] : '/';
                        if (ch >= 'a' && ch <= 'z') { frag.push_back(ch); continue; }
                        if (frag.size() >= 3 &&
                            (doc_name_words_.count(frag) || user_name_words_.count(frag) ||
                             email_fragments_.count(frag)))
                            flag = true;
                        frag.clear();
                    }
                }
                if (flag && enabled(pii_type::url))
                    add(out, i, j, pii_type::url, 0.9);
                else if (!flag)
                    protect.push_back(std::make_pair(i, j));
                i = j;
            }
        }

        /* Reads four decimal numbers from 0 to 255 joined by periods at i, without leading
           zeros, and sets end after the last one. */
        static bool read_ipv4(const utf8_text& t, std::size_t i, std::size_t& end)
        {
            using namespace privacy_impl;
            const std::size_t n = t.size();
            std::size_t j = i;
            for (int part = 0; part < 4; ++part)
            {
                if (part > 0)
                {
                    if (j >= n || t.cps[j] != '.') return false;
                    ++j;
                }
                const std::size_t s = j;
                int v = 0;
                while (j < n && is_ascii_digit(t.cps[j]) && j - s < 3)
                    v = v * 10 + static_cast<int>(t.cps[j++] - '0');
                if (j == s || (j < n && is_ascii_digit(t.cps[j])) || v > 255 || (j - s > 1 && t.cps[s] == '0'))
                    return false;
            }
            end = j;
            return true;
        }

        void scan_ip(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            const std::size_t n = t.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const char32_t c = t.cps[i];
                const char32_t prev = t.before(i);
                std::size_t j = i;
                if (enabled(pii_type::ipv4) && is_ascii_digit(c) && !is_word_char(prev) && prev != '.' &&
                    read_ipv4(t, i, j) && !is_word_char(t.at(j)) && !(t.at(j) == '.' && is_ascii_digit(t.at(j + 1))))
                {
                    add(out, i, j, pii_type::ipv4, 0.95);
                    i = j;
                    continue;
                }
                if (enabled(pii_type::ipv6) && (is_hex_cp(c) || c == ':') && !is_hex_cp(prev) && prev != ':' &&
                    !is_word_char(prev))
                {
                    j = i;
                    while (j < n && (is_hex_cp(t.cps[j]) || t.cps[j] == ':')) ++j;
                    // An IPv4 address may end the address: "::ffff:192.0.2.10".
                    std::size_t last = j, stop = j;
                    while (last > i && t.cps[last - 1] != ':') --last;
                    const bool v4 = t.at(j) == '.' && last > i && last < j && read_ipv4(t, last, stop);
                    if (!v4) stop = j;
                    std::string s;
                    for (std::size_t k = i; k < (v4 ? last : j); ++k) s.push_back(static_cast<char>(t.cps[k]));
                    if (v4) s += "0:0";
                    const std::size_t colons = static_cast<std::size_t>(std::count(s.begin(), s.end(), ':'));
                    std::array<unsigned char,16> a;
                    if (colons >= 2 && !is_word_char(t.at(stop)) && !(t.at(stop) == '.' && is_ascii_digit(t.at(stop + 1))) &&
                        (colons == 7 || s.find("::") != std::string::npos) && s.size() >= 3 && parse_ipv6(s, a))
                    {
                        add(out, i, stop, pii_type::ipv6, 0.9);
                        i = stop;
                    }
                }
            }
        }

        void scan_mac(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::mac_address)) return;
            const std::size_t n = t.size();
            for (std::size_t i = 0; i + 13 < n + 1; ++i)
            {
                if (!is_hex_cp(t.cps[i]) || !boundary_before(t, i)) continue;
                // aa:bb:cc:dd:ee:ff or aa-bb-...
                if (i + 17 <= n)
                {
                    const char32_t sep = t.cps[i + 2];
                    bool ok = (sep == ':' || sep == '-');
                    for (int g = 0; g < 6 && ok; ++g)
                    {
                        const std::size_t p = i + static_cast<std::size_t>(3 * g);
                        ok = is_hex_cp(t.cps[p]) && is_hex_cp(t.cps[p + 1]) && (g == 5 || t.cps[p + 2] == sep);
                    }
                    if (ok && boundary_after(t, i + 17))
                    {
                        add(out, i, i + 17, pii_type::mac_address, 0.95);
                        i += 16;
                        continue;
                    }
                }
                // aabb.ccdd.eeff
                if (i + 14 <= n)
                {
                    bool ok = true;
                    for (int g = 0; g < 3 && ok; ++g)
                    {
                        const std::size_t p = i + static_cast<std::size_t>(5 * g);
                        for (int k = 0; k < 4 && ok; ++k) ok = is_hex_cp(t.cps[p + static_cast<std::size_t>(k)]);
                        if (ok && g < 2) ok = t.cps[p + 4] == '.';
                    }
                    if (ok && boundary_after(t, i + 14))
                    {
                        add(out, i, i + 14, pii_type::mac_address, 0.9);
                        i += 13;
                    }
                }
            }
        }

        void scan_iban(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::iban)) return;
            static const std::map<std::string, std::size_t> lengths = {
                {"AD", 24}, {"AE", 23}, {"AL", 28}, {"AT", 20}, {"AZ", 28}, {"BA", 20}, {"BE", 16}, {"BG", 22},
                {"BH", 22}, {"BR", 29}, {"BY", 28}, {"CH", 21}, {"CR", 22}, {"CY", 28}, {"CZ", 24}, {"DE", 22},
                {"DK", 18}, {"DO", 28}, {"EE", 20}, {"EG", 29}, {"ES", 24}, {"FI", 18}, {"FO", 18}, {"FR", 27},
                {"GB", 22}, {"GE", 22}, {"GI", 23}, {"GL", 18}, {"GR", 27}, {"GT", 28}, {"HR", 21}, {"HU", 28},
                {"IE", 22}, {"IL", 23}, {"IQ", 23}, {"IS", 26}, {"IT", 27}, {"JO", 30}, {"KW", 30}, {"KZ", 20},
                {"LB", 28}, {"LI", 21}, {"LT", 20}, {"LU", 20}, {"LV", 21}, {"MA", 28}, {"MC", 27}, {"MD", 24},
                {"ME", 22}, {"MK", 19}, {"MR", 27}, {"MT", 31}, {"MU", 30}, {"NL", 18}, {"NO", 15}, {"PK", 24},
                {"PL", 28}, {"PS", 29}, {"PT", 25}, {"QA", 29}, {"RO", 24}, {"RS", 22}, {"SA", 24}, {"SC", 31},
                {"SE", 24}, {"SI", 19}, {"SK", 24}, {"SM", 27}, {"TN", 24}, {"TR", 26}, {"UA", 29}, {"VA", 22},
                {"VG", 24}, {"XK", 20} };
            const std::size_t n = t.size();
            for (std::size_t i = 0; i + 15 <= n; ++i)
            {
                const char32_t c0 = t.cps[i], c1 = t.cps[i + 1];
                if (!(c0 >= 'A' && c0 <= 'Z' && c1 >= 'A' && c1 <= 'Z' && is_ascii_digit(t.cps[i + 2]) &&
                      is_ascii_digit(t.cps[i + 3]) && boundary_before(t, i)))
                    continue;
                const std::string cc = { static_cast<char>(c0), static_cast<char>(c1) };
                const auto it = lengths.find(cc);
                if (it == lengths.end()) continue;
                std::string a;
                std::size_t j = i, end = i;
                while (j < n && a.size() < it->second)
                {
                    const char32_t c = t.cps[j];
                    if ((c >= 'A' && c <= 'Z') || is_ascii_digit(c))
                    {
                        a.push_back(static_cast<char>(c));
                        ++j;
                        end = j;
                    }
                    else if (c == ' ' && j + 1 < n &&
                             ((t.cps[j + 1] >= 'A' && t.cps[j + 1] <= 'Z') || is_ascii_digit(t.cps[j + 1])))
                        ++j;
                    else break;
                }
                if (a.size() == it->second && boundary_after(t, end) && iban_checksum_valid(a))
                {
                    add(out, i, end, pii_type::iban, 0.99);
                    i = end;
                }
            }
        }

        /* Collects a run of digits joined by single separators taken from seps. Returns the
           digits and sets end to the position after the last digit. */
        static std::string digit_run(const utf8_text& t, std::size_t i, const char* seps, std::size_t& end,
                                     std::size_t max_digits, char32_t* used_sep = nullptr)
        {
            std::string d;
            std::size_t j = i;
            char32_t sep_seen = 0;
            end = i;
            while (j < t.size() && d.size() < max_digits)
            {
                const char32_t c = t.cps[j];
                if (privacy_impl::is_ascii_digit(c)) { d.push_back(static_cast<char>(c)); ++j; end = j; continue; }
                bool is_sep = false;
                for (const char* s = seps; *s; ++s) if (c == static_cast<char32_t>(*s)) is_sep = true;
                if (is_sep && j + 1 < t.size() && privacy_impl::is_ascii_digit(t.cps[j + 1]) &&
                    (sep_seen == 0 || sep_seen == c))
                {
                    sep_seen = c;
                    ++j;
                    continue;
                }
                break;
            }
            if (used_sep) *used_sep = sep_seen;
            return d;
        }

        /* The groups of a run of digits joined by single separators taken from seps: for
           each group, the position after it and the number of digits up to it. digits
           receives the digits read. */
        static std::vector<std::pair<std::size_t, std::size_t>> digit_groups(
            const utf8_text& t, std::size_t i, const char* seps, std::size_t max_digits, std::string& digits)
        {
            std::vector<std::pair<std::size_t, std::size_t>> groups;
            digits.clear();
            std::size_t j = i;
            char32_t sep_seen = 0;
            while (j < t.size() && digits.size() < max_digits)
            {
                const char32_t c = t.cps[j];
                if (privacy_impl::is_ascii_digit(c))
                {
                    while (j < t.size() && privacy_impl::is_ascii_digit(t.cps[j]) && digits.size() < max_digits)
                        digits.push_back(static_cast<char>(t.cps[j++]));
                    groups.push_back(std::make_pair(j, digits.size()));
                    continue;
                }
                bool is_sep = false;
                for (const char* s = seps; *s; ++s) if (c == static_cast<char32_t>(*s)) is_sep = true;
                if (is_sep && j + 1 < t.size() && privacy_impl::is_ascii_digit(t.cps[j + 1]) &&
                    (sep_seen == 0 || sep_seen == c))
                {
                    sep_seen = c;
                    ++j;
                    continue;
                }
                break;
            }
            return groups;
        }

        bool context_before(const utf8_text& t, std::size_t i, const std::vector<std::string>& cues,
                            std::size_t window) const
        {
            return cue_before(t, i, cues, window);
        }

        void scan_digit_identifiers(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            const std::size_t n = t.size();
            static const std::vector<std::string> siret_cues = { "siren", "siret", "rcs", "immatricul*", "registre" };
            for (std::size_t i = 0; i < n; ++i)
            {
                const char32_t c = t.cps[i];
                const char32_t prev = t.before(i);
                const bool start_ok = !is_word_char(prev) && prev != '.' && prev != ',' && prev != '+' &&
                                      !(prev == '-' && is_ascii_digit(t.before(i - 1)));
                if (!start_ok || !(is_ascii_digit(c) || c == '+')) continue;

                // NIR: 1 85 03 75 123 456 78, Corsican departments 2A and 2B.
                if (enabled(pii_type::nir) && (c == '1' || c == '2' || c == '3' || c == '4' || c == '7' || c == '8'))
                {
                    std::string a;
                    std::size_t j = i, end = i;
                    while (j < n && a.size() < 15)
                    {
                        const char32_t ch = t.cps[j];
                        const bool corsican = (a.size() == 6 && a[5] == '2' && (ch == 'A' || ch == 'B'));
                        if (is_ascii_digit(ch) || corsican) { a.push_back(static_cast<char>(ch)); ++j; end = j; }
                        else if ((ch == ' ' || ch == '.' || ch == 0xA0) && j + 1 < n &&
                                 (is_ascii_digit(t.cps[j + 1]) ||
                                  (a.size() == 6 && (t.cps[j + 1] == 'A' || t.cps[j + 1] == 'B'))))
                            ++j;
                        else break;
                    }
                    if (a.size() == 15 && boundary_after(t, end))
                    {
                        const int month = (a[3] - '0') * 10 + (a[4] - '0');
                        bool month_ok =
                            (month >= 1 && month <= 12) || (month >= 20 && month <= 42) || (month >= 50 && month <= 99);
                        if (month_ok && nir_key_valid(a))
                        {
                            add(out, i, end, pii_type::nir, 0.97);
                            i = end;
                            continue;
                        }
                    }
                }

                // SIREN / SIRET announced by their context take precedence over cards.
                if (enabled(pii_type::siret) && is_ascii_digit(c))
                {
                    std::size_t end;
                    const std::string d = digit_run(t, i, " ", end, 14);
                    if ((d.size() == 9 || d.size() == 14) && boundary_after(t, end) && luhn_valid(d) &&
                        context_before(t, i, siret_cues, 48))
                    {
                        add(out, i, end, pii_type::siret, 0.95);
                        i = end;
                        continue;
                    }
                }

                /* Payment cards: Luhn, and the lengths and prefixes issuers use (13 digits for
                   Visa only, 15 for American Express). IMEI numbers pass Luhn too and are
                   told apart by their cue. */
                static const std::vector<std::string> device_cues = { "imei", "imsi", "iccid", "sim" };
                if (enabled(pii_type::payment_card) && c >= '2' && c <= '6' && !cue_before(t, i, device_cues, 24))
                {
                    /* The longest prefix ending a group that has the length of a card: an
                       expiry date or a code of four digits at most may follow the number
                       ("4970 ... 7893 12/27"). */
                    std::string all;
                    const auto groups = digit_groups(t, i, " -", 23, all);
                    bool taken_card = false;
                    for (std::size_t g = groups.size(); g-- > 0 && !taken_card;)
                    {
                        const std::size_t len = groups[g].second, end = groups[g].first;
                        if (all.size() - len > 4) continue;
                        const std::string d = all.substr(0, len);
                        const bool plausible = (len == 13 && d[0] == '4') || len == 16 ||
                            (len == 15 && (d.compare(0, 2, "34") == 0 || d.compare(0, 2, "37") == 0)) ||
                            (len == 19 && d[0] != '3') ||
                            (len == 14 &&
                             (d.compare(0, 2, "36") == 0 || d.compare(0, 2, "38") == 0 || d.compare(0, 2, "30") == 0));
                        if (plausible && boundary_after(t, end) && luhn_valid(d) &&
                            d.find_first_not_of(d[0]) != std::string::npos)
                        {
                            add(out, i, end, pii_type::payment_card, 0.95);
                            i = end;
                            taken_card = true;
                        }
                    }
                    if (taken_card) continue;
                }

                // SIRET without context: fourteen digits passing Luhn.
                if (enabled(pii_type::siret) && is_ascii_digit(c))
                {
                    std::size_t end;
                    const std::string d = digit_run(t, i, " ", end, 14);
                    if (d.size() == 14 && boundary_after(t, end) && luhn_valid(d))
                    {
                        add(out, i, end, pii_type::siret, 0.6);
                        i = end;
                        continue;
                    }
                }

                // Phone numbers.
                if (enabled(pii_type::phone))
                {
                    std::size_t end = i;
                    bool found = false;
                    if (c == '+')
                    {
                        // Groups of digits: for each, its end and the digits up to it.
                        std::vector<std::pair<std::size_t, std::size_t>> groups;
                        std::size_t digits = 0, j = i + 1;
                        while (j < n && digits <= 20)
                        {
                            const char32_t ch = t.cps[j];
                            if (is_ascii_digit(ch))
                            {
                                while (j < n && is_ascii_digit(t.cps[j])) { ++digits; ++j; }
                                groups.push_back(std::make_pair(j, digits));
                                continue;
                            }
                            if ((ch == ' ' || ch == '.' || ch == '-' || ch == 0xA0) && j + 1 < n &&
                                (is_ascii_digit(t.cps[j + 1]) || t.cps[j + 1] == '('))
                            {
                                ++j;
                                continue;
                            }
                            if (ch == '(' && j + 2 < n && is_ascii_digit(t.cps[j + 1]) && t.cps[j + 2] == ')')
                            {
                                ++digits;
                                j += 3;
                                groups.push_back(std::make_pair(j, digits));
                                continue;
                            }
                            break;
                        }
                        const std::size_t g = phone_prefix(groups, true);
                        if (g != std::string::npos && is_ascii_digit(t.at(i + 1)))
                        {
                            end = groups[g].first;
                            found = true;
                        }
                    }
                    else if (c == '0' && i + 1 < n)
                    {
                        if (t.cps[i + 1] == '0')
                        {
                            // The international prefix 00, on a whole run of 11 to 17 digits; no country code starts with 0.
                            const std::string d = digit_run(t, i, " .-", end, 18);
                            found = d.size() >= 11 && d.size() <= 17 && d[2] != '0';
                        }
                        else if (is_ascii_digit(t.cps[i + 1]))
                        {
                            char32_t sep = 0;
                            const std::string d = digit_run(t, i, " .-", end, 10, &sep);
                            if (d.size() == 10)
                            {
                                // French layout: 0X XX XX XX XX, or ten contiguous digits.
                                bool layout = true;
                                if (sep != 0)
                                {
                                    std::size_t digits = 0;
                                    for (std::size_t k = i; k < end; ++k)
                                    {
                                        if (is_ascii_digit(t.cps[k])) ++digits;
                                        else if (digits % 2 != 0) layout = false;
                                    }
                                }
                                found = layout;
                            }
                        }
                    }
                    if (found && boundary_after(t, end) && !(t.at(end) == '.' && is_ascii_digit(t.at(end + 1))))
                    {
                        // A layout that was checked outranks the longer runs read after a cue.
                        add(out, i, end, pii_type::phone, 0.9, 1);
                        i = end;
                        continue;
                    }
                }
            }
        }

        static bool nir_key_valid(const std::string& a)
        {
            std::string n = a.substr(0, 13);
            if (n[5] == '2' && (n[6] == 'A' || n[6] == 'B'))
            {
                const bool is_a = n[6] == 'A';
                n[5] = '1';
                n[6] = is_a ? '9' : '8';
            }
            long long r = 0;
            for (char ch : n)
            {
                if (ch < '0' || ch > '9') return false;
                r = (r * 10 + (ch - '0')) % 97;
            }
            const int key = (a[13] - '0') * 10 + (a[14] - '0');
            return key == 97 - static_cast<int>(r);
        }

        void scan_plates(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::license_plate)) return;
            const std::size_t n = t.size();
            auto up = [](char32_t c) { return c >= 'A' && c <= 'Z'; };
            for (std::size_t i = 0; i + 9 <= n; ++i)
            {
                if (!boundary_before(t, i)) continue;
                const char32_t s1 = t.cps[i + 2], s2 = t.cps[i + 6];
                if (up(t.cps[i]) && up(t.cps[i + 1]) && (s1 == '-' || s1 == ' ') && is_ascii_digit(t.cps[i + 3]) &&
                    is_ascii_digit(t.cps[i + 4]) && is_ascii_digit(t.cps[i + 5]) && s2 == s1 && up(t.cps[i + 7]) &&
                    up(t.cps[i + 8]) && boundary_after(t, i + 9))
                {
                    add(out, i, i + 9, pii_type::license_plate, s1 == '-' ? 0.9 : 0.7);
                    i += 8;
                }
            }
        }

        // Account names inside file paths: C:\Users\jdupont\, /home/jdupont/.
        void scan_user_paths(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::identifier)) return;
            static const char* markers[] = { "\\users\\", "/users/", "/home/", "\\documents and settings\\" };
            const std::size_t n = t.size();
            for (const char* m : markers)
            {
                const std::size_t len = std::char_traits<char>::length(m);
                for (std::size_t i = 0; i + len < n; ++i)
                {
                    bool match = true;
                    for (std::size_t k = 0; k < len && match; ++k)
                        match = to_lower(t.cps[i + k]) == static_cast<char32_t>(m[k]);
                    if (!match) continue;
                    const std::size_t b = i + len;
                    auto stop = [&](char32_t ch) {
                        return ch == '\\' || ch == '/' || ch == '"' || ch == '<' || ch == '>' || ch == '|' || ch == ':' ||
                               ch == '\t' || is_line_break(ch);
                    };
                    auto pause = [&](char32_t ch) { return is_space(ch) || ch == ',' || ch == ';' || ch == ')'; };
                    /* The account name ends at the next separator. A space, a comma or a
                       parenthesis ends it too, unless a separator comes within the next few
                       words: "C:\Users\Jean Dupont\Documents". */
                    std::size_t e = b;
                    while (e < n && !stop(t.cps[e]))
                    {
                        if (pause(t.cps[e]))
                        {
                            std::size_t q = e, spaces = 0;
                            while (q < n && !stop(t.cps[q]) && spaces <= 3 && q - e < 48)
                            {
                                if (t.cps[q] == ',' || t.cps[q] == ';' || t.cps[q] == ')') break;
                                if (is_space(t.cps[q])) ++spaces;
                                ++q;
                            }
                            if (q >= n || !(t.cps[q] == '\\' || t.cps[q] == '/')) break;
                            e = q;
                            break;
                        }
                        ++e;
                    }
                    while (e > b && (t.cps[e - 1] == '.' || is_space(t.cps[e - 1]))) --e;
                    std::string seg;
                    for (std::size_t k = b; k < e; ++k) utf8_append(seg, t.cps[k]);
                    const std::string key = fold_key(seg);
                    if (e - b >= 2 && key != "public" && key != "default" && key != "all users" && key != "shared" &&
                        key != "default user" && key != "administrator")
                        add(out, b, e, pii_type::identifier, 0.8);
                    i = std::max(i, e > 0 ? e - 1 : 0);
                }
            }
        }

    // ------------------------------------------------------------------------------------
    // Identifiers read from their layout, most of them after a cue: GPS coordinates, device
    // and SIM identifiers, UUIDs, vehicle numbers, national numbers of other countries,
    // French bank details and tax numbers, pre-2009 plates, phone numbers announced by a
    // cue, social handles and reference numbers announced by the noun they belong to.

        /* Matches a layout at i: 'd' a digit, 'A' a capital letter, 'X' a capital letter or
           a digit, 'h' a hexadecimal digit of either case, ' ' a space; any other character
           matches itself. The letters and digits matched are appended to compact. */
        static bool shape_at(const utf8_text& t, std::size_t i, const char* layout, std::size_t& end,
                             std::string& compact)
        {
            using namespace privacy_impl;
            compact.clear();
            std::size_t j = i;
            for (const char* l = layout; *l; ++l, ++j)
            {
                if (j >= t.size()) return false;
                const char32_t c = t.cps[j];
                const bool upper = c >= 'A' && c <= 'Z';
                switch (*l)
                {
                    case 'd': if (!is_ascii_digit(c)) return false; break;
                    case 'A': if (!upper) return false; break;
                    case 'X': if (!upper && !is_ascii_digit(c)) return false; break;
                    case 'h': if (!is_hex_cp(c)) return false; break;
                    case ' ': if (c != ' ') return false; continue;
                    default: if (c != static_cast<char32_t>(static_cast<unsigned char>(*l))) return false; continue;
                }
                compact.push_back(static_cast<char>(c));
            }
            end = j;
            return boundary_after(t, end) && !(t.at(end) == '.' && is_ascii_digit(t.at(end + 1))) &&
                   !(t.at(end) == '-' && privacy_impl::is_word_char(t.at(end + 1)));
        }

        /* True if one of the cues appears among the window code points before i as a whole
           word. A cue ending with '*' is a prefix: "tel*" is found in "Tél. :" and
           "telephone", not in "hotel"; "sim" is found in "SIM :", not in "Simon". */
        static bool cue_before(const utf8_text& t, std::size_t i, const std::vector<std::string>& cues,
                               std::size_t window = 40)
        {
            std::string before;
            const std::size_t start = i > window ? i - window : 0;
            for (std::size_t k = start; k < i; ++k)
                privacy_impl::utf8_append(before, t.cps[k]);
            const std::string key = privacy_impl::fold_key(before);
            auto letter = [](char ch) { return (ch >= 'a' && ch <= 'z') || static_cast<unsigned char>(ch) >= 0x80; };
            for (const std::string& c : cues)
            {
                const bool prefix = !c.empty() && c.back() == '*';
                const std::size_t len = prefix ? c.size() - 1 : c.size();
                if (len == 0) continue;
                for (std::size_t p = key.find(c.data(), 0, len); p != std::string::npos; p = key.find(c.data(), p + 1, len))
                {
                    if (p > 0 && letter(key[p - 1])) continue;
                    const std::size_t q = p + len;
                    if (!prefix && q < key.size() && (letter(key[q]) || (key[q] >= '0' && key[q] <= '9'))) continue;
                    return true;
                }
            }
            return false;
        }

        void scan_national_ids(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::identifier)) return;
            static const std::vector<std::string> be_cues = { "registre national", "numero national", "rijksregister*",
                                                              "niss", "insz", "national number", "n.n." };
            static const std::vector<std::string> nl_cues = { "bsn", "burgerservicenummer", "sofinummer", "sofi-nummer" };
            static const std::vector<std::string> de_cues = { "steuer*", "idnr", "identifikationsnummer", "tax id", "tin" };
            static const std::vector<std::string> pl_cues = { "pesel" };
            static const std::vector<std::string> pt_cues = { "nif", "contribuinte" };
            static const std::vector<std::string> us_cues = { "ssn", "social security" };
            static const std::vector<std::string> tax_cues = { "fiscal*", "spi", "impot*", "contribuable", "tax number" };
            static const std::vector<std::string> imei_cues = { "imei" };
            static const std::vector<std::string> imsi_cues = { "imsi" };
            static const std::vector<std::string> iccid_cues = { "iccid", "carte sim", "sim card", "numero sim", "sim" };
            static const std::vector<std::string> vin_cues = { "vin", "chassis", "numero de serie", "serial number",
                                                               "identification du vehicule", "vehicle identification",
                                                               "numero d'identification", "frame number" };
            static const std::vector<std::string> passport_cues = { "passeport", "passport" };
            // National numbers of other countries, mostly accepted after their usual label.
            static const std::vector<std::string> at_cues = { "svnr", "vsnr", "sv-nr", "sv-nummer",
                                                              "sozialversicherungsnummer", "versicherungsnummer" };
            static const std::vector<std::string> cz_cues = { "rodne cislo", "rodneho cisla", "birth number" };
            static const std::vector<std::string> dk_cues = { "cpr" };
            static const std::vector<std::string> fi_cues = { "hetu", "henkilotunnu*", "personbeteckning",
                                                              "personal identity code" };
            static const std::vector<std::string> se_cues = { "personnummer*", "personnr", "pers.nr",
                                                              "personal identity number" };
            static const std::vector<std::string> no_cues = { "fodselsnummer*", "fnr", "f.nr", "personnummer*" };
            static const std::vector<std::string> ie_cues = { "pps", "ppsn", "personal public service" };
            static const std::vector<std::string> ro_cues = { "cnp", "cod numeric personal" };
            static const std::vector<std::string> bg_cues = { "egn" };
            static const std::vector<std::string> hr_cues = { "oib" };
            static const std::vector<std::string> ee_cues = { "isikukood", "asmens kodas", "personal code",
                                                              "personal identification code" };
            static const std::vector<std::string> si_cues = { "emso" };
            static const std::vector<std::string> gr_cues = { "amka" };
            static const std::vector<std::string> nhs_cues = { "nhs" };
            static const std::vector<std::string> sin_cues = { "sin", "nas", "social insurance", "assurance sociale" };
            static const std::vector<std::string> cpf_cues = { "cpf" };
            static const std::vector<std::string> aadhaar_cues = { "aadhaar", "aadhar", "uidai" };
            static const std::vector<std::string> rut_cues = { "rut", "run", "rol unico" };
            static const std::vector<std::string> cuit_cues = { "cuit", "cuil" };
            static const std::vector<std::string> tr_cues = { "kimlik*", "t.c.", "tckn" };
            static const std::vector<std::string> il_cues = { "teudat zehut", "mispar zehut", "israeli id" };
            static const std::vector<std::string> za_cues = { "south african", "sa id", "identity number" };
            static const std::vector<std::string> tfn_cues = { "tfn", "tax file" };
            static const std::vector<std::string> rrn_cues = { "rrn", "resident registration" };
            static const std::vector<std::string> jp_cues = { "my number", "mynumber", "individual number", "kojin bango" };
            static const std::vector<std::string> is_cues = { "kennitala" };
            static const std::vector<std::string> itin_cues = { "itin", "taxpayer identification" };
            static const std::vector<std::string> cin_cues = { "cin", "cnie", "c.i.n", "carte d'identite nationale" };
            static const std::vector<std::string> nin_cues = { "nin", "numero d'identification nationa*",
                                                               "identifiant national" };
            static const std::vector<std::string> ramq_cues = { "ramq", "assurance maladie", "carte soleil" };
            const std::size_t n = t.size();
            std::string a;
            std::size_t end = 0;
            for (std::size_t i = 0; i < n; ++i)
            {
                const char32_t c = t.cps[i];
                if (!is_hex_cp(c) && !(c >= 'A' && c <= 'Z')) continue;
                const char32_t prev = t.before(i);
                if (is_word_char(prev) || prev == '-' || prev == '.' || prev == '/' || prev == '_') continue;
                auto take = [&](double confidence, int bonus) {
                    add(out, i, end, pii_type::identifier, confidence, bonus);
                    i = end - 1;
                };
                auto named = [&](const std::vector<std::string>& cues) { return cue_before(t, i, cues); };

                // UUIDs: advertising, cookie and device identifiers.
                if (shape_at(t, i, "hhhhhhhh-hhhh-hhhh-hhhh-hhhhhhhhhhhh", end, a) &&
                    is_uuid_shape(std::string(cp_substr_ascii(t, i, end))))
                { take(0.9, 12); continue; }

                if ((c >= 'A' && c <= 'Z') || is_ascii_digit(c))
                {
                    if (c >= 'A' && c <= 'Z' && shape_at(t, i, "XXXXXXXXXXXXXXXX", end, a) && it_fiscal_code_ok(a))
                    { take(0.95, 12); continue; }
                    if (c >= 'A' && c <= 'Z' && shape_at(t, i, "AAAAddddddAAAAAAXd", end, a) && mx_curp_ok(a))
                    { take(0.9, 12); continue; }
                    if (is_ascii_digit(c) && shape_at(t, i, "dddddddddddddddddX", end, a) && cn_resident_id_ok(a))
                    { take(0.85, 10); continue; }
                    if (shape_at(t, i, "XXXXXXXXXXXXXXXXX", end, a) && vin_shape_ok(a))
                    {
                        int letters = 0;
                        for (char ch : a) if (ch >= 'A' && ch <= 'Z') ++letters;
                        if (cue_before(t, i, vin_cues)) { take(0.95, 12); continue; }
                        if (letters >= 3 && vin_check_digit_ok(a)) { take(0.8, 8); continue; }
                    }
                }
                if (c >= 'A' && c <= 'Z')
                {
                    if ((shape_at(t, i, "AAddddddA", end, a) || shape_at(t, i, "AA dd dd dd A", end, a) ||
                         shape_at(t, i, "AA dddddd A", end, a)) && uk_nino_ok(a))
                    { take(0.85, 10); continue; }
                    if ((c == 'X' || c == 'Y' || c == 'Z') &&
                        (shape_at(t, i, "AdddddddA", end, a) || shape_at(t, i, "Addddddd-A", end, a) ||
                         shape_at(t, i, "A-ddddddd-A", end, a)) && es_dni_ok(a))
                    { take(0.9, 12); continue; }
                    // Quebec health insurance number and Maghreb national identity cards, after their label.
                    if ((shape_at(t, i, "AAAAdddddddd", end, a) || shape_at(t, i, "AAAA dddd dddd", end, a)) &&
                        named(ramq_cues))
                    { take(0.9, 10); continue; }
                    if ((shape_at(t, i, "AAdddddd", end, a) || shape_at(t, i, "Adddddd", end, a) ||
                         shape_at(t, i, "AAddddd", end, a) || shape_at(t, i, "Addddd", end, a)) && named(cin_cues))
                    { take(0.85, 10); continue; }
                    continue;
                }
                if (!is_ascii_digit(c)) continue;

                // Layouts with separators.
                if ((shape_at(t, i, "ddddddddA", end, a) || shape_at(t, i, "dddddddd-A", end, a) ||
                     shape_at(t, i, "dddddddd A", end, a)) && es_dni_ok(a))
                { take(0.9, 12); continue; }
                if (shape_at(t, i, "dd.dd.dd-ddd.dd", end, a) && be_national_number_ok(a)) { take(0.95, 12); continue; }
                if (shape_at(t, i, "ddd.dddd.dddd.dd", end, a) && ch_avs_ok(a)) { take(0.95, 12); continue; }
                if (shape_at(t, i, "ddd-dd-dddd", end, a) && (us_ssn_ok(a) || us_itin_ok(a))) { take(0.8, 8); continue; }
                if ((shape_at(t, i, "ddddd ddddd XXXXXXXXXXX dd", end, a) ||
                     shape_at(t, i, "dddddddddd XXXXXXXXXXX dd", end, a) ||
                     shape_at(t, i, "ddddddddddXXXXXXXXXXXdd", end, a)) && fr_rib_ok(a))
                { take(0.95, 12); continue; }
                if (shape_at(t, i, "dd-dddddd-dddddd-d", end, a) && luhn_ok(a)) { take(0.9, 12); continue; }
                if (shape_at(t, i, "ddAAddddd", end, a) && cue_before(t, i, passport_cues)) { take(0.95, 12); continue; }
                // Layouts of other countries.
                {
                    // Finnish HETU, whose century sign belongs to the value. A dash followed by
                    // digits only also writes dated serial numbers, so that form needs its label.
                    static const char* layouts[] = { "dddddd-dddX", "dddddd+dddX", "ddddddAdddX" };
                    bool found = false;
                    for (const char* layout : layouts)
                    {
                        if (!shape_at(t, i, layout, end, a)) continue;
                        const std::string v =
                            a.size() == 10 ? a.substr(0, 6) + static_cast<char>(t.cps[i + 6]) + a.substr(6) : a;
                        found = fi_hetu_ok(v) && (v[6] != '-' || !is_ascii_digit(v[10]) || named(fi_cues));
                        if (found) break;
                    }
                    if (found) { take(0.9, 12); continue; }
                }
                if (shape_at(t, i, "dddddd/dddd", end, a) && cz_birth_number_ok(a))
                {
                    const bool cued = named(cz_cues);
                    take(cued ? 0.95 : 0.75, cued ? 12 : 6);
                    continue;
                }
                if (shape_at(t, i, "dddddd/ddd", end, a) && named(cz_cues) && cz_birth_number_ok(a))
                {
                    take(0.9, 10);
                    continue;
                }
                if (shape_at(t, i, "dddddd-ddddddd", end, a) && kr_rrn_ok(a))
                {
                    const bool cued = named(rrn_cues);
                    take(cued ? 0.95 : 0.8, cued ? 12 : 8);
                    continue;
                }
                if (shape_at(t, i, "ddd.ddd.ddd-dd", end, a) && br_cpf_ok(a)) { take(0.9, 12); continue; }
                if (shape_at(t, i, "dd-dddddddd-d", end, a) && ar_cuit_ok(a)) { take(0.9, 12); continue; }
                if ((shape_at(t, i, "dd.ddd.ddd-X", end, a) || shape_at(t, i, "d.ddd.ddd-X", end, a)) && cl_rut_ok(a))
                { take(0.85, 10); continue; }
                if ((shape_at(t, i, "dddddddd-X", end, a) || shape_at(t, i, "ddddddd-X", end, a)) &&
                    named(rut_cues) && cl_rut_ok(a))
                { take(0.9, 10); continue; }
                if (shape_at(t, i, "ddd-ddd-ddd", end, a) && named(sin_cues) && ca_sin_ok(a)) { take(0.95, 12); continue; }
                if ((shape_at(t, i, "dddddd-dddd", end, a) || shape_at(t, i, "dddddd+dddd", end, a) ||
                     shape_at(t, i, "dddddddd-dddd", end, a)) &&
                    ((named(se_cues) && se_personnummer_ok(a)) || (named(dk_cues) && dk_cpr_ok(a)) ||
                     (named(is_cues) && is_kennitala_ok(a))))
                { take(0.95, 12); continue; }
                if ((shape_at(t, i, "dddddddAA", end, a) || shape_at(t, i, "dddddddA", end, a)) &&
                    named(ie_cues) && ie_pps_ok(a))
                { take(0.95, 12); continue; }

                /* Digit runs, plain or grouped by spaces. A number announced by its label and
                   passing its check outranks the layout of a phone number. */
                std::size_t e1 = 0, e2 = 0;
                const std::string compact = digit_run(t, i, "", e1, 21);
                const std::string spaced = digit_run(t, i, " ", e2, 21);
                const std::string* runs[2] = { &compact, &spaced };
                const std::size_t ends[2] = { e1, e2 };
                bool done = false;
                for (int r = 0; r < 2 && !done; ++r)
                {
                    const std::string& d = *runs[r];
                    end = ends[r];
                    if (r == 1 && end == e1) break;
                    if (!boundary_after(t, end) || (t.at(end) == '.' && is_ascii_digit(t.at(end + 1)))) continue;
                    auto cued = [&](const std::vector<std::string>& cues, bool (*ok)(const std::string&)) {
                        return named(cues) && ok(d);
                    };
                    switch (d.size())
                    {
                        case 8:
                            if (cued(tfn_cues, au_tfn_ok)) { take(0.9, 12); done = true; }
                            else if (named(cin_cues)) { take(0.85, 12); done = true; }
                            break;
                        case 9:
                            if (cue_before(t, i, nl_cues) && nl_bsn_ok(d)) { take(0.9, 12); done = true; }
                            else if (cue_before(t, i, pt_cues) && pt_nif_ok(d)) { take(0.9, 12); done = true; }
                            else if (cue_before(t, i, us_cues) && us_ssn_ok(d)) { take(0.9, 12); done = true; }
                            else if (cued(sin_cues, ca_sin_ok) || cued(tfn_cues, au_tfn_ok) || cued(il_cues, il_idnr_ok) ||
                                     cued(itin_cues, us_itin_ok))
                            { take(0.9, 12); done = true; }
                            break;
                        case 10:
                            if (cued(at_cues, at_svnr_ok) || cued(cz_cues, cz_birth_number_ok) || cued(bg_cues, bg_egn_ok) ||
                                cued(dk_cues, dk_cpr_ok) || cued(se_cues, se_personnummer_ok) || cued(nhs_cues, gb_nhs_ok) ||
                                cued(is_cues, is_kennitala_ok))
                            { take(0.9, 12); done = true; }
                            break;
                        case 11:
                            if (cue_before(t, i, be_cues) && be_national_number_ok(d)) { take(0.9, 12); done = true; }
                            else if (cue_before(t, i, de_cues) && de_tax_id_ok(d)) { take(0.9, 12); done = true; }
                            else if (cue_before(t, i, pl_cues) && pl_pesel_ok(d)) { take(0.9, 12); done = true; }
                            else if (cued(no_cues, no_fodselsnummer_ok) || cued(hr_cues, hr_oib_ok) ||
                                     cued(ee_cues, ee_lt_personal_code_ok) || cued(cpf_cues, br_cpf_ok) ||
                                     cued(gr_cues, gr_amka_ok) || cued(tr_cues, tr_kimlik_ok) || cued(cuit_cues, ar_cuit_ok))
                            { take(0.9, 12); done = true; }
                            break;
                        case 12:
                            if (cued(aadhaar_cues, in_aadhaar_ok) || cued(jp_cues, jp_my_number_ok) ||
                                cued(se_cues, se_personnummer_ok))
                            { take(0.9, 12); done = true; }
                            break;
                        case 18:
                            if (cn_resident_id_ok(d)) { take(0.85, 12); done = true; }
                            else if (named(nin_cues)) { take(0.85, 12); done = true; }
                            break;
                        case 13:
                            if (cued(ro_cues, ro_cnp_ok) || cued(si_cues, si_emso_ok) || cued(za_cues, za_idnr_ok) ||
                                cued(rrn_cues, kr_rrn_ok))
                            { take(0.9, 12); done = true; }
                            else if (ch_avs_ok(d)) { take(0.85, 12); done = true; }
                            else if (lu_national_number_ok(d)) { take(0.9, 12); done = true; }
                            else if (fr_tax_number_ok(d))
                            {
                                take(cue_before(t, i, tax_cues) ? 0.95 : 0.7, cue_before(t, i, tax_cues) ? 12 : 4);
                                done = true;
                            }
                            break;
                        case 15:
                            if (cue_before(t, i, imei_cues) && luhn_ok(d)) { take(0.95, 12); done = true; }
                            else if (cue_before(t, i, imsi_cues)) { take(0.9, 12); done = true; }
                            break;
                        case 19:
                        case 20:
                            if (d.compare(0, 2, "89") == 0 && luhn_ok(d))
                            {
                                take(cue_before(t, i, iccid_cues) ? 0.95 : 0.85, 10);
                                done = true;
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
        }

        static std::string cp_substr_ascii(const utf8_text& t, std::size_t b, std::size_t e)
        {
            std::string s;
            for (std::size_t k = b; k < e; ++k)
                s.push_back(t.cps[k] < 0x80 ? static_cast<char>(t.cps[k]) : '?');
            return s;
        }

        /* Reads a coordinate at i: an optional sign, degrees, and either decimal degrees
           (point or comma) or degrees, minutes and seconds; an optional hemisphere letter.
           Returns false if nothing coordinate-like starts at i. */
        static bool read_coordinate(const utf8_text& t, std::size_t i, std::size_t& end, double& value,
                                    int& decimals, char& hemisphere, bool& dms, char32_t& decimal_mark)
        {
            using namespace privacy_impl;
            const std::size_t n = t.size();
            std::size_t j = i;
            double sign = 1;
            if (j < n && (t.cps[j] == '-' || t.cps[j] == '+' || t.cps[j] == 0x2212))
            {
                if (t.cps[j] != '+') sign = -1;
                ++j;
            }
            std::size_t s = j;
            double deg = 0;
            while (j < n && is_ascii_digit(t.cps[j]) && j - s < 3) deg = deg * 10 + static_cast<double>(t.cps[j++] - '0');
            if (j == s || (j < n && is_ascii_digit(t.cps[j]))) return false;
            decimals = 0;
            dms = false;
            hemisphere = 0;
            decimal_mark = 0;
            value = deg;
            if (j + 1 < n && (t.cps[j] == '.' || t.cps[j] == ',') && is_ascii_digit(t.cps[j + 1]))
            {
                decimal_mark = t.cps[j];
                ++j;
                double scale = 0.1;
                while (j < n && is_ascii_digit(t.cps[j]))
                {
                    value += scale * static_cast<double>(t.cps[j++] - '0');
                    scale /= 10;
                    ++decimals;
                }
                if (j < n && t.cps[j] == 0xB0) ++j;
            }
            else if (j < n && t.cps[j] == 0xB0)
            {
                dms = true;
                ++j;
                for (int part = 0; part < 2; ++part)
                {
                    std::size_t k = j;
                    while (k < n && t.cps[k] == ' ') ++k;
                    const std::size_t ds = k;
                    double v = 0;
                    while (k < n && is_ascii_digit(t.cps[k]) && k - ds < 2)
                        v = v * 10 + static_cast<double>(t.cps[k++] - '0');
                    if (k == ds) break;
                    if (part == 1 && k + 1 < n && t.cps[k] == '.' && is_ascii_digit(t.cps[k + 1]))
                    {
                        ++k;
                        double scale = 0.1;
                        while (k < n && is_ascii_digit(t.cps[k]))
                        {
                            v += scale * static_cast<double>(t.cps[k++] - '0');
                            scale /= 10;
                        }
                    }
                    const char32_t mark = t.at(k);
                    const bool minute_mark = mark == '\'' || mark == 0x2032 || mark == 0x2019;
                    const bool second_mark = mark == '"' || mark == 0x2033 || mark == 0x201D ||
                                             (mark == '\'' && t.at(k + 1) == '\'');
                    if (v >= 60) return false;
                    if (part == 0 && minute_mark) { value += v / 60; j = k + 1; continue; }
                    if (part == 1 && second_mark) { value += v / 3600; j = k + (mark == '\'' ? 2 : 1); continue; }
                    break;
                }
            }
            else
            {
                return false;
            }
            std::size_t k = j;
            while (k < n && t.cps[k] == ' ' && k - j < 1) ++k;
            const char32_t h = t.at(k);
            if ((h == 'N' || h == 'S' || h == 'E' || h == 'W' || h == 'O') && !is_word_char(t.at(k + 1)))
            {
                hemisphere = static_cast<char>(h);
                j = k + 1;
                if (h == 'S' || h == 'W' || (h == 'O')) sign = -1;
            }
            value *= sign;
            end = j;
            return true;
        }

        void scan_geo(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::geolocation)) return;
            static const std::vector<std::string> cues = { "gps", "coordonnee*", "coordinate*", "latitude", "longitude",
                                                           "lat", "position", "localisation", "location", "geoloc*",
                                                           "geo:*", "lieu" };
            const std::size_t n = t.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const char32_t c = t.cps[i];
                if (!(is_ascii_digit(c) || c == '-' || c == '+' || c == 0x2212)) continue;
                const char32_t prev = t.before(i);
                if (is_word_char(prev) || prev == '.' || prev == ',' || is_ascii_digit(prev)) continue;
                if ((c == '-' || c == '+') && !is_ascii_digit(t.at(i + 1))) continue;
                std::size_t e1, e2;
                double lat, lon;
                int d1, d2;
                char h1, h2;
                bool dms1, dms2;
                char32_t m1, m2;
                if (!read_coordinate(t, i, e1, lat, d1, h1, dms1, m1)) continue;
                // Separator: comma, semicolon or slash with optional spaces, or spaces alone.
                std::size_t j = e1;
                while (j < n && t.cps[j] == ' ' && j - e1 < 3) ++j;
                bool punct = false;
                if (j < n && (t.cps[j] == ',' || t.cps[j] == ';' || t.cps[j] == '/'))
                {
                    if (t.cps[j] == ',' && m1 == ',' && j == e1) continue;   // "48,85,2,35" is ambiguous
                    punct = true;
                    ++j;
                    while (j < n && t.cps[j] == ' ' && j - e1 < 6) ++j;
                }
                if (j == e1) continue;
                if (!read_coordinate(t, j, e2, lon, d2, h2, dms2, m2)) continue;
                if (dms1 != dms2 || !boundary_after(t, e2) || (t.at(e2) == '.' && is_ascii_digit(t.at(e2 + 1))) ||
                    (t.at(e2) == ',' && is_ascii_digit(t.at(e2 + 1))))
                    continue;
                if (std::abs(lat) > 90 || std::abs(lon) > 180) continue;
                {
                    std::size_t q = e2;
                    while (q < n && t.cps[q] == ' ') ++q;
                    const char32_t unit = t.at(q);
                    if (unit == 0x20AC || unit == '$' || unit == '%' || unit == 0xA3) continue;   // amounts, rates
                }
                const bool hemispheres = h1 && h2;
                const bool cue = cue_before(t, i, cues, 32);
                // Two values ending in 0 with four decimals or more are more often amounts.
                const bool round = t.before(e1) == '0' && t.before(e2) == '0';
                bool ok;
                if (dms1) ok = hemispheres || cue;
                else ok = (d1 >= 4 && d2 >= 4 && (punct || hemispheres || cue) && (!round || hemispheres || cue)) ||
                          (d1 >= 2 && d2 >= 2 && (hemispheres || cue));
                if (!ok) continue;
                add(out, i, e2, pii_type::geolocation, 0.9);
                i = e2 - 1;
            }
        }

        void scan_phone_cues(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::phone)) return;
            static const std::vector<std::string> cues = { "tel*", "phone", "phones", "mobile", "portable", "gsm", "fax",
                                                           "cell", "cellphone", "whatsapp", "appeler au", "joindre au",
                                                           "call", "standard :", "standard telephonique",
                                                           "ligne directe", "direct line", "numero de telephone" };
            const std::size_t n = t.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const char32_t c = t.cps[i];
                if (!(is_ascii_digit(c) || c == '(')) continue;
                const char32_t prev = t.before(i);
                if (is_word_char(prev) || prev == '.' || prev == '+' || prev == '-' || prev == '/' || prev == ',') continue;
                // (555) 123-4567, with or without a cue.
                std::size_t end;
                std::string a;
                if (c == '(' && (shape_at(t, i, "(ddd) ddd-dddd", end, a) || shape_at(t, i, "(ddd)ddd-dddd", end, a) ||
                                 shape_at(t, i, "(ddd) ddd dddd", end, a)))
                {
                    add(out, i, end, pii_type::phone, 0.85, 1);
                    i = end - 1;
                    continue;
                }
                if (!cue_before(t, i, cues, 24)) continue;
                // Groups of digits: for each, its end and the digits up to it.
                std::vector<std::pair<std::size_t, std::size_t>> groups;
                std::size_t j = i, digits = 0;
                while (j < n && digits <= 20)
                {
                    const char32_t ch = t.cps[j];
                    if (is_ascii_digit(ch))
                    {
                        while (j < n && is_ascii_digit(t.cps[j])) { ++digits; ++j; }
                        groups.push_back(std::make_pair(j, digits));
                        continue;
                    }
                    if ((ch == ' ' || ch == '.' || ch == '-') && j + 1 < n &&
                        (is_ascii_digit(t.cps[j + 1]) || t.cps[j + 1] == '('))
                    { ++j; continue; }
                    if (ch == '(' && j + 2 < n && is_ascii_digit(t.cps[j + 1]))
                    {
                        std::size_t k = j + 1;
                        while (k < n && is_ascii_digit(t.cps[k]) && k - j <= 4) ++k;
                        if (k < n && t.cps[k] == ')')
                        {
                            digits += k - j - 1;
                            j = k + 1;
                            groups.push_back(std::make_pair(j, digits));
                            continue;
                        }
                    }
                    break;
                }
                const std::size_t g = phone_prefix(groups, false);
                if (g == std::string::npos)
                {
                    // The run is no phone number: none of its digits starts one.
                    if (!groups.empty()) i = groups.back().first - 1;
                    continue;
                }
                const std::size_t last = groups[g].first;
                const bool whole_run = g + 1 == groups.size();
                if (!boundary_after(t, last) || (whole_run && t.at(last) == '.' && is_ascii_digit(t.at(last + 1))) ||
                    t.at(last) == '/')
                    continue;
                // "entretien téléphonique du 16-02-2027": a date after the cue stays a date.
                {
                    std::string value;
                    for (std::size_t q = i; q < last; ++q)
                        value.push_back(static_cast<char>(t.cps[q]));
                    civil_date d;
                    date_format f;
                    if (parse_date(value, options_.date_order, d, f) && f.has_day())
                    {
                        i = last - 1;
                        continue;
                    }
                }
                add(out, i, last, pii_type::phone, 0.85);
                i = last - 1;
            }
        }

        /* The group a phone number read from a run of digit groups ends at, or npos. groups
           holds, for each group, its end and the digits up to it. A number in the French
           layout (0X XX XX XX XX) has ten digits. Otherwise the number is the longest prefix
           of 8 to 15 digits; without a country code, a final group of five digits after
           groups of three digits or less is a postal code and stays out. */
        static std::size_t phone_prefix(const std::vector<std::pair<std::size_t, std::size_t>>& groups,
                                        bool country_code)
        {
            auto size = [&](std::size_t g) { return groups[g].second - (g > 0 ? groups[g - 1].second : 0); };
            if (!country_code && groups.size() > 5)
            {
                bool pairs = true;
                for (std::size_t g = 0; g < 5 && pairs; ++g) pairs = size(g) == 2;
                if (pairs) return 4;
            }
            for (std::size_t g = groups.size(); g-- > 0;)
            {
                const std::size_t d = groups[g].second;
                if (d < 8 || d > 15) continue;
                bool postal = !country_code && g > 0 && size(g) == 5;
                for (std::size_t h = 0; h < g && postal; ++h) postal = size(h) <= 3;
                if (!postal) return g;
            }
            return std::string::npos;
        }

        void scan_handles(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::identifier)) return;
            static const std::unordered_set<std::string> stop = make_word_set(
                "param tparam return returns retval throws throw exception see since author version deprecated "
                "override media import charset font-face keyframes supports page namespace property interface "
                "implementation end protocol class selector synthesize dynamic brief file details note todo fixme "
                "example code link inheritdoc value here channel everyone all test before after beforeeach "
                "aftereach component injectable input output ngmodule entity table column id autowired service "
                "controller get post put delete patch mapping requestmapping getmapping postmapping bean "
                "configuration transactional data builder getter setter valid notnull json xml type types typedef "
                "var pre invariant ingroup defgroup addtogroup interface layer apply include mixin extend if else "
                "each for function use forward tailwind container");
            const std::size_t n = t.size();
            for (std::size_t i = 0; i + 2 < n; ++i)
            {
                if (t.cps[i] != '@') continue;
                const char32_t prev = t.before(i);
                if (!(i == 0 || is_space(prev) || prev == '(' || prev == ',' || prev == ':' || prev == '"' ||
                      prev == '\'' || prev == 0xAB || prev == '['))
                    continue;
                std::size_t j = i + 1;
                int letters = 0;
                while (j < n && (is_ascii_alnum_cp(t.cps[j]) || t.cps[j] == '_' || t.cps[j] == '.') && j - i <= 31)
                {
                    if ((t.cps[j] >= 'a' && t.cps[j] <= 'z') || (t.cps[j] >= 'A' && t.cps[j] <= 'Z')) ++letters;
                    ++j;
                }
                while (j > i + 1 && t.cps[j - 1] == '.') --j;
                if (j - i - 1 < 2 || j - i - 1 > 30 || letters == 0 || t.at(j) == '@' || t.at(j) == '(' ||
                    is_word_char(t.at(j)))
                    continue;
                std::string name;
                for (std::size_t k = i + 1; k < j; ++k)
                    name.push_back(static_cast<char>(t.cps[k] | (t.cps[k] >= 'A' && t.cps[k] <= 'Z' ? 0x20 : 0)));
                if (stop.count(name)) continue;
                add(out, i, j, pii_type::identifier, 0.8, 2);
                i = j - 1;
            }
        }

        void scan_old_plates(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::license_plate)) return;
            static const std::vector<std::string> cues = { "immatricul*", "plaque", "vehicule", "voiture", "camion",
                                                           "moto", "scooter", "auto", "automobile", "fourgon", "remorque" };
            const std::size_t n = t.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                if (!is_ascii_digit(t.cps[i]) || !boundary_before(t, i)) continue;
                std::size_t j = i;
                while (j < n && is_ascii_digit(t.cps[j]) && j - i < 5) ++j;
                if (j - i > 4 || j >= n || !(t.cps[j] == ' ' || t.cps[j] == '-')) continue;
                const char32_t sep = t.cps[j];
                std::size_t k = j + 1;
                while (k < n && t.cps[k] >= 'A' && t.cps[k] <= 'Z' && k - j <= 4) ++k;
                if (k - j - 1 < 1 || k - j - 1 > 3 || k >= n || t.cps[k] != sep) continue;
                std::size_t m = k + 1, e = m;
                while (e < n && (is_ascii_digit(t.cps[e]) || t.cps[e] == 'A' || t.cps[e] == 'B') && e - m < 4) ++e;
                std::string dep;
                for (std::size_t q = m; q < e; ++q) dep.push_back(static_cast<char>(t.cps[q]));
                const bool dep_ok =
                    (dep.size() == 2 && is_ascii_digit_char(dep[0]) && is_ascii_digit_char(dep[1]) && dep != "00" &&
                     dep != "20" && std::stoi(dep) <= 95) ||
                    dep == "2A" || dep == "2B" ||
                    (dep.size() == 3 && dep >= "971" && dep <= "976");
                if (!dep_ok || !boundary_after(t, e) || !cue_before(t, i, cues, 48)) continue;
                add(out, i, e, pii_type::license_plate, 0.85, 4);
                i = e - 1;
            }
        }

        /* User names written where software writes them: "user=jdupont", "session opened for
           jdupont", "created by jdupont", "Ticket 2024-123 ouvert par jdupont :", the chat
           lines "[10:42] jdupont: ..." and "[10:42] <jdupont>", and after "username is" or
           "identifiant de connexion est". After a phrase, the value is a login: no space, and
           in lower case or with a digit, a period or an underscore. */
        void scan_logins(const utf8_text& t, const std::vector<word_token>& words, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::identifier)) return;
            const std::size_t n = t.size();
            auto login_char = [&](char32_t c) {
                return is_ascii_alnum_cp(c) || c == '.' || c == '_' || c == '-' || (c >= 0xC0 && is_letter(c));
            };
            // The login starting at p, or an empty range.
            auto login_at = [&](std::size_t p, bool any_shape) {
                std::size_t e = p;
                bool letter = false, mark = false, upper = false;
                while (e < n && e - p <= 40 && login_char(t.cps[e]))
                {
                    if (is_letter(t.cps[e])) letter = true;
                    if (is_upper(t.cps[e])) upper = true;
                    if (is_ascii_digit(t.cps[e]) || t.cps[e] == '.' || t.cps[e] == '_') mark = true;
                    ++e;
                }
                while (e > p && (t.cps[e - 1] == '.' || t.cps[e - 1] == '-')) --e;
                const bool shape_ok = any_shape || mark || !upper;
                if (e - p < 2 || e - p > 40 || !letter || !shape_ok || (e < n && (is_word_char(t.cps[e]) || t.cps[e] == '@')))
                    return std::make_pair(p, p);
                return std::make_pair(p, e);
            };
            auto take = [&](std::pair<std::size_t, std::size_t> r) {
                if (r.second > r.first) add(out, r.first, r.second, pii_type::identifier, 0.85, 2);
            };
            // "user=jdupont".
            static const std::unordered_set<std::string> keys = make_word_set(
                "user username login usr uname account acct owner");
            for (std::size_t i = 1; i + 1 < n; ++i)
            {
                if (t.cps[i] != '=') continue;
                std::size_t k = i;
                while (k > 0 && is_letter(t.cps[k - 1])) --k;
                if (k == i || (k > 0 && is_word_char(t.cps[k - 1]))) continue;
                std::string key;
                for (std::size_t q = k; q < i; ++q) utf8_append(key, to_lower(t.cps[q]));
                if (!keys.count(key)) continue;
                std::size_t p = i + 1;
                if (p < n && (t.cps[p] == '"' || t.cps[p] == '\'')) ++p;
                take(login_at(p, true));
            }
            // "[10:42] jdupont:", "[10:42:05] <jdupont>".
            for (std::size_t i = 0; i < n; ++i)
            {
                if (t.cps[i] != '[' || (i > 0 && !is_line_break(t.cps[i - 1]))) continue;
                std::size_t q = i + 1, digits = 0, colons = 0;
                while (q < n && (is_ascii_digit(t.cps[q]) || t.cps[q] == ':'))
                {
                    if (t.cps[q] == ':') ++colons; else ++digits;
                    ++q;
                }
                if (q >= n || t.cps[q] != ']' || colons < 1 || colons > 2 || digits < 3) continue;
                ++q;
                while (q < n && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                const bool angle = q < n && t.cps[q] == '<';
                if (angle) ++q;
                const auto r = login_at(q, true);
                if (r.second > r.first && r.second < n && t.cps[r.second] == (angle ? '>' : ':')) take(r);
            }
            // "(pc-jdupont)", "(tel-mpoulain12)": a device named after its user.
            static const std::unordered_set<std::string> devices = make_word_set(
                "pc tel portable laptop desktop mobile phone ws poste");
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                const std::size_t e = words[k].e;
                if (!devices.count(words[k].key) || e + 1 >= n || t.cps[e] != '-' || words[k].b == 0 ||
                    t.cps[words[k].b - 1] != '(')
                    continue;
                const auto r = login_at(e + 1, false);
                if (r.second > r.first && t.at(r.second) == ')') take(r);
            }
            // After a phrase.
            static const std::vector<std::vector<std::string>> phrases = {
                { "session", "opened", "for" }, { "created", "by" }, { "modified", "by" }, { "opened", "by" },
                { "ouvert", "par" }, { "cree", "par" }, { "modifie", "par" }, { "username", "is" },
                { "login", "is" }, { "connexion", "est" }, { "utilisateur", "est" }, { "identifiant", "est" } };
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                for (const auto& phrase : phrases)
                {
                    if (k + 1 < phrase.size()) continue;
                    bool match = true;
                    for (std::size_t m = 0; m < phrase.size() && match; ++m)
                        match = words[k + 1 - phrase.size() + m].key == phrase[m];
                    if (!match) continue;
                    std::size_t p = words[k].e;
                    if (p >= n || t.cps[p] != ' ') continue;
                    ++p;
                    const auto r = login_at(p, false);
                    const char32_t after = t.at(r.second);
                    if (r.second > r.first && (after == 0 || after == ' ' || after == ':' || after == ',' ||
                                                after == '.' || after == ')' || after == '(' || is_line_break(after)))
                    {
                        std::string value;
                        for (std::size_t q = r.first; q < r.second; ++q) utf8_append(value, to_lower(t.cps[q]));
                        if (!common_words().count(value)) take(r);
                    }
                    break;
                }
            }
        }

        /* A reference that carries a year, without a cue: "INC-2026-0147", "LF-2026-0331". Two to
           six capitals, a year from 1990 to 2099 and three to six digits, joined by hyphens: the
           shape of a file, ticket or incident number, whose year would also tell how far the
           protected dates moved. */
        void scan_year_references(const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::identifier)) return;
            const std::size_t n = t.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                if (!(t.cps[i] >= 'A' && t.cps[i] <= 'Z') || is_word_char(t.before(i)) || t.before(i) == '-')
                    continue;
                std::size_t j = i;
                while (j < n && t.cps[j] >= 'A' && t.cps[j] <= 'Z') ++j;
                if (j - i < 2 || j - i > 6 || t.at(j) != '-') continue;
                std::size_t y = j + 1, d = y;
                while (d < n && is_ascii_digit(t.cps[d]) && d - y < 5) ++d;
                if (d - y != 4 || t.at(d) != '-') continue;
                const int year = (static_cast<int>(t.cps[y] - '0') * 10 + static_cast<int>(t.cps[y + 1] - '0')) * 100 +
                                 static_cast<int>(t.cps[y + 2] - '0') * 10 + static_cast<int>(t.cps[y + 3] - '0');
                if (year < 1990 || year > 2099) continue;
                std::size_t e = d + 1;
                while (e < n && is_ascii_digit(t.cps[e]) && e - d <= 7) ++e;
                if (e - d - 1 < 3 || e - d - 1 > 6 || is_word_char(t.at(e)) ||
                    (t.at(e) == '-' && is_word_char(t.at(e + 1))))
                    continue;
                add(out, i, e, pii_type::identifier, 0.8);
                i = e;
            }
        }

        void scan_cued_references(const utf8_text& t, const std::vector<word_token>& words,
                                  std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::identifier)) return;
            static const std::unordered_set<std::string> cues = make_word_set(
                "dossier client cliente contrat adherent adherente allocataire sinistre police matricule badge "
                "salarie salariee employe employee patient patiente ipp nda rpps adeli reservation commande "
                "facture abonne abonnee abonnement membre inscription etudiant etudiante candidat candidate "
                "beneficiaire assure assuree permis cni compte ticket booking order invoice customer member "
                "account policy claim student licence license case affaire requete pv proces-verbal plainte "
                "contravention avis reclamation reclamations remboursement resiliation demande colis parcel "
                "package envoi shipment tracking decision mrn sejour file request application permit ref "
                "reference tag company");
            static const std::unordered_set<std::string> fillers = make_word_set(
                "n no nº num numero number nr de du d des ref reference id identifiant code est is le la votre "
                "your vos the of our notre");
            const std::size_t n = t.size();
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                if (!cues.count(words[k].key)) continue;
                std::size_t p = words[k].e;
                std::size_t w = k + 1;
                int skipped = 0;
                // Fillers and punctuation between the noun and the value.
                while (p < n && p - words[k].e <= 30)
                {
                    const char32_t ch = t.cps[p];
                    // An opening quote too: reference "B4694342".
                    if (ch == ' ' || ch == ':' || ch == '#' || ch == 0xB0 || ch == 0xBA || ch == '.' || ch == '\'' ||
                        ch == 0x2019 || ch == '-' || ch == 0xA0 || ch == '"' || ch == 0xAB || ch == 0x201C)
                    { ++p; continue; }
                    if (w < words.size() && words[w].b == p && fillers.count(words[w].key) && skipped < 4 &&
                        !(is_ascii_digit(t.at(words[w].e))))
                    {
                        p = words[w].e;
                        ++w;
                        ++skipped;
                        continue;
                    }
                    break;
                }
                if (p >= n || !same_line(t, words[k].e, p) || !is_ascii_alnum_cp(t.cps[p])) continue;
                // The value: letters and digits with internal separators.
                std::size_t e = p, alnum = 0, digits = 0;
                while (e < n && alnum <= 32)
                {
                    const char32_t ch = t.cps[e];
                    if (is_ascii_alnum_cp(ch)) { ++alnum; if (is_ascii_digit(ch)) ++digits; ++e; continue; }
                    if ((ch == '-' || ch == '/' || ch == '.' || ch == '_') && e + 1 < n && is_ascii_alnum_cp(t.cps[e + 1]))
                    { ++e; continue; }
                    break;
                }
                if (alnum < 4 || alnum > 32 || digits < 2 || is_word_char(t.at(e))) continue;
                if (alnum == 4 && digits == 4 && (t.cps[p] == '1' || t.cps[p] == '2') && e == p + 4) continue;   // a year
                // "the file export_2010.csv": a file name.
                {
                    std::size_t dot = e;
                    while (dot > p && t.cps[dot - 1] != '.') --dot;
                    if (dot > p)
                    {
                        std::string extension;
                        for (std::size_t q = dot; q < e; ++q) utf8_append(extension, to_lower(t.cps[q]));
                        if (file_extensions().count(extension)) continue;
                    }
                }
                // "facture de 47370,72 €": an amount.
                if (digits == alnum && t.at(e) == ',' && is_ascii_digit(t.at(e + 1))) continue;
                {
                    std::size_t q = e;
                    while (q < n && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                    const char32_t c = t.at(q);
                    if (digits == alnum && (c == 0x20AC || c == '$' || c == 0xA3 ||
                                            (q + 2 < n && t.cps[q] == 'E' && t.cps[q + 1] == 'U' && t.cps[q + 2] == 'R')))
                        continue;
                }
                add(out, p, e, pii_type::identifier, 0.85, 2);
            }
        }

        // "luhn" runs on the digits of value, "iban" on its letters, in capitals, and digits.
        static bool pattern_check_ok(const std::string& check, const std::string& value)
        {
            if (check.empty()) return true;
            const bool luhn = check == "luhn";
            std::string a;
            for (char ch : value)
            {
                if (ch >= '0' && ch <= '9') a.push_back(ch);
                else if (!luhn && ch >= 'A' && ch <= 'Z') a.push_back(ch);
                else if (!luhn && ch >= 'a' && ch <= 'z') a.push_back(static_cast<char>(ch - 32));
            }
            if (luhn) return privacy_impl::luhn_ok(a);
            if (check == "iban") return privacy_impl::iban_checksum_valid(a);
            return false;
        }

        void scan_patterns(const std::string& text, const utf8_text& t, std::vector<candidate>& out) const
        {
            if (patterns_.empty()) return;
            /* Patterns are matched line by line, in windows of 2048 bytes that overlap by
               512 bytes, which bounds the recursion and the backtracking of std::regex on
               long paragraphs. A match is kept by the window it starts in before the
               overlap, so it is reported once; a match longer than the overlap may be cut
               at the end of a window. */
            const std::size_t window = 2048, overlap = 512, step = window - overlap;
            for (const compiled_pattern& p : patterns_)
            {
                if (!enabled(p.spec.type)) continue;
                std::size_t line = 0;
                while (line < text.size())
                {
                    std::size_t line_end = text.find('\n', line);
                    if (line_end == std::string::npos) line_end = text.size();
                    std::size_t covered = line;   // the end of the matches already reported
                    for (std::size_t b = line; b < line_end; b += step)
                    {
                        const std::size_t e = std::min(line_end, b + window);
                        const bool last_window = e == line_end;
                        auto flags = std::regex_constants::match_default;
                        // The text before and after the window is there: '^', '$' and '\b' see it.
                        if (b > line) flags |= std::regex_constants::match_prev_avail;
                        if (!last_window) flags |= std::regex_constants::match_not_eol;
                        try
                        {
                            const std::sregex_iterator last;
                            for (std::sregex_iterator it(text.begin() + static_cast<std::ptrdiff_t>(b),
                                                         text.begin() + static_cast<std::ptrdiff_t>(e), *p.re, flags);
                                 it != last; ++it)
                            {
                                const std::size_t whole = static_cast<std::size_t>((*it)[0].first - text.begin());
                                if (!last_window && whole >= b + step) break;
                                if (whole < covered) continue;
                                covered = static_cast<std::size_t>((*it)[0].second - text.begin());
                                const auto& m = (*it)[static_cast<int>(p.spec.group)];
                                if (!m.matched || m.length() == 0) continue;
                                const std::size_t mb = static_cast<std::size_t>(m.first - text.begin());
                                const std::size_t me = static_cast<std::size_t>(m.second - text.begin());
                                if (!pattern_check_ok(p.spec.check, std::string(m.first, m.second))) continue;
                                const std::size_t cb = static_cast<std::size_t>(
                                    std::lower_bound(t.offsets.begin(), t.offsets.end(), mb) - t.offsets.begin());
                                const std::size_t ce = static_cast<std::size_t>(
                                    std::lower_bound(t.offsets.begin(), t.offsets.end(), me) - t.offsets.begin());
                                // Patterns describe what the organisation knows: they outrank the built-in scanners.
                                if (ce > cb && ce <= t.size())
                                    add(out, cb, ce, p.spec.type, p.spec.confidence, 150, p.spec.type == pii_type::person);
                            }
                        }
                        catch (std::regex_error& err)
                        {
                            throw error("pii_detector: the pattern \"" + p.spec.regex + "\" failed on a line: " +
                                        err.what());
                        }
                        if (last_window) break;
                    }
                    line = line_end + 1;
                }
            }
        }

        void scan_recognizers(const std::string& text, const utf8_text& t, std::vector<candidate>& out) const
        {
            if (recognizers_.empty() || t.size() == 0) return;
            // The cache is keyed by the recognizer and the SHA-256 digest of the text.
            const std::array<unsigned char,32> digest = sha256_digest(text);
            for (const recognizer_entry& r : recognizers_)
            {
                std::vector<pii_span> spans;
                {
                    std::lock_guard<std::mutex> lock(cache_->m);
                    std::string key = std::to_string(r.id) + ':';
                    key.append(reinterpret_cast<const char*>(digest.data()), digest.size());
                    const auto it = cache_->spans.find(key);
                    if (it != cache_->spans.end())
                    {
                        spans = it->second;
                    }
                    else
                    {
                        spans = r.fn(text);
                        if (cache_->spans.size() >= 8192) cache_->spans.clear();
                        cache_->spans.emplace(key, spans);
                    }
                }
                add_recognized(text, t, spans, r.min_confidence, out);
            }
        }

        /* True if the words are terms of a medical record, a measure or a table heading, with
           function words between them: "Créatinine", "LDL-cholestérol", "Bilan de santé". */
        static bool technical_words(const std::string& text, const utf8_text& t,
                                    const std::vector<std::pair<std::size_t, std::size_t>>& words)
        {
            using namespace privacy_impl;
            bool any = false;
            for (const auto& w : words)
            {
                const std::string key = fold_key(cp_substr(t, text, w.first, w.second));
                std::size_t from = 0;
                while (from <= key.size())
                {
                    std::size_t to = key.find_first_of("-'", from);
                    if (to == std::string::npos) to = key.size();
                    const std::string part = key.substr(from, to - from);
                    if (technical_terms().count(part)) any = true;
                    else if (!part.empty() && !function_words().count(part)) return false;
                    from = to + 1;
                }
            }
            return any;
        }

        // The words of t in [b, e): letters and digits, with the hyphens and apostrophes inside them.
        static std::vector<std::pair<std::size_t, std::size_t>> span_words(const utf8_text& t, std::size_t b,
                                                                           std::size_t e)
        {
            using namespace privacy_impl;
            std::vector<std::pair<std::size_t, std::size_t>> words;
            for (std::size_t i = b; i < e;)
            {
                while (i < e && !is_word_char(t.cps[i])) ++i;
                std::size_t j = i;
                while (j < e && (is_word_char(t.cps[j]) ||
                                 ((t.cps[j] == '-' || t.cps[j] == '\'' || t.cps[j] == 0x2019) && j + 1 < e &&
                                  is_word_char(t.cps[j + 1]))))
                    ++j;
                if (j > i) words.push_back(std::make_pair(i, j));
                i = j;
            }
            return words;
        }

        // A capital letter alone, or a letter followed by a period, is an initial: "A. Ferguson".
        static bool initial_word(const utf8_text& t, const std::pair<std::size_t, std::size_t>& w)
        {
            return w.second == w.first + 1 && (privacy_impl::is_upper(t.cps[w.first]) || t.at(w.second) == '.');
        }

        // The first letter of the name in a word: after an elision, the one after the apostrophe ("d'Arras").
        static char32_t name_letter(const utf8_text& t, const std::pair<std::size_t, std::size_t>& w)
        {
            std::size_t first = w.first;
            for (std::size_t q = w.first + 1; q < w.second && q <= w.first + 2; ++q)
                if ((t.cps[q] == '\'' || t.cps[q] == 0x2019) && q + 1 < w.second) first = q + 1;
            return t.cps[first];
        }

        // The small words that belong to the name of a person ("de", "Le") or of a place ("sur", "les").
        static bool name_particle(pii_type type, const std::string& key)
        {
            using namespace privacy_impl;
            static const std::unordered_set<std::string> place_words = make_word_set("of upon on in by the et");
            if (type == pii_type::person) return person_particles().count(key) != 0;
            if (type == pii_type::location) return location_kept_words().count(key) || place_words.count(key);
            return false;
        }

        /* A piece of a learned recognizer that names nothing and would spoil the name it is
           merged with, made of whole words: a greeting alone ("Bonsoir" in "Bonsoir Maëlys"), a
           field label alone unless it ends the name of a place ("City" in "Quebec City", after a
           piece of the same type), function words that are no particle of a name ("Votre") and
           a common word in lower case ("commande"). Pieces of words, particles and logins are
           judged once merged by spurious_learned_span(). */
        bool spurious_learned_piece(const std::string& text, const utf8_text& t, std::size_t b, std::size_t e,
                                    pii_type type, bool after_same_type) const
        {
            using namespace privacy_impl;
            if (type != pii_type::person && type != pii_type::location && type != pii_type::organization)
                return false;
            if ((b > 0 && is_letter(t.cps[b - 1]) && is_letter(t.cps[b])) ||
                (e < t.size() && is_letter(t.cps[e - 1]) && is_letter(t.cps[e])))
                return false;
            const std::vector<std::pair<std::size_t, std::size_t>> words = span_words(t, b, e);
            if (words.empty()) return true;
            // A country names no town: "Cameroun" on the line after "Douala", "Sud" in "Afrique du Sud".
            if (type == pii_type::location && in_country_name(text, t, b, e)) return true;
            bool all_function = true, all_particles = true;
            for (const auto& w : words)
            {
                const std::string key = fold_key(cp_substr(t, text, w.first, w.second));
                if (initial_word(t, w) || !function_words().count(key)) all_function = false;
                if (!name_particle(type, key)) all_particles = false;
            }
            if (all_particles) return false;
            if (all_function) return true;
            if (technical_words(text, t, words)) return true;
            if (words.size() > 1 || initial_word(t, words[0])) return false;
            const std::string key = fold_key(cp_substr(t, text, words[0].first, words[0].second));
            const char32_t c = name_letter(t, words[0]);
            if (greeting_words().count(key)) return true;
            if (field_labels().count(key) && !(after_same_type && is_upper(c))) return true;
            return type != pii_type::location && is_letter(c) && !is_upper(c) && common_words().count(key);
        }

        /* Whether [b, e) is the name of a country, or a piece of one within its line ("Sud" in
           "Afrique du Sud"). The countries that are also the name of their capital
           ("Luxembourg", "Singapore") stay places when they are the whole span. */
        bool in_country_name(const std::string& text, const utf8_text& t, std::size_t b, std::size_t e) const
        {
            using namespace privacy_impl;
            static const std::unordered_set<std::string> cities = make_phrase_set(
                "luxembourg|singapour|singapore|monaco|djibouti|koweit|kuwait|panama|guatemala|mexico|saint-marin|"
                "san marino");
            const std::string key = fold_key(cp_substr(t, text, b, e));
            if (country_names().count(key)) return !cities.count(key);
            // The words around the span on its line, within a few words of it.
            std::size_t lb = b, le = e;
            while (lb > 0 && b - lb < 48 && !is_line_break(t.cps[lb - 1])) --lb;
            while (le < t.size() && le - e < 48 && !is_line_break(t.cps[le])) ++le;
            std::vector<std::pair<std::size_t, std::size_t>> words = span_words(t, lb, le);
            if (!words.empty() && words.back().second == le && le < t.size() && is_word_char(t.cps[le]))
                words.pop_back();
            if (!words.empty() && words.front().first == lb && lb > 0 && is_word_char(t.cps[lb - 1]))
                words.erase(words.begin());
            std::size_t first = words.size(), last = words.size();
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                if (first == words.size() && words[k].second > b) first = k;
                if (words[k].first < e) last = k;
            }
            if (first == words.size() || last == words.size() || first > last) return false;
            for (std::size_t i = first >= 3 ? first - 3 : 0; i <= first; ++i)
                for (std::size_t j = last; j < words.size() && j <= last + 3; ++j)
                    if ((i < first || j > last) &&
                        country_names().count(fold_key(cp_substr(t, text, words[i].first, words[j].second))))
                        return true;
            return false;
        }

        /* A span of a learned recognizer, its pieces merged, where the text holds no personal
           data. A login inside a longer token, a file name ("final.docx" in
           "rapport_final.docx"), a single character, or a common or function word in lower case
           ("month", "for", "<note>"). A person, place or organisation that is a piece of a word
           ("Rem" in "Remplacez"), function words alone ("Le"), a greeting or a field label alone
           ("Nom"), a common word in lower case ("commande"), a name or place that joins a
           capitalized word to lower case words ("Le projet", "Ajoutez la b"), a word in lower
           case, other than a first name, in a text that capitalizes its words ("de congés"), a
           country alone read as an organisation ("Pologne"), a country or a piece of one read
           as a place ("Maroc", "Sud" in "Afrique du Sud"), the civility "M." before a name,
           and a participle that opens a line before "à" ("Fait à Lyon"). */
        bool spurious_learned_span(const std::string& text, const utf8_text& t, std::size_t b, std::size_t e,
                                   pii_type type, bool capitalized_text) const
        {
            using namespace privacy_impl;
            const std::size_t n = t.size();
            if (type == pii_type::identifier)
            {
                if (b > 0 && (is_word_char(t.cps[b - 1]) || t.cps[b - 1] == '_')) return true;
                if (e == b + 1) return true;
                const std::string key = fold_key(cp_substr(t, text, b, e));
                const std::size_t dot = key.rfind('.');
                if (dot != std::string::npos && file_extensions().count(key.substr(dot + 1))) return true;
                bool lower_letters = true;
                for (std::size_t q = b; q < e && lower_letters; ++q)
                    if (!is_letter(t.cps[q]) || is_upper(t.cps[q])) lower_letters = false;
                return lower_letters && (common_words().count(key) || function_words().count(key) ||
                                         field_labels().count(key) || greeting_words().count(key));
            }
            // A number alone read as an address, which no street name follows: "60" in "60 – 100".
            if (type == pii_type::address)
            {
                for (std::size_t q = b; q < e; ++q)
                    if (!is_ascii_digit(t.cps[q]) && t.cps[q] != ' ') return false;
                std::size_t q = e;
                while (q < n && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                return !(q < n && is_letter(t.cps[q]));
            }
            if (type != pii_type::person && type != pii_type::location && type != pii_type::organization)
                return false;
            if (b > 0 && is_letter(t.cps[b - 1]) && is_letter(t.cps[b]) && !is_upper(t.cps[b])) return true;
            if (e < n && is_letter(t.cps[e - 1]) && is_letter(t.cps[e]) && !is_upper(t.cps[e])) return true;
            /* A piece of a short word, which no name of that length is cut from: the capitals of
               a unit glued to its prefix ("UI" in "mUI/L", "Hg" in "mmHg"), a letter of an
               acronym ("L" in "LDL-cholestérol"). */
            auto short_run = [&](std::size_t q) {
                std::size_t rb = q, re = q;
                while (rb > 0 && is_letter(t.cps[rb - 1])) --rb;
                while (re < n && is_letter(t.cps[re])) ++re;
                return re - rb <= 5;
            };
            if ((b > 0 && is_letter(t.cps[b - 1]) && is_letter(t.cps[b]) && short_run(b)) ||
                (e < n && is_letter(t.cps[e]) && is_letter(t.cps[e - 1]) && short_run(e - 1)))
                return true;
            // A place or an organisation of one letter.
            if (type != pii_type::person && e == b + 1) return true;
            const std::vector<std::pair<std::size_t, std::size_t>> words = span_words(t, b, e);
            if (words.empty()) return true;
            std::vector<std::string> keys;
            bool all_function = true;
            for (const auto& w : words)
            {
                keys.push_back(fold_key(cp_substr(t, text, w.first, w.second)));
                if (initial_word(t, w) || !function_words().count(keys.back())) all_function = false;
            }
            if (all_function) return true;
            if (technical_words(text, t, words)) return true;
            // A country alone names no organisation: "Pologne" on the line after a town.
            if (type == pii_type::organization && country_names().count(fold_key(cp_substr(t, text, b, e))))
                return true;
            if (type == pii_type::location && in_country_name(text, t, b, e)) return true;
            // The word before position b and its key, on the same line: "Monsieur" in "Monsieur M. Denis".
            auto word_before = [&](std::size_t q) {
                while (q > 0 && (t.cps[q - 1] == ' ' || t.cps[q - 1] == 0xA0)) --q;
                std::size_t r = q;
                while (r > 0 && is_letter(t.cps[r - 1])) --r;
                return fold_key(cp_substr(t, text, r, q));
            };
            /* "M." before a name is the civility, which the rules report as a title; after
               another civility it is an initial ("Monsieur M. Denis"). */
            if (type == pii_type::person && words.size() == 1 && e == b + 1 && t.cps[b] == 'M' && t.at(e) == '.')
            {
                const std::string before = word_before(b);
                if (!civility_words().count(before) && !person_titles().count(before)) return true;
            }
            // "Page 2", "Page 1 sur 3": the word of a page number.
            if (words.size() == 1 && (keys[0] == "page" || keys[0] == "seite" || keys[0] == "pagina"))
            {
                std::size_t q = e;
                if (q < n && t.cps[q] == ' ') ++q;
                if (q < n && is_ascii_digit(t.cps[q])) return true;
            }
            if (words.size() == 1 && !initial_word(t, words[0]))
            {
                // A common word in lower case is no name ("commande"); capitalized, it can be
                // one ("Paris", "West", "Page").
                const std::string& key = keys[0];
                /* A participle that opens a line before "à": "Fait à Lyon", "Remis à", "Livré
                   au". */
                if (type != pii_type::location && common_words().count(key) && !first_names().count(key) &&
                    !known_organizations().count(key) && word_before(words[0].first).empty())
                {
                    std::size_t q = words[0].second;
                    if (t.at(q) == ' ' || t.at(q) == 0xA0) ++q;
                    std::size_t r = q;
                    while (r < n && is_letter(t.cps[r])) ++r;
                    const std::string next = fold_key(cp_substr(t, text, q, r));
                    if ((next == "a" || next == "au" || next == "aux") && (t.at(r) == ' ' || t.at(r) == 0xA0))
                        return true;
                }
                const char32_t c = name_letter(t, words[0]);
                const bool lower = is_letter(c) && !is_upper(c);
                if (greeting_words().count(key) || field_labels().count(key)) return true;
                if (type != pii_type::location && lower && common_words().count(key)) return true;
                // "lrisJohnson", glued, keeps a capital.
                bool all_lower = lower;
                for (std::size_t q = words[0].first; q < words[0].second && all_lower; ++q)
                    if (is_upper(t.cps[q])) all_lower = false;
                if (type != pii_type::organization && capitalized_text && all_lower &&
                    !(type == pii_type::person && first_names().count(key) && !ambiguous_first_names().count(key)))
                    return true;
            }
            if (type == pii_type::organization)
            {
                /* A common word, a head noun, a trade, a legal form or a title alone names no
                   organisation ("Convocation", "Clinic", "Security", "Maître"), unless it is a known
                   name or a bank named by its network ("Sparkasse"). */
                static const std::unordered_set<std::string> networks = make_word_set(
                    "sparkasse volksbank raiffeisenbank");
                if (words.size() == 1 && !known_organizations().count(keys[0]) && !networks.count(keys[0]) &&
                    (common_words().count(keys[0]) || organization_edge_words().count(keys[0]) ||
                     person_titles().count(keys[0])))
                    return true;
                // "Bruit (quais de": a word in lower case alone, on a line that writes names with capitals.
                if (words.size() == 1 && !known_organizations().count(keys[0]))
                {
                    bool lower_word = is_letter(t.cps[b]);
                    for (std::size_t q = b; q < e && lower_word; ++q)
                        if (is_upper(t.cps[q])) lower_word = false;
                    std::size_t lb = b, le = e;
                    while (lb > 0 && !is_line_break(t.cps[lb - 1])) --lb;
                    while (le < n && !is_line_break(t.cps[le])) ++le;
                    for (std::size_t q = lb; q < le && lower_word; ++q)
                        if ((q < b || q >= e) && is_upper(t.cps[q])) return true;
                }
                /* "the embassy of Spain", "le consulat de Pologne": in a text that capitalizes its
                   names, a common word in lower case opens none, unless it is a head noun or a
                   particle ("ministère de la Santé", "de Mulder Vastgoed V.O.F."). */
                const char32_t c = name_letter(t, words[0]);
                if (capitalized_text && words.size() >= 2 && is_letter(c) && !is_upper(c) &&
                    common_words().count(keys[0]) && !organization_heads().count(keys[0]) &&
                    !person_particles().count(keys[0]))
                    return true;
                // "Le logiciel Excel": an article and a lower case word open no name.
                return words.size() >= 2 && function_words().count(keys[0]) && !is_upper(t.cps[words[1].first]) &&
                       is_letter(t.cps[words[1].first]);
            }
            bool capital = false, lower = false;
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                if (initial_word(t, words[k])) continue;
                const char32_t c = name_letter(t, words[k]);
                if (!is_letter(c)) continue;
                if (is_upper(c)) capital = true;
                else if (!name_particle(type, keys[k])) lower = true;
            }
            return capital && lower;
        }

        struct learned_piece
        {
            std::size_t b, e;       // code point indices
            pii_type type;
            double confidence;
            int parts;
        };

        /* The boundaries of the organisations a recognizer reads, made consistent with those of
           the annotation. The pieces of one name are joined: pieces of an organisation that meet
           or are one space apart ("Club" and "Dexiq"), that a comma, a bracket or a hyphen
           separates before a head noun, a legal form, a trade or a partner word ("DUNMORE CO.,"
           and "LTD."), that an ampersand or a conjunction separates next to one ("Thornfield y"
           and "Asociados"), or that such words alone separate ("Zenane", "Oil and", "Gas Nigeria
           Limited"). Head nouns alone join the person or the place they name ("Klinikum" and
           "Hambourg"), legal forms, trades and partner words alone the name before them
           ("Calance" and "Partners"), a legal form or a particle read as a person opens the name
           ("TOV" and "Talion", "Van" and "Damme Logistiek B.V."), and partners after a conjunction
           close it ("BARAN" and "I WSPÓLNICY S.A."). A person or a place read with a legal form, a
           partner word, a trade or the head noun of a firm or a town is an organisation ("Lumelle
           Bt.", "Stadt Karlsruhe"), and a place in brackets or after a comma that follows a legal
           form or a head noun is a place ("Larkspur ltée, São Paulo"). A capitalized head noun,
           trade or particle just before the name opens it ("Entreprise Meyer", "City of Sydney
           Council", "Le Corre Holdings Inc."), a legal form or a partner word just after it
           closes it ("Armium a.s.", "Kerexa SARL AU", "Foxglove A/S"), and a name made of head
           nouns or legal forms alone takes the capitalized words around it ("TALIS CLINIC", "Club
           Deportivo Madrid", "TOV Briora", "Instituto Désiré Carrasco"). A legal form ends the
           name ("Solix FZE" in "Solix FZE Save"), and in a line of capitals the name starts after
           the last word of the sentence ("ELEXA LTÉE" in "DIRIGE DEPUIS DIX ANS ELEXA LTÉE").
           Greetings, articles and function words that open or close the name are left out, and so
           are common words, in lower case or in a line of capitals. A legal form keeps its own
           period ("S.A.", "Pte. Ltd.") and leaves out the period that ends the sentence ("Goelia
           SNC."). */
        void normalize_learned_organizations(const std::string& text, const utf8_text& t,
                                             std::vector<learned_piece>& pieces) const
        {
            using namespace privacy_impl;
            typedef std::pair<std::size_t, std::size_t> token;
            const std::size_t n = t.size();
            const auto& edge = organization_edge_words();
            const auto& forms = organization_forms();
            const auto& partners = organization_partner_words();
            static const std::unordered_set<std::string> connectors = make_word_set(
                "y i e et and und of de du des di da do del della dei der van von");
            static const std::unordered_set<std::string> particles = make_word_set(
                "van von der den ter ten le la de du del di da dos");
            static const std::unordered_set<std::string> lower_particles = make_word_set("van von der den ter ten");
            // Abbreviations written with their period: "Inc.", "Ltda.", "Kft.", "Co.", "Ets.".
            static const std::unordered_set<std::string> period_forms = make_word_set(
                "inc corp kft bt zrt lda ltda co pte pvt cie bros sti ets ste univ assoc mfg ltd");
            // Legal forms written before the name: "TOV Briora", "SC Novion", "EURL Xanelle".
            static const std::unordered_set<std::string> prefix_forms = make_word_set(
                "tov ooo oao zao pao ao sc sas sasu sarl eurl sci snc selarl pt sia uab");
            // Articles that open the sentence rather than the name: "La Police nationale".
            static const std::unordered_set<std::string> articles = make_word_set(
                "le la les the el il lo der die das het");
            // The head nouns of a town and of its administration: "Stadt Karlsruhe", "Ålesund kommune".
            static const std::unordered_set<std::string> town_heads = make_word_set(
                "stadt kommune kommun gemeente gemeinde municipio ayuntamiento comune commune mairie prefeitura "
                "belediyesi landratsamt magistrat primaria");
            // Those written after the name of the town, in any case: "Stockholm kommun", "Oslo kommune".
            static const std::unordered_set<std::string> town_suffixes = make_word_set("kommun kommune belediyesi");
            // Head nouns of institutions named after a person: "Instituto Désiré Carrasco", "Lycée Victor Hugo".
            static const std::unordered_set<std::string> eponym_heads = make_word_set(
                "instituto institut institute istituto liceo lycee college colegio ecole escuela escola scuola "
                "universite universidad hopital hospital clinique clinica fondation fundacion fondazione "
                "foundation stiftung centre center museo musee museum teatro conservatoire conservatorio "
                "academie accademia");
            // Words before a name that its particle opens, as a value or after a preposition: "chez de Bruijn et Fils".
            static const std::unordered_set<std::string> openers = make_word_set(
                "chez contre entre avec rejoint rejoindre quitte quitter from at with joins joined de du des of");

            /* The words of a name: letters, digits and the periods, hyphens, apostrophes, slashes
               and ampersands inside or after them ("S.A.", "Inc.", "A/S", "AT&T"); the key drops
               the periods and the slashes. */
            auto tokens = [&](std::size_t b, std::size_t e) {
                std::vector<token> out;
                for (std::size_t i = b; i < e;)
                {
                    while (i < e && !is_word_char(t.cps[i])) ++i;
                    std::size_t j = i;
                    while (j < e && (is_word_char(t.cps[j]) || t.cps[j] == '.' || t.cps[j] == '&' ||
                                     ((is_hyphen(t.cps[j]) || is_apostrophe(t.cps[j]) || t.cps[j] == '/') &&
                                      j + 1 < e && is_word_char(t.cps[j + 1]))))
                        ++j;
                    if (j > i) out.push_back(std::make_pair(i, j));
                    i = j;
                }
                return out;
            };
            auto key_of = [&](const token& w) {
                std::string k = fold_key(cp_substr(t, text, w.first, w.second));
                k.erase(std::remove_if(k.begin(), k.end(), [](char c) { return c == '.' || c == '/'; }), k.end());
                return k;
            };
            auto inner_period = [&](const token& w) {
                for (std::size_t q = w.first; q + 1 < w.second; ++q)
                    if (t.cps[q] == '.' && is_word_char(t.cps[q + 1])) return true;
                return false;
            };
            auto capital = [&](std::size_t b) { return is_letter(t.cps[b]) && is_upper(t.cps[b]); };
            auto lower_word = [&](const token& w) { return is_letter(t.cps[w.first]) && !is_upper(t.cps[w.first]); };
            auto common = [&](const std::string& key) {
                return common_words().count(key) != 0 || function_words().count(key) != 0;
            };
            // A short legal form written as one: "AS", "a.s.", "A/S", "Oy", "Bt."; "As" and "Ab" open sentences.
            auto short_form = [&](const token& w, const std::string& key) {
                if (!organization_short_forms().count(key)) return false;
                bool lower = false, period = false;
                for (std::size_t q = w.first; q < w.second; ++q)
                {
                    if (t.cps[q] == '.' || t.cps[q] == '/') period = true;
                    else if (is_letter(t.cps[q]) && !is_upper(t.cps[q])) lower = true;
                }
                if (period || !lower) return true;
                return capital(w.first) && (key == "oy" || key == "bt" || key == "ky" || key == "sti");
            };
            auto form_word = [&](const token& w) {
                const std::string key = key_of(w);
                return forms.count(key) != 0 || short_form(w, key);
            };
            auto edge_word = [&](const token& w) {
                const std::string key = key_of(w);
                return edge.count(key) != 0 || short_form(w, key);
            };
            // A word of a firm, which no place holds: a legal form, a partner word, a head noun, a trade.
            auto firm_word = [&](const token& w) {
                const std::string key = key_of(w);
                return form_word(w) || partners.count(key) || organization_heads().count(key) ||
                       organization_sectors().count(key) || organization_trade_words().count(key);
            };
            // The words of compass points and sizes that open the names of places: "East Bradby".
            static const std::unordered_set<std::string> place_words = make_word_set(
                "north south east west upper lower new old great little saint sainte st");
            // The letter after the elided article or preposition that opens a word: "d'Entreprise", "l'Office".
            auto after_elision = [&](const token& w) {
                if (w.second < w.first + 3 || !is_apostrophe(t.cps[w.first + 1]) || !is_letter(t.cps[w.first + 2]))
                    return w.first;
                const char32_t c = to_lower(t.cps[w.first]);
                return c == 'd' || c == 'l' ? w.first + 2 : w.first;
            };
            auto all_edge = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                for (const auto& x : w)
                    if (!edge_word(x)) return false;
                return !w.empty();
            };
            // Head nouns and trades, without a legal form or a partner word: "Club Deportivo", "Klinikum".
            auto all_heads = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                for (const auto& x : w)
                {
                    const std::string k = key_of(x);
                    if (!edge.count(k) || forms.count(k) || partners.count(k)) return false;
                }
                return !w.empty();
            };
            auto all_prefix_forms = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                for (const auto& x : w)
                    if (!prefix_forms.count(key_of(x)) || lower_word(x)) return false;
                return !w.empty();
            };
            // A head noun of an institution and at most two words of the name of a person: "Instituto Désiré".
            auto eponym = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                if (w.size() < 2 || w.size() > 3 || !eponym_heads.count(key_of(w.front()))) return false;
                for (std::size_t i = 1; i < w.size(); ++i)
                    if (!capital(w[i].first) || common(key_of(w[i])) || edge_word(w[i])) return false;
                return true;
            };
            auto first_edge = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                return !w.empty() && edge_word(w.front());
            };
            auto last_edge = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                return !w.empty() && edge_word(w.back());
            };
            auto first_key = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                return w.empty() ? std::string() : key_of(w.front());
            };
            auto last_key = [&](const learned_piece& p) {
                const auto w = tokens(p.b, p.e);
                return w.empty() ? std::string() : key_of(w.back());
            };
            auto one_space = [&](std::size_t a, std::size_t b) {
                return b == a + 1 && (t.cps[a] == ' ' || t.cps[a] == 0xA0);
            };
            // One space, or an ampersand between two spaces: the gap between two words of a trade.
            auto word_gap = [&](std::size_t a, std::size_t b) {
                return one_space(a, b) || (b == a + 3 && one_space(a, a + 1) && t.cps[a + 1] == '&' &&
                                           one_space(a + 2, b));
            };
            // The gap between two pieces of a line, without its spaces: "", ",", "&", "(", "y".
            auto gap_key = [&](std::size_t a, std::size_t b) {
                std::string g;
                for (std::size_t q = a; q < b; ++q)
                    if (!(t.cps[q] == ' ' || t.cps[q] == 0xA0)) utf8_append(g, t.cps[q]);
                return fold_key(g);
            };
            // Words of a trade and conjunctions between two pieces: " Oil and ", " & Gas ".
            auto edge_gap = [&](std::size_t a, std::size_t b) {
                if (b < a + 3 || b - a > 24 || !one_space(a, a + 1) || !one_space(b - 1, b)) return false;
                const auto w = tokens(a + 1, b - 1);
                if (w.empty() || w.size() > 3 || w.front().first != a + 1 || w.back().second != b - 1) return false;
                bool trade = false;
                for (std::size_t i = 0; i < w.size(); ++i)
                {
                    if (i > 0 && !word_gap(w[i - 1].second, w[i].first)) return false;
                    if (capital(w[i].first) && edge_word(w[i])) trade = true;
                    else if (!connectors.count(key_of(w[i]))) return false;
                }
                return trade;
            };
            // The share of capitals among the letters of the line around pos.
            auto line_case = [&](std::size_t pos, std::size_t& upper) {
                std::size_t b = pos, e = pos, letters = 0;
                upper = 0;
                while (b > 0 && !is_line_break(t.cps[b - 1]) && pos - b < 200) --b;
                while (e < n && !is_line_break(t.cps[e]) && e - pos < 200) ++e;
                for (std::size_t q = b; q < e; ++q)
                    if (is_letter(t.cps[q]))
                    {
                        ++letters;
                        if (is_upper(t.cps[q])) ++upper;
                    }
                return letters;
            };
            // A line written in capitals: most of its letters are.
            auto caps_line = [&](std::size_t pos) {
                std::size_t upper = 0;
                const std::size_t letters = line_case(pos, upper);
                return letters >= 8 && upper * 10 >= letters * 8;
            };

            /* A person or a place read with a legal form, a partner word or the head noun of a
               town, or a person read with the head noun of a firm or a trade, is an organisation.
               A piece of an organisation of several words takes the words it cuts, unless a change
               of case marks a word glued to another and the whole word is no legal form ("Hollis
               Ap" in "Hollis ApS"), and leaves out the function words in lower case that open or
               close it ("S.L.U. to"). */
            for (std::size_t k = 0; k < pieces.size(); ++k)
            {
                learned_piece& p = pieces[k];
                const std::size_t floor = k > 0 ? pieces[k - 1].e : 0;
                const std::size_t ceiling = k + 1 < pieces.size() ? pieces[k + 1].b : n;
                if (p.type == pii_type::person || p.type == pii_type::location)
                {
                    const auto w = tokens(p.b, p.e);
                    bool organisation = false;
                    for (std::size_t i = 0; i < w.size() && !organisation; ++i)
                    {
                        const std::string key = key_of(w[i]);
                        // "Lumelle Bt.", "Elance Oyj", "TOV"; "BAKKER Sem" is a person.
                        bool written_form = false;
                        if (i > 0 && i + 1 == w.size() && form_word(w[i]) && !first_names().count(key))
                        {
                            bool lower = false;
                            for (std::size_t q = w[i].first; q < w[i].second; ++q)
                                if (is_letter(t.cps[q]) && !is_upper(t.cps[q])) lower = true;
                            written_form = inner_period(w[i]) || t.cps[w[i].second - 1] == '.' || !lower ||
                                           (forms.count(key) && key.size() >= 3) || short_form(w[i], key);
                        }
                        if (town_heads.count(key) || partners.count(key) || written_form ||
                            (w.size() == 1 && prefix_forms.count(key) && !lower_word(w[i]) && !inner_period(w[i])))
                            organisation = true;
                        else if (p.type == pii_type::person && key != "ste" &&
                                 (organization_heads().count(key) || organization_trade_words().count(key)))
                            organisation = true;
                    }
                    // "Ålesund kommune", "Stockholm kommun".
                    if (!organisation && p.type == pii_type::location && p.e + 1 < ceiling && one_space(p.e, p.e + 1))
                    {
                        const auto x = tokens(p.e + 1, std::min(ceiling, p.e + 20));
                        if (!x.empty() && x.front().first == p.e + 1 && town_suffixes.count(key_of(x.front())))
                        {
                            p.e = x.front().second;
                            organisation = true;
                        }
                    }
                    if (organisation) p.type = pii_type::organization;
                }
                if (p.type != pii_type::organization) continue;
                const bool several = tokens(p.b, p.e).size() >= 2;
                while (several && p.b > floor && is_letter(t.cps[p.b - 1]) && is_letter(t.cps[p.b]) &&
                       !(is_lower(t.cps[p.b - 1]) && is_upper(t.cps[p.b])))
                    --p.b;
                if (several && p.e < ceiling && is_letter(t.cps[p.e - 1]) && is_letter(t.cps[p.e]))
                {
                    std::size_t s = p.e, e = p.e;
                    while (s > p.b && is_letter(t.cps[s - 1])) --s;
                    while (e < ceiling && is_letter(t.cps[e])) ++e;
                    const bool glued = is_lower(t.cps[p.e - 1]) && is_upper(t.cps[p.e]);
                    if (!glued || forms.count(fold_key(cp_substr(t, text, s, e)))) p.e = e;
                }
                auto w = tokens(p.b, p.e);
                while (w.size() > 1 && lower_word(w.back()) && function_words().count(key_of(w.back())))
                {
                    w.pop_back();
                    p.e = w.back().second;
                }
                while (w.size() > 1 && lower_word(w.front()) && function_words().count(key_of(w.front())) &&
                       !particles.count(key_of(w.front())))
                {
                    w.erase(w.begin());
                    p.b = w.front().first;
                }
                // "l'Auto-école du Steir", "d'Harmonie Mutuelle": the elided article stays outside the name.
                if (p.e > p.b + 2 && (t.cps[p.b] == 'l' || t.cps[p.b] == 'd') && is_apostrophe(t.cps[p.b + 1]) &&
                    is_letter(t.cps[p.b + 2]) && is_upper(t.cps[p.b + 2]) && !is_letter(t.before(p.b)))
                    p.b += 2;
            }

            // The pieces of one name.
            for (bool changed = true; changed;)
            {
                changed = false;
                for (std::size_t k = 0; k + 1 < pieces.size() && !changed; ++k)
                {
                    learned_piece& p = pieces[k];
                    const learned_piece& q = pieces[k + 1];
                    const bool org_p = p.type == pii_type::organization, org_q = q.type == pii_type::organization;
                    const bool named_p = p.type == pii_type::person || p.type == pii_type::location;
                    const bool named_q = q.type == pii_type::person || q.type == pii_type::location;
                    if ((!org_p && !org_q) || q.b < p.e || q.b - p.e > 24 || !same_line(t, p.e, q.b)) continue;
                    const bool spaced = one_space(p.e, q.b), close = q.b - p.e <= 5;
                    const std::string g = close ? gap_key(p.e, q.b) : std::string();
                    bool join = false;
                    if (org_p && org_q)
                    {
                        join = q.b == p.e || spaced || (close && (g == "," || g == "(" || g == "-") && first_edge(q)) ||
                               (close && (g == "&" || connectors.count(g)) && (first_edge(q) || last_edge(p))) ||
                               edge_gap(p.e, q.b);
                    }
                    else if (org_p && named_q && spaced)
                    {
                        join = all_heads(p) || all_edge(q) || (q.type == pii_type::person && eponym(p));
                    }
                    else if (org_p && named_q && close)
                    {
                        // "Câmara Municipal de Porto", "Universidad de Salamanca", "SCP Payet-Gautier".
                        join = (all_heads(p) && (g == "de" || g == "of" || g == "di" || g == "do" || g == "da" ||
                                                 g == "del" || g == "du" || g == "des")) ||
                               (q.b == p.e + 1 && is_hyphen(t.cps[p.e]));
                    }
                    else if (named_p && org_q && spaced)
                    {
                        const std::size_t words_p = tokens(p.b, p.e).size(), words_q = tokens(q.b, q.e).size();
                        const std::string k_p = first_key(p);
                        join = all_edge(q) ||
                               (words_p == 1 && (edge.count(k_p) || particles.count(k_p) ||
                                                 (words_q >= 2 && forms.count(last_key(q)))));
                    }
                    else if (named_p && org_q && close)
                    {
                        // "BARAN I WSPÓLNICY S.A.", "Martin & Fils".
                        join = p.type == pii_type::person && (g == "&" || connectors.count(g)) &&
                               partners.count(first_key(q));
                    }
                    if (!join) continue;
                    p.e = q.e;
                    p.type = pii_type::organization;
                    p.confidence = std::min(p.confidence, q.confidence);
                    p.parts += q.parts;
                    pieces.erase(pieces.begin() + static_cast<long>(k + 1));
                    changed = true;
                }
            }

            // A place after the legal form or the head noun of a name: "Larkspur ltée, São Paulo".
            for (std::size_t k = 0; k + 1 < pieces.size(); ++k)
            {
                const learned_piece& p = pieces[k];
                learned_piece& q = pieces[k + 1];
                if (p.type != pii_type::organization || q.type != pii_type::organization || q.b < p.e ||
                    q.b - p.e > 3 || !same_line(t, p.e, q.b))
                    continue;
                const std::string g = gap_key(p.e, q.b);
                if ((g != "," && g != "(") || !last_edge(p)) continue;
                const auto w = tokens(q.b, q.e);
                bool place = !w.empty() && w.size() <= 3;
                for (const auto& x : w)
                {
                    const std::string key = key_of(x);
                    if (!capital(x.first) || firm_word(x) || (common(key) && !place_words.count(key))) place = false;
                }
                if (place) q.type = pii_type::location;
            }

            for (std::size_t k = 0; k < pieces.size(); ++k)
            {
                learned_piece& p = pieces[k];
                if (p.type != pii_type::organization) continue;
                const std::size_t floor = k > 0 ? pieces[k - 1].e : 0;
                const std::size_t ceiling = k + 1 < pieces.size() ? pieces[k + 1].b : n;
                const std::size_t limit = std::min(ceiling, n);
                const bool caps = caps_line(p.b), lower = lower_case_line(t, p.b);
                // The word that ends one space before position b, without its elided article ("d'Entreprise"),
                // or an empty range.
                auto word_before = [&](std::size_t b) {
                    if (b < 2 || !one_space(b - 1, b)) return std::make_pair(b, b);
                    std::size_t e = b - 1, s = e;
                    while (s > floor && (is_word_char(t.cps[s - 1]) ||
                                         ((is_hyphen(t.cps[s - 1]) || is_apostrophe(t.cps[s - 1])) && s - 1 > floor &&
                                          is_word_char(t.cps[s - 2]))))
                        --s;
                    // A word that the previous piece holds in part is no word of its own.
                    if (s == floor && floor > 0 && is_word_char(t.cps[floor - 1])) return std::make_pair(b, b);
                    return std::make_pair(after_elision(std::make_pair(s, e)), e);
                };
                // A value or a preposition before position b: "Employeur : de Mulder", "chez de Bruijn".
                auto opens_value = [&](std::size_t b) {
                    std::size_t q = b;
                    while (q > 0 && (t.cps[q - 1] == ' ' || t.cps[q - 1] == 0xA0)) --q;
                    if (q == 0) return true;
                    const char32_t c = t.cps[q - 1];
                    if (is_line_break(c) || c == ':' || c == '(' || c == ',' || c == ';' || is_hyphen(c) || c == '\t')
                        return true;
                    std::size_t r = q;
                    while (r > 0 && is_letter(t.cps[r - 1])) --r;
                    return r < q && openers.count(fold_key(cp_substr(t, text, r, q))) != 0;
                };

                /* A person read with an organisation, an elision between them: "Typhaine Marnier
                   d'Abeona S.r.l.", where a legal form, a head noun or a trade follows the elision. */
                {
                    const auto w = tokens(p.b, p.e);
                    for (std::size_t i = 2; i < w.size() && i <= 3; ++i)
                    {
                        const std::size_t a = after_elision(w[i]);
                        if (a == w[i].first) continue;
                        bool person = capital(a), firm = false;
                        for (std::size_t j = 0; j < i && person; ++j)
                            if (!capital(w[j].first) || edge_word(w[j]) || common(key_of(w[j]))) person = false;
                        for (std::size_t j = i + 1; j < w.size(); ++j)
                            if (firm_word(w[j])) firm = true;
                        if (person && firm) p.b = a;
                        break;
                    }
                }
                // In a line of capitals, the name starts after the last word of the sentence.
                if (caps)
                {
                    const auto w = tokens(p.b, p.e);
                    std::size_t cut = 0;
                    for (std::size_t i = 0; i + 1 < w.size(); ++i)
                    {
                        const std::string key = key_of(w[i]);
                        if (!common(key) || edge_word(w[i]) || connectors.count(key) ||
                            (i > 0 && connectors.count(key_of(w[i - 1]))))
                            continue;
                        // "LE RONNEC SERVICES": a particle before a family name.
                        if (particles.count(key) && !common(key_of(w[i + 1])) && !edge_word(w[i + 1])) continue;
                        for (std::size_t j = i + 1; j < w.size(); ++j)
                        {
                            const std::string kj = key_of(w[j]);
                            if (!common(kj) && !connectors.count(kj))
                            {
                                cut = i + 1;
                                break;
                            }
                        }
                    }
                    if (cut > 0) p.b = w[cut].first;
                }
                /* A capitalized head noun, a trade or a particle just before the name; a head noun and
                   its conjunction ("City of"); "Le" before a family name, even at the start of a
                   sentence ("Le Corre Holdings Inc."); a particle in lower case after a value or a
                   preposition ("chez de Bruijn et Fils"). A name made of head nouns, trades or legal
                   forms alone takes the words before it, and then their particle ("LE RONNEC" before
                   "SERVICES S.À R.L."). */
                for (int pass = 0; pass < 2; ++pass)
                {
                    for (int step = 0; step < 3; ++step)
                    {
                        const auto w = word_before(p.b);
                        if (w.first == w.second || w.first < floor) break;
                        const std::string key = key_of(w);
                        const bool head = (capital(w.first) || (lower && lower_word(w))) && edge.count(key) &&
                                          !partners.count(key);
                        bool particle = false;
                        if (particles.count(key))
                        {
                            const auto first = tokens(p.b, p.e);
                            const bool family = !first.empty() && capital(first.front().first) &&
                                                !common(key_of(first.front())) && !edge_word(first.front()) &&
                                                !first_names().count(key_of(first.front()));
                            if (!capital(w.first))
                                particle = lower_particles.count(key) ||
                                           ((key == "de" || key == "du") && family && opens_value(w.first));
                            else
                                particle = !at_sentence_start(t, w.first) || family;
                        }
                        // "City of", "Pharmacie de la", "tribunal de commerce de": a head noun and its conjunction.
                        if (!head && (!particle || !capital(w.first)) && capital(w.first) == caps &&
                            (connectors.count(key) || articles.count(key)))
                        {
                            auto h = word_before(w.first);
                            if (articles.count(key) && h.first < h.second && h.first >= floor &&
                                connectors.count(key_of(h)))
                                h = word_before(h.first);
                            else if (!connectors.count(key))
                                h = std::make_pair(w.first, w.first);
                            const std::string hk = h.first < h.second && h.first >= floor ? key_of(h) : std::string();
                            if (!hk.empty() && (capital(h.first) || lower) && edge.count(hk) && !partners.count(hk) &&
                                !forms.count(hk))
                            {
                                p.b = h.first;
                                continue;
                            }
                        }
                        if (!head && !particle) break;
                        p.b = w.first;
                    }
                    if (pass == 0 && all_edge(p))
                    {
                        for (int step = 0; step < 2; ++step)
                        {
                            const auto w = word_before(p.b);
                            if (w.first == w.second || w.first < floor) break;
                            const std::string key = key_of(w);
                            if (greeting_words().count(key) || person_titles().count(key) ||
                                civility_words().count(key))
                                break;
                            bool named;
                            if (capital(w.first))
                                named = !common(key) || (step == 0 && !caps && function_words().count(key) &&
                                                         !at_sentence_start(t, w.first));
                            else
                                named = lower && !common(key);
                            if (!named) break;
                            p.b = w.first;
                        }
                    }
                }
                // After the name: its bracket, the period of its abbreviation, a legal form or a partner
                // word, and the name that follows head nouns or legal forms alone.
                bool name_after = false;
                for (int round = 0; round < 8; ++round)
                {
                    std::size_t open = 0, shut = 0;
                    for (std::size_t q = p.b; q < p.e; ++q)
                    {
                        if (t.cps[q] == '(') ++open;
                        else if (t.cps[q] == ')') ++shut;
                    }
                    if (open > shut && p.e < limit && t.cps[p.e] == ')')
                    {
                        ++p.e;
                        continue;
                    }
                    const auto w = tokens(p.b, p.e);
                    if (w.empty()) break;
                    const token last = w.back();
                    if (p.e < limit && t.cps[p.e] == '.' && last.second == p.e && t.cps[p.e - 1] != '.' &&
                        (inner_period(last) || period_forms.count(key_of(last))))
                    {
                        ++p.e;
                        continue;
                    }
                    // "Oakridge Co., Ltd.": a legal form after the comma of another.
                    if (p.e + 2 < limit && t.cps[p.e] == ',' && one_space(p.e + 1, p.e + 2) && form_word(last))
                    {
                        const auto x = tokens(p.e + 2, std::min(limit, p.e + 20));
                        if (!x.empty() && x.front().first == p.e + 2)
                        {
                            const std::string key = key_of(x.front());
                            if (key == "ltd" || key == "limited" || key == "inc" || key == "llc")
                            {
                                p.e = x.front().second;
                                continue;
                            }
                        }
                    }
                    if (p.e + 1 >= limit || !one_space(p.e, p.e + 1)) break;
                    const auto x = tokens(p.e + 1, std::min(limit, p.e + 60));
                    if (x.empty() || x.front().first != p.e + 1) break;
                    const token& f = x.front();
                    const std::string key = key_of(f);
                    if (form_word(f) || partners.count(key) || (key == "au" && !lower_word(f) && form_word(last)) ||
                        town_suffixes.count(key))
                    {
                        p.e = f.second;
                        continue;
                    }
                    /* "Weber Bâtiment", "Le Dunnec Conseil", "penzance city council": a trade or a head
                       noun after a name, which common words alone do not make ("Crown Court"). */
                    bool named = false;
                    for (const auto& y : w)
                        if (!common(key_of(y)) && !edge_word(y)) named = true;
                    if (named && edge.count(key) && !form_word(last) && (capital(f.first) || (lower && lower_word(f))))
                    {
                        p.e = f.second;
                        continue;
                    }
                    // "Private Limited".
                    if ((key == "private" || key == "public") && capital(f.first) && x.size() > 1 &&
                        one_space(f.second, x[1].first) && key_of(x[1]) == "limited")
                    {
                        p.e = x[1].second;
                        continue;
                    }
                    if (name_after) break;
                    name_after = true;
                    const bool eponymous = eponym(p) || (w.size() == 1 && eponym_heads.count(key_of(last)));
                    if (!all_heads(p) && !all_prefix_forms(p) && !eponymous) break;
                    const std::size_t most = eponymous ? 3 - w.size() : 2;
                    std::size_t e = p.e, taken = 0;
                    for (std::size_t i = 0; i < x.size() && taken < most; ++i)
                    {
                        if (i > 0 && !one_space(x[i - 1].second, x[i].first)) break;
                        const std::string kx = key_of(x[i]);
                        const bool word = capital(x[i].first) ? !common(kx) && !greeting_words().count(kx) &&
                                                                    !person_titles().count(kx) && !form_word(x[i])
                                                              : lower && !common(kx);
                        if (!word) break;
                        e = x[i].second;
                        ++taken;
                    }
                    if (e == p.e) break;
                    p.e = e;
                }
                // A legal form ends the name: "Solix FZE" in "Solix FZE Save", "S.L.U." before "to Galis Ltd".
                learned_piece rest = p;
                rest.b = rest.e = 0;
                {
                    const auto w = tokens(p.b, p.e);
                    for (std::size_t i = 1; i + 1 < w.size(); ++i)
                    {
                        if (!form_word(w[i])) continue;
                        const std::string next = key_of(w[i + 1]);
                        if (form_word(w[i + 1]) || edge_word(w[i + 1]) || partners.count(next) || next == "au" ||
                            (connectors.count(next) && i + 2 < w.size() && form_word(w[i + 2])))
                            continue;
                        bool bracket = false, comma = false;
                        for (std::size_t q = w[i].second; q < w[i + 1].first; ++q)
                        {
                            if (t.cps[q] == '(') bracket = true;
                            if (t.cps[q] == ',') comma = true;
                        }
                        // "Galatis Mining (Pty) Ltd": a bracketed legal form belongs to the name.
                        if (bracket && form_word(w[i + 1])) continue;
                        std::size_t r = i + 1;
                        while (r < w.size() && function_words().count(key_of(w[r])) && lower_word(w[r])) ++r;
                        bool organisation = false, place = (bracket || comma) && r < w.size() && w.size() - r <= 3;
                        for (std::size_t j = r; j < w.size(); ++j)
                        {
                            const std::string key = key_of(w[j]);
                            if (form_word(w[j]) || partners.count(key)) organisation = true;
                            if (!capital(w[j].first) || firm_word(w[j]) || (common(key) && !place_words.count(key)))
                                place = false;
                        }
                        if (r < w.size() && capital(w[r].first) && (organisation || place))
                        {
                            rest.b = w[r].first;
                            rest.e = w.back().second;
                            rest.type = organisation ? pii_type::organization : pii_type::location;
                        }
                        p.e = w[i].second;
                        break;
                    }
                }
                /* Words that open or close the sentence around the name; neighbour is the word after
                   a leading word, the word before a trailing one. */
                auto droppable = [&](const token& w, const token& neighbour, bool leading) {
                    const std::string key = key_of(w);
                    if (key.empty() || edge_word(w)) return false;
                    if (greeting_words().count(key)) return true;
                    const bool lower_w = !capital(w.first);
                    if (leading)
                    {
                        const std::string kn = key_of(neighbour);
                        // An acronym after an article is no family name: "La CPAM de Saint-Brieuc".
                        bool acronym = !caps && neighbour.second - neighbour.first >= 2;
                        for (std::size_t q = neighbour.first; q < neighbour.second && acronym; ++q)
                            if (is_letter(t.cps[q]) && !is_upper(t.cps[q])) acronym = false;
                        const bool name_next = !common(kn) && !edge_word(neighbour) && !acronym;
                        // "LE RONNEC", "de Mulder", "Le Corre": a particle before a family name.
                        if (particles.count(key) && name_next) return false;
                        // "La Police nationale": an article before a common word.
                        if (articles.count(key) && !lower_w && !caps && !name_next) return true;
                    }
                    // "Kerexa SARL AU".
                    else if (key == "au" && !lower_w && form_word(neighbour))
                        return false;
                    if (function_words().count(key)) return lower_w || caps || !leading;
                    return common_words().count(key) && (lower_w || caps) && !first_names().count(key);
                };
                auto w = tokens(p.b, p.e);
                if (w.empty()) continue;
                std::size_t first = 0, last = w.size() - 1;
                while (first < last && droppable(w[first], w[first + 1], true)) ++first;
                while (last > first && droppable(w[last], w[last - 1], false)) --last;
                // A word that a hyphen continues after the name: "Calovia Finance Saint" in "Saint-Ouen-en-Retz".
                if (last > first + 1 && w[last].second < n && is_hyphen(t.cps[w[last].second]) &&
                    w[last].second + 1 < n && is_letter(t.cps[w[last].second + 1]))
                    --last;
                p.b = w[first].first;
                // The period of a legal form, and none of the sentence.
                std::size_t end = w[last].second;
                const std::string key = key_of(w[last]);
                const bool keeps_period = inner_period(w[last]) || period_forms.count(key);
                while (end > w[last].first + 1 && t.cps[end - 1] == '.' && t.cps[end - 2] == '.') --end;
                if (end > w[last].first && t.cps[end - 1] == '.' && !keeps_period) --end;
                if (end < limit && t.cps[end] == '.' && t.cps[end - 1] != '.' && keeps_period) ++end;
                p.e = end;
                // A bracket the name opens closes it: "Xanys Holdings (Pty) Ltd", "Lumivia (France)".
                std::size_t open = 0, shut = 0;
                for (std::size_t q = p.b; q < p.e; ++q)
                {
                    if (t.cps[q] == '(') ++open;
                    else if (t.cps[q] == ')') ++shut;
                }
                if (open > shut && p.e < limit && t.cps[p.e] == ')') ++p.e;
                if (rest.e > rest.b) pieces.insert(pieces.begin() + static_cast<long>(k + 1), rest);
            }
        }

        void add_recognized(const std::string& text, const utf8_text& t, const std::vector<pii_span>& spans,
                            double min_confidence, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            typedef learned_piece piece;
            std::vector<piece> found;
            const std::size_t bytes = t.offsets.back();
            for (const pii_span& s : spans)
            {
                if (s.end <= s.begin || s.end > bytes || s.confidence < min_confidence) continue;
                const pii_type type = s.type != pii_type::none ? s.type : pii_type_from_label(s.label);
                if (type == pii_type::none || !enabled(type)) continue;
                // A byte offset inside a code point takes the whole code point.
                const std::size_t b = static_cast<std::size_t>(
                    std::upper_bound(t.offsets.begin(), t.offsets.end(), s.begin) - t.offsets.begin()) - 1;
                const std::size_t e = static_cast<std::size_t>(
                    std::lower_bound(t.offsets.begin(), t.offsets.end(), s.end) - t.offsets.begin());
                if (e <= b || e > t.size()) continue;
                found.push_back(piece{ b, e, type, s.confidence, 1 });
            }
            std::sort(found.begin(), found.end(), [](const piece& x, const piece& y) { return x.b < y.b; });
            /* The gap between two pieces of one name: spaces and hyphens, at most two; the
               pieces of a login, and those of an address cut inside a word, meet ("v" and
               "wojcikski"). */
            auto name_gap = [&](const piece& m, const piece& p) {
                if (m.type == pii_type::identifier && p.type == pii_type::identifier) return p.b == m.e;
                // Two pieces of an address that meet inside a word: "u" and "l. Długa 5".
                if (m.type == pii_type::address && p.type == pii_type::address)
                    return p.b == m.e && p.b > 0 && is_letter(t.cps[p.b - 1]) && is_letter(t.cps[p.b]);
                bool gap = m.type == p.type && (p.type == pii_type::person || p.type == pii_type::location) &&
                           p.b >= m.e && p.b - m.e <= 2;
                for (std::size_t k = m.e; k < p.b && gap; ++k)
                    if (!(t.cps[k] == ' ' || is_hyphen(t.cps[k]))) gap = false;
                return gap;
            };
            /* A capitalized word read as an organisation one space before another piece of one may
               open its name: "Le" and "Moigne & Sons plc", "Vos" and "Diensten V.O.F.". */
            auto opens_organization = [&](std::size_t k) {
                const piece& p = found[k];
                return p.type == pii_type::organization && k + 1 < found.size() &&
                       found[k + 1].type == pii_type::organization && found[k + 1].b == p.e + 1 &&
                       (t.cps[p.e] == ' ' || t.cps[p.e] == 0xA0) && is_upper(t.cps[p.b]);
            };
            /* A state or a province written in capitals beside a town or a postal code stays
               readable: "Sydney NSW 2010", "São Paulo - SP, 01001-000". */
            static const std::unordered_set<std::string> states = make_word_set(
                "al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj "
                "nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy nsw vic qld sa tas act nt on qc bc "
                "ab mb sk ns nb nl pe yt nu ac ap am ba ce df es go mg pb pr pi rj rn rs ro rr sp se to");
            auto state_beside_place = [&](std::size_t k) {
                const piece& p = found[k];
                if (p.type != pii_type::location || p.e - p.b < 2 || p.e - p.b > 3) return false;
                for (std::size_t q = p.b; q < p.e; ++q)
                    if (t.cps[q] < 'A' || t.cps[q] > 'Z') return false;
                if (!states.count(fold_key(cp_substr(t, text, p.b, p.e)))) return false;
                return (k > 0 && found[k - 1].type == pii_type::location && found[k - 1].e <= p.b &&
                        p.b - found[k - 1].e <= 3) ||
                       (k + 1 < found.size() && found[k + 1].type == pii_type::location && p.e <= found[k + 1].b &&
                        found[k + 1].b - p.e <= 3);
            };
            std::vector<piece> pieces;
            for (std::size_t k = 0; k < found.size(); ++k)
                if (opens_organization(k) ||
                    (!state_beside_place(k) &&
                     !spurious_learned_piece(text, t, found[k].b, found[k].e, found[k].type,
                                             k > 0 && name_gap(found[k - 1], found[k]))))
                    pieces.push_back(found[k]);
            /* A given name next to a family name is one person, a postal code next to its
               city one place, and the pieces of a word one word: they are replaced as a
               whole, like the rules do. */
            std::vector<piece> joined;
            for (const piece& p : pieces)
            {
                if (!joined.empty())
                {
                    piece& m = joined.back();
                    if (name_gap(m, p))
                    {
                        m.e = p.e;
                        m.confidence = std::min(m.confidence, p.confidence);
                        ++m.parts;
                        continue;
                    }
                }
                joined.push_back(p);
            }
            normalize_learned_organizations(text, t, joined);
            // Two capitalized words after the first one: the text capitalizes its names too.
            std::size_t capitals = 0;
            for (std::size_t i = 1; i < t.size() && capitals < 2; ++i)
                if (is_upper(t.cps[i]) && !is_word_char(t.cps[i - 1])) ++capitals;
            std::vector<piece> merged;
            for (piece p : joined)
            {
                // The at sign of a handle belongs to it: "@hpetit38".
                if (p.type == pii_type::identifier && p.b > 0 && t.cps[p.b - 1] == '@' &&
                    (p.b == 1 || !is_word_char(t.cps[p.b - 2])))
                    --p.b;
                if (!spurious_learned_span(text, t, p.b, p.e, p.type, capitals >= 2)) merged.push_back(p);
            }
            /* A confident recognizer outranks the heuristics of the rules for the same
               categories, whose boundaries follow no annotation guideline, and stays below
               the validated scanners. What the rules established about the same value is
               kept: a name they found with a title stays strong, so that its bare mentions
               are still learned, whichever boundaries win. The rule candidates overlapping a
               piece are found from the running maximum of the piece ends. */
            std::vector<bool> strong(merged.size());
            std::vector<double> confidence(merged.size());
            std::vector<std::size_t> max_end(merged.size());
            for (std::size_t p = 0; p < merged.size(); ++p)
            {
                strong[p] = merged[p].type == pii_type::person && merged[p].confidence >= 0.9 && merged[p].parts >= 2;
                confidence[p] = merged[p].confidence;
                max_end[p] = std::max(p > 0 ? max_end[p - 1] : 0, merged[p].e);
            }
            /* An organisation of a lexicon of names or a postal code of another country with
               its town, matched whole, keeps its boundaries and its type against the pieces
               read inside it, which would leave the rest readable ("Bahn" in "Deutsche Bahn",
               "Santander" in "Banco Santander", "Tokyo 153" in "Tokyo 153-0063"), against a
               piece of another type read on the same words ("Vueling" read as a place), and
               against an organisation that holds a part of it only ("Courrier de Le Parlement"
               across "Parlement européen"). */
            std::vector<bool> inside_exact(merged.size(), false);
            /* A piece inside a candidate of the rules, or beyond it by the end of its last word after
               an apostrophe, which the rules leave out: "Le Floc'h" over "Le Floc". */
            auto within = [&](const candidate& r, const piece& m) {
                if (m.b < r.b || m.b >= r.e) return false;
                if (m.e <= r.e) return true;
                if (!is_apostrophe(t.cps[r.e]) || m.e < r.e + 2) return false;
                for (std::size_t q = r.e + 1; q < m.e; ++q)
                    if (!is_letter(t.cps[q])) return false;
                return true;
            };
            const std::size_t rule_candidates = out.size();
            for (std::size_t k = 0; k < rule_candidates; ++k)
            {
                const candidate& r = out[k];
                std::size_t p = static_cast<std::size_t>(
                    std::upper_bound(max_end.begin(), max_end.end(), r.b) - max_end.begin());
                for (; p < merged.size() && merged[p].b < r.e; ++p)
                {
                    if (merged[p].e <= r.b) continue;
                    if (r.exact && r.b <= merged[p].b && merged[p].e <= r.e &&
                        (r.b != merged[p].b || r.e != merged[p].e || r.type != merged[p].type))
                        inside_exact[p] = true;
                    // "Courrier de Le Parlement" across "Parlement européen".
                    if (r.exact && r.type == pii_type::organization && merged[p].type == pii_type::organization &&
                        !(merged[p].b <= r.b && r.e <= merged[p].e) && !(r.b <= merged[p].b && merged[p].e <= r.e))
                        inside_exact[p] = true;
                    /* A person that a civility or a name label announces stays a person when a piece
                       read inside it is an organisation, a place or an address with no word of a firm
                       or of a street: "Nom : LE FLOC'H", "Mme Le Floc'h". */
                    if (r.strong && r.type == pii_type::person && within(r, merged[p]) &&
                        (merged[p].type == pii_type::organization || merged[p].type == pii_type::location ||
                         merged[p].type == pii_type::address) &&
                        announced_person(text, t, r.b) && !firm_or_street_words(text, t, merged[p].b, merged[p].e))
                        merged[p].type = pii_type::person;
                    if (merged[p].type != r.type) continue;
                    strong[p] = strong[p] || r.strong;
                    confidence[p] = std::max(confidence[p], r.confidence);
                }
            }
            for (std::size_t p = 0; p < merged.size(); ++p)
                if (!inside_exact[p])
                    add(out, merged[p].b, merged[p].e, merged[p].type, confidence[p],
                        merged[p].confidence >= 0.7 ? 10 : 0, strong[p]);
        }

    // ------------------------------------------------------------------------------------

        static const std::unordered_set<std::string>& civility_abbreviations()
        {
            static const std::unordered_set<std::string> s =
                privacy_impl::make_word_set("m mme mmes mlle mlles mr mrs ms mx");
            return s;
        }

        /* The civilities and professional titles that open the value of a name field, up to
           the name: "Mme", "M. et Mme", "Dr", "Monsieur le Docteur" are skipped, the
           civilities among them recorded. After a title, M. is an initial ("Monsieur M.
           Denis"). Returns the position of the name. */
        std::size_t skip_leading_titles(const std::string& text, const utf8_text& t, std::size_t b, std::size_t e,
                                        std::vector<std::pair<std::size_t, std::size_t>>& civilities) const
        {
            using namespace privacy_impl;
            std::size_t k = b;
            bool after_title = false;
            for (int round = 0; round < 6 && k < e; ++round)
            {
                std::size_t j = k;
                while (j < e && is_letter(t.cps[j])) ++j;
                if (j == k) break;
                // "MonsieurVERDIER Pierre": a civility glued to the name.
                word_token run;
                run.b = k;
                run.e = j;
                const std::size_t glued = after_title ? 0 : glued_civility(t, run);
                if (glued > 0)
                {
                    civilities.push_back(std::make_pair(k, k + glued));
                    return k + glued;
                }
                const std::string key = fold_key(cp_substr(t, text, k, j));
                std::size_t end = j;
                if (end < e && t.cps[end] == '.') ++end;
                std::size_t next = end;
                while (next < e && (t.cps[next] == ' ' || t.cps[next] == 0xA0)) ++next;
                // A title is followed by a space, or by a period and a capital ("M.Pichon"), and
                // by more of the value.
                const bool glued_period = end > j && next == end && end < e && is_upper(t.cps[end]);
                if ((next == end && !glued_period) || next >= e) break;
                if (key == "m" && (end == j || after_title)) break;
                if (civility_words().count(key))
                {
                    civilities.push_back(std::make_pair(k, end));
                    k = next;
                    after_title = true;
                    // "M. et Mme Martin", "Mr and Mrs Smith".
                    std::size_t c = k;
                    while (c < e && is_letter(t.cps[c])) ++c;
                    const std::string conj = fold_key(cp_substr(t, text, k, c));
                    if ((conj == "et" || conj == "and") && c < e && t.cps[c] == ' ')
                    {
                        std::size_t q = c + 1, r = c + 1;
                        while (r < e && is_letter(t.cps[r])) ++r;
                        if (civility_words().count(fold_key(cp_substr(t, text, q, r))))
                        {
                            k = q;
                            after_title = false;
                        }
                    }
                    continue;
                }
                // Professional titles and the article of "Monsieur le Docteur" stay readable.
                if (person_titles().count(key) || ((key == "le" || key == "la") && after_title))
                {
                    k = next;
                    after_title = true;
                    continue;
                }
                break;
            }
            return k;
        }

        void type_value(const std::string& text, const utf8_text& t, std::size_t b, std::size_t e,
                        pii_type type, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            while (b < e && is_space(t.cps[b])) ++b;
            // The period of an abbreviation belongs to it: "Title: Mrs.", "Employer: Acme Inc.".
            auto abbreviation_period = [&](std::size_t p) {
                if (type == pii_type::title) return true;
                if (type != pii_type::organization) return false;
                std::size_t q = p;
                while (q > b && is_letter(t.cps[q - 1])) --q;
                const std::string key = fold_key(cp_substr(t, text, q, p));
                return key == "inc" || key == "co" || key == "corp";
            };
            while (e > b && (is_space(t.cps[e - 1]) || t.cps[e - 1] == ',' || t.cps[e - 1] == ';' ||
                             (t.cps[e - 1] == '.' && !abbreviation_period(e - 1))))
                --e;
            if (e <= b) return;
            switch (type)
            {
                case pii_type::person:
                {
                    /* "Titulaire : Mme VAILLANT Danielle", "Nom : M. et Mme MARTIN": the
                       civilities before the name are replaced as TITLE and a professional
                       title ("Dr", "Maitre") stays readable; the name is what follows. */
                    std::vector<std::pair<std::size_t, std::size_t>> civilities;
                    b = skip_leading_titles(text, t, b, e, civilities);
                    if (b >= e) return;
                    std::size_t stop = e;
                    for (std::size_t k = b; k < e; ++k)
                    {
                        const char32_t c = t.cps[k];
                        if (c == '<' || c == '(' || c == ',' || c == ';' || c == '[' || c == '|' || c == '/' ||
                            is_ascii_digit(c) || c == '@')
                        {
                            stop = k;
                            break;
                        }
                    }
                    while (stop > b && is_space(t.cps[stop - 1])) --stop;
                    // "Contact : Severin au 06...": a name ends with a name, not with "au".
                    static const std::unordered_set<std::string> trailing = make_word_set(
                        "au aux a de du des le la les et ou sur en par pour chez at on in of and or via tel");
                    while (stop > b)
                    {
                        std::size_t q = stop;
                        while (q > b && is_letter(t.cps[q - 1])) --q;
                        if (q == stop || q == b || !is_space(t.cps[q - 1]) || !is_lower(t.cps[q]) ||
                            !trailing.count(fold_key(cp_substr(t, text, q, stop))))
                            break;
                        stop = q;
                        while (stop > b && is_space(t.cps[stop - 1])) --stop;
                    }
                    if (stop <= b) return;
                    int words = 0;
                    bool name_chars = true;
                    for (std::size_t k = b; k < stop; ++k)
                    {
                        const char32_t c = t.cps[k];
                        if (is_letter(c)) { if (k == b || !is_letter(t.cps[k - 1])) ++words; }
                        else if (!(c == ' ' || c == 0xA0 || is_hyphen(c) || is_apostrophe(c) || c == '.'))
                            name_chars = false;
                    }
                    if (name_chars && words >= 1 && words <= 6)
                    {
                        const std::string value = cp_substr(t, text, b, stop);
                        const utf8_text vt(value);
                        bool capital = false, uncommon = false;
                        for (const word_token& w : tokenize_words(vt, value))
                        {
                            if (w.shape == 1 || w.shape == 2 || w.shape == 3) capital = true;
                            if (!common_words().count(w.key) && !person_titles().count(w.key) &&
                                !acronym_words().count(w.key) && !organization_heads().count(w.key) &&
                                !street_types().count(w.key))
                                uncommon = true;
                        }
                        if (capital && uncommon && field_labels().count(fold_key(value)) == 0)
                        {
                            add(out, b, stop, pii_type::person, 0.95, 8, true);
                            for (const auto& c : civilities) add(out, c.first, c.second, pii_type::title, 0.95, 8);
                        }
                    }
                    return;
                }
                case pii_type::address:
                {
                    /* "12 rue des Lilas, 35000 Rennes": the street and the place after the
                       last comma are replaced as an address and a location, which is how the
                       learned recognizers report them too. The country written after the place
                       stays readable, with the sentence that may follow it on the line: "60000
                       Oujda, Maroc", "Miami, FL 33130, États-Unis. Est-elle toujours valable ?". */
                    for (std::size_t k = e; k > b + 1; --k)
                    {
                        if (t.cps[k - 1] != ',') continue;
                        std::size_t q = k, r = e;
                        while (q < r && is_space(t.cps[q])) ++q;
                        for (std::size_t p = q; p < r; ++p)
                            if (t.cps[p] == '.') { r = p; break; }
                        while (r > q && is_space(t.cps[r - 1])) --r;
                        if (q < r && country_names().count(fold_key(cp_substr(t, text, q, r))))
                        {
                            e = k - 1;
                            while (e > b && is_space(t.cps[e - 1])) --e;
                        }
                        break;
                    }
                    if (e <= b) return;
                    std::size_t comma = e;
                    for (std::size_t k = e; k > b + 1; --k)
                        if (t.cps[k - 1] == ',') { comma = k - 1; break; }
                    if (comma < e)
                    {
                        std::size_t tail = comma + 1;
                        while (tail < e && is_space(t.cps[tail])) ++tail;
                        std::size_t d = tail;
                        while (d < e && is_ascii_digit(t.cps[d])) ++d;
                        const bool postal = d - tail >= 4 && d - tail <= 5 && d < e && t.cps[d] == ' ';
                        const bool city = tail < e && is_upper(t.cps[tail]) && is_ascii_digit(t.cps[b]);
                        std::size_t head = comma;
                        while (head > b && is_space(t.cps[head - 1])) --head;
                        if ((postal || city) && head > b && tail < e)
                        {
                            add(out, b, head, pii_type::address, 0.9, 8);
                            add(out, tail, e, pii_type::location, 0.9, 8);
                            return;
                        }
                    }
                    add(out, b, e, pii_type::address, 0.9, 8);
                    return;
                }
                case pii_type::organization:
                {
                    // "Raison sociale : TAVELIS SAS", "Employeur : Novexa (Rennes)".
                    std::size_t stop = e;
                    for (std::size_t k = b; k < e; ++k)
                        if (t.cps[k] == '(' || t.cps[k] == ';' || t.cps[k] == '|' || t.cps[k] == '[' || t.cps[k] == ',')
                        { stop = k; break; }
                    while (stop > b && is_space(t.cps[stop - 1])) --stop;
                    bool capital = false;
                    for (std::size_t k = b; k < stop; ++k)
                        if (is_upper(t.cps[k])) { capital = true; break; }
                    if (capital && stop - b >= 2 && stop - b <= 80)
                        add(out, b, stop, pii_type::organization, 0.9, 8);
                    return;
                }

                case pii_type::location:
                {
                    std::size_t stop = e;
                    for (std::size_t k = b; k < e; ++k)
                        if (t.cps[k] == ',' || t.cps[k] == '(' || t.cps[k] == ';') { stop = k; break; }
                    while (stop > b && is_space(t.cps[stop - 1])) --stop;
                    // A place has letters, or is a postal code; "N/A" or "-" in an empty cell is none.
                    static const std::unordered_set<std::string> placeholders = make_word_set(
                        "na nc nr nd none aucun aucune inconnu inconnue unknown neant");
                    bool letters = false, digits_only = stop > b;
                    for (std::size_t k = b; k < stop; ++k)
                    {
                        if (is_letter(t.cps[k])) letters = true;
                        if (!is_ascii_digit(t.cps[k])) digits_only = false;
                    }
                    const bool postal = digits_only && stop - b >= 4 && stop - b <= 5;
                    std::string key;
                    for (char ch : fold_key(cp_substr(t, text, b, stop)))
                        if (ch != '/' && ch != '.' && ch != ' ') key.push_back(ch);
                    if ((letters || postal) && !placeholders.count(key))
                        add(out, b, stop, pii_type::location, 0.9, 8);
                    return;
                }
                case pii_type::identifier:
                case pii_type::license_plate:
                case pii_type::siret:
                {
                    std::size_t k = b;
                    while (k < e && !is_word_char(t.cps[k])) ++k;
                    /* "ID number: NHS number 621 941 9995": the words that name the number come
                       before it, when the value has digits. */
                    bool value_digits = false;
                    for (std::size_t q = k; q < e && !value_digits; ++q)
                        if (is_ascii_digit(t.cps[q])) value_digits = true;
                    while (value_digits && k < e && is_letter(t.cps[k]))
                    {
                        std::size_t q = k;
                        while (q < e && is_letter(t.cps[q])) ++q;
                        if (q < e && (is_ascii_digit(t.cps[q]) || t.cps[q] == '-' || t.cps[q] == '_')) break;
                        k = q;
                        while (k < e && !is_word_char(t.cps[k])) ++k;
                    }
                    std::size_t j = k;
                    while (j < e &&
                           (is_ascii_alnum_cp(t.cps[j]) ||
                            ((t.cps[j] == '-' || t.cps[j] == '/' || t.cps[j] == '.' || t.cps[j] == '_' ||
                              t.cps[j] == ' ') &&
                             j + 1 < e && is_ascii_alnum_cp(t.cps[j + 1]) &&
                             (t.cps[j] != ' ' || is_ascii_digit(t.cps[j + 1])))))
                        ++j;
                    // "Username: <username>": a placeholder names no account.
                    const std::string key = fold_key(cp_substr(t, text, k, j));
                    if (j - k >= 3 && !field_labels().count(key) && !common_words().count(key))
                        add(out, k, j, pii_type::identifier, 0.85, 4);
                    return;
                }
                case pii_type::phone:
                {
                    std::size_t stop = e;
                    for (std::size_t k = b; k < e; ++k)
                        if (t.cps[k] == '(' && k > b && !is_ascii_digit(t.at(k + 1))) { stop = k; break; }
                    while (stop > b && is_space(t.cps[stop - 1])) --stop;
                    int digits = 0;
                    bool phone_chars = true;
                    for (std::size_t k = b; k < stop; ++k)
                    {
                        const char32_t c = t.cps[k];
                        if (is_ascii_digit(c)) ++digits;
                        else if (!(c == ' ' || c == '.' || c == '-' || c == '+' || c == '(' || c == ')' || c == 0xA0))
                            phone_chars = false;
                    }
                    if (phone_chars && digits >= 8 && digits <= 17)
                        add(out, b, stop, pii_type::phone, 0.9, 2);
                    return;
                }
                case pii_type::title:
                case pii_type::gender:
                case pii_type::nationality:
                {
                    /* Closed vocabularies: the value is replaced only when it belongs to its
                       vocabulary, so that "Titre : Rapport annuel" or "Genre : roman" stay
                       readable. A list ("francaise et marocaine", "M/F") gives one span per
                       item. */
                    auto conjunction_at = [&](std::size_t p) {
                        static const char32_t et[] = { ' ', 'e', 't', ' ' };
                        static const char32_t an[] = { ' ', 'a', 'n', 'd', ' ' };
                        bool a = p + 4 <= e, d = p + 5 <= e;
                        for (std::size_t q = 0; q < 4 && a; ++q) if (t.cps[p + q] != et[q]) a = false;
                        for (std::size_t q = 0; q < 5 && d; ++q) if (t.cps[p + q] != an[q]) d = false;
                        return a ? 4 : (d ? 5 : 0);
                    };
                    std::size_t k = b;
                    while (k < e)
                    {
                        while (k < e && (is_space(t.cps[k]) || t.cps[k] == ',' || t.cps[k] == '/' || t.cps[k] == ';'))
                            ++k;
                        if (k >= e || is_line_break(t.cps[k]) || t.cps[k] == '(') return;
                        std::size_t j = k, next = 0;
                        while (j < e && t.cps[j] != ',' && t.cps[j] != '/' && t.cps[j] != ';' && t.cps[j] != '(' &&
                               !is_line_break(t.cps[j]))
                        {
                            const int skip = conjunction_at(j);
                            if (skip > 0) { next = j + static_cast<std::size_t>(skip); break; }
                            ++j;
                        }
                        std::size_t ie = j;
                        while (ie > k && is_space(t.cps[ie - 1])) --ie;
                        if (ie == k || !attribute_value_ok(type, fold_key(cp_substr(t, text, k, ie)))) return;
                        // A period ends an abbreviation ("Mme.", "Mrs."), not a word ("Madame.").
                        if (t.cps[ie - 1] == '.' && !civility_abbreviations().count(fold_key(cp_substr(t, text, k, ie - 1))))
                            --ie;
                        add(out, k, ie, type, 0.95, 6);
                        k = next > 0 ? next : j;
                    }
                    return;
                }
                default:
                    return;
            }
        }

        void scan_labels(const std::string& text, const utf8_text& t, const std::vector<word_token>& words,
                         const detection_context& context, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (context.label != pii_type::none)
                type_value(text, t, 0, t.size(), context.label, out);

            // Labels that only count at the start of a segment, where mail headers put them.
            static const std::unordered_set<std::string> header_only = make_word_set("a de to from cc");
            const std::size_t n = t.size();
            /* A label without value ("Nom :" before a tab, or alone on its line) types the
               next segment of its line, or the next line. */
            pii_type pending = pii_type::none;
            // "Ne le : 12/03/1980" then "a : Rennes": the place of birth, not a recipient.
            static const std::unordered_set<std::string> birth_labels = make_phrase_set(
                "ne le|nee le|date de naissance|date of birth|dob|birth date");
            bool birth_line = false, previous_birth_line = false;
            std::size_t line_start = 0;
            while (line_start < n)
            {
                std::size_t line_end = line_start;
                while (line_end < n && !is_line_break(t.cps[line_end])) ++line_end;
                previous_birth_line = birth_line;
                birth_line = false;
                bool blank = true;
                for (std::size_t c = line_start; c < line_end && blank; ++c)
                    if (!is_space(t.cps[c])) blank = false;
                if (blank) pending = pii_type::none;
                std::size_t seg = line_start;
                while (seg < line_end)
                {
                    std::size_t seg_end = seg;
                    while (seg_end < line_end && t.cps[seg_end] != '\t' && t.cps[seg_end] != 0x2022 &&
                           t.cps[seg_end] != '|')
                        ++seg_end;
                    std::size_t seg_first = seg;
                    while (seg_first < seg_end && is_space(t.cps[seg_first])) ++seg_first;

                    /* Every colon of the segment may close a label made of the 1 to 5 words
                       before it; each value runs to the next label. */
                    struct found_label { std::size_t start, colon; pii_type type; };
                    std::vector<found_label> labels;
                    std::size_t value_floor = seg;
                    for (std::size_t c = seg; c < seg_end; ++c)
                    {
                        if (t.cps[c] != ':' && t.cps[c] != 0xFF1A) continue;
                        if (c + 1 < n && t.cps[c + 1] == '/') continue;
                        // Last word ending before the colon, only spaces or label punctuation between.
                        std::size_t last = words.size();
                        {
                            auto it = std::lower_bound(words.begin(), words.end(), c,
                                [](const word_token& w, std::size_t pos) { return w.e <= pos; });
                            if (it != words.begin())
                            {
                                const std::size_t w = static_cast<std::size_t>(it - words.begin()) - 1;
                                if (words[w].b >= value_floor) last = w;
                            }
                        }
                        if (last == words.size() || c - words[last].e > 8) continue;
                        bool tight = true;
                        for (std::size_t g = words[last].e; g < c && tight; ++g)
                            tight = is_space(t.cps[g]) || t.cps[g] == '.' || t.cps[g] == ')' || t.cps[g] == 0xB0 ||
                                    t.cps[g] == 0xBA || t.cps[g] == '*';
                        if (!tight) continue;
                        for (std::size_t len = 5; len >= 1; --len)
                        {
                            if (last + 1 < len) continue;
                            const std::size_t first = last + 1 - len;
                            if (words[first].b < value_floor || !same_line(t, words[first].b, c)) continue;
                            const std::string norm = normalize_label(cp_substr(t, text, words[first].b, c));
                            pii_type type = lookup_label(norm);
                            if (type == pii_type::none) continue;
                            if (birth_labels.count(norm)) birth_line = true;
                            if (norm == "a" && previous_birth_line) type = pii_type::location;
                            if (header_only.count(norm) && words[first].b != seg_first) continue;
                            labels.push_back(found_label{ words[first].b, c, type });
                            value_floor = c + 1;
                            break;
                        }
                    }
                    if (pending != pii_type::none)
                    {
                        type_value(text, t, seg, labels.empty() ? seg_end : labels[0].start, pending, out);
                        pending = pii_type::none;
                    }
                    for (std::size_t k = 0; k < labels.size(); ++k)
                    {
                        const std::size_t value_end = k + 1 < labels.size() ? labels[k + 1].start : seg_end;
                        type_value(text, t, labels[k].colon + 1, value_end, labels[k].type, out);
                    }
                    if (!labels.empty())
                    {
                        const found_label& lab = labels.back();
                        bool empty = true;
                        for (std::size_t c = lab.colon + 1; c < seg_end && empty; ++c)
                            if (!is_space(t.cps[c])) empty = false;
                        if (empty && (seg_end < line_end || (seg == line_start && lab.start == seg_first)))
                            pending = lab.type;
                    }
                    seg = seg_end + 1;
                }
                line_start = line_end + 1;
            }
        }

        /* Tables written as text under a header line of field labels: "| Nom | Sexe |
           Nationalite |" over its rows, "Civilite;Nom;Prenom" over lines of values, tab
           separated columns. Each cell of a row is typed by the label of its column, as a
           spreadsheet column is typed by its header, until a blank line or a line with
           another number of cells. A header has two labels at least, labels in half of its
           cells, and labels only when commas separate it. */
        void scan_tables(const std::string& text, const utf8_text& t, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            typedef std::vector<std::pair<std::size_t, std::size_t>> cell_list;
            const std::size_t n = t.size();
            auto split = [&](std::size_t ls, std::size_t le, char32_t sep, cell_list& cells) {
                cells.clear();
                std::size_t cb = ls;
                for (std::size_t q = ls; q <= le; ++q)
                {
                    if (q < le && t.cps[q] != sep) continue;
                    std::size_t b = cb, e = q;
                    while (b < e && is_space(t.cps[b])) ++b;
                    while (e > b && is_space(t.cps[e - 1])) --e;
                    cells.push_back(std::make_pair(b, e));
                    cb = q + 1;
                }
                // The borders of "| a | b |" make empty cells.
                if (sep == '|')
                {
                    if (!cells.empty() && cells.front().first == cells.front().second) cells.erase(cells.begin());
                    if (!cells.empty() && cells.back().first == cells.back().second) cells.pop_back();
                }
            };
            // "|---|:---:|" under a Markdown header.
            auto rule_line = [&](const cell_list& cells) {
                if (cells.empty()) return false;
                for (const auto& c : cells)
                {
                    if (c.first == c.second) return false;
                    for (std::size_t q = c.first; q < c.second; ++q)
                        if (t.cps[q] != '-' && t.cps[q] != ':' && t.cps[q] != ' ') return false;
                }
                return true;
            };
            static const char32_t separators[] = { '|', '\t', ';', ',' };
            std::vector<pii_type> columns;
            char32_t sep = 0;
            cell_list cells;
            std::size_t ls = 0;
            while (ls < n)
            {
                std::size_t le = ls;
                while (le < n && !is_line_break(t.cps[le])) ++le;
                bool blank = true;
                for (std::size_t q = ls; q < le && blank; ++q)
                    if (!is_space(t.cps[q])) blank = false;
                bool row = false;
                if (!columns.empty() && !blank)
                {
                    split(ls, le, sep, cells);
                    if (rule_line(cells)) row = true;
                    else if (cells.size() == columns.size())
                    {
                        row = true;
                        for (std::size_t c = 0; c < cells.size(); ++c)
                            if (columns[c] != pii_type::none && cells[c].first < cells[c].second)
                                type_value(text, t, cells[c].first, cells[c].second, columns[c], out);
                    }
                }
                if (!row)
                {
                    columns.clear();
                    for (std::size_t s = 0; s < 4 && !blank; ++s)
                    {
                        split(ls, le, separators[s], cells);
                        if (cells.size() < 2) continue;
                        std::vector<pii_type> types;
                        std::size_t labels = 0;
                        bool short_cells = true;
                        for (const auto& c : cells)
                        {
                            if (c.second - c.first > 40) short_cells = false;
                            const pii_type type = c.first < c.second ?
                                lookup_label(normalize_label(cp_substr(t, text, c.first, c.second))) : pii_type::none;
                            if (type != pii_type::none) ++labels;
                            types.push_back(type);
                        }
                        if (short_cells && labels >= 2 && 2 * labels >= cells.size() &&
                            (separators[s] != ',' || labels == cells.size()))
                        {
                            columns.swap(types);
                            sep = separators[s];
                        }
                        break;
                    }
                }
                ls = le + 1;
            }
        }

    // ------------------------------------------------------------------------------------

        void scan_addresses(const utf8_text& t, const std::vector<word_token>& words, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::address)) return;
            const auto& streets = street_types();
            const auto& kept = address_kept_words();
            /* English types follow the name and never open an address: "Dr Martin" is a
               doctor, "Oak Dr" a drive. They are read by the second loop below. */
            static const std::unordered_set<std::string> suffix_only = make_word_set(
                "street st road rd lane ln drive dr blvd court ct way terrace highway hwy ave close crescent "
                "grove gardens parade row walk mews hill park circle plaza parkway trail");
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                const word_token& w = words[k];
                if (!streets.count(w.key) || suffix_only.count(w.key) ||
                    (w.shape == 1 && at_sentence_start(t, w.b) && k + 1 < words.size() && words[k + 1].shape == 0))
                    continue;
                // Number before: "12", "12 bis", "12,", "12bis".
                std::size_t start = w.b;
                bool number = false;
                std::size_t p = w.b;
                while (p > 0 && (t.cps[p - 1] == ' ' || t.cps[p - 1] == 0xA0 || t.cps[p - 1] == ',')) --p;
                if (k > 0 && words[k - 1].e == p && (words[k - 1].key == "bis" || words[k - 1].key == "ter" ||
                    words[k - 1].key == "quater" || (words[k - 1].e - words[k - 1].b == 1 && words[k - 1].shape == 2)))
                {
                    p = words[k - 1].b;
                    while (p > 0 && (t.cps[p - 1] == ' ' || t.cps[p - 1] == 0xA0)) --p;
                }
                std::size_t q = p;
                while (q > 0 && is_ascii_digit(t.cps[q - 1]) && p - q < 5) --q;
                if (q < p && p - q <= 4 && boundary_before(t, q) && same_line(t, q, w.b))
                {
                    start = q;
                    number = true;
                }
                /* "le mail de Federica Marino" is an email, "à la place de Jean Martin" a
                   replacement: in lower case and without a number, the word "mail", or a type
                   followed by "de" and a first name, names no street. */
                if (!number && w.shape == 0 &&
                    (w.key == "mail" ||
                     (k + 2 < words.size() && (words[k + 1].key == "de" || words[k + 1].key == "d") &&
                      words[k + 2].shape == 1 && first_names().count(words[k + 2].key))))
                    continue;
                // Street name after the type.
                std::size_t end = w.e, cursor = w.e;
                std::size_t j = k + 1;
                int named = 0;
                while (j < words.size() && j <= k + 8)
                {
                    const word_token& x = words[j];
                    if (!same_line(t, cursor, x.b) || x.b - cursor > 7) break;
                    bool gap_ok = true;
                    for (std::size_t g = cursor; g < x.b; ++g)
                    {
                        const char32_t gc = t.cps[g];
                        if (!(gc == ' ' || gc == 0xA0 || is_hyphen(gc) || is_apostrophe(gc) || is_ascii_digit(gc)))
                            gap_ok = false;
                    }
                    if (!gap_ok) break;
                    if (kept.count(x.key) && x.shape != 1) { cursor = x.e; ++j; continue; }
                    if (x.shape == 1 || x.shape == 2 || x.shape == 3)
                    {
                        if (common_words().count(x.key) && !kept.count(x.key) && named > 0) break;
                        end = cursor = x.e;
                        ++named;
                        ++j;
                        continue;
                    }
                    break;
                }
                // "Chemin C:\Users": without a number, a name of one letter is no street.
                if (named == 0 || (!number && named == 1 && end - words[j - 1].b <= 1))
                    continue;
                // "rue du 18 Juin 1940": the year of the date that names the street.
                {
                    static const std::unordered_set<std::string> months = make_word_set(
                        "janvier fevrier mars avril mai juin juillet aout septembre octobre novembre decembre");
                    std::size_t y = end + 1;
                    while (y < t.size() && y - end <= 4 && is_ascii_digit(t.cps[y]))
                        ++y;
                    if (months.count(words[j - 1].key) && words[j - 1].e == end && t.at(end) == ' ' &&
                        y - end == 5 && !is_word_char(t.at(y)))
                        end = y;
                }
                add(out, start, end, pii_type::address, number ? 0.9 : 0.6);
            }

            // "221B Baker Street", "1600 Pennsylvania Avenue NW", "56 Woodland Way".
            const auto& english = english_street_types();
            for (std::size_t k = 1; k < words.size(); ++k)
            {
                const word_token& w = words[k];
                if (!english.count(w.key) || !(w.shape == 1 || w.shape == 2)) continue;
                // "PONT-COURT-EN-COGLES", "Chester-le-Street" are places.
                if (is_hyphen(t.before(w.b)) || is_hyphen(t.at(w.e))) continue;
                std::size_t first = k;
                int named = 0;
                while (first > 0 && k - first < 4)
                {
                    const word_token& x = words[first - 1];
                    if (!(x.shape == 1 || x.shape == 2 || x.shape == 3) || !same_line(t, x.e, words[first].b) ||
                        words[first].b - x.e > 2)
                        break;
                    --first;
                    ++named;
                }
                if (named == 0) continue;
                // The number before the name, with an optional letter (221B).
                std::size_t p = words[first].b;
                while (p > 0 && (t.cps[p - 1] == ' ' || t.cps[p - 1] == 0xA0)) --p;
                std::size_t q = p;
                if (q > 0 && t.cps[q - 1] >= 'A' && t.cps[q - 1] <= 'Z' && q > 1 && is_ascii_digit(t.cps[q - 2])) --q;
                std::size_t r = q;
                while (r > 0 && is_ascii_digit(t.cps[r - 1]) && q - r < 5) --r;
                if (r == q || !boundary_before(t, r) || !same_line(t, r, words[first].b)) continue;
                std::size_t end = w.e;
                if (k + 1 < words.size() && words[k + 1].b == end + 1 && words[k + 1].shape == 2 &&
                    words[k + 1].e - words[k + 1].b <= 2)
                {
                    static const std::unordered_set<std::string> compass = make_word_set("n s e w ne nw se sw");
                    if (compass.count(words[k + 1].key)) end = words[k + 1].e;
                }
                add(out, r, end, pii_type::address, 0.9);
            }

            /* Streets of other countries, the house number after the name: the type first ("ul.
               Długa 5", "Calle Sol, 86", "Rua das Flores, 143", "Via Roma 5"), after the name
               ("Kossuth Lajos utca 16", "Atatürk Caddesi No:12", "Karl Johans gate 1",
               "Berliner Str. 5") or at the end of it ("Hauptstraße 12", "Keizersgracht 123"). */
            static const std::unordered_set<std::string> type_first = make_word_set(
                "ul ulica calle avenida avda carrer rua travessa via viale piazza piazzale corso largo strada calea "
                "str bulevardul ulitsa prospekt vulytsia cesta trg namesti rruga odos jalan");
            // Types also read as other words, which open a street only with their period: "al.", "pl.".
            static const std::unordered_set<std::string> type_first_dotted = make_word_set("al pl");
            // Written in lower case after the name ("Széchenyi tér 1", "Karl Johans gate 1"), or in any case.
            static const std::unordered_set<std::string> type_after_lower = make_word_set(
                "utca ut korut ter gate gata gaten vei veien vej gade gatan vagen tie katu");
            static const std::unordered_set<std::string> type_after = make_word_set(
                "sokak sokagı sokagi caddesi cad bulvarı bulvari yolu strasse strase str gasse platz allee straat "
                "laan kade plein");
            static const char* const glued_types[] = { "strasse", "strase", "str", "gasse", "weg", "platz", "allee",
                "gracht", "straat", "laan", "markt", "damm", "ufer", "veien", "vegen", "gade", "gatan", "gata",
                "vagen", "katu", "utca", "ulica", "kade", "plein" };
            static const std::unordered_set<std::string> name_connectors = make_word_set(
                "de del della dei degli di da do das dos la las los le les du des van von der den e y i");
            const std::size_t n = t.size();
            auto spaces_only = [&](std::size_t a, std::size_t b) {
                for (std::size_t q = a; q < b; ++q)
                    if (!(t.cps[q] == ' ' || t.cps[q] == 0xA0 || is_hyphen(t.cps[q]) || is_apostrophe(t.cps[q])))
                        return false;
                return b - a <= 3;
            };
            auto name_word = [&](const word_token& x) {
                return (x.shape == 1 || x.shape == 2 || x.shape == 3) && !known_organizations().count(x.key) &&
                       !acronym_words().count(x.key);
            };
            // The end of the house number that follows position p: " 5", ", 86", " nr 14", " No:9", " 12a", " 5/7".
            auto number_after = [&](std::size_t p) {
                std::size_t q = p;
                if (q < n && t.cps[q] == '.') ++q;
                if (q < n && t.cps[q] == ',') ++q;
                while (q < n && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                std::size_t r = q;
                std::string label;
                while (r < n && r - q < 3 && t.cps[r] < 128 && is_letter(t.cps[r]))
                    label.push_back(static_cast<char>(to_lower(t.cps[r++])));
                if ((label == "nr" || label == "no" || label == "n") && r < n &&
                    (t.cps[r] == '.' || t.cps[r] == ':' || t.cps[r] == 0xB0 || t.cps[r] == ' '))
                {
                    while (r < n && (t.cps[r] == '.' || t.cps[r] == ':' || t.cps[r] == 0xB0 || t.cps[r] == ' ')) ++r;
                    q = r;
                }
                std::size_t d = q;
                while (d < n && is_ascii_digit(t.cps[d]) && d - q < 5) ++d;
                if (d == q || d - q > 4) return p;
                if (d < n && is_letter(t.cps[d]) && !(d + 1 < n && is_letter(t.cps[d + 1]))) ++d;
                else if (d + 1 < n && (t.cps[d] == '/' || t.cps[d] == '-') && is_ascii_digit(t.cps[d + 1]))
                {
                    ++d;
                    while (d < n && is_ascii_digit(t.cps[d])) ++d;
                }
                if (d < n && is_word_char(t.cps[d])) return p;
                return d;
            };
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                const word_token& w = words[k];
                if (k > 0 && is_ascii_digit(t.before(w.b))) continue;
                // The type first, then the name.
                const bool dotted = t.at(w.e) == '.';
                if (type_first.count(w.key) || (dotted && type_first_dotted.count(w.key)))
                {
                    std::size_t cursor = dotted ? w.e + 1 : w.e, end = cursor;
                    int named = 0;
                    for (std::size_t j = k + 1; j < words.size() && j <= k + 6; ++j)
                    {
                        const word_token& x = words[j];
                        if (!spaces_only(cursor, x.b) || x.b == cursor) break;
                        if (name_connectors.count(x.key) && x.shape == 0 && j + 1 < words.size() &&
                            name_word(words[j + 1]))
                        {
                            cursor = x.e;
                            continue;
                        }
                        if (!name_word(x)) break;
                        end = cursor = x.e;
                        ++named;
                    }
                    const std::size_t number = named > 0 ? number_after(end) : end;
                    if (number > end)
                    {
                        add(out, w.b, number, pii_type::address, 0.85);
                        continue;
                    }
                }
                // The type after the name, or at the end of it.
                const bool after = k > 0 && (type_after.count(w.key) ||
                                             (w.shape == 0 && type_after_lower.count(w.key)));
                bool glued = false;
                if (!after && name_word(w) && !common_words().count(w.key))
                    for (const char* suffix : glued_types)
                    {
                        const std::size_t m = std::strlen(suffix);
                        if (w.key.size() > m + 2 && w.key.compare(w.key.size() - m, m, suffix) == 0) glued = true;
                    }
                if (!glued && !after) continue;
                std::size_t first = k;
                if (!glued)
                {
                    while (first > 0 && k - first < 4 && name_word(words[first - 1]) &&
                           spaces_only(words[first - 1].e, words[first].b) && words[first - 1].e < words[first].b)
                        --first;
                    if (first == k) continue;
                }
                const std::size_t number = number_after(w.e);
                if (number > w.e) add(out, words[first].b, number, pii_type::address, 0.85);
            }
        }

        void scan_locations(const utf8_text& t, const std::vector<word_token>& words, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::location)) return;
            const std::size_t n = t.size();
            const auto& kept = location_kept_words();

            auto capitalized = [](const word_token& x) { return x.shape == 1 || x.shape == 2 || x.shape == 3; };
            // Large cities are common words of the lexicon, and places all the same.
            static const std::unordered_set<std::string> big_cities = make_word_set(
                "paris lyon marseille rennes nantes bordeaux lille toulouse nice brest");
            auto glue = [&](std::size_t a, std::size_t b) {
                return b == a + 1 &&
                    (t.cps[a] == ' ' || t.cps[a] == 0xA0 || is_hyphen(t.cps[a]) || is_apostrophe(t.cps[a]));
            };
            // End of a capitalized place name starting at words[k]: "Saint-Germain-en-Laye".
            auto extend_place = [&](std::size_t k) -> std::size_t {
                std::size_t end = words[k].e;
                std::size_t j = k + 1;
                while (j < words.size() && j < k + 7 && glue(end, words[j].b))
                {
                    const word_token& x = words[j];
                    // "CEDEX" routes the mail and stays readable.
                    if (x.key == "cedex") break;
                    // "Pointe-a-Pitre": a preposition inside a hyphenated name only.
                    const bool inner_a = x.key == "a" && is_hyphen(t.before(x.b)) && is_hyphen(t.at(x.e));
                    if (capitalized(x))
                    {
                        if (common_words().count(x.key) && !kept.count(x.key)) break;
                        end = x.e;
                        ++j;
                        continue;
                    }
                    if ((kept.count(x.key) || inner_a) && j + 1 < words.size() && glue(x.e, words[j + 1].b) &&
                        capitalized(words[j + 1]))
                    {
                        end = words[j + 1].e;
                        j += 2;
                        continue;
                    }
                    break;
                }
                return end;
            };

            // Postal code and city: 35000 Rennes, 75008 PARIS CEDEX 08.
            for (std::size_t i = 0; i + 6 < n; ++i)
            {
                if (!is_ascii_digit(t.cps[i]) || !boundary_before(t, i) || t.before(i) == '.' || t.before(i) == ',')
                    continue;
                std::size_t j = i;
                while (j < n && is_ascii_digit(t.cps[j])) ++j;
                if (j - i != 5 || !(t.at(j) == ' ' || t.at(j) == 0xA0)) { i = j; continue; }
                const std::size_t k = static_cast<std::size_t>(std::lower_bound(words.begin(), words.end(), j + 1,
                    [](const word_token& w, std::size_t pos) { return w.b < pos; }) - words.begin());
                if (k >= words.size() || words[k].b != j + 1) continue;
                if (!(words[k].shape == 1 || words[k].shape == 2)) continue;
                // Amounts and quantities: "35000 Euros", "12000 EUR HT".
                static const std::unordered_set<std::string> units = make_word_set(
                    "euro euros eur dollar dollars usd franc francs chf livre livres pound pounds gbp yen yuan "
                    "ht ttc km habitants hab");
                if (units.count(words[k].key)) { i = j; continue; }
                // "35000 Le Mans", "32050 Les Essarts-en-Retz": an article opens the name of a town.
                const bool article = (words[k].key == "le" || words[k].key == "la" || words[k].key == "les" ||
                                      words[k].key == "l") && k + 1 < words.size() && capitalized(words[k + 1]) &&
                                     glue(words[k].e, words[k + 1].b);
                if (common_words().count(words[k].key) && !article && !big_cities.count(words[k].key))
                    continue;
                // The town, without "CEDEX" and its number, which stay readable.
                const std::size_t end = extend_place(k);
                if (end > words[k].b)
                {
                    add(out, i, end, pii_type::location, 0.85);
                    i = end;
                }
            }

            /* "ne a Brest", "nee le 3 mars 1985 a Saint-Malo", "domiciliee a Saint-Malo",
               "originaire de Quimper", "Fait a Lyon", "born in Leeds"; "habite Melun". */
            for (std::size_t k = 0; k + 1 < words.size(); ++k)
            {
                const std::string prep = words[k].key;
                const bool direct = prep == "habite" || prep == "habitent" || prep == "reside" || prep == "resident";
                if (!(prep == "a" || prep == "au" || prep == "aux" || prep == "en" || prep == "in" || prep == "at" ||
                      prep == "de" || prep == "d" || direct))
                    continue;
                // "Bourac-en-Cogles": the preposition of a hyphenated name.
                if (is_hyphen(t.before(words[k].b)) || is_hyphen(t.at(words[k].e))) continue;
                const word_token& x = words[k + 1];
                const std::size_t gap = x.b - words[k].e;
                if (!capitalized(x) || !same_line(t, words[k].e, x.b) ||
                    gap != 1)
                    continue;
                if (common_words().count(x.key) && !big_cities.count(x.key)) continue;
                // "a Sylviane Etienne": a first name is a person.
                if (first_names().count(x.key) && !ambiguous_first_names().count(x.key)) continue;
                bool trigger = direct;
                for (std::size_t m = k; m > 0 && k - m < 10 && !trigger; --m)
                {
                    const word_token& y = words[m - 1];
                    if (!same_line(t, y.e, words[k].b)) break;
                    if (location_triggers().count(y.key)) trigger = true;
                    // "de" and "d'" follow their trigger at once: "originaire de".
                    if (prep == "de" || prep == "d") break;
                }
                if (!trigger) continue;
                add(out, x.b, extend_place(k + 1), pii_type::location, 0.85);
            }

            /* The place of a letter before its date: "Rennes, le 12 mars 2024", "PARIS, le 11
               juillet 2027", at the start of a line. */
            for (std::size_t k = 0; k + 1 < words.size(); ++k)
            {
                const word_token& x = words[k];
                if (!capitalized(x) || (x.b > 0 && !is_line_break(t.cps[x.b - 1]) && t.cps[x.b - 1] != ' ' &&
                                        t.cps[x.b - 1] != '\t'))
                    continue;
                std::size_t ls = x.b;
                while (ls > 0 && (t.cps[ls - 1] == ' ' || t.cps[ls - 1] == '\t')) --ls;
                if (ls > 0 && !is_line_break(t.cps[ls - 1])) continue;
                const std::size_t end = extend_place(k);
                if (end + 5 >= n || t.cps[end] != ',' || t.cps[end + 1] != ' ' ||
                    !(t.cps[end + 2] == 'l' || t.cps[end + 2] == 'L') || !(t.cps[end + 3] == 'e' ||
                                                                         t.cps[end + 3] == 'E') ||
                    t.cps[end + 4] != ' ' || !is_ascii_digit(t.cps[end + 5]))
                    continue;
                if (common_words().count(x.key) && !big_cities.count(x.key)) continue;
                add(out, x.b, end, pii_type::location, 0.85);
            }
        }

        /* Towns and postal codes of English-speaking addresses: "Los Angeles, CA 90012",
           "Gold Coast, QLD 4217", "Winnipeg, MB R3C 0V8", where the state stays readable, and
           "Manchester M1 1AE", "London SW1A 2AA". */
        void scan_foreign_postcodes(const std::string& text, const utf8_text& t,
                                    const std::vector<word_token>& words, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::location)) return;
            static const std::unordered_set<std::string> states = make_word_set(
                "al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj "
                "nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy nsw vic qld sa tas act nt on qc bc "
                "ab mb sk ns nb nl pe yt nu");
            const std::size_t n = t.size();
            auto upper = [&](std::size_t i) { return i < n && t.cps[i] >= 'A' && t.cps[i] <= 'Z'; };
            auto digit = [&](std::size_t i) { return i < n && is_ascii_digit(t.cps[i]); };
            // The capitalized words of a town that end at position e: "Los Angeles", "EAST THORNSTEAD".
            auto town_before = [&](std::size_t e) {
                auto it = std::lower_bound(words.begin(), words.end(), e,
                    [](const word_token& w, std::size_t pos) { return w.e < pos; });
                if (it == words.end() || it->e != e) return e;
                std::size_t k = static_cast<std::size_t>(it - words.begin()), first = k, count = 0;
                while (count < 3 && (words[first].shape == 1 || words[first].shape == 2) &&
                       !common_words().count(words[first].key))
                {
                    ++count;
                    if (first == 0 || words[first - 1].e + 1 != words[first].b ||
                        !(t.cps[words[first - 1].e] == ' ' || is_hyphen(t.cps[words[first - 1].e])))
                        break;
                    --first;
                }
                if (count == 0) return e;
                if (!(words[first].shape == 1 || words[first].shape == 2) || common_words().count(words[first].key))
                    ++first;
                return first <= k ? words[first].b : e;
            };
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                const word_token& w = words[k];
                // ", CA 90012", ", QLD 4217", ", MB R3C 0V8".
                if (w.shape == 2 && states.count(w.key) && w.b >= 3 && t.cps[w.b - 1] == ' ' && t.cps[w.b - 2] == ',' &&
                    w.e + 1 < n && t.cps[w.e] == ' ')
                {
                    const std::size_t z = w.e + 1;
                    std::size_t ze = z;
                    while (digit(ze)) ++ze;
                    const bool us = ze - z == 5 && !(digit(ze) || upper(ze));
                    const bool au = ze - z == 4 && !(digit(ze) || upper(ze));
                    const bool ca = upper(z) && digit(z + 1) && upper(z + 2) && t.at(z + 3) == ' ' && digit(z + 4) &&
                                    upper(z + 5) && digit(z + 6) && !is_word_char(t.at(z + 7));
                    if (us && t.at(ze) == '-' && digit(ze + 1) && digit(ze + 4)) ze += 5;
                    if (!(us || au || ca)) continue;
                    if (ca) ze = z + 7;
                    const std::size_t town = town_before(w.b - 2);
                    if (town < w.b - 2) add(out, town, w.b - 2, pii_type::location, 0.85, 4);
                    add(out, z, ze, pii_type::location, 0.85, 4);
                    continue;
                }
                // "Manchester M1 1AE", "London SW1A 2AA", "Newport NP20 1PA".
                if (w.shape == 1 && w.e + 7 <= n && t.cps[w.e] == ' ' && upper(w.e + 1))
                {
                    std::size_t q = w.e + 1, letters = 0;
                    while (upper(q) && letters < 2) { ++q; ++letters; }
                    if (!digit(q)) continue;
                    ++q;
                    if (digit(q) || upper(q)) ++q;
                    if (t.at(q) != ' ' || !digit(q + 1) || !upper(q + 2) || !upper(q + 3) || is_word_char(t.at(q + 4)))
                        continue;
                    const std::size_t town = town_before(w.e);
                    if (town < w.e) add(out, town, q + 4, pii_type::location, 0.85, 4);
                }
            }

            /* The postal codes of other countries with their town, in an address: before the
               town ("8004 Zurich", "00-238 Warszawa", "1015 CJ Amsterdam", "1100-053 Lisboa",
               "111 51 Stockholm", "L-1611 Luxembourg", "125009 Moscow") or after it ("Cairo
               11511", "Tel Aviv 6473921", "Tokyo 150-0002", "Dublin D02 X285", "Sydney NSW
               2000", "Gatineau (Québec) J8X 3X7", "São Paulo - SP, 01001-000"). The line is an
               address when the street comes before the code on the same line or on the line
               above, when the name of a country follows the place, or when the place is
               alone on its line. */
            static const std::unordered_set<std::string> connectors = make_word_set(
                "am an der den de del della di do da dos das du la las los el on upon op aan nad pod sur les le "
                "sous en lez aux au");
            // Common words that open the name of a town ("East London", "Tel Aviv", "Le Caire").
            static const std::unordered_set<std::string> town_prefixes = make_word_set(
                "east west north south new old tel le la les el al los las san santa santo sao port fort mount "
                "mont saint st ste great little upper lower grand bad den de rio");
            // Words written before a number that are no town: complements, units, labels.
            static const std::unordered_set<std::string> not_towns = make_word_set(
                "bp cs tsa cp po box postfach apartado casilla cedex case casier lot bat km no nr flat unit suite "
                "apt appt room floor etage bureau porte tel fax ref euro euros eur dollar dollars usd franc francs "
                "chf livre livres pound pounds gbp yen yuan ht ttc habitants hab");
            auto word_at = [&](std::size_t b) -> std::size_t {
                auto it = std::lower_bound(words.begin(), words.end(), b,
                    [](const word_token& w, std::size_t pos) { return w.b < pos; });
                return it != words.end() && it->b == b ? static_cast<std::size_t>(it - words.begin()) : words.size();
            };
            auto capital = [](const word_token& x) { return x.shape == 1 || x.shape == 2; };
            auto town_word = [&](const word_token& x) {
                return capital(x) && !common_words().count(x.key) && !not_towns.count(x.key);
            };
            auto glued = [&](std::size_t a, std::size_t b) {
                return b == a + 1 && (t.cps[a] == ' ' || is_hyphen(t.cps[a]));
            };
            auto segment_end = [&](std::size_t e) {
                return e >= n || t.cps[e] == ',' || is_line_break(t.cps[e]) || t.cps[e] == '\t' ||
                       (t.cps[e] == ' ' && (t.at(e + 1) == '(' || is_hyphen(t.at(e + 1))));
            };
            /* The end of the town that starts at b: capitalized words and their connectors, "St."
               before a name, and a last capitalized common word that ends the line or the
               segment ("Karlovy Vary"). */
            auto town_after = [&](std::size_t b) {
                std::size_t k = word_at(b);
                if (k == words.size() || !town_word(words[k])) return b;
                std::size_t end = words[k].e, count = 1;
                while (k + 1 < words.size() && count < 4)
                {
                    const word_token& x = words[k + 1];
                    const bool abbreviated = (words[k].key == "st" || words[k].key == "ste") && t.at(end) == '.' &&
                                             x.b == end + 2 && t.at(end + 1) == ' ' && capital(x);
                    if (!abbreviated && !glued(end, x.b)) break;
                    const bool last_word = capital(x) && segment_end(x.e) && !function_words().count(x.key);
                    if (!country_names().count(x.key) && (town_word(x) || abbreviated || last_word))
                    {
                        end = x.e;
                        ++k;
                        ++count;
                        continue;
                    }
                    if (connectors.count(x.key) && k + 2 < words.size() && glued(x.e, words[k + 2].b) &&
                        capital(words[k + 2]))
                    {
                        end = words[k + 2].e;
                        k += 2;
                        ++count;
                        continue;
                    }
                    break;
                }
                return end;
            };
            // The start of the town that ends at e, the end of its last word, or e when none does.
            auto town_before_code = [&](std::size_t e) {
                auto it = std::lower_bound(words.begin(), words.end(), e,
                    [](const word_token& w, std::size_t pos) { return w.e < pos; });
                if (it == words.end() || it->e != e || !town_word(*it)) return e;
                std::size_t first = static_cast<std::size_t>(it - words.begin()), count = 1;
                while (first > 0 && count < 4)
                {
                    const word_token& p = words[first - 1];
                    if (!glued(p.e, words[first].b)) break;
                    if ((town_word(p) && !country_names().count(p.key)) ||
                        (capital(p) && town_prefixes.count(p.key)))
                    {
                        --first;
                        ++count;
                        continue;
                    }
                    if (connectors.count(p.key) && first >= 2 && glued(words[first - 2].e, p.b) &&
                        (town_word(words[first - 2]) || (capital(words[first - 2]) &&
                                                          town_prefixes.count(words[first - 2].key))))
                    {
                        first -= 2;
                        ++count;
                        continue;
                    }
                    break;
                }
                return words[first].b;
            };
            // The end of the postal code that starts at i, or i when none does.
            auto code_end = [&](std::size_t i) {
                if (!boundary_before(t, i)) return i;
                if (upper(i))
                {
                    // A Canadian postal code: "J8X 3X7".
                    if (digit(i + 1) && upper(i + 2) && t.at(i + 3) == ' ' && digit(i + 4) && upper(i + 5) &&
                        digit(i + 6) && !is_word_char(t.at(i + 7)))
                        return i + 7;
                    // A country prefix ("L-1611", "CH-8001") or a letter and four digits ("C1043").
                    std::size_t j = i + 1;
                    if (upper(j)) ++j;
                    const bool prefix = t.at(j) == '-';
                    if (prefix) ++j;
                    else if (j != i + 1) j = i;
                    std::size_t d = j;
                    while (digit(d)) ++d;
                    if (j > i && !is_word_char(t.at(d)) && (d - j == 4 || (prefix && d - j == 5))) return d;
                    // An Eircode: "D02 X285", "A92 F3T7".
                    if (digit(i + 1) && (digit(i + 2) || upper(i + 2)) && t.at(i + 3) == ' ')
                    {
                        std::size_t q = i + 4, digits = 0;
                        while (q < i + 8 && (digit(q) || upper(q))) digits += digit(q++) ? 1 : 0;
                        if (q == i + 8 && digits > 0 && !is_word_char(t.at(q))) return q;
                    }
                    return i;
                }
                std::size_t j = i;
                while (digit(j)) ++j;
                const std::size_t d = j - i;
                if (d == 0) return i;
                if (t.at(j) == '-' && digit(j + 1))
                {
                    std::size_t q = j + 1;
                    while (digit(q)) ++q;
                    const std::size_t d2 = q - j - 1;
                    const bool ok = (d == 2 && d2 == 3) || (d == 4 && d2 == 3) || (d == 3 && d2 == 4) ||
                                    (d == 5 && d2 == 3);
                    return ok && !is_word_char(t.at(q)) ? q : i;
                }
                if (d == 3 && t.at(j) == ' ' && digit(j + 1) && digit(j + 2) && !is_word_char(t.at(j + 3)))
                    return j + 3;
                if (d == 4 && t.at(j) == ' ' && upper(j + 1) && upper(j + 2) && t.at(j + 3) == ' ')
                    return j + 3;
                return d >= 4 && d <= 7 && !is_word_char(t.at(j)) ? j : i;
            };
            auto has_digit_and_letter = [&](std::size_t a, std::size_t b) {
                bool d = false, l = false;
                for (std::size_t q = a; q < b && !(d && l); ++q)
                {
                    if (is_ascii_digit(t.cps[q])) d = true;
                    else if (is_letter(t.cps[q])) l = true;
                }
                return d && l;
            };
            // A street word in [a, b): "rue", "Road", "ul.", "Calle", "Hauptstraße", "Keizersgracht".
            static const std::unordered_set<std::string> foreign_street_words = make_word_set(
                "ul ulica utca ut calle avenida avda rua via viale piazza corso strada calea str bulevardul ulitsa "
                "prospekt vulytsia cesta trg namesti laan straat plein kade veien vegen vei vej gade gatan vagen "
                "katu tie kuja sokak caddesi cad marg nagar strasse strase gasse allee platz");
            // Folded keys: "Hauptstraße" reads "hauptstrase".
            static const char* const street_suffixes[] = { "strasse", "strase", "gasse", "weg", "platz", "allee",
                "gracht", "straat", "laan", "markt", "ring", "damm", "ufer", "veien", "vegen", "gade", "gatan",
                "vagen", "katu", "utca", "ulica", "quai", "kade", "plein" };
            auto street_hint = [&](std::size_t a, std::size_t b) {
                auto it = std::lower_bound(words.begin(), words.end(), a,
                    [](const word_token& w, std::size_t pos) { return w.b < pos; });
                for (; it != words.end() && it->e <= b; ++it)
                {
                    const std::string& key = it->key;
                    if (street_types().count(key) || english_street_types().count(key) ||
                        foreign_street_words.count(key))
                        return true;
                    for (const char* suffix : street_suffixes)
                    {
                        const std::size_t m = std::strlen(suffix);
                        if (key.size() > m && key.compare(key.size() - m, m, suffix) == 0) return true;
                    }
                }
                return false;
            };
            // A street before position b: on its line, after a comma, or on the line above when b opens its line.
            // With named, the street must also hold a street word: a year is no postal code ("2018 Toyota").
            auto street_before = [&](std::size_t b, bool named) {
                std::size_t q = b;
                while (q > 0 && (t.cps[q - 1] == ' ' || t.cps[q - 1] == 0xA0)) --q;
                if (q > 0 && t.cps[q - 1] == ',')
                {
                    std::size_t from = q - 1;
                    while (from > 0 && from + 80 > q && !is_line_break(t.cps[from - 1]) && t.cps[from - 1] != '\t')
                        --from;
                    return has_digit_and_letter(from, q - 1) && (!named || street_hint(from, q - 1));
                }
                if (q > 0 && !is_line_break(t.cps[q - 1])) return false;
                std::size_t e = q;
                while (e > 0 && (is_line_break(t.cps[e - 1]) || t.cps[e - 1] == ' ')) --e;
                if (e == 0 || q - e > 3) return false;
                std::size_t from = e;
                while (from > 0 && from + 80 > e && !is_line_break(t.cps[from - 1])) --from;
                return has_digit_and_letter(from, e) && (!named || street_hint(from, e));
            };
            /* A line that holds the place alone, as a paragraph of a letter or a cell does:
               "00-238 Warszawa", "1000 Bruxelles", "Cairo 11511". Four digits alone need a
               town that is no given name and no known organisation ("3008 Peugeot"). */
            auto alone = [&](std::size_t b, std::size_t e, std::size_t code_b, std::size_t code_e,
                             std::size_t town_b) {
                std::size_t q = b;
                while (q > 0 && (t.cps[q - 1] == ' ' || t.cps[q - 1] == 0xA0)) --q;
                if (!(q == 0 || is_line_break(t.cps[q - 1]) || t.cps[q - 1] == '\t')) return false;
                q = e;
                while (q < n && (t.cps[q] == ' ' || t.cps[q] == 0xA0)) ++q;
                if (!(q >= n || is_line_break(t.cps[q]) || t.cps[q] == '\t')) return false;
                bool digits = true;
                for (std::size_t r = code_b; r < code_e; ++r) digits = digits && digit(r);
                if (!digits || code_e - code_b >= 6) return true;
                const std::size_t k = word_at(town_b);
                return k < words.size() && !first_names().count(words[k].key) &&
                       !known_organizations().count(words[k].key);
            };
            // Four digits that could be a year.
            auto year_like = [&](std::size_t i, std::size_t ce) {
                if (ce != i + 4) return false;
                unsigned long v = 0;
                for (std::size_t q = i; q < ce; ++q)
                {
                    if (!digit(q)) return false;
                    v = v * 10 + static_cast<unsigned long>(t.cps[q] - '0');
                }
                return 1950 <= v && v <= 2039;
            };
            // The name of a country after position e: ", Suisse", " - Pays-Bas", the next line.
            auto country_after = [&](std::size_t e) {
                std::size_t q = e;
                while (q < n && q < e + 4 && (t.cps[q] == ' ' || t.cps[q] == 0xA0 || t.cps[q] == ',' ||
                                             is_hyphen(t.cps[q]) || is_line_break(t.cps[q])))
                    ++q;
                if (q == e || q >= n || !is_upper(t.cps[q])) return false;
                std::size_t r = q;
                for (int count = 0; count < 4 && r < n; ++count)
                {
                    while (r < n && (is_letter(t.cps[r]) || is_apostrophe(t.cps[r]))) ++r;
                    if (country_names().count(fold_key(cp_substr(t, text, q, r)))) return true;
                    if (r >= n || !(t.cps[r] == ' ' || is_hyphen(t.cps[r])) || r + 1 >= n || !is_letter(t.cps[r + 1]))
                        break;
                    ++r;
                }
                return false;
            };
            auto add_exact = [&](std::size_t b, std::size_t e) {
                const std::size_t count = out.size();
                add(out, b, e, pii_type::location, 0.85, 4);
                if (out.size() > count) out.back().exact = true;
            };
            for (std::size_t i = 0; i < n; ++i)
            {
                if (!(digit(i) || upper(i)) || !boundary_before(t, i)) continue;
                const std::size_t ce = code_end(i);
                if (ce == i) continue;
                // Before the town: "8004 Zurich", after a street. Five digits and their town are
                // read by scan_locations(), which knows the French towns and "CEDEX".
                bool five = ce == i + 5;
                for (std::size_t q = i; q < ce && five; ++q) five = digit(q);
                if (!five && (t.at(ce) == ' ' || t.at(ce) == 0xA0))
                {
                    std::size_t b = ce + 1;
                    if (t.at(b) == ' ') ++b;
                    const std::size_t te = town_after(b);
                    const bool year = year_like(i, ce);
                    if (te > b && (street_before(i, year) || country_after(te) || (!year && alone(i, te, i, ce, b))))
                    {
                        add_exact(i, te);
                        i = te;
                        continue;
                    }
                }
                // After the town: "Cairo 11511", "Sydney NSW 2000", "Gatineau (Québec) J8X 3X7".
                if (i < 2 || !(t.cps[i - 1] == ' ' || t.cps[i - 1] == 0xA0) || is_word_char(t.at(ce))) continue;
                std::size_t town_end = i - 1;
                std::size_t k = static_cast<std::size_t>(std::lower_bound(words.begin(), words.end(), town_end,
                    [](const word_token& w, std::size_t pos) { return w.e < pos; }) - words.begin());
                if (k < words.size() && words[k].e == town_end && words[k].shape == 2 && states.count(words[k].key) &&
                    words[k].b >= 2 && t.cps[words[k].b - 1] == ' ')
                    town_end = words[k].b - 1;
                else if (t.cps[i - 2] == ')' && ce == i + 7 && upper(i))
                {
                    std::size_t open = i - 2;
                    while (open > 0 && i - open < 32 && t.cps[open] != '(') --open;
                    if (t.cps[open] != '(' || open < 2 || t.cps[open - 1] != ' ') continue;
                    town_end = open - 1;
                }
                else if (t.cps[i - 2] == ',' && i >= 8 && upper(i - 3) && upper(i - 4) && t.cps[i - 5] == ' ' &&
                         is_hyphen(t.cps[i - 6]) && t.cps[i - 7] == ' ')
                {
                    town_end = i - 7;
                }
                const std::size_t town = town_before_code(town_end);
                const bool year = year_like(i, ce);
                if (town >= town_end ||
                    !(street_before(town, year) || country_after(ce) || (!year && alone(town, ce, i, ce, town))))
                    continue;
                if (town_end == i - 1)
                {
                    add_exact(town, ce);
                }
                else
                {
                    add_exact(town, town_end);
                    add_exact(i, ce);
                }
                i = ce;
            }
        }

        /* Companies and other organisations named by a legal form ("Dupont Conseil SAS",
           "SAS Iroise", "Acme Inc.", "Hall Healthcare plc"), by a head noun ("Cabinet
           Dupont", "Groupe Iroise", "Etude Le Calvez notaires"), by a sector word after a
           name ("Zorvane Finance", "Watkins Medical Practice", "Rennes Metropole") or by its
           partners ("Martin et Fils", "Webb & Sons"). The name stops at a civility, at a
           preposition that introduces it ("chez", "with"), at a function word ("Groupe de
           Travail") and at the place written after a branch ("Banque Opea Asnieres-sur-Seine").
           "la societe", "the company" introduce a name and stay outside it. */
        void scan_organizations(const std::string& text, const utf8_text& t, const std::vector<word_token>& words,
                                std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            (void)text;
            if (!enabled(pii_type::organization)) return;
            const auto& forms = organization_forms();
            const auto& heads = organization_heads();
            const auto& kept = organization_kept_words();
            static const std::unordered_set<std::string> connectors = make_word_set(
                "de du des d la le les l et and of the for pour");
            static const std::unordered_set<std::string> stops = make_word_set(
                "chez aupres avec entre par dans vers contre sans siege at with by from into via between against");
            static const std::unordered_set<std::string> descriptors = make_word_set(
                "societe ste entreprise entreprises company firme enseigne");
            static const std::unordered_set<std::string> determiners = make_word_set(
                "la le l une un the a this cette notre votre sa");
            // Legal forms read in a line in lower case: "weber conseil sarl", "bluewater ag".
            static const std::unordered_set<std::string> lower_forms = make_word_set(
                "sarl sas sasu eurl snc selarl sci scp gmbh ag bv nv llc llp plc ltd limited inc corp srl oy oyj kft "
                "zrt asbl sprl bvba");
            static const std::unordered_set<std::string> partners = make_word_set(
                "fils associes associe cie compagnie partenaires freres soeurs sons partners co company associates");
            static const std::unordered_set<std::string> professions = make_word_set(
                "notaires notaire avocats avocat huissiers huissier associes architectes geometres");
            static const std::unordered_set<std::string> function_words = make_word_set(
                "travail avis scientifique scientifiques technique techniques financier financiere direction "
                "pilotage suivi coordination projet projets recherche administration surveillance");
            const auto& sectors = organization_sectors();
            // "Novexa France", "Wexalis Ouest": a region closes a company name.
            static const std::unordered_set<std::string> tails = make_word_set("france europe ouest atlantique");
            static const std::unordered_set<std::string> particles = make_word_set("le la de du des van von di da del mac");
            // Words that qualify a sector word without naming a company: "Steering Group", "Grande Distribution".
            // Adjectives between a head noun and its name: "Hôpital privé des Glénan", "Centre hospitalier de Brest".
            static const std::unordered_set<std::string> qualifiers = make_word_set(
                "prive privee public publique mutualiste hospitalier hospitaliere universitaire regional regionale "
                "departemental departementale intercommunal intercommunale municipal municipale general generale "
                "medical medicale dentaire veterinaire pluridisciplinaire specialise specialisee psychiatrique");
            static const std::unordered_set<std::string> modifiers = make_word_set(
                "grand grande petit petite nouveau nouvelle haut haute bas basse premier premiere espace "
                "international internationale global steering working focus advisory research support project "
                "task study user customer parent press sales marketing quality safety private public digital "
                "local regional national european");

            /* The gap between two words of a name. A period followed by a space ends a
               sentence unless it closes an initial or an abbreviation of one letter ("S. A."). */
            auto joinable = [&](std::size_t a, std::size_t b) {
                if (b < a || b - a > 3 || !same_line(t, a, b)) return false;
                bool period = false, space = false;
                for (std::size_t g = a; g < b; ++g)
                {
                    const char32_t c = t.cps[g];
                    if (!(c == ' ' || c == 0xA0 || is_hyphen(c) || is_apostrophe(c) || c == '&' || c == '.'))
                        return false;
                    if (c == '.') period = true;
                    if (c == ' ' || c == 0xA0) space = true;
                }
                return !(period && space && a >= 2 && is_letter(t.cps[a - 2]));
            };
            auto capital = [&](const word_token& x) { return x.shape == 1 || x.shape == 2 || x.shape == 3; };
            auto spaced = [&](std::size_t a) { return words[a].e + 1 == words[a + 1].b && t.cps[words[a].e] == ' '; };
            // "Asnieres-sur-Seine", "Saint-Louis", "SAINTE-CROIX-LES-BAINS": the shape of a place.
            auto place_shaped = [&](std::size_t j) {
                const word_token& x = words[j];
                if (!is_hyphen(t.at(x.e)) || j + 1 >= words.size()) return false;
                static const std::unordered_set<std::string> place_links = make_word_set(
                    "sur sous les en de du la le aux saint sainte st ste");
                return x.key == "saint" || x.key == "sainte" || place_links.count(words[j + 1].key) != 0;
            };
            /* The start of a name: "Le Saux SCI" and, at the start of a line or a value, "de La
               Kervella SNC", "d'Antonini SELARL"; the hyphenated words of "Quintin-Etcheverry
               Batiment" and "Cherbourg-en-Cotentin Metropole". */
            auto name_start = [&](std::size_t first) {
                std::size_t q = first;
                while (q > 0 && is_hyphen(t.before(words[q].b)) && words[q - 1].e + 1 == words[q].b) --q;
                if (q < first && capital(words[q])) first = q;
                auto line_start = [&](std::size_t i) {
                    std::size_t p = words[i].b;
                    while (p > 0 && (t.cps[p - 1] == ' ' || t.cps[p - 1] == 0xA0)) --p;
                    return p == 0 || is_line_break(t.cps[p - 1]) || t.cps[p - 1] == ':' || t.cps[p - 1] == '\t';
                };
                if (first > 0 && particles.count(words[first - 1].key) && words[first - 1].shape == 1 &&
                    spaced(first - 1) && !common_words().count(words[first].key) && !heads.count(words[first].key) &&
                    !sectors.count(words[first].key))
                {
                    --first;
                    if (first > 0 && (words[first - 1].key == "de" || words[first - 1].key == "du" ||
                                      words[first - 1].key == "van" || words[first - 1].key == "von") &&
                        words[first - 1].shape == 0 && spaced(first - 1) && line_start(first - 1))
                        --first;
                }
                else if (first > 0 && words[first - 1].key == "d" && words[first - 1].e + 1 == words[first].b &&
                         is_apostrophe(t.cps[words[first - 1].e]) && line_start(first - 1))
                    --first;
                return first;
            };

            for (std::size_t k = 0; k < words.size(); ++k)
            {
                const word_token& w = words[k];
                /* In a line in lower case, a legal form after one to three words of which one at
                   least is neither a common word nor a trade: "weber conseil sarl", "kerix limited". */
                if (w.shape == 0 && lower_forms.count(w.key) && k > 0 && lower_case_line(t, w.b))
                {
                    std::size_t first = k;
                    int named = 0;
                    while (first > 0 && k - first < 3 && spaced(first - 1))
                    {
                        const word_token& x = words[first - 1];
                        const bool trade = sectors.count(x.key) || kept.count(x.key);
                        if (x.shape != 0 || stops.count(x.key) || descriptors.count(x.key) ||
                            privacy_impl::function_words().count(x.key) || person_titles().count(x.key) ||
                            forms.count(x.key) || (common_words().count(x.key) && !trade))
                            break;
                        if (!trade) ++named;
                        --first;
                    }
                    while (first < k && (sectors.count(words[first].key) || kept.count(words[first].key))) ++first;
                    if (named > 0 && first < k)
                    {
                        std::size_t end = w.e;
                        if (t.at(end) == '.' && (w.key == "inc" || w.key == "corp")) ++end;
                        add(out, words[first].b, end, pii_type::organization, 0.75);
                        continue;
                    }
                }
                // Legal form after the name: "Dupont Conseil SAS", "Acme Inc.", "Briys plc".
                // "IL A PERDU SA VOITURE": a possessive in a line of capitals.
                const bool possessive = w.shape == 2 && w.e - w.b <= 2 && k + 1 < words.size() &&
                    words[k + 1].shape == 2 && words[k + 1].e - words[k + 1].b >= 3 &&
                    words[k + 1].b - w.e <= 2 && same_line(t, w.e, words[k + 1].b) && !forms.count(words[k + 1].key);
                if (forms.count(w.key) && (w.shape == 2 || (w.shape == 1 && w.e - w.b >= 3) || w.key == "plc") &&
                    !possessive)
                {
                    /* The capitalized words before the form, of which one at least is neither
                       a common word nor a connector: "La SARL" names no company. */
                    std::size_t first = k;
                    int named = 0;
                    while (first > 0 && k - first < 6 && joinable(words[first - 1].e, words[first].b))
                    {
                        const word_token& x = words[first - 1];
                        if (stops.count(x.key) || descriptors.count(x.key) || person_titles().count(x.key)) break;
                        // "BAGGAGE IS LIMITED", "PASSED TO THE DELHI MUNICIPAL CORPORATION".
                        if (x.shape == 2 && privacy_impl::function_words().count(x.key) && !connectors.count(x.key))
                            break;
                        // "SAS Iroise et SELARL Martin": two companies.
                        if ((forms.count(x.key) && x.shape == 2) ||
                            ((x.key == "et" || x.key == "and") && !partners.count(words[first].key)))
                            break;
                        if (capital(x))
                        {
                            if (!common_words().count(x.key) && !connectors.count(x.key) && !forms.count(x.key))
                                ++named;
                        }
                        else if (!kept.count(x.key)) break;
                        --first;
                    }
                    /* Connectors, articles, short common words, the word opening a sentence ("Entre
                       Dupont SARL") and, in capitals, common words ("JOINED TALOVA SERVICES INC.") do
                       not open the name. */
                    while (first < k && (words[first].shape == 0 || connectors.count(words[first].key) ||
                                         (common_words().count(words[first].key) &&
                                          (words[first].e - words[first].b <= 3 ||
                                           at_sentence_start(t, words[first].b) ||
                                           (words[first].shape == 2 && !sectors.count(words[first].key) &&
                                            !kept.count(words[first].key))))))
                        ++first;
                    if (named > 0 && first < k)
                    {
                        first = name_start(first);
                        std::size_t end = w.e;
                        if (t.at(end) == '.' && (w.key == "inc" || w.key == "co" || w.key == "corp"))
                            ++end;
                        add(out, words[first].b, end, pii_type::organization, 0.85);
                        continue;
                    }
                    /* Legal form before the name: "SAS Iroise", "SELARL Martin", "SCI du Port". In
                       capitals only, followed on the line by a capitalized name. */
                    if (w.shape == 2 && w.e - w.b >= 3 && k + 1 < words.size() &&
                        words[k + 1].b - w.e <= 2 && same_line(t, w.e, words[k + 1].b))
                    {
                        std::size_t j = k + 1, last = k;
                        while (j < words.size() && j <= k + 4 && joinable(words[j - 1].e, words[j].b))
                        {
                            const word_token& x = words[j];
                            if (capital(x) && !forms.count(x.key) && !person_titles().count(x.key) &&
                                !(common_words().count(x.key) && x.shape == 2))
                                last = j;
                            else if (!(connectors.count(x.key) && x.shape == 0)) break;
                            ++j;
                        }
                        if (last > k && !common_words().count(words[last].key))
                        {
                            add(out, w.b, words[last].e, pii_type::organization, 0.8);
                            continue;
                        }
                    }
                }
                // Partners after the name: "Martin et Fils", "Webb & Sons", "Guerin & Associes".
                if (partners.count(w.key) && capital(w) && k > 0)
                {
                    std::size_t p = w.b;
                    while (p > 0 && (t.cps[p - 1] == ' ' || t.cps[p - 1] == 0xA0)) --p;
                    std::size_t name = k;
                    if (p > 0 && t.cps[p - 1] == '&')
                    {
                        std::size_t q = p - 1;
                        while (q > 0 && (t.cps[q - 1] == ' ' || t.cps[q - 1] == 0xA0)) --q;
                        if (words[k - 1].e == q) name = k - 1;
                    }
                    else if (k >= 2 && (words[k - 1].key == "et" || words[k - 1].key == "and") && spaced(k - 1) &&
                             spaced(k - 2))
                        name = k - 2;
                    if (name < k && capital(words[name]) && !common_words().count(words[name].key) &&
                        !person_titles().count(words[name].key) && same_line(t, words[name].e, w.b))
                    {
                        add(out, words[name_start(name)].b, w.e, pii_type::organization, 0.8);
                        continue;
                    }
                }
                // Sector word after a name: "Zorvane Finance", "Yarrow Building Society", "Rennes Metropole".
                if ((sectors.count(w.key) || tails.count(w.key)) && w.shape == 1 && k > 0 &&
                    !(k + 1 < words.size() && sectors.count(words[k + 1].key) && words[k + 1].shape == 1 &&
                      spaced(k)))
                {
                    const bool metropole = w.key == "metropole" || w.key == "agglomeration" || w.key == "agglo";
                    std::size_t first = k;
                    int name_words = 0;
                    while (first > 0 && k - first < 3 && spaced(first - 1))
                    {
                        const word_token& x = words[first - 1];
                        if (x.shape != 1 && !(metropole && capital(x))) break;
                        if (modifiers.count(x.key) || heads.count(x.key) || person_titles().count(x.key) ||
                            forms.count(x.key) || nationality_words().count(x.key) || particles.count(x.key))
                            break;
                        if (sectors.count(x.key)) { --first; continue; }
                        if (common_words().count(x.key) && !metropole) break;
                        --first;
                        ++name_words;
                    }
                    while (first < k && sectors.count(words[first].key)) ++first;
                    if (name_words > 0 && first < k)
                    {
                        add(out, words[name_start(first)].b, w.e, pii_type::organization, 0.75);
                        continue;
                    }
                }
                /* Head noun before the name: "Cabinet Dupont", "Universite de Rennes", "Hôpital privé
                   des Glénan", "Auto-école du Steir"; "Centre-Val de Loire" is a region. */
                const bool auto_school = w.key == "ecole" && k > 0 && words[k - 1].key == "auto" &&
                                         words[k - 1].shape == 1 && words[k - 1].e + 1 == w.b &&
                                         is_hyphen(t.cps[words[k - 1].e]);
                if (heads.count(w.key) && (w.shape == 1 || w.shape == 2 || auto_school) && !is_hyphen(t.at(w.e)) &&
                    (!is_hyphen(t.before(w.b)) || auto_school))
                {
                    // end follows the capitalized words, cursor the connectors between them too.
                    std::size_t j = k + 1, end = w.e, cursor = w.e, first_named = 0;
                    int named = 0, proper = 0;
                    bool branch_place = false, after_connector = false, qualified = auto_school;
                    while (j < words.size() && j <= k + 6 && joinable(cursor, words[j].b))
                    {
                        const word_token& x = words[j];
                        if (person_titles().count(x.key) || stops.count(x.key)) break;
                        if ((x.key == "et" || x.key == "and") &&
                            !(j + 1 < words.size() && partners.count(words[j + 1].key)))
                            break;
                        if (connectors.count(x.key) || (kept.count(x.key) && !capital(x)))
                        {
                            if (named == 0 && (x.key == "de" || x.key == "du" || x.key == "d")) branch_place = true;
                            cursor = x.e;
                            after_connector = true;
                        }
                        else if (named == 0 && !after_connector && x.shape == 0 && qualifiers.count(x.key))
                        {
                            cursor = x.e;
                            qualified = true;
                        }
                        else if (capital(x))
                        {
                            if (common_words().count(x.key) && !kept.count(x.key) && !tails.count(x.key) &&
                                (named > 0 || (x.shape == 2 && w.shape == 2)))
                                break;
                            // "Banque Ambrelis SAINTE-CROIX-LES-BAINS", "Caisse Ulmeo Saint-Louis".
                            if (named > 0 && !after_connector &&
                                (place_shaped(j) || (x.shape == 2 && words[j - 1].shape == 1)))
                                break;
                            if (named == 0) first_named = j;
                            end = cursor = x.e;
                            ++named;
                            after_connector = false;
                            if (!function_words.count(x.key) && !tails.count(x.key)) ++proper;
                        }
                        else break;
                        ++j;
                    }
                    // "l'agence de Lyon", "AGENCE DE VILLEBOEUF": a branch named by its place only.
                    static const std::unordered_set<std::string> branch_heads = make_word_set(
                        "agence agency office");
                    if (branch_heads.count(w.key) && branch_place) continue;
                    if (proper == 0) continue;
                    // "the Bank Holiday": a common word alone, right after the head.
                    if (named == 1 && first_named == k + 1 && common_words().count(words[first_named].key)) continue;
                    // "Etude Le Calvez notaires".
                    if (j < words.size() && professions.count(words[j].key) && words[j].b == end + 1 &&
                        t.cps[end] == ' ')
                        end = words[j].e;
                    /* "la societe Calixe Courtage", "the company Ulmeo": after an article the name
                       follows the descriptor, which otherwise opens it ("Societe Kerexa SARL"). */
                    const bool determined = k > 0 && determiners.count(words[k - 1].key) &&
                                            w.b - words[k - 1].e <= 2 && same_line(t, words[k - 1].e, w.b);
                    const std::size_t begin = descriptors.count(w.key) && first_named > k && determined ?
                                              words[first_named].b : (auto_school ? words[k - 1].b : w.b);
                    // A qualified head names a body whose place or name follows: above the place alone.
                    add(out, begin, end, pii_type::organization, 0.8, qualified ? 8 : 0);
                }
            }
        }

    // ------------------------------------------------------------------------------------

        bool name_like(const utf8_text& t, const word_token& w) const
        {
            using namespace privacy_impl;
            if (!(w.shape == 1 || w.shape == 2 || w.shape == 3))
                return false;
            const std::size_t len = w.e - w.b;
            const bool known = user_name_words_.count(w.key) || doc_name_words_.count(w.key);
            if (len == 1)
            {
                // An initial, or the first letter of "N'Diaye", "O'Connor", "M'Bami", "D'Souza".
                if (w.shape == 2 && is_apostrophe(t.at(w.e)) && is_upper(t.at(w.e + 1)))
                    return w.key == "n" || w.key == "o" || w.key == "m" || w.key == "d";
                if (w.shape != 2 || t.at(w.e) != '.') return false;
                if (!person_titles().count(w.key)) return true;
                // Inside a group of initials, M is an initial too: "J.-M. Dupont", "M.-P. Martin".
                const bool after_group = w.b >= 2 && is_hyphen(t.cps[w.b - 1]) && t.cps[w.b - 2] == '.';
                const bool before_group = is_hyphen(t.at(w.e + 1)) && is_upper(t.at(w.e + 2)) && t.at(w.e + 3) == '.';
                return after_group || before_group;
            }
            // The articles of Spanish place names: "San Antonio", "Los Angeles".
            static const std::unordered_set<std::string> place_articles = make_word_set("san los las");
            if (!known)
            {
                if (place_articles.count(w.key)) return false;
                if (common_words().count(w.key) || acronym_words().count(w.key) || person_titles().count(w.key) ||
                    organization_heads().count(w.key) || organization_forms().count(w.key) ||
                    street_types().count(w.key) || person_cues_before().count(w.key) || greeting_words().count(w.key))
                    return false;
                if (w.shape == 2 && (len < 3 || !has_vowel(w.key)) && !first_names().count(w.key))
                    return false;
            }
            return true;
        }

        void scan_persons(const std::string& text, const utf8_text& t, const std::vector<word_token>& words,
                          std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            if (!enabled(pii_type::person)) return;
            const auto& particles = person_particles();
            for (std::size_t k = 0; k < words.size(); ++k)
            {
                std::size_t first = k;
                if (!name_like(t, words[k]))
                {
                    /* "Le Goff", "DE GAULLE": a capitalized particle opens the span when the
                       name after it is already known, or when a civility comes before it
                       ("M. Le Goff", "Mme De Smet"). */
                    const bool known_next = k + 1 < words.size() &&
                        (doc_name_words_.count(words[k + 1].key) || user_name_words_.count(words[k + 1].key));
                    std::size_t title_end = k > 0 ? words[k - 1].e : 0;
                    if (k > 0 && t.at(title_end) == '.') ++title_end;
                    const bool after_title = k > 0 && person_titles().count(words[k - 1].key) &&
                        title_end <= words[k].b && words[k].b - title_end <= 2 && same_line(t, title_end, words[k].b);
                    /* After a title the particles may be in lower case, and two in a row: "M. de
                       La Guen", "Madame DE LA BRETON", "Monsieur d'Aubin". */
                    std::size_t name_at = k + 1;
                    if (after_title && particles.count(words[k].key) && name_at + 1 < words.size() &&
                        particles.count(words[name_at].key) && name_gap(t, words[k].e, words[name_at].b, false))
                        ++name_at;
                    // "Monsieur le Procureur", "Madame la Presidente" name a function.
                    static const std::unordered_set<std::string> lower_particles = make_word_set(
                        "de du des d van von di da del");
                    const bool capital = words[k].shape == 1 || words[k].shape == 2;
                    if (!(particles.count(words[k].key) && (capital || (after_title && lower_particles.count(words[k].key))) &&
                          (known_next || after_title) &&
                          name_at < words.size() && name_gap(t, words[name_at - 1].e, words[name_at].b, false) &&
                          name_like(t, words[name_at])))
                        continue;
                    first = k;
                    k = name_at;
                }
                // Span: name-like words joined by name gaps, particles allowed inside.
                std::size_t last = k;
                std::size_t j = k + 1;
                while (j < words.size() && j - k < 7)
                {
                    const bool initial = words[last].e - words[last].b == 1;
                    if (!name_gap(t, words[last].e, words[j].b, initial)) break;
                    if (name_like(t, words[j])) { last = j; ++j; continue; }
                    // "Amadou BA", "Thanh LY": a family name of two capitals after a first name.
                    static const std::unordered_set<std::string> two_letters = make_word_set(
                        "ok oh ah eh nu mi re fa go up am pm id tv ue eu uk ci ai ia es ex ie io op os ya cv rh");
                    const word_token& x = words[j];
                    if (j == last + 1 && x.shape == 2 && x.e - x.b == 2 && has_vowel(x.key) && !two_letters.count(x.key) &&
                        first_names().count(words[last].key) && !ambiguous_first_names().count(words[last].key) &&
                        !common_words().count(x.key) && !function_words().count(x.key) && !person_titles().count(x.key) &&
                        !is_word_char(t.at(x.e)))
                    {
                        last = j;
                        ++j;
                        break;
                    }
                    if (particles.count(words[j].key) && words[j].e - words[j].b <= 4 && j + 1 < words.size() &&
                        name_gap(t, words[j].e, words[j + 1].b, false) && name_like(t, words[j + 1]))
                    {
                        last = j + 1;
                        j += 2;
                        continue;
                    }
                    break;
                }
                k = first;
                /* "Copie de Hélène Le Goff", "Réclamation de Gwendal Courtois": a word before a
                   particle opens the name only if it is a first name or a known name; otherwise
                   the name starts after the particle when a first name follows it, or when the
                   word opens the sentence and the particle is in lower case. */
                if (last >= k + 2 && particles.count(words[k + 1].key) && !first_names().count(words[k].key) &&
                    !doc_name_words_.count(words[k].key) && !user_name_words_.count(words[k].key) &&
                    ((first_names().count(words[k + 2].key) && !ambiguous_first_names().count(words[k + 2].key)) ||
                     (at_sentence_start(t, words[k].b) && words[k + 1].shape == 0 &&
                      (words[k + 1].key == "de" || words[k + 1].key == "d"))))
                    k += 2;
                /* "Microsoft Teams", "Toyota Yaris": an organisation known by its own name opens a
                   product, unless a first name follows it ("BLOOMBERG Michael"). */
                if (last > k && known_organizations().count(words[k].key) && !doc_name_words_.count(words[k].key) &&
                    !user_name_words_.count(words[k].key))
                {
                    bool first_name = false;
                    for (std::size_t m = k + 1; m <= last; ++m)
                        if (first_names().count(words[m].key)) first_name = true;
                    if (!first_name) continue;
                }
                const std::size_t b = words[k].b;
                std::size_t e = words[last].e;
                if (words[last].e - words[last].b == 1 && t.at(e) == '.')
                {
                    // A trailing initial alone ("Dupont J.") keeps its period.
                    ++e;
                }

                double score = 0;
                bool strong = false;
                int name_words = 0, title_shapes = 0, upper_shapes = 0, firsts = 0, ambiguous = 0, lexicon = 0,
                    email_hits = 0;
                for (std::size_t m = k; m <= last; ++m)
                {
                    const word_token& x = words[m];
                    if (particles.count(x.key) && x.e - x.b <= 4) continue;
                    ++name_words;
                    if (x.shape == 1 || x.shape == 3) ++title_shapes;
                    if (x.shape == 2 && x.e - x.b > 1) ++upper_shapes;
                    if (ambiguous_first_names().count(x.key)) ++ambiguous;
                    else if (first_names().count(x.key)) ++firsts;
                    if (doc_name_words_.count(x.key) || user_name_words_.count(x.key)) ++lexicon;
                    if (email_fragments_.count(x.key)) ++email_hits;
                }
                const std::string span_key = fold_key(cp_substr(t, text, b, e));
                if (firsts > 0) score += 2;
                else if (ambiguous > 0) score += name_words >= 2 ? 2 : 1;   // "Jean Dupont" weighs more than "jean" alone
                if (name_words >= 2) score += 0.5;
                if (title_shapes > 0 && upper_shapes > 0) score += 1.5;
                if (lexicon > 0) score += 3;
                if (user_names_full_.count(span_key) || doc_names_full_.count(span_key)) { score += 4; }
                if (email_hits > 0) score += email_hits >= 2 ? 3 : 1.5;

                // "MmeLamia Chaoui": a civility glued to the first word.
                if (glued_civility(t, words[k]) > 0) { score += 3.5; strong = true; }
                // Context before: title or cue word, "M.  Dupont" with a doubled space too.
                if (k > 0 && same_line(t, words[k - 1].e, b) &&
                    (b - words[k - 1].e <= 2 || (b - words[k - 1].e == 3 && t.cps[words[k - 1].e] == '.')))
                {
                    const std::string& pk = words[k - 1].key;
                    if (person_titles().count(pk)) { score += 3.5; strong = true; }
                    else if (person_cues_before().count(pk)) { score += 2; }
                    else if (street_types().count(pk) || organization_heads().count(pk)) continue;
                }
                // Context after: "Jean Dupont, ne le", "Jean Dupont (nee".
                std::size_t a = last + 1;
                if (a < words.size() && words[a].b - e <= 3 && same_line(t, e, words[a].b) &&
                    person_cues_after().count(words[a].key))
                    score += 2;

                if (name_words == 1 && lexicon == 0 && at_sentence_start(t, b)) score -= 1;
                auto saint = [&](const std::string& key) {
                    return key == "saint" || key == "sainte" || key == "st" || key == "ste";
                };
                if (lexicon == 0 && saint(words[k].key)) continue;
                // "ANDREWS Brandon, M, Italian, carpenter": a nationality alone is not a name.
                if (name_words == 1 && lexicon == 0 && !strong && nationality_words().count(words[k].key)) continue;
                // "Saint-Jean-de-Luz" is a place, even when Jean is a known name.
                if (k > 0 && saint(words[k - 1].key) && words[k - 1].e + 1 == b && is_hyphen(t.cps[words[k - 1].e]))
                    continue;
                if (k > 0 && lexicon == 0 && same_line(t, words[k - 1].e, b) && b - words[k - 1].e <= 2 &&
                    (words[k - 1].key == "a" || words[k - 1].key == "en" || words[k - 1].key == "au") && firsts == 0)
                    continue;
                /* A determiner before the span makes it a common noun phrase ("La Pierre
                   Philosophale", "the Rose Garden") unless the name is known. */
                static const std::unordered_set<std::string> determiners = make_word_set(
                    "le la les un une ce cet cette ces son sa ses mon ma mes ton ta tes notre nos votre vos leur leurs "
                    "the an this that these those my his her its our their your");
                if (k > 0 && lexicon == 0 && firsts == 0 && same_line(t, words[k - 1].e, b) &&
                    b - words[k - 1].e <= 2 && determiners.count(words[k - 1].key) && !particles.count(words[k].key))
                    continue;
                if (upper_shapes >= 3 && title_shapes == 0 && lexicon == 0) score -= 2;

                if (score >= 2.5)
                {
                    if (score >= 3.5 && (name_words >= 2 || strong)) strong = true;
                    add(out, b, e, pii_type::person, std::min(1.0, score / 5.0), 0, strong);
                    k = last;
                }
            }
        }

        /* Terms of the document lexicon and of the user dictionary, looked up by their first
           word: the cost follows the number of words of the text, not that of the terms. */
        void scan_terms(const utf8_text& t, const std::vector<word_token>& words, std::vector<candidate>& out) const
        {
            if (doc_terms_.empty() && terms_.empty()) return;
            const std::vector<word_token> tokens = term_tokens(t, words);
            auto matches = [&](const std::vector<std::string>& keys, std::size_t k) {
                if (k + keys.size() > tokens.size()) return false;
                for (std::size_t m = 1; m < keys.size(); ++m)
                    if (tokens[k + m].key != keys[m] || tokens[k + m].b - tokens[k + m - 1].e > 2) return false;
                // A number is a whole token: "12345" is not found in "A12345".
                const word_token& first = tokens[k];
                const word_token& last = tokens[k + keys.size() - 1];
                if (is_digit_key(keys.front()) && !boundary_before(t, first.b)) return false;
                if (is_digit_key(keys.back()) && !boundary_after(t, last.e)) return false;
                return true;
            };
            for (std::size_t k = 0; k < tokens.size(); ++k)
            {
                const auto d = doc_term_index_.find(tokens[k].key);
                // Document terms need the capital letter of a proper noun.
                if (d != doc_term_index_.end() && tokens[k].shape != 0)
                    for (std::size_t index : d->second)
                        if (matches(doc_terms_[index].first, k))
                            add(out, tokens[k].b, tokens[k + doc_terms_[index].first.size() - 1].e,
                                doc_terms_[index].second, 0.85);
                const auto u = term_index_.find(tokens[k].key);
                if (u != term_index_.end())
                    for (std::size_t index : u->second)
                        if (matches(terms_[index].first, k))
                            add(out, tokens[k].b, tokens[k + terms_[index].first.size() - 1].e, terms_[index].second,
                                1.0, 10, terms_[index].second == pii_type::person);
            }
        }

        pii_detector_options options_;
        std::vector<bool> enabled_;
        std::unordered_set<std::string> user_names_full_;
        std::unordered_set<std::string> user_name_words_;
        std::unordered_set<std::string> doc_names_full_;
        std::unordered_set<std::string> doc_name_words_;
        std::unordered_set<std::string> email_fragments_;
        std::vector<std::pair<std::vector<std::string>, pii_type>> doc_terms_;
        std::unordered_set<std::string> doc_term_set_;
        std::unordered_map<std::string, std::vector<std::size_t>> doc_term_index_;   // first key to doc_terms_
        std::vector<std::pair<std::vector<std::string>, pii_type>> terms_;
        std::unordered_map<std::string, std::vector<std::size_t>> term_index_;       // first key to terms_
        std::unordered_set<std::string> allowed_;

        struct compiled_pattern
        {
            pii_pattern spec;
            std::shared_ptr<std::regex> re;
        };
        std::vector<compiled_pattern> patterns_;

        struct recognizer_entry
        {
            pii_recognizer fn;
            double min_confidence = 0.5;
            std::size_t id = 0;
        };
        std::vector<recognizer_entry> recognizers_;

        /* Recognizers run once per text: learn() and detect() see the same units, and the
           models behind them are not reentrant, so calls are serialized and cached. */
        struct recognizer_cache
        {
            std::mutex m;
            std::unordered_map<std::string, std::vector<pii_span>> spans;
        };
        std::shared_ptr<recognizer_cache> cache_ = std::make_shared<recognizer_cache>();
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_DETECTOR_H_
