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
            bool strong;            // usable to teach the document lexicon
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
                    add(c, b, e, pii_type::date, 0.95);
                }
            }
            scan_labels(text, t, words, context, c);
            scan_tables(text, t, c);
            scan_attributes(text, t, words, c);
            scan_addresses(t, words, c);
            scan_locations(t, words, c);
            scan_foreign_postcodes(t, words, c);
            scan_organizations(text, t, words, c);
            scan_public_bodies(text, t, words, c);
            scan_english_bodies(t, words, c);
            scan_persons(text, t, words, c);
            scan_terms(t, words, c);
            scan_cued_references(t, words, c);
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
                while (last + 1 < words.size() && last < j + 5 && words[last + 1].b == words[last].e + 1 &&
                       (is_hyphen(t.cps[words[last].e]) || t.cps[words[last].e] == ' ' ||
                        (is_apostrophe(t.cps[words[last].e]) && words[last].e - words[last].b == 1)) &&
                       (words[last + 1].shape != 0 || location_kept_words().count(words[last + 1].key)))
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
                    // "NHS number" is a field label.
                    if (k + 1 < words.size() && number_words.count(words[k + 1].key) && joined(w.e, words[k + 1].b))
                        continue;
                    std::size_t last = k;
                    const bool with_place = place_after(k, last);
                    add(out, w.b, with_place ? words[last].e : w.e, pii_type::organization, 0.9, 10);
                    if (with_place) k = last;
                    continue;
                }
                if (w.shape == 1 || w.shape == 2)
                {
                    std::size_t best = 0;
                    for (std::size_t len = 1; len <= 9 && k + len <= words.size(); ++len)
                    {
                        const std::size_t j = k + len - 1;
                        if (len > 1 && !joined(words[j - 1].e, words[j].b)) break;
                        if (names.count(fold_key(cp_substr(t, text, w.b, words[j].e)))) best = len;
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
                        add(out, words[first].b, words[last].e, pii_type::organization, 0.9, 10);
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
                "package envoi shipment tracking decision mrn sejour");
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
                    if (ch == ' ' || ch == ':' || ch == '#' || ch == 0xB0 || ch == 0xBA || ch == '.' || ch == '\'' ||
                        ch == 0x2019 || ch == '-' || ch == 0xA0)
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

        void add_recognized(const std::string& text, const utf8_text& t, const std::vector<pii_span>& spans,
                            double min_confidence, std::vector<candidate>& out) const
        {
            using namespace privacy_impl;
            struct piece { std::size_t b, e; pii_type type; double confidence; int parts; };
            std::vector<piece> pieces;
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
                // A greeting read as a name or a place: "Bonsoir, merci pour votre retour."
                if ((type == pii_type::person || type == pii_type::location || type == pii_type::organization) &&
                    greeting_words().count(fold_key(cp_substr(t, text, b, e))))
                    continue;
                pieces.push_back(piece{ b, e, type, s.confidence, 1 });
            }
            std::sort(pieces.begin(), pieces.end(), [](const piece& x, const piece& y) { return x.b < y.b; });
            /* A given name next to a family name is one person, a postal code next to its
               city one place: they are replaced as a whole, like the rules do. */
            std::vector<piece> merged;
            for (const piece& p : pieces)
            {
                if (!merged.empty())
                {
                    piece& m = merged.back();
                    bool gap = m.type == p.type && (p.type == pii_type::person || p.type == pii_type::location) &&
                               p.b >= m.e && p.b - m.e <= 2;
                    for (std::size_t k = m.e; k < p.b && gap; ++k)
                        if (!(t.cps[k] == ' ' || is_hyphen(t.cps[k]))) gap = false;
                    if (gap)
                    {
                        m.e = p.e;
                        m.confidence = std::min(m.confidence, p.confidence);
                        ++m.parts;
                        continue;
                    }
                }
                merged.push_back(p);
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
            const std::size_t rule_candidates = out.size();
            for (std::size_t k = 0; k < rule_candidates; ++k)
            {
                const candidate& r = out[k];
                std::size_t p = static_cast<std::size_t>(
                    std::upper_bound(max_end.begin(), max_end.end(), r.b) - max_end.begin());
                for (; p < merged.size() && merged[p].b < r.e; ++p)
                {
                    if (merged[p].type != r.type || merged[p].e <= r.b) continue;
                    strong[p] = strong[p] || r.strong;
                    confidence[p] = std::max(confidence[p], r.confidence);
                }
            }
            for (std::size_t p = 0; p < merged.size(); ++p)
                add(out, merged[p].b, merged[p].e, merged[p].type, confidence[p], merged[p].confidence >= 0.7 ? 10 : 0,
                    strong[p]);
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
                       learned recognizers report them too. */
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
                    if (j - k >= 3)
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
        void scan_foreign_postcodes(const utf8_text& t, const std::vector<word_token>& words,
                                    std::vector<candidate>& out) const
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
            static const std::unordered_set<std::string> partners = make_word_set(
                "fils associes associe cie compagnie partenaires freres soeurs sons partners co company associates");
            static const std::unordered_set<std::string> professions = make_word_set(
                "notaires notaire avocats avocat huissiers huissier associes architectes geometres");
            static const std::unordered_set<std::string> function_words = make_word_set(
                "travail avis scientifique scientifiques technique techniques financier financiere direction "
                "pilotage suivi coordination projet projets recherche administration surveillance");
            static const std::unordered_set<std::string> sectors = make_word_set(
                "conseil services immobilier transports industrie industries distribution batiment informatique "
                "ingenierie logistique assurances assurance courtage formation sante energie environnement "
                "patrimoine expertise finance finances medical consulting logistics engineering solutions systems "
                "holdings partners analytics healthcare properties capital technologies media group insurance "
                "bank law academy clinic motors plumbing bakery practice solicitors society surveyors "
                "accountants metropole agglomeration agglo");
            // "Novexa France", "Wexalis Ouest": a region closes a company name.
            static const std::unordered_set<std::string> tails = make_word_set("france europe ouest atlantique");
            static const std::unordered_set<std::string> particles = make_word_set("le la de du des van von di da del mac");
            // Words that qualify a sector word without naming a company: "Steering Group", "Grande Distribution".
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
                    if (first > 0 && (words[first - 1].key == "de" || words[first - 1].key == "van" ||
                                      words[first - 1].key == "von") &&
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
                    // Connectors, articles, short common words and the word opening a sentence
                    // ("Entre Dupont SARL") do not open the name.
                    while (first < k && (words[first].shape == 0 || connectors.count(words[first].key) ||
                                         (common_words().count(words[first].key) &&
                                          (words[first].e - words[first].b <= 3 ||
                                           at_sentence_start(t, words[first].b)))))
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
                // Head noun before the name: "Cabinet Dupont", "Universite de Rennes"; "Centre-Val de
                // Loire" is a region.
                if (heads.count(w.key) && (w.shape == 1 || w.shape == 2) && !is_hyphen(t.at(w.e)) &&
                    !is_hyphen(t.before(w.b)))
                {
                    // end follows the capitalized words, cursor the connectors between them too.
                    std::size_t j = k + 1, end = w.e, cursor = w.e, first_named = 0;
                    int named = 0, proper = 0;
                    bool branch_place = false, after_connector = false;
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
                    // "la societe Calixe Courtage": the name follows the descriptor.
                    const std::size_t begin = descriptors.count(w.key) && first_named > k ? words[first_named].b : w.b;
                    add(out, begin, end, pii_type::organization, 0.8);
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
            if (!known)
            {
                if (common_words().count(w.key) || acronym_words().count(w.key) || person_titles().count(w.key) ||
                    organization_heads().count(w.key) || organization_forms().count(w.key) ||
                    street_types().count(w.key))
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
