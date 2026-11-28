// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_FICTIVE_VALUES_H_
#define DLIB_PRIVACY_FICTIVE_VALUES_H_

#include "fictive_values_abstract.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "../crypto/sha256.h"
#include "../crypto/crypto_utils.h"
#include "pii_types.h"
#include "unicode_text.h"
#include "id_checks.h"
#include "pii_lexicons.h"
#include "date_formats.h"
#include "pseudonymizer.h"
#include "fictive_lexicons.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct fictive_options
    {
        std::vector<std::string> languages;
        long date_shift_days = 180;
        bool keep_city = true;
        numeric_date_order date_order = numeric_date_order::day_month_year;
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        // Keyed draws: HMAC-SHA-256 of a message and of a counter, read 8 bytes at a time.
        class fictive_stream
        {
        public:
            fictive_stream(const hmac_sha256& base, std::string message) : base_(base), message_(std::move(message)) {}
            ~fictive_stream()
            {
                secure_wipe(block_.data(), block_.size());
                if (!message_.empty()) secure_wipe(&message_[0], message_.size());
            }
            fictive_stream(fictive_stream&&) = default;
            fictive_stream(const fictive_stream&) = delete;
            fictive_stream& operator=(const fictive_stream&) = delete;

            std::uint64_t next()
            {
                if (used_ >= 32)
                {
                    hmac_sha256 mac = base_;
                    mac.update(message_);
                    const unsigned char c[4] = { static_cast<unsigned char>(counter_ >> 24),
                                                 static_cast<unsigned char>(counter_ >> 16),
                                                 static_cast<unsigned char>(counter_ >> 8),
                                                 static_cast<unsigned char>(counter_) };
                    mac.update(c, 4);
                    block_ = mac.finalize();
                    ++counter_;
                    used_ = 0;
                }
                std::uint64_t v = 0;
                for (int k = 0; k < 8; ++k) v = (v << 8) | block_[used_ + static_cast<std::size_t>(k)];
                used_ += 8;
                return v;
            }

            std::size_t below(std::size_t n) { return n == 0 ? 0 : static_cast<std::size_t>(next() % n); }
            char digit() { return static_cast<char>('0' + below(10)); }
            std::string digits(std::size_t n)
            {
                std::string s;
                for (std::size_t k = 0; k < n; ++k) s.push_back(digit());
                return s;
            }

        private:
            hmac_sha256 base_;
            std::string message_;
            std::array<unsigned char, 32> block_{};
            std::size_t used_ = 32;
            std::uint32_t counter_ = 0;
        };

        // A run of a value: 0 other characters, 1 letters, 2 digits.
        struct fictive_token
        {
            std::size_t begin, end;
            int kind;
        };

        /*
            The runs of letters, of digits and of other characters of a value. An apostrophe
            between letters stays in the word ("O'Brien"), unless an elided particle stands before
            it ("d'Arc", "l'Orme").
        */
        inline std::vector<fictive_token> fictive_tokens(const std::string& s)
        {
            std::vector<fictive_token> out;
            auto kind_of = [](char32_t c) { return is_letter(c) ? 1 : is_ascii_digit(c) ? 2 : 0; };
            std::size_t i = 0;
            while (i < s.size())
            {
                std::size_t j = i;
                const int kind = kind_of(utf8_next(s, j));
                std::size_t end = j;
                while (end < s.size())
                {
                    std::size_t k = end;
                    const char32_t d = utf8_next(s, k);
                    int dk = kind_of(d);
                    if (kind == 1 && dk == 0 && is_apostrophe(d) && k < s.size())
                    {
                        std::size_t m = k;
                        const std::string before = fold_key(s.substr(i, end - i));
                        if (is_letter(utf8_next(s, m)) && before != "d" && before != "l" && before != "qu") dk = 1;
                    }
                    if (dk != kind) break;
                    end = k;
                }
                if (!out.empty() && out.back().kind == 0 && kind == 0) out.back().end = end;
                else out.push_back(fictive_token{ i, end, kind });
                i = end;
            }
            return out;
        }

        // The case of a word: 2 capitals only (two letters at least), 1 lower case only, 0 otherwise.
        inline int fictive_case(const std::string& w)
        {
            std::size_t letters = 0, upper = 0, lower = 0, i = 0;
            while (i < w.size())
            {
                const char32_t c = utf8_next(w, i);
                if (!is_letter(c)) continue;
                ++letters;
                if (is_upper(c)) ++upper;
                else if (is_lower(c)) ++lower;
            }
            if (letters >= 2 && upper == letters) return 2;
            if (letters >= 1 && lower == letters) return 1;
            return 0;
        }

        inline std::string fictive_cased(const std::string& w, int c)
        {
            return c == 2 ? uppercase(w) : c == 1 ? lowercase(w) : w;
        }

        // A name in ASCII lower case, the form of the local part of an address and of a host name.
        inline std::string fictive_ascii(const std::string& w)
        {
            std::string out;
            for (char ch : fold_key(w))
                if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) out.push_back(ch);
            return out;
        }

        inline std::size_t fictive_letter_count(const std::string& w)
        {
            std::size_t n = 0, i = 0;
            while (i < w.size())
                if (is_letter(utf8_next(w, i))) ++n;
            return n;
        }

        // value with its digits replaced, in order, by those of digits.
        inline std::string fictive_with_digits(const std::string& v, const std::string& digits)
        {
            std::string out = v;
            std::size_t k = 0;
            for (char& ch : out)
                if (ch >= '0' && ch <= '9' && k < digits.size()) ch = digits[k++];
            return out;
        }

        // The sex a civility tells: 1 male, 2 female, 0 none.
        inline int fictive_civility_sex(const std::string& folded)
        {
            static const std::unordered_set<std::string> male = make_word_set(
                "m mr monsieur messieurs sir herr herrn hr sig signor signore sr senor don");
            static const std::unordered_set<std::string> female = make_word_set(
                "mrs ms miss mme mmes mlle mlles madame mademoiselle mesdames madam frau fr signora signorina sigra "
                "sra srta senora senorita dona dna");
            if (male.count(folded)) return 1;
            if (female.count(folded)) return 2;
            return 0;
        }

        // The sex of the civility written right before pos in text, its periods and spaces left out.
        inline int fictive_civility_before(const std::string& text, std::size_t pos)
        {
            std::size_t e = pos;
            while (e > 0 && (text[e - 1] == ' ' || text[e - 1] == '\t')) --e;
            if (e >= 2 && static_cast<unsigned char>(text[e - 2]) == 0xC2 &&
                static_cast<unsigned char>(text[e - 1]) == 0xA0)
                e -= 2;
            std::size_t b = e;
            while (b > 0 && (std::isalpha(static_cast<unsigned char>(text[b - 1])) || text[b - 1] == '.') && e - b < 16)
                --b;
            if (b == e) return 0;
            if (b > 0 && (std::isalnum(static_cast<unsigned char>(text[b - 1])) || (text[b - 1] & 0x80))) return 0;
            std::string w;
            for (std::size_t k = b; k < e; ++k)
                if (text[k] != '.') w.push_back(text[k]);
            return fictive_civility_sex(fold_key(w));
        }

        // The given names of a language and of a sex whose folded initial is initial.
        inline const std::vector<const std::string*>& fictive_given_pool(std::size_t language, int sex, char initial)
        {
            static const std::vector<std::vector<const std::string*>> pools = []() {
                std::vector<std::vector<const std::string*>> r(fictive_languages().size() * 2 * 26);
                const std::vector<fictive_language>& all = fictive_languages();
                for (std::size_t k = 0; k < all.size(); ++k)
                    for (int s = 0; s < 2; ++s)
                        for (const std::string& n : s == 0 ? all[k].male : all[k].female)
                        {
                            const std::string f = fold_key(n);
                            if (f.empty() || f[0] < 'a' || f[0] > 'z') continue;
                            r[(k * 2 + static_cast<std::size_t>(s)) * 26 + static_cast<std::size_t>(f[0] - 'a')]
                                .push_back(&n);
                        }
                return r;
            }();
            static const std::vector<const std::string*> none;
            if (initial < 'a' || initial > 'z' || language >= fictive_languages().size() || (sex != 1 && sex != 2))
                return none;
            return pools[(language * 2 + static_cast<std::size_t>(sex - 1)) * 26 +
                         static_cast<std::size_t>(initial - 'a')];
        }

        // The words of street types in the five languages, which a fictive address keeps.
        inline const std::unordered_set<std::string>& fictive_street_types()
        {
            static const std::unordered_set<std::string> s = []() {
                std::unordered_set<std::string> r = make_word_set(
                    "calle c avenida avda av plaza pza paseo camino carretera ronda travesia callejon glorieta "
                    "via viale piazza piazzale corso largo vicolo strada lungomare contrada localita "
                    "strasse strase str platz weg gasse allee ring damm ufer chaussee steig pfad markt");
                for (const std::string& w : street_types()) r.insert(w);
                for (const std::string& w : english_street_types()) r.insert(w);
                for (const std::string& w : address_kept_words()) r.insert(w);
                return r;
            }();
            return s;
        }

        // The endings of German street names written as one word: "Lindenstrasse", "Bahnhofplatz".
        inline std::size_t fictive_german_street_suffix(const std::string& folded)
        {
            static const char* const endings[] = { "strasse", "strase", "str", "weg", "platz", "gasse", "allee", "ring",
                                                   "damm", "ufer", "steig", "pfad", "chaussee", "markt" };
            for (const char* e : endings)
            {
                const std::size_t n = std::string(e).size();
                if (folded.size() > n + 2 && folded.compare(folded.size() - n, n, e) == 0) return n;
            }
            return 0;
        }

        inline std::string fictive_reserved_ranges()
        {
            return "fr 19900 26191 35301 46571 53649 63998; nanp 55501; gb 7700900 2079460 4960 1914980 2896496 "
                   "2920180 1632960; de 3023125 6990009 4066969 2214710 8999998 17139200 176040690; iban fr 990; "
                   "card 411111 555555 378282 601111; nir 13-19; ipv4 192.0.2 198.51.100 203.0.113; "
                   "ipv6 2001:db8; mail example.com example.org example.net; plate X Y Z";
        }
    }

// ----------------------------------------------------------------------------------------

    class fictive_substituter
    {
    public:
        fictive_substituter(
            const pseudonymization_key& corpus_key,
            const fictive_options& options = fictive_options()
        ) : options_(options), mac_(draw_mac(corpus_key))
        {
            options_.date_shift_days = std::min(3650L, std::max(7L, options_.date_shift_days));
            std::vector<std::string> languages;
            for (const std::string& l : options_.languages)
            {
                const std::string code = privacy_impl::lowercase(l);
                if (privacy_impl::fictive_language_of(code) &&
                    std::find(languages.begin(), languages.end(), code) == languages.end())
                    languages.push_back(code);
            }
            if (languages.empty()) languages.push_back("fr");
            options_.languages = languages;
            language_ = languages[0];
            country_ = privacy_impl::fictive_language_of(language_)->country;
        }

        const fictive_options& get_options() const { return options_; }

        std::string fictive(pii_type type, const std::string& value, const std::string& document = "") const
        {
            draw_context c;
            c.document = document;
            return write(type, value, c);
        }

        long date_shift_days(const std::string& document) const
        {
            privacy_impl::fictive_stream st(mac_, "D\x1f" + document);
            const long weeks = std::max(1L, options_.date_shift_days / 7);
            const long w = 1 + static_cast<long>(st.below(static_cast<std::size_t>(weeks)));
            return st.below(2) ? 7 * w : -7 * w;
        }

        static std::string lexicon_version()
        {
            static const std::string v = []() {
                using namespace privacy_impl;
                sha256 h;
                auto add = [&h](const std::string& s) {
                    h.update(s.data(), s.size());
                    h.update("\n", 1);
                };
                for (const fictive_language& l : fictive_languages())
                {
                    add(l.code + " " + l.country);
                    for (const std::vector<std::string>* list : { &l.male, &l.female, &l.stems, &l.endings, &l.plural,
                                                                  &l.plural_feminine, &l.masculine, &l.feminine,
                                                                  &l.vowel, &l.plain })
                        for (const std::string& w : *list) add(w);
                }
                for (std::size_t k = 0; k < 5; ++k)
                    for (int sex = 0; sex < 2; ++sex) add(fictive_given_reading(k)[sex]);
                for (const fictive_city& c : fictive_cities())
                    add(c.country + "|" + c.size + "|" + c.name + "|" + c.postal + "|" + c.state);
                add(fictive_initials());
                add(fictive_reserved_ranges());
                const std::array<unsigned char, 32> d = h.finalize();
                static const char hex[] = "0123456789abcdef";
                std::string out = "1.1.0-";
                for (std::size_t k = 0; k < 6; ++k)
                {
                    out.push_back(hex[d[k] >> 4]);
                    out.push_back(hex[d[k] & 15]);
                }
                return out;
            }();
            return v;
        }

    private:
        friend class fictive_batch;

        // The state of a batch that its draws read: the draws moved on, the sexes of the civilities.
        struct batch_state
        {
            std::unordered_map<std::string, unsigned> attempts;
            std::unordered_map<std::string, int> sexes;

            unsigned attempt(const std::string& key) const
            {
                const auto it = attempts.find(key);
                return it == attempts.end() ? 0 : it->second;
            }
        };

        struct draw_context
        {
            const batch_state* state = nullptr;
            int civility = 0;           // the sex of the civility written before the value
            std::string document;
            std::string bump;           // the key of the draw that the batch moves on when the value collides
        };

        static hmac_sha256 draw_mac(const pseudonymization_key& key)
        {
            if (!key.is_set()) throw crypto_error("fictive_substituter: the key of the corpus is not set");
            privacy_impl::wiped_string k(hkdf_sha256(key.bytes(), "dlib.privacy.fictive.v1", "draws", 32));
            return hmac_sha256(k.s);
        }

        // The draws of key, at the attempt the batch reached for it.
        privacy_impl::fictive_stream stream(const std::string& key, draw_context& c) const
        {
            const unsigned a = c.state ? c.state->attempt(key) : 0;
            c.bump = key;
            return privacy_impl::fictive_stream(mac_, key + "\x1f" + std::to_string(a));
        }

        std::string write(pii_type type, const std::string& value, draw_context& c) const
        {
            switch (type)
            {
                case pii_type::person: return person(value, c);
                case pii_type::organization: return organization(value, c);
                case pii_type::location: return location(value, c);
                case pii_type::address: return address(value, c);
                case pii_type::email: return email(value, c);
                case pii_type::phone: return phone(value, c);
                case pii_type::iban: return iban(value, c);
                case pii_type::payment_card: return card(value, c);
                case pii_type::nir: return nir(value, c);
                case pii_type::siret: return siret(value, c);
                case pii_type::identifier: return identifier(value, c);
                case pii_type::ipv4: return ipv4(value, c);
                case pii_type::ipv6: return ipv6(value, c);
                case pii_type::mac_address: return mac(value, c);
                case pii_type::url: return url(value, c);
                case pii_type::date: return date(value, c);
                case pii_type::license_plate: return plate(value, c);
                case pii_type::geolocation: return geolocation(value, c);
                case pii_type::secret: return secret(value, c);
                case pii_type::title:
                case pii_type::gender:
                case pii_type::nationality:
                case pii_type::none: break;
            }
            return value;
        }

    // ------------------------------------------------------------------------------------
    // Names.

        // The fictive initial of a folded letter.
        char initial_of(char folded) const
        {
            const std::string& initials = privacy_impl::fictive_initials();
            if (folded < 'a' || folded > 'z') folded = static_cast<char>('a' + static_cast<unsigned char>(folded) % 26);
            privacy_impl::fictive_stream st(mac_, std::string("I\x1f") + folded);
            return initials[st.below(initials.size())];
        }

        static std::size_t language_index(const std::string& code)
        {
            const std::vector<privacy_impl::fictive_language>& all = privacy_impl::fictive_languages();
            for (std::size_t k = 0; k < all.size(); ++k)
                if (all[k].code == code) return k;
            return 0;
        }

        static bool known_given(const std::string& folded)
        {
            return privacy_impl::fictive_given_names().count(folded) != 0 ||
                   privacy_impl::first_names().count(folded) != 0;
        }

        /*
            The language and the sex of a given name: its lexicon, the primary language first, then
            the civility before it, then the civilities of the batch, then a draw.
        */
        void given_traits(const std::string& folded, int civility, const draw_context& c, std::string& lang,
                          int& sex) const
        {
            lang = language_;
            sex = 0;
            const auto& names = privacy_impl::fictive_given_names();
            const auto it = names.find(folded);
            if (it != names.end())
            {
                std::vector<std::string> order = options_.languages;
                for (const privacy_impl::fictive_language& l : privacy_impl::fictive_languages())
                    if (std::find(order.begin(), order.end(), l.code) == order.end()) order.push_back(l.code);
                for (const std::string& code : order)
                {
                    const std::size_t k = language_index(code);
                    const bool male = (it->second >> (2 * k)) & 1u, female = (it->second >> (2 * k + 1)) & 1u;
                    if (!male && !female) continue;
                    lang = code;
                    if (male != female) sex = male ? 1 : 2;
                    break;
                }
            }
            if (sex == 0) sex = civility;
            if (sex == 0 && c.state)
            {
                const auto s = c.state->sexes.find(folded);
                if (s != c.state->sexes.end() && s->second > 0) sex = s->second;
            }
            if (sex == 0)
            {
                privacy_impl::fictive_stream st(mac_, "X\x1f" + folded);
                sex = 1 + static_cast<int>(st.below(2));
            }
        }

        // A given name; forced, when not 0, is the sex of the name (that of the first given name of a person).
        std::string given_name(const std::string& folded, int civility, draw_context& c, int forced = 0) const
        {
            std::string lang;
            int sex = 0;
            given_traits(folded, civility, c, lang, sex);
            if (forced == 1 || forced == 2) sex = forced;
            const char initial = static_cast<char>(initial_of(folded.empty() ? 'a' : folded[0]) | 0x20);
            const std::size_t k = language_index(lang);
            const std::vector<const std::string*>* pool = &privacy_impl::fictive_given_pool(k, sex, initial);
            const privacy_impl::fictive_language& L = privacy_impl::fictive_languages()[k];
            std::vector<const std::string*> all;
            if (pool->size() < 2)
            {
                for (const std::string& n : sex == 2 ? L.female : L.male) all.push_back(&n);
                pool = &all;
            }
            privacy_impl::fictive_stream st =
                stream("G\x1f" + lang + "\x1f" + std::to_string(sex) + "\x1f" + folded, c);
            for (int tries = 0; tries < 16; ++tries)
            {
                const std::string& n = *(*pool)[st.below(pool->size())];
                if (privacy_impl::fold_key(n) != folded) return n;
            }
            return privacy_impl::fold_key(*(*pool)[0]) != folded ? *(*pool)[0] : *(*pool)[pool->size() - 1];
        }

        // The language of a family name by its ending, else the primary language.
        std::string family_language(const std::string& f) const
        {
            auto ends = [&f](const char* e) {
                const std::size_t n = std::string(e).size();
                return f.size() > n + 1 && f.compare(f.size() - n, n, e) == 0;
            };
            for (const char* e : { "ez", "az", "oz", "iz" })
                if (ends(e)) return "es";
            for (const char* e : { "elli", "ini", "etti", "ucci", "oni", "otti", "one", "ari", "esi", "uzzi", "ino" })
                if (ends(e)) return "it";
            for (const char* e : { "mann", "berg", "burg", "stein", "meier", "mayer", "maier", "bauer", "hoff",
                                   "hofer", "huber", "dorf", "feld", "ler" })
                if (ends(e)) return "de";
            if (f.find("sch") != std::string::npos) return "de";
            for (const char* e : { "son", "ton", "ley", "field", "wood", "worth", "ford", "ham", "by" })
                if (ends(e)) return "en";
            return language_;
        }

        std::string family_name(const std::string& folded, draw_context& c) const
        {
            const std::string lang = family_language(folded);
            const privacy_impl::fictive_language& L = privacy_impl::fictive_languages()[language_index(lang)];
            privacy_impl::fictive_stream st = stream("F\x1f" + lang + "\x1f" + folded, c);
            for (int tries = 0; tries < 16; ++tries)
            {
                const std::string n = L.stems[st.below(L.stems.size())] + L.endings[st.below(L.endings.size())];
                if (privacy_impl::fold_key(n) != folded) return n;
            }
            return L.stems[0] + L.endings[1];
        }

        // An acronym of the length of the original, in capitals.
        std::string acronym(const std::string& w, draw_context& c) const
        {
            const std::string f = privacy_impl::fold_key(w);
            privacy_impl::fictive_stream st = stream("A\x1f" + f, c);
            const std::size_t n = privacy_impl::fictive_letter_count(w);
            for (int tries = 0; tries < 16; ++tries)
            {
                std::string out;
                for (std::size_t k = 0; k < n; ++k) out.push_back(static_cast<char>('A' + st.below(26)));
                if (privacy_impl::fold_key(out) != f) return out;
            }
            return std::string(n, 'X');
        }

        enum word_role { keep_word, given_word, family_word, initial_word };

        /*
            The role of each run of a person name: civilities, titles and particles stay, a letter
            alone is an initial, the given names come first unless the family name is written
            first in capitals ("DUPONT Jean") or the name ends with a given name ("Dupont Jean"),
            and a word alone after a civility is a family name. The words joined by a hyphen take
            the role of the first of them. civility receives the sex of a civility of the name.
        */
        std::vector<int> person_roles(const std::string& v, const std::vector<privacy_impl::fictive_token>& t,
                                      int civility_before, int& civility) const
        {
            using namespace privacy_impl;
            std::vector<int> roles(t.size(), keep_word);
            std::vector<std::size_t> words;
            civility = civility_before;
            for (std::size_t k = 0; k < t.size(); ++k)
            {
                if (t[k].kind != 1) continue;
                const std::string w = v.substr(t[k].begin, t[k].end - t[k].begin);
                const std::string f = fold_key(w);
                const int sex = fictive_civility_sex(f);
                if (sex != 0 && (k + 1 >= t.size() || words.empty()))
                {
                    if (civility == 0) civility = sex;
                    continue;
                }
                if (person_titles().count(f) && words.empty()) continue;
                if (person_particles().count(f) && fictive_case(w) != 2 && !words.empty()) continue;
                if (fictive_letter_count(w) == 1)
                {
                    roles[k] = initial_word;
                    continue;
                }
                words.push_back(k);
            }
            if (words.empty()) return roles;
            // The first run of each group of runs joined by hyphens.
            std::vector<std::size_t> heads;
            for (std::size_t n = 0; n < words.size(); ++n)
            {
                const std::size_t k = words[n];
                const bool joined = n > 0 && k >= 2 && words[n - 1] == k - 2 &&
                                    v.compare(t[k - 1].begin, t[k - 1].end - t[k - 1].begin, "-") == 0;
                if (!joined) heads.push_back(n);
            }
            std::vector<int> head_roles(heads.size(), family_word);
            auto word_at = [&](std::size_t h) { return v.substr(t[words[heads[h]]].begin, t[words[heads[h]]].end -
                                                                                         t[words[heads[h]]].begin); };
            const std::size_t m = heads.size();
            if (m == 1)
            {
                const std::string w = word_at(0);
                head_roles[0] = civility == 0 && known_given(fold_key(w)) ? given_word : family_word;
            }
            else if (fictive_case(word_at(0)) == 2 && fictive_case(word_at(m - 1)) != 2)
            {
                for (std::size_t h = 0; h < m; ++h)
                    head_roles[h] = fictive_case(word_at(h)) == 2 ? family_word : given_word;
            }
            else if (known_given(fold_key(word_at(m - 1))) && !known_given(fold_key(word_at(0))))
            {
                for (std::size_t h = 0; h + 1 < m; ++h) head_roles[h] = family_word;
                head_roles[m - 1] = given_word;
            }
            else
            {
                for (std::size_t h = 0; h + 1 < m; ++h) head_roles[h] = given_word;
                head_roles[m - 1] = family_word;
            }
            for (std::size_t h = 0; h < m; ++h)
            {
                const std::size_t last = h + 1 < m ? heads[h + 1] : words.size();
                for (std::size_t n = heads[h]; n < last; ++n) roles[words[n]] = head_roles[h];
            }
            return roles;
        }

        // The folded first given name of a person value, the one that a civility gives its sex to.
        std::string first_given(const std::string& v) const
        {
            const std::vector<privacy_impl::fictive_token> t = privacy_impl::fictive_tokens(v);
            int civility = 0;
            const std::vector<int> roles = person_roles(v, t, 1, civility);
            for (std::size_t k = 0; k < t.size(); ++k)
                if (roles[k] == given_word) return privacy_impl::fold_key(v.substr(t[k].begin, t[k].end - t[k].begin));
            return std::string();
        }

        std::string person(const std::string& v, draw_context& c) const
        {
            using namespace privacy_impl;
            const std::vector<fictive_token> t = fictive_tokens(v);
            int civility = 0;
            const std::vector<int> roles = person_roles(v, t, c.civility, civility);
            // The sex of the first given name is that of the others: "Jean-Marie" stays a male name.
            int sex = civility;
            for (std::size_t k = 0; k < t.size() && sex == civility; ++k)
            {
                if (roles[k] != given_word) continue;
                std::string lang;
                int s = 0;
                given_traits(fold_key(v.substr(t[k].begin, t[k].end - t[k].begin)), civility, c, lang, s);
                sex = s;
                break;
            }
            std::string out, given_key, family_key;
            for (std::size_t k = 0; k < t.size(); ++k)
            {
                const std::string w = v.substr(t[k].begin, t[k].end - t[k].begin);
                const std::string f = fold_key(w);
                switch (roles[k])
                {
                    case given_word:
                        out += fictive_cased(given_name(f, civility, c, sex), fictive_case(w));
                        given_key = c.bump;
                        break;
                    case family_word:
                        out += fictive_cased(family_name(f, c), fictive_case(w));
                        family_key = c.bump;
                        break;
                    case initial_word:
                    {
                        const char i = initial_of(f.empty() ? 'a' : f[0]);
                        out.push_back(fictive_case(w) == 1 ? static_cast<char>(i | 0x20) : i);
                        break;
                    }
                    default:
                        out += w;
                }
            }
            c.bump = family_key.empty() ? given_key : family_key;
            return out;
        }

    // ------------------------------------------------------------------------------------
    // Organisations, places and addresses.

        std::string organization(const std::string& v, draw_context& c) const
        {
            using namespace privacy_impl;
            const std::vector<fictive_token> t = fictive_tokens(v);
            std::string out, last;
            std::size_t skip_to = 0;
            for (const fictive_token& k : t)
            {
                if (k.begin < skip_to) continue;
                const std::string w = v.substr(k.begin, k.end - k.begin);
                if (k.kind == 2)
                {
                    privacy_impl::fictive_stream st = stream("ON\x1f" + fold_key(v) + "\x1f" + w, c);
                    out += st.digits(w.size());
                    continue;
                }
                if (k.kind != 1)
                {
                    out += w;
                    continue;
                }
                const std::string f = fold_key(w);
                const bool kept = organization_forms().count(f) || organization_short_forms().count(f) ||
                                  organization_heads().count(f) || organization_sectors().count(f) ||
                                  organization_partner_words().count(f) || organization_trade_words().count(f) ||
                                  organization_kept_words().count(f) || function_words().count(f) ||
                                  person_particles().count(f) || public_body_heads().count(f) ||
                                  public_body_words().count(f);
                if (kept)
                {
                    out += w;
                    continue;
                }
                const std::size_t run = person_run(v, t, static_cast<std::size_t>(&k - &t[0]));
                if (run > static_cast<std::size_t>(&k - &t[0]))
                {
                    draw_context pc = c;
                    out += person(v.substr(k.begin, t[run].end - k.begin), pc);
                    last = pc.bump;
                    skip_to = t[run].end;
                    continue;
                }
                const fictive_city* city = fictive_city_named(w);
                if (city)
                {
                    bool same = false;
                    const fictive_city* chosen = city_for(w, std::string(), c, same);
                    out += same ? w : fictive_cased(chosen->name, fictive_case(w));
                    continue;
                }
                const std::size_t letters = fictive_letter_count(w);
                if (fictive_case(w) == 2 && letters >= 2 && letters <= 5)
                    out += acronym(w, c);
                else
                    out += fictive_cased(family_name(f, c), fictive_case(w));
                last = c.bump;
            }
            c.bump = last;
            return out;
        }

        /*
            The runs of a value that name a person: a known given name followed, after a space, by
            capitalised words ("Fondation Jean Moulin"). Returns the token where the run ends, or k.
        */
        static std::size_t person_run(const std::string& v, const std::vector<privacy_impl::fictive_token>& t,
                                      std::size_t k)
        {
            using namespace privacy_impl;
            const std::string w = v.substr(t[k].begin, t[k].end - t[k].begin);
            if (fictive_case(w) != 0 || !known_given(fold_key(w))) return k;
            std::size_t e = k;
            while (e + 2 < t.size() && t[e + 1].kind == 0 && t[e + 1].end - t[e + 1].begin == 1 &&
                   (v[t[e + 1].begin] == ' ' || v[t[e + 1].begin] == '-') && t[e + 2].kind == 1)
            {
                const std::string n = v.substr(t[e + 2].begin, t[e + 2].end - t[e + 2].begin);
                const std::string f = fold_key(n);
                if (fictive_case(n) != 0 || organization_forms().count(f) || organization_sectors().count(f) ||
                    organization_heads().count(f) || organization_partner_words().count(f))
                    break;
                e += 2;
            }
            return e;
        }

        /*
            The fictive city of a city name: the city itself when keep_city and the lexicon knows it
            with 50,000 inhabitants or more (same true), else a city of the lexicon of its country,
            or of country, or of the country of the primary language, and of its size class.
        */
        const privacy_impl::fictive_city* city_for(const std::string& name, const std::string& country,
                                                   draw_context& c, bool& same) const
        {
            using namespace privacy_impl;
            const fictive_city* known = fictive_city_named(name);
            same = false;
            if (known && options_.keep_city && known->size != 'S')
            {
                same = true;
                return known;
            }
            const std::string where = known ? known->country : country.empty() ? country_ : country;
            const char size = known ? known->size : 'S';
            std::vector<const fictive_city*> pool;
            for (const fictive_city& x : fictive_cities())
                if (x.country == where && x.size == size) pool.push_back(&x);
            if (pool.empty())
                for (const fictive_city& x : fictive_cities())
                    if (x.country == country_) pool.push_back(&x);
            const std::string f = fold_key(name);
            privacy_impl::fictive_stream st = stream("C\x1f" + f, c);
            for (int tries = 0; tries < 16; ++tries)
            {
                const fictive_city* x = pool[st.below(pool.size())];
                if (fold_key(x->name) != f) return x;
            }
            return pool[0];
        }

        static bool upper_ascii(char ch) { return ch >= 'A' && ch <= 'Z'; }
        static bool digit_char(char ch) { return ch >= '0' && ch <= '9'; }

        /*
            The British postcode of v: its outward code [b, m) and, when written, its inward code
            [m', e). Returns false when v holds none.
        */
        static bool uk_postcode(const std::string& v, std::size_t& b, std::size_t& e, std::size_t& outward_end)
        {
            for (std::size_t i = 0; i + 2 <= v.size(); ++i)
            {
                if (i > 0 && (std::isalnum(static_cast<unsigned char>(v[i - 1])))) continue;
                std::size_t j = i;
                std::size_t letters = 0;
                while (j < v.size() && upper_ascii(v[j]) && letters < 2) { ++j; ++letters; }
                if (letters == 0 || j >= v.size() || !digit_char(v[j])) continue;
                ++j;
                if (j < v.size() && (digit_char(v[j]) || upper_ascii(v[j]))) ++j;
                const std::size_t out_end = j;
                if (out_end < v.size() && std::isalnum(static_cast<unsigned char>(v[out_end])) &&
                    !(v[out_end] == ' '))
                {
                    // "SW1A1AA" written without its space
                    if (out_end + 3 <= v.size() && digit_char(v[out_end]) && upper_ascii(v[out_end + 1]) &&
                        upper_ascii(v[out_end + 2]) &&
                        (out_end + 3 == v.size() || !std::isalnum(static_cast<unsigned char>(v[out_end + 3]))))
                    {
                        b = i;
                        outward_end = out_end;
                        e = out_end + 3;
                        return true;
                    }
                    continue;
                }
                std::size_t k = out_end;
                if (k < v.size() && v[k] == ' ') ++k;
                if (k + 3 <= v.size() && digit_char(v[k]) && upper_ascii(v[k + 1]) && upper_ascii(v[k + 2]) &&
                    (k + 3 == v.size() || !std::isalnum(static_cast<unsigned char>(v[k + 3]))))
                {
                    b = i;
                    outward_end = out_end;
                    e = k + 3;
                    return true;
                }
            }
            return false;
        }

        // A British inward code: a digit and two letters of those it uses.
        static std::string uk_inward(privacy_impl::fictive_stream& st)
        {
            static const char letters[] = "ABDEFGHJLNPQRSTUWXYZ";
            std::string s(1, st.digit());
            s.push_back(letters[st.below(sizeof(letters) - 1)]);
            s.push_back(letters[st.below(sizeof(letters) - 1)]);
            return s;
        }

        /*
            A place: a city, its postal code before or after it, a department in parentheses, a
            British postcode, an American state and its ZIP code. A city that the lexicon does not
            know takes a city of street_country when it is given, the country of the street of an
            address.
        */
        std::string location(const std::string& v, draw_context& c, const std::string& street_country = "") const
        {
            using namespace privacy_impl;
            const std::vector<fictive_token> t = fictive_tokens(v);
            std::size_t pb = 0, pe = 0, outward_end = 0;
            const bool uk = uk_postcode(v, pb, pe, outward_end);
            std::size_t postal = t.size(), state = t.size();
            std::string country;
            if (uk) country = "gb";
            for (std::size_t k = 0; k < t.size() && !uk; ++k)
            {
                const std::size_t n = t[k].end - t[k].begin;
                if (t[k].kind == 2 && (n == 5 || n == 4) && postal == t.size()) postal = k;
                if (t[k].kind == 1 && n == 2 && upper_ascii(v[t[k].begin]) && upper_ascii(v[t[k].begin + 1]) &&
                    k + 2 < t.size() && t[k + 2].kind == 2 && t[k + 2].end - t[k + 2].begin == 5)
                {
                    state = k;
                    postal = k + 2;
                    country = "us";
                }
            }
            // The words of the city: the runs of letters outside the postcode, the state and a CEDEX.
            std::size_t cb = std::string::npos, ce = 0;
            for (std::size_t k = 0; k < t.size(); ++k)
            {
                if (t[k].kind != 1 || k == state) continue;
                if (uk && t[k].begin >= pb && t[k].end <= pe) continue;
                const std::string f = fold_key(v.substr(t[k].begin, t[k].end - t[k].begin));
                if (f == "cedex" || f == "cs" || f == "bp") break;
                if (cb != std::string::npos && k > 0 && t[k - 1].kind == 0)
                {
                    const std::string sep = v.substr(t[k - 1].begin, t[k - 1].end - t[k - 1].begin);
                    if (sep.find(',') != std::string::npos || sep.find('(') != std::string::npos) break;
                }
                if (cb == std::string::npos) cb = t[k].begin;
                ce = t[k].end;
            }
            const std::string city_name = cb == std::string::npos ? std::string() : v.substr(cb, ce - cb);
            if (!city_name.empty() && country.empty())
            {
                const fictive_city* known = fictive_city_named(city_name);
                if (known) country = known->country;
                else if (!street_country.empty()) country = street_country;
                else if (postal != t.size() && country_ == "gb") country = "us";
            }
            bool same = false;
            const fictive_city* chosen = city_for(city_name.empty() ? v : city_name, country, c, same);
            if (same)
            {
                /* A kept city keeps its postal code but the part of it that names a few streets: the
                   inward code of a British postcode, the four digits of an American ZIP+4. */
                privacy_impl::fictive_stream st = stream("CP\x1f" + fold_key(v), c);
                if (uk && pe > outward_end + 1)
                {
                    const std::size_t ib = v[outward_end] == ' ' ? outward_end + 1 : outward_end;
                    return v.substr(0, ib) + uk_inward(st) + v.substr(pe);
                }
                if (postal + 2 < t.size() && v.compare(t[postal + 1].begin, t[postal + 1].end - t[postal + 1].begin,
                                                       "-") == 0 &&
                    t[postal + 2].kind == 2 && t[postal + 2].end - t[postal + 2].begin == 4)
                    return v.substr(0, t[postal + 2].begin) + st.digits(4) + v.substr(t[postal + 2].end);
                c.bump.clear();
                return v;
            }
            std::string out;
            std::size_t pos = 0;
            auto replace = [&](std::size_t b, std::size_t e, const std::string& with) {
                if (b < pos) return;
                out.append(v, pos, b - pos);
                out += with;
                pos = e;
            };
            // The pieces in order of position.
            struct piece { std::size_t b, e; std::string with; };
            std::vector<piece> pieces;
            if (!city_name.empty())
                pieces.push_back(piece{ cb, ce, fictive_cased(chosen->name, fictive_case(city_name)) });
            privacy_impl::fictive_stream st = stream("CP\x1f" + fold_key(v), c);
            if (uk)
            {
                std::string code = chosen->country == "gb" ? chosen->postal : "EC1A";
                pieces.push_back(piece{ pb, outward_end, code });
                if (pe > outward_end + 1 || (pe > outward_end && v[outward_end] != ' '))
                {
                    const std::size_t ib = v[outward_end] == ' ' ? outward_end + 1 : outward_end;
                    pieces.push_back(piece{ ib, pe, uk_inward(st) });
                }
            }
            else if (postal != t.size())
            {
                const std::string original = v.substr(t[postal].begin, t[postal].end - t[postal].begin);
                std::string code = chosen->postal;
                if (code.size() != original.size() || !std::all_of(code.begin(), code.end(), digit_char))
                    code = st.digits(original.size());
                pieces.push_back(piece{ t[postal].begin, t[postal].end, code });
                if (state != t.size() && !chosen->state.empty())
                    pieces.push_back(piece{ t[state].begin, t[state].end, chosen->state });
            }
            // A department in parentheses after the city: the first digits of the new postal code.
            for (std::size_t k = 0; k + 2 < t.size(); ++k)
            {
                if (t[k + 1].kind != 2 || t[k + 1].end - t[k + 1].begin != 2 || t[k].kind != 0) continue;
                if (v.compare(t[k].end - 1, 1, "(") != 0 || k + 1 == postal) continue;
                if (chosen->postal.size() >= 2 && digit_char(chosen->postal[0]) && digit_char(chosen->postal[1]))
                    pieces.push_back(piece{ t[k + 1].begin, t[k + 1].end, chosen->postal.substr(0, 2) });
            }
            std::sort(pieces.begin(), pieces.end(), [](const piece& a, const piece& b) { return a.b < b.b; });
            for (const piece& p : pieces) replace(p.b, p.e, p.with);
            out.append(v, pos, std::string::npos);
            c.bump = "C\x1f" + fold_key(city_name.empty() ? v : city_name);
            return out;
        }

        // The language of an address, told by its type of street, else the primary language.
        std::string address_language(const std::string& v, const std::vector<privacy_impl::fictive_token>& t) const
        {
            using namespace privacy_impl;
            static const std::unordered_set<std::string> fr = make_word_set(
                "rue avenue boulevard bd allee impasse chemin place quai route cours passage voie sentier "
                "lotissement residence hameau faubourg esplanade parvis ruelle venelle montee traverse promenade");
            static const std::unordered_set<std::string> en = make_word_set(
                "street st road rd lane ln drive dr court ct way terrace close crescent grove gardens parade row "
                "walk mews hill park circle plaza parkway trail");
            static const std::unordered_set<std::string> es = make_word_set(
                "calle c avenida avda plaza pza paseo camino carretera ronda travesia callejon glorieta");
            static const std::unordered_set<std::string> it = make_word_set(
                "via viale piazza piazzale corso largo vicolo strada lungomare contrada");
            static const std::unordered_set<std::string> de = make_word_set(
                "strasse strase str platz weg gasse allee ring damm ufer chaussee am an im auf zum zur");
            for (const fictive_token& k : t)
            {
                if (k.kind != 1) continue;
                const std::string f = fold_key(v.substr(k.begin, k.end - k.begin));
                if (fr.count(f)) return "fr";
                if (es.count(f)) return "es";
                if (it.count(f)) return "it";
                if (en.count(f)) return "en";
                if (de.count(f) || fictive_german_street_suffix(f)) return "de";
            }
            return language_;
        }

        // A word of street name of language lang, agreeing with the article before it.
        std::string street_word(const std::string& folded, const std::string& lang, const std::string& article,
                                draw_context& c) const
        {
            const privacy_impl::fictive_language& L = privacy_impl::fictive_languages()[language_index(lang)];
            const std::vector<std::string>* list = &L.plain;
            if (article == "des" || article == "les" || article == "aux" || article == "los" || article == "dei" ||
                article == "degli")
                list = &L.plural;
            else if (article == "las" || article == "delle") list = &L.plural_feminine;
            else if (article == "du" || article == "au" || article == "del") list = &L.masculine;
            else if (article == "la" || article == "della") list = &L.feminine;
            else if (article == "l") list = &L.vowel;
            if (list->empty()) list = !L.plain.empty() ? &L.plain : &L.masculine;
            if (list->empty()) return family_name(folded, c);
            privacy_impl::fictive_stream st = stream("S\x1f" + lang + "\x1f" + article + "\x1f" + folded, c);
            for (int tries = 0; tries < 16; ++tries)
            {
                const std::string& w = (*list)[st.below(list->size())];
                if (privacy_impl::fold_key(w) != folded) return w;
            }
            return (*list)[0];
        }

        /*
            The postal code and the city at the end of an address: from a postal code followed by a
            city, or the part after the last comma that names a known city or holds a postcode.
            Returns v.size() when there is none.
        */
        std::size_t address_tail(const std::string& v, const std::vector<privacy_impl::fictive_token>& t) const
        {
            std::size_t pb = 0, pe = 0, oe = 0;
            const std::size_t comma = v.rfind(',');
            if (uk_postcode(v, pb, pe, oe))
            {
                std::size_t b = comma != std::string::npos && comma < pb ? comma + 1 : pb;
                while (b < v.size() && v[b] == ' ') ++b;
                return b;
            }
            for (std::size_t k = 0; k < t.size(); ++k)
            {
                const std::size_t n = t[k].end - t[k].begin;
                if (t[k].kind != 2 || n != 5) continue;
                // An American ZIP code after its state: the city before them, after a comma.
                if (k >= 2 && t[k - 2].kind == 1 && t[k - 2].end - t[k - 2].begin == 2 &&
                    upper_ascii(v[t[k - 2].begin]) && upper_ascii(v[t[k - 2].begin + 1]))
                {
                    const std::size_t before = v.rfind(',', t[k - 2].begin);
                    if (before == std::string::npos || before == 0) return t[k - 2].begin;
                    const std::size_t start = v.rfind(',', before - 1);
                    std::size_t b = start == std::string::npos ? 0 : start + 1;
                    while (b < v.size() && v[b] == ' ') ++b;
                    return b;
                }
                if (k + 2 < t.size() && t[k + 2].kind == 1) return t[k].begin;
                if (k > 0) return t[k].begin;
            }
            if (comma != std::string::npos)
            {
                std::size_t b = comma + 1;
                while (b < v.size() && v[b] == ' ') ++b;
                if (privacy_impl::fictive_city_named(v.substr(b))) return b;
            }
            return v.size();
        }

        std::string address(const std::string& v, draw_context& c) const
        {
            using namespace privacy_impl;
            const std::vector<fictive_token> t = fictive_tokens(v);
            const std::size_t tail = address_tail(v, t);
            const std::string head = v.substr(0, tail);
            const std::vector<fictive_token> h = fictive_tokens(head);
            const std::string lang = address_language(head, h);
            const std::string key = fold_key(head);
            std::string out, article, last;
            for (std::size_t k = 0; k < h.size(); ++k)
            {
                const std::string w = head.substr(h[k].begin, h[k].end - h[k].begin);
                if (h[k].kind == 2)
                {
                    // A number of as many digits, none of them a leading zero.
                    privacy_impl::fictive_stream st = stream("N\x1f" + key + "\x1f" + std::to_string(k), c);
                    std::string n;
                    for (int tries = 0; tries < 16 && (n.empty() || n == w); ++tries)
                    {
                        n.clear();
                        for (std::size_t d = 0; d < w.size(); ++d)
                            n.push_back(d == 0 && w.size() > 1 ? static_cast<char>('1' + st.below(9)) : st.digit());
                        if (w.size() == 1 && n == "0") n = "1";
                    }
                    out += n;
                    continue;
                }
                if (h[k].kind != 1)
                {
                    out += w;
                    continue;
                }
                const std::string f = fold_key(w);
                static const std::unordered_set<std::string> articles = make_word_set(
                    "des les du au aux la l de d le los las del della delle dei degli dello di da the of am an im "
                    "zum zur der die das y e et");
                if (articles.count(f))
                {
                    // "de la", "de l'": the article is the second word.
                    if (!(f == "de" || f == "d" || f == "di" || f == "da" || f == "of" || f == "the")) article = f;
                    else article.clear();
                    out += w;
                    continue;
                }
                // A letter after the number, glued or after a space ("12b", "94 a"), stays.
                const bool after_number = k > 0 && (h[k - 1].kind == 2 ||
                                                    (k > 1 && h[k - 2].kind == 2 && h[k - 1].kind == 0 &&
                                                     head.compare(h[k - 1].begin, h[k - 1].end - h[k - 1].begin,
                                                                  " ") == 0));
                if (fictive_street_types().count(f) || f == "bis" || f == "ter" || f == "quater" ||
                    (fictive_letter_count(w) == 1 && after_number))
                {
                    article.clear();
                    out += w;
                    continue;
                }
                const std::size_t suffix = fictive_german_street_suffix(f);
                if (suffix)
                {
                    const std::string stem = street_word(f, "de", "", c);
                    // The ending as written ("strasse" or "straße"): the shortest end that folds to it.
                    const std::string ending = f.substr(f.size() - suffix);
                    std::size_t cut = w.size();
                    for (std::size_t p = w.size(); p-- > 0;)
                    {
                        if ((static_cast<unsigned char>(w[p]) & 0xC0) == 0x80) continue;
                        if (fold_key(w.substr(p)) == ending)
                        {
                            cut = p;
                            break;
                        }
                    }
                    out += fictive_cased(stem + lowercase(w.substr(cut)), fictive_case(w));
                    last = c.bump;
                    article.clear();
                    continue;
                }
                // A name of person in the street name: a run of capitalised words without article.
                if (article.empty() && fictive_case(w) == 0 && known_given(f))
                {
                    std::size_t e = k;
                    while (e + 2 < h.size() && h[e + 1].kind == 0 && head.compare(h[e + 1].begin, 1, " ") == 0 &&
                           h[e + 1].end - h[e + 1].begin == 1 && h[e + 2].kind == 1 &&
                           fictive_case(head.substr(h[e + 2].begin, h[e + 2].end - h[e + 2].begin)) == 0)
                        e += 2;
                    if (e > k)
                    {
                        draw_context pc = c;
                        out += person(head.substr(h[k].begin, h[e].end - h[k].begin), pc);
                        last = pc.bump;
                        k = e;
                        continue;
                    }
                }
                if (article.empty() && lang == "fr")
                    out += fictive_cased(family_name(f, c), fictive_case(w));
                else
                    out += fictive_cased(street_word(f, lang, article, c), fictive_case(w));
                last = c.bump;
                article.clear();
            }
            if (tail < v.size())
            {
                // A city the lexicon does not know takes a city of the country of the street.
                draw_context lc = c;
                const privacy_impl::fictive_language* street = privacy_impl::fictive_language_of(lang);
                const std::string where = lang == "en" || !street ? std::string() : street->country;
                out += location(v.substr(tail), lc, where);
                if (last.empty()) last = lc.bump;
            }
            c.bump = last;
            return out;
        }

    // ------------------------------------------------------------------------------------
    // Addresses of the internet.

        // The fictive host of a host name: a subdomain of example.com, example.org or example.net.
        std::string host(const std::string& name, draw_context& c) const
        {
            using namespace privacy_impl;
            std::vector<std::string> labels;
            std::string label;
            for (char ch : lowercase(name))
            {
                if (ch == '.')
                {
                    labels.push_back(label);
                    label.clear();
                }
                else label.push_back(ch);
            }
            labels.push_back(label);
            while (!labels.empty() && labels.back().empty()) labels.pop_back();
            if (labels.empty()) return "example.com";
            const std::string tld = labels.size() >= 2 ? labels.back() : std::string();
            std::size_t reg = labels.size() >= 2 ? labels.size() - 2 : 0;
            static const std::unordered_set<std::string> second = make_word_set("co com org net gov gouv ac edu");
            if (labels.size() >= 3 && tld.size() == 2 && second.count(labels[reg])) --reg;
            static const std::unordered_set<std::string> generic = make_word_set(
                "www mail smtp webmail mx intranet extranet portail portal api app login sso vpn owa autodiscover "
                "ftp m");
            const std::string base = tld == "org" ? "example.org" : tld == "net" ? "example.net" : "example.com";
            std::string out;
            for (std::size_t k = 0; k <= reg; ++k)
            {
                const std::string& l = labels[k];
                std::string w = generic.count(l) && k < reg ? l : fictive_ascii(family_name(fictive_ascii(l), c));
                if (w.empty()) w = "mail";
                out += w + ".";
            }
            return out + base;
        }

        // A word of the local part of an address or of a path: a given name, an initial, a common word, a family name.
        std::string name_word(const std::string& w, bool keep_common, draw_context& c) const
        {
            using namespace privacy_impl;
            const std::string f = fold_key(w);
            if (fictive_letter_count(w) == 1)
            {
                const char i = initial_of(f[0]);
                return std::string(1, fictive_case(w) == 1 ? static_cast<char>(i | 0x20) : i);
            }
            if (keep_common && (common_words().count(f) || url_kept_words().count(f) || function_words().count(f)))
                return w;
            const std::string n = known_given(f) ? given_name(f, 0, c) : family_name(f, c);
            const std::string a = fictive_ascii(n);
            const int cs = fictive_case(w);
            return cs == 2 ? uppercase(a) : cs == 1 ? a : titlecase(a);
        }

        std::string email(const std::string& v, draw_context& c) const
        {
            using namespace privacy_impl;
            const std::size_t at = v.rfind('@');
            if (at == std::string::npos || at == 0 || at + 1 >= v.size()) return identifier(v, c);
            const std::string local = v.substr(0, at);
            std::string out, last;
            for (const fictive_token& k : fictive_tokens(local))
            {
                const std::string w = local.substr(k.begin, k.end - k.begin);
                if (k.kind == 1)
                {
                    out += name_word(w, true, c);
                    last = c.bump;
                }
                else if (k.kind == 2)
                {
                    privacy_impl::fictive_stream st = stream("ED\x1f" + lowercase(v) + "\x1f" + w, c);
                    out += st.digits(w.size());
                }
                else out += w;
            }
            out += "@" + host(v.substr(at + 1), c);
            if (last.empty()) last = c.bump;
            c.bump = last;
            return out;
        }

        std::string url(const std::string& v, draw_context& c) const
        {
            using namespace privacy_impl;
            const std::string lower = lowercase(v);
            if (lower.compare(0, 7, "mailto:") == 0) return v.substr(0, 7) + email(v.substr(7), c);
            std::size_t hb = lower.find("://");
            hb = hb == std::string::npos ? 0 : hb + 3;
            std::size_t he = v.find_first_of("/?#:", hb);
            if (he == std::string::npos) he = v.size();
            std::string out = v.substr(0, hb);
            const std::string name = v.substr(hb, he - hb);
            std::uint32_t a4 = 0;
            if (parse_ipv4(name, a4)) out += ipv4(name, c);
            else out += host(name, c);
            const std::string host_key = c.bump;
            const std::string rest = v.substr(he);
            std::size_t pos = 0;
            // A port stays.
            if (!rest.empty() && rest[0] == ':')
            {
                std::size_t e = 1;
                while (e < rest.size() && digit_char(rest[e])) ++e;
                out += rest.substr(0, e);
                pos = e;
            }
            const std::string tail = rest.substr(pos);
            for (const fictive_token& k : fictive_tokens(tail))
            {
                const std::string w = tail.substr(k.begin, k.end - k.begin);
                // The short words of a path and the names of the parameters stay ("id", "q", "lang").
                if (k.kind == 1 && (privacy_impl::fictive_letter_count(w) <= 3 ||
                                    (k.end < tail.size() && tail[k.end] == '=')))
                    out += w;
                else if (k.kind == 1) out += name_word(w, true, c);
                else if (k.kind == 2)
                {
                    privacy_impl::fictive_stream st = stream("UD\x1f" + v + "\x1f" + std::to_string(k.begin), c);
                    out += st.digits(w.size());
                }
                else out += w;
            }
            c.bump = host_key;
            return out;
        }

    // ------------------------------------------------------------------------------------
    // Numbers.

        static std::string home_country_code(const std::string& country)
        {
            if (country == "fr") return "33";
            if (country == "gb") return "44";
            if (country == "de") return "49";
            if (country == "es") return "34";
            if (country == "it") return "39";
            return "1";
        }

        /*
            The country of a number written in its national form: the primary country when the number
            has its shape, else the first country of the languages of the corpus whose shape it has,
            else the first country of the lexicons whose shape it has, North America before Italy for
            ten digits without trunk prefix, else the primary country.
        */
        std::string national_country(const std::string& d) const
        {
            auto shaped = [&d](const std::string& country) {
                const std::size_t n = d.size();
                if (country == "fr") return n == 10 && d[0] == '0' && d[1] != '0';
                if (country == "gb") return n == 11 && d[0] == '0' && d[1] != '0';
                if (country == "de") return n >= 9 && n <= 12 && d[0] == '0' && d[1] != '0';
                if (country == "es") return n == 9 && d[0] >= '6' && d[0] <= '9';
                if (country == "it") return (n >= 9 && n <= 11 && d[0] == '0') || (n == 10 && d[0] == '3');
                return (n == 10 && d[0] >= '2' && d[0] <= '9') || (n == 11 && d[0] == '1');
            };
            if (shaped(country_)) return country_;
            for (const std::string& l : options_.languages)
            {
                const std::string country = privacy_impl::fictive_language_of(l)->country;
                if (shaped(country)) return country;
            }
            for (const char* country : { "fr", "gb", "de", "es", "us", "it" })
                if (shaped(country)) return country;
            return country_;
        }

        // A national significant number of country code cc in a range reserved for fiction when there is one.
        static std::string fictive_national(const std::string& cc, const std::string& nsn,
                                            privacy_impl::fictive_stream& st)
        {
            if (cc == "33" && nsn.size() == 9)
            {
                static const char* const blocks[] = { "19900", "26191", "35301", "46571", "53649" };
                const char d = nsn[0];
                if (d >= '1' && d <= '5') return blocks[d - '1'] + st.digits(4);
                if (d == '6' || d == '7') return "63998" + st.digits(4);
                return "19900" + st.digits(4);
            }
            if (cc == "1" && nsn.size() == 10) return nsn.substr(0, 3) + "55501" + st.digits(2);
            if (cc == "1" && nsn.size() == 7) return "55501" + st.digits(2);
            if (cc == "44" && !nsn.empty())
            {
                if (nsn[0] == '7') return "7700900" + st.digits(3);
                if (nsn.compare(0, 2, "20") == 0) return "2079460" + st.digits(3);
                static const char* const areas[] = { "113", "114", "115", "116", "117", "118", "121", "131", "141",
                                                     "151", "161" };
                for (const char* a : areas)
                    if (nsn.compare(0, 3, a) == 0) return std::string(a) + "4960" + st.digits(3);
                if (nsn.compare(0, 3, "191") == 0) return "1914980" + st.digits(3);
                if (nsn.compare(0, 2, "28") == 0) return "2896496" + st.digits(3);
                if (nsn.compare(0, 2, "29") == 0) return "2920180" + st.digits(3);
                return "1632960" + st.digits(3);
            }
            if (cc == "49" && nsn.size() >= 2)
            {
                if (nsn[0] == '1' && (nsn[1] == '5' || nsn[1] == '6' || nsn[1] == '7'))
                    return st.below(2) ? "17139200" + st.digits(2) : "176040690" + st.digits(2);
                if (nsn.compare(0, 2, "69") == 0) return "6990009" + st.digits(3);
                if (nsn.compare(0, 2, "40") == 0) return "4066969" + st.digits(3);
                if (nsn.compare(0, 3, "221") == 0) return "2214710" + st.digits(3);
                if (nsn.compare(0, 2, "89") == 0) return "8999998" + st.digits(3);
                return "3023125" + st.digits(3);
            }
            // No range of fiction is published: the first two digits stay, the others are drawn.
            std::string s = nsn.substr(0, std::min<std::size_t>(2, nsn.size()));
            while (s.size() < nsn.size()) s.push_back(st.digit());
            return s;
        }

        std::string phone(const std::string& v, draw_context& c) const
        {
            std::string d;
            for (char ch : v)
                if (digit_char(ch)) d.push_back(ch);
            if (d.size() < 6) return drawn_shape(v, "PHONE", c);
            const std::size_t first = v.find_first_not_of(" \t(");
            const bool plus = first != std::string::npos && v[first] == '+';
            const bool zeros = !plus && d.compare(0, 2, "00") == 0;
            std::string cc, lead, nsn;
            if (plus || zeros)
            {
                const std::string a = zeros ? d.substr(2) : d;
                const std::size_t n = privacy_impl::e164_country_code_length(a);
                cc = a.substr(0, std::min(n, a.size()));
                nsn = a.size() > n ? a.substr(n) : std::string();
                if (!nsn.empty() && nsn[0] == '0' && cc != "39")
                {
                    lead = "0";
                    nsn = nsn.substr(1);
                }
            }
            else
            {
                cc = home_country_code(national_country(d));
                if (cc == "1")
                {
                    if (d.size() == 11 && d[0] == '1') { lead = "1"; nsn = d.substr(1); }
                    else nsn = d;
                }
                else if (cc == "34" || cc == "39") nsn = d;
                else if (d[0] == '0') { lead = "0"; nsn = d.substr(1); }
                else nsn = d;
            }
            privacy_impl::fictive_stream st = stream("P\x1f" + cc + "\x1f" + nsn, c);
            std::string fn;
            for (int tries = 0; tries < 16 && (fn.empty() || fn == nsn); ++tries) fn = fictive_national(cc, nsn, st);
            const std::string digits = std::string(zeros ? "00" : "") + (plus || zeros ? cc : "") + lead + fn;
            if (digits.size() == d.size()) return privacy_impl::fictive_with_digits(v, digits);
            // Another length: the number written whole after its prefix.
            const std::string prefix = plus ? "+" + cc + " " : zeros ? "00" + cc + " " : lead;
            return prefix + fn;
        }

        // The digits of v drawn, its letters drawn in their case, its other characters kept.
        std::string drawn_shape(const std::string& v, const char* type, draw_context& c) const
        {
            privacy_impl::fictive_stream st = stream(std::string(type) + "\x1f" + privacy_impl::fold_key(v), c);
            for (int tries = 0; tries < 16; ++tries)
            {
                std::string out;
                std::size_t i = 0;
                while (i < v.size())
                {
                    const std::size_t at = i;
                    const char32_t ch = privacy_impl::utf8_next(v, i);
                    if (privacy_impl::is_ascii_digit(ch)) out.push_back(st.digit());
                    else if (ch < 0x80 && privacy_impl::is_letter(ch))
                        out.push_back(static_cast<char>((privacy_impl::is_upper(ch) ? 'A' : 'a') + st.below(26)));
                    else out.append(v, at, i - at);
                }
                if (out != v || tries == 15) return out;
            }
            return v;
        }

        // The characters of v that are letters or digits, in capitals, and their positions.
        static std::string alnum_of(const std::string& v, std::vector<std::size_t>* positions = nullptr)
        {
            std::string a;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                const char ch = v[i];
                if (std::isalnum(static_cast<unsigned char>(ch)))
                {
                    a.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
                    if (positions) positions->push_back(i);
                }
            }
            return a;
        }

        // v with its letters and digits replaced in order by those of a, the case of each place kept.
        static std::string with_alnum(const std::string& v, const std::vector<std::size_t>& positions,
                                      const std::string& a)
        {
            std::string out = v;
            for (std::size_t k = 0; k < positions.size() && k < a.size(); ++k)
            {
                const bool lower = std::islower(static_cast<unsigned char>(v[positions[k]])) != 0;
                out[positions[k]] = lower ? static_cast<char>(std::tolower(static_cast<unsigned char>(a[k]))) : a[k];
            }
            return out;
        }

        static int mod97(const std::string& digits)
        {
            int r = 0;
            for (char ch : digits) r = (r * 10 + (ch - '0')) % 97;
            return r;
        }

        static std::string two_digits(int v)
        {
            return std::string(1, static_cast<char>('0' + v / 10)) + static_cast<char>('0' + v % 10);
        }

        std::string iban(const std::string& v, draw_context& c) const
        {
            std::vector<std::size_t> pos;
            const std::string a = alnum_of(v, &pos);
            if (a.size() < 15 || !upper_ascii(a[0]) || !upper_ascii(a[1])) return drawn_shape(v, "IBAN", c);
            const std::string cc = a.substr(0, 2);
            privacy_impl::fictive_stream st = stream("IBAN\x1f" + a, c);
            for (int tries = 0; tries < 16; ++tries)
            {
                std::string bban;
                if (cc == "FR" && a.size() == 27)
                {
                    // A test establishment of the Banque de France, a branch, an account and the key of the RIB.
                    const std::string bank = "990" + st.digits(2), branch = st.digits(5), account = st.digits(11);
                    const int key = 97 - (89 * mod97(bank) + 15 * mod97(branch) + 3 * mod97(account)) % 97;
                    bban = bank + branch + account + two_digits(key);
                }
                else
                {
                    for (std::size_t k = 4; k < a.size(); ++k)
                        bban.push_back(upper_ascii(a[k]) ? static_cast<char>('A' + st.below(26)) : st.digit());
                }
                const std::string out = cc + privacy_impl::iban_check_digits(cc, bban) + bban;
                if (out != a) return with_alnum(v, pos, out);
            }
            return v;
        }

        std::string card(const std::string& v, draw_context& c) const
        {
            std::string d;
            for (char ch : v)
                if (digit_char(ch)) d.push_back(ch);
            if (d.size() < 12 || d.size() > 19) return drawn_shape(v, "CARD", c);
            std::string prefix = "411111";
            if (d[0] == '5' || d[0] == '2') prefix = "555555";
            else if (d[0] == '3') prefix = "378282";
            else if (d[0] == '6') prefix = "601111";
            privacy_impl::fictive_stream st = stream("CARD\x1f" + d, c);
            for (int tries = 0; tries < 16; ++tries)
            {
                std::string out = prefix + st.digits(d.size() - prefix.size() - 1);
                out.push_back(privacy_impl::luhn_check_digit(out));
                if (out != d) return privacy_impl::fictive_with_digits(v, out);
            }
            return v;
        }

        std::string nir(const std::string& v, draw_context& c) const
        {
            std::vector<std::size_t> pos;
            const std::string a = alnum_of(v, &pos);
            if (a.size() != 13 && a.size() != 15) return drawn_shape(v, "NIR", c);
            privacy_impl::fictive_stream st = stream("NIR\x1f" + a, c);
            for (int tries = 0; tries < 16; ++tries)
            {
                std::string n;
                n.push_back(a[0] == '1' || a[0] == '2' || a[0] == '7' || a[0] == '8' ? a[0] : '1');
                n += st.digits(2);
                // A month from 13 to 19, which the register does not give.
                n += "1";
                n.push_back(static_cast<char>('3' + st.below(7)));
                int department = 1 + static_cast<int>(st.below(95));
                if (department == 20) department = 21;
                n += two_digits(department);
                const int commune = 1 + static_cast<int>(st.below(989)), order = 1 + static_cast<int>(st.below(999));
                n += std::to_string(1000 + commune).substr(1);
                n += std::to_string(1000 + order).substr(1);
                n = n.substr(0, 13);
                if (a.size() == 15)
                {
                    long long r = 0;
                    for (char ch : n) r = (r * 10 + (ch - '0')) % 97;
                    n += two_digits(static_cast<int>(97 - r));
                }
                if (n != a) return with_alnum(v, pos, n);
            }
            return v;
        }

        std::string siret(const std::string& v, draw_context& c) const
        {
            std::string d;
            for (char ch : v)
                if (digit_char(ch)) d.push_back(ch);
            if (d.size() != 9 && d.size() != 14) return drawn_shape(v, "SIRET", c);
            // The register reserves no block for tests: the first digit keeps its class, 1 or 2 for a
            // public body and 3 to 9 for the others, and the other digits are drawn.
            const bool public_body = d[0] == '1' || d[0] == '2';
            std::string siren;
            {
                privacy_impl::fictive_stream st = stream("SIREN\x1f" + d.substr(0, 9), c);
                for (int tries = 0; tries < 16 && (siren.empty() || siren == d.substr(0, 9)); ++tries)
                {
                    siren.assign(1, static_cast<char>(public_body ? '1' + st.below(2) : '3' + st.below(7)));
                    siren += st.digits(7);
                    siren.push_back(privacy_impl::luhn_check_digit(siren));
                }
            }
            std::string out = siren;
            if (d.size() == 14)
            {
                privacy_impl::fictive_stream st = stream("SIRET\x1f" + d, c);
                out += st.digits(4);
                out.push_back(privacy_impl::luhn_check_digit(out));
            }
            return privacy_impl::fictive_with_digits(v, out);
        }

        std::string ipv4(const std::string& v, draw_context& c) const
        {
            // Read with leading zeros too ("010.001.002.003").
            std::uint32_t a = 0;
            int parts = 0;
            std::size_t i = 0;
            bool ok = true;
            while (ok && parts < 4)
            {
                std::size_t j = i;
                int n = 0;
                while (j < v.size() && digit_char(v[j]) && j - i < 3) n = n * 10 + (v[j++] - '0');
                ok = j > i && n <= 255;
                a = (a << 8) | static_cast<std::uint32_t>(n);
                ++parts;
                i = j;
                if (parts < 4) ok = ok && i < v.size() && v[i] == '.' && ++i;
            }
            if (!ok || i != v.size()) return drawn_shape(v, "IPV4", c);
            static const char* const nets[] = { "192.0.2.", "198.51.100.", "203.0.113." };
            privacy_impl::fictive_stream net(mac_, "IP4N\x1f" + std::to_string(a >> 8));
            const char* base = nets[net.below(3)];
            privacy_impl::fictive_stream st = stream("IP4\x1f" + std::to_string(a), c);
            for (int tries = 0; tries < 16; ++tries)
            {
                const std::string out = base + std::to_string(1 + st.below(254));
                if (out != v) return out;
            }
            return v;
        }

        std::string ipv6(const std::string& v, draw_context& c) const
        {
            std::array<unsigned char, 16> a{};
            if (!privacy_impl::parse_ipv6(v, a)) return drawn_shape(v, "IPV6", c);
            std::array<unsigned char, 16> b{};
            b[0] = 0x20;
            b[1] = 0x01;
            b[2] = 0x0d;
            b[3] = 0xb8;
            // Each group drawn from the groups before it: a prefix keeps its fictive prefix.
            const std::string key = "IP6\x1f" + std::string(reinterpret_cast<const char*>(a.data()), a.size());
            const unsigned attempt = c.state ? c.state->attempt(key) : 0;
            for (std::size_t g = 2; g < 8; ++g)
            {
                const std::string prefix(reinterpret_cast<const char*>(a.data()), 2 * g + 2);
                const std::string last = g == 7 ? "\x1f" + std::to_string(attempt) : std::string();
                privacy_impl::fictive_stream st(mac_, "IP6\x1f" + prefix + last);
                const std::uint64_t r = st.next();
                b[2 * g] = static_cast<unsigned char>(r >> 8);
                b[2 * g + 1] = static_cast<unsigned char>(r);
            }
            c.bump = key;
            std::string out = privacy_impl::render_ipv6(b);
            if (std::any_of(v.begin(), v.end(), [](char ch) { return ch >= 'A' && ch <= 'F'; }))
                out = privacy_impl::uppercase(out);
            return out;
        }

        std::string mac(const std::string& v, draw_context& c) const
        {
            std::vector<std::size_t> pos;
            const std::string a = alnum_of(v, &pos);
            if (a.size() != 12 || !std::all_of(a.begin(), a.end(), [](char ch) { return std::isxdigit(
                                                    static_cast<unsigned char>(ch)) != 0; }))
                return drawn_shape(v, "MAC", c);
            static const char hex[] = "0123456789ABCDEF";
            privacy_impl::fictive_stream st = stream("MAC\x1f" + a, c);
            std::string out = "02";
            for (int k = 0; k < 10; ++k) out.push_back(hex[st.below(16)]);
            return with_alnum(v, pos, out);
        }

        std::string identifier(const std::string& v, draw_context& c) const
        {
            using namespace privacy_impl;
            // An account name: letters in lower case, with dots, hyphens or underscores between them.
            const bool account = !v.empty() && std::all_of(v.begin(), v.end(), [](char ch) {
                return (ch >= 'a' && ch <= 'z') || ch == '.' || ch == '_' || ch == '-';
            }) && std::any_of(v.begin(), v.end(), [](char ch) { return ch >= 'a' && ch <= 'z'; });
            if (account)
            {
                const std::vector<fictive_token> t = fictive_tokens(v);
                std::string out, last;
                std::size_t words = 0;
                for (const fictive_token& k : t) words += k.kind == 1;
                for (const fictive_token& k : t)
                {
                    const std::string w = v.substr(k.begin, k.end - k.begin);
                    if (k.kind != 1) { out += w; continue; }
                    if (words == 1 && w.size() >= 5 && !known_given(w))
                    {
                        // "jdupont": an initial and a family name.
                        out.push_back(static_cast<char>(initial_of(w[0]) | 0x20));
                        out += fictive_ascii(family_name(w.substr(1), c));
                    }
                    else out += name_word(w, false, c);
                    last = c.bump;
                }
                c.bump = last;
                if (out != v) return out;
            }
            if (is_uuid_shape(v))
            {
                static const char hex[] = "0123456789abcdef";
                privacy_impl::fictive_stream st = stream("UUID\x1f" + lowercase(v), c);
                std::string out = v;
                for (std::size_t k = 0; k < out.size(); ++k)
                {
                    if (out[k] == '-' || k == 14 || k == 19) continue;
                    const char ch = hex[st.below(16)];
                    out[k] = std::isupper(static_cast<unsigned char>(v[k])) ? static_cast<char>(std::toupper(ch)) : ch;
                }
                return out;
            }
            // Other identifiers: a leading country code stays, the Luhn digit of a long number is valid.
            static const std::unordered_set<std::string> countries = make_word_set(
                "fr de es it gb uk be ch lu nl at pt ie pl se dk fi no cz sk hu ro bg gr hr si ee lv lt mt cy us ca");
            std::size_t keep = 0;
            if (v.size() > 4 && upper_ascii(v[0]) && upper_ascii(v[1]) && digit_char(v[2]) &&
                countries.count(lowercase(v.substr(0, 2))))
                keep = 2;
            std::string d;
            for (char ch : v)
                if (digit_char(ch)) d.push_back(ch);
            const bool luhn = keep == 0 && d.size() >= 8 && d.size() == alnum_of(v).size() && luhn_ok(d);
            const std::string body = drawn_shape(v.substr(keep), "ID", c);
            std::string out = v.substr(0, keep) + body;
            if (luhn)
            {
                std::string nd;
                for (char ch : out)
                    if (digit_char(ch)) nd.push_back(ch);
                nd.back() = luhn_check_digit(nd.substr(0, nd.size() - 1));
                out = fictive_with_digits(out, nd);
            }
            return out;
        }

        std::string plate(const std::string& v, draw_context& c) const
        {
            std::vector<std::size_t> pos;
            const std::string a = alnum_of(v, &pos);
            static const char letters[] = "ABCDEFGHJKLMNPQRSTVWXYZ";
            privacy_impl::fictive_stream st = stream("PLATE\x1f" + a, c);
            auto letter = [&]() { return letters[st.below(sizeof(letters) - 1)]; };
            // A plate of the SIV, AA-123-AA: the first letter X, Y or Z.
            if (a.size() == 7 && upper_ascii(a[0]) && upper_ascii(a[1]) && digit_char(a[2]) && digit_char(a[3]) &&
                digit_char(a[4]) && upper_ascii(a[5]) && upper_ascii(a[6]))
            {
                for (int tries = 0; tries < 16; ++tries)
                {
                    std::string out;
                    out.push_back("XYZ"[st.below(3)]);
                    out.push_back(letter());
                    out += std::to_string(1001 + st.below(999)).substr(1);
                    out.push_back(letter());
                    out.push_back(letter());
                    if (out != a) return with_alnum(v, pos, out);
                }
            }
            // A plate of the FNI, 1234 AB 35: the department stays.
            if (a.size() >= 6 && digit_char(a[a.size() - 1]) && digit_char(a[a.size() - 2]) && digit_char(a[0]))
            {
                std::string out;
                for (std::size_t k = 0; k + 2 < a.size(); ++k) out.push_back(digit_char(a[k]) ? st.digit() : letter());
                out += a.substr(a.size() - 2);
                if (out != a) return with_alnum(v, pos, out);
            }
            return drawn_shape(v, "PLATE", c);
        }

        std::string geolocation(const std::string& v, draw_context& c) const
        {
            privacy_impl::fictive_stream st = stream("GEO\x1f" + v, c);
            std::string out;
            bool fraction = false, after_mark = false;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                const char ch = v[i];
                if (!digit_char(ch))
                {
                    if (ch == '.' || ch == ',') fraction = i > 0 && digit_char(v[i - 1]);
                    else fraction = false;
                    // Minutes and seconds after a degree mark or a minute mark stay below 60.
                    after_mark = ch == '\'' || (static_cast<unsigned char>(ch) == 0xB0) ||
                                 (static_cast<unsigned char>(ch) == 0xC2 && i + 1 < v.size() &&
                                  static_cast<unsigned char>(v[i + 1]) == 0xB0);
                    if (static_cast<unsigned char>(ch) == 0xB0) after_mark = true;
                    out.push_back(ch);
                    continue;
                }
                if (fraction) out.push_back(st.digit());
                else if (after_mark && i + 1 < v.size() && digit_char(v[i + 1]))
                    out.push_back(static_cast<char>('0' + st.below(6)));
                else if (after_mark) out.push_back(st.digit());
                else out.push_back(ch);
                if (!fraction && !(i + 1 < v.size() && digit_char(v[i + 1]))) after_mark = false;
            }
            return out;
        }

        std::string date(const std::string& v, draw_context& c) const
        {
            civil_date d;
            date_format f;
            if (parse_date(v, options_.date_order, d, f) && format_date(d, f) == v)
            {
                const long days = date_shift_days(c.document);
                civil_date s = d;
                if (!f.has_month())
                    s.year = static_cast<int>(d.year + std::lround(days / 365.2425));
                else if (!f.has_day())
                {
                    const long m = 12L * d.year + (d.month - 1) + std::lround(days / 30.436875);
                    s.year = static_cast<int>(m / 12);
                    s.month = static_cast<int>(m % 12 + 1);
                }
                else
                    s = civil_from_days(days_from_civil(d.year, d.month, d.day) + days);
                const std::string out = format_date(s, f);
                if (!out.empty()) return out;
            }
            return drawn_shape(v, "DATE", c);
        }

        /*
            A secret: each character takes a character of its class that the secret does not hold, or
            of its class when the secret holds them all, and no run of four characters of the
            secret is left. The armour lines of a PEM block stay.
        */
        std::string secret(const std::string& v, draw_context& c) const
        {
            static const std::string lower = "abcdefghijklmnopqrstuvwxyz", upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                     digits = "0123456789", symbols = "!#$%&()*+,-./:;<=>?@[]^_{|}~";
            bool used[256] = { false };
            for (unsigned char ch : v) used[ch] = true;
            auto pool_of = [&](const std::string& cls) {
                std::string p;
                for (char ch : cls)
                    if (!used[static_cast<unsigned char>(ch)]) p.push_back(ch);
                return p.empty() ? cls : p;
            };
            const std::string pl = pool_of(lower), pu = pool_of(upper), pd = pool_of(digits), ps = pool_of(symbols),
                              pb = pool_of("+/");
            privacy_impl::fictive_stream st = stream("K\x1f" + v, c);
            const bool pem = v.compare(0, 5, "-----") == 0;
            std::string out;
            std::vector<const std::string*> cls;
            std::size_t i = 0;
            bool armour = false, line_start = true;
            while (i < v.size())
            {
                const std::size_t at = i;
                const char32_t ch = privacy_impl::utf8_next(v, i);
                if (line_start) armour = pem && (v.compare(at, 5, "-----") == 0 ||
                                                 v.find(':', at) < v.find('\n', at));
                line_start = ch == '\n';
                if (armour || ch == '\n' || ch == '\r' || (pem && ch == '='))
                {
                    out.append(v, at, i - at);
                    cls.push_back(nullptr);
                    continue;
                }
                const std::string* p = &pl;
                if (ch >= 'A' && ch <= 'Z') p = &pu;
                else if (ch >= '0' && ch <= '9') p = &pd;
                else if (pem && (ch == '+' || ch == '/')) p = &pb;
                else if (ch < 0x80 && !(ch >= 'a' && ch <= 'z')) p = &ps;
                out.push_back((*p)[st.below(p->size())]);
                cls.push_back(p);
            }
            // No run of four characters of the secret left.
            if (v.size() >= 4 && out.size() == cls.size())
            {
                std::unordered_set<std::string> grams;
                for (std::size_t k = 0; k + 4 <= v.size(); ++k) grams.insert(v.substr(k, 4));
                for (std::size_t k = 0; k + 4 <= out.size(); ++k)
                {
                    int tries = 0;
                    while (grams.count(out.substr(k, 4)) && tries < 64)
                    {
                        std::size_t m = k + 3;
                        while (m > k && !cls[m]) --m;
                        if (!cls[m]) break;
                        out[m] = (*cls[m])[st.below(cls[m]->size())];
                        ++tries;
                    }
                }
            }
            return out;
        }

        fictive_options options_;
        hmac_sha256 mac_;
        std::string language_, country_;
    };

// ----------------------------------------------------------------------------------------

    class fictive_batch
    {
    public:
        explicit fictive_batch(const fictive_substituter& s) : s_(s) { renew(); }

        ~fictive_batch() { erase(); }

        fictive_batch(const fictive_batch&) = delete;
        fictive_batch& operator=(const fictive_batch&) = delete;

        std::string record(pii_type type, const std::string& value)
        {
            if (type == pii_type::title || type == pii_type::gender || type == pii_type::nationality ||
                type == pii_type::none)
                return value;
            values_.push_back(std::make_pair(type, value));
            return "fv" + nonce_ + std::to_string(values_.size() - 1) + "q";
        }

        std::vector<std::string> finish(
            const std::vector<std::string>& written,
            const std::vector<std::string>& documents
        )
        {
            using namespace privacy_impl;
            // Where each placeholder stands in each text.
            struct place { std::size_t text, begin, end, index; };
            std::vector<place> places;
            const std::string mark = "fv" + nonce_;
            for (std::size_t t = 0; t < written.size(); ++t)
            {
                const std::string& w = written[t];
                std::size_t pos = 0;
                while ((pos = w.find(mark, pos)) != std::string::npos)
                {
                    std::size_t e = pos + mark.size();
                    std::size_t index = 0;
                    const std::size_t digits_begin = e;
                    while (e < w.size() && w[e] >= '0' && w[e] <= '9' && e - digits_begin < 12)
                        index = index * 10 + static_cast<std::size_t>(w[e++] - '0');
                    if (e == digits_begin || e >= w.size() || w[e] != 'q' || index >= values_.size())
                    {
                        pos += mark.size();
                        continue;
                    }
                    places.push_back(place{ t, pos, e + 1, index });
                    pos = e + 1;
                }
            }
            // The originals of the batch, by type, and the sexes their civilities tell.
            std::unordered_map<int, std::unordered_set<std::string>> originals;
            for (const auto& v : values_) originals[static_cast<int>(v.first)].insert(normal(v.first, v.second));
            fictive_substituter::batch_state state;
            for (const place& p : places)
            {
                const auto& v = values_[p.index];
                if (v.first != pii_type::person) continue;
                const int sex = fictive_civility_before(written[p.text], p.begin);
                if (sex == 0) continue;
                const std::string g = s_.first_given(v.second);
                if (g.empty()) continue;
                const auto it = state.sexes.find(g);
                if (it == state.sexes.end()) state.sexes[g] = sex;
                else if (it->second != sex) it->second = -1;
            }
            std::vector<std::string> out(written.size());
            std::vector<std::size_t> copied(written.size(), 0);
            for (const place& p : places)
            {
                const auto& v = values_[p.index];
                fictive_substituter::draw_context c;
                c.state = &state;
                c.document = p.text < documents.size() ? documents[p.text] : std::string();
                if (v.first == pii_type::person) c.civility = fictive_civility_before(written[p.text], p.begin);
                std::string fictive;
                for (int tries = 0; tries < 16; ++tries)
                {
                    c.bump.clear();
                    fictive = s_.write(v.first, v.second, c);
                    if (v.first == pii_type::date || c.bump.empty()) break;
                    const std::string n = normal(v.first, fictive);
                    if (n != normal(v.first, v.second) && !originals[static_cast<int>(v.first)].count(n)) break;
                    ++state.attempts[c.bump];
                }
                std::string& o = out[p.text];
                o.append(written[p.text], copied[p.text], p.begin - copied[p.text]);
                o += fictive;
                copied[p.text] = p.end;
                secure_wipe_string(fictive);
            }
            for (std::size_t t = 0; t < written.size(); ++t) out[t].append(written[t], copied[t], std::string::npos);
            erase();
            renew();
            return out;
        }

    private:
        // The form under which two values of a type are the same value.
        static std::string normal(pii_type type, const std::string& v)
        {
            switch (type)
            {
                case pii_type::phone:
                case pii_type::payment_card:
                case pii_type::siret:
                {
                    std::string d;
                    for (char ch : v)
                        if (ch >= '0' && ch <= '9') d.push_back(ch);
                    return d;
                }
                case pii_type::iban:
                case pii_type::nir:
                case pii_type::mac_address:
                {
                    std::string a;
                    for (char ch : v)
                        if (std::isalnum(static_cast<unsigned char>(ch)))
                            a.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
                    return a;
                }
                case pii_type::secret: return v;
                default: return privacy_impl::fold_key(v);
            }
        }

        static void secure_wipe_string(std::string& s)
        {
            if (!s.empty()) secure_wipe(&s[0], s.size());
            s.clear();
        }

        void erase()
        {
            for (auto& v : values_) secure_wipe_string(v.second);
            values_.clear();
        }

        void renew()
        {
            static const char hex[] = "0123456789abcdef";
            const pseudonymization_key k = pseudonymization_key::generate();
            nonce_.clear();
            for (std::size_t i = 0; i < 8; ++i)
            {
                const unsigned char b = static_cast<unsigned char>(k.bytes()[i]);
                nonce_.push_back(hex[b >> 4]);
                nonce_.push_back(hex[b & 15]);
            }
        }

        const fictive_substituter& s_;
        std::string nonce_;
        std::vector<std::pair<pii_type, std::string>> values_;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_FICTIVE_VALUES_H_
