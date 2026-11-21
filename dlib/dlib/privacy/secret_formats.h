// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_SECRET_FORMATS_H_
#define DLIB_PRIVACY_SECRET_FORMATS_H_

#include "secret_formats_abstract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct secret_statistics
    {
        std::size_t length = 0;
        std::size_t lower = 0;
        std::size_t upper = 0;
        std::size_t digits = 0;
        std::size_t joiners = 0;
        std::size_t symbols = 0;
        std::size_t other = 0;
        double entropy = 0;

        std::size_t letters() const { return lower + upper + other; }
    };

    enum class secret_evidence
    {
        format,
        assignment
    };

    struct secret_match
    {
        std::size_t begin = 0;
        std::size_t end = 0;
        secret_evidence evidence = secret_evidence::format;
        double confidence = 0;
        bool web_credentials = false;
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        inline bool secret_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
        inline bool secret_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
        inline bool secret_digit(unsigned char c) { return c >= '0' && c <= '9'; }
        inline bool secret_alnum(unsigned char c) { return secret_lower(c) || secret_upper(c) || secret_digit(c); }
        inline bool secret_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
        inline bool secret_line_break(unsigned char c) { return c == '\n' || c == '\r'; }
        inline bool secret_quote(unsigned char c) { return c == '"' || c == '\'' || c == '`'; }

        inline std::string secret_lowercase(const std::string& s, std::size_t b, std::size_t e)
        {
            std::string out;
            out.reserve(e - b);
            for (std::size_t i = b; i < e; ++i)
                out.push_back(secret_upper(static_cast<unsigned char>(s[i])) ? static_cast<char>(s[i] + 32) : s[i]);
            return out;
        }

        inline bool secret_ends_with(const std::string& s, const char* suffix)
        {
            const std::size_t n = std::strlen(suffix);
            return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
        }

        inline bool secret_starts_with(const std::string& s, std::size_t at, const char* prefix)
        {
            const std::size_t n = std::strlen(prefix);
            return at + n <= s.size() && s.compare(at, n, prefix) == 0;
        }

        inline bool secret_word_in(const std::string& word, const char* list)
        {
            // list holds words separated by single spaces.
            const std::size_t n = word.size();
            if (n == 0) return false;
            for (const char* p = list; *p;)
            {
                const char* q = std::strchr(p, ' ');
                const std::size_t len = q ? static_cast<std::size_t>(q - p) : std::strlen(p);
                if (len == n && std::memcmp(p, word.data(), n) == 0) return true;
                if (!q) break;
                p = q + 1;
            }
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    inline secret_statistics secret_stats(
        const std::string& text,
        std::size_t begin,
        std::size_t end
    )
    {
        using namespace privacy_impl;
        secret_statistics st;
        end = std::min(end, text.size());
        if (begin >= end) return st;
        std::size_t counts[256] = { 0 };
        for (std::size_t i = begin; i < end; ++i)
        {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            ++counts[c];
            if (secret_lower(c)) ++st.lower;
            else if (secret_upper(c)) ++st.upper;
            else if (secret_digit(c)) ++st.digits;
            else if (c >= 0x80) ++st.other;
            else if (c == '-' || c == '_' || c == '.' || c == '\'') ++st.joiners;
            else ++st.symbols;
        }
        st.length = end - begin;
        const double n = static_cast<double>(st.length);
        for (std::size_t k = 0; k < 256; ++k)
        {
            if (counts[k] == 0) continue;
            const double p = static_cast<double>(counts[k]) / n;
            st.entropy -= p * std::log2(p);
        }
        return st;
    }

    inline bool secret_like(
        const secret_statistics& st
    )
    {
        if (st.length < 4) return false;
        // Letters with digits, or a sign among letters or digits: "Tournesol2024", "p@ssword".
        if (st.digits > 0 && st.letters() > 0) return true;
        if (st.symbols > 0 && st.letters() + st.digits > 0) return true;
        // A long value whose characters repeat little: a key written in letters alone.
        return st.length >= 16 && st.entropy >= 3.5 && st.lower + st.upper + st.digits == st.length;
    }

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        /* True when every character of the value is one of the marks that hide a value, or the same
           character: "********", "xxxxxxxx", "XXXX-XXXX", "........", "????". */
        inline bool secret_masked(const std::string& v)
        {
            /* The bullet, the middle dot, the black circle and the ellipsis are written on several bytes:
               E2 80 A2, C2 B7, E2 97 8F and E2 80 A6. */
            auto mark = [](unsigned char c) {
                return c == '*' || c == 'x' || c == 'X' || c == '#' || c == '.' || c == '-' || c == '_' || c == '?' ||
                       c == 0xE2 || c == 0x80 || c == 0xA2 || c == 0xC2 || c == 0xB7 || c == 0x97 || c == 0x8F ||
                       c == 0xA6;
            };
            bool same = true, marks = true;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                const unsigned char c = static_cast<unsigned char>(v[i]);
                if (c != static_cast<unsigned char>(v[0])) same = false;
                if (!mark(c)) marks = false;
            }
            return same || marks;
        }

        // A path of the file system: the place of a key, never the key.
        inline bool secret_path(const std::string& v)
        {
            if (v.size() < 4) return false;
            const bool rooted = v[0] == '/' || v.compare(0, 2, "./") == 0 || v.compare(0, 3, "../") == 0 ||
                                v.compare(0, 2, "~/") == 0 || v.compare(0, 2, ".\\") == 0 ||
                                (secret_alnum(static_cast<unsigned char>(v[0])) && v[1] == ':' &&
                                 (v[2] == '\\' || v[2] == '/'));
            std::size_t separators = 0;
            for (unsigned char c : v)
            {
                if (c == '/' || c == '\\') ++separators;
                else if (!(secret_alnum(c) || c == '_' || c == '.' || c == '-' || c == ':' || c == '~' || c == ' '))
                    return false;
            }
            // A long value that mixes capitals, small letters and digits is a key of base 64, not a path.
            const secret_statistics st = secret_stats(v, 0, v.size());
            if (v.size() >= 20 && st.lower > 0 && st.upper > 0 && st.digits > 0) return false;
            const std::size_t dot = v.rfind('.');
            const bool extension = dot != std::string::npos && dot + 2 <= v.size() && v.size() - dot <= 6 &&
                                   v.find_first_of("/\\", dot) == std::string::npos;
            if (rooted) return separators >= 2 || extension;
            return separators >= 1 && extension;
        }

        // An identifier from b on, its parts joined by periods when dots is true: "getenv", "os.environ".
        inline bool secret_identifier_only(const std::string& v, std::size_t b = 0, bool dots = false)
        {
            if (b >= v.size()) return false;
            for (std::size_t i = b; i < v.size(); ++i)
            {
                const unsigned char c = static_cast<unsigned char>(v[i]);
                if (!(secret_alnum(c) || c == '_' || (dots && c == '.'))) return false;
            }
            return !secret_digit(static_cast<unsigned char>(v[b])) && v[b] != '.';
        }
    }

    inline bool secret_placeholder(
        const std::string& value
    )
    {
        using namespace privacy_impl;
        const std::size_t n = value.size();
        if (n == 0) return true;
        if (secret_masked(value)) return true;
        const char first = value[0], last = value[n - 1];
        // Fields left to fill and references to a value kept elsewhere.
        if ((first == '<' && last == '>') || (first == '[' && last == ']') || (first == '{' && last == '}') ||
            (first == '%' && last == '%' && n >= 3) || (first == '@' && last == '@' && n >= 3) ||
            (first == '(' && last == ')'))
            return true;
        if (value.compare(0, 2, "${") == 0 || value.compare(0, 2, "$(") == 0 || value.compare(0, 2, "#{") == 0 ||
            value.compare(0, 2, "{{") == 0 || value.compare(0, 2, "<%") == 0 || value.compare(0, 2, "__") == 0)
            return true;
        // A variable, an alias, a tag of the file format: "$DB_PASSWORD", "*default", "!vault".
        if ((first == '$' || first == '*' || first == '&') && secret_identifier_only(value, 1)) return true;
        if (first == '!' && n >= 4)
        {
            bool tag = true;
            for (std::size_t i = 1; i < n && tag; ++i)
                tag = secret_lower(static_cast<unsigned char>(value[i])) || value[i] == '_';
            if (tag) return true;
        }
        // A call or an index: the value is computed, "getenv(...)", "getpass.getpass()", "cfg[".
        {
            const std::size_t open = value.find_first_of("([");
            if (open != std::string::npos && open > 0 && secret_identifier_only(value.substr(0, open), 0, true))
                return true;
        }
        const std::string low = secret_lowercase(value, 0, n);
        static const char* const roots[] = { "env.", "process.env", "os.environ", "secrets.", "vault:", "ssm:",
            "arn:aws:", "file:", "secret:", "op://", "self.", "this.", "config.", "settings.", "request.", "req.",
            "args.", "params.", "options.", "ctx.", "props.", "var.", "local.", "data.", "module.", "azurerm_",
            "keyvault", "sm://", "http://", "https://" };
        for (const char* r : roots)
            if (low.compare(0, std::strlen(r), r) == 0) return true;
        if (secret_path(value)) return true;
        // Words that tell the state of a secret, in the five languages, and the values of code.
        static const char* const states =
            "none null nil undefined true false yes no oui non si ja nein tbd todo n/a na nc ok ko "
            "required requis requise obligatoire optionnel optional facultatif aucun aucune neant vide empty "
            "inconnu unknown oublie oubliee forgotten expire expiree expired masque masquee hidden cache cachee "
            "confidentiel confidential redacted removed supprime idem inchange unchanged voir see defini "
            "defined changed modifie reset reinitialise envoye sent fourni provided manquant missing invalide "
            "invalid incorrect correct valide valid obbligatorio obbligatoria facoltativo nessuna nessuno vuoto "
            "vuota sconosciuta dimenticata scaduta nascosta obligatoria obligatorio opcional ninguna ninguno "
            "vacio vacia desconocida olvidada caducada oculta erforderlich keine keins leer unbekannt vergessen "
            "abgelaufen versteckt geheim default defaut string str text secure securestring prompt ask input "
            "enabled disabled on off";
        if (secret_word_in(low, states)) return true;
        // The name of a secret in place of its value: "password", "YOUR_API_KEY", "my-secret-token".
        static const char* const names =
            "password passwd pwd pass secret token key apikey mdp motdepasse contrasena passwort kennwort "
            "passphrase credentials credential";
        if (secret_word_in(low, names)) return true;
        {
            bool letters_only = true, separated = false, capitals = true;
            for (unsigned char c : value)
            {
                if (c == '_' || c == '-' || c == ' ') separated = true;
                else if (!(secret_lower(c) || secret_upper(c))) letters_only = false;
                if (secret_lower(c)) capitals = false;
            }
            if (letters_only && (separated || capitals))
            {
                static const char* const parts[] = { "password", "passwd", "secret", "token", "apikey", "api_key",
                    "api-key", "motdepasse", "mot_de_passe", "passwort", "kennwort", "contrasena", "placeholder",
                    "changeit", "example", "exemple" };
                for (const char* p : parts)
                    if (low.find(p) != std::string::npos) return true;
            }
        }
        return false;
    }

// ----------------------------------------------------------------------------------------

    inline int secret_key_strength(
        const std::string& name
    )
    {
        using namespace privacy_impl;
        /* The words of the name: "DB_PASSWORD", "client-secret", "spring.datasource.password" and
           "apiKey" read "db password", "client secret", "spring datasource password" and "api key". */
        std::vector<std::string> words;
        std::string w;
        for (std::size_t i = 0; i < name.size(); ++i)
        {
            const unsigned char c = static_cast<unsigned char>(name[i]);
            const bool hump = secret_upper(c) && i > 0 && secret_lower(static_cast<unsigned char>(name[i - 1]));
            if (!secret_alnum(c) || hump)
            {
                if (!w.empty()) words.push_back(w);
                w.clear();
                if (!secret_alnum(c)) continue;
            }
            w.push_back(secret_upper(c) ? static_cast<char>(c + 32) : static_cast<char>(c));
        }
        if (!w.empty()) words.push_back(w);
        // "password2", "password_confirmation", "passwordNew" name the secret that "password" names.
        while (!words.empty())
        {
            std::string& tail = words.back();
            while (!tail.empty() && secret_digit(static_cast<unsigned char>(tail.back()))) tail.pop_back();
            if (!tail.empty() && !(words.size() >= 2 && secret_word_in(tail, "confirmation confirm repeat again new "
                                                                              "old current value plain clear")))
                break;
            words.pop_back();
        }
        if (words.empty() || words.size() > 8) return 0;
        if (words.size() >= 3 && words.back() == "passe" && words[words.size() - 2] == "de" &&
            words[words.size() - 3] == "mot")
            return 2;
        const std::string& last = words.back();
        const std::string before = words.size() >= 2 ? words[words.size() - 2] : std::string();

        // Short codes of digits: a PIN, the security code of a card, a code sent for one use.
        if (secret_word_in(last, "pin puk otp totp cvv cvc cvv2 cvn digicode")) return 3;
        if (last == "code" && secret_word_in(before, "pin puk access security verification auth authentication "
                                                     "recovery backup secret otp door gate"))
            return 3;
        static const char* const strong =
            "password passwd pwd passphrase passcode mdp motdepasse contrasena passwort kennwort apikey psk "
            "authorization";
        if (secret_word_in(last, strong)) return 2;
        for (const char* suffix : { "password", "passwd", "passwort", "kennwort", "apikey", "accesstoken",
                                    "authtoken", "secretkey", "privatekey", "accesskey" })
            if (secret_ends_with(last, suffix)) return 2;
        if (last == "pass" && words.size() >= 2) return 2;
        if (last == "secret" && words.size() >= 2) return 2;
        if (last == "token" && secret_word_in(before, "access auth authentication refresh api bearer session id "
                                                      "csrf xsrf bot personal oauth jwt registration deploy "
                                                      "security github gitlab slack npm vault"))
            return 2;
        if (last == "key" && secret_word_in(before, "api secret private access auth encryption decryption signing "
                                                    "master account shared storage subscription ssh wpa wifi hmac "
                                                    "aes jwt session security service webhook app application "
                                                    "consumer license licence crypt cipher"))
            return 2;
        if (last == "signature" && before == "access") return 2;
        // Names that often hold a secret, and as often another value: the value decides.
        if (secret_word_in(last, "secret token pass auth credentials credential sig signature sessionid sid "
                                 "jsessionid phpsessid session cookie clave jeton seed"))
            return 1;
        return 0;
    }

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        struct secret_prefix
        {
            const char* prefix;     // what the value starts with, its case exact
            std::size_t min_rest;   // the least number of characters after it
            std::size_t max_rest;   // the most, 0 for any
            const char* extra;      // the characters the rest may hold beyond letters and digits
            bool capitals;          // capitals and digits alone, a digit not needed

            secret_prefix(const char* prefix_, std::size_t min_rest_, std::size_t max_rest_, const char* extra_,
                          bool capitals_ = false)
                : prefix(prefix_), min_rest(min_rest_), max_rest(max_rest_), extra(extra_), capitals(capitals_) {}
        };

        /* The values that a prefix tells: the tokens of code hosting, cloud, payment, messaging and
           package services, each with the alphabet and the least length its issuer writes. */
        inline const std::vector<secret_prefix>& secret_prefixes()
        {
            static const std::vector<secret_prefix> p = {
                { "ghp_", 30, 0, "" }, { "gho_", 30, 0, "" }, { "ghu_", 30, 0, "" }, { "ghs_", 30, 0, "" },
                { "ghr_", 30, 0, "" }, { "github_pat_", 40, 0, "_" }, { "glpat-", 20, 0, "_-" },
                { "gldt-", 20, 0, "_-" }, { "glrt-", 20, 0, "_-" },
                { "xoxb-", 20, 0, "-" }, { "xoxp-", 20, 0, "-" }, { "xoxa-", 20, 0, "-" }, { "xoxr-", 20, 0, "-" },
                { "xoxs-", 20, 0, "-" }, { "xoxe-", 20, 0, "-" }, { "xapp-", 20, 0, "-" },
                { "sk_live_", 16, 0, "" }, { "sk_test_", 16, 0, "" }, { "rk_live_", 16, 0, "" },
                { "rk_test_", 16, 0, "" }, { "whsec_", 16, 0, "" },
                { "sk-", 32, 0, "_-" },
                { "AKIA", 16, 16, "", true }, { "ASIA", 16, 16, "", true },
                { "AIza", 35, 35, "_-" }, { "ya29.", 40, 0, "_-." },
                { "hf_", 30, 0, "" }, { "npm_", 36, 36, "" }, { "pypi-", 40, 0, "_-" },
                { "dop_v1_", 64, 64, "" }, { "doo_v1_", 64, 64, "" }, { "dor_v1_", 64, 64, "" },
                { "shpat_", 32, 32, "" }, { "shpca_", 32, 32, "" }, { "shppa_", 32, 32, "" },
                { "shpss_", 32, 32, "" },
                { "SG.", 60, 0, "_-." }, { "AGE-SECRET-KEY-1", 50, 0, "" }, { "hvs.", 24, 0, "_-" },
                { "PMAK-", 50, 0, "-" }, { "NRAK-", 27, 27, "" }, { "glsa_", 32, 0, "_" },
                { "lin_api_", 40, 40, "" }, { "figd_", 40, 0, "_-" }, { "ntn_", 40, 0, "" },
                { "tskey-", 20, 0, "-" }, { "sq0atp-", 22, 0, "_-" }, { "sq0csp-", 22, 0, "_-" },
                { "dapi", 32, 32, "" }, { "pat-", 36, 0, "-" }, { "ATATT", 60, 0, "_-=" } };
            return p;
        }

        inline bool secret_token_char(unsigned char c, const char* extra)
        {
            return secret_alnum(c) || (c != 0 && std::strchr(extra, c) != nullptr);
        }

        // The longest value read after a key, and the longest run of characters read as one token.
        const std::size_t secret_max_value = 4096;
        const std::size_t secret_max_token = 1024;

        /* The end of the value that a prefix tells at begin, or 0: its characters to the first that
           its alphabet does not hold, of the length its issuer writes, a digit and a letter among them.
           A run of more than secret_max_token characters is no token: its end is given in overrun, when
           given, so that a reader of the text looks for no token inside it. */
        inline std::size_t secret_prefixed_end(const std::string& text, std::size_t begin,
                                               std::size_t* overrun = nullptr)
        {
            static const std::array<bool, 256> opens = []() {
                std::array<bool, 256> first;
                first.fill(false);
                for (const secret_prefix& p : secret_prefixes()) first[static_cast<unsigned char>(p.prefix[0])] = true;
                return first;
            }();
            if (begin >= text.size() || !opens[static_cast<unsigned char>(text[begin])]) return 0;
            for (const secret_prefix& p : secret_prefixes())
            {
                if (!secret_starts_with(text, begin, p.prefix)) continue;
                const std::size_t from = begin + std::strlen(p.prefix);
                std::size_t e = from;
                while (e < text.size() && e - from <= secret_max_token &&
                       secret_token_char(static_cast<unsigned char>(text[e]), p.extra))
                    ++e;
                if (e - from > secret_max_token)
                {
                    if (overrun != nullptr) *overrun = std::max(*overrun, e);
                    continue;
                }
                // The period, the hyphen or the underscore that ends a sentence or a name is no part of it.
                while (e > from && !secret_alnum(static_cast<unsigned char>(text[e - 1]))) --e;
                const std::size_t rest = e - from;
                if (rest < p.min_rest || (p.max_rest != 0 && rest > p.max_rest)) continue;
                const secret_statistics st = secret_stats(text, from, e);
                if (p.capitals ? st.lower > 0 : (st.digits == 0 || st.lower + st.upper == 0)) continue;
                return e;
            }
            return 0;
        }

        inline bool secret_base64url(unsigned char c) { return secret_alnum(c) || c == '-' || c == '_'; }

        // The end of a JSON Web Token at begin, or 0: three parts joined by periods, the first two objects.
        inline std::size_t secret_jwt_end(const std::string& text, std::size_t begin)
        {
            if (!secret_starts_with(text, begin, "eyJ")) return 0;
            std::size_t e = begin;
            for (int part = 0; part < 3; ++part)
            {
                if (part < 2 && !secret_starts_with(text, e, "eyJ")) return 0;
                const std::size_t from = e;
                while (e < text.size() && secret_base64url(static_cast<unsigned char>(text[e]))) ++e;
                if (e - from < (part < 2 ? 10u : 0u)) return 0;
                if (part == 2) break;
                if (e >= text.size() || text[e] != '.') return 0;
                ++e;
            }
            return e;
        }

        /* The end of the hash of a password at begin, or 0, in the forms that the files of accounts
           write: "$2b$12$...", "$6$salt$...", "$argon2id$v=19$...". */
        inline std::size_t secret_hash_end(const std::string& text, std::size_t begin)
        {
            static const char* const schemes[] = { "$2a$", "$2b$", "$2y$", "$1$", "$5$", "$6$", "$y$", "$argon2id$",
                                                   "$argon2i$", "$argon2d$", "$pbkdf2", "$scrypt$" };
            bool known = false;
            for (const char* s : schemes)
                if (secret_starts_with(text, begin, s)) known = true;
            if (!known) return 0;
            std::size_t e = begin;
            while (e < text.size())
            {
                const unsigned char c = static_cast<unsigned char>(text[e]);
                if (!(secret_alnum(c) || c == '$' || c == '.' || c == '/' || c == '=' || c == ',' || c == '+' ||
                      c == '-'))
                    break;
                ++e;
            }
            while (e > begin && (text[e - 1] == '.' || text[e - 1] == ',')) --e;
            return e - begin >= 28 ? e : 0;
        }

        inline bool secret_boundary_before(const std::string& text, std::size_t i)
        {
            if (i == 0) return true;
            const unsigned char c = static_cast<unsigned char>(text[i - 1]);
            return !(secret_alnum(c) || c == '_' || c == '-' || c >= 0x80);
        }

        inline bool secret_boundary_after(const std::string& text, std::size_t i)
        {
            if (i >= text.size()) return true;
            const unsigned char c = static_cast<unsigned char>(text[i]);
            return !(secret_alnum(c) || c == '_' || c >= 0x80);
        }

        inline void add_secret(std::vector<secret_match>& out, std::size_t b, std::size_t e, secret_evidence evidence,
                               double confidence, bool web = false)
        {
            if (e <= b) return;
            secret_match m;
            m.begin = b;
            m.end = e;
            m.evidence = evidence;
            m.confidence = confidence;
            m.web_credentials = web;
            out.push_back(m);
        }

        // The offset after the line that closes a key, "-----END ... -----", which starts at i.
        inline std::size_t secret_key_block_end(const std::string& text, std::size_t i)
        {
            std::size_t e = i + 9;
            while (e < text.size() && e - i < 80 && !secret_line_break(static_cast<unsigned char>(text[e])))
            {
                if (secret_starts_with(text, e, "-----")) return e + 5;
                ++e;
            }
            return e;
        }

        inline bool secret_base64(unsigned char c) { return secret_alnum(c) || c == '+' || c == '/' || c == '='; }

        /* The lines of a private key between its opening and its closing lines, each as a value: the
           lines that name the key and its cipher stay readable. A line of the alphabet of base 64 is a
           part of the key, a line with a colon names the key or its cipher, and any other line ends a
           key whose closing line is missing. A key written on one line gives its pieces, between the
           escaped line breaks of a JSON or an environment value or between spaces. A part of less than
           16 characters is one when it ends a key whose closing line is read. Returns the offset after
           the block, or begin when no block of a private key opens there. */
        inline std::size_t secret_key_block(const std::string& text, std::size_t begin, std::vector<secret_match>& out)
        {
            if (!secret_starts_with(text, begin, "-----BEGIN ")) return begin;
            const std::size_t name = begin + 11;
            std::size_t head_end = name;
            while (head_end < text.size() && head_end - name < 48 && text[head_end] != '-' &&
                   !secret_line_break(static_cast<unsigned char>(text[head_end])))
                ++head_end;
            if (!secret_starts_with(text, head_end, "-----") ||
                text.substr(name, head_end - name).find("PRIVATE KEY") == std::string::npos)
                return begin;
            typedef std::pair<std::size_t, std::size_t> range;
            // The short parts read since the last part of 16 characters, and whether one was read.
            std::vector<range> pending;
            bool material = false;
            auto take = [&](const range& r) {
                if (r.second - r.first < 16) { pending.push_back(r); return; }
                add_secret(out, r.first, r.second, secret_evidence::format, 0.99);
                material = true;
                pending.clear();
            };
            auto close = [&](std::size_t end_line) {
                if (material)
                    for (const range& r : pending) add_secret(out, r.first, r.second, secret_evidence::format, 0.99);
                return secret_key_block_end(text, end_line);
            };
            std::size_t i = head_end + 5;
            while (i < text.size())
            {
                while (i < text.size() && secret_space(static_cast<unsigned char>(text[i]))) ++i;
                if (i >= text.size()) break;
                if (secret_starts_with(text, i, "-----END ")) return close(i);
                std::size_t e = i;
                while (e < text.size() && !secret_line_break(static_cast<unsigned char>(text[e]))) ++e;
                std::size_t stop = e;
                while (stop > i && secret_space(static_cast<unsigned char>(text[stop - 1]))) --stop;
                bool base64 = true, colon = false;
                for (std::size_t k = i; k < stop; ++k)
                {
                    if (!secret_base64(static_cast<unsigned char>(text[k]))) base64 = false;
                    if (text[k] == ':') colon = true;
                }
                if (base64)
                {
                    take(range(i, stop));
                    i = e;
                    continue;
                }
                // The pieces of the line, to its closing mark when it holds one.
                std::vector<range> pieces;
                bool escaped = false, parts = false;
                std::size_t end_mark = std::string::npos;
                for (std::size_t k = i; k < stop && end_mark == std::string::npos;)
                {
                    const char c = text[k];
                    if (c == ' ' || c == '\t' || c == '"' || c == '\'') { ++k; continue; }
                    if (c == '\\' && k + 1 < stop && (text[k + 1] == 'n' || text[k + 1] == 'r'))
                    {
                        escaped = true;
                        k += 2;
                        continue;
                    }
                    if (secret_starts_with(text, k, "-----END ")) { end_mark = k; break; }
                    std::size_t pe = k;
                    bool part = true;
                    while (pe < stop && text[pe] != ' ' && text[pe] != '\t' && text[pe] != '"' && text[pe] != '\'' &&
                           !(text[pe] == '\\' && pe + 1 < stop && (text[pe + 1] == 'n' || text[pe + 1] == 'r')))
                    {
                        if (!secret_base64(static_cast<unsigned char>(text[pe]))) part = false;
                        ++pe;
                    }
                    if (part)
                    {
                        pieces.push_back(range(k, pe));
                        if (pe - k >= 16) parts = true;
                    }
                    k = pe;
                }
                if (end_mark == std::string::npos && !escaped)
                {
                    // A line that names the key or its cipher holds no part of it; a line of words ends the key.
                    if (colon) { i = e; continue; }
                    if (!parts) return i;
                }
                for (const range& r : pieces) take(r);
                return end_mark != std::string::npos ? close(end_mark) : e;
            }
            return text.size();
        }

        /* The password of an address written with its credentials, "scheme://user:password@host": begin
           is the offset after "://". The password of a web address is marked so, since the whole address
           is a value of its own for the reader of web addresses. */
        inline void secret_address_password(const std::string& text, std::size_t scheme_begin, std::size_t begin,
                                            std::vector<secret_match>& out)
        {
            // The user, the password and the host: no more than 512 bytes are read for them.
            std::size_t e = begin;
            while (e < text.size() && e - begin < 512)
            {
                const unsigned char c = static_cast<unsigned char>(text[e]);
                if (secret_space(c) || c == '/' || c == '?' || c == '#' || c == '"' || c == '\'' || c == '<' ||
                    c == '>' || c == '`')
                    break;
                ++e;
            }
            // The last "@" before the host, then the first colon before it.
            std::size_t at = e;
            while (at > begin && text[at - 1] != '@') --at;
            if (at == begin || at >= e) return;
            --at;
            std::size_t colon = begin;
            while (colon < at && text[colon] != ':') ++colon;
            if (colon + 1 >= at) return;
            const std::string value = text.substr(colon + 1, at - colon - 1);
            if (secret_placeholder(value)) return;
            const std::string scheme = secret_lowercase(text, scheme_begin, begin - 3);
            add_secret(out, colon + 1, at, secret_evidence::format, 0.95, scheme == "http" || scheme == "https");
        }

        // The value that starts at begin and ends before limit, without the quotes around it.
        struct secret_value
        {
            std::size_t begin = 0;
            std::size_t end = 0;
            bool quoted = false;
        };

        /* Reads the value written at begin on its line: between its quotes, or to the first space; in
           the query of an address (query), to the next parameter; in a list of settings joined by
           semicolons, to the next one. The comma, the semicolon, the period and the brackets that
           close what holds the value are left out. No more than secret_max_value bytes are read past
           what a value may hold. */
        inline secret_value read_secret_value(const std::string& text, std::size_t begin, bool query)
        {
            secret_value v;
            std::size_t i = begin;
            while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
            v.begin = v.end = i;
            if (i >= text.size()) return v;
            const unsigned char q = static_cast<unsigned char>(text[i]);
            if (secret_quote(q))
            {
                std::size_t e = i + 1;
                while (e < text.size() && e - i <= secret_max_value && static_cast<unsigned char>(text[e]) != q &&
                       !secret_line_break(static_cast<unsigned char>(text[e])))
                {
                    if (text[e] == '\\' && e + 1 < text.size() && static_cast<unsigned char>(text[e + 1]) == q) ++e;
                    ++e;
                }
                if (e < text.size() && static_cast<unsigned char>(text[e]) == q)
                {
                    v.begin = i + 1;
                    v.end = e;
                    v.quoted = true;
                    return v;
                }
                v.begin = i = i + 1;
            }
            // A remark between parentheses in place of the value is read whole: "(sent apart)", "(voir le coffre)".
            if (text[i] == '(')
            {
                std::size_t close = i + 1;
                bool words = false;
                while (close < text.size() && close - i <= secret_max_value && text[close] != ')' &&
                       !secret_line_break(static_cast<unsigned char>(text[close])))
                {
                    if (text[close] == ' ') words = true;
                    ++close;
                }
                if (words && close < text.size() && text[close] == ')')
                {
                    v.end = close + 1;
                    return v;
                }
            }
            std::size_t e = i;
            while (e < text.size() && e - i <= secret_max_value && !secret_space(static_cast<unsigned char>(text[e])))
            {
                const unsigned char c = static_cast<unsigned char>(text[e]);
                if (query && (c == '&' || c == '#')) break;
                if (c == ';' && (e + 1 >= text.size() || secret_space(static_cast<unsigned char>(text[e + 1])) ||
                                 secret_upper(static_cast<unsigned char>(text[e + 1])) ||
                                 secret_lower(static_cast<unsigned char>(text[e + 1]))))
                    break;
                if (c == '"' || c == '<' || c == '>') break;
                ++e;
            }
            // What closes the sentence or the structure around the value.
            int round = 0, square = 0, curly = 0;
            for (std::size_t k = i; k < e; ++k)
            {
                if (text[k] == '(') ++round;
                if (text[k] == ')') --round;
                if (text[k] == '[') ++square;
                if (text[k] == ']') --square;
                if (text[k] == '{') ++curly;
                if (text[k] == '}') --curly;
            }
            for (bool moved = true; moved && e > i;)
            {
                moved = false;
                const char c = text[e - 1];
                if (c == ',' || c == ';' || c == '\'' || c == '`' ||
                    (c == '.' && e - i >= 2 && text[e - 2] != '.') ||
                    (c == ')' && round < 0) || (c == ']' && square < 0) || (c == '}' && curly < 0))
                {
                    if (c == ')') ++round;
                    if (c == ']') ++square;
                    if (c == '}') ++curly;
                    --e;
                    moved = true;
                }
            }
            v.end = e;
            return v;
        }

        /* Whether a value read after a key of the given strength (secret_key_strength()) is its secret.
           A word of letters alone, written without quotes before another word of its line, belongs to a
           sentence or to a list: "passwd: files systemd", "password: see below". */
        inline bool secret_value_accepted(const std::string& text, const secret_value& v, int strength)
        {
            const std::size_t n = v.end - v.begin;
            if (n < 3 || n > secret_max_value) return false;
            const std::string value = text.substr(v.begin, n);
            if (secret_placeholder(value)) return false;
            const secret_statistics st = secret_stats(text, v.begin, v.end);
            if (!v.quoted && st.digits == 0 && st.symbols == 0)
            {
                std::size_t k = v.end;
                while (k < text.size() && (text[k] == ' ' || text[k] == '\t')) ++k;
                if (k > v.end && k < text.size() &&
                    (secret_lower(static_cast<unsigned char>(text[k])) ||
                     secret_upper(static_cast<unsigned char>(text[k])) || static_cast<unsigned char>(text[k]) >= 0x80))
                    return false;
            }
            if (strength == 3)
                return (st.digits == n && n <= 10) || (secret_like(st) && n <= 64);
            if (strength == 2)
                return n >= 4;
            return n >= 8 && secret_like(st) && value.find(' ') == std::string::npos;
        }

        // Whether the name between b and e may name a secret: a cheap test before secret_key_strength().
        inline bool secret_key_candidate(const std::string& text, std::size_t b, std::size_t e)
        {
            static const char* const needles[] = { "pass", "pwd", "secret", "token", "key", "mdp", "kennwort",
                "contrasena", "pin", "puk", "otp", "cvv", "cvc", "cvn", "code", "auth", "identified", "psk", "sig",
                "sid", "session", "cookie", "clave", "jeton", "seed", "credential" };
            for (const char* needle : needles)
            {
                const std::size_t len = std::strlen(needle);
                for (std::size_t i = b; i + len <= e; ++i)
                {
                    std::size_t k = 0;
                    while (k < len)
                    {
                        const unsigned char c = static_cast<unsigned char>(text[i + k]);
                        if ((secret_upper(c) ? c + 32 : c) != static_cast<unsigned char>(needle[k])) break;
                        ++k;
                    }
                    if (k == len) return true;
                }
            }
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    inline std::vector<secret_match> find_secret_formats(
        const std::string& text
    )
    {
        using namespace privacy_impl;
        std::vector<secret_match> out;
        const std::size_t n = text.size();
        // The end of a run of characters too long for a token: no token is looked for inside it.
        std::size_t overrun = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == '-' && secret_starts_with(text, i, "-----BEGIN "))
            {
                const std::size_t after = secret_key_block(text, i, out);
                if (after > i) { i = after - 1; continue; }
            }
            if (c == ':' && secret_starts_with(text, i, "://") && i > 0)
            {
                std::size_t b = i;
                while (b > 0 && (secret_alnum(static_cast<unsigned char>(text[b - 1])) || text[b - 1] == '+' ||
                                 text[b - 1] == '.' || text[b - 1] == '-'))
                    --b;
                if (b < i) secret_address_password(text, b, i + 3, out);
                continue;
            }
            if (c == '$' && secret_boundary_before(text, i))
            {
                const std::size_t e = secret_hash_end(text, i);
                if (e > i)
                {
                    add_secret(out, i, e, secret_evidence::format, 0.95);
                    i = e - 1;
                }
                continue;
            }
            if (!secret_alnum(c) || !secret_boundary_before(text, i)) continue;
            std::size_t e = secret_jwt_end(text, i);
            if (e == 0 && i >= overrun) e = secret_prefixed_end(text, i, &overrun);
            if (e > i && secret_boundary_after(text, e))
            {
                add_secret(out, i, e, secret_evidence::format, 0.98);
                // The secret key written on the line of the identifier of an access key.
                if (secret_starts_with(text, i, "AKIA") || secret_starts_with(text, i, "ASIA"))
                {
                    std::size_t k = e;
                    while (k < n && k - e < 256 && !secret_line_break(static_cast<unsigned char>(text[k])))
                    {
                        std::size_t r = k;
                        while (r < n && (secret_alnum(static_cast<unsigned char>(text[r])) || text[r] == '+' ||
                                         text[r] == '/'))
                            ++r;
                        if (r - k == 40 && (k == 0 || !secret_alnum(static_cast<unsigned char>(text[k - 1]))))
                        {
                            const secret_statistics st = secret_stats(text, k, r);
                            if (st.digits > 0 && st.lower > 0 && st.upper > 0)
                                add_secret(out, k, r, secret_evidence::format, 0.9);
                        }
                        k = r > k ? r : k + 1;
                    }
                }
                i = e - 1;
                continue;
            }
            // "Bearer" and the token it presents.
            if ((c == 'B' || c == 'b') && secret_lowercase(text, i, std::min(n, i + 7)) == "bearer ")
            {
                std::size_t b = i + 7;
                while (b < n && text[b] == ' ') ++b;
                std::size_t r = b;
                while (r < n)
                {
                    const unsigned char x = static_cast<unsigned char>(text[r]);
                    if (!(secret_alnum(x) || x == '.' || x == '_' || x == '~' || x == '+' || x == '/' || x == '=' ||
                          x == '-'))
                        break;
                    ++r;
                }
                while (r > b && text[r - 1] == '.') --r;
                const secret_statistics st = secret_stats(text, b, r);
                if (r - b >= 20 && st.digits > 0 && st.lower + st.upper > 0 &&
                    !secret_placeholder(text.substr(b, r - b)))
                {
                    add_secret(out, b, r, secret_evidence::format, 0.95);
                    i = r - 1;
                }
            }
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline std::vector<secret_match> find_secret_assignments(
        const std::string& text
    )
    {
        using namespace privacy_impl;
        std::vector<secret_match> out;
        const std::size_t n = text.size();
        auto key_char = [](unsigned char c) { return secret_alnum(c) || c == '_' || c == '-' || c == '.'; };
        for (std::size_t i = 0; i < n; ++i)
        {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            // A key, then "=", ":", "=>" or ":=", the key between quotes or not.
            if (c == '=' || c == ':')
            {
                if (c == '=' && i + 1 < n && text[i + 1] == '=') { ++i; continue; }
                if (c == '=' && i > 0 && (text[i - 1] == '!' || text[i - 1] == '<' || text[i - 1] == '>' ||
                                         text[i - 1] == '=' || text[i - 1] == ':'))
                    continue;
                if (c == ':' && i + 1 < n && (text[i + 1] == '/' || text[i + 1] == ':')) continue;
                std::size_t e = i;
                while (e > 0 && (text[e - 1] == ' ' || text[e - 1] == '\t') && i - e < 8) --e;
                if (e > 0 && (text[e - 1] == ']' || text[e - 1] == ')')) --e;
                if (e > 0 && secret_quote(static_cast<unsigned char>(text[e - 1]))) --e;
                std::size_t b = e;
                while (b > 0 && key_char(static_cast<unsigned char>(text[b - 1])) && e - b < 64) --b;
                while (b < e && !secret_alnum(static_cast<unsigned char>(text[b]))) ++b;
                if (b >= e || (b > 0 && key_char(static_cast<unsigned char>(text[b - 1])) && e - b >= 64)) continue;
                if (!secret_key_candidate(text, b, e)) continue;
                const std::string key = text.substr(b, e - b);
                const int strength = secret_key_strength(key);
                if (strength == 0) continue;
                // The parameter of a query: "?token=...&", "&sig=...".
                std::size_t before = b;
                if (before > 0 && secret_quote(static_cast<unsigned char>(text[before - 1]))) --before;
                const bool query = c == '=' && before > 0 && (text[before - 1] == '?' || text[before - 1] == '&');
                std::size_t from = i + 1;
                if (from < n && (text[from] == '>' || text[from] == '=')) ++from;
                secret_value v = read_secret_value(text, from, query);
                // "Authorization: Bearer ...": the scheme is no part of the secret.
                if (strength == 2 && secret_ends_with(secret_lowercase(key, 0, key.size()), "authorization"))
                {
                    std::size_t word_end = v.begin;
                    while (word_end < v.end && text[word_end] != ' ') ++word_end;
                    const std::string scheme = secret_lowercase(text, v.begin, word_end);
                    if (secret_word_in(scheme, "bearer basic token digest apikey negotiate ntlm"))
                    {
                        if (v.quoted)
                        {
                            while (word_end < v.end && text[word_end] == ' ') ++word_end;
                            v.begin = word_end;
                        }
                        else
                        {
                            v = read_secret_value(text, v.end, false);
                        }
                    }
                    else if (v.end - v.begin < 16)
                    {
                        continue;
                    }
                }
                if (!secret_value_accepted(text, v, strength))
                {
                    // A run too long for a value holds no value: the keys written inside it name none.
                    if (v.end - v.begin > secret_max_value) i = v.end - 1;
                    continue;
                }
                add_secret(out, v.begin, v.end, secret_evidence::assignment, strength == 1 ? 0.8 : 0.9);
                i = v.end > i ? v.end - 1 : i;
                continue;
            }
            // The content of an element named as a secret: "<password>VALUE</password>".
            if (c == '>' && i > 1)
            {
                std::size_t b = i;
                while (b > 0 && key_char(static_cast<unsigned char>(text[b - 1])) && i - b < 64) --b;
                if (b == i || b == 0 || text[b - 1] != '<' || !secret_key_candidate(text, b, i)) continue;
                const int strength = secret_key_strength(text.substr(b, i - b));
                if (strength < 2) continue;
                secret_value v;
                v.begin = i + 1;
                while (v.begin < n && (text[v.begin] == ' ' || text[v.begin] == '\t')) ++v.begin;
                v.end = v.begin;
                while (v.end < n && text[v.end] != '<' && !secret_line_break(static_cast<unsigned char>(text[v.end])))
                    ++v.end;
                while (v.end > v.begin && secret_space(static_cast<unsigned char>(text[v.end - 1]))) --v.end;
                v.quoted = true;
                if (v.end >= n || text[v.end] != '<' || !secret_value_accepted(text, v, strength)) continue;
                add_secret(out, v.begin, v.end, secret_evidence::assignment, 0.9);
                i = v.end - 1;
                continue;
            }
            /* An option of a command, "--password VALUE", "-Password 'VALUE'", and a key before a value
               between quotes, "PASSWORD 'VALUE'", "IDENTIFIED BY 'VALUE'". */
            if ((c == '-' && (i == 0 || secret_space(static_cast<unsigned char>(text[i - 1])))) ||
                (secret_alnum(c) && secret_boundary_before(text, i) && (i == 0 || text[i - 1] != '-')))
            {
                std::size_t b = i;
                while (b < n && text[b] == '-') ++b;
                const std::size_t dashes = b - i;
                std::size_t e = b;
                while (e < n && key_char(static_cast<unsigned char>(text[e])) && e - b < 64) ++e;
                if (e == b || e >= n || (text[e] != ' ' && text[e] != '\t')) { i = e > i ? e - 1 : i; continue; }
                if (dashes == 0)
                {
                    // Without dashes, a key announces the value between quotes that follows it, or "IDENTIFIED BY".
                    std::size_t k = e;
                    while (k < n && (text[k] == ' ' || text[k] == '\t')) ++k;
                    const bool quote = k < n && secret_quote(static_cast<unsigned char>(text[k]));
                    if (!quote && !(e - b == 10 && secret_lowercase(text, b, e) == "identified"))
                    {
                        i = e - 1;
                        continue;
                    }
                }
                const std::string key = text.substr(b, e - b);
                const std::string low = secret_lowercase(key, 0, key.size());
                // "curl -u user:password", "curl --user user:password".
                if (dashes > 0 && (low == "u" || low == "user"))
                {
                    std::size_t line = i;
                    while (line > 0 && !secret_line_break(static_cast<unsigned char>(text[line - 1])) && i - line < 512)
                        --line;
                    const secret_value v = read_secret_value(text, e, false);
                    std::size_t colon = v.begin;
                    while (colon < v.end && text[colon] != ':') ++colon;
                    if (colon + 4 <= v.end && text.substr(line, i - line).find("curl") != std::string::npos &&
                        !secret_placeholder(text.substr(colon + 1, v.end - colon - 1)))
                    {
                        add_secret(out, colon + 1, v.end, secret_evidence::assignment, 0.9);
                        i = v.end - 1;
                        continue;
                    }
                }
                if (!secret_key_candidate(text, b, e)) { i = e - 1; continue; }
                int strength = secret_key_strength(key);
                std::size_t from = e;
                if (dashes == 0 && low == "identified")
                {
                    // "IDENTIFIED BY 'value'", "IDENTIFIED WITH plugin BY 'value'".
                    const std::string ahead = secret_lowercase(text, e, std::min(n, e + 52));
                    const std::size_t by = ahead.find(" by ");
                    if (by == std::string::npos || ahead.find_first_of("\r\n") < by)
                    {
                        i = e - 1;
                        continue;
                    }
                    from = e + by + 4;
                    strength = 2;
                }
                if (strength < (dashes > 0 ? 1 : 2)) { i = e - 1; continue; }
                const secret_value v = read_secret_value(text, from, false);
                // Without dashes, the words of a sentence follow a key as often as its value: quotes decide.
                if (dashes == 0 && !v.quoted) { i = e - 1; continue; }
                if (dashes > 0 && !v.quoted && v.begin < n && text[v.begin] == '-') { i = e - 1; continue; }
                if (!secret_value_accepted(text, v, strength))
                {
                    i = v.end - v.begin > secret_max_value ? v.end - 1 : e - 1;
                    continue;
                }
                add_secret(out, v.begin, v.end, secret_evidence::assignment, 0.9);
                i = v.end > i ? v.end - 1 : i;
            }
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline std::vector<secret_match> find_secrets(
        const std::string& text
    )
    {
        std::vector<secret_match> out = find_secret_formats(text);
        const std::vector<secret_match> assigned = find_secret_assignments(text);
        out.insert(out.end(), assigned.begin(), assigned.end());
        // By position, the surest first at one position; a match inside another kept leaves.
        std::sort(out.begin(), out.end(), [](const secret_match& a, const secret_match& b) {
            if (a.begin != b.begin) return a.begin < b.begin;
            if (a.confidence != b.confidence) return a.confidence > b.confidence;
            return a.end > b.end;
        });
        std::vector<secret_match> kept;
        for (const secret_match& m : out)
        {
            if (!kept.empty() && m.begin < kept.back().end)
            {
                /* A format read inside the value of an assignment keeps the bounds of the format when it
                   covers the value, and the value otherwise: "token=ghp_..." and "password=a$2b$x". */
                if (m.end > kept.back().end && m.evidence == secret_evidence::format &&
                    kept.back().evidence == secret_evidence::assignment && m.begin == kept.back().begin)
                    kept.back() = m;
                continue;
            }
            kept.push_back(m);
        }
        return kept;
    }

// ----------------------------------------------------------------------------------------

    inline void secret_chunk(
        const std::string& text,
        std::size_t& begin,
        std::size_t& end
    )
    {
        using namespace privacy_impl;
        /* A secret is written whole between spaces, quotes or the signs of an assignment: what a
           reader found of it grows to these limits, then leaves out the punctuation of its sentence. */
        auto opens = [](unsigned char c) {
            return secret_space(c) || secret_quote(c) || c == '=' || c == ':' || c == '(' || c == '[' || c == '{' ||
                   c == '<' || c == '>' || c == ',' || c == ';';
        };
        auto closes = [](unsigned char c) {
            return secret_space(c) || secret_quote(c) || c == ')' || c == ']' || c == '}' || c == '<' || c == '>' ||
                   c == ',' || c == ';';
        };
        end = std::min(end, text.size());
        if (begin >= end) return;
        const std::size_t first = begin > secret_max_token ? begin - secret_max_token : 0;
        const std::size_t last = std::min(text.size(), end + secret_max_token);
        while (begin > first && !opens(static_cast<unsigned char>(text[begin - 1]))) --begin;
        while (end < last && !closes(static_cast<unsigned char>(text[end]))) ++end;
        while (end > begin + 1 && (text[end - 1] == '.' || text[end - 1] == ':')) --end;
        // Never inside a character of several bytes.
        while (begin > 0 && (static_cast<unsigned char>(text[begin]) & 0xC0) == 0x80) --begin;
        while (end < text.size() && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) ++end;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_SECRET_FORMATS_H_
