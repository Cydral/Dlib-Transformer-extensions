// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PSEUDONYMIZER_H_
#define DLIB_PSEUDONYMIZER_H_

#include "pseudonymizer_abstract.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../crypto.h"
#include "pii_types.h"
#include "pii_lexicons.h"
#include "date_formats.h"
#include "unicode_text.h"
#include "id_checks.h"

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pseudonymization_key
    {
    public:
        pseudonymization_key() {}

        pseudonymization_key(const pseudonymization_key&) = default;
        pseudonymization_key& operator=(const pseudonymization_key&) = default;

        ~pseudonymization_key()
        {
            if (!key_.empty())
                secure_wipe(&key_[0], key_.size());
        }

        static pseudonymization_key generate()
        {
            pseudonymization_key k;
            k.key_ = generate_secure_random_bytes(32);
            return k;
        }

        static pseudonymization_key from_bytes(const std::string& bytes)
        {
            if (bytes.size() != 32)
                throw crypto_error("pseudonymization_key: a key is exactly 32 bytes");
            pseudonymization_key k;
            k.key_ = bytes;
            return k;
        }

        static pseudonymization_key from_passphrase(
            const std::string& passphrase,
            unsigned long iterations = 600000,
            const std::string& salt = "dlib.privacy.passphrase.v1"
        )
        {
            pseudonymization_key k;
            k.key_ = pbkdf2_hmac_sha256(passphrase, salt, iterations, 32);
            return k;
        }

        void save(const std::string& filename) const
        {
            if (!is_set())
                throw crypto_error("pseudonymization_key: no key to save");
            std::string content =
                "# dlib pseudonymization key: keep this file secret, it restores every document protected with it\n"
                "v1:" + to_hex(key_) + "\n";
            bool ok = false;
#if !defined(_WIN32)
            /* O_EXCL: an existing key is never overwritten, losing it would make every
               document protected with it impossible to restore. */
            const int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
            if (fd < 0)
            {
                secure_wipe(&content[0], content.size());
                throw crypto_error("pseudonymization_key: cannot create " + filename + " (it must not exist yet)");
            }
            std::size_t done = 0;
            while (done < content.size())
            {
                const ssize_t n = ::write(fd, content.data() + done, content.size() - done);
                if (n <= 0) break;
                done += static_cast<std::size_t>(n);
            }
            ok = done == content.size() && ::fsync(fd) == 0;
            ok = ::close(fd) == 0 && ok;
#else
            if (std::ifstream(filename))
            {
                secure_wipe(&content[0], content.size());
                throw crypto_error("pseudonymization_key: cannot create " + filename + " (it must not exist yet)");
            }
            {
                std::ofstream out(filename, std::ios::binary | std::ios::trunc);
                out << content;
                out.flush();
                ok = static_cast<bool>(out);
            }
#endif
            secure_wipe(&content[0], content.size());
            if (!ok)
            {
                // A partial file would block the next attempt.
                std::remove(filename.c_str());
                throw crypto_error("pseudonymization_key: cannot write " + filename);
            }
        }

        static pseudonymization_key load(const std::string& filename)
        {
            std::ifstream in(filename);
            if (!in)
                throw crypto_error("pseudonymization_key: cannot open " + filename);
            std::string line;
            while (std::getline(in, line))
            {
                if (line.compare(0, 3, "v1:") == 0)
                {
                    std::string hex = line.substr(3);
                    secure_wipe(&line[0], line.size());
                    while (!hex.empty() && (hex.back() == '\r' || hex.back() == ' '))
                        hex.pop_back();
                    std::string bytes = from_hex(hex);
                    if (!hex.empty()) secure_wipe(&hex[0], hex.size());
                    pseudonymization_key k;
                    try { k = from_bytes(bytes); }
                    catch (...) { if (!bytes.empty()) secure_wipe(&bytes[0], bytes.size()); throw; }
                    if (!bytes.empty()) secure_wipe(&bytes[0], bytes.size());
                    return k;
                }
                if (!line.empty()) secure_wipe(&line[0], line.size());
            }
            throw crypto_error("pseudonymization_key: " + filename + " holds no v1 key");
        }

        bool is_set() const { return key_.size() == 32; }
        const std::string& bytes() const { return key_; }

    private:
        std::string key_;
    };

// ----------------------------------------------------------------------------------------

    enum class date_policy
    {
        shift,
        encrypt,
        keep
    };

    struct pseudonymizer_options
    {
        std::string context;
        date_policy dates = date_policy::shift;
        unsigned long max_shift_weeks = 260;
        numeric_date_order date_order = numeric_date_order::day_month_year;
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        inline const char* crockford_alphabet() { return "0123456789ABCDEFGHJKMNPQRSTVWXYZ"; }

        inline int crockford_value(char c)
        {
            if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
            if (c == 'O') return 0;
            if (c == 'I' || c == 'L') return 1;
            const char* p = std::strchr(crockford_alphabet(), c);
            return (p && c != '\0') ? static_cast<int>(p - crockford_alphabet()) : -1;
        }

        inline std::string crockford_encode(std::uint32_t v, int chars)
        {
            std::string s(static_cast<std::size_t>(chars), '0');
            for (int i = chars - 1; i >= 0; --i)
            {
                s[static_cast<std::size_t>(i)] = crockford_alphabet()[v & 31];
                v >>= 5;
            }
            return s;
        }

        inline bool crockford_decode(const std::string& s, std::uint32_t& v)
        {
            v = 0;
            if (s.empty() || s.size() > 6) return false;
            for (char c : s)
            {
                const int d = crockford_value(c);
                if (d < 0) return false;
                v = (v << 5) | static_cast<std::uint32_t>(d);
            }
            return true;
        }

        inline bool is_ascii_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
        inline bool is_ascii_upper(char c) { return c >= 'A' && c <= 'Z'; }
        inline bool is_ascii_lower(char c) { return c >= 'a' && c <= 'z'; }
        inline bool is_ascii_digit_char(char c) { return c >= '0' && c <= '9'; }

        inline bool all_ascii(const std::string& s)
        {
            for (char c : s)
                if (static_cast<unsigned char>(c) >= 0x80) return false;
            return true;
        }

        inline bool all_digits(const std::string& s)
        {
            if (s.empty()) return false;
            for (char c : s)
                if (!is_ascii_digit_char(c)) return false;
            return true;
        }

        /*
            Characters kept verbatim between the encrypted words of a tag body. They carry
            the structure of the value, and none of them can appear inside a code (codes
            are ASCII letters, an optional final '!', or digits), so the body splits back
            into the same words. '!', '#', '[' and ']' are excluded because the tag grammar uses
            them.
        */
        inline bool is_verbatim_separator(char32_t c)
        {
            switch (c)
            {
                case ' ': case 0xA0: case 0x202F: case 0x2009:
                case '-': case 0x2010: case 0x2011:
                case '\'': case 0x2019: case 0x2BC:
                case '.': case ',': case '&': case '(': case ')': case '/': case ':': case ';':
                case '+': case '_': case '@': case '=': case '?': case '~': case '%': case '*':
                    return true;
                default:
                    return false;
            }
        }

        struct word_segment
        {
            std::string text;
            bool separator;
        };

        inline std::vector<word_segment> split_words(const std::string& value)
        {
            std::vector<word_segment> out;
            std::size_t i = 0;
            while (i < value.size())
            {
                const std::size_t start = i;
                const char32_t c = utf8_next(value, i);
                const bool sep = is_verbatim_separator(c);
                if (out.empty() || out.back().separator != sep)
                    out.push_back(word_segment{ std::string(), sep });
                out.back().text.append(value, start, i - start);
            }
            return out;
        }

        /* 0 lower, 1 title, 2 upper, 3 mixed. A class is reported only when applying it to
           the lower case form gives the word back exactly. */
        inline int word_case_class(const std::string& w)
        {
            const std::string lw = lowercase(w);
            if (w == lw) return 0;
            if (w == titlecase(lw)) return 1;
            if (w == uppercase(lw)) return 2;
            return 3;
        }

        inline std::string apply_word_class(const std::string& lw, int cls)
        {
            if (cls == 1) return titlecase(lw);
            if (cls == 2) return uppercase(lw);
            return lw;
        }

        inline bool luhn_valid(const std::string& d) { return luhn_ok(d); }

        inline char luhn_check_digit(const std::string& payload)
        {
            return static_cast<char>('0' + luhn_digit(payload));
        }

        // A string that wipes its content when it goes out of scope, for derived keys.
        struct wiped_string
        {
            std::string s;
            explicit wiped_string(std::string v) : s(std::move(v)) {}
            wiped_string(const wiped_string&) = delete;
            wiped_string& operator=(const wiped_string&) = delete;
            ~wiped_string() { if (!s.empty()) secure_wipe(&s[0], s.size()); }
            const char* data() const { return s.data(); }
        };

        inline int mod97_of_alnum(const std::string& s)
        {
            int r = 0;
            for (char c : s)
            {
                if (is_ascii_digit_char(c))
                {
                    r = (r * 10 + (c - '0')) % 97;
                }
                else
                {
                    const int v = (c >= 'a' ? c - 'a' : c - 'A') + 10;
                    r = (r * 100 + v) % 97;
                }
            }
            return r;
        }

        inline bool iban_checksum_valid(const std::string& a)
        {
            if (a.size() < 5) return false;
            return mod97_of_alnum(a.substr(4) + a.substr(0, 4)) == 1;
        }

        inline std::string iban_check_digits(const std::string& cc, const std::string& bban)
        {
            const int r = mod97_of_alnum(bban + cc + "00");
            const int check = 98 - r;
            std::string s = std::to_string(check);
            return s.size() < 2 ? "0" + s : s;
        }

        inline std::size_t e164_country_code_length(const std::string& d)
        {
            if (d.empty()) return 1;
            if (d[0] == '1' || d[0] == '7') return 1;
            static const char* two[] = { "20", "27", "30", "31", "32", "33", "34", "36", "39", "40", "41",
                "43", "44", "45", "46", "47", "48", "49", "51", "52", "53", "54", "55", "56", "57", "58",
                "60", "61", "62", "63", "64", "65", "66", "81", "82", "84", "86", "90", "91", "92", "93",
                "94", "95", "98" };
            if (d.size() >= 2)
                for (const char* t : two)
                    if (d.compare(0, 2, t) == 0) return 2;
            return 3;
        }

        inline bool parse_ipv4(const std::string& s, std::uint32_t& a)
        {
            a = 0;
            int parts = 0;
            std::size_t i = 0;
            while (parts < 4)
            {
                if (i >= s.size() || !is_ascii_digit_char(s[i])) return false;
                std::size_t j = i;
                int v = 0;
                while (j < s.size() && is_ascii_digit_char(s[j]) && j - i < 3)
                    v = v * 10 + (s[j++] - '0');
                if (j < s.size() && is_ascii_digit_char(s[j])) return false;
                if (v > 255 || (j - i > 1 && s[i] == '0')) return false;
                a = (a << 8) | static_cast<std::uint32_t>(v);
                ++parts;
                i = j;
                if (parts < 4)
                {
                    if (i >= s.size() || s[i] != '.') return false;
                    ++i;
                }
            }
            return i == s.size();
        }

        inline std::string render_ipv4(std::uint32_t a)
        {
            return std::to_string(a >> 24) + "." + std::to_string((a >> 16) & 255) + "." +
                   std::to_string((a >> 8) & 255) + "." + std::to_string(a & 255);
        }

        inline bool parse_ipv6(const std::string& s, std::array<unsigned char,16>& a)
        {
            std::vector<int> head, tail;
            bool compressed = false;
            std::size_t i = 0;
            if (s.compare(0, 2, "::") == 0) { compressed = true; i = 2; }
            while (i < s.size())
            {
                std::size_t j = i;
                int v = 0;
                while (j < s.size() && j - i < 4 && std::isxdigit(static_cast<unsigned char>(s[j])))
                {
                    const char c = s[j];
                    v = v * 16 + (is_ascii_digit_char(c) ? c - '0' : (c | 0x20) - 'a' + 10);
                    ++j;
                }
                if (j == i) return false;
                if (j < s.size() && std::isxdigit(static_cast<unsigned char>(s[j]))) return false;
                (compressed ? tail : head).push_back(v);
                i = j;
                if (i == s.size()) break;
                if (s[i] != ':') return false;
                if (i + 1 < s.size() && s[i + 1] == ':')
                {
                    if (compressed) return false;
                    compressed = true;
                    i += 2;
                    if (i == s.size()) break;
                }
                else
                {
                    ++i;
                    if (i == s.size()) return false;
                }
            }
            const std::size_t total = head.size() + tail.size();
            if ((!compressed && total != 8) || (compressed && total > 7)) return false;
            std::vector<int> groups(head);
            groups.insert(groups.end(), 8 - total, 0);
            groups.insert(groups.end(), tail.begin(), tail.end());
            for (int k = 0; k < 8; ++k)
            {
                a[2 * k] = static_cast<unsigned char>(groups[static_cast<std::size_t>(k)] >> 8);
                a[2 * k + 1] = static_cast<unsigned char>(groups[static_cast<std::size_t>(k)] & 255);
            }
            return true;
        }

        inline std::string render_ipv6(const std::array<unsigned char,16>& a)
        {
            /* RFC 5952: lower case, no leading zeros, the longest run of two or more zero
               groups (the first one on a tie) replaced by "::". */
            int g[8];
            for (int k = 0; k < 8; ++k)
                g[k] = (a[2 * k] << 8) | a[2 * k + 1];
            int best_start = -1, best_len = 0;
            for (int k = 0; k < 8;)
            {
                if (g[k] != 0) { ++k; continue; }
                int e = k;
                while (e < 8 && g[e] == 0) ++e;
                if (e - k > best_len && e - k >= 2) { best_start = k; best_len = e - k; }
                k = e;
            }
            std::string out;
            char buf[8];
            for (int k = 0; k < 8; ++k)
            {
                if (k == best_start)
                {
                    out += "::";
                    k += best_len - 1;
                    continue;
                }
                if (!out.empty() && out.back() != ':')
                    out += ":";
                std::snprintf(buf, sizeof(buf), "%x", g[k]);
                out += buf;
            }
            return out;
        }

        inline std::string percent_decode(const std::string& s)
        {
            std::string out;
            for (std::size_t i = 0; i < s.size(); ++i)
            {
                if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(static_cast<unsigned char>(s[i + 1])) &&
                    std::isxdigit(static_cast<unsigned char>(s[i + 2])))
                {
                    out.push_back(static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16)));
                    i += 2;
                }
                else
                {
                    out.push_back(s[i]);
                }
            }
            return out;
        }
    }

    inline std::string percent_encode_tag(const std::string& tag)
    {
        static const char hex[] = "0123456789ABCDEF";
        std::string out;
        for (unsigned char c : tag)
        {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                c == '-' || c == '.' || c == '_' || c == '~')
            {
                out.push_back(static_cast<char>(c));
            }
            else
            {
                out.push_back('%');
                out.push_back(hex[c >> 4]);
                out.push_back(hex[c & 15]);
            }
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    class pseudonymizer
    {
    public:
        pseudonymizer(
            const pseudonymization_key& key,
            const pseudonymizer_options& options = pseudonymizer_options()
        ) :
            options_(options),
            word_ff1_(privacy_impl::wiped_string(derive(key, "ff1.word")).data(), 32, 26),
            digit_ff1_(privacy_impl::wiped_string(derive(key, "ff1.digits")).data(), 32, 10),
            hex_ff1_(privacy_impl::wiped_string(derive(key, "ff1.alnum")).data(), 32, 16),
            alnum36_ff1_(privacy_impl::wiped_string(derive(key, "ff1.alnum")).data(), 32, 36),
            alnum62_ff1_(privacy_impl::wiped_string(derive(key, "ff1.alnum")).data(), 32, 62),
            ip_(privacy_impl::wiped_string(derive(key, "ip")).data(), 32),
            check_key_(derive(key, "check"))
        {
            const privacy_impl::wiped_string date_key(derive(key, "date"));
            std::array<unsigned char,32> h = hmac_sha256_digest(date_key.s, "date-shift");
            std::uint64_t v = 0;
            for (int i = 0; i < 8; ++i)
                v = (v << 8) | h[static_cast<std::size_t>(i)];
            secure_wipe(h.data(), h.size());
            // A shift of 1 to max_shift_weeks weeks, either way; the bound is clamped to [1, 5200].
            const long weeks = static_cast<long>(
                std::min<unsigned long>(5200, std::max<unsigned long>(1, options_.max_shift_weeks)));
            long k = static_cast<long>(v % static_cast<std::uint64_t>(2 * weeks)) - weeks;
            if (k >= 0) ++k;
            shift_days_ = 7 * k;

            letter_lengths_.assign(1, 0);
            for (std::size_t n = 1; n <= 512; ++n)
                letter_lengths_.push_back(numerals_needed(n, 26));
        }

        pseudonymizer(const pseudonymizer&) = delete;
        pseudonymizer& operator=(const pseudonymizer&) = delete;

        ~pseudonymizer()
        {
            if (!check_key_.empty()) secure_wipe(&check_key_[0], check_key_.size());
            clear_cache();
        }

        // Forgets the tags computed so far, erasing the values they were computed from.
        void clear_cache() const
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            for (auto& kv : cache_)
            {
                std::string& k = const_cast<std::string&>(kv.first);
                if (!k.empty()) secure_wipe(&k[0], k.size());
            }
            cache_.clear();
        }

        const pseudonymizer_options& get_options() const { return options_; }

        struct restore_report
        {
            std::size_t restored = 0;
            std::size_t rejected = 0;
        };

        long date_shift_days() const { return shift_days_; }

        // True for the values the options leave in clear: dates under date_policy::keep.
        bool keeps_in_clear(pii_type type) const
        {
            return type == pii_type::date && options_.dates == date_policy::keep;
        }

        double shift_date_serial(double serial, bool forward) const
        {
            return serial + (forward ? shift_days_ : -shift_days_);
        }

        std::string key_fingerprint() const
        {
            const std::array<unsigned char,32> h = hmac_sha256_digest(check_key_, "dlib.privacy.fingerprint");
            return to_hex(h.data(), 6);
        }

        std::string tag(pii_type type, const std::string& value) const
        {
            if (attribute_type(type))
            {
                const std::string t = attribute_tag(type, value);
                if (decodes_to(t, value)) return t;
            }
            const std::string cache_key = std::string(pii_type_keyword(type)) + '\x1F' + value;
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                const auto it = cache_.find(cache_key);
                if (it != cache_.end())
                    return it->second;
            }

            std::string body;
            bool ok = false;
            switch (type)
            {
                case pii_type::person:
                case pii_type::organization:
                case pii_type::location:
                case pii_type::address:
                case pii_type::email:
                case pii_type::url:
                    body = encode_words(type, value);
                    ok = true;
                    break;
                case pii_type::phone:
                case pii_type::iban:
                case pii_type::payment_card:
                case pii_type::nir:
                case pii_type::siret:
                    ok = spaced_code(type, value, body, true);
                    break;
                case pii_type::mac_address: ok = code_mac(value, body, true); break;
                case pii_type::license_plate: ok = code_plate(value, body, true); break;
                case pii_type::identifier:
                {
                    std::string digits;
                    for (char c : value) if (c >= '0' && c <= '9') digits.push_back(c);
                    const bool luhn = digits.size() >= 8 && value.find_first_of(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") == std::string::npos &&
                        privacy_impl::luhn_valid(digits);
                    ok = code_identifier(value, body, true, true, luhn);
                    break;
                }
                case pii_type::geolocation: ok = code_geo(value, body, true); break;
                case pii_type::ipv4: ok = code_ipv4(value, body, true); break;
                case pii_type::ipv6: ok = code_ipv6(value, body, true); break;
                case pii_type::date:
                    if (options_.dates == date_policy::shift)
                        ok = date_forward(value, body);
                    break;
                case pii_type::title:
                case pii_type::gender:
                case pii_type::nationality:
                case pii_type::none: break;
            }
            if (!ok)
                body = encode_opaque(type, value);

            std::string result = std::string("[[") + pii_type_keyword(type) + ":" + body + "#" +
                                 check_code(type, value) + "]]";
            /* Every tag is decoded once before it is used: a value its codec cannot give
               back (a body holding "]]", a non canonical check digit) takes the opaque code. */
            if (!decodes_to(result, value))
            {
                result = std::string("[[") + pii_type_keyword(type) + ":" + encode_opaque(type, value) + "#" +
                         check_code(type, value) + "]]";
                if (!decodes_to(result, value))
                    throw crypto_error("pseudonymizer::tag(): the value cannot be encoded");
            }
            std::lock_guard<std::mutex> lock(cache_mutex_);
            if (cache_.size() >= max_cache_entries) 
            {
                for (auto& kv : cache_)
                {
                    std::string& k = const_cast<std::string&>(kv.first);
                    if (!k.empty()) secure_wipe(&k[0], k.size());
                }
                cache_.clear();
            }
            cache_[cache_key] = result;
            return result;
        }

        std::string apply(const std::string& text, const std::vector<pii_entity>& entities) const
        {
            std::vector<pii_entity> sorted(entities);
            std::sort(sorted.begin(), sorted.end(), [](const pii_entity& a, const pii_entity& b)
                      { return a.begin < b.begin || (a.begin == b.begin && a.end > b.end); });
            std::string out;
            std::size_t pos = 0;
            for (const pii_entity& e : sorted)
            {
                if (e.begin < pos || e.end > text.size() || e.end <= e.begin || e.type == pii_type::none ||
                    keeps_in_clear(e.type))
                    continue;
                out.append(text, pos, e.begin - pos);
                out += tag(e.type, text.substr(e.begin, e.end - e.begin));
                pos = e.end;
            }
            out.append(text, pos, std::string::npos);
            return out;
        }

        std::string restore(const std::string& text, restore_report* report = nullptr) const
        {
            return restore_percent(restore_raw(text, report), report);
        }

        struct tag_occurrence
        {
            std::size_t begin = 0;
            std::size_t end = 0;
            pii_type type = pii_type::none;
            std::string original;
        };

        std::vector<tag_occurrence> find_tags(const std::string& text, restore_report* report = nullptr) const
        {
            std::vector<tag_occurrence> found;
            scan_tags(text, "[[", "]]", false, report, [&](std::size_t b, std::size_t e, pii_type type,
                                                             const std::string& original) {
                tag_occurrence o;
                o.begin = b;
                o.end = e;
                o.type = type;
                o.original = original;
                found.push_back(o);
            });
            return found;
        }

        bool decode(const std::string& tag_text, pii_type& type, std::string& original) const
        {
            if (tag_text.size() < 9 || tag_text.compare(0, 2, "[[") != 0 ||
                tag_text.compare(tag_text.size() - 2, 2, "]]") != 0)
                return false;
            return decode_inner(tag_text.substr(2, tag_text.size() - 4), type, original);
        }

    private:
        // The context is folded into every derivation through the HKDF salt.
        std::string derive(const pseudonymization_key& key, const char* purpose) const
        {
            if (!key.is_set())
                throw crypto_error("pseudonymizer: the key is not set");
            return hkdf_sha256(key.bytes(), "dlib.privacy.v1|" + options_.context, purpose, 32);
        }

        std::string check_code(pii_type type, const std::string& original, const std::string& nonce = "") const
        {
            const std::string message = nonce.empty() ? std::string(pii_type_keyword(type)) + ":" + original :
                                        std::string(pii_type_keyword(type)) + ":" + nonce + ":" + original;
            const std::array<unsigned char,32> h = hmac_sha256_digest(check_key_, message);
            const std::uint32_t v = (static_cast<std::uint32_t>(h[0]) << 7) | (h[1] >> 1);
            return privacy_impl::crockford_encode(v, 3);
        }

    // ------------------------------------------------------------------------------------
    /* Civilities, sexes and nationalities take few values: a deterministic code would give
       them away through its frequency or its length. Their code starts with a random nonce
       of four Crockford characters, mixed into the encryption tweak and into the check,
       and the value is padded with zero bytes to a multiple of 16 bytes before encryption,
       so that "M." and "Madame" give codes of the same length and two occurrences of the
       same value give two different tags. */

        static bool attribute_type(pii_type type)
        {
            return type == pii_type::title || type == pii_type::gender || type == pii_type::nationality;
        }

        static std::string attribute_tweak(pii_type type, const std::string& nonce)
        {
            return std::string("attribute.") + pii_type_keyword(type) + "." + nonce;
        }

        std::string attribute_tag(pii_type type, const std::string& value) const
        {
            const std::string random = generate_secure_random_bytes(3);
            const std::uint32_t v = (static_cast<std::uint32_t>(static_cast<unsigned char>(random[0])) << 16) |
                                    (static_cast<std::uint32_t>(static_cast<unsigned char>(random[1])) << 8) |
                                    static_cast<std::uint32_t>(static_cast<unsigned char>(random[2]));
            const std::string nonce = privacy_impl::crockford_encode(v & 0xFFFFF, 4);
            std::string plain = value;
            plain.resize(std::max<std::size_t>(16, (plain.size() + 15) / 16 * 16), '\0');
            return std::string("[[") + pii_type_keyword(type) + ":" + nonce +
                   encode_letters(plain, attribute_tweak(type, nonce)) + "#" + check_code(type, value, nonce) + "]]";
        }

        bool decode_attribute(pii_type type, const std::string& body, const std::string& check,
                              std::string& original) const
        {
            if (body.size() <= 4) return false;
            std::string nonce = body.substr(0, 4);
            for (char& c : nonce)
                if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
            std::uint32_t v = 0;
            if (!privacy_impl::crockford_decode(nonce, v)) return false;
            nonce = privacy_impl::crockford_encode(v, 4);
            std::string value;
            if (!decode_letters(body.substr(4), attribute_tweak(type, nonce), value)) return false;
            const std::string expected = check_code(type, value, nonce);
            if (expected.size() != check.size() || !constant_time_equal(expected.data(), check.data(), check.size()))
                return false;
            original = value;
            return true;
        }

    // ------------------------------------------------------------------------------------
    /* Letter codes: FF1 over 26 letters of the big-endian integer formed by the plaintext
       bytes, padded with zero bytes to at least 3 so that 26^m stays above the minimum
       FF1 domain. The code length m identifies the byte length, which is why it is
       recovered without any length field. */

        std::string encode_letters(std::string plain, const std::string& tweak) const
        {
            while (plain.size() < 3)
                plain.push_back('\0');
            const std::size_t m = letters_for_bytes(plain.size());
            const std::vector<std::uint32_t> x = bytes_to_numerals(plain, 26, m);
            const std::vector<std::uint32_t> y = word_ff1_.encrypt(x, tweak);
            std::string s;
            s.reserve(y.size());
            for (std::uint32_t d : y)
                s.push_back(static_cast<char>('a' + d));
            return s;
        }

        bool decode_letters(const std::string& lower_letters, const std::string& tweak, std::string& plain) const
        {
            const std::size_t m = lower_letters.size();
            std::size_t n = 0;
            if (m <= letter_lengths_.back())
            {
                for (std::size_t k = 3; k < letter_lengths_.size(); ++k)
                {
                    if (letter_lengths_[k] == m) { n = k; break; }
                    if (letter_lengths_[k] > m) return false;
                }
            }
            else
            {
                // Beyond the table: about m * log(26) / log(256) bytes, checked exactly.
                const std::size_t guess = m * 4700 / 8000;
                for (std::size_t k = guess > 2 ? guess - 2 : 1; k <= guess + 2 && n == 0; ++k)
                    if (numerals_needed(k, 26) == m) n = k;
            }
            if (n == 0) return false;
            std::vector<std::uint32_t> y;
            y.reserve(m);
            for (char c : lower_letters)
            {
                if (c < 'a' || c > 'z') return false;
                y.push_back(static_cast<std::uint32_t>(c - 'a'));
            }
            const std::vector<std::uint32_t> x = word_ff1_.decrypt(y, tweak);
            if (!numerals_to_bytes(x, 26, n, plain)) return false;
            while (!plain.empty() && plain.back() == '\0')
                plain.pop_back();
            if (plain.empty() || plain.find('\0') != std::string::npos) return false;
            return privacy_impl::utf8_valid(plain);
        }

        std::size_t letters_for_bytes(std::size_t n) const
        {
            return n < letter_lengths_.size() ? letter_lengths_[n] : numerals_needed(n, 26);
        }

        std::string digits_code(const std::string& digits, const std::string& tweak, bool forward) const
        {
            std::vector<std::uint32_t> x;
            for (char c : digits) x.push_back(static_cast<std::uint32_t>(c - '0'));
            const std::vector<std::uint32_t> y = forward ? digit_ff1_.encrypt(x, tweak) : digit_ff1_.decrypt(x, tweak);
            std::string s;
            for (std::uint32_t d : y) s.push_back(static_cast<char>('0' + d));
            return s;
        }

    // ------------------------------------------------------------------------------------
    // Word codes for names, organisations, places, addresses, e-mails and URLs.

        const std::unordered_set<std::string>& kept_words(pii_type t) const
        {
            static const std::unordered_set<std::string> none;
            switch (t)
            {
                case pii_type::person: return privacy_impl::person_particles();
                case pii_type::organization: return privacy_impl::organization_kept_words();
                case pii_type::location: return privacy_impl::location_kept_words();
                case pii_type::address: return privacy_impl::address_kept_words();
                case pii_type::url: return privacy_impl::url_kept_words();
                default: return none;
            }
        }

        // An e-mail keeps its top level domain: the last word after the last '.'.
        static std::size_t verbatim_tld_index(pii_type t, const std::vector<privacy_impl::word_segment>& segs)
        {
            if (t != pii_type::email || segs.size() < 3) return std::string::npos;
            const std::size_t last = segs.size() - 1;
            if (segs[last].separator) return std::string::npos;
            const std::string& before = segs[last - 1].text;
            bool has_at = false;
            for (std::size_t k = 0; k + 1 < segs.size(); ++k)
                if (segs[k].separator && segs[k].text.find('@') != std::string::npos) has_at = true;
            if (!has_at || before.empty() || before.back() != '.') return std::string::npos;
            return last;
        }

        std::string encode_word(const std::string& w) const
        {
            using namespace privacy_impl;
            if (all_digits(w) && w.size() >= 6)
                return digits_code(w, "word.digits", true);
            const int cls = word_case_class(w);
            if (cls == 3)
                return encode_letters(w, "word.exact") + "!";
            const std::string code = encode_letters(lowercase(w), "word");
            if (cls == 1) { std::string c = code; c[0] = static_cast<char>(c[0] - 32); return c; }
            if (cls == 2) return uppercase(code);
            return code;
        }

        bool decode_word_code(const std::string& token, std::string& word) const
        {
            using namespace privacy_impl;
            if (all_digits(token))
            {
                if (token.size() < 6) return false;
                word = digits_code(token, "word.digits", false);
                return true;
            }
            if (token.size() >= 2 && token.back() == '!')
            {
                const std::string letters = token.substr(0, token.size() - 1);
                for (char c : letters) if (!is_ascii_lower(c)) return false;
                return decode_letters(letters, "word.exact", word);
            }
            int upper = 0, lower = 0;
            for (char c : token)
            {
                if (is_ascii_upper(c)) ++upper;
                else if (is_ascii_lower(c)) ++lower;
                else return false;
            }
            int cls;
            if (upper == 0) cls = 0;
            else if (lower == 0) cls = 2;
            else if (upper == 1 && is_ascii_upper(token[0])) cls = 1;
            else return false;
            std::string lw;
            if (!decode_letters(lowercase(token), "word", lw)) return false;
            word = apply_word_class(lw, cls);
            return true;
        }

        std::string encode_words(pii_type type, const std::string& value) const
        {
            using namespace privacy_impl;
            const std::vector<word_segment> segs = split_words(value);
            const std::unordered_set<std::string>& kept = kept_words(type);
            const std::size_t tld = verbatim_tld_index(type, segs);
            std::string body;
            for (std::size_t k = 0; k < segs.size(); ++k)
            {
                const word_segment& s = segs[k];
                if (s.separator || k == tld || kept.count(fold_key(s.text)) != 0)
                    body += s.text;
                else
                    body += encode_word(s.text);
            }
            return body;
        }

        /* Each word of a body may be verbatim or a code. The usual reading is tried first;
           words that also exist in the kept vocabulary give a second reading, and the tag
           check decides between the combinations. */
        void decode_words(pii_type type, const std::string& body, std::vector<std::string>& candidates) const
        {
            using namespace privacy_impl;
            const std::vector<word_segment> segs = split_words(body);
            const std::unordered_set<std::string>& kept = kept_words(type);
            const std::size_t tld = verbatim_tld_index(type, segs);
            std::vector<std::vector<std::string>> options(segs.size());
            std::size_t combos = 1;
            for (std::size_t k = 0; k < segs.size(); ++k)
            {
                const word_segment& s = segs[k];
                if (s.separator || k == tld)
                {
                    options[k].push_back(s.text);
                    continue;
                }
                /* Every word outside the kept vocabulary is encrypted, so a word that
                   reads as a code is one. A kept word may also read as a code by chance:
                   both readings are tried. */
                std::string decoded;
                const bool is_code = decode_word_code(s.text, decoded);
                const bool is_kept = kept.count(fold_key(s.text)) != 0;
                options[k].push_back(is_kept || !is_code ? s.text : decoded);
                if (is_kept && is_code) options[k].push_back(decoded);
                if (combos <= 64) combos *= options[k].size();
            }
            if (combos > 64)
            {
                std::string first;
                for (std::size_t k = 0; k < segs.size(); ++k)
                    first += options[k][0];
                candidates.push_back(first);
                return;
            }
            std::vector<std::size_t> idx(segs.size(), 0);
            for (;;)
            {
                std::string s;
                for (std::size_t k = 0; k < segs.size(); ++k)
                    s += options[k][idx[k]];
                candidates.push_back(s);
                std::size_t k = 0;
                while (k < segs.size())
                {
                    if (++idx[k] < options[k].size()) break;
                    idx[k] = 0;
                    ++k;
                }
                if (k == segs.size()) break;
            }
        }

        /* Any byte string: a final 0x01 byte keeps empty values, final zero bytes and
           invalid UTF-8 apart from the padding. */
        std::string encode_opaque(pii_type type, const std::string& value) const
        {
            return encode_letters(value + '\x01', std::string("opaque.") + pii_type_keyword(type)) + "!";
        }

        bool decode_opaque(const std::string& letters, const std::string& keyword, std::string& value) const
        {
            const std::size_t m = letters.size();
            std::size_t n = 0;
            for (std::size_t k = 3; k < letter_lengths_.size() && n == 0; ++k)
                if (letter_lengths_[k] == m) n = k;
            if (n == 0 && m > letter_lengths_.back())
            {
                const std::size_t guess = m * 4700 / 8000;
                for (std::size_t k = guess > 2 ? guess - 2 : 1; k <= guess + 2 && n == 0; ++k)
                    if (numerals_needed(k, 26) == m) n = k;
            }
            if (n == 0) return false;
            std::vector<std::uint32_t> y;
            for (char c : letters)
            {
                if (c < 'a' || c > 'z') return false;
                y.push_back(static_cast<std::uint32_t>(c - 'a'));
            }
            std::string plain;
            if (!numerals_to_bytes(word_ff1_.decrypt(y, std::string("opaque.") + keyword), 26, n, plain)) return false;
            while (!plain.empty() && plain.back() == '\0') plain.pop_back();
            if (plain.empty() || plain.back() != '\x01') return false;
            plain.pop_back();
            value = plain;
            return true;
        }

    // ------------------------------------------------------------------------------------
    /* Structured identifiers. Each function works in both directions; forward == true
       encrypts value into out, forward == false decrypts a body into the original. */

        /* Numbers written with typographic spaces ("06 12 34 56 78" with no-break or
           narrow no-break spaces): the digit codecs see plain spaces, and the original
           space characters are put back in the code, so that the value comes back exactly. */
        bool spaced_code(pii_type type, const std::string& v, std::string& out, bool forward) const
        {
            auto code = [&](const std::string& a, std::string& b) {
                switch (type)
                {
                    case pii_type::phone: return code_phone(a, b, forward);
                    case pii_type::iban: return code_iban(a, b, forward);
                    case pii_type::payment_card: return code_card(a, b, forward);
                    case pii_type::nir: return code_nir(a, b, forward);
                    case pii_type::siret: return code_siret(a, b, forward);
                    default: return false;
                }
            };
            std::string plain;
            std::vector<std::pair<std::size_t, std::string>> spaces;
            std::size_t i = 0;
            while (i < v.size())
            {
                const std::size_t at = i;
                const char32_t c = privacy_impl::utf8_next(v, i);
                if (c == 0xA0 || c == 0x202F || c == 0x2009 || c == 0x2007 || c == 0x2008)
                {
                    spaces.push_back(std::make_pair(plain.size(), v.substr(at, i - at)));
                    plain.push_back(' ');
                }
                else
                {
                    plain.append(v, at, i - at);
                }
            }
            if (spaces.empty())
                return code(v, out);
            std::string coded;
            if (!code(plain, coded) || coded.size() != plain.size())
                return false;
            out.clear();
            std::size_t k = 0;
            for (std::size_t j = 0; j < coded.size(); ++j)
            {
                if (k < spaces.size() && spaces[k].first == j && coded[j] == ' ') out += spaces[k++].second;
                else out.push_back(coded[j]);
            }
            return k == spaces.size();
        }

        bool code_phone(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            std::vector<std::size_t> pos;
            std::string d;
            for (std::size_t i = 0; i < v.size(); ++i)
                if (is_ascii_digit_char(v[i])) { pos.push_back(i); d.push_back(v[i]); }
            if (d.size() < 6 || d.size() > 20) return false;
            const std::size_t first = v.find_first_not_of(" \t(");
            std::size_t kept;
            if (first != std::string::npos && v[first] == '+')
                kept = e164_country_code_length(d) + 1;
            else if (d.size() >= 10 && d[0] == '0' && d[1] == '0')
                kept = 2 + e164_country_code_length(d.substr(2)) + 1;
            else if (d[0] == '0')
                kept = 2;
            else
                kept = 1;
            if (d.size() < kept + 6)
                kept = d.size() - 6;
            const std::string y = digits_code(d.substr(kept), "PHONE", forward);
            out = v;
            for (std::size_t k = 0; k < y.size(); ++k)
                out[pos[kept + k]] = y[k];
            return true;
        }

        bool code_iban(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            std::vector<std::size_t> pos;
            std::string a;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                const char c = v[i];
                if (is_ascii_lower(c)) return false;
                if (is_ascii_upper(c) || is_ascii_digit_char(c)) { pos.push_back(i); a.push_back(c); }
                else if (c != ' ') return false;
            }
            if (a.size() < 15 || a.size() > 34) return false;
            if (!is_ascii_upper(a[0]) || !is_ascii_upper(a[1]) || !is_ascii_digit_char(a[2]) ||
                !is_ascii_digit_char(a[3]))
                return false;
            if (forward && !iban_checksum_valid(a)) return false;
            const std::string cc = a.substr(0, 2);
            std::string bban = a.substr(4);
            std::vector<std::size_t> dpos, lpos;
            std::string digits, letters;
            for (std::size_t k = 0; k < bban.size(); ++k)
            {
                if (is_ascii_digit_char(bban[k])) { dpos.push_back(k); digits.push_back(bban[k]); }
                else { lpos.push_back(k); letters.push_back(bban[k]); }
            }
            if (digits.size() < 6) return false;
            const std::string nd = digits_code(digits, "IBAN.digits", forward);
            for (std::size_t k = 0; k < dpos.size(); ++k) bban[dpos[k]] = nd[k];
            if (letters.size() >= 5)
            {
                std::vector<std::uint32_t> x;
                for (char c : letters) x.push_back(static_cast<std::uint32_t>(c - 'A'));
                const std::vector<std::uint32_t> y =
                    forward ? word_ff1_.encrypt(x, "IBAN.letters") : word_ff1_.decrypt(x, "IBAN.letters");
                for (std::size_t k = 0; k < lpos.size(); ++k) bban[lpos[k]] = static_cast<char>('A' + y[k]);
            }
            const std::string na = cc + iban_check_digits(cc, bban) + bban;
            out = v;
            for (std::size_t k = 0; k < na.size(); ++k)
                out[pos[k]] = na[k];
            return true;
        }

        bool code_card(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            std::vector<std::size_t> pos;
            std::string d;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (is_ascii_digit_char(v[i])) { pos.push_back(i); d.push_back(v[i]); }
                else if (v[i] != ' ' && v[i] != '-') return false;
            }
            if (d.size() < 13 || d.size() > 19) return false;
            if (forward && !luhn_valid(d)) return false;
            const std::string head = d.substr(0, 6);
            const std::string mid = digits_code(d.substr(6, d.size() - 7), "CARD", forward);
            const std::string nd = head + mid + luhn_check_digit(head + mid);
            out = v;
            for (std::size_t k = 0; k < nd.size(); ++k) out[pos[k]] = nd[k];
            return true;
        }

        bool code_nir(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            std::vector<std::size_t> pos;
            std::string a;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                const char c = v[i];
                if (is_ascii_digit_char(c) || c == 'A' || c == 'B' || c == 'a' || c == 'b')
                {
                    pos.push_back(i);
                    a.push_back(c);
                }
                else if (c != ' ' && c != '.' && c != '-') return false;
            }
            if (a.size() != 15) return false;
            const bool corsica = a[5] == '2' && (a[6] == 'A' || a[6] == 'B' || a[6] == 'a' || a[6] == 'b');
            std::string digits;
            std::vector<std::size_t> dk;
            for (std::size_t k = 0; k < 13; ++k)
            {
                if (corsica && (k == 5 || k == 6)) continue;
                if (!is_ascii_digit_char(a[k])) return false;
                digits.push_back(a[k]);
                dk.push_back(k);
            }
            if (!is_ascii_digit_char(a[13]) || !is_ascii_digit_char(a[14])) return false;
            auto key_of = [&](const std::string& s13) {
                std::string n = s13;
                if (corsica)
                {
                    const bool is_a = (n[6] == 'A' || n[6] == 'a');
                    n[5] = '1';
                    n[6] = is_a ? '9' : '8';
                }
                long long r = 0;
                for (char c : n) r = (r * 10 + (c - '0')) % 97;
                const int key = 97 - static_cast<int>(r);
                std::string s = std::to_string(key);
                return s.size() < 2 ? "0" + s : s;
            };
            if (forward && key_of(a.substr(0, 13)) != a.substr(13, 2)) return false;
            const std::string nd = digits_code(digits, corsica ? "NIR.corsica" : "NIR", forward);
            std::string na = a;
            for (std::size_t k = 0; k < dk.size(); ++k) na[dk[k]] = nd[k];
            const std::string key = key_of(na.substr(0, 13));
            na[13] = key[0];
            na[14] = key[1];
            out = v;
            for (std::size_t k = 0; k < na.size(); ++k) out[pos[k]] = na[k];
            return true;
        }

        bool code_siret(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            std::vector<std::size_t> pos;
            std::string d;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (is_ascii_digit_char(v[i])) { pos.push_back(i); d.push_back(v[i]); }
                else if (v[i] != ' ' && v[i] != '.') return false;
            }
            if (d.size() != 9 && d.size() != 14) return false;
            if (forward && (!luhn_valid(d) || !luhn_valid(d.substr(0, 9)))) return false;
            /* The SIREN gets the same code alone and inside its SIRET, so that both still
               join; the establishment number (NIC) is kept, and both check digits are
               recomputed. */
            const std::string siren = digits_code(d.substr(0, 8), "SIREN", forward);
            std::string nd = siren + luhn_check_digit(siren);
            if (d.size() == 14)
            {
                const std::string body = nd + d.substr(9, 4);
                nd = body + luhn_check_digit(body);
            }
            out = v;
            for (std::size_t k = 0; k < nd.size(); ++k) out[pos[k]] = nd[k];
            return true;
        }

        bool code_mac(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            std::vector<std::size_t> pos;
            bool upper = false, lower = false;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                const char c = v[i];
                if (std::isxdigit(static_cast<unsigned char>(c)))
                {
                    pos.push_back(i);
                    if (is_ascii_upper(c)) upper = true;
                    if (is_ascii_lower(c)) lower = true;
                }
                else if (c != ':' && c != '-' && c != '.') return false;
            }
            if (pos.size() != 12 || (upper && lower)) return false;
            std::vector<std::uint32_t> x;
            for (std::size_t k = 6; k < 12; ++k)
            {
                const char c = v[pos[k]];
                x.push_back(static_cast<std::uint32_t>(is_ascii_digit_char(c) ? c - '0' : (c | 0x20) - 'a' + 10));
            }
            /* Cycle walking within the class "holds a hexadecimal letter" or "digits only":
               the device half keeps its class, so the case of a value with letters stays
               readable from its code. */
            auto has_letter = [](const std::vector<std::uint32_t>& h) {
                for (std::uint32_t d : h) if (d >= 10) return true;
                return false;
            };
            const bool cls = has_letter(x);
            std::vector<std::uint32_t> y = x;
            do
            {
                y = forward ? hex_ff1_.encrypt(y, "MAC") : hex_ff1_.decrypt(y, "MAC");
            } while (has_letter(y) != cls);
            const char* digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
            out = v;
            for (std::size_t k = 0; k < 6; ++k)
                out[pos[6 + k]] = digits[y[k]];
            return true;
        }

        bool code_plate(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            std::vector<std::size_t> pos;
            std::string a;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (is_ascii_alpha(v[i]) || is_ascii_digit_char(v[i])) { pos.push_back(i); a.push_back(v[i]); }
                else if (v[i] != '-' && v[i] != ' ') return false;
            }
            if (a.size() != 7) return false;
            const bool lower = is_ascii_lower(a[0]);
            for (int k : { 0, 1, 5, 6 })
                if (!is_ascii_alpha(a[static_cast<std::size_t>(k)]) ||
                    is_ascii_lower(a[static_cast<std::size_t>(k)]) != lower)
                    return false;
            for (int k : { 2, 3, 4 })
                if (!is_ascii_digit_char(a[static_cast<std::size_t>(k)])) return false;
            auto li = [&](char c) { return static_cast<std::uint64_t>((c | 0x20) - 'a'); };
            std::uint64_t n = ((li(a[0]) * 26 + li(a[1])) * 26 + li(a[5])) * 26 + li(a[6]);
            n = n * 1000 + static_cast<std::uint64_t>((a[2] - '0') * 100 + (a[3] - '0') * 10 + (a[4] - '0'));
            const std::uint64_t domain = 26ull * 26 * 26 * 26 * 1000;
            // Cycle walking keeps the permutation inside the plate space.
            std::vector<std::uint32_t> x(9);
            do
            {
                for (int k = 8; k >= 0; --k)
                {
                    x[static_cast<std::size_t>(k)] = static_cast<std::uint32_t>(n % 10);
                    n /= 10;
                }
                x = forward ? digit_ff1_.encrypt(x, "PLATE") : digit_ff1_.decrypt(x, "PLATE");
                n = 0;
                for (std::uint32_t d : x) n = n * 10 + d;
            } while (n >= domain);
            const int digits = static_cast<int>(n % 1000);
            std::uint64_t l = n / 1000;
            char letters[4];
            for (int k = 3; k >= 0; --k) { letters[k] = static_cast<char>((lower ? 'a' : 'A') + l % 26); l /= 26; }
            std::string na = { letters[0],
                               letters[1],
                               static_cast<char>('0' + digits / 100),
                               static_cast<char>('0' + (digits / 10) % 10),
                               static_cast<char>('0' + digits % 10),
                               letters[2],
                               letters[3] };
            out = v;
            for (std::size_t k = 0; k < 7; ++k) out[pos[k]] = na[k];
            return true;
        }

        bool code_identifier(const std::string& v, std::string& out, bool forward, bool uuid_codec = true,
                             bool luhn = false) const
        {
            using namespace privacy_impl;
            if (!all_ascii(v)) return false;
            if (uuid_codec && is_uuid_shape(v) && v.find_first_not_of("0123456789-") != std::string::npos)
            {
                /* Device, advertising and cookie identifiers stay hexadecimal UUIDs. The
                   code is walked until it holds a letter, so that the case stays readable;
                   UUIDs made of digits only take the general codec. */
                std::vector<std::uint32_t> x;
                for (char c : v)
                    if (c != '-')
                        x.push_back(static_cast<std::uint32_t>(is_ascii_digit_char(c) ? c - '0' : (c | 0x20) - 'a' + 10));
                auto has_letter = [](const std::vector<std::uint32_t>& h) {
                    for (std::uint32_t d : h) if (d >= 10) return true;
                    return false;
                };
                std::vector<std::uint32_t> y = x;
                do
                {
                    y = forward ? hex_ff1_.encrypt(y, "UUID") : hex_ff1_.decrypt(y, "UUID");
                } while (!has_letter(y));
                bool upper = false;
                for (char c : v) if (is_ascii_upper(c)) upper = true;
                const char* digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
                out = v;
                std::size_t k = 0;
                for (std::size_t i = 0; i < out.size(); ++i)
                    if (out[i] != '-') out[i] = digits[y[k++]];
                return true;
            }
            std::vector<std::size_t> pos;
            std::string a;
            int letters = 0;
            bool upper = false, lower = false;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                const char c = v[i];
                if (is_ascii_alpha(c) || is_ascii_digit_char(c))
                {
                    pos.push_back(i);
                    a.push_back(c);
                    if (is_ascii_alpha(c)) { ++letters; if (is_ascii_upper(c)) upper = true; else lower = true; }
                }
                else if (c != '-' && c != '/' && c != '.' && c != ' ' && c != '_') return false;
            }
            if (letters == 0)
            {
                if (a.size() < 6) return false;
                std::string nd;
                if (luhn && a.size() >= 8)
                {
                    // IMEI, ICCID and other Luhn numbers: the payload is encrypted, the check digit recomputed.
                    if (forward && !luhn_valid(a)) return false;
                    nd = digits_code(a.substr(0, a.size() - 1), "ID.luhn", forward);
                    nd.push_back(luhn_check_digit(nd));
                }
                else
                {
                    nd = digits_code(a, "ID", forward);
                }
                out = v;
                for (std::size_t k = 0; k < nd.size(); ++k) out[pos[k]] = nd[k];
                return true;
            }
            if (a.size() < 4) return false;
            const bool mixed = upper && lower;
            const ff1_cipher& c = mixed ? alnum62_ff1_ : alnum36_ff1_;
            const std::string alphabet = mixed
                ? "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                : (upper ? "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" : "0123456789abcdefghijklmnopqrstuvwxyz");
            auto acceptable = [&](const std::string& s) {
                bool u = false, l = false, any = false;
                for (char ch : s)
                {
                    if (is_ascii_upper(ch)) { u = true; any = true; }
                    if (is_ascii_lower(ch)) { l = true; any = true; }
                }
                return mixed ? (u && l) : any;
            };
            std::string s = a;
            do
            {
                s = forward ? c.encrypt(s, alphabet, mixed ? "ID.62" : "ID.36")
                            : c.decrypt(s, alphabet, mixed ? "ID.62" : "ID.36");
            } while (!acceptable(s));
            out = v;
            for (std::size_t k = 0; k < s.size(); ++k) out[pos[k]] = s[k];
            return true;
        }

        // GPS coordinates: every digit is encrypted, signs, separators and hemispheres kept.
        bool code_geo(const std::string& v, std::string& out, bool forward) const
        {
            using namespace privacy_impl;
            std::string digits;
            std::vector<std::size_t> pos;
            for (std::size_t i = 0; i < v.size(); ++i)
                if (is_ascii_digit_char(v[i])) { pos.push_back(i); digits.push_back(v[i]); }
            if (digits.size() < 6) return false;
            const std::string nd = digits_code(digits, "GEO", forward);
            out = v;
            for (std::size_t k = 0; k < nd.size(); ++k) out[pos[k]] = nd[k];
            return true;
        }

        bool code_ipv4(const std::string& v, std::string& out, bool forward) const
        {
            std::uint32_t a;
            if (!privacy_impl::parse_ipv4(v, a)) return false;
            out = privacy_impl::render_ipv4(forward ? ip_.encrypt_ipv4(a) : ip_.decrypt_ipv4(a));
            return true;
        }

        bool code_ipv6(const std::string& v, std::string& out, bool forward) const
        {
            std::array<unsigned char,16> a;
            if (!privacy_impl::parse_ipv6(v, a) || privacy_impl::render_ipv6(a) != v) return false;
            out = privacy_impl::render_ipv6(forward ? ip_.encrypt_ipv6(a) : ip_.decrypt_ipv6(a));
            return true;
        }

        bool date_forward(const std::string& v, std::string& body) const
        {
            civil_date d;
            date_format f;
            if (!parse_date(v, options_.date_order, d, f)) return false;
            if (format_date(d, f) != v) return false;
            const civil_date s = civil_from_days(days_from_civil(d.year, d.month, d.day) + shift_days_);
            body = format_date(s, date_format());
            if (!f.is_iso())
                body += "~" + privacy_impl::crockford_encode(f.pack(), 4);
            return true;
        }

        bool date_backward(const std::string& body, std::string& original) const
        {
            if (body.size() < 10) return false;
            const std::string iso = body.substr(0, 10);
            date_format f;
            if (body.size() > 10)
            {
                std::uint32_t packed;
                if (body[10] != '~' || !privacy_impl::crockford_decode(body.substr(11), packed)) return false;
                f = date_format::unpack(packed);
            }
            // YYYY-MM-DD, read directly: a shifted date may fall outside the years find_dates() accepts.
            if (iso[4] != '-' || iso[7] != '-') return false;
            for (int k : { 0, 1, 2, 3, 5, 6, 8, 9 })
                if (!privacy_impl::is_ascii_digit_char(iso[static_cast<std::size_t>(k)])) return false;
            civil_date d;
            d.year = std::stoi(iso.substr(0, 4));
            d.month = std::stoi(iso.substr(5, 2));
            d.day = std::stoi(iso.substr(8, 2));
            if (!is_valid_civil_date(d.year, d.month, d.day)) return false;
            const civil_date o = civil_from_days(days_from_civil(d.year, d.month, d.day) - shift_days_);
            original = format_date(o, f);
            return true;
        }

    // ------------------------------------------------------------------------------------

        bool decode_inner(const std::string& inner, pii_type& type, std::string& original) const
        {
            using namespace privacy_impl;
            const std::size_t colon = inner.find(':');
            const std::size_t hash = inner.rfind('#');
            if (colon == std::string::npos || hash == std::string::npos || hash < colon || inner.size() - hash != 4)
                return false;
            const std::string keyword = inner.substr(0, colon);
            type = pii_type::none;
            for (pii_type t : all_pii_types())
                if (keyword == pii_type_keyword(t)) type = t;
            if (type == pii_type::none) return false;
            const std::string body = inner.substr(colon + 1, hash - colon - 1);
            std::string check = inner.substr(hash + 1);
            for (char& c : check)
                if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 32);
            if (body.empty()) return false;
            if (attribute_type(type) && decode_attribute(type, body, check, original))
                return true;

            std::vector<std::string> candidates;
            if (body.size() >= 2 && body.back() == '!')
            {
                std::string v;
                if (decode_opaque(body.substr(0, body.size() - 1), keyword, v))
                    candidates.push_back(v);
            }
            std::string v;
            switch (type)
            {
                case pii_type::person:
                case pii_type::organization:
                case pii_type::location:
                case pii_type::address:
                case pii_type::email:
                case pii_type::url:
                    decode_words(type, body, candidates);
                    break;
                case pii_type::phone:
                case pii_type::iban:
                case pii_type::payment_card:
                case pii_type::nir:
                case pii_type::siret:
                    if (spaced_code(type, body, v, false)) candidates.push_back(v);
                    break;
                case pii_type::mac_address: if (code_mac(body, v, false)) candidates.push_back(v); break;
                case pii_type::license_plate: if (code_plate(body, v, false)) candidates.push_back(v); break;
                case pii_type::identifier:
                    if (code_identifier(body, v, false)) candidates.push_back(v);
                    // A UUID shaped body may also come from the general alphanumeric codec.
                    if (privacy_impl::is_uuid_shape(body) && code_identifier(body, v, false, false))
                        candidates.push_back(v);
                    // A body of digits with a valid Luhn digit may come from the Luhn codec.
                    if (code_identifier(body, v, false, false, true)) candidates.push_back(v);
                    break;
                case pii_type::geolocation: if (code_geo(body, v, false)) candidates.push_back(v); break;
                case pii_type::ipv4: if (code_ipv4(body, v, false)) candidates.push_back(v); break;
                case pii_type::ipv6: if (code_ipv6(body, v, false)) candidates.push_back(v); break;
                case pii_type::date: if (date_backward(body, v)) candidates.push_back(v); break;
                case pii_type::title:
                case pii_type::gender:
                case pii_type::nationality:
                case pii_type::none: break;
            }
            for (const std::string& c : candidates)
            {
                const std::string expected = check_code(type, c);
                if (expected.size() == check.size() && constant_time_equal(expected.data(), check.data(), check.size()))
                {
                    original = c;
                    return true;
                }
            }
            return false;
        }

        static std::size_t find_ci(const std::string& text, const std::string& needle, std::size_t from)
        {
            for (std::size_t i = from; i + needle.size() <= text.size(); ++i)
            {
                bool ok = true;
                for (std::size_t k = 0; k < needle.size() && ok; ++k)
                {
                    char a = text[i + k];
                    if (a >= 'a' && a <= 'z') a = static_cast<char>(a - 32);
                    ok = a == needle[k];
                }
                if (ok) return i;
            }
            return std::string::npos;
        }

        /* Calls found(begin, end, type, original) for each verified tag of text, left to
           right. A tag body never holds the opening marker, so a candidate is the text
           between an opening marker and the first closing marker with no other opening
           marker in between: the scan stays linear, and a tag preceded by '[' or by a
           stray opening marker is still found. A rejected candidate is one that has the
           shape of a tag, a keyword and a check. */
        template <typename callback>
        void scan_tags(const std::string& text, const char* open, const char* close, bool percent,
                       restore_report* report, callback found) const
        {
            const std::string o(open), c(close);
            auto find = [&](const std::string& needle, std::size_t from) {
                return percent ? find_ci(text, needle, from) : text.find(needle, from);
            };
            std::size_t pos = 0, q = std::string::npos;
            bool q_valid = false;
            for (;;)
            {
                const std::size_t p = find(o, pos);
                if (p == std::string::npos) break;
                if (!q_valid || q < p + o.size())
                {
                    q = find(c, p + o.size());
                    q_valid = true;
                }
                if (q == std::string::npos) break;
                const std::size_t inner = find(o, p + 1);
                if (inner != std::string::npos && inner < q)
                {
                    pos = inner;
                    continue;
                }
                const std::string raw = text.substr(p + o.size(), q - p - o.size());
                const std::string body = percent ? privacy_impl::percent_decode(raw) : raw;
                pii_type type;
                std::string original;
                if (decode_inner(body, type, original))
                {
                    found(p, q + c.size(), type, original);
                    if (report) ++report->restored;
                    pos = q + c.size();
                    q_valid = false;
                }
                else
                {
                    const std::size_t colon = body.find(':');
                    if (report && colon != std::string::npos && body.find('#', colon) != std::string::npos)
                        ++report->rejected;
                    pos = q;
                }
            }
        }

        std::string restore_raw(const std::string& text, restore_report* report) const
        {
            std::string out;
            std::size_t pos = 0;
            scan_tags(text, "[[", "]]", false, report, [&](std::size_t b, std::size_t e, pii_type,
                                                         const std::string& original) {
                out.append(text, pos, b - pos);
                out += original;
                pos = e;
            });
            out.append(text, pos, std::string::npos);
            return out;
        }

        std::string restore_percent(const std::string& text, restore_report* report) const
        {
            if (text.find('%') == std::string::npos)
                return text;
            std::string out;
            std::size_t pos = 0;
            scan_tags(text, "%5B%5B", "%5D%5D", true, report, [&](std::size_t b, std::size_t e, pii_type,
                                                                 const std::string& original) {
                out.append(text, pos, b - pos);
                out += original;
                pos = e;
            });
            out.append(text, pos, std::string::npos);
            return out;
        }

        // True if tag decodes, with its check, back to exactly value.
        bool decodes_to(const std::string& tag, const std::string& value) const
        {
            if (tag.size() < 4 || tag.find("]]") != tag.size() - 2 || tag.find("[[", 1) != std::string::npos)
                return false;
            pii_type type;
            std::string original;
            return decode_inner(tag.substr(2, tag.size() - 4), type, original) && original == value;
        }

        pseudonymizer_options options_;
        ff1_cipher word_ff1_;
        ff1_cipher digit_ff1_;
        ff1_cipher hex_ff1_;
        ff1_cipher alnum36_ff1_;
        ff1_cipher alnum62_ff1_;
        prefix_preserving_ip_cipher ip_;
        std::string check_key_;
        long shift_days_ = 0;
        std::vector<std::size_t> letter_lengths_;
        static const std::size_t max_cache_entries = 1 << 16;
        mutable std::mutex cache_mutex_;
        mutable std::unordered_map<std::string, std::string> cache_;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PSEUDONYMIZER_H_
