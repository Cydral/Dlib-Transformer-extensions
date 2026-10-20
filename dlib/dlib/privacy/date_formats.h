// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_DATE_FORMATS_H_
#define DLIB_PRIVACY_DATE_FORMATS_H_

#include "date_formats_abstract.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "unicode_text.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct civil_date
    {
        int year = 0;
        int month = 0;
        int day = 0;
    };

    inline long days_from_civil(int y, int m, int d)
    {
        y -= m <= 2 ? 1 : 0;
        const long era = (y >= 0 ? y : y - 399) / 400;
        const long yoe = y - era * 400;
        const long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
        const long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097 + doe - 719468;
    }

    inline civil_date civil_from_days(long z)
    {
        z += 719468;
        const long era = (z >= 0 ? z : z - 146096) / 146097;
        const long doe = z - era * 146097;
        const long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const long y = yoe + era * 400;
        const long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const long mp = (5 * doy + 2) / 153;
        const long d = doy - (153 * mp + 2) / 5 + 1;
        const long m = mp < 10 ? mp + 3 : mp - 9;
        civil_date r;
        r.year = static_cast<int>(y + (m <= 2 ? 1 : 0));
        r.month = static_cast<int>(m);
        r.day = static_cast<int>(d);
        return r;
    }

    inline int days_in_month(int y, int m)
    {
        static const int days[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        if (m == 2)
            return ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 29 : 28;
        return (m >= 1 && m <= 12) ? days[m - 1] : 0;
    }

    inline bool is_valid_civil_date(int y, int m, int d)
    {
        return m >= 1 && m <= 12 && d >= 1 && d <= days_in_month(y, m);
    }

// ----------------------------------------------------------------------------------------

    enum class numeric_date_order
    {
        day_month_year,
        month_day_year
    };

    struct date_format
    {
        std::uint8_t layout = 2;
        std::uint8_t separator = 1;
        std::uint8_t day_style = 1;
        std::uint8_t month_style = 1;
        std::uint8_t month_case = 0;
        std::uint8_t two_digit_year = 0;
        std::uint8_t comma = 0;

        std::uint32_t pack() const
        {
            return static_cast<std::uint32_t>(layout) | (static_cast<std::uint32_t>(separator) << 3) |
                   (static_cast<std::uint32_t>(day_style) << 5) | (static_cast<std::uint32_t>(month_style) << 7) |
                   (static_cast<std::uint32_t>(month_case) << 13) | (static_cast<std::uint32_t>(two_digit_year) << 15) |
                   (static_cast<std::uint32_t>(comma) << 16);
        }

        static date_format unpack(std::uint32_t v)
        {
            date_format f;
            f.layout = static_cast<std::uint8_t>(v & 7);
            f.separator = static_cast<std::uint8_t>((v >> 3) & 3);
            f.day_style = static_cast<std::uint8_t>((v >> 5) & 3);
            f.month_style = static_cast<std::uint8_t>((v >> 7) & 63);
            f.month_case = static_cast<std::uint8_t>((v >> 13) & 3);
            f.two_digit_year = static_cast<std::uint8_t>((v >> 15) & 1);
            f.comma = static_cast<std::uint8_t>((v >> 16) & 1);
            return f;
        }

        bool is_iso() const
        {
            return layout == 2 && separator == 1 && day_style == 1 && month_style == 1 && two_digit_year == 0;
        }

        bool has_day() const { return layout <= 4; }
        bool has_month() const { return layout <= 6; }
    };

    struct date_match
    {
        std::size_t begin = 0;
        std::size_t end = 0;
        civil_date date;
        date_format format;
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        struct month_name_variant
        {
            int month;
            const char* text;
        };

        inline const std::vector<month_name_variant>& month_name_variants()
        {
            static const std::vector<month_name_variant> v = {
                {1, "janvier"}, {2, "f\xC3\xA9vrier"}, {3, "mars"}, {4, "avril"}, {5, "mai"}, {6, "juin"},
                {7, "juillet"}, {8, "ao\xC3\xBBt"}, {9, "septembre"}, {10, "octobre"}, {11, "novembre"},
                {12, "d\xC3\xA9" "cembre"},
                {2, "fevrier"}, {8, "aout"}, {12, "decembre"},
                {1, "janv."}, {2, "f\xC3\xA9vr."}, {2, "f\xC3\xA9v."}, {4, "avr."}, {7, "juil."}, {9, "sept."},
                {10, "oct."}, {11, "nov."}, {12, "d\xC3\xA9" "c."}, {2, "fevr."},
                {1, "janv"}, {2, "f\xC3\xA9vr"}, {4, "avr"}, {7, "juil"},
                {1, "january"}, {2, "february"}, {3, "march"}, {4, "april"}, {5, "may"}, {6, "june"},
                {7, "july"}, {8, "august"}, {9, "september"}, {10, "october"}, {11, "november"}, {12, "december"},
                {1, "jan"}, {2, "feb"}, {3, "mar"}, {4, "apr"}, {6, "jun"}, {7, "jul"}, {8, "aug"},
                {9, "sep"}, {9, "sept"}, {10, "oct"}, {11, "nov"}, {12, "dec"},
                {1, "jan."}, {2, "feb."}, {3, "mar."}, {4, "apr."}, {6, "jun."}, {7, "jul."}, {8, "aug."},
                {9, "sep."}, {12, "dec."}
            };
            return v;
        }

        inline std::string apply_case(const std::string& lower_text, int case_style)
        {
            if (case_style == 1) return titlecase(lower_text);
            if (case_style == 2) return uppercase(lower_text);
            return lower_text;
        }

        // Matches a month name at code point i. Returns the variant index or -1.
        inline int match_month_name(const utf8_text& t, std::size_t i, std::size_t& end, int& case_style)
        {
            const auto& variants = month_name_variants();
            int best = -1;
            std::size_t best_len = 0;
            int best_case = 0;
            for (std::size_t v = 0; v < variants.size(); ++v)
            {
                const std::string name = variants[v].text;
                std::size_t k = 0, j = i;
                bool ok = true;
                int upper = 0, lower = 0;
                bool first_upper = false;
                std::size_t n = 0;
                while (k < name.size())
                {
                    const char32_t want = utf8_next(name, k);
                    if (j >= t.size()) { ok = false; break; }
                    const char32_t got = t.cps[j];
                    if (to_lower(got) != want) { ok = false; break; }
                    if (is_upper(got)) { ++upper; if (n == 0) first_upper = true; }
                    else if (is_lower(got)) ++lower;
                    ++j;
                    ++n;
                }
                if (!ok) continue;
                if (name.back() != '.' && j < t.size() && is_letter(t.cps[j])) continue;
                int cs;
                if (upper == 0) cs = 0;
                else if (lower == 0) cs = 2;
                else if (first_upper && upper == 1) cs = 1;
                else continue;
                if (n > best_len)
                {
                    best = static_cast<int>(v);
                    best_len = n;
                    best_case = cs;
                    end = j;
                }
            }
            case_style = best_case;
            return best;
        }

        inline bool read_digits(const utf8_text& t, std::size_t& j, int max_digits, int& value, int& count)
        {
            value = 0;
            count = 0;
            while (j < t.size() && count < max_digits && is_ascii_digit(t.cps[j]))
            {
                value = value * 10 + static_cast<int>(t.cps[j] - '0');
                ++j;
                ++count;
            }
            if (j < t.size() && is_ascii_digit(t.cps[j]))
                return false;
            return count > 0;
        }

        inline bool is_date_space(char32_t c) { return c == ' ' || c == 0xA0; }

        inline int expand_two_digit_year(int yy) { return yy < 50 ? 2000 + yy : 1900 + yy; }

        inline bool plausible_year(int y) { return y >= 1800 && y <= 2199; }

        // A year written without its day: a narrower range, since four digits alone are often something else.
        inline bool plausible_partial_year(int y) { return y >= 1900 && y <= 2099; }

        // The lower case word made of the letters that end at code point e, and its first code point.
        inline std::string letters_before(const utf8_text& t, std::size_t e, std::size_t& b)
        {
            b = e;
            while (b > 0 && is_letter(t.cps[b - 1]))
                --b;
            std::string w;
            for (std::size_t k = b; k < e; ++k)
                utf8_append(w, to_lower(t.cps[k]));
            return w;
        }

        inline bool is_range_dash(char32_t c) { return c == '-' || c == 0x2013 || c == 0x2014; }

        enum class year_cue
        {
            none,
            alone,      // "en 2019", "since 2019", "Année de naissance : 1985"
            range       // "de 2015 à 2019", "between 2015 and 2019": only with the second year
        };

        /* What the words before the year starting at code point i make of it. A year alone
           is a date of the document after a word of time: a preposition ("en", "depuis",
           "in", "since"), a part of a year ("fin", "mi-", "early", "summer"), "année",
           "l'an", "class of", or the label of a birth year. */
        inline year_cue year_cue_before(const utf8_text& t, std::size_t i)
        {
            static const std::vector<std::string> alone = {
                "en", "depuis", "avant", "apres", "fin", "debut", "courant", "vers", "annee", "printemps",
                "ete", "automne", "hiver", "session", "promotion", "promo", "in", "since", "until", "till",
                "before", "after", "around", "circa", "early", "late", "spring", "summer", "autumn", "fall",
                "winter", "year", "born"
            };
            static const std::vector<std::string> range = { "de", "entre", "from", "between" };
            static const std::vector<std::string> labels = { "naissance", "birth", "dob" };
            auto in = [](const std::vector<std::string>& v, const std::string& w) {
                for (const std::string& s : v)
                    if (s == w) return true;
                return false;
            };
            if (i == 0)
                return year_cue::none;
            std::size_t b;
            if (t.cps[i - 1] == '-')
            {
                // "mi-2021", "mid-2021".
                const std::string w = fold_key(letters_before(t, i - 1, b));
                return (w == "mi" || w == "mid") && !is_letter(t.before(b)) ? year_cue::alone : year_cue::none;
            }
            std::size_t k = i;
            while (k > 0 && t.cps[k - 1] == ' ' && i - k < 2)
                --k;
            if (k == i)
                return year_cue::none;
            if (k > 0 && t.cps[k - 1] == ':')
            {
                // A field label: "Année de naissance : 1985", "Year of birth: 1985".
                --k;
                while (k > 0 && t.cps[k - 1] == ' ')
                    --k;
                const std::string w = fold_key(letters_before(t, k, b));
                return in(labels, w) ? year_cue::alone : year_cue::none;
            }
            const std::string lower = letters_before(t, k, b);
            if (lower.empty() || is_ascii_digit(t.before(b)))
                return year_cue::none;
            if (lower == "d\xC3\xA8s")
                return year_cue::alone;
            const std::string w = fold_key(lower);
            if (w == "an")
            {
                // "l'an 2019", not "un an".
                return b >= 2 && is_apostrophe(t.cps[b - 1]) && to_lower(t.cps[b - 2]) == 'l' &&
                       !is_letter(t.before(b - 2)) ? year_cue::alone : year_cue::none;
            }
            if (w == "of")
            {
                // "class of 2015".
                std::size_t c = b;
                if (c == 0 || t.cps[c - 1] != ' ')
                    return year_cue::none;
                return fold_key(letters_before(t, c - 1, c)) == "class" ? year_cue::alone : year_cue::none;
            }
            // A word glued to an apostrophe before it is still the word: "jusqu'en 2025".
            if (in(alone, w))
                return year_cue::alone;
            if (w == "de" && b > 0 && t.cps[b - 1] == ' ')
            {
                // "à la session de 2011", "promotion de 2015".
                std::size_t c;
                const std::string before = fold_key(letters_before(t, b - 1, c));
                if (before == "session" || before == "promotion")
                    return year_cue::alone;
            }
            if (in(range, w))
                return year_cue::range;
            return year_cue::none;
        }

        // True if the year that ends at code point j stands on its own: no letter, digit or number after it.
        inline bool year_ends(const utf8_text& t, std::size_t j)
        {
            const char32_t a = t.at(j);
            if (is_word_char(a))
                return false;
            const char32_t b = t.at(j + 1);
            if ((a == '/' || a == '-' || a == '.' || a == ',') && is_ascii_digit(b))
                return false;
            if (a == ':' && is_word_char(b))
                return false;
            return true;
        }

        // True if a unit or a counted noun follows the number that ends at code point j: "en 2000 exemplaires".
        inline bool quantity_after(const utf8_text& t, std::size_t j)
        {
            static const std::vector<std::string> units = {
                "an", "ans", "annee", "annees", "year", "years", "mois", "month", "months", "semaine", "semaines",
                "week", "weeks", "jour", "jours", "day", "days", "heure", "heures", "hour", "hours", "h", "min",
                "minutes", "km", "m", "m2", "cm", "mm", "kg", "g", "t", "tonne", "tonnes", "ton", "tons", "l",
                "litre", "litres", "liter", "liters", "euro", "euros", "eur", "dollar", "dollars", "usd", "franc",
                "francs", "chf", "livres", "pounds", "gbp", "exemplaire", "exemplaires", "copie", "copies",
                "copy", "page", "pages", "mot", "mots", "word", "words", "personne", "personnes", "person",
                "persons", "people", "habitant", "habitants", "inhabitants", "residents", "salarie", "salaries",
                "employe", "employes", "employee", "employees", "membre", "membres", "member", "members",
                "unite", "unites", "unit", "units", "piece", "pieces", "point", "points", "fois", "times", "voix",
                "votes", "signatures", "places", "lits", "beds", "patients", "dossiers", "cas", "cases",
                "articles", "lignes", "lines", "octets", "bytes", "mo", "go", "ko", "mb", "gb", "kb", "metre",
                "metres", "meter", "meters", "kilometres", "kilometers", "miles", "pas", "steps", "kcal",
                "participants", "candidats", "candidates", "eleves", "students", "etudiants", "enfants",
                "children", "clients", "customers", "utilisateurs", "users", "visiteurs", "visitors",
                "entreprises", "companies", "logements", "emplois", "jobs", "postes", "projets", "projects"
            };
            std::size_t k = j;
            if (t.at(k) == ' ')
                ++k;
            const char32_t c = t.at(k);
            if (c == '%' || c == 0x20AC || c == '$' || c == 0xA3)
                return true;
            std::size_t e = k;
            while (e < t.size() && is_word_char(t.cps[e]))
                ++e;
            if (e == k || k == j)
                return false;
            std::string w;
            for (std::size_t n = k; n < e; ++n)
                utf8_append(w, to_lower(t.cps[n]));
            w = fold_key(w);
            for (const std::string& u : units)
                if (u == w) return true;
            return false;
        }

        // True if the text between the start of the line and code point i is empty or a bullet.
        inline bool starts_line(const utf8_text& t, std::size_t i)
        {
            std::size_t k = i;
            while (k > 0 && (t.cps[k - 1] == ' ' || t.cps[k - 1] == '\t'))
                --k;
            if (k > 0 && (t.cps[k - 1] == '-' || t.cps[k - 1] == '*' || t.cps[k - 1] == 0x2022 ||
                          t.cps[k - 1] == 0x2013 || t.cps[k - 1] == 0x2014))
            {
                --k;
                while (k > 0 && (t.cps[k - 1] == ' ' || t.cps[k - 1] == '\t'))
                    --k;
            }
            return k == 0 || t.cps[k - 1] == '\n' || t.cps[k - 1] == '\r';
        }

        // True if a colon follows code point j, as in the lines of a CV: "2015 – 2019 : ingénieur".
        inline bool colon_after(const utf8_text& t, std::size_t j)
        {
            std::size_t k = j;
            while (k < t.size() && t.cps[k] == ' ')
                ++k;
            return t.at(k) == ':' && !is_word_char(t.at(k + 1));
        }

        /* A second year after the year that ends at code point j: "2015-2019", "2015 – 2019",
           "2015 à 2019", "2015 and 2019". Sets [b, e) and the year. */
        inline bool second_year(const utf8_text& t, std::size_t j, std::size_t& b, std::size_t& e, int& year)
        {
            static const std::vector<std::string> links = {
                "a", "au", "et", "to", "and", "until", "till", "through", "thru"
            };
            std::size_t k = j;
            if (t.at(k) == ' ')
                ++k;
            if (is_range_dash(t.at(k)))
            {
                ++k;
                if (t.at(k) == ' ')
                    ++k;
            }
            else
            {
                if (k == j)
                    return false;
                std::size_t w = k;
                while (w < t.size() && is_letter(t.cps[w]))
                    ++w;
                std::string word;
                for (std::size_t n = k; n < w; ++n)
                    utf8_append(word, to_lower(t.cps[n]));
                word = fold_key(word);
                bool link = false;
                for (const std::string& s : links)
                    if (s == word) link = true;
                if (!link || t.at(w) != ' ')
                    return false;
                k = w + 1;
            }
            int count;
            b = k;
            e = k;
            if (!read_digits(t, e, 4, year, count) || count != 4 || !plausible_partial_year(year) ||
                is_word_char(t.before(b)))
                return false;
            return year_ends(t, e) && !quantity_after(t, e);
        }

        // A number or a reference before code point i: "n° 03/2022", "Réf. 10/2025", "#12/2024".
        inline bool reference_before(const utf8_text& t, std::size_t i)
        {
            std::size_t k = i;
            while (k > 0 && (t.cps[k - 1] == ' ' || t.cps[k - 1] == ':' || t.cps[k - 1] == '.'))
                --k;
            const char32_t c = t.before(k);
            if (c == 0xB0 || c == 0xBA || c == '#' || c == 0x2116)
                return true;
            std::size_t b;
            const std::string w = fold_key(letters_before(t, k, b));
            return w == "no" || w == "nr" || w == "num" || w == "numero" || w == "ref" || w == "reference" ||
                   w == "dossier" || w == "affaire" || w == "lot" || w == "facture" || w == "invoice" ||
                   w == "number" || w == "version";
        }
    }

// ----------------------------------------------------------------------------------------

    inline std::vector<date_match> find_dates(
        const std::string& text,
        numeric_date_order order = numeric_date_order::day_month_year
    )
    {
        using namespace privacy_impl;
        const utf8_text t(text, true);
        std::vector<date_match> out;
        const std::size_t n = t.size();
        std::size_t i = 0;
        while (i < n)
        {
            const char32_t c = t.cps[i];
            const char32_t prev = t.before(i);
            bool matched = false;
            date_match m;

            if (is_ascii_digit(c) && !is_word_char(prev) &&
                !((prev == '/' || prev == '-' || prev == '.') && i >= 2 && is_ascii_digit(t.cps[i - 2])))
            {
                // Numeric layouts.
                std::size_t j = i;
                int a, la;
                if (read_digits(t, j, 4, a, la) && j < n && (t.cps[j] == '/' || t.cps[j] == '-' || t.cps[j] == '.'))
                {
                    const char32_t sep = t.cps[j];
                    ++j;
                    int b, lb;
                    if (read_digits(t, j, 2, b, lb) && j < n && t.cps[j] == sep)
                    {
                        ++j;
                        int cval, lc;
                        if (read_digits(t, j, 4, cval, lc))
                        {
                            const char32_t after = t.at(j);
                            bool boundary = !is_word_char(after) &&
                                !((after == '/' || after == '-' || after == '.') && is_ascii_digit(t.at(j + 1)));
                            date_format f;
                            f.separator = static_cast<std::uint8_t>(sep == '/' ? 0 : (sep == '-' ? 1 : 2));
                            civil_date d;
                            bool ok = false;
                            if (la == 4 && (lc == 1 || lc == 2))
                            {
                                if (after == 'T' && is_ascii_digit(t.at(j + 1)))
                                    boundary = true;
                                d.year = a; d.month = b; d.day = cval;
                                f.layout = 2;
                                f.month_style = (lb == 2) ? 1 : 0;
                                f.day_style = (lc == 2) ? 1 : 0;
                                ok = plausible_year(d.year);
                            }
                            else if ((la == 1 || la == 2) && (lc == 2 || lc == 4))
                            {
                                int day = a, month = b;
                                int dlen = la, mlen = lb;
                                bool mdy = (order == numeric_date_order::month_day_year);
                                if (a > 12 && b <= 12) mdy = false;
                                else if (b > 12 && a <= 12) mdy = true;
                                if (mdy) { day = b; month = a; dlen = lb; mlen = la; }
                                d.day = day; d.month = month;
                                d.year = (lc == 4) ? cval : expand_two_digit_year(cval);
                                f.layout = mdy ? 1 : 0;
                                f.day_style = (dlen == 2) ? 1 : 0;
                                f.month_style = (mlen == 2) ? 1 : 0;
                                f.two_digit_year = (lc == 2) ? 1 : 0;
                                ok = plausible_year(d.year);
                            }
                            if (ok && boundary && is_valid_civil_date(d.year, d.month, d.day))
                            {
                                m.begin = t.offsets[i];
                                m.end = t.offsets[j];
                                m.date = d;
                                m.format = f;
                                matched = true;
                                i = j;
                            }
                        }
                    }
                }

                // Text layout "12 mars 1985", "1er mars 1985", "12th March 1985".
                if (!matched)
                {
                    std::size_t j = i;
                    int day, ld;
                    if (read_digits(t, j, 2, day, ld))
                    {
                        date_format f;
                        f.layout = 3;
                        f.day_style = (ld == 2) ? 1 : 0;
                        if (to_lower(t.at(j)) == 'e' && to_lower(t.at(j + 1)) == 'r' && !is_letter(t.at(j + 2)))
                        {
                            if (day == 1) { f.day_style = 2; j += 2; }
                        }
                        else if (is_letter(t.at(j)) && is_letter(t.at(j + 1)) && !is_letter(t.at(j + 2)))
                        {
                            const std::string suf = { static_cast<char>(to_lower(t.at(j))),
                                                      static_cast<char>(to_lower(t.at(j + 1))) };
                            if (suf == "st" || suf == "nd" || suf == "rd" || suf == "th") { f.day_style = 3; j += 2; }
                        }
                        if (j < n && is_date_space(t.cps[j]) && t.cps[j] == ' ')
                        {
                            std::size_t k = j + 1, mend = 0;
                            int cs = 0;
                            const int variant = match_month_name(t, k, mend, cs);
                            if (variant >= 0 && mend < n && t.cps[mend] == ' ')
                            {
                                std::size_t y = mend + 1;
                                int year, ly;
                                if (read_digits(t, y, 4, year, ly) && ly == 4 && !is_word_char(t.at(y)) &&
                                    plausible_year(year))
                                {
                                    const int month = month_name_variants()[static_cast<std::size_t>(variant)].month;
                                    if (is_valid_civil_date(year, month, day))
                                    {
                                        f.month_style = static_cast<std::uint8_t>(variant);
                                        f.month_case = static_cast<std::uint8_t>(cs);
                                        m.begin = t.offsets[i];
                                        m.end = t.offsets[y];
                                        m.date.year = year; m.date.month = month; m.date.day = day;
                                        m.format = f;
                                        matched = true;
                                        i = y;
                                    }
                                }
                            }
                        }
                    }
                }

                // Month and year in figures: "03/2022", "10.2025", not a reference "n° 03/2022".
                if (!matched)
                {
                    std::size_t j = i;
                    int month, lm;
                    if (read_digits(t, j, 2, month, lm) && lm == 2 && month >= 1 && month <= 12 && j < n &&
                        (t.cps[j] == '/' || t.cps[j] == '.'))
                    {
                        const char32_t sep = t.cps[j];
                        std::size_t y = j + 1;
                        int year, ly;
                        if (read_digits(t, y, 4, year, ly) && ly == 4 && plausible_partial_year(year) &&
                            year_ends(t, y) && !reference_before(t, i))
                        {
                            date_format f;
                            f.layout = 6;
                            f.separator = static_cast<std::uint8_t>(sep == '/' ? 0 : 2);
                            f.day_style = 0;
                            f.month_style = 1;
                            m.begin = t.offsets[i];
                            m.end = t.offsets[y];
                            m.date.year = year; m.date.month = month; m.date.day = 0;
                            m.format = f;
                            matched = true;
                            i = y;
                        }
                    }
                }

                /* A year alone, after a word of time ("en 2019", "since 2019", "fin 2022") or at
                   the head of a line before a colon, as in a CV; with the second year of a range
                   ("de 2015 à 2019", "2015-2019 :"). */
                if (!matched)
                {
                    std::size_t j = i;
                    int year, ly;
                    if (read_digits(t, j, 4, year, ly) && ly == 4 && plausible_partial_year(year))
                    {
                        const year_cue cue = year_cue_before(t, i);
                        std::size_t b2 = 0, e2 = 0;
                        int year2 = 0;
                        const bool pair = second_year(t, j, b2, e2, year2);
                        const bool line = starts_line(t, i) && colon_after(t, pair ? e2 : j);
                        const bool first = pair ? (cue != year_cue::none || line)
                                                : (year_ends(t, j) && !quantity_after(t, j) &&
                                                   (cue == year_cue::alone || line));
                        if (first)
                        {
                            date_format f;
                            f.layout = 7;
                            f.separator = 0;
                            f.day_style = 0;
                            f.month_style = 0;
                            m.begin = t.offsets[i];
                            m.end = t.offsets[j];
                            m.date.year = year; m.date.month = 0; m.date.day = 0;
                            m.format = f;
                            matched = true;
                            i = j;
                            if (pair)
                            {
                                out.push_back(m);
                                m.begin = t.offsets[b2];
                                m.end = t.offsets[e2];
                                m.date.year = year2;
                                i = e2;
                            }
                        }
                    }
                }
            }
            else if (is_letter(c) && !is_letter(prev))
            {
                // Text layout "March 12, 1985".
                std::size_t mend = 0;
                int cs = 0;
                const int variant = match_month_name(t, i, mend, cs);
                if (variant >= 0 && mend < n && t.cps[mend] == ' ')
                {
                    std::size_t j = mend + 1;
                    int day, ld;
                    if (read_digits(t, j, 2, day, ld))
                    {
                        date_format f;
                        f.layout = 4;
                        f.day_style = (ld == 2 && day < 10) ? 1 : 0;
                        if (is_letter(t.at(j)) && is_letter(t.at(j + 1)) && !is_letter(t.at(j + 2)))
                        {
                            const std::string suf = { static_cast<char>(to_lower(t.at(j))),
                                                      static_cast<char>(to_lower(t.at(j + 1))) };
                            if (suf == "st" || suf == "nd" || suf == "rd" || suf == "th") { f.day_style = 3; j += 2; }
                        }
                        if (t.at(j) == ',') { f.comma = 1; ++j; }
                        if (t.at(j) == ' ')
                        {
                            std::size_t y = j + 1;
                            int year, ly;
                            if (read_digits(t, y, 4, year, ly) && ly == 4 && !is_word_char(t.at(y)) &&
                                plausible_year(year))
                            {
                                const int month = month_name_variants()[static_cast<std::size_t>(variant)].month;
                                if (is_valid_civil_date(year, month, day))
                                {
                                    f.month_style = static_cast<std::uint8_t>(variant);
                                    f.month_case = static_cast<std::uint8_t>(cs);
                                    m.begin = t.offsets[i];
                                    m.end = t.offsets[y];
                                    m.date.year = year; m.date.month = month; m.date.day = day;
                                    m.format = f;
                                    matched = true;
                                    i = y;
                                }
                            }
                        }
                    }
                }

                // Month and year: "mars 2022", "March, 2022", "sept. 2019".
                if (!matched && variant >= 0)
                {
                    std::size_t j = mend;
                    std::uint8_t comma = 0;
                    if (t.at(j) == ',')
                    {
                        comma = 1;
                        ++j;
                    }
                    std::size_t y = j;
                    while (y < n && t.cps[y] == ' ' && y - j < 3)
                        ++y;
                    const std::size_t spaces = y - j;
                    int year, ly;
                    if ((spaces == 1 || spaces == 2) && read_digits(t, y, 4, year, ly) && ly == 4 &&
                        plausible_partial_year(year) && year_ends(t, y) && !quantity_after(t, y))
                    {
                        date_format f;
                        f.layout = 5;
                        f.separator = static_cast<std::uint8_t>(spaces - 1);
                        f.day_style = 0;
                        f.month_style = static_cast<std::uint8_t>(variant);
                        f.month_case = static_cast<std::uint8_t>(cs);
                        f.comma = comma;
                        m.begin = t.offsets[i];
                        m.end = t.offsets[y];
                        m.date.year = year;
                        m.date.month = month_name_variants()[static_cast<std::size_t>(variant)].month;
                        m.date.day = 0;
                        m.format = f;
                        matched = true;
                        i = y;
                    }
                }
            }

            if (matched)
                out.push_back(m);
            else
                ++i;
        }
        return out;
    }

    inline std::string format_date(
        const civil_date& d,
        const date_format& f
    )
    {
        using namespace privacy_impl;
        auto two = [](int v) {
            std::string s = std::to_string(v);
            return s.size() < 2 ? "0" + s : s;
        };
        auto day_text = [&](void) -> std::string {
            switch (f.day_style)
            {
                case 1: return two(d.day);
                case 2: return d.day == 1 ? std::string("1er") : std::to_string(d.day);
                case 3:
                {
                    const int r100 = d.day % 100, r10 = d.day % 10;
                    const char* suf = (r100 >= 11 && r100 <= 13)
                        ? "th"
                        : (r10 == 1 ? "st" : (r10 == 2 ? "nd" : (r10 == 3 ? "rd" : "th")));
                    return std::to_string(d.day) + suf;
                }
                default: return std::to_string(d.day);
            }
        };
        const std::string year_text = f.two_digit_year ? two(((d.year % 100) + 100) % 100) : std::to_string(d.year);

        if (f.layout <= 2 || f.layout == 6)
        {
            const char sep = "/-."[f.separator < 3 ? f.separator : 0];
            const std::string mo = f.month_style == 1 ? two(d.month) : std::to_string(d.month);
            const std::string da = f.day_style == 1 ? two(d.day) : std::to_string(d.day);
            if (f.layout == 0) return da + sep + mo + sep + year_text;
            if (f.layout == 1) return mo + sep + da + sep + year_text;
            if (f.layout == 6) return mo + sep + year_text;
            return year_text + sep + mo + sep + da;
        }
        if (f.layout == 7)
            return year_text;

        /* A month name variant only fits its own month; other months use the variant of the
           same family at the same position (full names, abbreviations), found by month. */
        const auto& variants = month_name_variants();
        std::string name;
        if (f.month_style < variants.size())
        {
            const month_name_variant& orig = variants[f.month_style];
            if (orig.month == d.month)
            {
                name = orig.text;
            }
            else
            {
                const std::size_t family_start = f.month_style < 15 ? 0 :
                                                 f.month_style < 25 ? 15 :
                                                 f.month_style < 29 ? 25 :
                                                 f.month_style < 41 ? 29 :
                                                 f.month_style < 53 ? 41 : 53;
                /* Families: 0-11 French full, 15-24/25-28 French abbreviations, 29-40 English
                   full, 41-52 English abbreviations, 53-61 English abbreviations with period. */
                const std::size_t family_end = family_start == 0 ? 12 : family_start == 15 ? 25 :
                                               family_start == 25 ? 29 : family_start == 29 ? 41 :
                                               family_start == 41 ? 53 : variants.size();
                for (std::size_t v = family_start; v < family_end && name.empty(); ++v)
                    if (variants[v].month == d.month) name = variants[v].text;
                if (name.empty())
                {
                    // Months without an abbreviation in this family are written in full.
                    const std::size_t full_start = (family_start >= 29) ? 29 : 0;
                    for (std::size_t v = full_start; v < full_start + 12 && name.empty(); ++v)
                        if (variants[v].month == d.month) name = variants[v].text;
                }
            }
        }
        name = apply_case(name, f.month_case);
        if (f.layout == 3)
            return day_text() + " " + name + " " + year_text;
        if (f.layout == 5)
            return name + (f.comma ? "," : "") + (f.separator == 1 ? "  " : " ") + year_text;
        return name + " " + day_text() + (f.comma ? "," : "") + " " + year_text;
    }

    inline bool parse_date(
        const std::string& text,
        numeric_date_order order,
        civil_date& date,
        date_format& format
    )
    {
        const std::vector<date_match> m = find_dates(text, order);
        if (m.size() == 1 && m[0].begin == 0 && m[0].end == text.size())
        {
            date = m[0].date;
            format = m[0].format;
            return true;
        }
        // A year alone: the words that made it a date are outside the value.
        if (text.size() == 4 && text.find_first_not_of("0123456789") == std::string::npos &&
            privacy_impl::plausible_partial_year(std::stoi(text)))
        {
            date = civil_date();
            date.year = std::stoi(text);
            format = date_format();
            format.layout = 7;
            format.separator = 0;
            format.day_style = 0;
            format.month_style = 0;
            return true;
        }
        return false;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DATE_FORMATS_H_
