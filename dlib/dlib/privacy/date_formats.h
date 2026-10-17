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
                        if (t.at(j) == 'e' && t.at(j + 1) == 'r' && !is_letter(t.at(j + 2)))
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

        if (f.layout <= 2)
        {
            const char sep = "/-."[f.separator < 3 ? f.separator : 0];
            const std::string mo = f.month_style == 1 ? two(d.month) : std::to_string(d.month);
            const std::string da = f.day_style == 1 ? two(d.day) : std::to_string(d.day);
            if (f.layout == 0) return da + sep + mo + sep + year_text;
            if (f.layout == 1) return mo + sep + da + sep + year_text;
            return year_text + sep + mo + sep + da;
        }

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
        if (m.size() != 1 || m[0].begin != 0 || m[0].end != text.size())
            return false;
        date = m[0].date;
        format = m[0].format;
        return true;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DATE_FORMATS_H_
