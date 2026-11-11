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
        std::uint8_t connector = 0;

        /* The fields in 21 bits. The bits of the formats that French and English dates
           needed come first, in the 17 bits they always took, so that a tag written before
           the other languages decodes the same; the connector and the two high bits of the
           month name variant follow. */
        std::uint32_t pack() const
        {
            return static_cast<std::uint32_t>(layout) | (static_cast<std::uint32_t>(separator) << 3) |
                   (static_cast<std::uint32_t>(day_style) << 5) |
                   (static_cast<std::uint32_t>(month_style & 63) << 7) |
                   (static_cast<std::uint32_t>(month_case) << 13) | (static_cast<std::uint32_t>(two_digit_year) << 15) |
                   (static_cast<std::uint32_t>(comma) << 16) | (static_cast<std::uint32_t>(connector & 3) << 17) |
                   (static_cast<std::uint32_t>(month_style >> 6) << 19);
        }

        static date_format unpack(std::uint32_t v)
        {
            date_format f;
            f.layout = static_cast<std::uint8_t>(v & 7);
            f.separator = static_cast<std::uint8_t>((v >> 3) & 3);
            f.day_style = static_cast<std::uint8_t>((v >> 5) & 3);
            f.month_style = static_cast<std::uint8_t>(((v >> 7) & 63) | (((v >> 19) & 3) << 6));
            f.month_case = static_cast<std::uint8_t>((v >> 13) & 3);
            f.two_digit_year = static_cast<std::uint8_t>((v >> 15) & 1);
            f.comma = static_cast<std::uint8_t>((v >> 16) & 1);
            f.connector = static_cast<std::uint8_t>((v >> 17) & 3);
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
                {9, "sep."}, {12, "dec."},
                // 62: Italian, full names then abbreviations.
                {1, "gennaio"}, {2, "febbraio"}, {3, "marzo"}, {4, "aprile"}, {5, "maggio"}, {6, "giugno"},
                {7, "luglio"}, {8, "agosto"}, {9, "settembre"}, {10, "ottobre"}, {11, "novembre"}, {12, "dicembre"},
                {1, "gen."}, {2, "feb."}, {3, "mar."}, {4, "apr."}, {5, "mag."}, {6, "giu."}, {7, "lug."},
                {8, "ago."}, {9, "set."}, {10, "ott."}, {11, "nov."}, {12, "dic."},
                // 86: Spanish.
                {1, "enero"}, {2, "febrero"}, {3, "marzo"}, {4, "abril"}, {5, "mayo"}, {6, "junio"}, {7, "julio"},
                {8, "agosto"}, {9, "septiembre"}, {10, "octubre"}, {11, "noviembre"}, {12, "diciembre"},
                {1, "ene."}, {2, "feb."}, {3, "mar."}, {4, "abr."}, {5, "may."}, {6, "jun."}, {7, "jul."},
                {8, "ago."}, {9, "sep."}, {10, "oct."}, {11, "nov."}, {12, "dic."},
                // 110: German, full names, "März" without its umlaut, then abbreviations.
                {1, "januar"}, {2, "februar"}, {3, "m\xC3\xA4rz"}, {4, "april"}, {5, "mai"}, {6, "juni"},
                {7, "juli"}, {8, "august"}, {9, "september"}, {10, "oktober"}, {11, "november"}, {12, "dezember"},
                {3, "marz"}, {3, "maerz"},
                {1, "jan."}, {2, "feb."}, {3, "m\xC3\xA4r."}, {4, "apr."}, {6, "jun."}, {7, "jul."}, {8, "aug."},
                {9, "sep."}, {10, "okt."}, {11, "nov."}, {12, "dez."}
            };
            return v;
        }

        /* The languages of the month name variants, by the first index of each: French (0),
           English (29), Italian (62), Spanish (86) and German (110). */
        enum class month_language { french = 0, english = 1, italian = 2, spanish = 3, german = 4 };

        inline month_language language_of_variant(std::size_t v)
        {
            if (v < 29) return month_language::french;
            if (v < 62) return month_language::english;
            if (v < 86) return month_language::italian;
            if (v < 110) return month_language::spanish;
            return month_language::german;
        }

        /* The first and last index of the family of variant v, names written alike, and the
           family of full names of its language, used when a month has no abbreviation. */
        inline void variant_family(std::size_t v, std::size_t& first, std::size_t& last, std::size_t& full)
        {
            static const std::size_t starts[] = { 0, 15, 25, 29, 41, 53, 62, 74, 86, 98, 110, 124, 135 };
            std::size_t k = 0;
            while (k + 2 < sizeof(starts) / sizeof(starts[0]) && v >= starts[k + 1]) ++k;
            first = starts[k];
            last = starts[k + 1];
            // The names of French without accents (12 to 14) take the accented ones of 0 to 11.
            if (first == 0) last = 12;
            if (first == 110) last = 122;
            const month_language lang = language_of_variant(v);
            full = lang == month_language::french ? 0 : lang == month_language::english ? 29 :
                   lang == month_language::italian ? 62 : lang == month_language::spanish ? 86 : 110;
        }

        /* The variant of language lang written like variant v, or v when that language
           writes the name otherwise: "novembre" French or Italian, "August" English or German. */
        inline int variant_in_language(int v, month_language lang)
        {
            const auto& variants = month_name_variants();
            if (v < 0 || language_of_variant(static_cast<std::size_t>(v)) == lang) return v;
            const std::string text = variants[static_cast<std::size_t>(v)].text;
            for (std::size_t k = 0; k < variants.size(); ++k)
                if (language_of_variant(k) == lang && text == variants[k].text) return static_cast<int>(k);
            return v;
        }

        /* The language that the word written just before code point i gives to a month name
           that several languages share, or -1: "il 3 novembre" is Italian, "le 3 novembre"
           French, "am 3. August" German. */
        inline int language_before(const utf8_text& t, std::size_t i)
        {
            std::size_t k = i;
            while (k > 0 && t.cps[k - 1] == ' ') --k;
            if (k == i && !(k > 0 && is_apostrophe(t.cps[k - 1]))) return -1;
            std::size_t e = k;
            if (e > 0 && is_apostrophe(t.cps[e - 1])) --e;
            std::size_t b = e;
            while (b > 0 && is_letter(t.cps[b - 1])) --b;
            std::string w;
            for (std::size_t q = b; q < e; ++q)
                utf8_append(w, to_lower(t.cps[q]));
            w = fold_key(w);
            if (e < k)
            {
                // "dall'8 marzo", "nell'aprile": an elided Italian preposition.
                if (w == "dall" || w == "dell" || w == "nell" || w == "all" || w == "l" || w == "sull")
                    return static_cast<int>(month_language::italian);
                return -1;
            }
            static const char* const italian[] = { "il", "del", "dal", "al", "nel", "entro", "fino", "giorno", "data",
                                                   "li", "addi", "sabato", "domenica", "lunedi", "martedi",
                                                   "mercoledi", "giovedi", "venerdi" };
            static const char* const spanish[] = { "el", "desde", "hasta", "dia", "fecha", "a", "lunes", "martes",
                                                   "miercoles", "jueves", "viernes", "sabado", "domingo" };
            static const char* const french[] = { "le", "du", "au", "des", "depuis", "lundi", "mardi", "mercredi",
                                                  "jeudi", "vendredi", "samedi", "dimanche" };
            static const char* const german[] = { "am", "vom", "zum", "bis", "seit", "ab", "den", "im", "montag",
                                                  "dienstag", "mittwoch", "donnerstag", "freitag", "samstag",
                                                  "sonntag", "stand" };
            static const char* const english[] = { "on", "of", "by", "from", "since", "monday", "tuesday",
                                                   "wednesday", "thursday", "friday", "saturday", "sunday" };
            for (const char* x : italian) if (w == x) return static_cast<int>(month_language::italian);
            for (const char* x : spanish) if (w == x) return static_cast<int>(month_language::spanish);
            for (const char* x : french) if (w == x) return static_cast<int>(month_language::french);
            for (const char* x : german) if (w == x) return static_cast<int>(month_language::german);
            for (const char* x : english) if (w == x) return static_cast<int>(month_language::english);
            return -1;
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
                "winter", "year", "born",
                // Italian, Spanish and German.
                "nel", "dal", "fino", "anno", "entro", "estate", "inverno", "primavera", "autunno", "inizio",
                "nato", "nata", "desde", "hasta", "ano", "verano", "invierno", "otono", "principios", "finales",
                "mediados", "nacido", "nacida", "seit", "bis", "jahr", "jahre", "sommer", "fruhjahr", "fruhling",
                "herbst", "anfang", "ende", "mitte", "geboren"
            };
            static const std::vector<std::string> range = { "de", "entre", "from", "between", "da", "tra", "fra",
                                                             "von", "zwischen" };
            static const std::vector<std::string> labels = { "naissance", "birth", "dob", "nascita", "nacimiento",
                                                             "geburt", "geburtsjahr", "jahrgang" };
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
                "entreprises", "companies", "logements", "emplois", "jobs", "postes", "projets", "projects",
                "anni", "anno", "mesi", "giorni", "ore", "persone", "dipendenti", "abitanti", "pezzi", "copie",
                "pagine", "utenti", "clienti", "studenti", "posti", "anos", "meses", "dias", "horas", "personas",
                "empleados", "habitantes", "unidades", "copias", "paginas", "usuarios", "alumnos", "plazas",
                "jahren", "monate", "monaten", "tage", "tagen", "stunden", "personen", "mitarbeiter",
                "mitarbeitende", "einwohner", "stuck", "seiten", "nutzer", "kunden", "exemplare", "schuler",
                "platze", "betten"
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
                "a", "au", "et", "to", "and", "until", "till", "through", "thru", "al", "e", "y", "und", "bis",
                "hasta", "fino"
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
                   w == "number" || w == "version" || w == "n" || w == "rif" || w == "prot" || w == "protocollo" ||
                   w == "pratica" || w == "fattura" || w == "factura" || w == "expediente" || w == "rechnung" ||
                   w == "az" || w == "aktenzeichen" || w == "nummer";
        }

        /* The words of the end of validity of a card or a document before code point i: "expiry
           date, 07/29", "exp. 07/29", "valid thru 07/29", "scadenza 07/29", "caducidad: 07/29",
           "gültig bis 07/29", "date d'expiration : 07/29". */
        inline bool expiry_before(const utf8_text& t, std::size_t i)
        {
            static const std::vector<std::string> alone = {
                "expiry", "expiration", "expires", "expire", "exp", "expira", "expiracion", "scadenza", "scad",
                "caducidad", "caduca", "vencimiento", "vence", "ablaufdatum", "verfallsdatum", "ablauf", "validite" };
            static const std::vector<std::string> dates = { "date", "datum", "data", "fecha" };
            static const std::vector<std::string> until = { "thru", "through", "until", "bis", "hasta", "fino" };
            static const std::vector<std::string> valid = { "valid", "good", "gultig", "valida", "valido", "valable" };
            auto in = [](const std::vector<std::string>& words, const std::string& w) {
                for (const std::string& x : words)
                    if (x == w) return true;
                return false;
            };
            std::size_t k = i;
            while (k > 0 && i - k < 4 && (t.cps[k - 1] == ' ' || t.cps[k - 1] == ':' || t.cps[k - 1] == '.' ||
                                          t.cps[k - 1] == ',' || t.cps[k - 1] == 0xA0))
                --k;
            std::size_t b;
            const std::string w = fold_key(letters_before(t, k, b));
            if (in(alone, w)) return true;
            if (!in(dates, w) && !in(until, w)) return false;
            std::size_t q = b;
            while (q > 0 && b - q < 3 && (t.cps[q - 1] == ' ' || t.cps[q - 1] == 0xA0)) --q;
            std::size_t b2;
            const std::string w2 = fold_key(letters_before(t, q, b2));
            return in(dates, w) ? in(alone, w2) : in(valid, w2);
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

                /* Text layout "12 mars 1985", "1er mars 1985", "12th March 1985", and those of
                   German, Spanish and Italian: "12. März 1985", "12 de marzo de 1985", "1° marzo
                   1985". A month name that several languages write alike takes the language
                   its connector or the word before the date gives it. */
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
                        else if (t.at(j) == '.' && t.at(j + 1) == ' ')
                        {
                            f.connector = 1;
                            ++j;
                        }
                        else if (t.at(j) == 0xB0 && day == 1)
                        {
                            f.connector = 3;
                            ++j;
                        }
                        if (j < n && is_date_space(t.cps[j]) && t.cps[j] == ' ')
                        {
                            std::size_t k = j + 1, mend = 0;
                            const bool de = f.connector == 0 && to_lower(t.at(k)) == 'd' &&
                                            to_lower(t.at(k + 1)) == 'e' && t.at(k + 2) == ' ';
                            if (de)
                            {
                                f.connector = 2;
                                k += 3;
                            }
                            int cs = 0;
                            int variant = match_month_name(t, k, mend, cs);
                            if (variant >= 0)
                            {
                                static const month_language by_connector[] = {
                                    month_language::french, month_language::german, month_language::spanish,
                                    month_language::italian };
                                if (f.connector != 0)
                                {
                                    const month_language lang = by_connector[f.connector];
                                    variant = variant_in_language(variant, lang);
                                    if (language_of_variant(static_cast<std::size_t>(variant)) != lang) variant = -1;
                                }
                                else
                                {
                                    const int lang = language_before(t, i);
                                    if (lang >= 0)
                                        variant = variant_in_language(variant, static_cast<month_language>(lang));
                                }
                            }
                            if (variant >= 0 && mend < n && t.cps[mend] == ' ')
                            {
                                std::size_t y = mend + 1;
                                if (f.connector == 2)
                                {
                                    if (to_lower(t.at(y)) == 'd' && to_lower(t.at(y + 1)) == 'e' && t.at(y + 2) == ' ')
                                        y += 3;
                                    else
                                        y = n;
                                }
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
                        const bool read = read_digits(t, y, 4, year, ly);
                        // The end of validity of a card in two figures: "expiry date, 07/29".
                        const bool expiry = read && ly == 2 && sep == '/' && year_ends(t, y) && expiry_before(t, i);
                        if (expiry) year += 2000;
                        if (read && (ly == 4 || expiry) && plausible_partial_year(year) && year_ends(t, y) &&
                            !reference_before(t, i))
                        {
                            date_format f;
                            f.layout = 6;
                            f.separator = static_cast<std::uint8_t>(sep == '/' ? 0 : 2);
                            f.day_style = 0;
                            f.month_style = 1;
                            f.two_digit_year = expiry ? 1 : 0;
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

                // Month and year: "mars 2022", "March, 2022", "sept. 2019", "marzo de 2022".
                if (!matched && variant >= 0)
                {
                    std::size_t j = mend;
                    std::uint8_t comma = 0, connector = 0;
                    int month_variant = variant;
                    if (t.at(j) == ' ' && to_lower(t.at(j + 1)) == 'd' && to_lower(t.at(j + 2)) == 'e' &&
                        t.at(j + 3) == ' ' && is_ascii_digit(t.at(j + 4)))
                    {
                        const int spanish = variant_in_language(variant, month_language::spanish);
                        if (language_of_variant(static_cast<std::size_t>(spanish)) == month_language::spanish)
                        {
                            month_variant = spanish;
                            connector = 2;
                            j += 3;
                        }
                    }
                    else
                    {
                        const int lang = language_before(t, i);
                        if (lang >= 0) month_variant = variant_in_language(variant, static_cast<month_language>(lang));
                    }
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
                        f.month_style = static_cast<std::uint8_t>(month_variant);
                        f.month_case = static_cast<std::uint8_t>(cs);
                        f.comma = comma;
                        f.connector = connector;
                        m.begin = t.offsets[i];
                        m.end = t.offsets[y];
                        m.date.year = year;
                        m.date.month = month_name_variants()[static_cast<std::size_t>(month_variant)].month;
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
                /* Families: 0-11 French full, 15-24/25-28 French abbreviations, 29-40 English
                   full, 41-52 and 53-61 English abbreviations, 62-73 and 74-85 Italian, 86-97 and
                   98-109 Spanish, 110-121 and 124-134 German. */
                std::size_t first, last, full;
                variant_family(f.month_style, first, last, full);
                for (std::size_t v = first; v < last && name.empty(); ++v)
                    if (variants[v].month == d.month) name = variants[v].text;
                // Months without an abbreviation in this family are written in full.
                for (std::size_t v = full; v < full + 12 && name.empty(); ++v)
                    if (variants[v].month == d.month) name = variants[v].text;
            }
        }
        name = apply_case(name, f.month_case);
        if (f.layout == 3)
        {
            // "12. März 1985", "12 de marzo de 1985", "1° marzo 1985".
            if (f.connector == 1) return day_text() + ". " + name + " " + year_text;
            if (f.connector == 2) return day_text() + " de " + name + " de " + year_text;
            if (f.connector == 3) return (d.day == 1 ? std::string("1\xC2\xB0") : day_text()) + " " + name + " " +
                                         year_text;
            return day_text() + " " + name + " " + year_text;
        }
        if (f.layout == 5)
        {
            if (f.connector == 2) return name + " de " + year_text;
            return name + (f.comma ? "," : "") + (f.separator == 1 ? "  " : " ") + year_text;
        }
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
        // The end of validity of a card, "07/29": the words that made it a date are outside the value.
        if (text.size() == 5 && text[2] == '/' && text.find_first_not_of("0123456789", 3) == std::string::npos &&
            text.find_first_not_of("0123456789") == 2)
        {
            const int month = std::stoi(text.substr(0, 2));
            if (month < 1 || month > 12) return false;
            date = civil_date();
            date.year = 2000 + std::stoi(text.substr(3));
            date.month = month;
            format = date_format();
            format.layout = 6;
            format.separator = 0;
            format.day_style = 0;
            format.month_style = 1;
            format.two_digit_year = 1;
            return true;
        }
        return false;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DATE_FORMATS_H_
