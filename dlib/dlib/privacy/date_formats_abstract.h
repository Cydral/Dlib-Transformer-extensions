// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PRIVACY_DATE_FORMATS_ABSTRACT_H_
#ifdef DLIB_PRIVACY_DATE_FORMATS_ABSTRACT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dlib
{

    /*!
        Calendar dates as written in French and English documents: recognition with the
        layout of the source text, and rendering of any date in that same layout.

        The pair is what makes date pseudonymization reversible. A date such as "1er mars
        1985" or "03/12/85" is reduced to a civil date and a date_format that records what
        is needed to write it again: field order, separator, zero padding, ordinal suffix,
        month name spelling and case, two digit year. The pseudonymizer shifts the civil
        date and keeps the format; restoration shifts back and renders with the recorded
        format.

        Dates written without their day are read too, since a month or a year left in clear
        would tell how far the other dates moved: a month and its year ("mars 2022", "March,
        2022", "03/2022") and a year that dates something ("en 2019", "since 2019", "fin
        2022", the years of a CV line "2015 - 2019 :").
    !*/

// ----------------------------------------------------------------------------------------

    struct civil_date
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A date of the proleptic Gregorian calendar.
        !*/
        int year = 0;
        int month = 0;      // 1 to 12, 0 when the date is a year alone
        int day = 0;        // 1 to 31, 0 when the date has no day
    };

    long days_from_civil(
        int y,
        int m,
        int d
    );
    /*!
        requires
            - 1 <= m <= 12
        ensures
            - Returns the number of days from 1970-01-01 to y-m-d in the proleptic
              Gregorian calendar (H. Hinnant's algorithm), negative before the epoch.
    !*/

    civil_date civil_from_days(
        long z
    );
    /*!
        ensures
            - Returns the date z days after 1970-01-01, the inverse of days_from_civil():
              days_from_civil(r.year, r.month, r.day) == z for the returned r.
    !*/

    int days_in_month(
        int y,
        int m
    );
    /*!
        ensures
            - Returns the number of days of month m of year y, leap years included, or 0
              when m is not in [1, 12].
    !*/

    bool is_valid_civil_date(
        int y,
        int m,
        int d
    );
    /*!
        ensures
            - Returns true if y-m-d exists in the Gregorian calendar.
    !*/

// ----------------------------------------------------------------------------------------

    enum class numeric_date_order
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                How an all-numeric date whose first two fields are both <= 12 is read.
                Unambiguous dates (13/03/1985, 03/13/1985) are read correctly under
                either setting.
        !*/
        day_month_year,
        month_day_year
    };

    struct date_format
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The written layout of a date.

                layout         - 0 numeric D M Y, 1 numeric M D Y, 2 numeric Y M D,
                                 3 text "12 mars 1985", 4 text "March 12, 1985",
                                 5 month and year "mars 1985", 6 numeric M Y "03/1985",
                                 7 year alone "1985"
                separator      - numeric layouts: the index of the separator in "/-.";
                                 layout 5: 0 for one space before the year, 1 for two
                day_style      - 0 plain, 1 zero padded, 2 French ordinal ("1er"),
                                 3 English ordinal ("1st", "12th")
                month_style    - numeric layouts: 0 plain or 1 padded; text layouts: the
                                 index of the month name variant (full name, abbreviation,
                                 with or without accents and final period)
                month_case     - 0 lower, 1 title, 2 upper
                two_digit_year - 1 when the year has two digits
                comma          - 1 when a comma follows the day ("March 12, 1985") or the
                                 month ("March, 1985")
        !*/
        std::uint8_t layout = 2;
        std::uint8_t separator = 1;
        std::uint8_t day_style = 1;
        std::uint8_t month_style = 1;
        std::uint8_t month_case = 0;
        std::uint8_t two_digit_year = 0;
        std::uint8_t comma = 0;

        std::uint32_t pack() const;
        /*!
            ensures
                - Returns the fields packed into 17 bits, for a tag.
        !*/

        static date_format unpack(std::uint32_t v);
        /*!
            ensures
                - Returns the format f such that f.pack() == v for every v returned by
                  pack().
        !*/

        bool is_iso() const;
        /*!
            ensures
                - Returns true for the layout YYYY-MM-DD.
        !*/

        bool has_day() const;
        /*!
            ensures
                - Returns true for the layouts of a complete date (0 to 4).
        !*/

        bool has_month() const;
        /*!
            ensures
                - Returns true for every layout but the year alone (7).
        !*/
    };

    struct date_match
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A date found in a text: its byte span [begin, end), its value and the
                layout it is written in.
        !*/
        std::size_t begin = 0;
        std::size_t end = 0;
        civil_date date;
        date_format format;
    };

// ----------------------------------------------------------------------------------------

    std::vector<date_match> find_dates(
        const std::string& text,
        numeric_date_order order = numeric_date_order::day_month_year
    );
    /*!
        ensures
            - Returns the non-overlapping dates of the UTF-8 text, left to right, each with
              its byte span, value and format: numeric dates with '/', '-' or '.' (years
              of two or four digits), ISO dates (a following time "T10:30" allowed), and
              text dates in French and English with full or abbreviated month names.
            - A complete date has a year from 1800 to 2199 and a day that exists in its
              month. A two digit year below 50 reads 20YY, the others 19YY.
            - Dates without a day have a year of four digits from 1900 to 2099 that no
              letter, digit or unit follows ("en 2000 exemplaires" is no date):
                - a month name and its year, with an optional comma after the month and
                  one or two spaces before the year (layout 5);
                - a month of two digits, '/' or '.', and the year (layout 6), unless a
                  number sign or a reference word comes before ("n° 03/2022");
                - a year alone (layout 7) after a word of time: en, depuis, dès, avant,
                  après, fin, début, courant, vers, année, l'an, mi-, printemps, été,
                  automne, hiver, session, promotion, in, since, until, till, before, after,
                  around, circa, early, late, mid-, spring, summer, autumn, fall, winter,
                  year, born, class of; after the label of a birth year ("Année de
                  naissance : 1985", "Year of birth: 1985"); with the second year of a range
                  after de, entre, from or between ("de 2015 à 2019", "between 2015 and
                  2019"), a dash or a linking word (à, au, et, to, and, until, through); or
                  at the head of a line before a colon, alone or as a range ("2015 : BTS",
                  "2015 - 2019 : technicien"). Each year of a range is reported apart.
              Such a date has 0 for the parts it lacks.
            - A no-break space reads as a space and invisible marks are skipped, as in
              pii_detector.
    !*/

    std::string format_date(
        const civil_date& d,
        const date_format& f
    );
    /*!
        requires
            - is_valid_civil_date(d.year, d.month, d.day) for a layout with a day, a month
              of 1 to 12 for a layout with a month
        ensures
            - Writes d in the layout f. A month name variant is kept for its own month;
              another month is written with the variant of the same family (French full
              names, French abbreviations, English full names, English abbreviations),
              and in full when that family has no abbreviation for it.
            - For a date returned by find_dates(), the result is the matched text unless
              the text holds what a date_format does not record: a no-break space or an
              invisible mark, a zero padded "01er", an ordinal suffix in capitals ("1ST").
              Callers that need the source text back compare the two, as the
              pseudonymizer does before it shifts a date.
    !*/

    bool parse_date(
        const std::string& text,
        numeric_date_order order,
        civil_date& date,
        date_format& format
    );
    /*!
        ensures
            - Returns true if the whole of text is one date as find_dates() reads it, and
              sets #date and #format accordingly.
            - Four digits alone from 1900 to 2099 read as a year alone (layout 7): the words
              that made them a date are outside the text.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DATE_FORMATS_ABSTRACT_H_

