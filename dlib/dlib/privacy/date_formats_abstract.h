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
    !*/

// ----------------------------------------------------------------------------------------

    struct civil_date
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A date of the proleptic Gregorian calendar.
        !*/
        int year = 0;
        int month = 0;      // 1 to 12
        int day = 0;        // 1 to 31
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
                                 3 text "12 mars 1985", 4 text "March 12, 1985"
                separator      - numeric layouts: the index of the separator in "/-."
                day_style      - 0 plain, 1 zero padded, 2 French ordinal ("1er"),
                                 3 English ordinal ("1st", "12th")
                month_style    - numeric layouts: 0 plain or 1 padded; text layouts: the
                                 index of the month name variant (full name, abbreviation,
                                 with or without accents and final period)
                month_case     - 0 lower, 1 title, 2 upper
                two_digit_year - 1 when the year has two digits
                comma          - 1 when a comma follows the day ("March 12, 1985")
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
            - Only complete dates (day, month and year) are reported, with a year from
              1800 to 2199, and only when the day exists in that month. A two digit year
              below 50 reads 20YY, the others 19YY.
            - A no-break space reads as a space and invisible marks are skipped, as in
              pii_detector.
    !*/

    std::string format_date(
        const civil_date& d,
        const date_format& f
    );
    /*!
        requires
            - is_valid_civil_date(d.year, d.month, d.day)
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
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DATE_FORMATS_ABSTRACT_H_

