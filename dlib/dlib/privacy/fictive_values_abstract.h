// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PRIVACY_FICTIVE_VALUES_ABSTRACT_H_
#ifdef DLIB_PRIVACY_FICTIVE_VALUES_ABSTRACT_H_

#include <string>
#include <vector>
#include "pii_types_abstract.h"
#include "date_formats_abstract.h"
#include "pseudonymizer_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct fictive_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings of a corpus written with fictive values.

                languages       - the languages of the corpus among "fr", "en", "de", "es"
                                  and "it". The first one is the language of the names found
                                  in no lexicon and gives the country of the places, the
                                  telephone numbers and the addresses whose country the value
                                  does not tell. Empty or unknown entries read as "fr".
                date_shift_days - the bound of the shift of the dates, in days, clamped to
                                  [7, 3650]
                keep_city       - true: a postal code and its city stay as they are when the
                                  city is one of the cities of 50,000 inhabitants or more of
                                  the lexicon; any other place, and every place when false,
                                  takes a city of the same country and of the same size class
                                  (fewer than 50,000 inhabitants when its size is unknown)
                date_order      - how numeric dates such as 03/04/2026 are read
        !*/
        std::vector<std::string> languages;
        long date_shift_days = 180;
        bool keep_city = true;
        numeric_date_order date_order = numeric_date_order::day_month_year;
    };

// ----------------------------------------------------------------------------------------

    class fictive_substituter
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Plausible fictive values for the corpora of training: each value found takes a
                value of the same type, of the same language and of the same form, drawn by
                HMAC-SHA-256 under the key of the corpus from the value itself, so that a value
                takes the same fictive value in every text of the corpus and no table is kept.
                Nothing links a fictive value to its original without the key, and nothing in
                this object or in what it writes gives the original back: this writing is not
                reversible.

                The words of a value are drawn one by one, from their folded form, so that the
                variants of an entity stay linked whatever their case: "Jean Dupont", "M. Dupont",
                "J. Dupont", "DUPONT Jean" and jean.dupont@acme.fr become "Lucien Morel",
                "M. Morel", "L. Morel", "MOREL Lucien" and lucien.morel@garnier.example.com when
                Jean draws Lucien, Dupont draws Morel and acme draws Garnier.

                VALUES BY TYPE
                    Persons: a given name takes a given name of its language and of its sex
                        (the lexicon of the name, else the civility that precedes it in the
                        text, else a draw), whose initial is the fictive initial of its own, so
                        that "J." and "Jean" keep one initial; a family name takes a family name
                        of its language made of a stem and an ending; particles, initials and
                        civilities keep their places, and the case of each word is kept.
                    Organisations: the legal form, the word of the trade or of the kind
                        (Banque, Hôpital, Université, Industries), the partners and the small
                        words stay; an acronym takes an acronym of its length, a given name a
                        given name, a city the city of a place, and the other words family names.
                    Places and addresses: the number of the street takes another number of as
                        many digits, the words of its name the words of street names of its
                        language that agree with the article before them, the names of persons
                        in it fictive names; the type of the street and the small words stay. A
                        postal code and its city follow keep_city, a replaced city taking a city
                        of the lexicon of its country and of its size class with its postal code.
                    E-mail addresses: the words of the local part are drawn as the names they
                        are, the domain becomes a subdomain of example.com, example.org or
                        example.net (RFC 2606), whose label is the family name drawn for the
                        label of the original domain. URLs take the same host.
                    Telephone numbers: the ranges reserved for fiction where the country
                        publishes them (France: 01 99 00, 02 61 91, 03 53 01, 04 65 71, 05 36 49,
                        06 39 98; North America: 555-0100 to 555-0199; the drama numbers of
                        Ofcom in Great Britain and of the Bundesnetzagentur in Germany); in the
                        other countries, digits drawn after the country code and the first two
                        national digits.
                    IBAN: France, the bank code 990xx of the test establishments of the Banque
                        de France, the key of the RIB and the check digits recomputed; other
                        countries, the digits and the letters of the national part drawn and
                        the check digits recomputed. Payment cards: the prefix of the test
                        numbers that the processors publish for their network (411111, 555555,
                        378282, 601111) and a valid Luhn digit; the networks reserve no such
                        prefix, so the number drawn may be one that a bank issued. NIR: the sex
                        kept, a month from 13 to 19, which the register does not give, and a
                        valid key. SIREN and SIRET: a first digit of the same class (1 or 2 for
                        a public body, 3 to 9 for the others), the other digits drawn, valid
                        Luhn digits, a SIRET keeping the fictive SIREN of its SIREN; the
                        register reserves no block for tests, so a fictive SIREN may be the
                        number of a registered unit.
                    IP addresses: the ranges of documentation 192.0.2.0/24, 198.51.100.0/24,
                        203.0.113.0/24 (RFC 5737) and 2001:db8::/32 (RFC 3849), the addresses of
                        one /24 staying in one range and an IPv6 prefix keeping its fictive prefix.
                        MAC addresses: a locally administered address.
                    Dates: a shift of a whole number of weeks, at least one and at most
                        date_shift_days, the same for every date of a document, into the past
                        or the future: durations, order and weekdays are kept. A month and its
                        year move by the nearest whole number of months, a year alone by the
                        nearest whole number of years, possibly none.
                    Secrets: a random value of the same length whose characters keep the class
                        of the character in their place (lower case, upper case, digit, other)
                        and avoid every character of the original; when a class holds no other
                        character, the value still shares no run of four characters with it. The
                        armour lines of a PEM block stay.
                    Identifiers: account names take a family name, or an initial and a family
                        name, in lower case; other identifiers keep their shape (letters, digits,
                        case, separators, a leading country code, the Luhn validity of a number
                        of eight digits or more, the form of a UUID). License plates: the
                        layout kept, the letters of a SIV plate starting with X, Y or Z, a
                        series the register reaches in decades. GPS coordinates: the degrees
                        kept, the decimals drawn.
                    Civilities, sexes and nationalities stay as they are: the fictive given
                        names follow them.

                A drawn value never equals its original: the next draw is taken. A date, which
                moves by the shift of its document, is the one value that may stay as it was.

            THREAD SAFETY
                The const methods may be called from several threads at once.
        !*/
    public:
        fictive_substituter(
            const pseudonymization_key& corpus_key,
            const fictive_options& options = fictive_options()
        );
        /*!
            ensures
                - #get_options() == options, date_shift_days clamped
                - The keys of the draws are derived with HKDF-SHA-256, and the corpus key
                  itself is not kept.
            throws
                - crypto_error if corpus_key.is_set() == false.
        !*/

        const fictive_options& get_options() const;

        std::string fictive(
            pii_type type,
            const std::string& value,
            const std::string& document = ""
        ) const;
        /*!
            requires
                - type != pii_type::none
            ensures
                - Returns the fictive value of value, read as a value of type alone: the sex of
                  a given name comes from its lexicon or from a draw, and no other value of a
                  batch is checked (fictive_batch does both). document names the document whose
                  dates move together; the same (type, value, document) always gives the same
                  result under a given key and options.
        !*/

        long date_shift_days(
            const std::string& document
        ) const;
        /*!
            ensures
                - Returns the shift of the dates of document, in days: a nonzero multiple of 7
                  whose magnitude is at most get_options().date_shift_days. It depends on the
                  key and document only.
        !*/

        static std::string lexicon_version(
        );
        /*!
            ensures
                - Returns the version of the lexicons and of the reserved ranges compiled in,
                  "1.1.0-" followed by 12 hexadecimal digits of the SHA-256 of their content,
                  which the reports give with the versions of the models.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class fictive_batch
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The texts of one request written with fictive values, in two passes. The first
                writes each text with document_anonymizer, whose value writer is record(): each
                value found takes a placeholder. finish() then writes the fictive value of each
                placeholder, knowing every original value of the batch:
                    - no fictive value equals an original value of the same type in the batch:
                      the next draw is taken, for every occurrence of the batch (a value whose
                      first draw is the original of another text may so take another fictive
                      value in another batch, which a large lexicon makes rare);
                    - a civility that precedes a given name in one text of the batch gives its
                      sex to the occurrences of that name in the other texts, when the lexicon
                      tells none.
                The batch keeps the values it records in memory only, until finish() or its
                destruction, which erase them.
        !*/
    public:
        explicit fictive_batch(
            const fictive_substituter& s
        );
        /*!
            requires
                - s outlives this object
        !*/

        std::string record(
            pii_type type,
            const std::string& value
        );
        /*!
            ensures
                - Returns value itself for a civility, a sex or a nationality, which stay as they
                  are, and otherwise a placeholder of letters and digits that holds no delimiter,
                  quote or space, so that the CSV and text writers keep it whole.
        !*/

        std::vector<std::string> finish(
            const std::vector<std::string>& written,
            const std::vector<std::string>& documents
        );
        /*!
            requires
                - written holds the texts written with the placeholders of this batch
                - documents.size() == written.size(), or documents is empty
            ensures
                - Returns written with each placeholder replaced by its fictive value; the dates
                  of written[i] move by date_shift_days(documents[i]) (an empty document when
                  documents is empty).
                - The values recorded are erased: record() starts a new batch.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_FICTIVE_VALUES_ABSTRACT_H_
