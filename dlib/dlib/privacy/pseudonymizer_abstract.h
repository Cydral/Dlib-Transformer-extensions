// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PSEUDONYMIZER_ABSTRACT_H_
#ifdef DLIB_PSEUDONYMIZER_ABSTRACT_H_

#include <string>
#include <vector>
#include "pii_types_abstract.h"
#include "date_formats_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pseudonymization_key
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The 256 bit master secret from which every pseudonymization key is
                derived. Holding it is what allows protected documents to be restored;
                it takes the place of the correspondence table other tools write next to
                the protected document.

                The key file is a text file: a comment line and "v1:" followed by 64
                hexadecimal digits. On POSIX systems it is created with mode 0600.

                The key bytes are erased from memory when the object is destroyed, and
                the temporary copies made while loading and saving are erased too.
        !*/
    public:
        pseudonymization_key();
        /*!
            ensures
                - #is_set() == false
        !*/

        static pseudonymization_key generate();
        /*!
            ensures
                - Returns a key drawn from the CSPRNG of the operating system.
                - The returned key is set.
            throws
                - crypto_error if the operating system provides no random bytes.
        !*/

        static pseudonymization_key from_bytes(const std::string& bytes);
        /*!
            ensures
                - Returns the key made of these 32 bytes.
            throws
                - crypto_error if bytes.size() != 32.
        !*/

        static pseudonymization_key from_passphrase(
            const std::string& passphrase,
            unsigned long iterations = 600000,
            const std::string& salt = "dlib.privacy.passphrase.v1"
        );
        /*!
            ensures
                - Derives the key with PBKDF2-HMAC-SHA-256. With the default salt, a
                  fixed and published application salt, the same passphrase always
                  yields the same key and no file needs to be kept; the passphrase then
                  has to be long and unpredictable, since anyone can try passphrases
                  against that salt. An organisation passes a salt of its own to make the
                  derivation specific to it. The default iteration count follows current
                  OWASP guidance.
            throws
                - crypto_error if iterations == 0.
        !*/

        void save(const std::string& filename) const;
        /*!
            ensures
                - Writes the key to a new file, readable by its owner only and
                  synchronized to the disk on POSIX systems.
            throws
                - crypto_error if the key is not set, or if the file already exists or
                  cannot be written: a key file is never overwritten, since every
                  document protected with it would become impossible to restore. A file
                  left incomplete by a failed write is removed.
        !*/

        static pseudonymization_key load(const std::string& filename);
        /*!
            ensures
                - Returns the key of the first "v1:" line of the file.
            throws
                - crypto_error if the file cannot be read or holds no valid key.
        !*/

        bool is_set() const;
        /*!
            ensures
                - Returns true if the object holds a key of 32 bytes.
        !*/

        const std::string& bytes() const;
        /*!
            ensures
                - Returns the 32 key bytes, or an empty string when is_set() == false.
        !*/
    };

// ----------------------------------------------------------------------------------------

    enum class date_policy
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                How dates are protected.
                shift   - every date moves into the past by the same secret number of
                          weeks: order, intervals, ages at a given date and weekdays are
                          preserved, and a date of the past never shows up in the future.
                          A month and its year ("mars 2022", "03/2022") move by the nearest
                          whole number of months, a year alone by the nearest whole number
                          of years, so that no date left without its day gives the shift
                          away
                encrypt - dates become opaque codes, only equality is preserved
                keep    - dates are left in clear: apply() and the document handlers
                          leave them untouched (see keeps_in_clear())
        !*/
        shift,
        encrypt,
        keep
    };

    struct pseudonymizer_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The configuration of a pseudonymizer.

                context         - a domain separation label mixed into every derived key.
                                  Documents protected under the same key and context
                                  share their pseudonyms (Dupont gets the same code in all
                                  of them); a different context makes them unlinkable.
                                  Restoration needs the context used for protection.
                dates           - the date policy
                max_shift_weeks - the largest shift of date_policy::shift, in weeks into
                                  the past, clamped to [1, 5200]
                date_order      - how numeric dates such as 03/04/2026 are read
        !*/
        std::string context;
        date_policy dates = date_policy::shift;
        unsigned long max_shift_weeks = 260;
        numeric_date_order date_order = numeric_date_order::day_month_year;
    };

// ----------------------------------------------------------------------------------------

    class pseudonymizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Vaultless, reversible and operation-preserving pseudonymization of
                personal data.

                Each detected value is replaced by a self-contained tag

                    [[TYPE:BODY#CHK]]

                where TYPE is the pii_type keyword, BODY is the value encrypted with a
                code chosen per type so that the operations meaningful for that type keep
                working on the protected text, and CHK is a 15 bit HMAC of the original
                value (three Crockford base32 characters) that authenticates the tag under
                the key and context.

                BODY CODES
                    Names, organisations, places, addresses, e-mails, URLs:
                        each word is encrypted separately with FF1 over the 26 letter
                        alphabet after case folding, and the case of the original word is
                        carried by the case of its code. "Dupont" and "DUPONT" become
                        "Zkwrtuvqahb" and "ZKWRTUVQAHB", and the code of a word is the same
                        wherever it occurs, so "Jean Dupont", "M. Dupont" and
                        jean.dupont@acme.fr visibly share their codes. Street types,
                        legal forms and the prepositions of places and organisations stay
                        readable ("rue", "SAS", "de"); the particles of a person name are
                        encrypted like its other words, since "Le", "van" or "bin" would
                        tell the language of the name. The top level domain of an e-mail
                        stays readable. Digit groups of six or
                        more are encrypted as digits. Words with irregular case are
                        encrypted exactly and marked with a trailing '!'. The length of a
                        code follows the length of its word, so a protected text still
                        tells a short name from a long one.
                    Civilities, sexes and nationalities (title, gender, nationality):
                        these values take few forms, so a deterministic code would give
                        them away through its frequency or its length. Their body starts
                        with a random nonce mixed into the encryption and the check, and
                        the value is padded to a multiple of 16 bytes: "M." and "Madame"
                        give bodies of the same length, and two occurrences of the same
                        value give two different tags.
                    Phone numbers: digits after the country code and first national digit
                        are encrypted with FF1 over the decimal alphabet; layout kept.
                    IBAN, payment cards, NIR, SIREN/SIRET: the variable digits are
                        encrypted and the check digits are recomputed, so the protected
                        value still passes mod 97, Luhn or NIR key validation. The
                        country code of an IBAN and the issuer prefix of a card are kept.
                    MAC addresses: the vendor prefix is kept, the device part encrypted.
                    Identifiers: digits stay digits, and a number of eight digits or more
                        that passes Luhn keeps a valid Luhn digit; alphanumeric values are
                        encrypted over 36 or 62 symbols with cycle walking, case
                        preserved. A UUID (device, advertising or cookie identifier) stays
                        a UUID: its 32 hexadecimal digits are encrypted over the
                        hexadecimal alphabet. The check digits of national numbers other
                        than those above are not recomputed.
                    GPS coordinates: every digit is encrypted, the signs, separators,
                        degree marks and hemisphere letters are kept.
                    License plates (AA-123-AA): encrypted within the plate space.
                    IPv4 and IPv6: Crypto-PAn, so shared prefixes (subnets) are preserved.
                    Dates (date_policy::shift): ISO 8601 rendering of the shifted date,
                        YYYY-MM-DD, YYYY-MM for a month and its year, YYYY for a year
                        alone, followed by "~" and a format code when the original was not
                        ISO.
                    Anything a dedicated code cannot represent, and every value whose tag
                        would not decode back to it, takes an exact FF1 encryption of the
                        whole value, rendered as lower case letters followed by '!'.

                FF1 is never run below its minimum domain: short values are padded with
                zero bytes inside the plaintext, and the padding is removed on decryption.

            RESTORATION
                restore() finds tags in any text, including text derived from a protected
                document such as a summary or an answer produced by a language model, and
                replaces each tag whose check verifies by the original value. Tags that do
                not verify (altered, forged, or produced under another key or context) are
                left in place and counted. The scan takes time linear in the length of the
                text; brackets nested in a tag mark the innermost one.

            THREAD SAFETY
                A pseudonymizer may be shared between threads; its memoization cache is
                protected by a mutex.
        !*/
    public:
        pseudonymizer(
            const pseudonymization_key& key,
            const pseudonymizer_options& options = pseudonymizer_options()
        );
        /*!
            ensures
                - #get_options() == options
                - The derived keys are computed and the key itself is not kept.
            throws
                - crypto_error if key.is_set() == false.
        !*/

        const pseudonymizer_options& get_options() const;
        /*!
            ensures
                - Returns the options given to the constructor.
        !*/

        std::string tag(pii_type type, const std::string& value) const;
        /*!
            requires
                - type != pii_type::none
            ensures
                - Returns the tag replacing value, which decode() reads back as (type,
                  value). Except for title, gender and nationality, the same (type, value)
                  always yields the same tag under a given key and context.
                - Under date_policy::encrypt a date takes the opaque code. The policy keep
                  is applied by apply() and the document handlers, which leave dates in
                  clear; tag() itself still protects a date it is given.
            throws
                - crypto_error if the value cannot be encoded, which the opaque code
                  rules out for any value of a reasonable size.
        !*/

        std::string apply(const std::string& text, const std::vector<pii_entity>& entities) const;
        /*!
            ensures
                - Returns text with every entity span replaced by its tag. Overlapping
                  entities are resolved in favour of the first in text order, the longest
                  first at the same position. Entities of type none, outside the text,
                  empty, or of a type kept in clear (keeps_in_clear()) are skipped.
        !*/

        bool keeps_in_clear(pii_type type) const;
        /*!
            ensures
                - Returns true for the values the options leave in clear: dates under
                  date_policy::keep.
        !*/

        struct restore_report
        {
            std::size_t restored = 0;   // tags replaced by their original value
            std::size_t rejected = 0;   // tags left in place because their check fails
        };

        std::string restore(const std::string& text, restore_report* report = nullptr) const;
        /*!
            ensures
                - Returns text with every verified tag replaced by its original value.
                  Tags are recognized in their raw form and percent-encoded inside URIs.
                - If report != nullptr, the counts are added to *report.
        !*/

        struct tag_occurrence
        {
            std::size_t begin = 0;      // byte range of the tag in the text
            std::size_t end = 0;
            pii_type type = pii_type::none;
            std::string original;
        };

        std::vector<tag_occurrence> find_tags(const std::string& text, restore_report* report = nullptr) const;
        /*!
            ensures
                - Returns the verified raw tags of text, in order, with their original
                  values, without changing text. Document handlers use it to restore values
                  inside structures where a tag may span several pieces of text.
                - If report != nullptr, the verified and rejected tags are counted in it.
        !*/

        bool decode(const std::string& tag_text, pii_type& type, std::string& original) const;
        /*!
            ensures
                - If tag_text is exactly one verified tag, sets #type and #original and
                  returns true. Returns false otherwise.
        !*/

        long date_shift_days() const;
        /*!
            ensures
                - Returns the secret shift of date_policy::shift in days: a negative
                  multiple of 7 whose magnitude is at most 7 times max_shift_weeks, once
                  clamped to [1, 5200]. It depends on the key and the context only.
        !*/

        long date_shift_months() const;
        /*!
            ensures
                - Returns the shift of the dates written without a day: date_shift_days()
                  in months of 30.436875 days, rounded to the nearest, and -1 when that
                  gives 0.
        !*/

        long date_shift_years() const;
        /*!
            ensures
                - Returns the shift of the years written alone: date_shift_days() in years
                  of 365.2425 days, rounded to the nearest, and -1 when that gives 0.
        !*/

        double shift_date_serial(double serial, bool forward) const;
        /*!
            ensures
                - Returns a spreadsheet date serial number shifted by date_shift_days()
                  (forward) or back. Used for date-typed spreadsheet cells, which stay
                  dates and keep working in formulas.
        !*/

        std::string key_fingerprint() const;
        /*!
            ensures
                - Returns 12 hexadecimal digits identifying (key, context) without
                  revealing either; reports carry it so the right key can be found.
        !*/

        void clear_cache() const;
        /*!
            ensures
                - Forgets the tags computed so far and erases the values they were
                  computed from. The cache holds at most 65536 values and is cleared when
                  full and when the object is destroyed.
        !*/
    };

// ----------------------------------------------------------------------------------------

    std::string percent_encode_tag(const std::string& tag);
    /*!
        ensures
            - Returns tag with every byte outside [A-Za-z0-9-._~] percent-encoded, the
              form used when a tag replaces part of a URI (hyperlink targets).
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PSEUDONYMIZER_ABSTRACT_H_

