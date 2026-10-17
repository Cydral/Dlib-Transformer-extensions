// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PII_TYPES_ABSTRACT_H_
#ifdef DLIB_PII_TYPES_ABSTRACT_H_

#include <cstddef>
#include <string>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class pii_type
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                The categories of personal data handled by the detector and the
                pseudonymizer. The keyword of each value, given by pii_type_keyword(), is
                also the keyword of the tags that replace the data, so a reader of a
                protected document sees [[PERSON:...]], [[IBAN:...]] or [[DATE:...]] and
                knows what was there.
        !*/
        none,
        person,         // natural person names
        organization,   // company, institution and public body names
        location,       // cities, birth places, postal code with city
        address,        // street addresses
        email,
        phone,
        iban,
        payment_card,
        nir,            // French social security number, key validated
        siret,          // French company identifiers SIREN and SIRET, Luhn validated
        identifier,     // passport, identity card, licence, customer, file, employee, device
                        // and foreign national numbers, account names
        ipv4,
        ipv6,
        mac_address,
        url,
        date,
        license_plate,
        geolocation,    // GPS coordinates
        title,          // civility before a name or after its label: M., Mme, Mrs, Ms
        gender,         // the value of a sex or gender field
        nationality     // a nationality written after its label or cue
    };

    struct pii_entity
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One detected piece of personal data: a byte span [begin, end) of a UTF-8
                text, its type and a confidence in [0, 1].
        !*/
        std::size_t begin = 0;
        std::size_t end = 0;
        pii_type type = pii_type::none;
        double confidence = 0;
    };

    struct pii_span
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One span reported by a learned recognizer or read from an annotated
                dataset: the byte range [begin, end) of a UTF-8 text, the label it
                carries (a fine label such as "GIVEN_NAME", "CITY" or "STREET_ADDRESS",
                or a tag keyword such as "EMAIL"), the pii_type its value is replaced as,
                and a confidence in [0, 1].
        !*/
        std::size_t begin = 0;
        std::size_t end = 0;
        std::string label;
        pii_type type = pii_type::none;
        double confidence = 1;
    };

// ----------------------------------------------------------------------------------------

    const std::vector<pii_type>& all_pii_types(
    );
    /*!
        ensures
            - Returns every value of pii_type except none, in the order of the enum.
    !*/

    const char* pii_type_keyword(
        pii_type t
    );
    /*!
        ensures
            - Returns the tag keyword of t: PERSON, ORG, LOCATION, ADDRESS, EMAIL, PHONE,
              IBAN, CARD, NIR, SIRET, ID, IPV4, IPV6, MAC, URL, DATE, PLATE, GEO, TITLE,
              GENDER or NATIONALITY, and "NONE" for pii_type::none.
    !*/

    pii_type pii_type_from_keyword(
        const std::string& keyword
    );
    /*!
        ensures
            - Returns the type whose keyword equals keyword, without regard to case, a
              '-' reading as '_'. Command line names are accepted too: ORGANIZATION,
              ORGANISATION, PAYMENT_CARD, CREDIT_CARD, IDENTIFIER, MAC_ADDRESS,
              LICENSE_PLATE, LICENCE_PLATE, SIREN, GEOLOCATION, GPS, COORDINATES,
              CIVILITY, HONORIFIC, SEX and CITIZENSHIP.
            - Returns pii_type::none for any other string.
    !*/

    const std::vector<std::string>& pii_fine_labels(
    );
    /*!
        ensures
            - Returns the labels of the learned recognizers, in the order of their class
              indices: GIVEN_NAME, FAMILY_NAME, USERNAME, STREET_ADDRESS, POSTAL_CODE,
              CITY and ORGANIZATION. They split the categories that rules find hardest
              (names, places, addresses, organisations) into the parts a model can learn.
    !*/

    pii_type pii_type_from_label(
        const std::string& label
    );
    /*!
        ensures
            - Returns the type a value carrying this label is replaced as: GIVEN_NAME and
              FAMILY_NAME give person, POSTAL_CODE and CITY give location, STREET_ADDRESS
              gives address, ORGANIZATION gives organization and USERNAME gives
              identifier. Any other label is read as a tag keyword with
              pii_type_from_keyword().
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_TYPES_ABSTRACT_H_

