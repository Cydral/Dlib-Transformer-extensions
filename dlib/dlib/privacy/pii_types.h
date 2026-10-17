// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_TYPES_H_
#define DLIB_PII_TYPES_H_

#include "pii_types_abstract.h"

#include <cstddef>
#include <string>
#include <vector>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class pii_type
    {
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
        std::size_t begin = 0;
        std::size_t end = 0;
        pii_type type = pii_type::none;
        double confidence = 0;
    };

    struct pii_span
    {
        std::size_t begin = 0;
        std::size_t end = 0;
        std::string label;
        pii_type type = pii_type::none;
        double confidence = 1;
    };

// ----------------------------------------------------------------------------------------

    inline const std::vector<pii_type>& all_pii_types()
    {
        static const std::vector<pii_type> types = {
            pii_type::person, pii_type::organization, pii_type::location, pii_type::address,
            pii_type::email, pii_type::phone, pii_type::iban, pii_type::payment_card,
            pii_type::nir, pii_type::siret, pii_type::identifier, pii_type::ipv4,
            pii_type::ipv6, pii_type::mac_address, pii_type::url, pii_type::date,
            pii_type::license_plate, pii_type::geolocation, pii_type::title, pii_type::gender,
            pii_type::nationality };
        return types;
    }

    inline const char* pii_type_keyword(pii_type t)
    {
        switch (t)
        {
            case pii_type::person: return "PERSON";
            case pii_type::organization: return "ORG";
            case pii_type::location: return "LOCATION";
            case pii_type::address: return "ADDRESS";
            case pii_type::email: return "EMAIL";
            case pii_type::phone: return "PHONE";
            case pii_type::iban: return "IBAN";
            case pii_type::payment_card: return "CARD";
            case pii_type::nir: return "NIR";
            case pii_type::siret: return "SIRET";
            case pii_type::identifier: return "ID";
            case pii_type::ipv4: return "IPV4";
            case pii_type::ipv6: return "IPV6";
            case pii_type::mac_address: return "MAC";
            case pii_type::url: return "URL";
            case pii_type::date: return "DATE";
            case pii_type::license_plate: return "PLATE";
            case pii_type::geolocation: return "GEO";
            case pii_type::title: return "TITLE";
            case pii_type::gender: return "GENDER";
            case pii_type::nationality: return "NATIONALITY";
            case pii_type::none: break;
        }
        return "NONE";
    }

    inline pii_type pii_type_from_keyword(const std::string& keyword)
    {
        std::string k;
        for (char ch : keyword)
            k.push_back((ch >= 'a' && ch <= 'z') ? static_cast<char>(ch - 32) : (ch == '-' ? '_' : ch));
        for (pii_type t : all_pii_types())
            if (k == pii_type_keyword(t))
                return t;
        if (k == "ORGANIZATION" || k == "ORGANISATION") return pii_type::organization;
        if (k == "PAYMENT_CARD" || k == "CREDIT_CARD") return pii_type::payment_card;
        if (k == "IDENTIFIER") return pii_type::identifier;
        if (k == "MAC_ADDRESS") return pii_type::mac_address;
        if (k == "LICENSE_PLATE" || k == "LICENCE_PLATE") return pii_type::license_plate;
        if (k == "SIREN") return pii_type::siret;
        if (k == "GEOLOCATION" || k == "GPS" || k == "COORDINATES") return pii_type::geolocation;
        if (k == "CIVILITY" || k == "HONORIFIC") return pii_type::title;
        if (k == "SEX") return pii_type::gender;
        if (k == "CITIZENSHIP") return pii_type::nationality;
        return pii_type::none;
    }

    inline const std::vector<std::string>& pii_fine_labels()
    {
        static const std::vector<std::string> labels = {
            "GIVEN_NAME", "FAMILY_NAME", "USERNAME", "STREET_ADDRESS", "POSTAL_CODE", "CITY", "ORGANIZATION" };
        return labels;
    }

    inline pii_type pii_type_from_label(const std::string& label)
    {
        if (label == "GIVEN_NAME" || label == "FAMILY_NAME") return pii_type::person;
        if (label == "POSTAL_CODE" || label == "CITY") return pii_type::location;
        if (label == "STREET_ADDRESS") return pii_type::address;
        if (label == "ORGANIZATION") return pii_type::organization;
        if (label == "USERNAME") return pii_type::identifier;
        return pii_type_from_keyword(label);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_TYPES_H_
