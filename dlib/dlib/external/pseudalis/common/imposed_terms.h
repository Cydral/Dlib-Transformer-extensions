// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_IMPOSED_TERMS_H_
#define PSEUDALIS_COMMON_IMPOSED_TERMS_H_

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include <dlib/privacy/unicode_text.h>

#include "json.h"

namespace pseudalis
{
    /*
        The terms the administrators impose: the name of a client, of a project, of a contract
        or of an organisation that the detector does not know.  Each occurrence of a term is
        replaced with the tag of its type before the detector reads anything else of the text
        (dlib::pii_imposed_term): its words and numbers in the same order, as whole words,
        separated by at most two characters of one line, matching their case and their accents
        when case_sensitive is true, ignoring both otherwise.  The service keeps them in its
        database, which only SYSTEM and the administrators read, and gives them to the engine.
    */
    const std::size_t max_imposed_terms = 100;
    // In characters (code points), once the spaces at both ends are left out.
    const std::size_t max_imposed_term_length = 200;

    struct imposed_term
    {
        std::string term;
        std::string type = "ORG";
        bool case_sensitive = false;
    };

    // The types an imposed term may take, in the order the interface lists them.
    inline const std::vector<std::string>& imposed_term_types()
    {
        static const std::vector<std::string> types = { "ORG", "PERSON", "LOCATION", "ADDRESS", "ID", "EMAIL",
                                                        "PHONE", "URL" };
        return types;
    }

    /*
        Reads a list [{"term", "type", "case_sensitive"}], each term trimmed of the spaces at its
        ends.  Returns false with a message, a code and the position of the term at fault (-1 for
        the list itself) when the list is not one the service takes:
            terms_invalid    not a list, an entry without a term, a term empty, longer than
                             max_imposed_term_length, on several lines, with a control character
                             or invalid UTF-8, or without a letter or a digit;
            terms_too_many   more than max_imposed_terms;
            terms_type       a type out of imposed_term_types();
            terms_duplicate  a term given twice with the same case sensitivity (without it, two
                             terms that differ by their case or their accents are the same).
    */
    inline bool parse_imposed_terms(const json& list, std::vector<imposed_term>& terms, std::string& message,
                                    std::string& code, long long& index)
    {
        using namespace dlib::privacy_impl;
        terms.clear();
        index = -1;
        if (!list.is_array())
        {
            message = "the terms must be a list";
            code = "terms_invalid";
            return false;
        }
        if (list.size() > max_imposed_terms)
        {
            message = "at most " + std::to_string(max_imposed_terms) + " terms may be imposed";
            code = "terms_too_many";
            return false;
        }
        std::set<std::string> seen;
        for (std::size_t i = 0; i < list.size(); ++i)
        {
            const json& item = list[i];
            index = static_cast<long long>(i);
            imposed_term t;
            const std::string raw = item.is_object() ? item["term"].str() : std::string();
            // The spaces at both ends go; what remains is the term.
            std::size_t b = 0, e = raw.size();
            while (b < e && (raw[b] == ' ' || raw[b] == '\t')) ++b;
            while (e > b && (raw[e - 1] == ' ' || raw[e - 1] == '\t')) --e;
            t.term = raw.substr(b, e - b);
            bool ok = !t.term.empty() && utf8_valid(t.term);
            std::size_t length = 0;
            bool word = false;
            for (std::size_t k = 0; ok && k < t.term.size();)
            {
                const char32_t c = utf8_next(t.term, k);
                ++length;
                if (c < 0x20 || c == 0x7F || is_line_break(c)) ok = false;
                if (is_word_char(c)) word = true;
            }
            if (!ok || !word || length > max_imposed_term_length)
            {
                message = "each term must be a text of one line, of 1 to " + std::to_string(max_imposed_term_length) +
                          " characters, with a letter or a digit";
                code = "terms_invalid";
                return false;
            }
            t.type = item["type"].str("ORG");
            bool known = false;
            for (const std::string& k : imposed_term_types()) known = known || k == t.type;
            if (!known)
            {
                message = "the type of a term must be one of ORG, PERSON, LOCATION, ADDRESS, ID, EMAIL, PHONE, URL";
                code = "terms_type";
                return false;
            }
            t.case_sensitive = item["case_sensitive"].as_bool(false);
            const std::string key = (t.case_sensitive ? "1" : "0") + (t.case_sensitive ? t.term : fold_key(t.term));
            if (!seen.insert(key).second)
            {
                message = "the term \"" + t.term + "\" is given twice";
                code = "terms_duplicate";
                return false;
            }
            terms.push_back(t);
        }
        index = -1;
        return true;
    }

    inline json imposed_terms_json(const std::vector<imposed_term>& terms)
    {
        json list = json::array();
        for (const imposed_term& t : terms)
        {
            json item = json::object();
            item["term"] = t.term;
            item["type"] = t.type;
            item["case_sensitive"] = t.case_sensitive;
            list.push_back(item);
        }
        return list;
    }
}

#endif // PSEUDALIS_COMMON_IMPOSED_TERMS_H_
