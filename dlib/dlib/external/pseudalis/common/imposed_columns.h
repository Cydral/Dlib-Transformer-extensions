// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_IMPOSED_COLUMNS_H_
#define PSEUDALIS_COMMON_IMPOSED_COLUMNS_H_

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include <dlib/privacy/unicode_text.h>

#include "json.h"

namespace pseudalis
{
    /*
        The columns the administrators impose on the CSV files with a header, the column of the
        names of the clients or of their e-mail addresses in an export for instance.  Every
        value below the name of such a column in the header of a file takes the substitute of the
        type of the column, without detection and without tag, whatever the tags of the rest of
        the file (dlib::pii_imposed_column); the record of the file keeps each substitute with its
        tag, as for a file protected without tags.  A name is compared without regard to case and
        accents, the spaces at its ends left out.  The service keeps the columns in its database,
        which only SYSTEM and the administrators read, and gives them to the engine.
    */
    const std::size_t max_imposed_columns = 100;
    // In characters (code points), once the spaces at both ends are left out.
    const std::size_t max_imposed_column_length = 200;

    struct imposed_column
    {
        std::string name;
        std::string type = "PERSON";
    };

    /*
        The types an imposed column may take, in the order the interface lists them.  IP stands for
        the IP addresses of both forms, each value taking the substitute of its own, IPv4 or IPv6,
        with its prefix length, zone or port kept.
    */
    inline const std::vector<std::string>& imposed_column_types()
    {
        static const std::vector<std::string> types = { "PERSON", "ORG", "LOCATION", "ADDRESS", "EMAIL", "PHONE",
                                                        "ID", "DATE", "IBAN", "CARD", "NIR", "SIRET", "URL", "IP",
                                                        "MAC", "PLATE", "GEO", "TITLE", "GENDER", "NATIONALITY" };
        return types;
    }

    /*
        Reads a list [{"column", "type"}], each name trimmed of the spaces at its ends.  Returns
        false with a message, a code and the position of the column at fault (-1 for the list
        itself) when the list is not one the service takes:
            columns_invalid    not a list, an entry without a name, a name empty, longer than
                               max_imposed_column_length, on several lines, with a control
                               character or invalid UTF-8, or without a letter or a digit;
            columns_too_many   more than max_imposed_columns;
            columns_type       a type out of imposed_column_types();
            columns_duplicate  a name given twice, without regard to case and accents.
    */
    inline bool parse_imposed_columns(const json& list, std::vector<imposed_column>& columns, std::string& message,
                                      std::string& code, long long& index)
    {
        using namespace dlib::privacy_impl;
        columns.clear();
        index = -1;
        if (!list.is_array())
        {
            message = "the columns must be a list";
            code = "columns_invalid";
            return false;
        }
        if (list.size() > max_imposed_columns)
        {
            message = "at most " + std::to_string(max_imposed_columns) + " columns may be imposed";
            code = "columns_too_many";
            return false;
        }
        std::set<std::string> seen;
        for (std::size_t i = 0; i < list.size(); ++i)
        {
            const json& item = list[i];
            index = static_cast<long long>(i);
            imposed_column c;
            const std::string raw = item.is_object() ? item["column"].str() : std::string();
            std::size_t b = 0, e = raw.size();
            while (b < e && (raw[b] == ' ' || raw[b] == '\t')) ++b;
            while (e > b && (raw[e - 1] == ' ' || raw[e - 1] == '\t')) --e;
            c.name = raw.substr(b, e - b);
            bool ok = !c.name.empty() && utf8_valid(c.name);
            std::size_t length = 0;
            bool word = false;
            for (std::size_t k = 0; ok && k < c.name.size();)
            {
                const char32_t ch = utf8_next(c.name, k);
                ++length;
                if (ch < 0x20 || ch == 0x7F || is_line_break(ch)) ok = false;
                if (is_word_char(ch)) word = true;
            }
            if (!ok || !word || length > max_imposed_column_length)
            {
                message = "each column must be named by a text of one line, of 1 to " +
                          std::to_string(max_imposed_column_length) + " characters, with a letter or a digit";
                code = "columns_invalid";
                return false;
            }
            c.type = item["type"].str("PERSON");
            bool known = false;
            for (const std::string& k : imposed_column_types()) known = known || k == c.type;
            if (!known)
            {
                message = "the type of a column must be one of the types of the tags";
                code = "columns_type";
                return false;
            }
            if (!seen.insert(fold_key(c.name)).second)
            {
                message = "the column \"" + c.name + "\" is given twice";
                code = "columns_duplicate";
                return false;
            }
            columns.push_back(c);
        }
        index = -1;
        return true;
    }

    inline json imposed_columns_json(const std::vector<imposed_column>& columns)
    {
        json list = json::array();
        for (const imposed_column& c : columns)
        {
            json item = json::object();
            item["column"] = c.name;
            item["type"] = c.type;
            list.push_back(item);
        }
        return list;
    }
}

#endif // PSEUDALIS_COMMON_IMPOSED_COLUMNS_H_
