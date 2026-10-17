// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_DOCUMENT_REPORT_H_
#define DLIB_PRIVACY_DOCUMENT_REPORT_H_

#include "document_report_abstract.h"

#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "pii_types.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct document_report
    {
        std::string format;
        std::string operation;
        std::string key_fingerprint;
        std::map<std::string, std::size_t> entities;
        std::size_t text_units = 0;
        std::size_t tags_restored = 0;
        std::size_t tags_rejected = 0;
        std::size_t dates_shifted = 0;
        std::vector<std::string> removed;
        std::vector<std::string> notes;

        void count(pii_type t) { ++entities[pii_type_keyword(t)]; }

        std::size_t total_entities() const
        {
            std::size_t n = 0;
            for (const auto& e : entities) n += e.second;
            return n;
        }

        void note(const std::string& message)
        {
            for (const std::string& n : notes)
                if (n == message) return;
            notes.push_back(message);
        }

        std::string to_json() const
        {
            auto esc = [](const std::string& s) {
                std::string o;
                for (char c : s)
                {
                    if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
                    else if (c == '\n') o += "\\n";
                    else if (static_cast<unsigned char>(c) < 0x20) o += " ";
                    else o.push_back(c);
                }
                return o;
            };
            std::ostringstream o;
            o << "{\n  \"operation\": \"" << esc(operation) << "\",\n  \"format\": \"" << esc(format)
              << "\",\n  \"key_fingerprint\": \"" << esc(key_fingerprint) << "\",\n  \"text_units\": " << text_units
              << ",\n  \"entities\": {";
            bool first = true;
            for (const auto& e : entities)
            {
                o << (first ? "" : ",") << "\n    \"" << esc(e.first) << "\": " << e.second;
                first = false;
            }
            o << (entities.empty() ? "" : "\n  ") << "},\n  \"entities_total\": " << total_entities()
              << ",\n  \"tags_restored\": " << tags_restored << ",\n  \"tags_rejected\": " << tags_rejected
              << ",\n  \"dates_shifted\": " << dates_shifted << ",\n  \"removed\": [";
            for (std::size_t i = 0; i < removed.size(); ++i)
                o << (i ? ", " : "") << "\"" << esc(removed[i]) << "\"";
            o << "],\n  \"notes\": [";
            for (std::size_t i = 0; i < notes.size(); ++i)
                o << (i ? "," : "") << "\n    \"" << esc(notes[i]) << "\"";
            o << (notes.empty() ? "" : "\n  ") << "]\n}\n";
            return o.str();
        }
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DOCUMENT_REPORT_H_
