// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PRIVACY_DOCUMENT_REPORT_ABSTRACT_H_
#ifdef DLIB_PRIVACY_DOCUMENT_REPORT_ABSTRACT_H_

#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include "pii_types_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct document_report
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The account of one anonymization or restoration run. It never contains
                original values: counts by type, the names of the parts or objects that
                were removed, and notes about limitations met, so that it can be stored
                next to the protected document.

                format          - the name of the document format
                operation       - "anonymize" or "restore"
                key_fingerprint - the fingerprint of the key used (pseudonymizer::
                                  key_fingerprint()), which identifies the key without
                                  revealing it
                entities        - the number of tags written, by tag keyword. A value
                                  that a PDF line end cuts gets one tag for each part,
                                  so that restoration counts the same tags.
                text_units      - the number of text units read: paragraphs, cells,
                                  lines of text, metadata fields
                tags_restored   - restoration: the number of tags put back
                tags_rejected   - restoration: the number of tags left in place because
                                  their check fails under the key and context in use
                                  (another key, another context, an altered tag)
                dates_shifted   - the number of dates moved by the secret shift
                removed         - the parts, pictures, objects and properties removed
                notes           - the limitations met, each written once
        !*/
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

        void count(pii_type t);
        /*!
            ensures
                - Adds one to entities[pii_type_keyword(t)].
        !*/

        std::size_t total_entities() const;
        /*!
            ensures
                - Returns the sum of the counts of entities.
        !*/

        void note(const std::string& message);
        /*!
            ensures
                - Appends message to notes unless notes already holds it.
        !*/

        std::string to_json() const;
        /*!
            ensures
                - Returns the report as a JSON object with the fields above and
                  entities_total, control characters of the strings written as spaces.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DOCUMENT_REPORT_ABSTRACT_H_

