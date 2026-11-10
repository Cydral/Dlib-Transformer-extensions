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

    struct document_substitution
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A value written without its tag: its substitute and the tag it stands for.
                text is true for a value of the text of a document written without its tag
                (document_anonymizer::set_untagged_text()) or of an imposed column
                (document_anonymizer::set_imposed_columns()), false for a value of a name or
                of a property.
        !*/
        std::string substitute;
        std::string tag;
        bool text = false;
    };

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
                encoding        - the encoding in which a text file (formats txt, csv and
                                  xml) was written: "utf-8", "utf-8-bom", "utf-16le",
                                  "utf-16be", "utf-16le-nobom", "utf-16be-nobom" or
                                  "windows-1252", the encoding it was read in, but for a
                                  text of plain ASCII that restore() writes in Windows-1252
                                  (document_anonymizer::set_ascii_windows_1252()); empty
                                  for the other formats
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
                substitutions   - anonymization of names and properties without tags
                                  (document_anonymizer::set_untagged_properties()), and of
                                  the text without tags (document_anonymizer::
                                  set_untagged_text()): each substitute written
                                  (pseudonymizer::substitute()) with the tag it stands for,
                                  which the caller keeps to restore it, once for each place
                                  it is written in, and each value of an imposed column
                                  (document_anonymizer::set_imposed_columns()) once
                substitutes_restored - restoration: the number of substitutes put back
                pictures_removed - restoration: the number of picture texts removed
                                  (document_anonymizer::set_picture_text_finder())
        !*/
        std::string format;
        std::string encoding;
        std::string operation;
        std::string key_fingerprint;
        std::map<std::string, std::size_t> entities;
        std::size_t text_units = 0;
        std::size_t tags_restored = 0;
        std::size_t tags_rejected = 0;
        std::size_t dates_shifted = 0;
        std::vector<std::string> removed;
        std::vector<std::string> notes;
        std::vector<document_substitution> substitutions;
        std::size_t substitutes_restored = 0;
        std::size_t pictures_removed = 0;

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
                  entities_total, control characters of the strings written as spaces; the
                  substitutions are given by their number.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_DOCUMENT_REPORT_ABSTRACT_H_

