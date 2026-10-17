// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OOXML_PACKAGE_ABSTRACT_H_
#ifdef DLIB_OOXML_PACKAGE_ABSTRACT_H_

#include <string>
#include <vector>
#include "../zip/zip_archive_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class ooxml_kind
    {
        unknown,
        wordprocessing,     // .docx, .docm, .dotx
        spreadsheet,        // .xlsx, .xlsm, .xltx
        presentation        // .pptx, .pptm, .potx
    };

    struct ooxml_relationship
    {
        std::string id;
        std::string type;       // full relationship type URI
        std::string target;     // as written
        bool external = false;  // TargetMode="External"
    };

    class ooxml_package
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An Office Open XML package (ECMA-376 Part 2, Open Packaging Conventions)
                held in memory: its parts, the content types declared for them and the
                relationships between them.

                The package is identified from its content: the target of the root
                officeDocument relationship and its content type tell a document, a
                workbook and a presentation apart, whatever the file is called.

                Parts are named without the leading slash ("word/document.xml"). As in
                OPC, part names are compared with ASCII letters case folded: every
                member function accepts a name in any case, and part_names() returns
                the names as stored. XML parts stored in UTF-16 are converted to UTF-8
                when the package is read, their XML declaration included.
        !*/
    public:
        explicit ooxml_package(const std::string& zip_bytes, const zip_limits& limits = zip_limits());
        /*!
            ensures
                - Reads every member of the archive, folders excepted, as a part.
            throws
                - zip_error if the archive cannot be read.
                - xml_error if [Content_Types].xml is not well-formed.
                - error if it has no [Content_Types].xml, or if two members name the
                  same part (after removing a leading slash and folding case).
        !*/

        ooxml_kind kind() const;
        /*!
            ensures
                - Returns the kind given by the content type of main_part(), or
                  ooxml_kind::unknown.
            throws
                - xml_error if the package relationships are not well-formed.
        !*/

        std::string main_part() const;
        /*!
            ensures
                - Returns the target of the root officeDocument relationship, or "".
            throws
                - xml_error if the package relationships are not well-formed.
        !*/

        std::vector<std::string> part_names() const;
        /*!
            ensures
                - Returns the names of the parts, each once, in archive order followed
                  by the parts added since.
        !*/

        bool has_part(const std::string& name) const;
        /*!
            ensures
                - Returns true if the package holds a part with this name.
        !*/

        const std::string& part(const std::string& name) const;
        /*!
            ensures
                - Returns the content of the part.
            throws
                - error if there is no such part.
        !*/

        void set_part(const std::string& name, const std::string& content);
        /*!
            ensures
                - Replaces the content of the part, keeping its stored name, or adds the
                  part at the end of part_names().
        !*/

        void remove_part(const std::string& name);
        /*!
            ensures
                - Removes the part and its content type override, if any.
        !*/

        std::string content_type(const std::string& name) const;
        /*!
            ensures
                - Returns the content type of the part: its override, else the default
                  for its extension, else "".
        !*/

        void set_override(const std::string& name, const std::string& content_type);
        /*!
            ensures
                - #content_type(name) == content_type
        !*/

        static std::string rels_name(const std::string& source);
        /*!
            ensures
                - Returns the relationships part of source: "_rels/.rels" for the package
                  itself (source == ""), "word/_rels/document.xml.rels" for
                  "word/document.xml".
        !*/

        std::vector<ooxml_relationship> relationships(const std::string& source) const;
        /*!
            ensures
                - Returns the relationships declared in rels_name(source), in order, or
                  an empty vector when that part does not exist.
            throws
                - xml_error if the relationships part is not well-formed.
        !*/

        static std::string resolve_target(const std::string& source, const std::string& target);
        /*!
            ensures
                - Resolves an internal relationship target against the folder of source,
                  "../media/image1.png" from "word/document.xml" giving
                  "media/image1.png" relative to the package root with ".." folded and
                  percent escapes decoded.
        !*/

        std::string target_part(const std::string& source, const std::string& target) const;
        /*!
            ensures
                - Returns the stored name of the part the target designates: the part
                  named by resolve_target(source, target), else the part named by the
                  same path with its percent escapes kept, else
                  resolve_target(source, target).
        !*/

        std::vector<std::string> collect_garbage();
        /*!
            ensures
                - Removes every part that is not reachable from the package
                  relationships, together with orphaned relationship parts and content
                  type overrides, and returns the names removed.
            throws
                - xml_error if a relationships part is not well-formed.
        !*/

        std::string save(int level = 6) const;
        /*!
            ensures
                - Returns a new archive, [Content_Types].xml first, every member with the
                  neutral timestamp of zip_archive_writer.
            throws
                - zip_error if the archive would exceed the limits of zip_archive_writer.
        !*/

        static bool is_rels(const std::string& name);
        /*!
            ensures
                - Returns true if name is a relationships part: a ".rels" file in a
                  "_rels" folder.
        !*/

        static std::string rels_source(const std::string& rels);
        /*!
            requires
                - is_rels(rels) == true
            ensures
                - Returns the part whose relationships rels holds, "" for "_rels/.rels":
                  the inverse of rels_name().
        !*/

        static bool ends_with(const std::string& s, const std::string& suffix);
        /*!
            ensures
                - Returns true if s ends with suffix.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OOXML_PACKAGE_ABSTRACT_H_
