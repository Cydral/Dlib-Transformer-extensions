// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_XML_SCANNER_ABSTRACT_H_
#ifdef DLIB_XML_SCANNER_ABSTRACT_H_

#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class xml_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Thrown by xml_tokenize() on input that is not well-formed enough to be
                rewritten safely.
        !*/
    };

    struct xml_attribute
    {
        std::string name;           // qualified name, e.g. "w:author"
        std::size_t value_begin;    // byte range of the raw (escaped) value, quotes excluded
        std::size_t value_end;
    };

    struct xml_token
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One lexical unit of an XML document, located by byte offsets into the
                source. For text and CDATA tokens, [content_begin, content_end) is the
                raw character data; for tags it equals [begin, end).
        !*/
        enum token_kind { start_tag, end_tag, empty_tag, text, cdata, comment, processing_instruction, doctype };
        token_kind kind;
        std::size_t begin;
        std::size_t end;
        std::size_t content_begin;
        std::size_t content_end;
        std::string name;
        std::vector<xml_attribute> attributes;
    };

    std::vector<xml_token> xml_tokenize(const std::string& xml);
    /*!
        requires
            - xml is encoded in UTF-8 or another encoding compatible with ASCII, such
              as the output of xml_to_utf8().
        ensures
            - Splits xml into tokens that cover it completely and in order, so that any
              part of the document can be rewritten by byte range while the rest is
              copied unchanged. Unlike dlib::xml_parser, which reports SAX events, it
              keeps attributes in document order and exposes source offsets.
            - Entities are never expanded and DOCTYPE declarations are passed through as
              opaque tokens.
        throws
            - xml_error on an unterminated tag, comment, CDATA section or attribute, an
              attribute without a quoted value, or a tag without a name.
    !*/

    std::vector<xml_token> html_tokenize(const std::string& html);
    /*!
        ensures
            - Splits html into tokens like xml_tokenize(), with the tolerance of HTML
              parsers instead of errors: attribute values may be unquoted or absent (an
              attribute without value gets an empty value range at the end of its
              name), void elements such as <br> and <img> are empty tags, a '<' that
              starts no tag is text, the contents of script and style elements are one
              text token, and a construct left open at the end of the input ends there.
            - Adjacent text is merged into one token. The tokens cover html completely
              and in order.
    !*/

    std::string xml_to_utf8(const std::string& xml);
    /*!
        ensures
            - Returns xml converted to UTF-8 when it is UTF-16: with a byte order mark,
              or without one when it starts with '<' encoded on two bytes. The
              encoding named by its XML declaration becomes "UTF-8", and unpaired
              surrogates become U+FFFD.
            - Returns xml unchanged otherwise.
    !*/

    std::string xml_unescape(const std::string& raw);
    /*!
        ensures
            - Decodes the five predefined entities and numeric character references,
              leading zeros included. Unknown entities, references to surrogates or
              beyond U+10FFFF, and references longer than 64 characters are kept
              verbatim.
    !*/

    std::string xml_escape_text(const std::string& text);
    std::string xml_escape_attribute(const std::string& text);
    /*!
        ensures
            - Escapes text for use as character data, or as a double quoted attribute
              value (where tabs and line breaks are written as character references so
              that attribute value normalization keeps them).
    !*/

    std::string xml_local_name(const std::string& qname);
    /*!
        ensures
            - Returns qname without its prefix: "author" for "w:author".
    !*/

    std::string xml_prefix(const std::string& qname);
    /*!
        ensures
            - Returns the prefix of qname, "" when it has none: "w" for "w:author".
    !*/

    class xml_namespace_tracker
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Follows the xmlns declarations in scope while tokens are visited in
                order, so that elements are identified by namespace URI and local name
                whatever prefix a producer chose. Lookups take a time logarithmic in
                the number of prefixes in scope.
        !*/
    public:
        void enter(const std::string& source, const xml_token& start_or_empty_tag);
        /*!
            requires
                - start_or_empty_tag is a token of source.
            ensures
                - Opens the scope of the element: the xmlns declarations of its
                  attributes are in force until the matching leave().
                - #depth() == depth() + 1
        !*/

        void leave();
        /*!
            ensures
                - Closes the innermost scope opened by enter(), if any.
                - #depth() == max(depth() - 1, 0)
        !*/

        std::string resolve(const std::string& qname, bool is_attribute = false) const;
        /*!
            ensures
                - Returns the namespace URI bound to the prefix of qname, the default
                  namespace for unprefixed element names, or "" for unprefixed attributes.
        !*/

        std::size_t depth() const;
        /*!
            ensures
                - Returns the number of scopes open.
        !*/
    };

    class xml_edit_list
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Byte range replacements and insertions over a source document, applied
                in one pass.
        !*/
    public:
        void replace(std::size_t begin, std::size_t end, const std::string& text);
        /*!
            requires
                - begin <= end
            ensures
                - Records that the bytes [begin, end) are to be replaced by text.
        !*/

        void insert(std::size_t at, const std::string& text);
        /*!
            ensures
                - Records that text is to be inserted before the byte at.
        !*/

        bool empty() const;
        /*!
            ensures
                - Returns true if no edit is recorded.
        !*/

        bool covers(std::size_t begin, std::size_t end) const;
        /*!
            ensures
                - Returns true if a recorded replacement of a non-empty range contains
                  [begin, end).
        !*/

        std::string apply(const std::string& source) const;
        /*!
            ensures
                - Returns source with the edits applied. Edits are taken in order of
                  position; at one position insertions come first, in the order they
                  were recorded, then the widest replacement. An edit that starts inside
                  a range already replaced is dropped, as is an edit that ends beyond
                  source.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_XML_SCANNER_ABSTRACT_H_
