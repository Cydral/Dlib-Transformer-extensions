// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DOCUMENT_ANONYMIZER_ABSTRACT_H_
#ifdef DLIB_DOCUMENT_ANONYMIZER_ABSTRACT_H_

#include <string>
#include "pii_detector_abstract.h"
#include "pseudonymizer_abstract.h"
#include "document_report_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class document_format
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                The document formats handled by document_anonymizer.

                text            - plain text in UTF-8 (with or without BOM), UTF-16 (with
                                  or without BOM) or Windows-1252; also logs, Markdown,
                                  JSON, source code
                delimited       - CSV, TSV and other delimiter-separated values
                markup          - XML, HTML, SVG and other markup
                docx, xlsx,
                pptx            - Office Open XML packages, including the template and
                                  macro-enabled variants
                pdf             - PDF documents
                legacy_office   - binary Office files (.doc, .xls, .ppt) and password
                                  protected Office files, which are refused
                unknown         - anything else, refused
        !*/
        unknown,
        text,
        delimited,
        markup,
        docx,
        xlsx,
        pptx,
        pdf,
        legacy_office
    };

    const char* document_format_name(document_format f);
    /*!
        ensures
            - Returns "txt", "csv", "xml", "docx", "xlsx", "pptx", "pdf", "legacy-office"
              or "unknown".
    !*/

    document_format detect_document_format(
        const std::string& bytes,
        const std::string& name_hint = ""
    );
    /*!
        ensures
            - Identifies the format from the content: PDF, ZIP and OLE2 signatures, the
              package parts of an Office Open XML archive, byte order marks and the layout
              of UTF-16 text, then the text and markup heuristics. name_hint, a file name,
              is only used to tell delimited files from other text when the content is
              ambiguous: a ".csv", ".tsv" or ".tab" extension makes its delimiter the
              likely one.
    !*/

    document_format detect_document_file_format(
        const std::string& filename
    );
    /*!
        ensures
            - Classifies the file with detect_file_type() (dlib/data_io), then refines the
              answer: a ZIP archive is opened to recognize an Office package whatever its
              extension, UTF-16 text is recognized with or without its byte order mark, and
              a text file is checked for delimiter-separated values.
            - Returns document_format::unknown if the file cannot be read.
    !*/

// ----------------------------------------------------------------------------------------

    class document_anonymizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The single entry point for protecting and restoring documents. It
                dispatches on the format and applies the same pseudonymizer and detector
                to every format, so that one key yields the same tags across a whole
                corpus of Word, Excel, PowerPoint, PDF and text files.

                Text formats are handled here:
                    text      - the encoding is detected and kept; detection runs in two
                                passes over paragraphs (blocks separated by blank lines,
                                cut at a line break past 64 KiB).
                    delimited - each cell is a unit. A header row made of field labels
                                ("nom", "email", "telephone") types its columns, and a
                                label cell types the cell on its right, as in key and
                                value files; the labels themselves stay readable. Whole
                                rows feed the learning pass, so that a name established in
                                one row is replaced in every cell, and detection runs cell
                                by cell, so that a value never spills over a column.
                                Quoting is kept, and added where a tag needs it. A file
                                that does not parse is processed as plain text, with a
                                note.
                    markup    - character data, CDATA sections, comments and attribute
                                values are rewritten, nothing else. HTML and SVG are read
                                with the tolerance of HTML parsers (unquoted and valueless
                                attributes, script and style contents); other markup must
                                be well-formed XML and is read with the same tolerance,
                                with a note, when it is not. Character data split by
                                inline elements (<b>, <span>, <a>...) is read as one unit;
                                script and style contents are read too, since a name in a
                                string literal is still a name. Namespace declarations,
                                xml: attributes and, in HTML and SVG, the presentational
                                attributes (class, style, width, d...) are skipped; tags
                                written inside URI attributes (href, src...) are
                                percent-encoded. Processing instructions and the document
                                type declaration stay as written.
                The other formats are delegated to ooxml_anonymizer and pdf_anonymizer.

                Restoration of text and delimited files gives back the original bytes when
                every tag verifies, with one exception: a Windows-1252 file whose protected
                form holds only ASCII is restored in UTF-8, since nothing tells its
                encoding any more (the reports note it). In markup, everything outside the
                replaced values keeps its exact bytes; a restored value is written with
                plain characters, and a value that spanned several inline elements comes
                back in the first of them. A tag never forms "]]>" with the text after it:
                in character data its last bracket is written &#93;, and a CDATA section is
                cut in two after it, then joined again by the restoration.

            THREAD SAFETY
                One object processes one document at a time.
        !*/
    public:
        document_anonymizer(
            const pseudonymizer& p,
            pii_detector& detector
        );
        /*!
            requires
                - p and detector outlive this object.
        !*/

        std::string anonymize(
            const std::string& bytes,
            document_format format,
            document_report& report,
            const std::string& name_hint = ""
        );
        /*!
            ensures
                - Returns the protected document, in the same format and, for text
                  formats, in the same encoding.
                - name_hint, the file name, helps to find the delimiter of a delimited
                  file.
                - report receives the operation, the format, the key fingerprint, the
                  counts by type, the removed content and the notes; it never receives
                  original values.
            throws
                - error if format is unknown or legacy_office, and error, zip_error or
                  pdf_error if the document cannot be read.
        !*/

        std::string restore(
            const std::string& bytes,
            document_format format,
            document_report& report,
            const std::string& name_hint = ""
        );
        /*!
            ensures
                - Returns the document with every verified tag replaced by its original
                  value, dates shifted back where they were shifted in place.
                - report receives the counts of restored and rejected tags.
            throws
                - error if format is unknown or legacy_office, and error, zip_error or
                  pdf_error if the document cannot be read.
        !*/

        std::string anonymize_text(
            const std::string& utf8_text,
            document_report& report
        );
        /*!
            ensures
                - Protects a UTF-8 string as plain text, such as a prompt about to be sent
                  to a language model. The answer can be restored with
                  pseudonymizer::restore().
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DOCUMENT_ANONYMIZER_ABSTRACT_H_

