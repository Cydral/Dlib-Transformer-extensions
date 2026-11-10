// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DOCUMENT_ANONYMIZER_ABSTRACT_H_
#ifdef DLIB_DOCUMENT_ANONYMIZER_ABSTRACT_H_

#include <string>
#include <vector>
#include "../documents/document_image_abstract.h"
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

    struct pii_imposed_column
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A column of the delimited files whose values are all personal data of one
                type, the column of the names of the clients or of their e-mail addresses in
                an export for instance (document_anonymizer::set_imposed_columns()).

                name - the name of the column in the header of a file, compared without
                       regard to case and accents, the spaces at its ends left out
                       ("Prénom" names the columns headed "PRENOM" and " prénom ")
                type - the type of every value of the column
        !*/
        std::string name;
        pii_type type = pii_type::none;
    };

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
                                Quoting is kept, and added where a value needs it (a
                                delimiter, a line break); a cell written without quotes
                                keeps the quotes it holds as they are. A file that does
                                not parse is processed as plain text, with a note. The
                                columns that set_imposed_columns() names take
                                substitutes, without detection.
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

        void set_pdf_layout(
            bool keep
        );
        /*!
            ensures
                - #pdf_layout() == keep
        !*/

        bool pdf_layout(
        ) const;
        /*!
            ensures
                - Returns true, the default, when a PDF keeps its pages: they are written
                  again with their graphics, the text drawn with standard fonts and the
                  pictures replaced (pdf_anonymizer). Returns false when a PDF is rebuilt
                  as text only.
        !*/

        void set_package_limits(
            const zip_limits& limits
        );
        /*!
            ensures
                - #package_limits() == limits: the bounds within which the parts of Word,
                  Excel and PowerPoint documents are decompressed
                  (ooxml_anonymizer::set_package_limits()).
        !*/

        const zip_limits& package_limits(
        ) const;

        void set_untagged_properties(
            bool untagged
        );
        /*!
            ensures
                - #untagged_properties() == untagged
        !*/

        bool untagged_properties(
        ) const;
        /*!
            ensures
                - Returns false, the default, when the properties of the documents (title,
                  subject, authors, keywords, description, company, manager and the other
                  fields of the core, extended and custom properties of Word, Excel and
                  PowerPoint, the document information of a PDF) take tags like the text.
                - Returns true when anonymize() writes them with substitutes instead
                  (pseudonymizer::substitute()), which the windows of properties show as
                  plain values, and lists each one with its tag in report.substitutions;
                  restore() then gives back the substitutes that the resolver of
                  set_substitute_resolver() knows. A PDF otherwise loses its document
                  information, which the protected file then carries.
        !*/

        void set_substitute_resolver(
            const pseudonymizer::substitute_resolver& resolve
        );
        /*!
            ensures
                - restore() and restore_name() give resolve the candidates of the places
                  that hold substitutes (pseudonymizer::find_substitutes()): the properties
                  when untagged_properties() is true, and the names. An empty function, the
                  default, restores no substitute.
        !*/

        void set_untagged_text(
            bool untagged
        );
        /*!
            ensures
                - #untagged_text() == untagged
        !*/

        bool untagged_text(
        ) const;
        /*!
            ensures
                - Returns false, the default, when anonymize() and anonymize_text() replace
                  each value of the text of a document by its tag.
                - Returns true when they write its substitute instead (pseudonymizer::
                  substitute()), a protected value without the brackets, the type and the
                  check of its tag: in the text formats, the cells and the markup, in the
                  runs, cells, formulas and links of Word, Excel and PowerPoint, in the text
                  of the pages of a PDF. Each substitute is listed with its tag in
                  report.substitutions, text set to true, once for each place it is written
                  in, so that the caller can keep them: a substitute carries no check, and
                  only a resolver that knows it gives its value back
                  (set_text_substitute_resolver()). The names of the sheets of a workbook
                  and the tags that a document keeps for them in its custom properties are
                  written as without untagged text.
        !*/

        void set_text_substitute_resolver(
            const pseudonymizer::substitute_resolver& resolve
        );
        /*!
            ensures
                - restore() gives resolve the candidates of the text of the document, in
                  every place where anonymize() writes values (pseudonymizer::
                  find_substitutes()), besides its tags: the substitutes that resolve knows,
                  outside the tags and the texts of the pictures, take their original values
                  back, counted in report.substitutes_restored.
                - anonymize() and anonymize_text() leave as they are the substitutes that
                  resolve knows, an untagged document protected again.
                - An empty function, the default, restores no substitute of the text.
        !*/

        void set_imposed_columns(
            const std::vector<pii_imposed_column>& columns
        );
        /*!
            ensures
                - #imposed_columns() == columns, without those of type none, those whose
                  name holds nothing but spaces and the later ones of a name given twice.
        !*/

        const std::vector<pii_imposed_column>& imposed_columns(
        ) const;
        /*!
            ensures
                - Returns the columns of the delimited files whose values are replaced whole,
                  without detection, empty by default:
                    - The header of a file is the first of its first 16 rows that holds the
                      name of an imposed column in a cell. Its cells stay as they are, and the
                      labels among them type their columns as in any header.
                    - anonymize() writes the substitute of each value below the name of an
                      imposed column (pseudonymizer::substitute() for the type of the column),
                      with or without untagged text, the spaces and line breaks at its ends
                      kept and an empty cell left empty. The tag of a value stands for that
                      value, and a substitute that the text resolver knows for the type of
                      the column (set_text_substitute_resolver()), a file protected before,
                      stays as it is. Each value is counted in report.entities and listed with
                      its tag in report.substitutions, text set to true, once: a substitute
                      carries no check, and only a resolver that knows it gives its value back.
                      A civility, a sex or a nationality takes another substitute at each
                      place, listed each time. A date stays in clear under date_policy::keep.
                    - The values are not read by the detector: each is recorded once as a
                      value of the type of its column (pii_detector::learn_value()), so that
                      the other cells of the file, a comment that names a client for
                      instance, give it up too. The whole rows that the learning pass reads
                      leave them out.
                    - restore() gives each value below the name of an imposed column, whole, to
                      the text resolver: the first tag returned that decodes to a value of the
                      type of the column takes its place, counted in
                      report.substitutes_restored; the other values are restored as in any
                      cell.
                    - A file is found delimited by its header. In a file named .csv, .tsv or
                      .tab, a title may stand above the header, and a first line "sep=;"
                      written by Excel gives the delimiter; such a file may also hold one
                      column alone, its name alone on its line. A text given as
                      document_format::text, or under another name, is read as delimited
                      when its header holds the delimiter and the two records that follow it
                      hold it as many times.
                    - The values of the sheets of a workbook and of the other formats are
                      detected as before.
        !*/

        std::string anonymize_name(
            const std::string& name,
            document_report& report
        );
        /*!
            ensures
                - Returns name, a file name without its extension or any other short text
                  that cannot hold a tag, with each value found replaced by its substitute,
                  the underscores of the name read as spaces. The detector keeps what it
                  learned from the document processed before, so that a name it established
                  there is found in the name of its file too. A person or an organization
                  found with the common words that open it, "Contrat" in "Contrat Jean
                  Dupont", leaves those words as they are: a name read without a sentence
                  around it takes them in. The substitutes that the resolver of
                  set_substitute_resolver() knows, written by an earlier call, stay as they
                  are.
                - Each substitute is counted in report.entities and listed with its tag in
                  report.substitutions.
        !*/

        std::string restore_name(
            const std::string& name,
            document_report& report
        ) const;
        /*!
            ensures
                - Returns name with the substitutes that the resolver of
                  set_substitute_resolver() knows replaced by their original values, counted
                  in report.substitutes_restored.
        !*/

        void set_picture_text_finder(
            const document_picture_text_finder& find
        );
        /*!
            ensures
                - restore() then removes from every text the ranges that find gives, the texts
                  that a describer wrote in the place of pictures, with the tags they hold,
                  and counts them in report.pictures_removed: in the text formats, the cells
                  and the markup; in Word documents, from their runs; on slides, the gray
                  text box of the picture with its text; in PDF documents, the text written
                  over the gray box, the box staying. A restoration made without the
                  original thus leaves no trace of a picture it cannot give back.
                - An empty function, the default, removes nothing.
        !*/

        void set_image_describer(
            const document_image_describer& describer
        );
        /*!
            ensures
                - anonymize() then gives describer the pictures of the documents and
                  writes the text it returns in their place, as it is returned (a
                  describer whose text may hold personal data protects it first, with
                  anonymize_text() of another document_anonymizer and its own detector):
                    - Word documents: a picture of a run (w:drawing, w:pict, or their
                      mc:AlternateContent) becomes a run text;
                    - PowerPoint presentations: a picture (p:pic, or its
                      mc:AlternateContent) becomes a gray text box at its position;
                    - PDF documents that keep their pages: the text is written over the
                      gray box of the picture (pdf_anonymizer::set_image_describer()).
                  Each image part, or image object of a PDF, is given once. The pictures
                  of Excel workbooks, the pictures for which describer returns an empty
                  string and those of the other formats are removed as before.
                - An empty function, the default, removes the pictures without text.
                  restore() never uses it.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DOCUMENT_ANONYMIZER_ABSTRACT_H_

