// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OOXML_ANONYMIZER_ABSTRACT_H_
#ifdef DLIB_OOXML_ANONYMIZER_ABSTRACT_H_

#include <string>
#include "../documents/document_image_abstract.h"
#include "pii_detector_abstract.h"
#include "pseudonymizer_abstract.h"
#include "document_report_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class ooxml_anonymizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Anonymization and restoration of Word, Excel and PowerPoint packages
                (DOCX, XLSX, PPTX and their template and macro-enabled variants).

                The output is a rebuilt package, in the sense of content disarm and
                reconstruction: every XML part is re-emitted with only its personal data
                and its pictures changed, unreferenced parts are dropped, and the archive
                is written anew with neutral timestamps. The input is never modified. A
                part that is not well-formed XML could hold personal data that cannot be
                read: the package is then refused and nothing is written.

                WHAT IS TREATED
                    Text      - paragraphs are read across their runs, so a name split by
                                formatting ("Jean " in one run, "Dupont" in the next) is
                                found and replaced as one value; the tag goes to the run of
                                the first character. A value is cut at the boundary of a
                                hyperlink, field, revision or content control inside the
                                paragraph, so that each part keeps its own tag. Deleted
                                revisions, field codes, comments, footnotes, headers and
                                footers, notes pages, text boxes, content controls (their
                                dates, list items and placeholder text), legacy form fields
                                (default values, drop-down entries), mail merge sources,
                                shared strings, inline strings, formula results, cell
                                comments, threaded comments, charts and drawings that are
                                kept as text, tables, pivot caches and pivot tables,
                                external link caches and revision logs are treated the same
                                way.
                    Formulas  - the string literals of cell formulas, validation lists,
                                conditional formats, filters and defined names are
                                anonymized; cell references stay as written.
                    Sheets    - a sheet whose name holds personal data is renamed "SheetN",
                                the formulas and defined names that refer to it follow, and
                                its original name is kept as a tag in a custom property, so
                                that restoration gives it back.
                    Structure - table header cells and left-hand label cells type the
                                cells they govern ("Nom" | "DUPONT"); spreadsheet columns
                                are typed from their header row, and identifiers stored as
                                numbers in typed columns become tagged text cells.
                    Metadata  - authors of the document, of revisions and of comments,
                                comment author lists, people and person lists, the user
                                who shares a workbook and the folder it was saved in,
                                company, manager, title, subject, keywords and description,
                                the values of custom properties, hyperlink lists of the
                                extended properties, external hyperlink targets
                                (percent-encoded tags) and local file paths. The lists of
                                parts of the extended properties (titles of slides, names
                                of sheets, fonts and themes, which Office writes again
                                when it saves) are removed.
                    Dates     - dates held in typed fields (date-typed spreadsheet cells,
                                document and revision timestamps), which cannot carry a
                                tag, are shifted in place unless dates are kept; a custom
                                property "dlib.privacy" records it so that restoration
                                shifts them back. Under date_policy::encrypt, dates written
                                in text become opaque tags and typed dates are shifted.
                    Pictures  - pictures, charts kept as objects, SmartArt diagrams, ink,
                                OLE objects, ActiveX controls, imported content (altChunk),
                                video, audio, thumbnails, printer settings and VBA projects
                                are removed with their relationships; picture fills become
                                no fill; a picture still referenced from a place that
                                cannot lose it is replaced by a blank 1x1 image of the same
                                format.

            REPORT
                The report counts the values replaced in the parts written: those of a
                part removed with what referred to it, such as the data of a chart, are
                not counted.

            RESTORATION
                restore() replaces the verified tags found in the same places, gives
                renamed sheets their names back and shifts dates back when the marker is
                present. Removed pictures and objects cannot be restored. Text boxes and
                cells keep the text form of what they hold: an identifier stored as a
                number comes back as a text cell, and a value whose runs had different
                formatting comes back in the formatting of its first run.
        !*/
    public:
        ooxml_anonymizer(
            const pseudonymizer& p,
            pii_detector* detector,
            document_report& report
        );
        /*!
            requires
                - p, report and, when it is not null, *detector outlive this object.
            ensures
                - restore() does not use the detector, which may be null for it.
        !*/

        void set_image_describer(
            const document_image_describer& describer
        );
        /*!
            ensures
                - anonymize() then gives describer each image part shown by a picture of
                  a Word document (w:drawing, w:pict, or their mc:AlternateContent, in a
                  run) or of a slide (p:pic, or its mc:AlternateContent), once, and puts
                  the text it returns in the place of the picture: a run text in a Word
                  document, a gray text box at the position of the picture on a slide.
                  The text is written as it is returned. A picture for which describer
                  returns an empty string, and the pictures of workbooks, are removed as
                  before. An empty function, the default, removes every picture.
                - restore() never uses it.
        !*/

        void set_package_limits(
            const zip_limits& limits
        );
        /*!
            ensures
                - #package_limits() == limits: the bounds within which the parts of a package
                  are decompressed (zip_archive_reader), the defaults of zip_limits until
                  then.  A caller that knows the size of the files it takes keeps them in
                  proportion, since the memory of the processing grows with the size of the
                  parts decompressed.
        !*/

        const zip_limits& package_limits(
        ) const;

        std::string anonymize(const std::string& package_bytes);
        /*!
            ensures
                - Returns the bytes of the protected package.
            throws
                - error if the detector is null or a part is not well-formed XML, and
                  zip_error or error if the package cannot be read.
        !*/

        std::string restore(const std::string& package_bytes);
        /*!
            ensures
                - Returns the bytes of the restored package.
            throws
                - error if a part is not well-formed XML, and zip_error or error if the
                  package cannot be read.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OOXML_ANONYMIZER_ABSTRACT_H_

