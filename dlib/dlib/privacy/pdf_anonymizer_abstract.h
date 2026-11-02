// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PDF_ANONYMIZER_ABSTRACT_H_
#ifdef DLIB_PDF_ANONYMIZER_ABSTRACT_H_

#include <string>
#include "../documents/document_image_abstract.h"
#include "pii_detector_abstract.h"
#include "pseudonymizer_abstract.h"
#include "document_report_abstract.h"
#include "../documents/pdf_text_abstract.h"
#include "../documents/pdf_rewrite_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pdf_anonymizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Anonymization and restoration of PDF documents by reconstruction.

                A PDF describes how its pages are drawn. Personal data can sit in page
                text, in form fields, in annotations, in metadata, in hidden or clipped
                text, in images and in incremental updates that keep earlier versions.
                An edit made in place would leave those copies behind, so the protected
                PDF is written anew from what was read, in the spirit of content disarm
                and reconstruction:

                    1. The positioned text of every page is extracted (see
                       read_pdf_text()), including the visible values of filled forms.
                    2. Lines are rebuilt into text units. Lines of a paragraph are joined,
                       with a space where a line wraps and a line break where it ends, so
                       that a name broken over two lines is still found; the cells of a
                       table row become separate units, typed by the label on their left
                       ("Nom :" | "DUPONT") or by the header of their column.
                    3. Detection runs in two passes over the whole document, as for the
                       other formats, and each value is replaced by its tag. A value read
                       across a line end is cut there: each line keeps a tag for its part,
                       and the report counts one tag for each part.
                    4. A new PDF is written. By default it keeps the pages as they look
                       (see rewrite_pdf_text()): vector graphics, colors, tables, page
                       boxes and the order of painting stay, every text run is drawn
                       again in place with the standard font of its style, each tag over
                       the extent of the value it replaces, and pictures become gray
                       boxes. With keep_layout set to false, the PDF is rebuilt as text
                       only, each text segment at its original position, size and
                       direction in Helvetica (see write_pdf_text()).

                In both modes, only text that was read and checked is written: the
                original fonts, pictures, annotations, attachments, bookmarks, scripts,
                metadata, alternate texts and earlier revisions are left out, and text
                that cannot be decoded is not copied. The text-only mode also leaves out
                the vector graphics. The report lists what was removed, without values.

            RESTORATION
                restore() reads a PDF produced by anonymize(), or any PDF containing
                tags, and replaces the verified tags. A PDF protected with its layout
                is restored with its layout: each original value is drawn where its tag
                was, in the font of its text; the pictures replaced by gray boxes are
                not restored. A text-only PDF is restored as text only, each segment at
                its original place in Helvetica. Characters outside the repertoire of the
                standard fonts (WinAnsi and Latin Extended-A), such as Greek, Cyrillic or
                CJK text, are written as '?'. A PDF that holds no tag and was not written
                by anonymize() is returned unchanged.
        !*/
    public:
        pdf_anonymizer(
            const pseudonymizer& p,
            pii_detector* detector,
            document_report& report,
            bool keep_layout = true
        );
        /*!
            requires
                - p, report and, when it is not null, *detector outlive this object.
            ensures
                - anonymize() keeps the pages of the document when keep_layout is true,
                  and rebuilds it as text only otherwise. A document already written by
                  one mode is protected again in that mode.
                - restore() does not use the detector, which may be null for it.
        !*/

        void set_image_describer(
            const document_image_describer& describer
        );
        /*!
            ensures
                - anonymize() then gives each picture of a PDF that keeps its pages to
                  describer, and writes the text it returns over the gray box of the
                  picture (see rewrite_pdf_text()). An empty function, the default,
                  leaves the gray boxes empty. restore() never uses it.
        !*/

        std::string anonymize(const std::string& pdf_bytes);
        /*!
            ensures
                - Returns the bytes of the protected PDF.
            throws
                - error if the detector is null.
                - pdf_error if the PDF cannot be read, in particular if it is encrypted.
        !*/

        std::string restore(const std::string& pdf_bytes);
        /*!
            ensures
                - Returns the bytes of the restored PDF.
            throws
                - pdf_error if the PDF cannot be read, in particular if it is encrypted.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PDF_ANONYMIZER_ABSTRACT_H_
