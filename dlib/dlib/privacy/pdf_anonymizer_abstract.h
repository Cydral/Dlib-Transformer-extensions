// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PDF_ANONYMIZER_ABSTRACT_H_
#ifdef DLIB_PDF_ANONYMIZER_ABSTRACT_H_

#include <string>
#include "pii_detector_abstract.h"
#include "pseudonymizer_abstract.h"
#include "document_report_abstract.h"
#include "../documents/pdf_text_abstract.h"

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
                PDF is written anew from the text alone, in the spirit of content disarm
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
                    4. A new PDF is written with the same pages, each text segment at its
                       original position, size and direction (see write_pdf_text()).

                Everything that is not text is removed: images, vector graphics,
                annotations, attachments, bookmarks, scripts, metadata and earlier
                revisions. Text that cannot be decoded is not copied. The report lists
                what was removed, without values.

            RESTORATION
                restore() reads a PDF produced by anonymize(), or any PDF containing
                tags, replaces the verified tags and writes the result the same way. Each
                segment of the restored PDF holds its original text at its original
                place, in the standard Helvetica font: the original fonts, pictures and
                graphics are not restored, and characters outside the repertoire of the
                writer (WinAnsi and Latin Extended-A), such as Greek, Cyrillic or CJK
                text, are written as '?'.
        !*/
    public:
        pdf_anonymizer(
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

