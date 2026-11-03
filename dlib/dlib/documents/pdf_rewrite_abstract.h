// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PDF_REWRITE_ABSTRACT_H_
#ifdef DLIB_PDF_REWRITE_ABSTRACT_H_

#include <cstddef>
#include <string>
#include <vector>
#include "document_image_abstract.h"
#include "pdf_text_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pdf_text_replacement
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Text to draw in the place of a range of a segment of layout_pdf_text():
                bytes begin to end of the text of segment number segment of page number
                page.
        !*/
        std::size_t page = 0;
        std::size_t segment = 0;
        std::size_t begin = 0, end = 0;
        std::string text;   // UTF-8
    };

    struct pdf_rewrite_report
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                What a rewrite left out or changed.
        !*/
        std::size_t images = 0;             // pictures replaced by gray boxes
        std::size_t annotations = 0;        // annotations left out: comments, links, attachments...
        std::size_t drawn_annotations = 0;  // appearances of form fields, free texts and stamps drawn
                                            // into their page
        std::size_t unencodable = 0;        // characters drawn as '?'
        std::size_t described_images = 0;   // gray boxes that received the text of the describer
        std::size_t scanned_pages = 0;      // scans whose recognized text is drawn visible
    };

    std::string rewrite_pdf_text(
        const std::string& pdf_bytes,
        const std::vector<pdf_text_replacement>& replacements,
        pdf_rewrite_report* report = nullptr,
        const document_image_describer& describer = document_image_describer(),
        const std::vector<std::pair<std::string, std::string>>& info =
            std::vector<std::pair<std::string, std::string>>()
    );
    /*!
        requires
            - Each replacement refers to the segments of
              layout_pdf_text(read_pdf_text(pdf_bytes)).
        ensures
            - Returns a new PDF that keeps the pages of pdf_bytes as they look: page boxes,
              rotation, vector graphics, colors, transparency, optional content and the
              order in which everything is painted. Every text showing operator is
              replaced, in place, by the same text drawn with the standard font of the
              same style (Helvetica, Times or Courier; regular, bold, italic), each run
              at the position, size, direction, rise and color of the original and
              condensed or expanded to its width, so that the text of a page reads where
              it was. The text of a replacement is drawn over the extent of the range it
              replaces, across the runs and the lines of its segment, and the rest of
              those runs where they were.
            - Text drawn twice at the same place (fake bold) takes the replacements of
              the text it duplicates.
            - Only the text read by read_pdf_text() is drawn: a glyph that could not be
              decoded is drawn as U+FFFD and counted in report->unencodable, so no
              original glyph and no original font reaches the output. WinAnsi
              characters and the Latin Extended-A letters are drawn exactly, with a
              ToUnicode map; other characters are drawn with their ASCII stand-ins or
              as '?'.
            - Pictures, inline or not, are replaced by uniform gray boxes of the same
              place and size.  When describer is given, it receives each picture object
              (not the inline ones) once, as a file: a JPEG as it is stored, the other
              pictures of 8 bits per component in gray, RGB or CMYK, indexed or not, as
              PNG, with location "page <n>" of its first page; JPEG 2000, JBIG2, CCITT
              and mask pictures are not given.  The text it returns is written over each
              gray box of that picture larger than 24 x 10 points, in centered lines of
              Helvetica as large as fit (10 points at most, 3 at least, the last line
              shortened with "..." when the text does not fit), and counted in
              report->described_images. The appearances of form fields, free texts and stamps
              are drawn into their page, the value of a text field without appearance
              too; every annotation is left out, with the form fields, document
              information, XMP metadata, bookmarks, named destinations, embedded files,
              scripts, actions, structure tree, page thumbnails, comments of content
              streams and the alternate texts of marked content.
            - A page whose text is all invisible over pictures (a scan with its
              recognized text) has its text drawn visible, since its pictures become
              gray boxes.
            - The catalog carries /DlibPrivacyLayout true. Rewriting the output again
              keeps it as it is, apart from the replacements.
            - The new document information dictionary holds info, each value written as
              a text string in UTF-16 (the keys keep their letters and digits), and is
              left out when info is empty.
            - When report is not null, *report tells what was replaced or left out.
        throws
            - pdf_error if the file cannot be read (see read_pdf_text()).
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PDF_REWRITE_ABSTRACT_H_
