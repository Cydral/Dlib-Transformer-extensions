// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PDF_TEXT_ABSTRACT_H_
#ifdef DLIB_PDF_TEXT_ABSTRACT_H_

#include <cstddef>
#include <string>
#include <vector>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pdf_error : public error
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Thrown when a PDF cannot be read: damaged beyond repair, encrypted, or
                using a construct this reader does not handle.
        !*/
    };

    struct pdf_text_span
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One run of text as drawn by a text showing operator, in default user
                space: (x, y) is the baseline origin in points, width the advance along
                the baseline, font_size the rendered size and angle the direction of the
                baseline in degrees, counterclockwise from the x axis.
        !*/
        std::string text;           // UTF-8, U+FFFD for glyphs that could not be decoded
        double x = 0, y = 0;
        double width = 0;
        double font_size = 0;
        double angle = 0;
    };

    struct pdf_page_text
    {
        double x0 = 0, y0 = 0, x1 = 612, y1 = 792;   // MediaBox
        int rotate = 0;                               // /Rotate, 0, 90, 180 or 270
        std::vector<pdf_text_span> spans;
        std::size_t images = 0;                       // images drawn on the page
        std::size_t annotations = 0;                  // annotations whose text is not kept
    };

    struct pdf_text_document
    {
        std::vector<pdf_page_text> pages;
        std::size_t undecodable_glyphs = 0;
        bool produced_by_dlib = false;
        std::vector<std::string> notes;     // reading problems met, by page
        std::vector<std::string> features;  // non-text content present in the file
        /*!
            features names what the file holds besides page text, without any value:
            "document information (Author, Title)", "XMP metadata", "bookmarks",
            "embedded files", "JavaScript", "form fields", "structure tree"...
        !*/
    };

    pdf_text_document read_pdf_text(
        const std::string& pdf_bytes
    );
    /*!
        ensures
            - Extracts the positioned text of every page.
            - Cross-reference tables and streams, hybrid files, incremental updates and
              object streams are read, with a reconstruction by scanning when the
              cross-reference data is damaged. Streams may use the Flate (with PNG and
              TIFF predictors), LZW, ASCIIHex, ASCII85 and RunLength filters.
            - Text is decoded through ToUnicode maps, the standard simple encodings with
              their Differences (Adobe Glyph List names of the Latin repertoire),
              Type 3 fonts, and composite fonts with Identity, UCS-2 or UTF-16 encodings
              or embedded CMaps. A glyph whose code has no known character, such as a
              code renamed by Differences to a private glyph name, is counted in
              undecodable_glyphs and read as U+FFFD; so is text drawn with a font
              missing from the resources. Form XObjects are followed; images are
              counted and skipped.
            - The appearances of form fields, free text annotations and stamps are read
              as page text (the visible value of a filled form is kept); a text field
              without appearance contributes its value. Other annotations, such as
              comments, links and attachments, are counted in annotations.
            - A Flate stream that is damaged or truncated is read up to the damage, and
              notes says so. notes also reports text drawn with a missing font and
              shown strings read in part.
            - Numbers are read and written independently of the C locale.
            - Bounds: a decoded stream holds at most 64 MiB, all the streams of a
              document 1 GiB (streams without filter included), an array or a
              dictionary 2^20 elements, a document 2^23 parsed objects and 2^27 content
              operators, and a shown string is read up to 1 MiB. Chains of indirect
              objects and nested forms are depth limited and cycles are detected.
        throws
            - pdf_error if the file is encrypted, cannot be parsed at all or exceeds one
              of the document bounds.
    !*/

    struct pdf_text_segment
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A piece of a text line: spans joined along the baseline, with spaces
                inserted where the gap between two spans shows a word boundary. A wide
                gap starts a new segment, which keeps table columns and multi-column
                layouts apart. Segments of the same line share their line index.
        !*/
        std::string text;
        double x = 0, y = 0;
        double width = 0;
        double font_size = 0;
        double angle = 0;
        std::size_t line = 0;
    };

    std::vector<std::vector<pdf_text_segment>> layout_pdf_text(
        const pdf_text_document& doc
    );
    /*!
        ensures
            - Returns, page by page, the segments in reading order: for each baseline
              direction, upright text first, lines from top to bottom and segments
              along the baseline.
            - Text drawn twice at the same place (fake bold, shadows) is kept once.
            - For a document produced by write_pdf_text(), the runs written for one
              segment are joined back into that segment.
    !*/

    std::string write_pdf_text(
        const pdf_text_document& layout_source,
        const std::vector<std::vector<pdf_text_segment>>& pages,
        std::size_t* unencodable = nullptr
    );
    /*!
        requires
            - pages.size() == layout_source.pages.size()
        ensures
            - Returns a new PDF holding only the given text: same page count, page boxes
              and rotation, each segment drawn at its original position, size and
              direction with the standard Helvetica font, horizontally condensed when
              needed so that it does not run into the next segment of its line.
            - WinAnsi characters and the Latin Extended-A letters (Polish, Czech,
              Hungarian, Romanian, Turkish, Baltic names) are written exactly, with a
              ToUnicode map so that text extraction returns them. Ligatures, arrows and
              typographic spaces are written with their usual ASCII equivalents; any
              other character is written as '?' and counted in *unencodable when it is
              not null.
            - Nothing else is carried over: no images, vector graphics, annotations,
              form fields, bookmarks, attachments, scripts or metadata. The catalog
              carries /DlibPrivacy true so that layout_pdf_text() recognizes the file.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PDF_TEXT_ABSTRACT_H_
