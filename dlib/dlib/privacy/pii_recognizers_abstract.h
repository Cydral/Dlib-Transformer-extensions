// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PII_RECOGNIZERS_ABSTRACT_H_
#ifdef DLIB_PII_RECOGNIZERS_ABSTRACT_H_

#include <istream>
#include <memory>
#include <string>
#include <vector>
#include "pii_detector_abstract.h"
#include "pii_sequence_tagger_abstract.h"
#include "pii_span_model_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class pii_span_recognizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A trained span model of either size the anonymizers ship,
                pii_span_model_tiny or pii_span_model_small, read from its serialized
                form without knowing its size beforehand.

            THREAD SAFETY
                find() may be called from several threads (see pii_span_model).
        !*/
    public:
        pii_span_recognizer(
        );
        /*!
            ensures
                - #empty() == true
        !*/

        explicit pii_span_recognizer(
            std::istream& in
        );
        /*!
            ensures
                - Reads the model serialized at the current position of in.
            throws
                - serialization_error if in holds no tiny or small span model.
        !*/

        const std::string& size(
        ) const;
        /*!
            ensures
                - Returns "tiny", "small", or an empty string when empty().
        !*/

        bool empty(
        ) const;

        std::vector<pii_span> find(
            const std::string& text
        ) const;
        /*!
            ensures
                - Returns the spans of the model over text, or nothing when empty().
        !*/

        static std::string serialized_size(
            std::istream& in
        );
        /*!
            ensures
                - Returns the size of the span model serialized at the current position of
                  in, "tiny" or "small", or an empty string for another content. The
                  position of in is left where it was.
        !*/
    };

// ----------------------------------------------------------------------------------------

    enum class pii_leading_recognizer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Which of a span model and a sequence tagger leads where both read the same
                text (see combine_pii_recognizers()).
        !*/
        span_model,
        tagger
    };

    std::vector<pii_span> combine_pii_recognizers(
        const std::string& text,
        const std::vector<pii_span>& model_spans,
        const std::vector<pii_span>& tagger_spans,
        pii_leading_recognizer leading,
        double min_confidence = 0.5,
        double second_min_confidence = 0.9
    );
    /*!
        ensures
            - Combines the spans of a span model and of a sequence tagger over the same
              text. A span of the leading recognizer needs min_confidence and counts a
              tenth more than its confidence; a span of the other recognizer needs
              second_min_confidence and counts its confidence. The spans are taken from
              the one that counts most to the one that counts least, each kept when it
              overlaps none kept before; at equal counts the leading recognizer comes
              first.
            - On the spans labelled "SECRET" the span model leads whatever leading is: a
              secret is a run of characters, which the span model reads byte by byte and
              the tagger word by word. The secrets that the span model reads with
              min_confidence at least are kept before every other span, since a secret
              left readable costs more than a value given another type.
            - A town that a leading tagger reads with less than second_min_confidence is
              kept only where the span model reads a town or a postal code too, with a
              confidence of 0.2 at least: its context alone tells a town from any other
              capitalized word.
            - An account name written in lower-case letters alone ("jdupont") that a
              leading tagger reads with less than second_min_confidence in running text is
              kept only where the span model reads an account name too, with a confidence
              of 0.2 at least: it has the look of any word of a sentence. Where a layout
              writes a value (at the start of the text or of a line, after a tab, a colon,
              an equals sign, a semicolon or a vertical bar, or right after a comma or a
              '>'), and with a digit, a sign or a capital, it follows the general rule.
            - The less confident name parts of the other recognizer (at least
              min_confidence) that complete a name kept, separated from it by spaces or a
              hyphen only, are kept too, so that the parts of a name are replaced
              together.
            - Returns the spans sorted by their beginning; no two of them overlap when
              neither input holds overlapping spans.
    !*/

    pii_leading_recognizer default_leading_recognizer(
        const pii_span_recognizer& model
    );
    /*!
        ensures
            - Returns the recognizer that leads beside model in attach_pii_recognizers(),
              the one that scores higher alone on the evaluation sets of the anonymizers:
              the tagger beside a tiny span model, the span model when it is a small one.
    !*/

    void attach_pii_recognizers(
        pii_detector& detector,
        std::shared_ptr<const pii_sequence_tagger> tagger,
        std::shared_ptr<const pii_span_recognizer> model,
        pii_leading_recognizer leading,
        double min_confidence = 0.5,
        double second_min_confidence = 0.9
    );
    /*!
        ensures
            - Adds to detector the recognizers given, either of which may be null or
              empty: both through combine_pii_recognizers() with leading, or the one given
              alone with min_confidence.
            - The parts of a name and the organisations that the span model reads with a
              confidence of 0.2 at least, below the confidence it needs to be kept and
              where no span kept lies, also reach the detector, under the label "WEAK": no
              values themselves, they are the weak evidence that a capitalized word joined
              by a conjunction to a person found is another person ("Martin and Dupont").
            - The detector keeps shared ownership of the models.
    !*/

    void attach_pii_recognizers(
        pii_detector& detector,
        std::shared_ptr<const pii_sequence_tagger> tagger,
        std::shared_ptr<const pii_span_recognizer> model,
        double min_confidence = 0.5,
        double second_min_confidence = 0.9
    );
    /*!
        ensures
            - Same as above, with default_leading_recognizer(*model) as the leading
              recognizer.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_RECOGNIZERS_ABSTRACT_H_
