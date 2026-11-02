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

    std::vector<pii_span> combine_pii_recognizers(
        const std::string& text,
        const std::vector<pii_span>& model_spans,
        const std::vector<pii_span>& tagger_spans,
        double min_confidence = 0.5,
        double tagger_min_confidence = 0.9
    );
    /*!
        ensures
            - Combines the spans of a span model and of a sequence tagger over the same
              text, the span model leading: its spans of at least min_confidence are kept,
              and the tagger, whose word features also fire on capitalized words that are
              not personal data, only fills their gaps, with its spans of at least
              tagger_min_confidence where the span model kept nothing, and with its less
              confident name parts (at least min_confidence) that complete a name already
              found, so that the parts of a name are replaced together.
            - A span the model reads as an organisation with a confidence below 0.9 gives
              way to the name parts the tagger reads over all its words with more
              confidence ("Lucie Rey" alone on its line), and a single word the model
              reads as a town or an organisation gives way to the given name the tagger
              reads there, with a confidence of 0.9 at least and close to that of the
              model or above, when the tagger also reads the family name that follows it
              ("Malo Cozic", "Nolwenn Le Floc'h").
            - Returns the spans sorted by their beginning.
    !*/

    void attach_pii_recognizers(
        pii_detector& detector,
        std::shared_ptr<const pii_sequence_tagger> tagger,
        std::shared_ptr<const pii_span_recognizer> model,
        double min_confidence = 0.5,
        double tagger_min_confidence = 0.9
    );
    /*!
        ensures
            - Adds to detector the recognizers given, either of which may be null or
              empty: both through combine_pii_recognizers(), or the one given alone with
              min_confidence.
            - The detector keeps shared ownership of the models.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_RECOGNIZERS_ABSTRACT_H_
