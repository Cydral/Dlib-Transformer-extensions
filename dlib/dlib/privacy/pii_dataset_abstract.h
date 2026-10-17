// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PII_DATASET_ABSTRACT_H_
#ifdef DLIB_PII_DATASET_ABSTRACT_H_

#include <string>
#include <vector>
#include "pii_types_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pii_example
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One annotated text of a training or evaluation set: UTF-8 bytes and the
                spans of personal data they hold, sorted by position and not overlapping.
                Each span carries a label: one of pii_fine_labels() for the categories
                the learned recognizers are trained on, or a tag keyword (EMAIL, PHONE,
                DATE, IBAN...) for the categories the validated scanners already find.
                The second kind is kept so that a whole detection system can be scored
                against a complete annotation, and is ignored when a model is trained.
        !*/
        std::string text;
        std::vector<pii_span> spans;
    };

    pii_example parse_pii_example(
        const std::string& line
    );
    /*!
        ensures
            - Parses one line of the dataset format:
                  text <TAB> spans
              where text escapes the backslash, tab, line feed and carriage return as
              \\, \t, \n and \r, and a '#' that opens the text as \#, and spans is a
              comma separated list of begin:end:LABEL items, byte offsets into the
              unescaped text. An example without personal data has an empty span list
              (the tab may be omitted). A final carriage return is ignored.
            - The type of every span is set from its label with pii_type_from_label();
              the spans are sorted by position.
        throws
            - error if the line is malformed: a bad escape, more than one raw tab, a
              span outside the text, not on UTF-8 character boundaries, crossing a tab or
              a line break of the text, overlapping another span, or whose label is
              neither a fine label nor a tag keyword.
    !*/

    std::string format_pii_example(
        const pii_example& example
    );
    /*!
        requires
            - the spans of example are sorted, non-overlapping and inside example.text
        ensures
            - Returns the line parse_pii_example() reads back as example, without its
              final line feed. A text opening with '#' is written \# so that the line is
              not read as a comment.
    !*/

    std::vector<pii_example> load_pii_dataset(
        const std::string& filename
    );
    /*!
        ensures
            - Reads a dataset file: one example per line, lines starting with '#' and
              empty lines ignored, a UTF-8 byte order mark at the start of the file
              skipped.
        throws
            - error naming the file and the line if a line is malformed or the file
              cannot be read.
    !*/

    void save_pii_dataset(
        const std::vector<pii_example>& examples,
        const std::string& filename
    );
    /*!
        ensures
            - Writes the examples in the format read by load_pii_dataset().
        throws
            - error if the file cannot be written.
    !*/

// ----------------------------------------------------------------------------------------

    unsigned long pii_bio_class_count(
    );
    /*!
        ensures
            - Returns 1 + 2 * pii_fine_labels().size(): the class O, then for each fine
              label k the classes B (first unit of a span, 1 + 2k) and I (following units,
              2 + 2k). Both learned recognizers label their units with these classes.
    !*/

    unsigned long pii_bio_class(
        const std::string& label,
        bool inside
    );
    /*!
        ensures
            - Returns the B class of label, or its I class when inside is true; 0 (the O
              class) if label is not a fine label.
    !*/

    std::string pii_bio_class_label(
        unsigned long cls
    );
    /*!
        ensures
            - Returns the fine label of a B or I class, "" for the O class.
    !*/

    bool pii_bio_class_inside(
        unsigned long cls
    );
    /*!
        ensures
            - Returns true if cls is an I class.
    !*/

    bool pii_bio_transition_allowed(
        unsigned long previous,
        unsigned long cls
    );
    /*!
        ensures
            - Returns false only for an I class that does not continue a span of the same
              label: I-X may follow B-X or I-X and nothing else. Both recognizers decode
              under this constraint, so that every span they report starts with a B.
    !*/

    std::vector<unsigned long> pii_spans_to_bio(
        const std::vector<std::pair<std::size_t, std::size_t>>& units,
        const std::vector<pii_span>& spans,
        std::size_t* misaligned = nullptr
    );
    /*!
        requires
            - units are the byte ranges of the units being labelled (bytes, words),
              sorted and non-overlapping
            - spans do not overlap; they need not be sorted
        ensures
            - Returns one class per unit: the B class of the label of the span the unit
              starts, the I class of the span it continues, O otherwise. Spans whose label
              is not a fine label are ignored.
            - A unit belongs to a span when it overlaps it, so that a unit only partly
              covered by a span takes its class. A span that does not start and end on
              unit boundaries is counted in *misaligned when misaligned is not null.
    !*/

    std::vector<pii_span> pii_bio_to_spans(
        const std::vector<std::pair<std::size_t, std::size_t>>& units,
        const std::vector<unsigned long>& classes,
        const std::vector<double>& confidences = std::vector<double>()
    );
    /*!
        requires
            - units.size() == classes.size()
            - confidences is empty or has one value per unit
        ensures
            - Returns the spans encoded by classes: a B unit followed by the I units of
              the same label, from the start of its first unit to the end of its last. An
              I unit that does not continue a span starts one.
            - The confidence of a span is the mean of the confidences of its units, 1 when
              confidences is empty.
    !*/

// ----------------------------------------------------------------------------------------

    std::vector<pii_span> coarsen_pii_spans(
        const std::string& text,
        const std::vector<pii_span>& spans
    );
    /*!
        requires
            - spans are sorted, do not overlap and lie inside text
        ensures
            - Returns the spans as the pseudonymizer replaces them: each label becomes the
              keyword of its type (GIVEN_NAME and FAMILY_NAME give PERSON, POSTAL_CODE and
              CITY give LOCATION...), and adjacent spans of a person or a place, separated
              by at most two spaces (no-break spaces included) or hyphens, are merged into
              one, as pii_detector merges the spans of its recognizers. This is the level
              at which a rule-based detector and a model can be compared. A merged span
              takes the lowest confidence of its parts.
    !*/

    struct pii_score
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The counts of a span level evaluation for one label.
        !*/
        std::string label;
        std::size_t true_positives = 0;
        std::size_t false_positives = 0;
        std::size_t false_negatives = 0;

        double precision() const;
        /*!
            ensures
                - Returns true_positives / (true_positives + false_positives), 1 when
                  nothing was predicted.
        !*/

        double recall() const;
        /*!
            ensures
                - Returns true_positives / (true_positives + false_negatives), 1 when
                  nothing was expected.
        !*/

        double f1() const;
        /*!
            ensures
                - Returns the harmonic mean of precision() and recall(), 0 when both are 0.
        !*/
    };

    struct pii_evaluation
    {
        std::vector<pii_score> labels;     // one entry per label, sorted by label
        pii_score overall;                 // micro averaged over all labels

        std::string to_string() const;
        /*!
            ensures
                - Returns a table with one line per label and a total line: true
                  positives, false positives, false negatives, precision, recall and F1.
        !*/
    };

    enum class span_match
    {
        exact,      // same label, same begin and end
        overlap     // same label, overlapping ranges
    };

    pii_evaluation evaluate_pii_spans(
        const std::vector<std::vector<pii_span>>& truth,
        const std::vector<std::vector<pii_span>>& predicted,
        span_match match = span_match::exact,
        const std::vector<std::string>& labels = std::vector<std::string>()
    );
    /*!
        requires
            - truth.size() == predicted.size()
        ensures
            - Scores predicted against truth, text by text. With span_match::exact a
              predicted span is a true positive when a true span of the same label has the
              same begin and end, so that a boundary error counts as a false positive and a
              false negative; with span_match::overlap, when it overlaps a true span of the
              same label not matched yet. Each true span matches one prediction at most.
            - When labels is not empty, only the spans carrying one of these labels are
              counted, in truth and in predictions alike, and every one of them has an
              entry in the result, counts at zero included.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_DATASET_ABSTRACT_H_
