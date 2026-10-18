// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PII_SEQUENCE_TAGGER_ABSTRACT_H_
#ifdef DLIB_PII_SEQUENCE_TAGGER_ABSTRACT_H_

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "pii_types_abstract.h"
#include "pii_dataset_abstract.h"
#include "../svm/sequence_labeler_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct pii_token
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One unit of the tagger: a run of letters and digits, or a single
                punctuation character, given by its byte range in the text.
        !*/
        std::size_t begin = 0;
        std::size_t end = 0;
    };

    std::vector<pii_token> tokenize_for_tagging(
        const std::string& text
    );
    /*!
        ensures
            - Splits the text into runs of letters, digits and underscores, and single
              punctuation characters; spaces separate tokens and belong to none. A period
              between two word characters stays inside its run (olga.horvath,
              12.03.1985), except after the initial of "J.Dupont", and an at sign that
              opens a word stays with it (@jdupont35). Hyphens and apostrophes are tokens
              of their own, so that "Jean-Pierre", "d'Arc" and "Saint-Malo" are made of
              several tokens and a span boundary can fall between them.
            - Words glued together are split where the case or the kind of character
              tells a new word starts: "DupontMarie", "DUPONTMarie", "75008Paris".
            - Invisible marks (soft hyphens, zero width spaces) are skipped; they stay
              inside the byte range of the token that holds them.
            - Returns the tokens in order, with byte offsets into text.
    !*/

// ----------------------------------------------------------------------------------------

    class pii_tagger_features
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The feature extractor of pii_sequence_tagger, implementing the interface
                of dlib::sequence_labeler (see sequence_labeler_abstract.h). A sequence is
                the list of the hashed features of each token, computed once by
                make_sequence(); get_features() conjoins them with the label of the token
                and adds the transition from the previous label.

                The features of a token are: its folded form, prefixes and suffixes, shape
                and length; the folded forms and shapes of its two neighbours on each
                side and the bigrams they form with it; its membership of the lexicons
                of the detector (first names, titles, particles, street types, legal
                forms, organisation heads, place triggers, common words); whether it
                opens or closes a line, follows a tab or a colon; and, after a field
                label such as "Nom :" or "Ville", the type that label announces.

                Each feature is hashed from its name, a separator byte and its value into
                32 bits, then mixed with the label into one of 2^hash_bits weights, which
                keeps the model size independent of the vocabulary.

            THREAD SAFETY
                Instances are immutable once constructed and may be used concurrently.
        !*/
    public:
        struct unit
        {
            std::vector<std::uint32_t> features;    // hashed base features of a token
            bool opens_segment;                     // first token of a line or a cell
        };
        typedef std::vector<unit> sequence_type;

        explicit pii_tagger_features(
            unsigned long hash_bits = 20
        );
        /*!
            requires
                - 10 <= hash_bits <= 26
            ensures
                - #hash_bits() == hash_bits
        !*/

        sequence_type make_sequence(
            const std::string& text,
            const std::vector<pii_token>& tokens
        ) const;
        /*!
            requires
                - tokens are the tokens of text, as tokenize_for_tagging() returns them
            ensures
                - Returns one unit per token: its base features, before their conjunction
                  with a label, and whether it opens a line or a cell.
        !*/

        unsigned long hash_bits() const;
        /*!
            ensures
                - Returns the number of bits of the emission weight index.
        !*/

        unsigned long num_features() const;
        /*!
            ensures
                - Returns 2^hash_bits() + num_labels()^2 + num_labels(): the emission
                  weights, then one weight per label transition and one per first label.
        !*/

        unsigned long order() const;
        /*!
            ensures
                - Returns 1: a label depends on the label of the previous token.
        !*/

        unsigned long num_labels() const;
        /*!
            ensures
                - Returns pii_bio_class_count().
        !*/

        unsigned long emission_index(std::uint32_t feature, unsigned long label) const;
        /*!
            requires
                - label < num_labels()
            ensures
                - Returns the index, below 2^hash_bits(), of the weight of a base feature
                  conjoined with a label.
        !*/

        template <typename EXP>
        bool reject_labeling(
            const sequence_type& x,
            const matrix_exp<EXP>& y,
            unsigned long position
        ) const;
        /*!
            ensures
                - Returns true for an I class that does not continue a span of the same
                  label (see pii_bio_transition_allowed()), and for an I class on the
                  first token of a line or a cell: a span never crosses a line break or a
                  tab.
        !*/

        template <typename feature_setter, typename EXP>
        void get_features(
            feature_setter& set_feature,
            const sequence_type& x,
            const matrix_exp<EXP>& y,
            unsigned long position
        ) const;
        /*!
            ensures
                - Calls set_feature() with the emission index of every base feature of the
                  token at position, conjoined with its label y(0), and with the index of
                  the transition from y(1), or of the first label when position is 0.
        !*/
    };

    void serialize(const pii_tagger_features& item, std::ostream& out);
    void deserialize(pii_tagger_features& item, std::istream& in);
    /*!
        deserialize() throws serialization_error if the stream does not hold a
        pii_tagger_features or if its hash size lies outside [10, 26].
    !*/

// ----------------------------------------------------------------------------------------

    struct pii_tagger_training_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The parameters of pii_sequence_tagger::train().
        !*/
        double c = 10;                      // the SVM C, larger values fit the data closer
        double epsilon = 0.05;              // stopping tolerance of the cutting plane solver
        unsigned long max_iterations = 200; // solver iterations
        unsigned long num_threads = 2;
        unsigned long hash_bits = 20;       // in [10, 26]
        double entity_loss = 2;             // cost of a wrong label on a span unit, O counting 1
        unsigned long max_cache_size = 0;   // labelings the solver keeps per example
        bool verbose = false;
    };

    class pii_sequence_tagger
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A learned recognizer of given names, family names, usernames, street
                addresses, postal codes, cities and organisations (pii_fine_labels()). It
                is the chain structured model of dlib::sequence_labeler, trained as a
                structural SVM over the tokens of tokenize_for_tagging(), each labelled
                with a BIO class.

                The model complements the rules of pii_detector: its features describe the
                context in which a name or a place appears ("demeurant", "chez", "Signé
                par", a field label) besides the word itself, so that it also labels
                names absent from its training data. Plug it into a detector with

                    detector.add_recognizer([&](const std::string& t) { return tagger.find(t); });

            THREAD SAFETY
                find() is const and may run concurrently with itself.
        !*/
    public:
        pii_sequence_tagger();
        /*!
            ensures
                - #is_trained() == false
        !*/

        void train(
            const std::vector<pii_example>& examples,
            const pii_tagger_training_options& options = pii_tagger_training_options()
        );
        /*!
            requires
                - examples holds at least one example with a token
                - options.c > 0, options.epsilon > 0, options.max_iterations > 0,
                  options.num_threads > 0, options.entity_loss >= 0
                - 10 <= options.hash_bits <= 26
            ensures
                - Trains the model on the spans of examples whose label is a fine label;
                  the other spans are ignored.
                - The memory used grows with the examples, by about 15 kB for an example
                  of 50 tokens, and by about as much again per example for each labeling
                  options.max_cache_size lets the solver keep: a cache of 5 makes the
                  training about three times faster and needs about six times the
                  memory.
                - #is_trained() == true
                - #misaligned_spans() == the number of training spans that do not start
                  and end on token boundaries
                - The object changes only when training succeeds.
        !*/

        bool is_trained() const;
        /*!
            ensures
                - Returns true once train() has completed or a trained model was
                  deserialized.
        !*/

        std::vector<pii_span> find(
            const std::string& text
        ) const;
        /*!
            requires
                - is_trained() == true
            ensures
                - Returns the spans found in the text, sorted, with their fine label, the
                  type it maps to and a confidence in [0, 1]: the logistic of the margin
                  between the score of the predicted class and that of the O class,
                  averaged over the tokens of the span. No span crosses a line break or a
                  tab.
                - An empty result means that nothing was found.
        !*/

        const pii_tagger_features& get_feature_extractor() const;
        /*!
            ensures
                - Returns the feature extractor the model was trained with, the default
                  one before training.
        !*/

        std::size_t misaligned_spans() const;
        /*!
            ensures
                - Returns the number of training spans that did not start and end on token
                  boundaries during the last train(), 0 after deserialization.
        !*/
    };

    void serialize(const pii_sequence_tagger& item, std::ostream& out);
    void deserialize(pii_sequence_tagger& item, std::istream& in);
    /*!
        The weights are written in single precision, which halves the file and changes no
        decision. deserialize() throws serialization_error if the stream does not hold a
        pii_sequence_tagger; item is left unchanged when it throws.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_SEQUENCE_TAGGER_ABSTRACT_H_

