// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PII_SPAN_MODEL_ABSTRACT_H_
#ifdef DLIB_PII_SPAN_MODEL_ABSTRACT_H_

#include <iostream>
#include <string>
#include <vector>
#include "pii_types_abstract.h"
#include "pii_dataset_abstract.h"
#include "../dnn.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class pii_span_optimizer
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                The update rule of pii_span_model::train().

                adamw      - dlib::adamw
                adam_atan2 - dlib::adam_atan2: the arctangent of the two moments
                             replaces their quotient, so the step needs no epsilon,
                             is bounded and ignores the scale of the gradient
        !*/
        adamw,
        adam_atan2
    };

    struct pii_span_model_training_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The parameters of pii_span_model::train().

                Training reads windows of bytes cut from the examples laid end to end,
                a line feed between two examples, so that no window is padded and the
                attention of the model never sees a padding position. Windows of several
                lengths, drawn with the given weights, teach the model short units such
                as table cells and long ones such as paragraphs. A window that starts
                inside a span starts it with its B class.
        !*/
        unsigned long steps = 20000;            // optimizer steps
        unsigned long batch_size = 32;          // windows per step
        std::vector<long> window_lengths = { 32, 64, 128, 256 };   // each in [16, 4096]
        std::vector<double> window_weights = { 1, 2, 3, 4 };       // one per length, not all 0
        double learning_rate = 1e-3;            // peak learning rate
        double min_learning_rate = 1e-5;        // end of the cosine decay
        unsigned long warmup_steps = 500;       // linear warm-up
        double weight_decay = 0.01;             // decoupled weight decay
        pii_span_optimizer optimizer = pii_span_optimizer::adamw;   // moments 0.9 and 0.98
        bool continue_training = false;         // start from the current weights of a trained model
        unsigned long report_every = 200;       // steps between two log lines
        unsigned long evaluate_every = 2000;    // steps between two evaluations on dev, 0 for none
        unsigned long evaluate_examples = 500;  // dev examples used by those evaluations
        unsigned long seed = 1;                 // the window draws and the initial weights
        std::vector<int> gpus = { 0 };          // devices used by dnn_trainer
    };

// ----------------------------------------------------------------------------------------

    template <long EMBEDDING_DIM, long NUM_LAYERS, long NUM_HEADS>
    class pii_span_model
    {
        /*!
            REQUIREMENTS ON TEMPLATE ARGUMENTS
                - EMBEDDING_DIM % NUM_HEADS == 0 and EMBEDDING_DIM / NUM_HEADS is even

            WHAT THIS OBJECT REPRESENTS
                A small transformer specialized in one task: for an input string, tell
                whether it holds personal data of the categories of pii_fine_labels(),
                and where, as byte offsets with the label of each span.

                The model reads UTF-8 bytes, so it needs no tokenizer: spelling variants
                and OCR damage reach it unchanged, and it reports exact offsets. It is an
                encoder: a stack of NUM_LAYERS bidirectional pre-norm blocks (fused
                grouped query attention with rotary positions, SwiGLU feed-forward,
                RMSNorm) over byte embeddings, and a linear head giving each byte one of
                the pii_bio_class_count() BIO classes, trained with
                loss_cross_entropy_per_token. Spans are decoded with a Viterbi pass that
                only allows an I class after the B or I of the same label, and keeps line
                breaks and tabs outside every span. As an encoder, it reports substrings
                of its input only.

                Invisible marks (soft hyphens, zero width spaces and joiners, byte order
                marks) are left out of what the model reads, and the offsets it reports
                point into the original text.

                Texts longer than the window are read through overlapping windows; each
                byte takes the prediction of the window in which it is most central.

            THREAD SAFETY
                find() may be called from several threads; calls are serialized.
        !*/
    public:
        static constexpr long num_classes = 15;  // pii_bio_class_count()
        static constexpr long vocabulary = 260;  // the 256 byte values and a few reserved ids

        typedef ... net_type;
        /*!
            The network: loss_cross_entropy_per_token<linear<num_classes, rms_norm<
                gqa_transformer_unified::transformer_stack<NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS,
                NUM_HEADS, embeddings<vocabulary, EMBEDDING_DIM, input<matrix<int,0,1>>>, false,
                8, 3, linear, EMBEDDING_DIM / NUM_HEADS, false, false>>>>
            (no adaptive computation, not causal).
        !*/

        static constexpr long max_window = 4096;

        pii_span_model();
        /*!
            ensures
                - #is_trained() == false
                - #window() == 256
                - #parameter_count() == 0
        !*/

        long window() const;
        /*!
            ensures
                - Returns the length of the windows find() reads long texts through: the
                  longest window length of the last training, or the value given to
                  set_window().
        !*/

        void set_window(long length);
        /*!
            requires
                - 16 <= length <= max_window
            ensures
                - #window() == length. A window longer than the training windows asks the
                  model to read positions it never saw during training.
        !*/

        void train(
            const std::vector<pii_example>& training_set,
            const std::vector<pii_example>& dev_set,
            const pii_span_model_training_options& options = pii_span_model_training_options(),
            std::ostream* log = nullptr
        );
        /*!
            requires
                - training_set holds an example with a non-empty text
                - options.window_lengths is not empty and has one weight per length, each
                  length in [16, max_window], each weight >= 0 and their sum > 0
                - options.batch_size > 0, options.learning_rate > 0,
                  options.min_learning_rate >= 0, options.weight_decay >= 0
            ensures
                - Trains the model with options.optimizer, a linear warm-up and a cosine
                  decay of the learning rate. Spans whose label is not a fine label are
                  ignored.
                - When options.continue_training is true and is_trained() == true, starts
                  from the current weights: a complementary training, on new data mixed
                  with the data the model was trained on so that it keeps what it knows,
                  usually with a lower learning rate. The solver state starts afresh.
                - Otherwise seeds std::rand() with options.seed, since the layers draw their
                  initial weights from it; with the same data, options and device, two
                  trainings give the same model.
                - When log is not null, writes every options.report_every steps the mean
                  loss of the steps since the previous line. When dev_set is not empty, it
                  also writes every options.evaluate_every steps and at the end the loss
                  on windows that cover its first options.evaluate_examples examples once,
                  the same windows each time, and the exact span F1 on those examples; a
                  complementary training writes them before its first step too.
                - Besides training_set, the training keeps two bytes per byte of its texts:
                  each byte and its class.
                - #is_trained() == true
                - #window() == the longest of options.window_lengths
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
                - Returns the spans of the text, sorted, each with its fine label, its type
                  and a confidence in [0, 1]: the mean probability the model gives the
                  decoded class over the bytes of the span. Spans start and end on UTF-8
                  character boundaries, carry no surrounding space or separator and are
                  extended to the limits of the words they cut, except where a change of
                  case marks a word glued to another ("DupontMarie"), and never across a
                  neighbouring span.
                - An empty result means that nothing was found.
        !*/

        size_t parameter_count() const;
        /*!
            ensures
                - Returns the number of parameters of the trained network, 0 before
                  training.
        !*/
    };

    template <long D, long L, long H>
    void serialize(const pii_span_model<D, L, H>& item, std::ostream& out);
    template <long D, long L, long H>
    void deserialize(pii_span_model<D, L, H>& item, std::istream& in);
    /*!
        The stream records the three template arguments, the window and the network.
        deserialize() throws serialization_error when the arguments differ from those of
        item, naming the size the model was trained with, or when the window lies outside
        [16, max_window]; item is left unchanged when it throws.
    !*/

    typedef pii_span_model<64, 2, 2>   pii_span_model_tiny;
    typedef pii_span_model<128, 4, 4>  pii_span_model_small;
    typedef pii_span_model<256, 6, 8>  pii_span_model_base;
    /*!
        Three sizes of the same network, by embedding dimension, layers and heads. A smaller
        size trains and runs in less time, a larger one has more capacity for a large
        corpus; a model file is read back with the size it was trained with.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_SPAN_MODEL_ABSTRACT_H_
