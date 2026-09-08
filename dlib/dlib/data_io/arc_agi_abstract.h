// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_ARC_AGI_ABSTRACT_H_
#ifdef DLIB_ARC_AGI_ABSTRACT_H_

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <random>
#include "../matrix.h"
#include "../dir_nav.h"
#include "../serialize.h"

namespace dlib
{
    // Type aliases for ARC-AGI data structures
    using arc_grid_t = matrix<unsigned char>;
    using arc_token_sequence_t = matrix<int, 0, 1>;

    // Token vocabulary for the Hierarchical Reasoning Model
    enum arc_token_id : int
    {
        COLOR_0 = 0, COLOR_1 = 1, COLOR_2 = 2, COLOR_3 = 3, COLOR_4 = 4,
        COLOR_5 = 5, COLOR_6 = 6, COLOR_7 = 7, COLOR_8 = 8, COLOR_9 = 9,
        TOKEN_SEP_IO = 10,
        TOKEN_SEP_PAIR = 11,
        TOKEN_QUERY_START = 12,
        TOKEN_GEN_START = 13,
        TOKEN_END_OF_OUTPUT = 14,
        TOKEN_PADDING = 15,
        TOKEN_ROW_END = 16
    };

    // Vocabulary size constants
    constexpr int ARC_VOCAB_SIZE_COLORS = 10;
    constexpr int ARC_VOCAB_SIZE_TOTAL = arc_token_id::TOKEN_ROW_END + 1;

    struct arc_task_pair
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Represents a single Input/Output example pair within an ARC task.
                Each pair demonstrates a transformation pattern that the model must
                learn.
        !*/

        arc_grid_t input;
        arc_grid_t output;
        long input_rows;
        long input_cols;
        long output_rows;
        long output_cols;

        friend void serialize(const arc_task_pair& item, std::ostream& out);
        friend void deserialize(arc_task_pair& item, std::istream& in);
        /*!
            ensures
                - Standard dlib serialization protocol. All six fields are persisted.
        !*/
    };

    struct arc_task
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Represents a complete ARC-AGI reasoning task containing:
                - Multiple training pairs demonstrating a pattern;
                - One or more test pairs where the model must predict outputs.
        !*/

        std::string task_id;
        std::vector<arc_task_pair> train_pairs;
        std::vector<arc_task_pair> test_pairs;

        friend void serialize(const arc_task& item, std::ostream& out);
        friend void deserialize(arc_task& item, std::istream& in);
        /*!
            ensures
                - Standard dlib serialization protocol.
        !*/
    };

    class arc_agi_manager
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object provides utilities for loading, accessing, and preparing
                ARC-AGI (Abstraction and Reasoning Corpus for Artificial General
                Intelligence) data for training Transformer-based models such as the
                Hierarchical Reasoning Model (HRM).

                The ARC-AGI dataset consists of visual reasoning tasks where each
                task contains:
                - Training pairs: Input/Output grid examples demonstrating a pattern;
                - Test pairs: Input grids where the model must predict the output.

                Each grid is a 2D matrix of integers (0-9) representing colors,
                with maximum dimensions of 30x30.

                TOKENIZATION STRATEGY
                    Grids are tokenized row-by-row with TOKEN_ROW_END markers inserted
                    at the end of each row. This encoding preserves dimensional
                    information implicitly, allowing the model to learn and generate
                    grids of arbitrary dimensions (1x1 to 30x30, including non-square
                    grids) without requiring explicit dimension specification.

                The dataset is available from: https://github.com/fchollet/ARC-AGI
        !*/

    public:
        arc_agi_manager();
        /*!
            ensures
                - Constructs an empty arc_agi_manager object.
        !*/

        void load_data(
            const std::string& training_path,
            const std::string& evaluation_path
        );
        /*!
            ensures
                - Loads the ARC-AGI dataset from the specified directories.
                - training_path should contain JSON files for training tasks.
                - evaluation_path should contain JSON files for evaluation tasks.
                - Each JSON file represents one task with training and test pairs.
                - Task IDs are extracted from filenames (without .json extension).
            throws
                - std::runtime_error if directories cannot be accessed or files
                  cannot be parsed.
        !*/

        const arc_task& get_training_task(size_t index) const;
        const arc_task& get_evaluation_task(size_t index) const;
        /*!
            requires
                - index < num_training_tasks() (resp. num_evaluation_tasks()).
            ensures
                - Returns the task at the specified index.
            throws
                - std::out_of_range if index is out of bounds.
        !*/

        const arc_task& get_training_task_by_id(const std::string& task_id) const;
        const arc_task& get_evaluation_task_by_id(const std::string& task_id) const;
        /*!
            ensures
                - Returns the task with the specified task_id.
            throws
                - std::runtime_error if task_id is not found.
        !*/

        size_t num_training_tasks() const;
        size_t num_evaluation_tasks() const;
        /*!
            ensures
                - Returns the number of loaded tasks of the corresponding kind.
        !*/

        void serialize(std::ostream& out) const;
        void deserialize(std::istream& in);
        /*!
            ensures
                - Writes / reads the entire dataset to / from the stream using
                  dlib's serialization format. The output can be saved as a .dat
                  file for faster loading on subsequent runs.
            throws
                - serialization_error on invalid data format.
        !*/

        static arc_token_sequence_t tokenize_input_context(
            const arc_task& task,
            const arc_task_pair& test_pair
        );
        /*!
            ensures
                - Builds the token sequence the model uses as context for the
                  specified test pair:
                      for each training pair p:
                          [p.input flattened with TOKEN_ROW_END]
                          TOKEN_SEP_IO
                          [p.output flattened with TOKEN_ROW_END]
                          TOKEN_SEP_PAIR
                      TOKEN_QUERY_START
                      [test_pair.input flattened with TOKEN_ROW_END]
                      TOKEN_GEN_START
                - Each grid is encoded with TOKEN_ROW_END markers at the end of
                  every row to preserve dimensional information.
                - Returns a column vector of int tokens.
        !*/

        static arc_token_sequence_t tokenize_target_output(
            const arc_task_pair& test_pair
        );
        /*!
            ensures
                - Builds the token sequence the model must predict for the
                  specified test pair:
                      [test_pair.output flattened with TOKEN_ROW_END]
                      TOKEN_END_OF_OUTPUT
                - Returns a column vector of int tokens.
        !*/

        // ----------------------------------------------------------------------
        // Training-data builders
        //
        // Three strategies are provided to feed an autoregressive transformer
        // with ARC-AGI samples. They differ in how the demonstration pairs of a
        // task are exposed to the model, and serve different stages of training.
        // ----------------------------------------------------------------------

        static void prepare_training_data_pair_only(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch,
            bool debug = false
        );
        /*!
            requires
                - window_len > 1.
            ensures
                - For every pair p in task.train_pairs and task.test_pairs,
                  builds the sequence
                      [p.input flattened] TOKEN_GEN_START [p.output flattened]
                      TOKEN_END_OF_OUTPUT
                  and produces one training sample (window of size window_len,
                  next-token label) at every position inside the output portion.
                - The model is NOT exposed to other pairs while predicting a given
                  output. Useful as a warmup phase to teach the encoding scheme
                  before introducing few-shot demonstrations.
                - Left-pads with TOKEN_PADDING when the window extends past the
                  start of the sequence.
            throws
                - DLIB_CASSERT if window_len <= 1.
        !*/

        static void prepare_training_data_sliding_window(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch,
            bool debug = false
        );
        /*!
            requires
                - window_len > 1.
            ensures
                - Builds the FULL task sequence:
                      [train_in_1] SEP_IO [train_out_1] SEP_PAIR ...
                      QUERY_START [test_in] GEN_START [test_out] END_OF_OUTPUT ...
                  and emits one training sample at every position of that
                  sequence, including positions that fall inside demonstration
                  pairs (i.e. the model is trained to predict every token, not
                  only the test output).
                - The test_pair outputs are exposed to the model during training
                  in this mode. Suitable for tasks where the model should learn
                  generic next-token modelling over the full task structure.
                - Left-pads with TOKEN_PADDING when needed.
            throws
                - DLIB_CASSERT if window_len <= 1.
        !*/

        static void prepare_training_data_batch(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch,
            bool debug = false
        );
        /*!
            requires
                - window_len > 1.
            ensures
                - Implements the held-out few-shot training strategy used to teach
                  ARC-style reasoning:
                    * For every pair p in task.train_pairs (held-out role):
                        - Builds a synthetic task whose demonstrations are the
                          OTHER train_pairs and whose target is p.
                        - Concatenates tokenize_input_context(synthetic, p) and
                          tokenize_target_output(p) to obtain the full sequence.
                        - Locates the TOKEN_GEN_START position and considers only
                          the prediction positions strictly inside the output.
                        - Keeps only positions whose target token is a color
                          (COLOR_0..COLOR_9), TOKEN_ROW_END, or TOKEN_END_OF_OUTPUT,
                          discarding positions whose target is structural or
                          padding.
                        - Randomly subsamples those positions to at most 256 per
                          held-out pair to bound the per-task sample budget.
                        - For each retained position, extracts a left-padded
                          window of length window_len ending at that position.
                          Windows with more than 80% padding tokens are dropped.
                    * task.test_pairs are NOT used (their labels are unknown at
                      training time; they are reserved for evaluation).
                - This strategy exposes the model to multiple demonstration pairs
                  while predicting one held-out target, which is the canonical
                  setup for learning few-shot reasoning over ARC tasks.
                - The RNG is a thread-local std::mt19937 seeded once per thread.
            throws
                - DLIB_CASSERT if window_len <= 1.
        !*/

        static arc_grid_t detokenize_to_grid(
            const arc_token_sequence_t& tokens,
            long start_idx = 0
        );
        /*!
            requires
                - tokens contains a valid tokenized grid sequence with
                  TOKEN_ROW_END markers.
            ensures
                - Reconstructs a grid from a tokenized sequence:
                  * TOKEN_ROW_END terminates the current row;
                  * Parsing stops at TOKEN_END_OF_OUTPUT, TOKEN_SEP_IO, or
                    TOKEN_SEP_PAIR;
                  * Color tokens (COLOR_0..COLOR_9) become grid cells;
                  * Other tokens (including TOKEN_PADDING) are ignored.
                - Returns the reconstructed matrix.
                - Returns an empty matrix (0x0) if no valid row is found.
                - Grid dimensions are inferred from the row structure of the
                  token stream.
            throws
                - DLIB_CASSERT if row lengths are inconsistent.

            EXAMPLE
                tokens = [1, 2, 3, ROW_END, 4, 5, 6, ROW_END, END_OF_OUTPUT]
                returns the 2x3 grid [[1, 2, 3], [4, 5, 6]].
        !*/
    };


// ----------------------------------------------------------------------------------------

    struct arc_dataset
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A prepared ARC dataset: every grid already flattened into a fixed window,
                and the index arrays saying which examples belong to which puzzle and which
                puzzles belong to which group.

                A model reading this data sees one input grid and no worked examples of the
                same task. What tells it which transformation to apply is the puzzle
                identifier, carried as the first token and looked up in the same embedding
                table as the colours. An identifier the model never trained on therefore
                carries no meaning, and a puzzle can only be asked about by name if it was
                named during training.
        !*/

        long puzzle_of (long example) const;
        /*!
            ensures
                - returns where in this split the example's puzzle sits.
                - This is a position, not a name. Use identifier_of() to name a puzzle.
        !*/

        int identifier_of (long example) const;
        /*!
            requires
                - 0 <= example < num_examples()
            ensures
                - returns the identifier of the puzzle the example belongs to, which is the
                  number a network looks up.
                - This is not puzzle_of(). A split built under the reference protocol stores
                  identifiers assigned while the training split was built, so an evaluation
                  puzzle sitting at position three may be named twenty. Passing the position
                  names a different puzzle, silently, and only on the protocol where the
                  comparison is meant to happen.
        !*/

        int largest_identifier () const;
        /*!
            ensures
                - returns the largest identifier this split can name, which is what an
                  embedding table has to reach for the split to be usable at all.
        !*/

        long vocabulary      () const;
        long sequence_length () const;
        /*!
            ensures
                - vocabulary() is the colours plus this split's own puzzle count, a lower
                  bound on the table a network needs. Under the reference protocol the
                  bound that matters is colour_vocab + largest_identifier() + 1.
                - sequence_length() is the window plus the one identifier token.
        !*/
    };

// ----------------------------------------------------------------------------------------

    arc_dataset load_arc_dataset (const std::string& filename);
    /*!
        ensures
            - reads a file written by slm_tools/build_arc_dataset.py.
        throws
            - std::runtime_error if the file is missing, is not a prepared ARC file, has a
              version this build does not know, or ends early.
    !*/

    matrix<int, 0, 1>           arc_make_input (const arc_dataset& d, long example, int puzzle_id);
    matrix<unsigned long, 0, 1> arc_make_label (const arc_dataset& d, long example);
    /*!
        requires
            - 0 <= example < d.num_examples() and puzzle_id >= 0
        ensures
            - arc_make_input returns the identifier token followed by the window.
            - arc_make_label returns the answer, its first position labelled with the pad
              since the identifier is given rather than predicted.
            - passing arc_blank_puzzle_id withholds the identifier, which asks the model to
              work without being told which rule applies.
    !*/

    std::vector<unsigned long> arc_predict_sequence (const float* logits, long row_stride,
                                                     long sequence_length, long vocabulary);
    /*!
        requires
            - logits points at sequence_length rows of row_stride floats
            - row_stride >= vocabulary > 0
        ensures
            - returns the most likely colour at each position, read in one pass. Generating
              the window token by token would be nine hundred forward passes to answer one
              puzzle, and the loss is per position, so every position already has its
              logits after a single pass.
    !*/

    arc_grid_t arc_decode_grid (const std::vector<unsigned long>& predicted,
                                long seq_len, long grid_side);
    /*!
        requires
            - grid_side > 0, seq_len == grid_side * grid_side, predicted.size() >= seq_len
        ensures
            - returns the grid the prediction describes, its extent taken from the largest
              row and column carrying a colour.
            - returns an empty grid when the window is entirely padding: an empty answer is
              a wrong answer rather than a malformed one.
    !*/

} // namespace dlib

#endif // DLIB_ARC_AGI_ABSTRACT_H_
