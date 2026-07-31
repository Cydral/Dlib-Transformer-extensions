// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DISTILLATION_TRACES_ABSTRACT_H_
#ifdef DLIB_DISTILLATION_TRACES_ABSTRACT_H_

#include <functional>
#include <string>
#include <vector>

#include "../matrix.h"
#include "../tokenizer/hf_tokenizer_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    std::string tokenizer_fingerprint (
        const hf_tokenizer& tok
    );
    /*!
        ensures
            - Returns a sixteen character hexadecimal digest of tok's whole vocabulary.
            - Two tokenizers agreeing on their size while disagreeing on one entry produce
              different digests, which is the point: the size alone would let such a pair
              pass.
    !*/

// ----------------------------------------------------------------------------------------

    struct distillation_header
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                What a trace file declares about itself.

                - tokenizer_id: the fingerprint of the tokenizer the teacher used, which
                  the student must share.
                - teacher_name: recorded for the report and never checked.
                - vocab_size, top_k, window_len: the shape of every window in the file.
                - windows: how many the file holds, known only after reading it.
        !*/

        std::string tokenizer_id;
        std::string teacher_name;
        long vocab_size;
        long top_k;
        long window_len;
        long windows;
    };

// ----------------------------------------------------------------------------------------

    struct distillation_window
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One window of a recording: the tokens the teacher read, the hard target of
                every position, and the teacher's highest scoring entries there.

                - tokens is window_len by 1, the input the student will be given.
                - hard is window_len by 1, the target of every position or the ignore index.
                - ids and logits are window_len by top_k, row t holding what the teacher
                  would have predicted at position t.

                The hard targets travel with the traces so that training needs this file and
                nothing else: the corpus, the tokenizer and the teacher are all upstream of
                it and none is required again once it exists.
        !*/

        matrix<int, 0, 1> tokens;
        matrix<unsigned long, 0, 1> hard;
        matrix<unsigned long> ids;
        matrix<float> logits;
    };

// ----------------------------------------------------------------------------------------

    const std::string& distillation_magic (
    );
    /*!
        ensures
            - Returns the string every trace file begins with.
    !*/

// ----------------------------------------------------------------------------------------

    class distillation_writer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A trace file being written, one window at a time.

                Streaming rather than accumulating, so that a recording pass over a corpus
                larger than memory never holds more than one window. A flag precedes every
                window and a cleared flag closes the file, so nothing in the payload can be
                mistaken for an end marker and a reader knows whether a window follows
                before trying to parse one.

                WHY THE COUNT IS WRITTEN LAST

                    A recording pass does not know how many windows a corpus yields until it
                    has read it all. Writing the count at the end rather than patching a
                    header in place means an interrupted run leaves a file that says so,
                    instead of one that looks complete and is not.
        !*/

    public:

        distillation_writer (
            const std::string& path,
            const distillation_header& head
        );
        /*!
            ensures
                - Creates path and writes head to it. head.windows is ignored.
            throws
                - std::runtime_error if path cannot be written.
        !*/

        void write (
            const distillation_window& w
        );
        /*!
            requires
                - w.tokens.nr() == the window_len given at construction.
                - w.ids.nr() == window_len and w.ids.nc() == the top_k given at
                  construction.
                - w.logits has the same shape as w.ids.
            ensures
                - Appends w to the file.
                - #count() == count() + 1
        !*/

        long finish (
        );
        /*!
            ensures
                - Closes the file, recording how many windows it holds.
                - Returns that number.
                - A file whose writer was destroyed without this call is incomplete, and
                  distillation_reader::read reports it as such.
        !*/

        long count (
        ) const;
        /*!
            ensures
                - Returns how many windows have been written so far.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class distillation_reader
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A trace file being read, in batches a caller sizes.

                A recording of a real corpus does not fit in memory, so windows are handed
                over a few at a time and the file is walked once per epoch.
        !*/

    public:

        explicit distillation_reader (
            const std::string& path
        );
        /*!
            ensures
                - Opens path and reads its header.
            throws
                - std::runtime_error if path cannot be opened or is not a trace file.
        !*/

        const distillation_header& header (
        ) const;

        void check_tokenizer (
            const hf_tokenizer& tok
        ) const;
        /*!
            ensures
                - Does nothing when tok's fingerprint matches the one in the header.
            throws
                - std::runtime_error naming both fingerprints otherwise.

            WHY THIS EXISTS

                Distillation compares distributions position by position, which only means
                something if both models number their tokens identically. A student built on
                another tokenizer would read the recorded ids as perfectly valid integers
                standing for other words, and would train on nonsense without a single error
                being raised. This makes that disagreement impossible to miss rather than
                impossible to detect.
        !*/

        long read (
            long n,
            std::vector<distillation_window>& out
        );
        /*!
            ensures
                - Reads at most n windows from the current position into out and returns how
                  many were read.
                - Returns 0 once the file is exhausted, which is how a caller loops over a
                  recording larger than memory.
            throws
                - std::runtime_error if the file stops in the middle of a window, which
                  means the recording pass did not finish.
        !*/

        void rewind (
        );
        /*!
            ensures
                - The next read() returns the first window again, for a further epoch.
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <
        typename engine_type
        >
    long record_distillation_traces (
        engine_type& teacher,
        const std::vector<matrix<int, 0, 1>>& X,
        const std::vector<matrix<unsigned long, 0, 1>>& Y,
        long top_k,
        unsigned long ignore_label,
        distillation_writer& out,
        const std::function<void(long, long)>& progress = std::function<void(long, long)>()
    );
    /*!
        requires
            - X.size() == Y.size(), and every Y[i] has as many entries as X[i].
            - top_k > 0 and top_k <= the teacher's vocabulary size.
            - out was constructed with the same window length and top_k.
            - engine_type provides:
                - void set_context(long capacity, long keep);
                - const tensor& forward_prefill(const std::vector<int>& ids);
              and the second returns logits laid out as (1, 1, seq_len, vocab_size).
              Both engines of this library qualify, as does anything else that does.
        ensures
            - Runs teacher over every window of X and writes to out what it predicted, the
              top_k highest scoring entries of every position with their raw logits.
            - Positions whose hard target equals ignore_label are written zeroed rather than
              sorted. The loss skips them, and on a supervised set that is two thirds of the
              file: sorting a vocabulary of fifty thousand entries there would triple the
              recording pass for nothing.
            - Every window is shown to the teacher from a cleared cache, so that it sees
              each one on its own, exactly as the student will.
            - Calls progress(done, total) after each window when it is set.
            - Returns out.count(). The caller still has to call out.finish().

        WHY THE DATASET IS AN ARGUMENT RATHER THAN A CORPUS

            The windows and hard targets come from whichever dataset builder suits the
            corpus, the plain causal one or the masked supervised one. The emitter adds the
            only thing a teacher can contribute and stays indifferent to how the text was
            cut, which is what lets one recording pass serve a knowledge corpus and a
            question-and-answer set alike.
    !*/

// ----------------------------------------------------------------------------------------

    double distillation_bytes_per_window (
        long window_len,
        long top_k
    );
    /*!
        ensures
            - Returns the size one window occupies on disk, so that a report can warn before
              a corpus turns into a file nobody expected.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DISTILLATION_TRACES_ABSTRACT_H_
