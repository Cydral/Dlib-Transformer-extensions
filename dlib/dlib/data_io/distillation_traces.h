// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Distillation traces: what a teacher predicted, recorded once and reused.
//
// A teacher is asked what it would predict at every position of a corpus, and the answer
// is kept. Training a student then costs nothing more than reading a file, and several
// students of different shapes can be raised on the same recording without the teacher
// ever running again. That reuse is the whole reason the traces are a file rather than a
// call: on a model of a few billion parameters the recording pass takes hours, and paying
// it once per corpus instead of once per student is what makes a model factory practical.
//
// WHAT IS KEPT, AND WHAT IS DROPPED
//
// The full distribution is out of reach: fifty thousand floats per position is a hundred
// megabytes for a single window. Only the highest scoring entries are kept, sixty-four or
// so, which hold nearly all of the mass a trained model puts anywhere. Their raw logits
// are stored rather than probabilities, so that a temperature can still be chosen at
// training time; probabilities would have frozen that choice at recording time.
//
// The token ids are kept too, and the hard targets beside them, so that training needs the
// traces and nothing else. The corpus, the tokenizer and the teacher are all upstream of
// this file and none of them is required again once it exists.
//
// THE TOKENIZER FINGERPRINT
//
// Distillation compares distributions position by position, which only means something if
// both models number their tokens identically. A student built on another tokenizer would
// read the recorded ids as perfectly valid integers standing for other words, and would
// train on nonsense without a single error being raised. The fingerprint below makes that
// disagreement impossible to miss rather than impossible to detect.

#ifndef DLIB_DISTILLATION_TRACES_H_
#define DLIB_DISTILLATION_TRACES_H_

#include "distillation_traces_abstract.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../dnn.h"
#include "../tokenizer/hf_tokenizer.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    inline std::string tokenizer_fingerprint(const hf_tokenizer& tok)
    {
        /* FNV-1a over the vocabulary, which is what has to match: two tokenizers agreeing
           on their size while disagreeing on one entry would otherwise pass. */
        std::uint64_t h = 1469598103934665603ULL;
        auto mix = [&h](const void* data, size_t n)
        {
            const unsigned char* p = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < n; ++i)
            {
                h ^= p[i];
                h *= 1099511628211ULL;
            }
        };
        const std::uint64_t n = tok.size();
        mix(&n, sizeof(n));
        for (std::uint64_t i = 0; i < n; ++i)
        {
            const std::string& s = tok.id_to_token(static_cast<int>(i));
            mix(s.data(), s.size());
            mix("\0", 1);
        }

        static const char* digits = "0123456789abcdef";
        std::string out(16, '0');
        for (int i = 15; i >= 0; --i, h >>= 4) out[static_cast<size_t>(i)] = digits[h & 0xF];
        return out;
    }

// ----------------------------------------------------------------------------------------

    struct distillation_header
    {
        std::string tokenizer_id;    // fingerprint of the tokenizer both models share
        std::string teacher_name;    // recorded for the report, never checked
        long vocab_size = 0;
        long top_k = 0;
        long window_len = 0;
        long windows = 0;
    };

    /* One window: the tokens the teacher read, the hard target of every position, and the
       teacher's highest scoring entries there.

       ids and logits are laid out window_len rows by top_k columns, so row t holds what the
       teacher would have predicted at position t. A row whose hard target is the ignore
       label carries no useful soft target either, and the writer leaves it zeroed. */
    struct distillation_window
    {
        matrix<int, 0, 1> tokens;
        matrix<unsigned long, 0, 1> hard;
        matrix<unsigned long> ids;
        matrix<float> logits;
    };

    inline const std::string& distillation_magic()
    {
        static const std::string m = "dlib_distillation_traces";
        return m;
    }

// ----------------------------------------------------------------------------------------

    class distillation_writer
    {
        /*!
            Streams windows to disk as they are produced, so that a recording pass over a
            large corpus never holds more than one window in memory. The header carries a
            window count that is only known at the end, so it is written last and the file
            is closed by finish(); a file whose writer was destroyed without it is
            incomplete and the reader says so.
        !*/

    public:

        distillation_writer(const std::string& path, const distillation_header& head)
            : path_(path), head_(head), written_(0)
        {
            out_.open(path, std::ios::binary);
            if (!out_) throw std::runtime_error("cannot write " + path);
            serialize(distillation_magic(), out_);
            serialize(head_.tokenizer_id, out_);
            serialize(head_.teacher_name, out_);
            serialize(head_.vocab_size, out_);
            serialize(head_.top_k, out_);
            serialize(head_.window_len, out_);
        }

        void write(const distillation_window& w)
        {
            DLIB_CASSERT(w.tokens.nr() == head_.window_len,
                "A window must be " << head_.window_len << " tokens long, got " << w.tokens.nr());
            DLIB_CASSERT(w.ids.nr() == head_.window_len && w.ids.nc() == head_.top_k,
                "The teacher entries must be " << head_.window_len << " by " << head_.top_k);
            DLIB_CASSERT(w.logits.nr() == w.ids.nr() && w.logits.nc() == w.ids.nc(),
                "Every recorded id must carry a logit");
            /* A flag ahead of every window, and a cleared one to close the file. Nothing
               in the payload can be mistaken for an end marker this way, and a reader knows
               whether a window follows before trying to parse one. */
            serialize(true, out_);
            serialize(w.tokens, out_);
            serialize(w.hard, out_);
            serialize(w.ids, out_);
            serialize(w.logits, out_);
            ++written_;
        }

        long finish()
        {
            /* The count closes the file rather than opening it: a recording pass does not
               know how many windows a corpus yields until it has read it all, and a header
               rewritten in place would leave a truncated run looking complete. */
            serialize(false, out_);
            serialize(written_, out_);
            out_.close();
            return written_;
        }

        long count() const { return written_; }

    private:
        std::string path_;
        distillation_header head_;
        std::ofstream out_;
        long written_;
    };

// ----------------------------------------------------------------------------------------

    class distillation_reader
    {
    public:

        explicit distillation_reader(const std::string& path) : path_(path)
        {
            in_.open(path, std::ios::binary);
            if (!in_) throw std::runtime_error("cannot open " + path);

            std::string magic;
            deserialize(magic, in_);
            if (magic != distillation_magic())
                throw std::runtime_error("'" + path + "' is not a distillation trace file");

            deserialize(head_.tokenizer_id, in_);
            deserialize(head_.teacher_name, in_);
            deserialize(head_.vocab_size, in_);
            deserialize(head_.top_k, in_);
            deserialize(head_.window_len, in_);
            body_ = in_.tellg();
        }

        const distillation_header& header() const { return head_; }

        void check_tokenizer(const hf_tokenizer& tok) const
        {
            const std::string mine = tokenizer_fingerprint(tok);
            if (mine != head_.tokenizer_id)
                throw std::runtime_error(
                    "'" + path_ + "' was recorded against another tokenizer (" +
                    head_.tokenizer_id + " against " + mine + "). Distillation compares "
                    "distributions position by position, so the two models must number "
                    "their tokens identically; recorded ids would otherwise stand for other "
                    "words while remaining perfectly valid integers.");
        }

        /* Reads up to n windows from the current position, returning how many were read.
           Zero means the file is exhausted, which is how a caller loops over a recording
           larger than memory. */
        long read(long n, std::vector<distillation_window>& out)
        {
            out.clear();
            for (long i = 0; i < n; ++i)
            {
                bool more = false;
                try { deserialize(more, in_); }
                catch (const serialization_error&) { break; }
                if (!more) break;

                distillation_window w;
                try
                {
                    deserialize(w.tokens, in_);
                    deserialize(w.hard, in_);
                    deserialize(w.ids, in_);
                    deserialize(w.logits, in_);
                }
                catch (const serialization_error& e)
                {
                    throw std::runtime_error("'" + path_ + "' stops in the middle of a "
                        "window: the recording pass did not finish. " + e.what());
                }
                out.push_back(std::move(w));
            }
            return static_cast<long>(out.size());
        }

        void rewind()
        {
            in_.clear();
            in_.seekg(body_);
        }

    private:
        std::string path_;
        std::ifstream in_;
        distillation_header head_;
        std::streampos body_;
    };

// ----------------------------------------------------------------------------------------

    /* Records what a teacher predicts over a prepared dataset.

       Takes the windows and hard targets a dataset builder already produced, whether the
       plain causal one or the masked supervised one, and adds the only thing the teacher
       can contribute: its highest scoring entries at every position. The emitter is
       therefore indifferent to how the corpus was cut, which is what lets one recording
       serve a knowledge corpus and a question-and-answer set alike.

       Positions the dataset marks as ignored are written zeroed rather than sorted. The
       loss skips them, and on a supervised set that is two thirds of the file: sorting a
       vocabulary of fifty thousand entries there would triple the recording pass for
       nothing.

       The engine is named by type rather than by class, and only three of its members are
       used, the same three the chat service asks for: set_context, forward_prefill, and the
       shape of what the second returns. Any engine of this library qualifies. */
    template <typename engine_type>
    long record_distillation_traces(
        engine_type& teacher,
        const std::vector<matrix<int, 0, 1>>& X,
        const std::vector<matrix<unsigned long, 0, 1>>& Y,
        long top_k,
        unsigned long ignore_label,
        distillation_writer& out,
        const std::function<void(long, long)>& progress = std::function<void(long, long)>(),
        const std::function<bool()>& is_cancelled = std::function<bool()>())
    {
        DLIB_CASSERT(X.size() == Y.size(),
            "Every window must carry its hard targets: " << X.size() << " against " << Y.size());
        DLIB_CASSERT(top_k > 0, "top_k must be positive");

        const long windows = static_cast<long>(X.size());
        std::vector<int> ids;
        std::vector<long> order;

        for (long w = 0; w < windows; ++w)
        {
            const matrix<int, 0, 1>& tokens = X[static_cast<size_t>(w)];
            const matrix<unsigned long, 0, 1>& hard = Y[static_cast<size_t>(w)];
            const long len = tokens.nr();
            DLIB_CASSERT(hard.nr() == len,
                "A window of " << len << " tokens needs as many targets, got " << hard.nr());

            ids.assign(len, 0);
            for (long t = 0; t < len; ++t) ids[static_cast<size_t>(t)] = tokens(t);

            /* One window at a time from a clean cache: the teacher must see each window on
               its own, exactly as the student will. */
            teacher.set_context(len, 0);
            const tensor& logits = teacher.forward_prefill(ids);
            DLIB_CASSERT(logits.nr() == len,
                "The teacher returned " << logits.nr() << " positions for a window of " << len);
            const long vocab = logits.nc();
            DLIB_CASSERT(top_k <= vocab, "top_k exceeds the vocabulary");

            distillation_window rec;
            rec.tokens = tokens;
            rec.hard = hard;
            rec.ids.set_size(len, top_k);
            rec.logits.set_size(len, top_k);
            rec.ids = 0;
            rec.logits = 0.0f;

            const float* row = logits.host();
            for (long t = 0; t < len; ++t)
            {
                if (hard(t) == ignore_label) continue;

                const size_t base = tensor_index(logits, 0, 0, t, 0);
                order.resize(static_cast<size_t>(vocab));
                for (long c = 0; c < vocab; ++c) order[static_cast<size_t>(c)] = c;

                std::nth_element(order.begin(), order.begin() + top_k, order.end(),
                    [&](long a, long b) { return row[base + a] > row[base + b]; });
                std::sort(order.begin(), order.begin() + top_k,
                    [&](long a, long b) { return row[base + a] > row[base + b]; });

                for (long k = 0; k < top_k; ++k)
                {
                    rec.ids(t, k) = static_cast<unsigned long>(order[static_cast<size_t>(k)]);
                    rec.logits(t, k) = row[base + order[static_cast<size_t>(k)]];
                }
            }

            out.write(rec);
            if (progress) progress(w + 1, windows);
            /* Asked after the window is written, so an interrupted pass leaves a file whose
               last window is whole. Hours of recording are worth keeping. */
            if (is_cancelled && is_cancelled()) break;
        }
        return out.count();
    }

// ----------------------------------------------------------------------------------------

    /* Bytes one window of traces occupies, for a report that has to warn before a corpus
       turns into a file nobody expected. */
    inline double distillation_bytes_per_window(long window_len, long top_k)
    {
        const double per_entry = sizeof(std::uint32_t) + sizeof(float);
        return window_len * (top_k * per_entry + sizeof(int) + sizeof(std::uint32_t));
    }
}

#endif // DLIB_DISTILLATION_TRACES_H_
