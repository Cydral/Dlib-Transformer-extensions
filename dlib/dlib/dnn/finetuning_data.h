// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Tokenized fine-tuning datasets, from prepared corpora to trainer input.
//
// data_io/language_model_data.h reads the corpus files and turns token sequences into
// training windows, but it knows nothing of tokenizers or conversation formats, and
// cannot: tokenizer/ already depends on data_io/, so the reverse dependency would close
// a cycle. This header is where the two meet, the training-side counterpart of
// text_generation.h.
//
// One rule governs everything here: a fine-tuned model is used through the inference
// path, so the prompt it is trained on must be the byte-for-byte prompt that path will
// build. The prompt of a supervised example is therefore produced by encode_turn(), the
// same function the interactive loop and the served endpoint call, rather than by a
// layout invented for training. A model taught on turn markers it will never see again
// spends its capacity on the difference.

#ifndef DLIB_DNN_FINETUNING_DATA_H_
#define DLIB_DNN_FINETUNING_DATA_H_

#include "finetuning_data_abstract.h"

#include <string>
#include <vector>

#include "text_generation.h"
#include "../data_io/language_model_data.h"

namespace dlib
{
    /* Token sequences of a plain corpus, for the knowledge-alignment stage.

       append_eos closes every document with the tokenizer's end-of-sequence marker. It
       matters when the windows are built with packing, where documents are concatenated
       into one stream: without a separator the model learns that the last sentence of a
       standards publication is followed by the first sentence of an unrelated one. */
    inline std::vector<std::vector<int>> tokenize_documents(
        const hf_tokenizer& tok,
        const std::vector<std::string>& documents,
        bool append_eos = true)
    {
        std::vector<std::vector<int>> out;
        out.reserve(documents.size());
        for (const std::string& d : documents)
        {
            if (d.empty()) continue;
            /* Special-token parsing is off: corpus text is prose, and a document quoting
               a marker such as <|im_end|> must be tokenized as the characters it is made
               of, not silently promoted to the control token of the same name. */
            std::vector<int> ids = tok.encode(d, false, false, false, false);
            if (ids.empty()) continue;
            if (append_eos && tok.eos_id() >= 0) ids.push_back(tok.eos_id());
            out.push_back(std::move(ids));
        }
        return out;
    }

    /* Token ids of one supervised example, split into the context and the target.

       The prompt is the first turn of a conversation: the template's system block, the
       user message and the assistant header, exactly as encode_turn() renders it at
       inference. The response is the reference answer followed by the end-of-sequence
       marker, which is supervised like any other target position because stopping is
       part of what the model must learn; an answer trained without it runs on until the
       token budget is exhausted. */
    inline supervised_example encode_supervised_example(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const chat_record& record,
        bool supervise_eos = true)
    {
        supervised_example ex;
        ex.prompt = encode_turn(tok, fmt, record.system, record.user, /*first_turn=*/true);
        ex.response = tok.encode(record.assistant, false, false, false, false);
        if (supervise_eos && tok.eos_id() >= 0) ex.response.push_back(tok.eos_id());
        return ex;
    }

    inline std::vector<supervised_example> encode_supervised_examples(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::vector<chat_record>& records,
        bool supervise_eos = true)
    {
        std::vector<supervised_example> out;
        out.reserve(records.size());
        for (const chat_record& r : records)
        {
            supervised_example ex = encode_supervised_example(tok, fmt, r, supervise_eos);
            if (!ex.prompt.empty() && !ex.response.empty()) out.push_back(std::move(ex));
        }
        return out;
    }

    /* Length statistics of a set of supervised examples, in tokens.

       Reported before the windows are built, because the window length is the one
       hyper-parameter that silently destroys a fine-tuning run: set below the bulk of
       the examples it discards most of the corpus, set far above it every batch is
       mostly padding and the run is slow for nothing. The percentiles say where to put
       it; the caller then reads dataset_report to see what the choice actually cost. */
    struct length_profile
    {
        size_t examples = 0;
        long   max_total = 0;
        long   p50_total = 0;
        long   p90_total = 0;
        long   p99_total = 0;
        double mean_prompt = 0.0;
        double mean_response = 0.0;
        /* Every total length, sorted. Kept rather than summarized so that an arbitrary
           coverage can be asked for after the fact; a few hundred kilobytes on a corpus
           of a hundred thousand examples. */
        std::vector<long> sorted_totals;

        // Shortest window that holds the given fraction of the examples.
        long quantile(double q) const
        {
            if (sorted_totals.empty()) return 0;
            if (q <= 0.0) return sorted_totals.front();
            if (q >= 1.0) return sorted_totals.back();
            const size_t i = static_cast<size_t>(q * (sorted_totals.size() - 1) + 0.5);
            return sorted_totals[i];
        }

        // Fraction of the examples a window of the given length would keep whole.
        double coverage_at(long window_len) const
        {
            if (sorted_totals.empty()) return 0.0;
            const long capacity = window_len + 1;
            size_t kept = 0;
            for (long t : sorted_totals) { if (t <= capacity) ++kept; else break; }
            return static_cast<double>(kept) / sorted_totals.size();
        }

        std::string describe() const
        {
            std::ostringstream o;
            o << "examples      : " << examples << "\n"
              << "total tokens  : median " << p50_total << ", p90 " << p90_total
              << ", p99 " << p99_total << ", max " << max_total << "\n"
              << "mean split    : " << static_cast<long>(mean_prompt + 0.5) << " prompt + "
              << static_cast<long>(mean_response + 0.5) << " response";
            return o.str();
        }
    };

    inline length_profile profile_lengths(const std::vector<supervised_example>& examples)
    {
        length_profile p;
        p.examples = examples.size();
        if (examples.empty()) return p;

        std::vector<long> totals;
        totals.reserve(examples.size());
        double sp = 0.0, sr = 0.0;
        for (const supervised_example& e : examples)
        {
            const long np = static_cast<long>(e.prompt.size());
            const long nr = static_cast<long>(e.response.size());
            totals.push_back(np + nr);
            sp += np;
            sr += nr;
        }
        std::sort(totals.begin(), totals.end());
        p.sorted_totals = std::move(totals);
        p.max_total = p.sorted_totals.back();
        p.p50_total = p.quantile(0.50);
        p.p90_total = p.quantile(0.90);
        p.p99_total = p.quantile(0.99);
        p.mean_prompt = sp / examples.size();
        p.mean_response = sr / examples.size();
        return p;
    }

    /* Shortest window that keeps the requested fraction of the examples whole, rounded up
       to a multiple of granularity and capped.

       The window length is a property of the fine-tuning data, not of the model: nothing
       in the architecture ties them, rotary positions extend to any length and the KV
       cache is sized at inference. Training on short windows and serving on long ones is
       normal, and it is what keeps a stage affordable, since attention cost grows with the
       square of the window while almost every example may sit far below the model's
       capacity.

       A window of L scores L targets, the last of them the token just past it, so L + 1
       tokens fit. Coverage is a deliberate trade: pushing it to one lets a handful of
       outliers set the cost of every batch, which is why the default stops at 95 percent
       and leaves the remainder to the overflow policy. */
    inline long suggest_window_length(
        const length_profile& profile,
        double coverage = 0.95,
        long granularity = 64,
        long max_window = 0)
    {
        DLIB_CASSERT(granularity > 0, "granularity must be positive");
        if (profile.sorted_totals.empty()) return granularity;

        const long needed = profile.quantile(coverage) - 1;
        long window = ((std::max<long>(needed, 1) + granularity - 1) / granularity) * granularity;
        if (max_window > 0 && window > max_window) window = max_window;
        return std::max<long>(window, granularity);
    }
}

#endif // DLIB_DNN_FINETUNING_DATA_H_
