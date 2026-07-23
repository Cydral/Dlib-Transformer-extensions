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
        auto at = [&](double q) {
            size_t i = static_cast<size_t>(q * (totals.size() - 1) + 0.5);
            return totals[i];
        };
        p.max_total = totals.back();
        p.p50_total = at(0.50);
        p.p90_total = at(0.90);
        p.p99_total = at(0.99);
        p.mean_prompt = sp / examples.size();
        p.mean_response = sr / examples.size();
        return p;
    }
}

#endif // DLIB_DNN_FINETUNING_DATA_H_
