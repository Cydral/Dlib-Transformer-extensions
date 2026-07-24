// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_FINETUNING_DATA_ABSTRACT_H_
#ifdef DLIB_DNN_FINETUNING_DATA_ABSTRACT_H_

#include "text_generation_abstract.h"
#include "../data_io/language_model_data_abstract.h"

namespace dlib
{
    /*!
        WHAT THIS FILE REPRESENTS
            The step between a prepared corpus and the tensors a trainer consumes:
            tokenization of plain documents, and of supervised records through the
            model's own conversation template.

            It lives in dnn/ rather than in data_io/ because it needs the tokenizer and
            the chat template, and tokenizer/ already depends on data_io/; putting it
            there would close a dependency cycle.

            Everything here follows one rule: the prompt a model is fine-tuned on must
            be the byte-for-byte prompt the inference path will build for it. The prompt
            of a supervised example therefore comes from encode_turn(), the function the
            interactive loop and the served endpoint already call.

        TYPICAL USAGE
            // --- knowledge alignment ---
            std::vector<std::string> docs;
            load_document_corpus("nist_corpus.txt", docs);
            std::vector<matrix<int,0,1>> X;
            std::vector<matrix<unsigned long,0,1>> Y;
            dataset_report rep = build_causal_lm_dataset(
                tokenize_documents(tok, docs), window, window,
                pad_id, ignore_label, true, X, Y);

            // --- task alignment ---
            std::vector<chat_record> records;
            load_chat_records("cve_qa.txt", records);
            std::vector<supervised_example> ex =
                encode_supervised_examples(tok, fmt, records);
            std::cout << profile_lengths(ex).describe() << "\n";
            dataset_report rep = build_supervised_finetuning_dataset(
                ex, window, pad_id, ignore_label,
                sequence_overflow_policy::skip, X, Y);
    !*/

// ----------------------------------------------------------------------------------------

    std::vector<std::vector<int>> tokenize_documents(
        const hf_tokenizer& tok,
        const std::vector<std::string>& documents,
        bool append_eos = true
    );
    /*!
        ensures
            - Returns the token ids of each non-empty document, in order.
            - Special-token parsing is disabled: corpus prose quoting a marker such as
              <|im_end|> is tokenized as the characters it is made of rather than being
              promoted to the control token of the same name.
            - With append_eos each sequence is closed by the tokenizer's end-of-sequence
              marker, which is what keeps packed windows from teaching the model that one
              document continues into the next.
    !*/

// ----------------------------------------------------------------------------------------

    supervised_example encode_supervised_example(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const chat_record& record,
        bool supervise_eos = true
    );
    /*!
        ensures
            - #prompt is encode_turn(tok, fmt, record.system, record.user, true), that is
              the template's system block, the user message and the assistant header,
              rendered exactly as the inference path renders them.
            - #response is the tokenized answer, followed by the end-of-sequence marker
              when supervise_eos is set. That position is supervised like any other:
              stopping is part of what the model has to learn, and an answer trained
              without it runs until the token budget is exhausted.
    !*/

    std::vector<supervised_example> encode_supervised_examples(
        const hf_tokenizer& tok,
        const chat_template_formatter& fmt,
        const std::vector<chat_record>& records,
        bool supervise_eos = true
    );
    /*!
        ensures
            - Applies encode_supervised_example() to each record, dropping those whose
              prompt or response tokenizes to nothing.
    !*/

// ----------------------------------------------------------------------------------------

    struct length_profile
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Token-length statistics of a set of supervised examples: the number of
                examples, the median, 90th, 99th percentile and maximum of the total
                length, and the mean split between prompt and response.

                The window length is the one hyper-parameter that ruins a fine-tuning run
                without saying so. Below the bulk of the examples it discards most of the
                corpus; far above it, every batch is mostly padding. The percentiles say
                where to put it, and dataset_report then says what the choice cost.
        !*/
    };

    length_profile profile_lengths(
        const std::vector<supervised_example>& examples
    );
    /*!
        ensures
            - Returns the length statistics of the given examples, with every field zero
              when the set is empty.
            - #sorted_totals holds every total length in ascending order, so an arbitrary
              coverage can be asked for after the fact through quantile() and
              coverage_at().
    !*/

    long suggest_window_length(
        const length_profile& profile,
        double coverage = 0.95,
        long granularity = 64,
        long max_window = 0
    );
    /*!
        requires
            - granularity > 0
        ensures
            - Returns the shortest window keeping the requested fraction of the examples
              whole, rounded up to a multiple of granularity, capped at max_window when
              that is non-zero, and never below granularity.
            - The window length is a property of the fine-tuning data, not of the model:
              nothing in the architecture ties them, rotary positions extend to any length
              and the KV cache is sized at inference. Training on short windows and serving
              on long ones is normal, and it is what keeps a stage affordable, since
              attention cost grows with the square of the window while almost every example
              may sit far below the model's capacity.
            - Coverage is a deliberate trade: pushing it to one lets a handful of outliers
              set the cost of every batch, which is why the default stops short and leaves
              the remainder to the overflow policy of the dataset builder.
    !*/
}

#endif // DLIB_DNN_FINETUNING_DATA_ABSTRACT_H_
