// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_RUNTIME_H_
#define ALBIREO_ENGINE_RUNTIME_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model_config.h"

namespace albireo
{

    struct forward_batch
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The tokens one forward pass computes, possibly from several sequences:
                the next token of every sequence being generated, and a chunk of the
                prompt of a sequence being read.  Every product of the pass then runs once
                over all of them, which is what batching buys: the weights are read once
                for the whole batch instead of once per sequence.

                segments lists, for each sequence present, the range of tokens that belong
                to it.  The tokens of a segment are consecutive positions that extend the
                sequence: the first one sits at the current length of the sequence.

                logit_rows lists the tokens whose next-token distribution is needed, the
                last token of each segment that ends a prompt or continues a generation.
                With hidden_only, the pass stops at their normalized hidden states, which
                is what an embedding model returns.
        !*/
        struct segment
        {
            int sequence = -1;
            long start = 0;
            long count = 0;
        };

        std::vector<int> tokens;
        std::vector<segment> segments;
        std::vector<long> logit_rows;
        bool hidden_only = false;

        void clear()
        {
            tokens.clear();
            segments.clear();
            logit_rows.clear();
            hidden_only = false;
        }
    };

    class model_runtime
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The forward pass of one loaded model and the storage of the keys and values
                of its sequences.  The scheduler drives it through this interface, so that
                the processor runtime, with its quantized kernels, and the tensor runtime,
                with the devices of dlib, are interchangeable.

                Sequences are numbered by the runtime.  A sequence keeps its positions
                between calls until it is truncated or released, which is how the
                scheduler reuses the beginning of a conversation from one request to the
                next.
        !*/
    public:
        virtual ~model_runtime() {}

        virtual const model_config& config() const = 0;

        // Sequences.
        virtual int create_sequence() = 0;
        virtual void release_sequence(int seq) = 0;
        virtual long sequence_length(int seq) const = 0;
        virtual void truncate_sequence(int seq, long length) = 0;
        // Makes room for the given total length; false when memory is exhausted.
        virtual bool reserve_sequence(int seq, long length) = 0;
        // Makes dst hold the first length positions of src, whatever it held before;
        // false when memory is exhausted.  This is how several requests start from the
        // same cached prefix, such as a long system prompt, without computing it again.
        virtual bool copy_sequence(int src, int dst, long length) = 0;

        // Runs the batch, appends its keys and values to the sequences and writes, for each
        // entry of logit_rows, a row of config().vocab logits into logits, unless the batch
        // asks for the hidden states only.
        virtual void forward(const forward_batch& batch, std::vector<float>& logits) = 0;

        // The hidden states after the final normalization of the logit_rows of the last
        // forward, config().d_model floats each.
        virtual const std::vector<float>& hidden() const = 0;

        // Memory held for keys and values, in bytes, and the most it may hold.
        virtual size_t kv_bytes_used() const = 0;
        virtual size_t kv_bytes_budget() const = 0;
        virtual size_t kv_bytes_per_token() const = 0;

        // Returns to the system the memory of the keys and values no sequence holds.
        virtual void trim_memory() = 0;

        // Bytes of weights the runtime keeps resident (mapped files count for their size).
        virtual size_t weight_bytes() const = 0;

        virtual std::string description() const = 0;
    };

}

#endif // ALBIREO_ENGINE_RUNTIME_H_
