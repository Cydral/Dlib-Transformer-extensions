// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_CPU_RUNTIME_H_
#define ALBIREO_ENGINE_CPU_RUNTIME_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache.h"
#include "model_file.h"
#include "offload.h"
#include "qkernels.h"
#include "runtime.h"
#include "thread_pool.h"

namespace albireo
{

    class cpu_runtime : public model_runtime
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The forward pass of a decoder-only model on the processor.

                The weights stay in the memory mapping of the GGUF file, in their
                quantized form, and every product goes through the kernels of
                qkernels.h.  Activations are single-precision floats; keys and values are
                stored in a kv_pool, in half precision or in 8-bit integers by groups of
                kv_q8_group values (kv_format), one paged sequence per request slot.

                The numerical path follows the reference implementations of the
                supported families: pre-normalization residual blocks with RMS norms,
                grouped-query attention with rotary positions (consecutive pairs or
                halves, with linear, YaRN, llama 3 or LongRoPE frequency adjustments),
                optional QKV biases and per-head QK normalization, SwiGLU feed-forward
                blocks or mixtures of SwiGLU experts, and the global scale factors of the
                families that declare them.  The fused matrices of phi3 (attn_qkv, and
                ffn_up with the gate) are used as row ranges, one product per range.

                Attention runs over the paged keys of each sequence directly, by tiles: a
                task takes the heads that share one key/value head for up to tile_rows
                rows of consecutive tokens of one sequence, and reads the cache block after
                block with a running maximum and sum (the softmax of FlashAttention), so
                that every key and value read from memory serves all the rows of the tile.
                When the tiles are fewer than the threads, as in the generation of a single
                answer, the positions of each tile are split into ranges that tasks read
                for all the heads of the group, and the partial sums of the ranges are
                merged (the split of FlashDecoding).

                With a matmul_offload, the products with the weight matrices go through it,
                and it decides, product by product, whether a device computes them.  When the
                offload computes whole layers (layer_pass), the runtime of a dense model hands
                it the layers of its passes, once the logits of passes of the device over
                probe_tokens tokens, and over the tokens from which it expands the weights by
                tiles, agreed with those of the processor alone: the hidden states then stay
                on the device, and the processor keeps the attention.  The batches with
                DeepStack slices keep the products, as does every pass after a failure of the
                device.

                The hybrid decoders (model_config::layer_types) take a path of their own:
                their residual stream is not normalized, each block turning it on its sphere
                towards the output of its mixer, then towards the output of its mixture of
                experts.  A Mamba-3 layer keeps, for each sequence, the state of its channels,
                the angles of its rotations and what the trapezoid rule reads of the previous
                token, in single precision; an attention layer keeps its keys and values in
                the kv_pool, which holds the attention layers only, and the last key and value
                before their shift.  The tokens of a sequence are read in order by each layer,
                the heads of a Mamba-3 layer in parallel.  Such a state cannot give back the
                tokens it read: the runtime saves it at the end of the passes that read several
                tokens of a sequence, the chunks of a prompt, and a sequence goes back to these
                lengths only (restorable_length()), saved_states of them at most.

                When the offload computes the layers of a hybrid decoder (hybrid_pass), the
                runtime hands it the passes once logits of the device agreed with those of the
                processor alone, as for a dense decoder: the hidden states, the Mamba-3 layers,
                the experts and the geodesic updates then stay on the device, and the processor
                keeps the attention proper of the attention layers.  The states of the Mamba-3
                layers of a sequence the device computes live on the device; the runtime brings
                them back to save them at the end of a chunk of a prompt, to copy the sequence,
                to compute a pass itself and before its model rests.  A pass the device fails
                returns to the processor with the states its sequences had before it; a
                sequence whose states were lost with the device goes back to its saved states.

                A batch that gives the rotary positions of its tokens (forward_batch::
                positions) rotates each pair of a head by the position component its
                section names (model_config::rope_sections), and the rows it gives as
                embeddings take them in place of the embedding table, their DeepStack
                slices being added to their hidden states after the first layers.  A batch
                without them runs the path of the text models unchanged.
        !*/
    public:
        cpu_runtime(std::shared_ptr<model_file> file, const model_config& config, thread_pool& pool,
            size_t kv_budget_bytes, long kv_block_positions = 256, kv_format kv_type = kv_format::f16);

        const model_config& config() const override { return cfg_; }

        int create_sequence() override;
        void release_sequence(int seq) override;
        long sequence_length(int seq) const override;
        void truncate_sequence(int seq, long length) override;
        bool reserve_sequence(int seq, long length) override;
        bool copy_sequence(int src, int dst, long length) override;
        long restorable_length(int seq, long length) const override;

        void forward(const forward_batch& batch, std::vector<float>& logits) override;

        size_t kv_bytes_used() const override { return kv_.used_blocks() * kv_.block_bytes(); }
        size_t kv_bytes_budget() const override { return kv_.max_blocks() * kv_.block_bytes(); }
        size_t kv_bytes_per_token() const override
        {
            return 2 * static_cast<size_t>(kv_.layers()) * cfg_.n_kv_heads * kv_row_bytes(kv_.format(), cfg_.head_dim);
        }
        kv_format kv_type() const { return kv_.format(); }
        // The check of the memory of the system before each new block of keys and values (kv_pool::set_guard).
        void set_kv_guard(std::function<bool(size_t, size_t)> guard) { kv_.set_guard(std::move(guard)); }
        void trim_memory() override;
        void set_busy(bool busy) override;
        size_t weight_bytes() const override { return static_cast<size_t>(file_->size()); }

        std::string description() const override;

        // Hands the products, or the layers, of the following passes to a device.
        void set_offload(std::unique_ptr<matmul_offload> offload);

        // The bytes of the file of the model whose products the device computes from its copies
        // (matmul_offload::mapped_bytes()), 0 without a device.
        size_t offloaded_bytes() const { return offload_ ? offload_->mapped_bytes() : 0; }

        // The bytes of the copies of the device the system charges to the commit of the engine
        // (matmul_offload::charged_bytes()), 0 without a device.
        size_t charged_bytes() const { return offload_ ? offload_->charged_bytes() : 0; }

        /*!
            requires
                - no forward pass has run, or the rotary divisors stay the same.
            ensures
                - Chooses the rotary divisors of a model with LongRoPE for sequences of up to
                  positions tokens (select_rope_factors()).
        !*/
        void set_context(long positions);

        const std::vector<float>& hidden() const override { return hidden_out_; }

    private:
        struct layer
        {
            std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, bq, bk, bv;
            weight_matrix wq, wk, wv, wo;
            weight_matrix w_gate, w_up, w_down;
            weight_matrix router;
            std::vector<weight_matrix> e_gate, e_up, e_down;

            /*
                The layers of a hybrid decoder, which use wq, wo, q_norm, k_norm, router, e_up
                and e_down above with their own tensors: the projections of a Mamba-3 layer
                and the constants of its heads, the factors of the keys and values of an
                attention with the projections of their shifts, of the weights of the noise
                heads and of the gate, the projections to and from the latent space of the
                experts, the shared expert, and the scale and bias of the two geodesic updates.
            */
            weight_matrix ssm_in, ssm_dyn;
            std::vector<float> ssm_b_norm, ssm_c_norm, ssm_b_bias, ssm_c_bias, ssm_dt_bias, ssm_d;
            std::vector<float> mimo_x, mimo_z, mimo_o;
            weight_matrix wa_k, wb_k, wa_v, wb_v, shift_k, shift_v, noise, attn_gate;
            std::vector<float> softmax_scaler;
            weight_matrix latent_down, latent_up, shared_up, shared_down;
            std::vector<float> expert_bias;
            float mixer_scale = 1.0f, mixer_bias = 0.0f, ffn_scale = 1.0f, ffn_bias = 0.0f;
        };

        // The states of the layers of a hybrid decoder for a sequence of length tokens.
        struct recurrent_state
        {
            long length = 0;
            std::vector<float> values;
        };

        // What a sequence of a hybrid decoder keeps beside its keys and values.
        struct recurrent_sequence
        {
            recurrent_state live;
            // The states saved, by increasing length.
            std::vector<recurrent_state> saved;
            // The length the prompt being read reached at the end of the pass before, -1 when the
            // next pass begins another prompt.
            long reading = -1;
            // Whether the last saved state is one of the chunks of that prompt, which its next
            // chunk replaces.
            bool own = false;
            // Whether live holds the states of the Mamba-3 layers; the device alone holds them otherwise.
            bool on_host = true;
            // Whether these states were lost with the device: the sequence goes back to a saved state.
            bool lost = false;
        };

        // The states a sequence of a hybrid decoder keeps saved, at most.
        static const size_t saved_states = 2;

        // The buffers of one thread of a Mamba-3 layer.
        struct mixer_work
        {
            aligned_floats k, q, cos, sin, before, now, y, o, gate;
        };

        // The rows of queries of an attention tile, at most: the scores of a tile over one
        // block of the cache stay within a few tens of kilobytes.
        static const long tile_rows = 64;

        // The positions of a range of the split of the attention, at least (attention()).
        static const long part_positions = 256;

        // The tokens of the passes that compare the layers of the device with the processor.
        static const long probe_tokens = 8;

        /*
            The relative distance of the logits of the device and of the processor a probe allows,
            and the distance its probe of the attention of the device over keys and values in 8
            bits allows, whose rounding turns the small differences of the products into larger
            ones.
        */
        static constexpr double probe_tolerance = 0.01;
        static constexpr double q8_probe_tolerance = 0.05;

        /*
            The tokens of the passes, at most, that the device attends token by token when it can,
            as the steps of a generation, and that the runtime times apart from the longer ones:
            a device attends a longer pass, as a chunk of a prompt, over all its tokens together
            (layer_pass::attends_many()), or leaves its attention to the processor, which reads
            each block of keys and values once for a tile of tokens.
        */
        static const long attention_tokens = few_attention_tokens;

        // The tokens of the probe of the attention of the longer passes of the device over keys and values in 8 bits.
        static const long long_attention_probe_tokens = 2 * few_attention_tokens;

        // Failures for lack of device memory in a row after which the layers stay on the processor.
        static const int max_layer_pass_memory_failures = 4;

        // The passes that follow a check of the layer pass put off for lack of room, before the next one.
        static const int layer_pass_check_interval = 64;

        // The passes, at most, between two probes of an attention of the device that found no room.
        static const int max_attention_check_interval = 4096;

        // Consecutive tokens of one segment of the batch, which attend together.
        struct attention_tile
        {
            int sequence;
            long start;     // first token in the batch
            long count;
        };

        // The buffers of one thread of the attention.
        struct attention_work
        {
            aligned_floats q, s, o, max, sum, scratch;
        };

        void rms_norm_rows(const float* x, const float* gamma, float* y, long rows, long width, float eps);
        void ensure_rope(long positions);
        void apply_rope(float* rows, long n_rows, long n_heads, const std::vector<long>& positions);
        // The rotation of the tokens of a batch that gives their positions: the cosines and
        // sines of every pair of each token, gathered once per pass (prepare_given_positions),
        // then applied to the queries and keys of every layer.
        void prepare_given_positions(const forward_batch& batch, long T);
        void apply_given_rope(float* rows, long n_rows, long n_heads);
        // The rows of the batch that come from an encoder: their embeddings, and the slice of
        // a DeepStack layer added after it.
        void check_given_embeddings(const forward_batch& batch, long T) const;
        void add_given_embeddings(const forward_batch& batch, long slice);
        // The keys and values of the tokens of a batch, written in the blocks of their sequences.
        void store_keys_values(long layer, const forward_batch& batch);
        void attention(long layer, const forward_batch& batch, const std::vector<long>& positions);
        void moe_ffn(const layer& L, long n_tokens);
        void embed(const std::vector<int>& tokens);
        // The forward pass of a hybrid decoder, and its blocks: each writes in tmp_ the output
        // that geodesic_update() then turns the hidden states towards.
        void load_hybrid_layer(const model_file& f, long layer);
        /*
            The forward pass of a hybrid decoder, with the layers on the device through
            hybrid_pass_ when on_device is true; false when the device did not take the pass,
            which then did nothing.  A pass the device fails leaves the states of its sequences
            as they were before it and throws device_failure.
        */
        bool run_hybrid(const forward_batch& batch, std::vector<float>& logits, bool on_device);
        void mamba3_mixer(long layer, const forward_batch& batch);
        // With on_device, the device computes the products of the layer around the attention.
        void differential_attention(long layer, const forward_batch& batch, bool on_device);
        // The description of the model for the device, or false when its experts do not lie one after the other.
        bool describe_hybrid(hybrid_model& m) const;
        /*
            Brings the states of the Mamba-3 layers of a sequence back from the device when it
            alone holds them; false when the device failed, the states then being lost.
        */
        bool fetch_states(int seq);
        // The same for every sequence, as before the device may lose them.
        void fetch_all_states();
        // What the attention layers keep of the last token of each sequence of a pass, saved and brought back.
        void save_attention_states(const forward_batch& batch);
        void restore_states(const forward_batch& batch);
        // The probes of the layers of the device for a hybrid decoder, as probe_layer_pass() and check_layer_pass().
        int probe_hybrid_pass(long n, int attempt, std::string& detail);
        void check_hybrid_pass();
        void drop_hybrid_pass(const std::string& why);
        void latent_experts(const layer& L, long n_tokens);
        void geodesic_update(const float* g, float scale, float bias, long layer, long n_tokens);
        // The state of a sequence of a hybrid decoder at one of the lengths it holds, or null.
        const recurrent_state* state_at(int seq, long length) const;
        /*
            The forward pass, with the layers on the device through layer_pass_ when on_device is
            true; false when the device did not take the pass, which then did nothing.
        */
        bool run(const forward_batch& batch, std::vector<float>& logits, bool on_device);
        bool takes_layer_pass(const forward_batch& batch) const;
        // The tokens of a pass whose attention the device may compute, in plan_; false when they cannot be planned.
        bool plan_attention(const forward_batch& batch, bool given);
        int probe_layer_pass(long n, kv_format format, bool attention, double tolerance, bool& attended,
            std::string& detail);
        // probe_layer_pass() for a pass the device attends, made again without its graphs of launches when it differs.
        int probe_attended_pass(long n, kv_format format, double tolerance, bool& attended, std::string& detail);
        void check_layer_pass();
        // Probes again the attention of the device, which found no room at its probe.
        void check_attention();
        // The size of the probes of the layer pass.
        long probe_size() const;
        void drop_layer_pass(const std::string& why);
        /*
            The seconds of a pass of n tokens of a sequence of its own, with the layers on the
            device (on_device) or as the processor computes them without it: the shortest of
            timed passes that follow warm passes of the same sequence, each of n tokens.
            Negative when the keys and values or the device found no room for the pass.
        */
        double time_pass(long n, int warm, int timed, bool on_device);
        /*
            Times the passes of the device against the processor, for the passes of a few tokens
            (single tokens that follow one another, as a generation) and for the longer ones (a
            pass of many tokens, when many > 1), and sets device_few_ and device_many_; a measure
            that found no room leaves the device its passes.  detail describes the measures.
            Returns false when the processor computes both kinds faster.
        */
        bool time_passes(long many, std::string& detail);
        // Whether the device serves the passes of the size of batch (device_few_, device_many_).
        bool device_serves(const forward_batch& batch) const;
        // Y = X W^T through the offload when there is one, with the kernels of qkernels.h
        // otherwise.
        void product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride,
            qk_path path = qk_path::automatic);
        /*
            The products of group_, which read nothing of one another, for a pass of n_tokens
            tokens: in one parallel section for the few tokens of a generation step
            (qk_matmul_group()), unless the offload measures or computes one of them, and one
            after the other, as product() computes them, for the passes of a prompt.
        */
        void products(long n_tokens);
        void add_product(const weight_matrix& W, const float* X, long n, long x_stride, float* Y, long y_stride);
        /*
            The rows of the buffers of the experts for the tokens routed_ gives each expert, and
            their number.  Together, the experts follow one another, expert_rows_ holding the
            first row of each, -1 for an expert no token reaches; otherwise each expert takes
            the buffers in turn.
        */
        long lay_out_experts(long n_experts, bool together);

        std::shared_ptr<model_file> file_;
        model_config cfg_;
        thread_pool& pool_;
        std::unique_ptr<matmul_offload> offload_;
        // Destroyed before offload_, which they use: the layers of a dense decoder, or of a hybrid one.
        std::unique_ptr<layer_pass> layer_pass_;
        std::unique_ptr<hybrid_pass> hybrid_pass_;
        bool layer_pass_checked_ = false;
        int layer_pass_memory_failures_ = 0;
        int layer_pass_wait_ = 0;
        /*
            The passes before the attention of the device, whose probe found no room, is probed
            again (0 when no probe is due), and their number, which doubles at each new lack of
            room.
        */
        int attention_wait_ = 0;
        int attention_interval_ = 0;
        /*
            Whether the device serves the passes of at most attention_tokens tokens, those of a
            generation, and the longer ones, those of the prompts: each kind stays on the device
            only when the device computes it faster than the processor (time_passes()).
        */
        bool device_few_ = true;
        bool device_many_ = true;
        // The products of the probe passes of the processor do not go to the offload.
        bool host_only_ = false;
        // Whether the device may attend the passes (false in a probe of the layers alone), and attended the last one.
        bool attention_allowed_ = true;
        bool attended_ = false;
        // Whether the attention of the longer passes of the device differed from the processor at its probe.
        bool long_attention_differs_ = false;
        attention_plan plan_;
        kv_pool kv_;
        std::vector<std::unique_ptr<kv_sequence>> sequences_;
        std::vector<int> free_sequences_;
        // The pool of each sequence: kv_, or the pool in half precision of a probe of the device.
        std::vector<kv_pool*> sequence_pools_;
        kv_pool& pool_of(int seq) const { return *sequence_pools_[static_cast<size_t>(seq)]; }

        std::vector<layer> layers_;
        /*
            Hybrid decoders: the layer of the pool that holds the keys and values of each
            attention layer (-1 for the others), the place of the state of each layer in the
            values of a recurrent_state and their number, and the states of the sequences.
        */
        std::vector<long> kv_layers_;
        std::vector<size_t> state_offsets_;
        size_t state_floats_ = 0;
        std::vector<recurrent_sequence> recurrent_;
        std::vector<mixer_work> mixer_work_;
        aligned_floats proj_, dyn_, factor_a_, factor_b_, shift_, noise_, latent_, routed_latent_;
        std::vector<float> last_keys_;
        // The segments of a pass of the device, and what its attention layers kept before it.
        std::vector<hybrid_pass::segment> device_segments_;
        std::vector<float> attention_states_;
        std::vector<double> turns_;
        weight_matrix token_embd_, output_;
        std::vector<float> output_norm_;

        // Rotary tables: cos and sin for every position computed so far.
        std::vector<float> rope_cos_, rope_sin_;
        long rope_positions_ = 0;
        // The position component of each rotary pair (0 time, 1 row, 2 column), and the
        // cosines and sines of the tokens of a batch that gives their positions.
        std::vector<int> pair_component_;
        std::vector<float> given_cos_, given_sin_;

        // Work buffers, reused from one pass to the next.
        aligned_floats x_, xn_, q_, k_, v_, ctx_, tmp_, gate_, up_;
        aligned_floats router_logits_, expert_x_, expert_gate_, expert_up_, expert_y_;
        std::vector<float> hidden_out_, router_probs_;
        std::vector<long> router_order_;
        std::vector<std::vector<std::pair<long, float>>> routed_;
        std::vector<qk_scratch> scratch_;
        std::vector<qk_product> group_;
        // The first row of each expert a few tokens reach in the buffers of the experts, -1 for the others.
        std::vector<long> expert_rows_;
        std::vector<attention_work> attention_work_;
        std::vector<attention_tile> tiles_;
        // The partial sums of the ranges of a split attention, and the sequence of each token of a batch.
        aligned_floats partial_;
        std::vector<int> token_sequences_;
        std::vector<long> positions_;
    };

}

#endif // ALBIREO_ENGINE_CPU_RUNTIME_H_
