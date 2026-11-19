// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_TESTS_TINY_GGUF_H_
#define ALBIREO_TESTS_TINY_GGUF_H_

/*
    A GGUF file of a tiny model of the qwen3 architecture, written in memory, for the tests
    that load models: two layers of width 64, a byte-level vocabulary with the control
    tokens of ChatML, a chat template naming them (by default that of a model whose
    reasoning a request switches, or the one given to build()), and weights in single
    precision drawn from a fixed sequence.  The file computes nothing meaningful, and loads
    and runs as a published model does.

    Every field of the header is recorded with its offset, its size and a label, so that a
    test can damage one field at a time and check that the loader refuses the file.

    build_hybrid() writes a tiny model of the olala architecture instead: four layers of
    width 64, two Mamba-3 layers with several inputs and outputs and two differential
    attentions, each followed by a mixture of experts in a latent space beside a shared
    expert, with the same vocabulary.
*/

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace albireo_test
{

    struct gguf_field
    {
        std::string label;   // "kv qwen3.block_count", "dims token_embd.weight", ...
        size_t offset = 0;
        size_t size = 0;
    };

    struct tiny_gguf
    {
        std::vector<unsigned char> bytes;
        std::vector<gguf_field> fields;
        size_t header_end = 0;   // first byte of the tensor data

        const gguf_field* find(const std::string& label) const
        {
            for (const auto& f : fields)
                if (f.label == label) return &f;
            return nullptr;
        }

        // A copy with the field of the given label set to value (little endian, truncated
        // to the size of the field).
        std::vector<unsigned char> with(const std::string& label, uint64_t value) const
        {
            std::vector<unsigned char> b = bytes;
            const gguf_field* f = find(label);
            if (f)
                for (size_t i = 0; i < f->size; ++i) b[f->offset + i] = static_cast<unsigned char>(value >> (8 * i));
            return b;
        }

        bool save(const std::string& path, const std::vector<unsigned char>& b) const
        {
            std::FILE* f = std::fopen(path.c_str(), "wb");
            if (!f) return false;
            const bool ok = b.empty() || std::fwrite(b.data(), 1, b.size(), f) == b.size();
            return std::fclose(f) == 0 && ok;
        }
    };

    class tiny_gguf_writer
    {
    public:
        // The width of the heads, 16 by default; 32 gives rows of whole groups of 8-bit keys and values.
        tiny_gguf_writer& head_dim(long hd)
        {
            head_dim_ = hd;
            return *this;
        }

        // Biases of the queries, keys and values in every layer, as the models of Qwen2 have.
        tiny_gguf_writer& biases(bool on)
        {
            biases_ = on;
            return *this;
        }

        // The chat template of a model whose reasoning a request switches.
        static std::string switchable_template()
        {
            return "{%- if tools %}<tool_call>{%- endif %}{{- '<|im_start|>' + message.role + '\\n' }}"
                "{%- if enable_thinking is defined %}<think>{%- endif %}";
        }

        // The chat template of a model that reasons before every answer, as the thinking variants of Qwen3-VL.
        static std::string thinking_template()
        {
            return "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + "
                "'<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}"
                "{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}";
        }

        tiny_gguf build(const std::string& chat_template = switchable_template())
        {
            out_ = tiny_gguf();
            const long d = 64, ff = 128, heads = 4, kv_heads = 2, layers = 2;
            const long hd = head_dim_;
            std::vector<std::string> tokens;
            std::vector<int32_t> types;
            const long vocab = vocabulary(tokens, types);

            std::vector<tensor> list;
            list.push_back({ "token_embd.weight", { uint64_t(d), uint64_t(vocab) } });
            for (long l = 0; l < layers; ++l)
            {
                const std::string p = "blk." + std::to_string(l) + ".";
                list.push_back({ p + "attn_norm.weight", { uint64_t(d) } });
                list.push_back({ p + "attn_q.weight", { uint64_t(d), uint64_t(heads * hd) } });
                list.push_back({ p + "attn_k.weight", { uint64_t(d), uint64_t(kv_heads * hd) } });
                list.push_back({ p + "attn_v.weight", { uint64_t(d), uint64_t(kv_heads * hd) } });
                list.push_back({ p + "attn_output.weight", { uint64_t(heads * hd), uint64_t(d) } });
                if (biases_)
                {
                    list.push_back({ p + "attn_q.bias", { uint64_t(heads * hd) } });
                    list.push_back({ p + "attn_k.bias", { uint64_t(kv_heads * hd) } });
                    list.push_back({ p + "attn_v.bias", { uint64_t(kv_heads * hd) } });
                }
                list.push_back({ p + "attn_q_norm.weight", { uint64_t(hd) } });
                list.push_back({ p + "attn_k_norm.weight", { uint64_t(hd) } });
                list.push_back({ p + "ffn_norm.weight", { uint64_t(d) } });
                list.push_back({ p + "ffn_gate.weight", { uint64_t(d), uint64_t(ff) } });
                list.push_back({ p + "ffn_up.weight", { uint64_t(d), uint64_t(ff) } });
                list.push_back({ p + "ffn_down.weight", { uint64_t(ff), uint64_t(d) } });
            }
            list.push_back({ "output_norm.weight", { uint64_t(d) } });

            u32(0x46554747u);
            u32(3);
            field("tensor_count", 8);
            u64(list.size());
            field("kv_count", 8);
            const size_t kv_count_at = out_.bytes.size();
            u64(0);
            uint64_t kv = 0;
            kv_str("general.architecture", "qwen3"); ++kv;
            kv_str("general.name", "tiny"); ++kv;
            kv_u64("qwen3.block_count", layers); ++kv;
            kv_u64("qwen3.context_length", 4096); ++kv;
            kv_u64("qwen3.embedding_length", d); ++kv;
            kv_u64("qwen3.feed_forward_length", ff); ++kv;
            kv_u64("qwen3.attention.head_count", heads); ++kv;
            kv_u64("qwen3.attention.head_count_kv", kv_heads); ++kv;
            kv_u64("qwen3.attention.key_length", hd); ++kv;
            kv_u64("qwen3.attention.value_length", hd); ++kv;
            kv_u64("qwen3.rope.dimension_count", hd); ++kv;
            kv_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f); ++kv;
            kv_f32("qwen3.rope.freq_base", 1000000.0f); ++kv;
            kv_u64("general.alignment", 32); ++kv;
            kv_str("tokenizer.ggml.model", "gpt2"); ++kv;
            kv_str("tokenizer.ggml.pre", "qwen2"); ++kv;
            kv_strings("tokenizer.ggml.tokens", tokens); ++kv;
            kv_ints("tokenizer.ggml.token_type", types); ++kv;
            kv_strings("tokenizer.ggml.merges", { "\xc4\xa0 t", "h e" }); ++kv;
            kv_u64("tokenizer.ggml.eos_token_id", 258); ++kv;
            kv_u64("tokenizer.ggml.bos_token_id", 256); ++kv;
            kv_u64("tokenizer.ggml.add_bos_token", 0); ++kv;
            kv_str("tokenizer.chat_template", chat_template); ++kv;
            for (size_t i = 0; i < 8; ++i) out_.bytes[kv_count_at + i] = static_cast<unsigned char>(kv >> (8 * i));

            write_tensors(list);
            return out_;
        }

        /*
            A model of the olala architecture.  The values of its vectors stand around those of
            a trained model: scales and factors around one, steps of the states below one.
        */
        tiny_gguf build_hybrid()
        {
            out_ = tiny_gguf();
            const long d = 64, layers = 4;
            const long ssm_heads = 8, ssm_hd = 16, state = 16, rank = 2, angles = 4;
            const long heads = 8, kv_heads = 2, hd = 32, tpa = 2, signal = heads - kv_heads;
            const long experts = 8, used = 2, hidden = 32, latent = 16, shared = 48;
            const std::string types = "MVMV";
            std::vector<std::string> tokens;
            std::vector<int32_t> token_types;
            const long vocab = vocabulary(tokens, token_types);

            auto u = [](long v) { return static_cast<uint64_t>(v); };
            std::vector<tensor> list;
            // The hidden states have a length of one: an output matrix of this size gives logits of a few units.
            list.push_back({ "output.weight", { u(d), u(vocab) }, 0.0f, 8.0f });
            list.push_back({ "token_embd.weight", { u(d), u(vocab) } });
            for (long l = 0; l < layers; ++l)
            {
                const std::string p = "blk." + std::to_string(l) + ".";
                list.push_back({ p + "geo_mixer_scale", { 1 }, 1.0f, 0.2f });
                list.push_back({ p + "geo_mixer_bias", { 1 }, 0.02f, 0.02f });
                list.push_back({ p + "geo_mlp_scale", { 1 }, 1.0f, 0.2f });
                list.push_back({ p + "geo_mlp_bias", { 1 }, 0.02f, 0.02f });
                if (types[static_cast<size_t>(l)] == 'M')
                {
                    list.push_back({ p + "ssm_in.weight", { u(d), u(ssm_heads * (2 * ssm_hd + 3)) }, 0.0f, 0.8f });
                    list.push_back({ p + "ssm_in_dyn.weight", { u(d), u(2 * rank * state + angles) }, 0.0f, 0.8f });
                    list.push_back({ p + "attn_output.weight", { u(ssm_heads * ssm_hd), u(d) } });
                    list.push_back({ p + "ssm_b_norm.weight", { u(state) }, 1.0f, 0.2f });
                    list.push_back({ p + "ssm_c_norm.weight", { u(state) }, 1.0f, 0.2f });
                    list.push_back({ p + "ssm_b_bias", { u(state), u(rank), u(ssm_heads) }, 1.0f, 0.6f });
                    list.push_back({ p + "ssm_c_bias", { u(state), u(rank), u(ssm_heads) }, 1.0f, 0.6f });
                    list.push_back({ p + "ssm_dt_bias", { u(ssm_heads) }, -1.0f, 1.0f });
                    list.push_back({ p + "ssm_d", { u(ssm_heads) }, 1.0f, 0.6f });
                    list.push_back({ p + "ssm_mimo_x", { u(ssm_hd), u(rank), u(ssm_heads) }, 0.5f, 0.4f });
                    list.push_back({ p + "ssm_mimo_z", { u(ssm_hd), u(rank), u(ssm_heads) }, 1.0f, 0.6f });
                    list.push_back({ p + "ssm_mimo_o", { u(ssm_hd), u(rank), u(ssm_heads) }, 0.5f, 0.4f });
                }
                else
                {
                    list.push_back({ p + "attn_q.weight", { u(d), u(heads * hd) }, 0.0f, 0.8f });
                    list.push_back({ p + "attn_wa_k.weight", { u(d), u(kv_heads * tpa) }, 0.0f, 1.6f });
                    list.push_back({ p + "attn_wb_k.weight", { u(d), u(tpa * hd) }, 0.0f, 1.6f });
                    list.push_back({ p + "attn_wa_v.weight", { u(d), u(kv_heads * tpa) }, 0.0f, 1.6f });
                    list.push_back({ p + "attn_wb_v.weight", { u(d), u(tpa * hd) }, 0.0f, 1.6f });
                    list.push_back({ p + "attn_shift_k.weight", { u(d), u(kv_heads) }, 0.0f, 1.6f });
                    list.push_back({ p + "attn_shift_v.weight", { u(d), u(kv_heads) }, 0.0f, 1.6f });
                    list.push_back({ p + "attn_lambda.weight", { u(d), u(kv_heads) }, 0.0f, 1.6f });
                    list.push_back({ p + "attn_gate.weight", { u(d), u(signal * hd) }, 0.0f, 0.8f });
                    list.push_back({ p + "attn_output.weight", { u(signal * hd), u(d) } });
                    list.push_back({ p + "attn_q_norm.weight", { u(hd) }, 1.0f, 0.2f });
                    list.push_back({ p + "attn_k_norm.weight", { u(hd) }, 1.0f, 0.2f });
                    list.push_back({ p + "attn_softmax_scaler", { u(heads) }, 2.0f, 2.0f });
                }
                list.push_back({ p + "ffn_gate_inp.weight", { u(d), u(experts) }, 0.0f, 4.0f });
                list.push_back({ p + "exp_probs_b", { u(experts) }, 0.0f, 0.1f });
                list.push_back({ p + "ffn_latent_down.weight", { u(d), u(latent) }, 0.0f, 0.8f });
                list.push_back({ p + "ffn_latent_up.weight", { u(latent), u(d) } });
                list.push_back({ p + "ffn_up_exps.weight", { u(latent), u(hidden), u(experts) }, 0.0f, 0.8f });
                list.push_back({ p + "ffn_down_exps.weight", { u(hidden), u(latent), u(experts) } });
                list.push_back({ p + "ffn_up_shexp.weight", { u(d), u(shared) }, 0.0f, 0.8f });
                list.push_back({ p + "ffn_down_shexp.weight", { u(shared), u(d) } });
            }

            u32(0x46554747u);
            u32(3);
            field("tensor_count", 8);
            u64(list.size());
            field("kv_count", 8);
            const size_t kv_count_at = out_.bytes.size();
            u64(0);
            uint64_t kv = 0;
            kv_str("general.architecture", "olala"); ++kv;
            kv_str("general.name", "tiny hybrid"); ++kv;
            kv_u64("olala.block_count", layers); ++kv;
            kv_u64("olala.context_length", 4096); ++kv;
            kv_u64("olala.embedding_length", d); ++kv;
            kv_u64("olala.feed_forward_length", 0); ++kv;
            kv_u64("olala.attention.head_count", heads); ++kv;
            kv_u64("olala.attention.head_count_kv", kv_heads); ++kv;
            kv_u64("olala.attention.key_length", hd); ++kv;
            kv_u64("olala.attention.value_length", hd); ++kv;
            kv_f32("olala.attention.layer_norm_rms_epsilon", 1e-6f); ++kv;
            kv_u64("olala.attention.sliding_window", 2048); ++kv;
            kv_f32("olala.attn_logit_softcapping", 2.0f); ++kv;
            kv_u64("olala.expert_count", experts); ++kv;
            kv_u64("olala.expert_used_count", used); ++kv;
            kv_u64("olala.expert_feed_forward_length", hidden); ++kv;
            kv_u64("olala.expert_shared_feed_forward_length", shared); ++kv;
            kv_u64("olala.expert_shared_count", 1); ++kv;
            kv_f32("olala.expert_weights_scale", 2.45f); ++kv;
            kv_u64("olala.expert_gating_func", 2); ++kv;
            kv_u64("olala.moe_latent_size", latent); ++kv;
            kv_u64("olala.ssm.inner_size", ssm_heads * ssm_hd); ++kv;
            kv_u64("olala.ssm.state_size", state); ++kv;
            kv_u64("olala.ssm.group_count", 1); ++kv;
            kv_u64("olala.mamba_mimo_dim", rank); ++kv;
            kv_u64("olala.mamba_headdim", ssm_hd); ++kv;
            kv_u64("olala.num_signal_heads", signal); ++kv;
            kv_u64("olala.num_noise_heads", kv_heads); ++kv;
            kv_u64("olala.tpa_rank", tpa); ++kv;
            kv_u64("olala.num_rope_angles", angles); ++kv;
            kv_str("olala.layer_types", types); ++kv;
            kv_f32("olala.gate_bias", 1.15f); ++kv;
            kv_u64("olala.gate_attn", 1); ++kv;
            kv_u64("olala.geodesic_update", 1); ++kv;
            kv_u64("olala.final_norm", 0); ++kv;
            kv_str("tokenizer.ggml.model", "gpt2"); ++kv;
            kv_str("tokenizer.ggml.pre", "qwen2"); ++kv;
            kv_strings("tokenizer.ggml.tokens", tokens); ++kv;
            kv_ints("tokenizer.ggml.token_type", token_types); ++kv;
            kv_strings("tokenizer.ggml.merges", { "\xc4\xa0 t", "h e" }); ++kv;
            kv_u64("tokenizer.ggml.eos_token_id", 258); ++kv;
            kv_u64("tokenizer.ggml.add_bos_token", 0); ++kv;
            // The words of the template that tell the format and its capabilities: channels, effort, tools.
            kv_str("tokenizer.chat_template", "{%- if tools %}{%- endif %}Reasoning effort: {{ reasoning_effort }}"
                "{{- '<|im_start|>' + message.role + '<|channel_start|>' }}"); ++kv;
            for (size_t i = 0; i < 8; ++i) out_.bytes[kv_count_at + i] = static_cast<unsigned char>(kv >> (8 * i));
            write_tensors(list);
            return out_;
        }

    private:
        long head_dim_ = 16;
        bool biases_ = false;

        /*
            A tensor in single precision.  Its values are drawn around center, spread wide; without
            them, a vector holds ones (a norm) and the other tensors, biases included, values
            around zero.
        */
        struct tensor
        {
            tensor(const std::string& n, const std::vector<uint64_t>& shape, float c = 0.0f, float s = -1.0f)
                : name(n), dims(shape), center(c), spread(s) {}
            std::string name;
            std::vector<uint64_t> dims;
            float center;
            float spread;
        };

        // The byte-level vocabulary with its control tokens, padded to a multiple of 32; returns its size.
        static long vocabulary(std::vector<std::string>& tokens, std::vector<int32_t>& types)
        {
            tokens = byte_tokens();
            types.assign(tokens.size(), 1);
            const char* control[] = { "<|endoftext|>", "<|im_start|>", "<|im_end|>" };
            const char* user_defined[] = { "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
                "<think>", "</think>" };
            for (const char* t : control) { tokens.push_back(t); types.push_back(3); }
            for (const char* t : user_defined) { tokens.push_back(t); types.push_back(4); }
            while (tokens.size() % 32 != 0)
            {
                tokens.push_back("[PAD" + std::to_string(tokens.size()) + "]");
                types.push_back(5);
            }
            return static_cast<long>(tokens.size());
        }

        // The descriptions of the tensors, which end the header, then their values.
        void write_tensors(const std::vector<tensor>& list)
        {
            uint64_t offset = 0;
            std::vector<uint64_t> offsets;
            for (const auto& t : list)
            {
                str(t.name, "name " + t.name);
                field("ndim " + t.name, 4);
                u32(static_cast<uint32_t>(t.dims.size()));
                for (size_t k = 0; k < t.dims.size(); ++k)
                {
                    field("dim" + std::to_string(k) + " " + t.name, 8);
                    u64(t.dims[k]);
                }
                field("type " + t.name, 4);
                u32(0);   // F32
                field("offset " + t.name, 8);
                u64(offset);
                offsets.push_back(offset);
                uint64_t n = 1;
                for (uint64_t x : t.dims) n *= x;
                offset += (n * 4 + 31) / 32 * 32;
            }
            while (out_.bytes.size() % 32 != 0) out_.bytes.push_back(0);
            out_.header_end = out_.bytes.size();
            uint32_t state = 12345;
            for (size_t i = 0; i < list.size(); ++i)
            {
                while (out_.bytes.size() < out_.header_end + offsets[i]) out_.bytes.push_back(0);
                uint64_t n = 1;
                for (uint64_t x : list[i].dims) n *= x;
                // The vectors are norms, of ones, except the biases.
                const bool norm = list[i].dims.size() == 1 && list[i].name.find(".bias") == std::string::npos;
                const bool given = list[i].spread >= 0.0f;
                const float center = given ? list[i].center : norm ? 1.0f : 0.0f;
                const float spread = given ? list[i].spread : norm ? 0.0f : 0.4f;
                for (uint64_t k = 0; k < n; ++k)
                {
                    state = state * 1664525u + 1013904223u;
                    const float v = center + (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * spread;
                    unsigned char b[4];
                    std::memcpy(b, &v, 4);
                    out_.bytes.insert(out_.bytes.end(), b, b + 4);
                }
            }
        }


        static std::vector<std::string> byte_tokens()
        {
            // The printable bytes stand for themselves; the others take the code points
            // from 256 on, as in the byte-level vocabularies of GPT-2.
            std::vector<std::string> t(256);
            int extra = 0;
            for (int b = 0; b < 256; ++b)
            {
                const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
                const int cp = printable ? b : 256 + extra++;
                std::string s;
                if (cp < 0x80) s += static_cast<char>(cp);
                else
                {
                    s += static_cast<char>(0xC0 | (cp >> 6));
                    s += static_cast<char>(0x80 | (cp & 0x3F));
                }
                t[static_cast<size_t>(b)] = s;
            }
            return t;
        }

        void field(const std::string& label, size_t size)
        {
            gguf_field f;
            f.label = label;
            f.offset = out_.bytes.size();
            f.size = size;
            out_.fields.push_back(f);
        }
        void u32(uint32_t v)
        {
            for (int i = 0; i < 4; ++i) out_.bytes.push_back(static_cast<unsigned char>(v >> (8 * i)));
        }
        void u64(uint64_t v)
        {
            for (int i = 0; i < 8; ++i) out_.bytes.push_back(static_cast<unsigned char>(v >> (8 * i)));
        }
        void str(const std::string& s, const std::string& label)
        {
            field("length " + label, 8);
            u64(s.size());
            out_.bytes.insert(out_.bytes.end(), s.begin(), s.end());
        }
        void key(const std::string& k, uint32_t type)
        {
            str(k, "key " + k);
            field("vtype " + k, 4);
            u32(type);
        }
        void kv_u64(const std::string& k, uint64_t v)
        {
            key(k, 10);
            field("kv " + k, 8);
            u64(v);
        }
        void kv_f32(const std::string& k, float v)
        {
            key(k, 6);
            field("kv " + k, 4);
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            u32(bits);
        }
        void kv_str(const std::string& k, const std::string& v)
        {
            key(k, 8);
            str(v, "value " + k);
        }
        void kv_strings(const std::string& k, const std::vector<std::string>& v)
        {
            key(k, 9);
            field("element type " + k, 4);
            u32(8);
            field("count " + k, 8);
            u64(v.size());
            for (size_t i = 0; i < v.size(); ++i) str(v[i], k + "[" + std::to_string(i) + "]");
        }
        void kv_ints(const std::string& k, const std::vector<int32_t>& v)
        {
            key(k, 9);
            field("element type " + k, 4);
            u32(5);
            field("count " + k, 8);
            u64(v.size());
            for (int32_t x : v) u32(static_cast<uint32_t>(x));
        }

        tiny_gguf out_;
    };

}

#endif // ALBIREO_TESTS_TINY_GGUF_H_
