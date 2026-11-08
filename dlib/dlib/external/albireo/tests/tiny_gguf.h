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
            std::vector<std::string> tokens = byte_tokens();
            std::vector<int32_t> types(tokens.size(), 1);
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
            const long vocab = static_cast<long>(tokens.size());

            struct tensor { std::string name; std::vector<uint64_t> dims; };
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
                const bool norm = list[i].dims.size() == 1;
                for (uint64_t k = 0; k < n; ++k)
                {
                    state = state * 1664525u + 1013904223u;
                    const float v = norm ? 1.0f : (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 0.4f;
                    unsigned char b[4];
                    std::memcpy(b, &v, 4);
                    out_.bytes.insert(out_.bytes.end(), b, b + 4);
                }
            }
            return out_;
        }

    private:
        long head_dim_ = 16;

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
