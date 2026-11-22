// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/data_io/gguf_dequantize.h>
#include <dlib/data_io/gguf_model_spec.h>
#include <dlib/data_io/gguf_reader.h>
#include <dlib/cuda/gguf_blocks.h>
#include <dlib/dnn.h>
#include <dlib/dnn/runtime_transformer.h>
#include <dlib/rand.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.gguf");

// ----------------------------------------------------------------------------------------

    /*
        A GGUF file written in memory: a few metadata entries of each kind, a tensor of
        floats, a tensor of half floats and a tensor of Q8_0 blocks, the data aligned on 32
        bytes.  The offset of each field the tests damage is recorded under a name.
    */
    struct gguf_image
    {
        std::vector<unsigned char> bytes;
        std::vector<pair<string, size_t>> fields;
        size_t data_start = 0;

        size_t at(const string& name) const
        {
            for (const auto& f : fields)
                if (f.first == name) return f.second;
            DLIB_TEST_MSG(false, "no field " << name);
            return 0;
        }

        std::vector<unsigned char> with_u64(const string& name, uint64_t v) const
        {
            std::vector<unsigned char> b = bytes;
            for (int i = 0; i < 8; ++i) b[at(name) + i] = static_cast<unsigned char>(v >> (8 * i));
            return b;
        }

        std::vector<unsigned char> with_u32(const string& name, uint32_t v) const
        {
            std::vector<unsigned char> b = bytes;
            for (int i = 0; i < 4; ++i) b[at(name) + i] = static_cast<unsigned char>(v >> (8 * i));
            return b;
        }

    };

    class gguf_builder
    {
    public:
        // The file, with the file type given in its header when it is not negative.
        gguf_image build(long file_type = -1)
        {
            out = gguf_image();
            u32(0x46554747u);
            mark("version");
            u32(3);
            mark("tensor count");
            u64(3);
            u64(file_type < 0 ? 5 : 6);     // metadata entries
            key("general.architecture", 8);
            str("llama");
            if (file_type >= 0)
            {
                key("general.file_type", 4);
                u32(static_cast<uint32_t>(file_type));
            }
            key("general.alignment", 4);
            u32(32);
            key("llama.block_count", 4);
            u32(2);
            key("llama.rope.freq_base", 6);
            f32(10000.0f);
            key("tokenizer.ggml.tokens", 9);
            u32(8);
            u64(3);
            str("<s>");
            str("</s>");
            str("hello");

            tensor("weights", { 4, 2 }, 0, 0);          // F32, 32 bytes
            tensor("halves", { 8 }, 1, 32);             // F16, 16 bytes
            tensor("blocks", { 32, 2 }, 8, 64);         // Q8_0, 2 blocks of 34 bytes
            while (out.bytes.size() % 32 != 0) out.bytes.push_back(0);
            out.data_start = out.bytes.size();

            for (int i = 0; i < 8; ++i) f32(0.5f * i - 1.0f);
            const uint16_t halves[8] = { 0x3C00, 0x4000, 0xBC00, 0x0000, 0x3800, 0x4200, 0x7BFF, 0x8000 };
            for (uint16_t h : halves) u16(h);
            pad_to(out.data_start + 64);
            for (int b = 0; b < 2; ++b)
            {
                u16(b == 0 ? 0x3C00 : 0x3800);         // scale 1, then 0.5
                for (int k = 0; k < 32; ++k)
                    out.bytes.push_back(static_cast<unsigned char>(static_cast<int8_t>(k - 16)));
            }
            return out;
        }

    private:
        gguf_image out;

        void mark(const string& name) { out.fields.push_back(make_pair(name, out.bytes.size())); }
        void put(uint64_t v, int n)
        {
            for (int i = 0; i < n; ++i) out.bytes.push_back(static_cast<unsigned char>(v >> (8 * i)));
        }
        void u16(uint16_t v) { put(v, 2); }
        void u32(uint32_t v) { put(v, 4); }
        void u64(uint64_t v) { put(v, 8); }
        void f32(float f) { uint32_t v; std::memcpy(&v, &f, 4); u32(v); }
        void str(const string& s) { u64(s.size()); out.bytes.insert(out.bytes.end(), s.begin(), s.end()); }
        void pad_to(size_t n) { while (out.bytes.size() < n) out.bytes.push_back(0); }

        void key(const string& k, uint32_t type)
        {
            mark("key " + k);
            str(k);
            u32(type);
        }

        void tensor(const string& name, const std::vector<uint64_t>& dims, uint32_t type, uint64_t offset)
        {
            mark("name " + name);
            str(name);
            u32(static_cast<uint32_t>(dims.size()));
            for (size_t d = 0; d < dims.size(); ++d)
            {
                mark("dim" + to_string(d) + " " + name);
                u64(dims[d]);
            }
            mark("type " + name);
            u32(type);
            mark("offset " + name);
            u64(offset);
        }
    };

    // The message of the exception the reader throws for the bytes, empty when it reads them.
    string refusal(const std::vector<unsigned char>& bytes)
    {
        try
        {
            gguf_reader r(bytes.data(), bytes.size());
            return string();
        }
        catch (const std::runtime_error& e)
        {
            return e.what();
        }
    }

// ----------------------------------------------------------------------------------------

    void test_intact()
    {
        const gguf_image image = gguf_builder().build();
        gguf_reader r(image.bytes.data(), image.bytes.size());
        DLIB_TEST(r.version() == 3);
        DLIB_TEST(r.size() == image.bytes.size());
        DLIB_TEST(r.alignment() == 32);
        DLIB_TEST(r.data_offset() == image.data_start);
        DLIB_TEST(r.get_str("general.architecture") == "llama");
        DLIB_TEST(r.get_int("llama.block_count") == 2);
        DLIB_TEST(std::abs(r.get_double("llama.rope.freq_base") - 10000.0) < 1e-6);
        DLIB_TEST(r.at("tokenizer.ggml.tokens").arr_str.size() == 3);
        DLIB_TEST(r.at("tokenizer.ggml.tokens").arr_str[2] == "hello");
        DLIB_TEST(r.get_int("no.such.key", 7) == 7);
        DLIB_TEST(r.tensors().size() == 3);

        const gguf_tensor_info* w = r.find_tensor("weights");
        const gguf_tensor_info* h = r.find_tensor("halves");
        const gguf_tensor_info* q = r.find_tensor("blocks");
        DLIB_TEST(w && h && q);
        DLIB_TEST(r.find_tensor("missing") == nullptr);
        DLIB_TEST(w->nbytes == 32 && h->nbytes == 16 && q->nbytes == 68);
        DLIB_TEST(q->is_quantized() && !w->is_quantized());

        std::vector<float> values;
        gguf_read_dequantized(r, *w, values);
        DLIB_TEST(values.size() == 8);
        for (int i = 0; i < 8; ++i) DLIB_TEST(values[i] == 0.5f * i - 1.0f);
        gguf_read_dequantized(r, *h, values);
        const float halves[8] = { 1.0f, 2.0f, -1.0f, 0.0f, 0.5f, 3.0f, 65504.0f, 0.0f };
        for (int i = 0; i < 8; ++i) DLIB_TEST(values[i] == halves[i]);
        gguf_read_dequantized(r, *q, values);
        DLIB_TEST(values.size() == 64);
        for (int k = 0; k < 32; ++k)
        {
            DLIB_TEST(values[k] == static_cast<float>(k - 16));
            DLIB_TEST(values[32 + k] == 0.5f * static_cast<float>(k - 16));
        }

        // The format the file is known by: the one of most of its bytes without a file type,
        // the file type declared when the file holds tensors of its base format.
        DLIB_TEST(string(ggml_type_name(ggml_type::Q5_K)) == "Q5_K");
        DLIB_TEST(string(ggml_type_name(static_cast<ggml_type>(999))) == "unknown");
        DLIB_TEST(gguf_quantization_name(r) == "Q8_0");
        const std::pair<long, const char*> named[] = { { 7, "Q8_0" }, { 1, "F16" }, { 0, "F32" }, { 17, "Q8_0" },
            { 15, "Q8_0" }, { 999, "Q8_0" } };
        for (const auto& n : named)
        {
            const gguf_image typed = gguf_builder().build(n.first);
            gguf_reader t(typed.bytes.data(), typed.bytes.size());
            DLIB_TEST_MSG(gguf_quantization_name(t) == n.second, "file type " << n.first << ": "
                << gguf_quantization_name(t));
            DLIB_TEST(t.tensors().size() == 3);
        }

        // The same container read from a file.
        const string path = "gguf_test_file.gguf";
        {
            ofstream f(path.c_str(), ios::binary);
            f.write(reinterpret_cast<const char*>(image.bytes.data()),
                static_cast<std::streamsize>(image.bytes.size()));
        }
        {
            gguf_reader file(path);
            DLIB_TEST(file.tensors().size() == 3);
            std::vector<uint8_t> raw;
            file.read_tensor_raw(*file.find_tensor("blocks"), raw);
            DLIB_TEST(raw.size() == 68);
            DLIB_TEST(std::equal(raw.begin(), raw.end(), image.bytes.begin() + image.data_start + 64));
        }
        std::remove(path.c_str());
    }

    void test_damaged()
    {
        const gguf_image image = gguf_builder().build();
        DLIB_TEST(refusal(image.bytes).empty());

        // Truncations, at every byte of the header and through the data.
        for (size_t cut = 0; cut < image.bytes.size(); cut += (cut < image.data_start ? 1 : 17))
        {
            const std::vector<unsigned char> b(image.bytes.begin(),
                image.bytes.begin() + static_cast<std::ptrdiff_t>(cut));
            DLIB_TEST_MSG(!refusal(b).empty(), "truncated at " << cut);
        }
        std::vector<unsigned char> magic = image.bytes;
        magic[0] = 'X';
        DLIB_TEST(refusal(magic).find("bad magic") != string::npos);
        DLIB_TEST(refusal(image.with_u32("version", 4)).find("version") != string::npos);
        DLIB_TEST(refusal(image.with_u64("tensor count", uint64_t(1) << 40)).find("corrupt header") != string::npos);

        // A key and a tensor name given twice: llama.rope.freq_base renamed general.architecture,
        // of the same length, and the tensor blocks renamed halves.
        {
            std::vector<unsigned char> b = image.bytes;
            const string name = "general.architecture";
            for (size_t i = 0; i < name.size(); ++i)
                b[image.at("key llama.rope.freq_base") + 8 + i] = static_cast<unsigned char>(name[i]);
            const string m = refusal(b);
            DLIB_TEST_MSG(m.find("duplicate metadata key 'general.architecture'") != string::npos, m);
        }
        {
            std::vector<unsigned char> b = image.bytes;
            const string name = "halves";
            for (size_t i = 0; i < name.size(); ++i)
                b[image.at("name blocks") + 8 + i] = static_cast<unsigned char>(name[i]);
            const string m = refusal(b);
            DLIB_TEST_MSG(m.find("duplicate tensor name 'halves'") != string::npos, m);
        }

        // Tensor data off the alignment, rows that split the blocks, data beyond the end.
        string m = refusal(image.with_u64("offset halves", 40));
        DLIB_TEST_MSG(m.find("alignment") != string::npos, m);
        {
            std::vector<unsigned char> b = image.with_u64("dim0 blocks", 16);
            for (int i = 0; i < 8; ++i)
                b[image.at("dim1 blocks") + i] = static_cast<unsigned char>(uint64_t(4) >> (8 * i));
            m = refusal(b);
            DLIB_TEST_MSG(m.find("block-aligned") != string::npos, m);
        }
        m = refusal(image.with_u64("offset blocks", 1024));
        DLIB_TEST_MSG(m.find("beyond the end") != string::npos, m);
        DLIB_TEST(!refusal(image.with_u64("dim0 weights", uint64_t(1) << 62)).empty());
        DLIB_TEST(!refusal(image.with_u32("type weights", 99)).empty());
    }

    void test_bounded()
    {
        const gguf_image image = gguf_builder().build();
        gguf_reader r(image.bytes.data(), image.bytes.size());
        DLIB_TEST(gguf_bounded_int(r, "llama.block_count", 0, 0, 1024) == 2);
        DLIB_TEST(gguf_bounded_int(r, "llama.absent", 5, 0, 1024) == 5);
        DLIB_TEST(gguf_bounded_int(r, "tokenizer.ggml.tokens", 9, 0, 1024) == 9);
        DLIB_TEST(gguf_bounded_int(r, "llama.rope.freq_base", 0, 0, 100000) == 10000);
        DLIB_TEST(gguf_bounded_double(r, "llama.rope.freq_base", 0.0, 1.0, 1e12) == 10000.0);
        bool thrown = false;
        try { gguf_bounded_int(r, "llama.block_count", 0, 3, 1024); }
        catch (const std::runtime_error&) { thrown = true; }
        DLIB_TEST(thrown);
        thrown = false;
        try { gguf_bounded_int(r, "general.architecture", 0, 0, 10); }
        catch (const std::runtime_error&) { thrown = true; }
        DLIB_TEST(thrown);
        thrown = false;
        try { gguf_bounded_double(r, "llama.rope.freq_base", 0.0, 0.0, 100.0); }
        catch (const std::runtime_error&) { thrown = true; }
        DLIB_TEST(thrown);
    }

// ----------------------------------------------------------------------------------------

    /*
        A tiny hybrid decoder of the olala family written in memory, in single precision: four
        layers of width 64, two Mamba-3 layers with several inputs and outputs and two
        differential attentions, each followed by a mixture of experts in a latent space
        beside a shared expert.  Its values come from a fixed sequence, around those of a
        trained model: scales and factors around one.
    */
    class hybrid_builder
    {
    public:
        std::vector<unsigned char> build()
        {
            const long d = 64, layers = 4, vocab = 288;
            const long ssm_heads = 8, ssm_hd = 16, state = 16, rank = 2, angles = 4;
            const long heads = 8, kv_heads = 2, hd = 32, tpa = 2, signal = heads - kv_heads;
            const long experts = 8, used = 2, hidden = 32, latent = 16, shared = 48;
            const string types = "MVMV";
            list.clear();
            out.clear();
            add("output.weight", { d, vocab }, 0.0f, 8.0f);
            add("token_embd.weight", { d, vocab }, 0.0f, 0.4f);
            for (long l = 0; l < layers; ++l)
            {
                const string p = "blk." + to_string(l) + ".";
                add(p + "geo_mixer_scale", { 1 }, 1.0f, 0.2f);
                add(p + "geo_mixer_bias", { 1 }, 0.02f, 0.02f);
                add(p + "geo_mlp_scale", { 1 }, 1.0f, 0.2f);
                add(p + "geo_mlp_bias", { 1 }, 0.02f, 0.02f);
                if (types[static_cast<size_t>(l)] == 'M')
                {
                    add(p + "ssm_in.weight", { d, ssm_heads * (2 * ssm_hd + 3) }, 0.0f, 0.8f);
                    add(p + "ssm_in_dyn.weight", { d, 2 * rank * state + angles }, 0.0f, 0.8f);
                    add(p + "attn_output.weight", { ssm_heads * ssm_hd, d }, 0.0f, 0.4f);
                    add(p + "ssm_b_norm.weight", { state }, 1.0f, 0.2f);
                    add(p + "ssm_c_norm.weight", { state }, 1.0f, 0.2f);
                    add(p + "ssm_b_bias", { state, rank, ssm_heads }, 1.0f, 0.6f);
                    add(p + "ssm_c_bias", { state, rank, ssm_heads }, 1.0f, 0.6f);
                    add(p + "ssm_dt_bias", { ssm_heads }, -1.0f, 1.0f);
                    add(p + "ssm_d", { ssm_heads }, 1.0f, 0.6f);
                    add(p + "ssm_mimo_x", { ssm_hd, rank, ssm_heads }, 0.5f, 0.4f);
                    add(p + "ssm_mimo_z", { ssm_hd, rank, ssm_heads }, 1.0f, 0.6f);
                    add(p + "ssm_mimo_o", { ssm_hd, rank, ssm_heads }, 0.5f, 0.4f);
                }
                else
                {
                    add(p + "attn_q.weight", { d, heads * hd }, 0.0f, 0.8f);
                    add(p + "attn_wa_k.weight", { d, kv_heads * tpa }, 0.0f, 1.6f);
                    add(p + "attn_wb_k.weight", { d, tpa * hd }, 0.0f, 1.6f);
                    add(p + "attn_wa_v.weight", { d, kv_heads * tpa }, 0.0f, 1.6f);
                    add(p + "attn_wb_v.weight", { d, tpa * hd }, 0.0f, 1.6f);
                    add(p + "attn_shift_k.weight", { d, kv_heads }, 0.0f, 1.6f);
                    add(p + "attn_shift_v.weight", { d, kv_heads }, 0.0f, 1.6f);
                    add(p + "attn_lambda.weight", { d, kv_heads }, 0.0f, 1.6f);
                    add(p + "attn_gate.weight", { d, signal * hd }, 0.0f, 0.8f);
                    add(p + "attn_output.weight", { signal * hd, d }, 0.0f, 0.4f);
                    add(p + "attn_q_norm.weight", { hd }, 1.0f, 0.2f);
                    add(p + "attn_k_norm.weight", { hd }, 1.0f, 0.2f);
                    add(p + "attn_softmax_scaler", { heads }, 2.0f, 2.0f);
                }
                add(p + "ffn_gate_inp.weight", { d, experts }, 0.0f, 4.0f);
                add(p + "exp_probs_b", { experts }, 0.0f, 0.1f);
                add(p + "ffn_latent_down.weight", { d, latent }, 0.0f, 0.8f);
                add(p + "ffn_latent_up.weight", { latent, d }, 0.0f, 0.4f);
                add(p + "ffn_up_exps.weight", { latent, hidden, experts }, 0.0f, 0.8f);
                add(p + "ffn_down_exps.weight", { hidden, latent, experts }, 0.0f, 0.4f);
                add(p + "ffn_up_shexp.weight", { d, shared }, 0.0f, 0.8f);
                add(p + "ffn_down_shexp.weight", { shared, d }, 0.0f, 0.4f);
            }

            u32(0x46554747u);
            u32(3);
            u64(list.size());
            const size_t count_at = out.size();
            u64(0);
            n_keys = 0;
            text("general.architecture", "olala");
            text("general.name", "tiny hybrid");
            number("olala.block_count", layers);
            number("olala.context_length", 4096);
            number("olala.embedding_length", d);
            number("olala.vocab_size", vocab);
            number("olala.attention.head_count", heads);
            number("olala.attention.head_count_kv", kv_heads);
            number("olala.attention.key_length", hd);
            real("olala.attention.layer_norm_rms_epsilon", 1e-6f);
            number("olala.attention.sliding_window", 2048);
            real("olala.attn_logit_softcapping", 2.0f);
            number("olala.expert_count", experts);
            number("olala.expert_used_count", used);
            number("olala.expert_feed_forward_length", hidden);
            number("olala.expert_shared_feed_forward_length", shared);
            number("olala.expert_shared_count", 1);
            real("olala.expert_weights_scale", 2.45f);
            number("olala.expert_gating_func", 2);
            number("olala.moe_latent_size", latent);
            number("olala.ssm.inner_size", ssm_heads * ssm_hd);
            number("olala.ssm.state_size", state);
            number("olala.ssm.group_count", 1);
            number("olala.mamba_mimo_dim", rank);
            number("olala.mamba_headdim", ssm_hd);
            number("olala.num_signal_heads", signal);
            number("olala.num_noise_heads", kv_heads);
            number("olala.tpa_rank", tpa);
            number("olala.num_rope_angles", angles);
            text("olala.layer_types", types);
            real("olala.gate_bias", 1.15f);
            number("olala.gate_attn", 1);
            number("olala.geodesic_update", 1);
            number("olala.final_norm", 0);
            for (int i = 0; i < 8; ++i) out[count_at + i] = static_cast<unsigned char>(n_keys >> (8 * i));

            uint64_t offset = 0;
            std::vector<uint64_t> offsets;
            for (const auto& t : list)
            {
                str(t.name);
                u32(static_cast<uint32_t>(t.dims.size()));
                uint64_t n = 1;
                for (long x : t.dims)
                {
                    u64(static_cast<uint64_t>(x));
                    n *= static_cast<uint64_t>(x);
                }
                u32(0);     // F32
                u64(offset);
                offsets.push_back(offset);
                offset += (n * 4 + 31) / 32 * 32;
            }
            while (out.size() % 32 != 0) out.push_back(0);
            const size_t data = out.size();
            uint32_t state_of_sequence = 12345;
            for (size_t i = 0; i < list.size(); ++i)
            {
                while (out.size() < data + offsets[i]) out.push_back(0);
                uint64_t n = 1;
                for (long x : list[i].dims) n *= static_cast<uint64_t>(x);
                for (uint64_t k = 0; k < n; ++k)
                {
                    state_of_sequence = state_of_sequence * 1664525u + 1013904223u;
                    const float v = list[i].center
                        + (static_cast<float>(state_of_sequence >> 8) / 16777216.0f - 0.5f) * list[i].spread;
                    uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    u32(bits);
                }
            }
            return out;
        }

    private:
        struct entry { string name; std::vector<long> dims; float center; float spread; };
        std::vector<entry> list;
        std::vector<unsigned char> out;
        uint64_t n_keys = 0;

        void add(const string& name, const std::vector<long>& dims, float center, float spread)
        {
            list.push_back(entry{ name, dims, center, spread });
        }
        void put(uint64_t v, int n)
        {
            for (int i = 0; i < n; ++i) out.push_back(static_cast<unsigned char>(v >> (8 * i)));
        }
        void u32(uint32_t v) { put(v, 4); }
        void u64(uint64_t v) { put(v, 8); }
        void str(const string& s) { u64(s.size()); out.insert(out.end(), s.begin(), s.end()); }
        void text(const string& k, const string& v) { str(k); u32(8); str(v); ++n_keys; }
        void number(const string& k, uint64_t v) { str(k); u32(10); u64(v); ++n_keys; }
        void real(const string& k, float f)
        {
            str(k);
            u32(6);
            uint32_t v;
            std::memcpy(&v, &f, 4);
            u32(v);
            ++n_keys;
        }
    };

    // tt::geodesic_update() against its formula in double precision.
    void test_geodesic_update()
    {
        dlib::rand rnd;
        for (int trial = 0; trial < 3; ++trial)
        {
            const long rows = 5, nc = 48;
            resizable_tensor x(1, 1, rows, nc), g(1, 1, rows, nc);
            for (float& v : x) v = rnd.get_random_gaussian();
            // Outputs as large as those of the mixers of the hybrid decoders, then a null row.
            for (float& v : g) v = static_cast<float>(rnd.get_random_gaussian() * (trial == 2 ? 1e12 : 1.0));
            if (trial == 1) std::fill(g.host(), g.host() + nc, 0.0f);
            const float scale = 1.3f, bias = 0.02f;
            const long depth = 3;
            resizable_tensor want(x), start(x);
            const double quarter = 0.78539816339744830962;
            for (long r = 0; r < rows; ++r)
            {
                float* xr = want.host() + r * nc;
                const float* gr = g.host() + r * nc;
                double xx = 0, xg = 0, tt2 = 0;
                for (long i = 0; i < nc; ++i)
                {
                    xx += static_cast<double>(xr[i]) * xr[i];
                    xg += static_cast<double>(xr[i]) * gr[i];
                }
                std::vector<double> t(nc);
                for (long i = 0; i < nc; ++i)
                {
                    t[i] = gr[i] - xg / std::max(xx, 1e-12) * xr[i];
                    tt2 += t[i] * t[i];
                }
                const double norm = std::max(std::sqrt(tt2), 1e-8), radius = std::max(std::sqrt(xx), 1e-6);
                double theta = std::min(norm / radius, quarter);
                theta = std::min((theta * scale + bias) / depth, quarter);
                for (long i = 0; i < nc; ++i)
                    xr[i] = static_cast<float>(xr[i] * std::cos(theta) + t[i] / norm * radius * std::sin(theta));
            }
            tt::geodesic_update(x, g, scale, bias, depth);
            DLIB_TEST_MSG(max(abs(mat(x) - mat(want))) < 1e-5, max(abs(mat(x) - mat(want))));
            // A row that turns keeps its length.
            double before = 0, after = 0;
            for (long i = 0; i < nc; ++i)
            {
                before += static_cast<double>(start.host()[2 * nc + i]) * start.host()[2 * nc + i];
                after += static_cast<double>(x.host()[2 * nc + i]) * x.host()[2 * nc + i];
            }
            DLIB_TEST(std::abs(std::sqrt(before) - std::sqrt(after)) < 1e-4);
        }
    }

    /*
        The hybrid decoders in the runtime engine: the detection of the family, the logits of
        a token sequence against those tests/reference_forward.py of Albireo computes for
        this model with numpy, and the steps of one token against the pass of the whole
        sequence.  With CUDA, the pass runs the device kernels of tt::mamba3_step() and of
        tt::geodesic_update().
    */
    void test_hybrid_decoder()
    {
        const std::vector<unsigned char> bytes = hybrid_builder().build();
        gguf_reader g(bytes.data(), bytes.size());
        const model_spec spec = detect_model(g);
        DLIB_TEST(spec.family == arch_family::olala && spec.hybrid() && spec.hybrid_supported);
        DLIB_TEST(spec.layer_types == "MVMV" && spec.ssm_heads == 8 && spec.ssm_head_dim == 16);
        DLIB_TEST(spec.ssm_state == 16 && spec.ssm_rank == 2 && spec.ssm_angles == 4 && spec.tpa_rank == 2);
        DLIB_TEST(spec.d_latent == 16 && spec.d_ffn_expert == 32 && spec.d_ffn_shared == 48);
        DLIB_TEST(std::abs(spec.attention_scale - 1.0 / 32) < 1e-12 && spec.vocab_size == 288);
        // The runtime engine takes the model; the template network has no such layers.
        DLIB_TEST(check_compatibility(spec, false).ok);
        DLIB_TEST(!check_compatibility(spec, true).ok);

        runtime_transformer rt;
        for (int at_rest = 0; at_rest < 2; ++at_rest)
        {
            rt.set_quantized_at_rest(at_rest != 0);
            rt.load(g, spec, gguf_load_options());
            std::vector<int> tokens(24);
            for (int i = 0; i < 24; ++i) tokens[static_cast<size_t>(i)] = i + 1;
            const tensor& all = rt.forward_prefill(tokens);
            DLIB_TEST(all.nr() == 24 && all.nc() == 288);
            std::vector<float> whole(all.host(), all.host() + all.size());
            // Logits of the reference, at the first, at a middle and at the last position.
            const long ids[5] = { 0, 1, 97, 200, 287 };
            const long rows[3] = { 0, 11, 23 };
            const long best[3] = { 101, 17, 4 };
            const double want[3][5] = {
                { 2.064247, -3.492518, -0.050489, 1.429517, -0.063549 },
                { 0.097271, -3.198817, 0.103718, -2.553112, 2.278232 },
                { -1.557517, 0.276659, -0.921414, 0.911741, 1.237635 } };
            for (int r = 0; r < 3; ++r)
            {
                const float* row = whole.data() + rows[r] * 288;
                for (int i = 0; i < 5; ++i)
                    DLIB_TEST_MSG(std::abs(row[ids[i]] - want[r][i]) < 2e-3, "row " << rows[r] << " id " << ids[i]
                        << ": " << row[ids[i]] << " for " << want[r][i]);
                DLIB_TEST(std::max_element(row, row + 288) - row == best[r]);
            }

            // A context, the first half read in one pass, then one token at a time.
            rt.set_context(26);
            const std::vector<int> head(tokens.begin(), tokens.begin() + 12);
            rt.forward_prefill(head);
            for (int t = 12; t < 24; ++t)
            {
                const tensor& one = rt.step(tokens[static_cast<size_t>(t)]);
                const float* o = one.host();
                double worst = 0;
                for (long v = 0; v < 288; ++v)
                {
                    const double want_v = whole[static_cast<size_t>(t * 288 + v)];
                    worst = std::max(worst, std::abs(o[v] - want_v));
                }
                DLIB_TEST_MSG(worst < 1e-4, "step " << t << ": " << worst);
            }
            DLIB_TEST(rt.cache_length() == 24);
            // The layers cannot give up the tokens they read: a full context refuses the next one.
            rt.step(1);
            rt.step(2);
            bool thrown = false;
            try { rt.step(3); }
            catch (const std::runtime_error&) { thrown = true; }
            DLIB_TEST(thrown);
            // Nor does a runtime archive hold such a decoder.
            thrown = false;
            std::ostringstream sout;
            try { rt.save(sout); }
            catch (const std::runtime_error&) { thrown = true; }
            DLIB_TEST(thrown);
        }
    }

    // The bytes of rows rows of cols random weights in a GGUF format, with scales of a finite size.
    std::vector<uint8_t> random_weight_rows(int type, long rows, long cols, dlib::rand& rnd)
    {
        const size_t row_bytes = gguf_blocks::row_bytes(type, cols);
        std::vector<uint8_t> bytes(static_cast<size_t>(rows) * row_bytes);
        for (auto& b : bytes) b = static_cast<uint8_t>(rnd.get_random_32bit_number() & 0xFF);
        auto half = [&](uint8_t* p) {
            // A half between 2^-7 and 2^-4.
            const uint32_t h = ((8 + rnd.get_random_32bit_number() % 3) << 10) | (rnd.get_random_32bit_number() % 1024);
            p[0] = static_cast<uint8_t>(h & 0xFF);
            p[1] = static_cast<uint8_t>(h >> 8);
        };
        for (long r = 0; r < rows; ++r)
        {
            uint8_t* row = bytes.data() + static_cast<size_t>(r) * row_bytes;
            switch (type)
            {
            case gguf_blocks::f32:
                for (long i = 0; i < cols; ++i)
                {
                    const float v = rnd.get_random_gaussian();
                    std::memcpy(row + 4 * i, &v, 4);
                }
                break;
            case gguf_blocks::bf16:
                for (long i = 0; i < cols; ++i)
                {
                    const float v = rnd.get_random_gaussian();
                    uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    row[2 * i] = static_cast<uint8_t>((bits >> 16) & 0xFF);
                    row[2 * i + 1] = static_cast<uint8_t>(bits >> 24);
                }
                break;
            case gguf_blocks::q8_0:
                for (long g = 0; g < cols / 32; ++g) half(row + g * 34);
                break;
            case gguf_blocks::q4_K:
                for (long b = 0; b < cols / 256; ++b)
                {
                    half(row + b * 144);
                    half(row + b * 144 + 2);
                }
                break;
            case gguf_blocks::q5_K:
                for (long b = 0; b < cols / 256; ++b)
                {
                    half(row + b * 176);
                    half(row + b * 176 + 2);
                }
                break;
            default:
                for (long b = 0; b < cols / 256; ++b) half(row + b * 210 + 208);
                break;
            }
        }
        return bytes;
    }

    // A tensor whose floats store bytes, as tt::gguf_matvec() reads the weights of a matrix.
    void store_bytes(resizable_tensor& t, const std::vector<uint8_t>& bytes)
    {
        t.set_size(1, 1, 1, static_cast<long long>((bytes.size() + 3) / 4));
        t = 0;
        std::memcpy(t.host(), bytes.data(), bytes.size());
    }

    void store_integers(resizable_tensor& t, const std::vector<int32_t>& values)
    {
        t.set_size(1, 1, 1, static_cast<long long>(values.size()));
        std::memcpy(t.host(), values.data(), values.size() * sizeof(int32_t));
    }

    /*
        The operations of the mixtures of experts of the hybrid decoders against their
        definitions: tt::expert_routes() against a sort of the ranked scores, with experts of
        equal rank; tt::gguf_matvec_picked() against the products of the rows its picks name,
        in double precision, with one row of activations for several matrices and with one
        row each; tt::squared_relu() and tt::mix_rows().
    */
    void test_expert_operations()
    {
        dlib::rand rnd;
        {
            const long n = 5, experts = 37, top_k = 6;
            resizable_tensor logits(n, experts), bias(1, experts), picks(n, top_k), weights(n, top_k);
            for (float& v : logits) v = 2 * rnd.get_random_gaussian();
            for (float& v : bias) v = 0.1f * rnd.get_random_gaussian();
            // Eight experts of equal rank lead the second row: the six of lower index are taken.
            for (long e = 0; e < experts; ++e)
            {
                logits.host()[experts + e] = e >= 20 && e % 2 == 0 ? 1.5f : -1.0f;
                if (e >= 20) bias.host()[e] = 0;
            }
            for (int normalize = 0; normalize < 2; ++normalize)
            {
                const float scale = 2.45f;
                tt::expert_routes(picks, weights, logits, bias, top_k, normalize != 0, scale);
                for (long t = 0; t < n; ++t)
                {
                    std::vector<float> score(experts), ranked(experts);
                    std::vector<long> order(experts);
                    for (long e = 0; e < experts; ++e)
                    {
                        score[e] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(
                            logits.host()[t * experts + e]))));
                        ranked[e] = score[e] + bias.host()[e];
                        order[e] = e;
                    }
                    std::stable_sort(order.begin(), order.end(), [&](long a, long b) { return ranked[a] > ranked[b]; });
                    float selected = 0;
                    for (long k = 0; k < top_k; ++k) selected += score[order[k]];
                    for (long k = 0; k < top_k; ++k)
                    {
                        int32_t pick;
                        std::memcpy(&pick, picks.host() + t * top_k + k, sizeof(pick));
                        DLIB_TEST_MSG(pick == order[k], "row " << t << " rank " << k << ": " << pick << " for "
                            << order[k]);
                        float want = score[order[k]];
                        if (normalize) want /= selected + 1e-20f;
                        DLIB_TEST(std::abs(weights.host()[t * top_k + k] - want * scale) < 1e-6);
                    }
                }
            }
        }
        const int types[6] = { gguf_blocks::f32, gguf_blocks::bf16, gguf_blocks::q8_0, gguf_blocks::q4_K,
            gguf_blocks::q5_K, gguf_blocks::q6_K };
        for (int type : types)
        {
            const bool k_quant = type == gguf_blocks::q4_K || type == gguf_blocks::q5_K || type == gguf_blocks::q6_K;
            const long cols = k_quant ? 512 : 96, rows = 11, matrices = 7;
            const size_t row_bytes = gguf_blocks::row_bytes(type, cols);
            const std::vector<uint8_t> bytes = random_weight_rows(type, rows * matrices, cols, rnd);
            resizable_tensor w;
            store_bytes(w, bytes);
            for (long per_row : { 1L, 3L })
            {
                const std::vector<int32_t> chosen = { 4, 0, 6, 6, 2, 5 };
                const long p = static_cast<long>(chosen.size());
                resizable_tensor picks, x(p / per_row, cols), y(p, rows);
                store_integers(picks, chosen);
                for (float& v : x) v = rnd.get_random_gaussian();
                tt::gguf_matvec_picked(y, x, w, type, rows, cols, row_bytes, picks, per_row);
                double worst = 0;
                float v[32];
                for (long j = 0; j < p; ++j)
                    for (long r = 0; r < rows; ++r)
                    {
                        const uint8_t* row = bytes.data() + static_cast<size_t>(chosen[j] * rows + r) * row_bytes;
                        double sum = 0, size = 0;
                        for (long g = 0; g < cols / 32; ++g)
                        {
                            gguf_blocks::decode_group(type, row, g, v);
                            for (int i = 0; i < 32; ++i)
                            {
                                const double term = static_cast<double>(v[i])
                                    * x.host()[(j / per_row) * cols + g * 32 + i];
                                sum += term;
                                size += std::abs(term);
                            }
                        }
                        worst = std::max(worst, std::abs(y.host()[j * rows + r] - sum) / (size + 1e-9));
                    }
                DLIB_TEST_MSG(worst < 1e-5, "type " << type << ", " << per_row << " per row: " << worst);
            }
        }
        {
            resizable_tensor x(3, 50), start;
            for (float& v : x) v = rnd.get_random_gaussian();
            start = x;
            tt::squared_relu(x);
            for (size_t i = 0; i < x.size(); ++i)
            {
                const float r = std::max(start.host()[i], 0.0f);
                DLIB_TEST(x.host()[i] == r * r);
            }
            const long n = 4, per_row = 3, width = 21;
            resizable_tensor src(n * per_row, width), weights(n, per_row), mixed(n, width);
            for (float& v : src) v = rnd.get_random_gaussian();
            for (float& v : weights) v = rnd.get_random_gaussian();
            tt::mix_rows(mixed, src, weights, per_row);
            for (long t = 0; t < n; ++t)
                for (long i = 0; i < width; ++i)
                {
                    float sum = 0;
                    for (long j = 0; j < per_row; ++j)
                        sum += weights.host()[t * per_row + j] * src.host()[(t * per_row + j) * width + i];
                    DLIB_TEST(std::abs(mixed.host()[t * width + i] - sum) < 1e-6);
                }
        }
    }

#ifdef DLIB_USE_CUDA
    /*
        The device kernels of the operations of the mixtures of experts against the processor:
        the routes of tokens of 8 to 1,000 experts, the picked products for every kind of
        kernel, matrices whose rows do not fill the last block of their warps and activations
        that start on 16 bytes or not, then the squares and the weighted sums.
    */
    void test_expert_operations_devices()
    {
        dlib::rand rnd;
        const long shapes[4][3] = { { 1, 8, 2 }, { 3, 256, 6 }, { 9, 1000, 64 }, { 2, 300, 1 } };
        for (const auto& c : shapes)
        {
            const long n = c[0], experts = c[1], top_k = c[2];
            resizable_tensor logits(n, experts), bias(1, experts);
            for (float& v : logits) v = 2 * rnd.get_random_gaussian();
            for (float& v : bias) v = 0.1f * rnd.get_random_gaussian();
            resizable_tensor p1(n, top_k), w1(n, top_k), p2(n, top_k), w2(n, top_k);
            cpu::expert_routes(p1, w1, logits, bias, top_k, true, 2.45f);
            cuda::expert_routes(p2, w2, logits, bias, top_k, true, 2.45f);
            DLIB_TEST_MSG(std::memcmp(p1.host(), p2.host(), p1.size() * sizeof(float)) == 0, "experts of " << experts);
            DLIB_TEST_MSG(max(abs(mat(w1) - mat(w2))) < 1e-6, max(abs(mat(w1) - mat(w2))));
        }
        const int types[6] = { gguf_blocks::f32, gguf_blocks::bf16, gguf_blocks::q8_0, gguf_blocks::q4_K,
            gguf_blocks::q5_K, gguf_blocks::q6_K };
        for (int type : types)
        {
            const bool k_quant = type == gguf_blocks::q4_K || type == gguf_blocks::q5_K || type == gguf_blocks::q6_K;
            for (int shape = 0; shape < 2; ++shape)
            {
                // Rows that fill their blocks of 8 warps or not; columns beyond one chunk of shared memory.
                const long cols = k_quant ? (shape ? 1536 : 256) : (shape ? 1120 : 32);
                const long rows = shape ? 21 : 16, matrices = 9;
                const size_t row_bytes = gguf_blocks::row_bytes(type, cols);
                resizable_tensor w;
                store_bytes(w, random_weight_rows(type, rows * matrices, cols, rnd));
                for (long per_row : { 1L, 4L })
                {
                    std::vector<int32_t> chosen(12);
                    for (auto& e : chosen) e = static_cast<int32_t>(rnd.get_random_32bit_number() % matrices);
                    const long p = static_cast<long>(chosen.size());
                    resizable_tensor picks, y1(p, rows), y2(p, rows);
                    store_integers(picks, chosen);
                    // One float before the rows moves them off 16 bytes.
                    for (long shift = 0; shift < 2; ++shift)
                    {
                        resizable_tensor room(1, 1, 1, (p / per_row) * cols + 4);
                        for (float& v : room) v = rnd.get_random_gaussian();
                        alias_tensor rows_of(p / per_row, cols);
                        auto x = rows_of(room, static_cast<size_t>(shift));
                        cpu::gguf_matvec_picked(y1, x, w, type, rows, cols, row_bytes, picks, per_row);
                        cuda::gguf_matvec_picked(y2, x, w, type, rows, cols, row_bytes, picks, per_row);
                        const float apart = max(abs(mat(y1) - mat(y2)));
                        DLIB_TEST_MSG(apart < 2e-4 * (1 + max(abs(mat(y1)))), "type " << type << ", " << rows << "x"
                            << cols << ", " << per_row << " per row, shift " << shift << ": " << apart);
                    }
                }
            }
        }
        resizable_tensor x(7, 333), y;
        for (float& v : x) v = rnd.get_random_gaussian();
        y = x;
        cpu::squared_relu(x);
        cuda::squared_relu(y);
        DLIB_TEST(max(abs(mat(x) - mat(y))) < 1e-6);
        const long n = 5, per_row = 6, width = 384;
        resizable_tensor src(n * per_row, width), weights(n, per_row), m1(n, width), m2(n, width);
        for (float& v : src) v = rnd.get_random_gaussian();
        for (float& v : weights) v = rnd.get_random_gaussian();
        cpu::mix_rows(m1, src, weights, per_row);
        cuda::mix_rows(m2, src, weights, per_row);
        DLIB_TEST_MSG(max(abs(mat(m1) - mat(m2))) < 1e-5, max(abs(mat(m1) - mat(m2))));
    }

    // The device kernels of tt::mamba3_step() against the processor, over several tokens of several geometries.
    void test_mamba3_step_devices()
    {
        dlib::rand rnd;
        const long geometries[4][5] = { { 3, 5, 8, 2, 2 }, { 4, 16, 16, 1, 4 }, { 2, 7, 12, 4, 0 },
            { 6, 8, 32, 3, 16 } };
        for (const auto& c : geometries)
        {
            const long H = c[0], P = c[1], N = c[2], R = c[3], S = c[4];
            resizable_tensor norms(2, N), bias(H, 2 * R, N), misc(2, H), mxz(H, 2 * R, P), mo(H, R, P);
            for (float& v : norms) v = 1 + 0.2f * rnd.get_random_gaussian();
            for (float& v : bias) v = 1 + 0.3f * rnd.get_random_gaussian();
            for (float& v : misc) v = 0.5f * rnd.get_random_gaussian();
            for (float& v : mxz) v = 0.7f + 0.3f * rnd.get_random_gaussian();
            for (float& v : mo) v = 0.5f + 0.2f * rnd.get_random_gaussian();
            resizable_tensor s1(1, 1, 1, H * S + H * P * N + H * R * N + H * P), s2;
            s1 = 0;
            s2 = s1;
            resizable_tensor o1(1, 1, 1, H * P), o2(1, 1, 1, H * P), in(1, 1, 1, H * (2 * P + 3));
            resizable_tensor dyn(1, 1, 1, 2 * R * N + S);
            for (int t = 0; t < 8; ++t)
            {
                for (float& v : in) v = rnd.get_random_gaussian();
                for (float& v : dyn) v = 1.5f * rnd.get_random_gaussian();
                cpu::mamba3_step(o1, s1, in, dyn, norms, bias, misc, mxz, mo, 1e-6f);
                cuda::mamba3_step(o2, s2, in, dyn, norms, bias, misc, mxz, mo, 1e-6f);
                const float outputs = max(abs(mat(o1) - mat(o2))), states = max(abs(mat(s1) - mat(s2)));
                DLIB_TEST_MSG(outputs < 1e-3 * (1 + max(abs(mat(o1)))), outputs);
                DLIB_TEST_MSG(states < 1e-3 * (1 + max(abs(mat(s1)))), states);
            }
            // Several tokens in one launch, after the tokens above.
            const long T = 5;
            resizable_tensor many_in(T, 1, 1, H * (2 * P + 3)), many_dyn(T, 1, 1, 2 * R * N + S);
            resizable_tensor m1(T, 1, 1, H * P), m2(T, 1, 1, H * P);
            for (float& v : many_in) v = rnd.get_random_gaussian();
            for (float& v : many_dyn) v = 1.5f * rnd.get_random_gaussian();
            cpu::mamba3_step(m1, s1, many_in, many_dyn, norms, bias, misc, mxz, mo, 1e-6f);
            cuda::mamba3_step(m2, s2, many_in, many_dyn, norms, bias, misc, mxz, mo, 1e-6f);
            const float outputs = max(abs(mat(m1) - mat(m2))), states = max(abs(mat(s1) - mat(s2)));
            DLIB_TEST_MSG(outputs < 1e-3 * (1 + max(abs(mat(m1)))), outputs);
            DLIB_TEST_MSG(states < 1e-3 * (1 + max(abs(mat(s1)))), states);
        }
        // Rows narrower and wider than a block of the device, the wider ones with outputs far beyond their rows.
        const long widths[2] = { 40, 1000 };
        for (const long width : widths)
        {
            resizable_tensor x(2, 1, 3, width), g(2, 1, 3, width);
            for (float& v : x) v = rnd.get_random_gaussian();
            for (float& v : g) v = (width > 256 ? 3e6f : 3.0f) * rnd.get_random_gaussian();
            resizable_tensor y(x);
            cpu::geodesic_update(x, g, 1.2f, 0.01f, 2);
            cuda::geodesic_update(y, g, 1.2f, 0.01f, 2);
            DLIB_TEST_MSG(max(abs(mat(x) - mat(y))) < 1e-5, max(abs(mat(x) - mat(y))));
        }
    }
#endif

    /*
        The operations of the longer passes of a decoder, on the device of the build, against
        the processor code and against the operations they stand for: tt::gguf_gemm() in every
        arithmetic the device computes with, through tiles of one row to the whole matrix,
        against cpu::gguf_matvec(); tt::add_rms_norm_rows() against the sum followed by
        cpu::rms_norm_rows(); tt::rope_rows() of two sets of rows against a call each;
        tt::paged_attention_rows() against cpu::paged_attention(), in half precision and in 8
        bits, from the start of a sequence and from further on, in one run of the tokens and in
        runs of five.
    */
    void test_long_pass_operations()
    {
        dlib::rand rnd;
        const int types[5] = { gguf_blocks::f32, gguf_blocks::bf16, gguf_blocks::q8_0, gguf_blocks::q4_K,
            gguf_blocks::q6_K };
        const gguf_precision precisions[3] = { gguf_precision::single, gguf_precision::tf32, gguf_precision::half };
        for (int type : types)
        {
            const bool k_quant = type == gguf_blocks::q4_K || type == gguf_blocks::q6_K;
            const long rows = 21, cols = k_quant ? 512 : 160;
            const size_t row_bytes = gguf_blocks::row_bytes(type, cols);
            resizable_tensor w;
            store_bytes(w, random_weight_rows(type, rows, cols, rnd));
            for (long n : { 3L, 70L })
            {
                resizable_tensor x(n, cols), expected(n, rows), y(n, rows), saturated(1);
                for (float& v : x) v = rnd.get_random_gaussian();
                cpu::gguf_matvec(expected, x, w, type, rows, cols, row_bytes);
                const float size = 1 + max(abs(mat(expected)));
                for (gguf_precision p : precisions)
                {
                    if (!tt::gguf_gemm_computes(p)) continue;
                    const bool half = p == gguf_precision::half;
                    for (long tile : { 1L, 6L, rows })
                    {
                        const size_t activations = half ? (static_cast<size_t>(n) * cols * 2 + 15) / 16 * 16 : 0;
                        resizable_tensor work;
                        work.set_size(static_cast<long long>((activations + static_cast<size_t>(tile) * cols
                            * (half ? 2 : 4) + 3) / 4));
                        saturated = 0;
                        y = 1e30f;
                        tt::gguf_gemm(y, x, w, type, rows, cols, row_bytes, work, p, saturated);
                        const float apart = max(abs(mat(expected) - mat(y)));
                        // The floats agree within their rounding; the tensor cores and the halves keep 10 and 11 bits.
                        DLIB_TEST_MSG(apart < (p == gguf_precision::single ? 2e-4f : 2e-2f) * size, "type " << type
                            << ", " << n << " rows, tiles of " << tile << ", arithmetic " << static_cast<int>(p) << ": "
                            << apart);
                        DLIB_TEST_MSG(saturated.host()[0] == 0, "type " << type << ": no value beyond the halves");
                    }
                }
            }
        }
        DLIB_TEST(tt::gguf_gemm_computes(gguf_precision::single));
        if (tt::gguf_gemm_computes(gguf_precision::half))
        {
            // An activation beyond the halves is told.
            const long rows = 5, cols = 64, n = 4;
            resizable_tensor w, x(n, cols), y(n, rows), saturated(1), work;
            store_bytes(w, random_weight_rows(gguf_blocks::f32, rows, cols, rnd));
            for (float& v : x) v = rnd.get_random_gaussian();
            x.host()[cols + 7] = -1e6f;
            work.set_size(n * cols + rows * cols);
            saturated = 0;
            tt::gguf_gemm(y, x, w, gguf_blocks::f32, rows, cols, gguf_blocks::row_bytes(gguf_blocks::f32, cols), work,
                gguf_precision::half, saturated);
            DLIB_TEST(saturated.host()[0] != 0);
        }

        for (long width : { 48L, 256L, 700L })
        {
            const long rows = 4;
            resizable_tensor x(rows, width), addend(rows, width), gamma(1, width), expected(rows, width);
            resizable_tensor normalized(rows, width);
            for (float& v : x) v = 3 * rnd.get_random_gaussian();
            for (float& v : addend) v = 3 * rnd.get_random_gaussian();
            for (float& v : gamma) v = 1 + 0.3f * rnd.get_random_gaussian();
            resizable_tensor sum(x);
            for (size_t i = 0; i < sum.size(); ++i) sum.host()[i] += 0.7f * addend.host()[i];
            cpu::rms_norm_rows(expected, sum, gamma, 1e-6f);
            tt::add_rms_norm_rows(normalized, x, addend, 0.7f, gamma, 1e-6f);
            DLIB_TEST_MSG(max(abs(mat(sum) - mat(x))) < 1e-5, max(abs(mat(sum) - mat(x))));
            DLIB_TEST_MSG(max(abs(mat(expected) - mat(normalized))) < 1e-4, max(abs(mat(expected) - mat(normalized))));
        }

        for (int neox = 0; neox < 2; ++neox)
        {
            const long n = 33, q_heads = 4, k_heads = 2, hd = 32, dims = 24;
            resizable_tensor q(n, q_heads * hd), k(n, k_heads * hd), angles(n, dims);
            for (float& v : q) v = rnd.get_random_gaussian();
            for (float& v : k) v = rnd.get_random_gaussian();
            for (float& v : angles) v = rnd.get_random_gaussian();
            resizable_tensor q_alone(q), k_alone(k);
            cpu::rope_rows(q_alone, angles, q_heads, hd, dims, neox != 0);
            cpu::rope_rows(k_alone, angles, k_heads, hd, dims, neox != 0);
            tt::rope_rows(q, q_heads, k, k_heads, angles, hd, dims, neox != 0);
            DLIB_TEST_MSG(max(abs(mat(q) - mat(q_alone))) < 1e-5, max(abs(mat(q) - mat(q_alone))));
            DLIB_TEST_MSG(max(abs(mat(k) - mat(k_alone))) < 1e-5, max(abs(mat(k) - mat(k_alone))));
        }

        for (int q8 = 0; q8 < 2; ++q8)
            for (long rep : { 1L, 4L })
            {
                kv_page_layout layout;
                layout.layers = 2;
                layout.heads = 2;
                layout.head_dim = 64;
                layout.block_positions = 8;
                layout.q8 = q8 != 0;
                const long hd = layout.head_dim, q_heads = layout.heads * rep, kvd = layout.heads * hd;
                const long layer = 1, page_count = 6, first_entry = 2;
                // The blocks of the sequence lie in pages out of order, after other entries of the table.
                const std::vector<int32_t> table = { 5, 5, 3, 0, 4, 1 };
                resizable_tensor pages, tables;
                pages.set_size(static_cast<long long>((static_cast<size_t>(page_count) * layout.page_bytes() + 3) / 4));
                pages = 0;
                store_integers(tables, table);
                for (long first_position : { 0L, 11L })
                {
                    const long n = 13, span = first_position + n;
                    resizable_tensor keys(span, kvd), values(span, kvd), slots, tokens;
                    for (float& v : keys) v = rnd.get_random_gaussian();
                    for (float& v : values) v = rnd.get_random_gaussian();
                    std::vector<int32_t> where, plan;
                    for (long p = 0; p < span; ++p)
                        where.push_back(table[static_cast<size_t>(first_entry + p / layout.block_positions)]
                            * static_cast<int32_t>(layout.block_positions)
                            + static_cast<int32_t>(p % layout.block_positions));
                    for (long t = 0; t < n; ++t)
                    {
                        plan.push_back(static_cast<int32_t>(first_position + t));
                        plan.push_back(static_cast<int32_t>(first_entry));
                    }
                    store_integers(slots, where);
                    store_integers(tokens, plan);
                    cpu::paged_kv_store(pages, layout, layer, keys, values, slots);
                    resizable_tensor q(n, q_heads * hd), expected(n, q_heads * hd), out(n, q_heads * hd);
                    for (float& v : q) v = rnd.get_random_gaussian();
                    cpu::paged_attention(expected, q, q_heads, 0.2f, pages, layout, layer, tokens, tables, span);
                    for (long run : { n, 5L })
                    {
                        resizable_tensor work;
                        work.set_size(static_cast<long long>(tt::paged_attention_rows_scratch(n, q_heads, layout,
                            first_position, run)));
                        out = 1e30f;
                        tt::paged_attention_rows(out, q, q_heads, 0.2f, pages, layout, layer, tables, first_entry,
                            first_position, work);
                        const float apart = max(abs(mat(expected) - mat(out)));
                        DLIB_TEST_MSG(apart < 1e-4, "attention of consecutive tokens, " << (q8 ? "8 bits" : "halves")
                            << ", " << q_heads << " heads, from position " << first_position << ", runs of " << run
                            << ": " << apart);
                    }
                }
            }
    }

    // tt::mamba3_step() over several tokens in one call against one call per token.
    void test_mamba3_step_tokens()
    {
        dlib::rand rnd;
        const long geometries[3][5] = { { 3, 5, 8, 2, 2 }, { 2, 7, 12, 4, 0 }, { 6, 8, 32, 3, 16 } };
        for (const auto& c : geometries)
        {
            const long H = c[0], P = c[1], N = c[2], R = c[3], S = c[4];
            const long T = 6, in_w = H * (2 * P + 3), dyn_w = 2 * R * N + S;
            resizable_tensor norms(2, N), bias(H, 2 * R, N), misc(2, H), mxz(H, 2 * R, P), mo(H, R, P);
            for (float& v : norms) v = 1 + 0.2f * rnd.get_random_gaussian();
            for (float& v : bias) v = 1 + 0.3f * rnd.get_random_gaussian();
            for (float& v : misc) v = 0.5f * rnd.get_random_gaussian();
            for (float& v : mxz) v = 0.7f + 0.3f * rnd.get_random_gaussian();
            for (float& v : mo) v = 0.5f + 0.2f * rnd.get_random_gaussian();
            resizable_tensor in(T, 1, 1, in_w), dyn(T, 1, 1, dyn_w), together(T, 1, 1, H * P), apart(T, 1, 1, H * P);
            for (float& v : in) v = rnd.get_random_gaussian();
            for (float& v : dyn) v = 1.5f * rnd.get_random_gaussian();
            resizable_tensor s1(1, 1, 1, H * S + H * P * N + H * R * N + H * P), s2;
            s1 = 0;
            s2 = s1;
            tt::mamba3_step(together, s1, in, dyn, norms, bias, misc, mxz, mo, 1e-6f);
            resizable_tensor one_in(1, 1, 1, in_w), one_dyn(1, 1, 1, dyn_w), one_out(1, 1, 1, H * P);
            for (long t = 0; t < T; ++t)
            {
                std::memcpy(one_in.host_write_only(), in.host() + t * in_w, sizeof(float) * in_w);
                std::memcpy(one_dyn.host_write_only(), dyn.host() + t * dyn_w, sizeof(float) * dyn_w);
                tt::mamba3_step(one_out, s2, one_in, one_dyn, norms, bias, misc, mxz, mo, 1e-6f);
                std::memcpy(apart.host() + t * H * P, one_out.host(), sizeof(float) * H * P);
            }
            const float outputs = max(abs(mat(together) - mat(apart))), states = max(abs(mat(s1) - mat(s2)));
            DLIB_TEST_MSG(outputs <= 1e-5 * (1 + max(abs(mat(apart)))), outputs);
            DLIB_TEST_MSG(states <= 1e-5 * (1 + max(abs(mat(s2)))), states);
        }
    }

// ----------------------------------------------------------------------------------------

    class gguf_tester : public tester
    {
    public:
        gguf_tester (
        ) :
            tester ("test_gguf",
                    "Runs tests on the GGUF reader (metadata, tensors, dequantization, damaged files) and on "
                    "the hybrid decoders of the runtime engine.")
        {}

        void perform_test (
        )
        {
            print_spinner();
            test_intact();
            print_spinner();
            test_damaged();
            print_spinner();
            test_bounded();
            print_spinner();
            test_geodesic_update();
            test_mamba3_step_tokens();
            print_spinner();
            test_long_pass_operations();
            print_spinner();
            test_expert_operations();
            print_spinner();
            test_hybrid_decoder();
#ifdef DLIB_USE_CUDA
            print_spinner();
            test_mamba3_step_devices();
            print_spinner();
            test_expert_operations_devices();
#endif
        }
    } a;

}
