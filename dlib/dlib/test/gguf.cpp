// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/data_io/gguf_dequantize.h>
#include <dlib/data_io/gguf_reader.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
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
        vector<unsigned char> bytes;
        vector<pair<string, size_t>> fields;
        size_t data_start = 0;

        size_t at(const string& name) const
        {
            for (const auto& f : fields)
                if (f.first == name) return f.second;
            DLIB_TEST_MSG(false, "no field " << name);
            return 0;
        }

        vector<unsigned char> with_u64(const string& name, uint64_t v) const
        {
            vector<unsigned char> b = bytes;
            for (int i = 0; i < 8; ++i) b[at(name) + i] = static_cast<unsigned char>(v >> (8 * i));
            return b;
        }

        vector<unsigned char> with_u32(const string& name, uint32_t v) const
        {
            vector<unsigned char> b = bytes;
            for (int i = 0; i < 4; ++i) b[at(name) + i] = static_cast<unsigned char>(v >> (8 * i));
            return b;
        }

    };

    class gguf_builder
    {
    public:
        gguf_image build()
        {
            out = gguf_image();
            u32(0x46554747u);
            mark("version");
            u32(3);
            mark("tensor count");
            u64(3);
            u64(5);     // metadata entries
            key("general.architecture", 8);
            str("llama");
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

        void tensor(const string& name, const vector<uint64_t>& dims, uint32_t type, uint64_t offset)
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
    string refusal(const vector<unsigned char>& bytes)
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

        vector<float> values;
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
            vector<uint8_t> raw;
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
            const vector<unsigned char> b(image.bytes.begin(), image.bytes.begin() + static_cast<std::ptrdiff_t>(cut));
            DLIB_TEST_MSG(!refusal(b).empty(), "truncated at " << cut);
        }
        vector<unsigned char> magic = image.bytes;
        magic[0] = 'X';
        DLIB_TEST(refusal(magic).find("bad magic") != string::npos);
        DLIB_TEST(refusal(image.with_u32("version", 4)).find("version") != string::npos);
        DLIB_TEST(refusal(image.with_u64("tensor count", uint64_t(1) << 40)).find("corrupt header") != string::npos);

        // A key and a tensor name given twice: llama.rope.freq_base renamed general.architecture,
        // of the same length, and the tensor blocks renamed halves.
        {
            vector<unsigned char> b = image.bytes;
            const string name = "general.architecture";
            for (size_t i = 0; i < name.size(); ++i)
                b[image.at("key llama.rope.freq_base") + 8 + i] = static_cast<unsigned char>(name[i]);
            const string m = refusal(b);
            DLIB_TEST_MSG(m.find("duplicate metadata key 'general.architecture'") != string::npos, m);
        }
        {
            vector<unsigned char> b = image.bytes;
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
            vector<unsigned char> b = image.with_u64("dim0 blocks", 16);
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

    class gguf_tester : public tester
    {
    public:
        gguf_tester (
        ) :
            tester ("test_gguf",
                    "Runs tests on the GGUF reader: metadata, tensors, dequantization and damaged files.")
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
        }
    } a;

}
