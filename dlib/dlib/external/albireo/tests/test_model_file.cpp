// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the loading of damaged GGUF files.

        test_model_file [folder]

    A tiny model is written to the folder (the current one by default), then copies of it
    with one field of the header damaged: truncated files, counts and sizes beyond every
    bound, dimensions whose products overflow, offsets outside the file, an absurd
    vocabulary, a broken alignment, non-finite numbers.  Each copy goes through the loading
    of the engine (model_file, read_model_config, the tokenizer and cpu_runtime) and must be
    refused with an exception, without a crash and without any allocation of more than
    256 MB; the intact file must load and compute a pass.
*/

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <dlib/tokenizer/hf_tokenizer.h>

#include "../engine/cpu_runtime.h"
#include "tiny_gguf.h"

namespace
{
    // The largest allocation a load may ask for; beyond, the allocator refuses it and the
    // test counts it.
    const size_t allocation_limit = size_t(256) << 20;
    std::atomic<long> large_allocations{0};

    void* allocate(size_t n)
    {
        if (n > allocation_limit)
        {
            ++large_allocations;
            throw std::bad_alloc();
        }
        if (void* p = std::malloc(n ? n : 1)) return p;
        throw std::bad_alloc();
    }
}

void* operator new(size_t n) { return allocate(n); }
void* operator new[](size_t n) { return allocate(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

using namespace albireo;
using albireo_test::tiny_gguf;

namespace
{
    int failures = 0;

    void expect(bool ok, const std::string& what)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s\n", what.c_str());
        }
    }

    // The loading of the engine, then a pass over a few tokens; the reason of a refusal.
    std::string load(const std::string& path)
    {
        try
        {
            auto file = std::make_shared<model_file>(path);
            const model_config cfg = read_model_config(*file);
            dlib::hf_tokenizer tok;
            tok.load_from_gguf(file->reader());
            if (static_cast<long>(tok.size()) != cfg.vocab) return "vocabulary";
            thread_pool pool(2);
            cpu_runtime rt(file, cfg, pool, size_t(4) << 20, 16);
            const int seq = rt.create_sequence();
            if (!rt.reserve_sequence(seq, 8)) return "reserve";
            forward_batch b;
            b.tokens = { 1, 2, 3, 4 };
            b.segments.push_back({ seq, 0, 4 });
            b.logit_rows.push_back(3);
            std::vector<float> logits;
            rt.forward(b, logits);
            if (static_cast<long>(logits.size()) != cfg.vocab) return "logits";
            for (float x : logits)
                if (!std::isfinite(x)) return "non-finite logits";
            return std::string();
        }
        catch (const std::exception& e)
        {
            return std::string("refused: ") + e.what();
        }
    }

    void refused(const tiny_gguf& model, const std::string& name, const std::vector<unsigned char>& bytes,
        const std::string& path)
    {
        expect(model.save(path, bytes), "write " + path);
        const long before = large_allocations.load();
        const std::string r = load(path);
        expect(r.compare(0, 8, "refused:") == 0, name + " loaded (" + (r.empty() ? std::string("no error") : r) + ")");
        expect(large_allocations.load() == before, name + " asked for more than 256 MB");
    }
}

int main(int argc, char** argv)
{
    const std::string folder = argc > 1 ? std::string(argv[1]) + "/" : std::string();
    const std::string path = folder + "test_model_file.gguf";
    const tiny_gguf model = albireo_test::tiny_gguf_writer().build();

    // The intact file.
    expect(model.save(path, model.bytes), "write the model");
    const std::string intact = load(path);
    expect(intact.empty(), "the intact model does not load: " + intact);
    std::printf("  intact model: %s\n", intact.empty() ? "loaded, one pass computed" : intact.c_str());

    long cases = 0;
    auto damaged = [&](const std::string& name, const std::vector<unsigned char>& bytes) {
        refused(model, name, bytes, path);
        ++cases;
    };

    // Truncated files.
    for (size_t cut : { size_t(0), size_t(3), size_t(8), size_t(20), size_t(100), model.header_end / 2,
        model.header_end - 1, model.header_end + 100, model.bytes.size() - 1 })
    {
        std::vector<unsigned char> b(model.bytes.begin(), model.bytes.begin() + static_cast<std::ptrdiff_t>(cut));
        damaged("a file truncated at " + std::to_string(cut) + " bytes", b);
    }

    // Counts, lengths and dimensions beyond every bound.
    const uint64_t huge[] = { uint64_t(1) << 31, uint64_t(1) << 40, uint64_t(1) << 62, (uint64_t(1) << 62) + 2,
        ~uint64_t(0) };
    const char* fields[] = { "tensor_count", "kv_count", "count tokenizer.ggml.tokens", "count tokenizer.ggml.merges",
        "count tokenizer.ggml.token_type", "length key qwen3.block_count", "length value tokenizer.chat_template",
        "length name token_embd.weight", "dim0 token_embd.weight", "dim1 token_embd.weight",
        "dim0 blk.0.ffn_down.weight", "offset token_embd.weight", "offset output_norm.weight",
        "kv qwen3.block_count", "kv qwen3.embedding_length", "kv qwen3.feed_forward_length",
        "kv qwen3.attention.head_count", "kv qwen3.attention.head_count_kv", "kv qwen3.attention.key_length",
        "kv qwen3.attention.value_length", "kv qwen3.rope.dimension_count", "kv qwen3.context_length",
        "kv general.alignment" };
    for (const char* f : fields)
    {
        expect(model.find(f) != nullptr, std::string("no field ") + f);
        for (uint64_t v : huge) damaged(std::string(f) + " = " + std::to_string(v), model.with(f, v));
    }
    // Values a sane file never holds.
    damaged("no layer", model.with("kv qwen3.block_count", 0));
    damaged("more layers than tensors", model.with("kv qwen3.block_count", 1000));
    damaged("heads not a multiple of the key/value heads", model.with("kv qwen3.attention.head_count_kv", 3));
    damaged("an odd head dimension", model.with("kv qwen3.attention.key_length", 15));
    damaged("an alignment of 0", model.with("kv general.alignment", 0));
    damaged("an alignment of 48", model.with("kv general.alignment", 48));
    damaged("a tensor of no dimension", model.with("ndim blk.0.attn_q.weight", 0));
    damaged("a tensor of 9 dimensions", model.with("ndim blk.0.attn_q.weight", 9));
    damaged("an unknown tensor type", model.with("type blk.0.attn_q.weight", 99));
    damaged("an unknown metadata type", model.with("vtype qwen3.block_count", 99));
    damaged("an unknown element type", model.with("element type tokenizer.ggml.token_type", 99));
    damaged("a tensor at the end of the file", model.with("offset blk.1.ffn_down.weight",
        model.bytes.size() - model.header_end - 16));
    damaged("an empty vocabulary", model.with("count tokenizer.ggml.tokens", 0));
    const uint32_t nan = 0x7FC00000u, inf = 0x7F800000u;
    damaged("a rotary base NaN", model.with("kv qwen3.rope.freq_base", nan));
    damaged("an infinite rotary base", model.with("kv qwen3.rope.freq_base", inf));
    damaged("an infinite epsilon", model.with("kv qwen3.attention.layer_norm_rms_epsilon", inf));

    std::remove(path.c_str());
    std::printf("  %ld damaged files refused, no allocation beyond %zu MB\n", cases, allocation_limit >> 20);
    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
