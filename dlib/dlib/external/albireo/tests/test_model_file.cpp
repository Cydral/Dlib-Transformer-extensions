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

    A copy of more than 2 GiB, whose last tensor lies beyond 2 GiB, the rest a hole that
    takes no room on the disk, must load as well, through the mapped file and through a
    file stream: the standard library of Visual Studio counts a stream buffer in an int.
    The values of that tensor are distinct from those of every other tensor, so that a read
    at another offset shows: each value read is checked, and the engine must compute the
    logits of the same model written in a small file.
*/

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    // The loading of the engine, then a pass over a few tokens, whose logits go to out when
    // given; the reason of a refusal.
    std::string load(const std::string& path, std::vector<float>* out = nullptr)
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
            if (out) out->swap(logits);
            return std::string();
        }
        catch (const std::exception& e)
        {
            return std::string("refused: ") + e.what();
        }
    }

    // Sets the field of the given label in bytes, as tiny_gguf::with() does in a copy.
    void set_field(const tiny_gguf& model, std::vector<unsigned char>& bytes, const std::string& label, uint64_t value)
    {
        const albireo_test::gguf_field* f = model.find(label);
        expect(f != nullptr, "no field " + label);
        if (f)
            for (size_t i = 0; i < f->size; ++i) bytes[f->offset + i] = static_cast<unsigned char>(value >> (8 * i));
    }

    /*
        A copy of the file with one character of a string changed in place: the string whose
        length field bears the label "length " + label, its character at index set to c.
    */
    std::vector<unsigned char> renamed(const tiny_gguf& model, const std::string& label, size_t index, char c)
    {
        std::vector<unsigned char> b = model.bytes;
        const albireo_test::gguf_field* f = model.find("length " + label);
        expect(f != nullptr, "no string " + label);
        if (f) b[f->offset + f->size + index] = static_cast<unsigned char>(c);
        return b;
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

    // Writes head at the start of a file and tail at offset at, the bytes between them a hole
    // that takes no room on the disk (a sparse file on NTFS).
    bool write_sparse(const std::string& path, const std::vector<unsigned char>& head, uint64_t at,
        const std::vector<unsigned char>& tail)
    {
#ifdef _WIN32
        HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        DWORD done = 0;
        DeviceIoControl(h, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &done, nullptr);
        auto put = [&](uint64_t offset, const std::vector<unsigned char>& b) {
            LARGE_INTEGER p;
            p.QuadPart = static_cast<LONGLONG>(offset);
            DWORD written = 0;
            return SetFilePointerEx(h, p, nullptr, FILE_BEGIN) != 0
                && WriteFile(h, b.data(), static_cast<DWORD>(b.size()), &written, nullptr) != 0
                && written == b.size();
        };
        const bool ok = put(0, head) && put(at, tail);
        return CloseHandle(h) != 0 && ok;
#else
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return false;
        auto put = [&](uint64_t offset, const std::vector<unsigned char>& b) {
            return ::pwrite(fd, b.data(), b.size(), static_cast<off_t>(offset)) == static_cast<ssize_t>(b.size());
        };
        const bool ok = put(0, head) && put(at, tail);
        return ::close(fd) == 0 && ok;
#endif
    }

    // The bytes of a tensor read by r, or an empty vector with the reason in why.
    std::vector<uint8_t> tensor_bytes(dlib::gguf_reader& r, const std::string& name, std::string& why)
    {
        try
        {
            for (const auto& t : r.tensors())
                if (t.name == name)
                {
                    std::vector<uint8_t> raw;
                    r.read_tensor_raw(t, raw);
                    return raw;
                }
            why = "no tensor " + name;
        }
        catch (const std::exception& e)
        {
            why = e.what();
        }
        return std::vector<uint8_t>();
    }

    // The value at index k of output_norm.weight in the large file: no other tensor holds it.
    float pattern(size_t k)
    {
        return 0.5f + static_cast<float>(k) / 64.0f;
    }

    // Whether raw holds the floats pattern(0), pattern(1)... of a tensor of count values.
    bool holds_pattern(const std::vector<uint8_t>& raw, size_t count)
    {
        if (raw.size() != count * sizeof(float)) return false;
        for (size_t k = 0; k < count; ++k)
        {
            float v;
            std::memcpy(&v, raw.data() + k * sizeof(float), sizeof(float));
            if (v != pattern(k)) return false;
        }
        return true;
    }

    /*
        The tiny model with its last tensor, output_norm.weight, filled with the pattern and
        moved 2 GiB further, read in every way the engine reads it.  The engine must compute
        the logits of the same model written without the hole.
    */
    void large_file(const tiny_gguf& model, const std::string& folder)
    {
        const char* field = "offset output_norm.weight";
        const albireo_test::gguf_field* f = model.find(field);
        expect(f != nullptr && f->size == 8, std::string("no field ") + field);
        if (!f) return;
        uint64_t original = 0;
        for (size_t i = 0; i < 8; ++i) original |= static_cast<uint64_t>(model.bytes[f->offset + i]) << (8 * i);
        const size_t start = model.header_end + static_cast<size_t>(original);
        const size_t count = (model.bytes.size() - start) / sizeof(float);
        std::vector<unsigned char> bytes = model.bytes;
        for (size_t k = 0; k < count; ++k)
        {
            const float v = pattern(k);
            std::memcpy(&bytes[start + k * sizeof(float)], &v, sizeof(float));
        }

        const std::string small_path = folder + "test_model_file_pattern.gguf";
        std::vector<float> reference;
        expect(model.save(small_path, bytes), "write " + small_path);
        const std::string r0 = load(small_path, &reference);
        std::remove(small_path.c_str());
        expect(r0.empty(), "the model with the pattern does not load: " + r0);

        const std::string path = folder + "test_model_file_large.gguf";
        const uint64_t moved = original + (uint64_t(1) << 31);
        std::vector<unsigned char> head = bytes;
        set_field(model, head, field, moved);
        const std::vector<unsigned char> tail(head.begin() + static_cast<std::ptrdiff_t>(start), head.end());
        head.resize(start);
        const uint64_t size = model.header_end + moved + tail.size();
        if (!write_sparse(path, head, model.header_end + moved, tail))
        {
            expect(false, "write the file of more than 2 GiB");
            return;
        }
        std::vector<float> logits;
        const std::string r = load(path, &logits);
        expect(r.empty(), "the file of more than 2 GiB does not load: " + r);
        bool same = r.empty() && r0.empty() && logits.size() == reference.size();
        for (size_t i = 0; same && i < logits.size(); ++i)
            same = std::fabs(logits[i] - reference[i]) <= 1e-5f * std::max(1.0f, std::fabs(reference[i]));
        expect(same, "the logits computed from the file of more than 2 GiB differ from those of the small file");
        try
        {
            mapped_file m(path);
            dlib::gguf_reader mapped(m.data(), m.size());
            std::string why;
            expect(mapped.size() == size, "the mapped reader gives " + std::to_string(mapped.size()) + " bytes");
            expect(holds_pattern(tensor_bytes(mapped, "output_norm.weight", why), count),
                "the mapped reader does not read the values of the tensor beyond 2 GiB " + why);
            dlib::gguf_reader stream(path);
            expect(stream.size() == size, "the file reader gives " + std::to_string(stream.size()) + " bytes");
            expect(holds_pattern(tensor_bytes(stream, "output_norm.weight", why), count),
                "the file reader does not read the values of the tensor beyond 2 GiB " + why);
        }
        catch (const std::exception& e)
        {
            expect(false, std::string("the file of more than 2 GiB is refused: ") + e.what());
        }
        std::remove(path.c_str());
        std::printf("  file of %llu bytes, last tensor beyond 2 GiB: %s\n", static_cast<unsigned long long>(size),
            !r.empty() ? r.c_str() : same ? "loaded, the values read and the logits checked"
            : "loaded, other logits");
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
    // On the alignment, so that the bounds of the data, and not the alignment, refuse it.
    damaged("a tensor at the end of the file", model.with("offset blk.1.ffn_down.weight",
        (model.bytes.size() - model.header_end) / 32 * 32 - 32));
    damaged("an empty vocabulary", model.with("count tokenizer.ggml.tokens", 0));
    const uint32_t nan = 0x7FC00000u, inf = 0x7F800000u;
    damaged("a rotary base NaN", model.with("kv qwen3.rope.freq_base", nan));
    damaged("an infinite rotary base", model.with("kv qwen3.rope.freq_base", inf));
    damaged("an infinite epsilon", model.with("kv qwen3.attention.layer_norm_rms_epsilon", inf));
    // Names given twice, which the readers of the file could take for different entries.
    damaged("a metadata key given twice", renamed(model, "key tokenizer.ggml.eos_token_id", 15, 'b'));
    damaged("a tensor name given twice", renamed(model, "name blk.0.attn_k.weight", 11, 'q'));
    // Tensor data off the alignment, and rows that split the blocks of a quantized type.
    damaged("a tensor off the alignment", model.with("offset token_embd.weight", 4));
    {
        std::vector<unsigned char> b = model.with("type blk.0.attn_q.weight", 8);   // Q8_0, blocks of 32
        set_field(model, b, "dim0 blk.0.attn_q.weight", 80);
        set_field(model, b, "dim1 blk.0.attn_q.weight", 64);
        damaged("rows of 80 values in blocks of 32", b);
    }

    std::remove(path.c_str());
    std::printf("  %ld damaged files refused, no allocation beyond %zu MB\n", cases, allocation_limit >> 20);
    large_file(model, folder);
    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
