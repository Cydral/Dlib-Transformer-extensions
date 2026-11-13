// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks and measures the OpenVINO graph of a model (engine/ov_graph.h) against the
    processor runtime.

        test_ov_graph model.gguf [device] [threads] [f16|u8|f16-mask|u8-mask] [prompt] [chunk] [steps]

    device is a device of OpenVINO, CPU by default; threads the threads of both runtimes,
    those of the machine by default; f16 or u8 the keys and values of the graph (half
    precision by default), -mask giving the processor the graph of the other devices, whose
    attention takes the causal mask as an input.  The program reads a text of prompt tokens
    (512) by passes of chunk tokens (256), then writes steps tokens (32) one at a time, each
    step reading the next token of the text, through the processor runtime with keys and
    values in half precision, which is the reference, then through the graph:
        1. the logits of the last token of each pass and of each step, compared as
           distributions: the share of steps where both runtimes give the same most
           probable token, and the divergence of the distribution of the graph from the
           reference;
        2. the speeds of both runtimes: the first pass of the prompt, the passes that follow
           it, which attend to the positions before them, and the steps;
        3. the operations of the sequences: a sequence cut inside its positions, then read
           further; a second sequence that copies the start of the first; both stepping in
           one batch, then again after trim_memory();
        4. the logits of the processor runtime with keys and values in 8 bits, the default
           of the engine, against the reference, as a measure of the agreement to expect.

    The program fails when the graph gives another most probable token than the reference
    in more than one step out of eight, or when its mean divergence exceeds 0.05.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <dlib/tokenizer/hf_tokenizer.h>

#include "../common/system_info.h"
#include "../engine/cpu_runtime.h"
#include "../engine/ov_graph.h"

using namespace albireo;
using clock_type = std::chrono::steady_clock;

namespace
{
    double seconds_since(clock_type::time_point t)
    {
        return std::chrono::duration<double>(clock_type::now() - t).count();
    }

    struct agreement
    {
        long rows = 0;
        long same_top = 0;
        double divergence = 0;
        double max_divergence = 0;

        void add(const float* ref, const float* x, long n)
        {
            long a = 0, b = 0;
            double mr = ref[0], mx = x[0];
            for (long i = 1; i < n; ++i)
            {
                if (ref[i] > ref[a]) a = i;
                if (x[i] > x[b]) b = i;
                mr = std::max(mr, static_cast<double>(ref[i]));
                mx = std::max(mx, static_cast<double>(x[i]));
            }
            double zr = 0, zx = 0;
            for (long i = 0; i < n; ++i)
            {
                zr += std::exp(ref[i] - mr);
                zx += std::exp(x[i] - mx);
            }
            const double lzr = std::log(zr), lzx = std::log(zx);
            double kl = 0;
            for (long i = 0; i < n; ++i)
            {
                const double lr = ref[i] - mr - lzr;
                const double lx = x[i] - mx - lzx;
                kl += std::exp(lr) * (lr - lx);
            }
            ++rows;
            if (a == b) ++same_top;
            divergence += kl;
            max_divergence = std::max(max_divergence, kl);
        }

        double top_share() const { return rows ? static_cast<double>(same_top) / rows : 1.0; }
        double mean() const { return rows ? divergence / rows : 0.0; }
    };

    struct timings
    {
        double first_pass = 0;      // seconds of the first pass of the prompt
        long first_tokens = 0;
        double later_passes = 0;
        long later_tokens = 0;
        double steps = 0;
        long step_count = 0;
    };

    // The resident memory of the process that no file backs, in megabytes, on Linux; 0 elsewhere.
    long resident_megabytes()
    {
        long kb = 0;
        if (FILE* f = std::fopen("/proc/self/status", "r"))
        {
            char line[256];
            while (std::fgets(line, sizeof(line), f))
                if (std::sscanf(line, "RssAnon: %ld", &kb) == 1) break;
            std::fclose(f);
        }
        return kb / 1024;
    }

    // The hidden state of the last token of the latest pass.
    std::vector<float>& last_hidden()
    {
        static std::vector<float> h;
        return h;
    }

    double cosine(const std::vector<float>& a, const std::vector<float>& b)
    {
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / std::sqrt(std::max(1e-30, na * nb));
    }

    // One pass of tokens [first, first + count) of the text for one sequence, with the logits of its last token.
    void run_pass(model_runtime& rt, int seq, const std::vector<int>& text, long first, long count,
        std::vector<float>& logits)
    {
        forward_batch b;
        b.tokens.assign(text.begin() + first, text.begin() + first + count);
        forward_batch::segment s;
        s.sequence = seq;
        s.start = 0;
        s.count = count;
        b.segments.push_back(s);
        b.logit_rows.push_back(count - 1);
        b.hidden_only = rt.config().is_embedding();
        if (!rt.reserve_sequence(seq, rt.sequence_length(seq) + count))
            throw std::runtime_error("no room for the keys and values");
        rt.forward(b, logits);
        // An embedding model gives its hidden states, compared as the logits of a language model.
        if (logits.empty()) logits = rt.hidden();
        last_hidden() = rt.hidden();
    }

    // The prompt by chunks, then the steps: the logits of every pass and step, one row each.
    std::vector<std::vector<float>> script(model_runtime& rt, int seq, const std::vector<int>& text, long prompt,
        long chunk, long steps, timings& t)
    {
        std::vector<std::vector<float>> rows;
        std::vector<float> logits;
        {
            // A first pass and step on a sequence of their own prepare the kernels and buffers of both shapes.
            const int warm = rt.create_sequence();
            run_pass(rt, warm, text, 0, std::min(chunk, prompt), logits);
            run_pass(rt, warm, text, std::min(chunk, prompt), 1, logits);
            rt.release_sequence(warm);
        }
        for (long at = 0; at < prompt; at += chunk)
        {
            const long n = std::min(chunk, prompt - at);
            const auto start = clock_type::now();
            run_pass(rt, seq, text, at, n, logits);
            const double s = seconds_since(start);
            if (at == 0)
            {
                t.first_pass += s;
                t.first_tokens += n;
            }
            else
            {
                t.later_passes += s;
                t.later_tokens += n;
            }
            rows.push_back(logits);
        }
        const auto start = clock_type::now();
        for (long i = 0; i < steps; ++i)
        {
            run_pass(rt, seq, text, prompt + i, 1, logits);
            rows.push_back(logits);
        }
        t.steps += seconds_since(start);
        t.step_count += steps;
        return rows;
    }

    void report_speed(const char* name, const timings& t)
    {
        std::printf("  %-22s first pass %7.1f tokens/s, later passes %7.1f tokens/s, steps %6.2f tokens/s\n", name,
            t.first_tokens / std::max(1e-9, t.first_pass), t.later_tokens / std::max(1e-9, t.later_passes),
            t.step_count / std::max(1e-9, t.steps));
    }

    void compare(const char* what, const std::vector<std::vector<float>>& ref,
        const std::vector<std::vector<float>>& x, long vocab, agreement& a)
    {
        agreement local;
        for (size_t i = 0; i < ref.size() && i < x.size(); ++i)
        {
            local.add(ref[i].data(), x[i].data(), vocab);
            a.add(ref[i].data(), x[i].data(), vocab);
        }
        std::printf("  %-40s same top token %5.1f%%, divergence mean %.5f, max %.5f\n", what,
            100.0 * local.top_share(), local.mean(), local.max_divergence);
    }

    std::vector<int> make_text(const dlib::hf_tokenizer& tok, long count)
    {
        const std::string paragraph =
            "The history of the city goes back to the time when the river was the main road of the region, and the "
            "merchants who came from the south built their houses along its banks. Over the centuries the town grew "
            "into a market where wool, grain and salt were traded, and its people learned to read and write in the "
            "schools that the monks had opened near the old bridge. In the spring of every year, a fair brought "
            "musicians, painters and travelling scholars, who argued about astronomy, medicine and law in the inns "
            "of the lower town until late at night. ";
        std::vector<int> text;
        while (static_cast<long>(text.size()) < count)
        {
            const std::vector<int> part = tok.encode(paragraph);
            text.insert(text.end(), part.begin(), part.end());
        }
        text.resize(static_cast<size_t>(count));
        return text;
    }
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: test_ov_graph model.gguf [device] [threads] [f16|u8|f16-mask|u8-mask] [prompt] [chunk] "
            "[steps]\n");
        return 2;
    }
    try
    {
        const std::string device = argc > 2 ? argv[2] : "CPU";
        const int threads = argc > 3 && std::atoi(argv[3]) > 0 ? std::atoi(argv[3])
            : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        const std::string kv = argc > 4 ? argv[4] : "f16";
        const bool kv_half = kv.compare(0, 2, "u8") != 0;
        const bool explicit_mask = kv.size() > 5 && kv.compare(kv.size() - 5, 5, "-mask") == 0;
        const long prompt = argc > 5 ? std::max(2L, std::atol(argv[5])) : 512;
        const long chunk = argc > 6 ? std::max(1L, std::atol(argv[6])) : 256;
        const long steps = argc > 7 ? std::max(1L, std::atol(argv[7])) : 32;

        auto file = std::make_shared<model_file>(argv[1]);
        model_config cfg = read_model_config(*file);
        const std::string why = ov_graph_unsupported(cfg, *file);
        if (!why.empty())
        {
            std::printf("the graph does not compute this model: %s\n", why.c_str());
            return 0;
        }
        dlib::hf_tokenizer tok;
        tok.load_from_gguf(file->reader());
        const long total = prompt + steps + 64;
        const std::vector<int> text = make_text(tok, total);
        select_rope_factors(cfg, total);
        const long vocab = cfg.is_embedding() ? cfg.d_model : cfg.vocab;
        std::printf("model: %s\nOpenVINO %s, devices:", describe(cfg).c_str(), ov_graph_version().c_str());
        for (const std::string& d : ov_graph_devices()) std::printf(" %s", d.c_str());
        std::printf("\n");

        const size_t budget = size_t(4) << 30;
        thread_pool pool(threads);
        file->warm_up(false);

        // The reference: the processor runtime with keys and values in half precision.
        timings t_ref, t_q8, t_graph;
        std::vector<std::vector<float>> ref, q8, graph;
        {
            cpu_runtime rt(file, cfg, pool, budget, 256, kv_format::f16);
            rt.set_context(total);
            const int s = rt.create_sequence();
            ref = script(rt, s, text, prompt, chunk, steps, t_ref);
        }
        const std::vector<float> ref_hidden = last_hidden();
        {
            cpu_runtime rt(file, cfg, pool, budget, 256, kv_format::q8);
            rt.set_context(total);
            const int s = rt.create_sequence();
            q8 = script(rt, s, text, prompt, chunk, steps, t_q8);
        }
        if (cfg.is_embedding())
            std::printf("embedding model: the hidden states are compared as the logits of a language model\n");

        ov_graph_settings settings;
        settings.device = device;
        settings.kv_half = kv_half;
        settings.explicit_mask = explicit_mask;
        settings.threads = threads;
        // The folder of the system for temporary files, where the graph writes its weights and deletes them.
        const char* temp = std::getenv("TEMP");
        if (!temp) temp = std::getenv("TMPDIR");
        settings.weights_dir = std::string(temp ? temp : "/tmp") + "/albireo-test-ov-graph";
        const uint64_t commit_before = process_commit_bytes();
        const long rss_before = resident_megabytes();
        std::unique_ptr<ov_graph_runtime> rt = make_ov_graph_runtime(file, cfg, settings, budget);
        rt->set_context(total);
        std::printf("graph: %s\nbuilt and compiled in %.1f s", rt->description().c_str(), rt->build_seconds());
        const uint64_t commit_after = process_commit_bytes();
        if (commit_after > 0)
            std::printf(", the process commits %llu MB more",
                static_cast<unsigned long long>((commit_after - std::min(commit_after, commit_before)) >> 20));
        const long rss_after = resident_megabytes();
        if (rss_after > 0)
            std::printf(", anonymous resident memory %ld MB, %ld MB more", rss_after, rss_after - rss_before);
        std::printf("\n");
        const int s1 = rt->create_sequence();
        graph = script(*rt, s1, text, prompt, chunk, steps, t_graph);
        const std::vector<float> graph_hidden = last_hidden();

        std::printf("speed (%d threads):\n", threads);
        report_speed("processor, f16", t_ref);
        report_speed("processor, 8 bits", t_q8);
        report_speed(("graph, " + device).c_str(), t_graph);

        std::printf("agreement with the processor runtime in half precision:\n");
        agreement all, scratch;
        compare("processor runtime, keys and values in 8 bits", ref, q8, vocab, scratch);
        compare(("graph on " + device).c_str(), ref, graph, vocab, all);

        // The hidden states after the final normalization, which an embedding model returns.
        const double hidden_cosine = cosine(ref_hidden, graph_hidden);
        std::printf("  cosine of the hidden states of the last step: %.6f\n", hidden_cosine);

        // The operations of the sequences, against the processor runtime doing the same.
        {
            cpu_runtime cr(file, cfg, pool, budget, 256, kv_format::f16);
            cr.set_context(total);
            const int c1 = cr.create_sequence();
            timings ignored;
            script(cr, c1, text, prompt, chunk, steps, ignored);
            const long cut = prompt / 2 + 3;
            cr.truncate_sequence(c1, cut);
            rt->truncate_sequence(s1, cut);
            std::vector<std::vector<float>> a, b;
            std::vector<float> la, lb;
            run_pass(cr, c1, text, cut, 5, la);
            run_pass(*rt, s1, text, cut, 5, lb);
            a.push_back(la);
            b.push_back(lb);
            for (long i = 0; i < 6; ++i)
            {
                run_pass(cr, c1, text, cut + 5 + i, 1, la);
                run_pass(*rt, s1, text, cut + 5 + i, 1, lb);
                a.push_back(la);
                b.push_back(lb);
            }
            compare("cut inside the prompt, then read further", a, b, vocab, all);

            // A second sequence copies the start of the first, then both step in one batch.
            const long shared = prompt / 4 + 1;
            const int c2 = cr.create_sequence();
            const int s2 = rt->create_sequence();
            if (!cr.copy_sequence(c1, c2, shared) || !rt->copy_sequence(s1, s2, shared))
                throw std::runtime_error("no room to copy a sequence");
            std::vector<std::vector<float>> ca, cb, ta, tb;
            for (long i = 0; i < 6; ++i)
            {
                // The memory both runtimes give back between two passes, before the last two steps; the caches stay.
                if (i == 4)
                {
                    cr.trim_memory();
                    rt->trim_memory();
                }
                forward_batch batch;
                const long p1 = cr.sequence_length(c1), p2 = cr.sequence_length(c2);
                if (p1 != rt->sequence_length(s1) || p2 != rt->sequence_length(s2))
                    throw std::runtime_error("the lengths of the sequences differ between the runtimes");
                batch.tokens = { text[static_cast<size_t>(p1)], text[static_cast<size_t>(p2)] };
                forward_batch::segment g1, g2;
                g1.sequence = c1; g1.start = 0; g1.count = 1;
                g2.sequence = c2; g2.start = 1; g2.count = 1;
                batch.segments = { g1, g2 };
                batch.logit_rows = { 0, 1 };
                batch.hidden_only = cfg.is_embedding();
                if (!cr.reserve_sequence(c1, p1 + 1) || !cr.reserve_sequence(c2, p2 + 1)
                    || !rt->reserve_sequence(s1, p1 + 1) || !rt->reserve_sequence(s2, p2 + 1))
                    throw std::runtime_error("no room for the keys and values");
                cr.forward(batch, la);
                if (la.empty()) la = cr.hidden();
                batch.segments[0].sequence = s1;
                batch.segments[1].sequence = s2;
                rt->forward(batch, lb);
                if (lb.empty()) lb = rt->hidden();
                for (long r = 0; r < 2; ++r)
                {
                    (i < 4 ? ca : ta).push_back(std::vector<float>(la.begin() + r * vocab,
                        la.begin() + (r + 1) * vocab));
                    (i < 4 ? cb : tb).push_back(std::vector<float>(lb.begin() + r * vocab,
                        lb.begin() + (r + 1) * vocab));
                }
            }
            compare("copied start, two sequences in one batch", ca, cb, vocab, all);
            compare("the same after trim_memory", ta, tb, vocab, all);
        }

        if (cfg.multimodal_rope() || cfg.n_deepstack > 0)
        {
            /*
                An image of 4 x 4 tokens after 5 tokens of text: its rows come from an encoder, with
                DeepStack slices, and take the multimodal positions of a grid; the text and four
                steps follow.
            */
            const long img_start = 5, grid = 4, img = grid * grid, n = 40;
            const long W = cfg.input_width(), d = cfg.d_model;
            const weight_matrix embd = file->matrix("token_embd.weight", cfg.vocab, d);
            forward_batch batch;
            batch.tokens.assign(text.begin(), text.begin() + n);
            uint32_t state = 12345;
            auto noise = [&state]() {
                state = state * 1664525u + 1013904223u;
                return static_cast<float>((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
            };
            long next = 0;
            for (long t = 0; t < n; ++t)
            {
                long p0 = t, p1 = t, p2 = t;
                if (t >= img_start && t < img_start + img)
                {
                    const long k = t - img_start;
                    p0 = img_start;
                    p1 = img_start + k / grid;
                    p2 = img_start + k % grid;
                }
                else if (t >= img_start + img)
                {
                    p0 = p1 = p2 = img_start + grid + (t - img_start - img);
                }
                batch.positions.insert(batch.positions.end(), { p0, p1, p2 });
                next = std::max(next, std::max(p0, std::max(p1, p2)) + 1);
                if (t >= img_start && t < img_start + img)
                {
                    batch.embedding_rows.push_back(t);
                    std::vector<float> row(static_cast<size_t>(W));
                    qk_dequantize_row(embd.type, embd.data + static_cast<size_t>(text[static_cast<size_t>(t)])
                        * embd.row_bytes, row.data(), d);
                    for (long i = 0; i < d; ++i) row[static_cast<size_t>(i)] += 0.01f * noise();
                    for (long i = d; i < W; ++i) row[static_cast<size_t>(i)] = 0.05f * noise();
                    batch.embeddings.insert(batch.embeddings.end(), row.begin(), row.end());
                }
            }
            forward_batch::segment seg;
            seg.start = 0;
            seg.count = n;
            batch.logit_rows.push_back(n - 1);
            cpu_runtime cr(file, cfg, pool, budget, 256, kv_format::f16);
            cr.set_context(total);
            const int c = cr.create_sequence();
            const int g = rt->create_sequence();
            std::vector<std::vector<float>> a, b;
            std::vector<float> la, lb;
            for (long step = 0; step <= 4; ++step)
            {
                if (step > 0)
                {
                    batch.tokens = { text[static_cast<size_t>(n + step - 1)] };
                    batch.positions = { next, next, next };
                    batch.embedding_rows.clear();
                    batch.embeddings.clear();
                    batch.logit_rows = { 0 };
                    seg.count = 1;
                    ++next;
                }
                seg.sequence = c;
                batch.segments = { seg };
                if (!cr.reserve_sequence(c, cr.sequence_length(c) + seg.count)
                    || !rt->reserve_sequence(g, rt->sequence_length(g) + seg.count))
                    throw std::runtime_error("no room for the keys and values");
                cr.forward(batch, la);
                batch.segments[0].sequence = g;
                rt->forward(batch, lb);
                a.push_back(la);
                b.push_back(lb);
            }
            compare("image rows, multimodal positions, DeepStack", a, b, vocab, all);
        }

        std::printf("graph overall: same top token %.1f%%, divergence mean %.5f, max %.5f\n", 100.0 * all.top_share(),
            all.mean(), all.max_divergence);
        if (all.top_share() < 0.875 || all.mean() > 0.05 || hidden_cosine < 0.99)
        {
            std::printf("FAILED: the graph departs from the processor runtime\n");
            return 1;
        }
        std::printf("passed\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::printf("FAILED: %s\n", e.what());
        return 1;
    }
}
