// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Checks the sequence mixers the engine holds beside its attention: the recurrence of a
    Mamba-3 layer (engine/mamba3.h), the projections of Tensor Product Attention and the
    combination of the two maps of a differential attention (engine/differential_tpa.h).

        test_mixers mixers.golden

    mixers.golden (tests/data) holds small cases with their results, computed in double
    precision by tools/make_mixer_golden.py, an implementation of the reference functions of
    the authors of each mechanism.  The program compares the engine with them, then checks
    what the runtime of a model relies on: a layer gives the same numbers however its tokens
    and its heads are cut among the calls, and a sequence that takes back a copy of its state
    reads the tokens that follow as it did the first time.  It returns 0 when every check
    passes, and when the processor lacks the instructions it was built for, which it tells.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../engine/differential_tpa.h"
#include "../engine/mamba3.h"

using namespace albireo;

namespace
{
    int failures = 0;

    void check(bool ok, const std::string& what)
    {
        if (!ok)
        {
            ++failures;
            std::printf("FAILED: %s\n", what.c_str());
        }
    }

    struct array
    {
        std::vector<long> dims;
        std::vector<float> data;
        long at(size_t i) const { return static_cast<long>(std::lround(data[i])); }
    };

    std::map<std::string, array> golden;

    bool load(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        char magic[8];
        if (!in.read(magic, 8) || std::memcmp(magic, "ALBGOLD1", 8) != 0) return false;
        uint32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), 4);
        for (uint32_t i = 0; i < count && in; ++i)
        {
            uint32_t length = 0, dims = 0;
            in.read(reinterpret_cast<char*>(&length), 4);
            if (!in || length > 256) return false;
            std::string name(length, ' ');
            in.read(&name[0], length);
            in.read(reinterpret_cast<char*>(&dims), 4);
            if (!in || dims > 8) return false;
            array a;
            size_t total = 1;
            for (uint32_t d = 0; d < dims; ++d)
            {
                uint32_t n = 0;
                in.read(reinterpret_cast<char*>(&n), 4);
                a.dims.push_back(static_cast<long>(n));
                total *= n;
            }
            if (!in || total > (1u << 24)) return false;
            a.data.resize(total);
            in.read(reinterpret_cast<char*>(a.data.data()), static_cast<std::streamsize>(total * sizeof(float)));
            golden[name] = a;
        }
        return static_cast<bool>(in) && golden.size() == count;
    }

    const array& get(const char* name)
    {
        const auto it = golden.find(name);
        if (it == golden.end()) throw std::runtime_error(std::string("the reference file holds no ") + name);
        return it->second;
    }

    // The array of a case, whose names start with the prefix of the case.
    const array& get(const std::string& prefix, const char* name)
    {
        return get((prefix + name).c_str());
    }

    // The larger of two distances; a distance that is no number wins, and stays.
    double worse(double worst, double d)
    {
        return (worst != worst || d <= worst) ? worst : d;
    }

    // The largest distance of two arrays, each value against 1 + its reference.
    double distance(const float* a, const float* b, size_t n)
    {
        double worst = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double reference = b[i];
            worst = worse(worst, std::fabs(a[i] - reference) / (1.0 + std::fabs(reference)));
        }
        return worst;
    }

    double distance(const std::vector<float>& a, const array& b)
    {
        if (a.size() != b.data.size()) return 1e30;
        return distance(a.data(), b.data.data(), a.size());
    }

    // The largest distance of two arrays of angles, around the circle.
    double angle_distance(const std::vector<float>& a, const array& b)
    {
        if (a.size() != b.data.size()) return 1e30;
        const double turn = 6.283185307179586;
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double d = std::fmod(std::fabs(static_cast<double>(a[i]) - b.data[i]), turn);
            worst = worse(worst, d <= turn - d ? d : turn - d);
        }
        return worst;
    }

    bool same(const std::vector<float>& a, const std::vector<float>& b)
    {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    }

    bool same(const mamba3_state& a, const mamba3_state& b)
    {
        return same(a.angle, b.angle) && same(a.ssm, b.ssm) && same(a.k, b.k) && same(a.v, b.v);
    }

    template <typename F>
    bool throws(F f)
    {
        try { f(); }
        catch (const std::exception&) { return true; }
        return false;
    }

    // What the single-precision arithmetic of a layer may leave between it and the reference.
    const double tolerance = 2e-5;

    mamba3_geometry geometry_of(const array& g)
    {
        mamba3_geometry m;
        m.d_inner = g.at(0);
        m.head_dim = g.at(1);
        m.n_heads = m.d_inner / m.head_dim;
        m.d_state = g.at(2);
        m.groups = g.at(3);
        m.rank = g.at(4);
        m.mimo = g.at(5) != 0;
        m.n_angles = g.at(6);
        m.norm_before_gate = g.at(7) != 0;
        m.pairwise = g.at(8) != 0;
        return m;
    }

    void test_mamba3(const std::string& name)
    {
        const std::string p = "m3." + name + ".";
        const mamba3_geometry g = geometry_of(get(p, "geometry"));
        const long length = get(p, "geometry").at(9);
        mamba3_parameters w;
        w.dt_bias = get(p, "dt_bias").data;
        w.b_bias = get(p, "b_bias").data;
        w.c_bias = get(p, "c_bias").data;
        w.b_norm = get(p, "b_norm").data;
        w.c_norm = get(p, "c_norm").data;
        w.d = get(p, "d").data;
        if (g.mimo)
        {
            w.mimo_x = get(p, "mimo_x").data;
            w.mimo_z = get(p, "mimo_z").data;
            w.mimo_o = get(p, "mimo_o").data;
        }
        if (g.norm_before_gate) w.out_norm = get(p, "out_norm").data;
        const mamba3_mixer layer(g, w);
        const array& proj = get(p, "proj");
        check(proj.dims.size() == 2 && proj.dims[0] == length && proj.dims[1] == g.projection_width(),
            name + ": the rows of the reference have the width of the layer");
        if (proj.dims.size() != 2 || proj.dims[1] != g.projection_width()) return;
        const long width = g.projection_width();

        // The whole sequence in one call.
        mamba3_work work;
        mamba3_state whole = layer.initial_state();
        std::vector<float> y(static_cast<size_t>(length * g.d_inner));
        layer.forward(whole, proj.data.data(), length, y.data(), work);
        const double dy = distance(y, get(p, "y"));
        const double ds = worse(distance(whole.ssm, get(p, "ssm")), worse(distance(whole.k, get(p, "k")),
            distance(whole.v, get(p, "v"))));
        const double da = angle_distance(whole.angle, get(p, "angle"));
        check(dy <= tolerance, name + ": outputs against the reference, distance " + std::to_string(dy));
        check(ds <= tolerance, name + ": state against the reference, distance " + std::to_string(ds));
        check(da <= 1e-4, name + ": angles against the reference, distance " + std::to_string(da));

        // The same tokens in two calls, each for two ranges of heads with buffers of their own.
        mamba3_work first_work, second_work;
        mamba3_state cut = layer.initial_state();
        std::vector<float> y_cut(y.size(), -1.0f);
        const long split = length / 3, half = g.n_heads / 2;
        const long counts[2] = { split, length - split };
        long done = 0;
        for (const long count : counts)
        {
            layer.forward(cut, proj.data.data() + done * width, count, y_cut.data() + done * g.d_inner, second_work,
                half, g.n_heads);
            layer.forward(cut, proj.data.data() + done * width, count, y_cut.data() + done * g.d_inner, first_work,
                0, half);
            done += count;
        }
        check(same(y, y_cut) && same(whole, cut),
            name + ": the calls and the ranges of heads leave the numbers as they are");

        // A sequence that goes back to a copy of its state reads the same tokens to the same numbers.
        mamba3_state back = layer.initial_state();
        std::vector<float> y_back(y.size());
        layer.forward(back, proj.data.data(), split, y_back.data(), work);
        const mamba3_state snapshot = back;
        std::vector<float> scratch(static_cast<size_t>((length - split) * g.d_inner));
        layer.forward(back, proj.data.data() + split * width, length - split, scratch.data(), work);
        back = snapshot;
        layer.forward(back, proj.data.data() + split * width, length - split, y_back.data() + split * g.d_inner, work);
        check(same(y, y_back) && same(whole, back),
            name + ": a state taken back gives the same tokens the same numbers");
        check(whole.bytes() == sizeof(float) * static_cast<size_t>(g.n_heads * g.n_angles + g.d_inner * g.d_state
            + g.rank * g.n_heads * g.d_state + g.d_inner), name + ": the size of a state");
        std::printf("  mamba-3 %-14s %ld heads of %ld, state %ld, rank %ld%s%s: outputs %.1e, state %.1e\n",
            name.c_str(), g.n_heads, g.head_dim, g.d_state, g.rank, g.pairwise ? "" : ", halved pairs",
            g.norm_before_gate ? ", normalized" : "", dy, ds);
    }

    void test_geometry()
    {
        const mamba3_geometry a = mamba3_layer(24, 2.0, 8, 16, 1, 0.5, false, 1, false);
        check(a.d_inner == 48 && a.n_heads == 6 && a.n_angles == 4 && a.rank == 1 && a.pairwise && !a.mimo
            && a.projection_width() == 2 * 48 + 2 * 16 + 3 * 6 + 4, "the layer of the reference module without MIMO");
        const mamba3_geometry b = mamba3_layer(20, 2.0, 10, 20, 1, 1.0, true, 2, true);
        check(b.d_inner == 40 && b.n_heads == 4 && b.n_angles == 10 && b.rank == 2 && !b.pairwise && b.mimo
            && b.norm_before_gate && b.projection_width() == 2 * 40 + 2 * 20 * 2 + 3 * 4 + 10,
            "the layer of the reference module with MIMO");
        // An odd number of rotated values loses one: 18 * 0.5 = 9 values, 4 pairs.
        check(mamba3_layer(24, 2.0, 8, 18, 2, 0.5, false, 1, false).n_angles == 4, "an odd rotated width");
        check(mamba3_layer(24, 2.0, 8, 16, 1, 0.5, false, 7, false).rank == 1, "the rank is 1 without MIMO");
        check(throws([] { mamba3_layer(24, 2.0, 7, 16, 1, 0.5, false, 1, false); }),
            "channels that are not whole heads");
        check(throws([] { mamba3_layer(24, 2.0, 8, 16, 1, 0.3, false, 1, false); }), "another share of rotated values");
        check(throws([] { mamba3_layer(24, 2.0, 8, 16, 4, 0.5, false, 1, false); }),
            "groups that do not divide the heads");
        check(throws([] { mamba3_layer(24, 2.0, 8, 1, 1, 1.0, false, 1, false); }), "a state too narrow to rotate");
        check(throws([] { mamba3_layer(24, 2.0, 8, 16, 1, 0.5, true, 0, false); }), "a MIMO layer without rank");
        check(throws([] { mamba3_layer(24, 1e300, 8, 16, 1, 0.5, false, 1, false); }), "channels beyond a layer");
        check(throws([] { mamba3_layer(1 << 20, 2.0, 8, 16, 1, 0.5, false, 1, false); }), "channels beyond a layer");
        check(throws([] { mamba3_layer(24, 2.0, 8, 1 << 20, 1, 0.5, false, 1, false); }), "a state beyond a layer");
        // Each size within its bound, and a state of 2^30 floats.
        mamba3_geometry large = mamba3_layer(24, 2.0, 8, 16, 1, 0.5, false, 1, false);
        large.head_dim = 64;
        large.n_heads = 4096;
        large.d_inner = 262144;
        large.d_state = 4096;
        large.n_angles = 1024;
        check(throws([&] { check_mamba3(large); }), "a state of more floats than a layer holds");
        large.d_state = 1024;
        large.n_angles = 256;
        check(!throws([&] { check_mamba3(large); }), "the largest state of a layer");
        mamba3_geometry odd = a;
        odd.pairwise = false;
        odd.d_state = 17;
        check(throws([&] { check_mamba3(odd); }), "halved pairs of an odd width");
        mamba3_parameters w;
        w.dt_bias.assign(6, 0.0f);
        w.b_bias.assign(6 * 16, 0.0f);
        w.c_bias.assign(6 * 16, 0.0f);
        w.b_norm.assign(16, 1.0f);
        w.c_norm.assign(16, 1.0f);
        w.d.assign(6, 1.0f);
        check(!throws([&] { mamba3_mixer m(a, w); }), "parameters of the sizes of the layer");
        mamba3_parameters wrong = w;
        wrong.dt_bias.push_back(0.0f);
        check(throws([&] { mamba3_mixer m(a, wrong); }), "a parameter of another size");
        wrong = w;
        wrong.mimo_x.assign(6 * 8, 1.0f);
        check(throws([&] { mamba3_mixer m(a, wrong); }), "MIMO weights given to a layer without MIMO");
        wrong = w;
        wrong.out_norm.assign(48, 1.0f);
        check(throws([&] { mamba3_mixer m(a, wrong); }), "output weights given to a layer that does not normalize");
        const mamba3_mixer layer(a, w);
        const mamba3_geometry wide = mamba3_layer(24, 2.0, 8, 32, 1, 0.5, false, 1, false);
        mamba3_parameters wide_w = w;
        wide_w.b_bias.assign(6 * 32, 0.0f);
        wide_w.c_bias.assign(6 * 32, 0.0f);
        wide_w.b_norm.assign(32, 1.0f);
        wide_w.c_norm.assign(32, 1.0f);
        mamba3_state other = mamba3_mixer(wide, wide_w).initial_state(), own = layer.initial_state();
        std::vector<float> row(static_cast<size_t>(a.projection_width()), 0.5f), out(static_cast<size_t>(a.d_inner));
        mamba3_work work;
        check(throws([&] { layer.forward(other, row.data(), 1, out.data(), work); }), "the state of another layer");
        check(throws([&] { layer.forward(own, row.data(), 1, out.data(), work, 0, a.n_heads + 1); }),
            "a range beyond the heads");
        check(!throws([&] { layer.forward(own, row.data(), 1, out.data(), work, 3, 3); }), "an empty range of heads");
        std::printf("  mamba-3 geometry: the layers of the reference module, and the sizes refused\n");
    }

    void test_rotation()
    {
        const array& q = get("rot.q");
        const array& k = get("rot.k");
        const array& angle = get("rot.angle");
        const long R = q.dims[0], H = q.dims[1], N = q.dims[2], S = angle.dims[1];
        std::vector<float> cos(angle.data.size()), sin(angle.data.size());
        for (size_t i = 0; i < angle.data.size(); ++i)
        {
            cos[i] = std::cos(angle.data[i]);
            sin[i] = std::sin(angle.data[i]);
        }
        const char* labels[2] = { "halved", "pairwise" };
        for (int which = 0; which < 2; ++which)
        {
            std::vector<float> qo(q.data), ko(k.data);
            for (long r = 0; r < R; ++r)
            {
                for (long h = 0; h < H; ++h)
                {
                    const float* c = cos.data() + h * S;
                    const float* sn = sin.data() + h * S;
                    mamba3_rotate(qo.data() + (r * H + h) * N, c, sn, N, S, which == 1);
                    mamba3_rotate(ko.data() + (r * H + h) * N, c, sn, N, S, which == 1);
                }
            }
            const std::string p = std::string("rot.") + labels[which] + ".";
            const double d = worse(distance(qo, get(p, "q_out")), distance(ko, get(p, "k_out")));
            check(d <= 1e-5, std::string("the rotation by ") + labels[which] + " pairs, distance " + std::to_string(d));
        }
        std::printf("  mamba-3 rotation: the pairs (2i, 2i + 1) and the pairs (i, i + d_state / 2)\n");
    }

    // y = x W^T, W of rows x cols values.
    std::vector<float> product(const array& w, const float* x, long tokens)
    {
        const long rows = w.dims[0], cols = w.dims[1];
        std::vector<float> y(static_cast<size_t>(tokens * rows));
        for (long t = 0; t < tokens; ++t)
        {
            for (long r = 0; r < rows; ++r)
            {
                double sum = 0.0;
                for (long c = 0; c < cols; ++c)
                    sum += static_cast<double>(w.data[static_cast<size_t>(r * cols + c)]) * x[t * cols + c];
                y[static_cast<size_t>(t * rows + r)] = static_cast<float>(sum);
            }
        }
        return y;
    }

    void test_tpa()
    {
        const array& g = get("tpa.geometry");
        const long heads = g.at(1), head_dim = g.at(2), rank = g.at(3), q_rank = g.at(4), tokens = g.at(5);
        const array& x = get("tpa.x");
        const char* which[3] = { "q", "k", "v" };
        double worst = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            const long ranks = i == 0 ? q_rank : rank;
            const std::vector<float> a = product(get("tpa.W_A_", which[i]), x.data.data(), tokens);
            const std::vector<float> b = product(get("tpa.W_B_", which[i]), x.data.data(), tokens);
            // The rows of each array apart by more than their widths, the floats between them kept.
            const long wa = heads * ranks, wb = ranks * head_dim, wo = heads * head_dim;
            const long sa = wa + 3, sb = wb + 5, so = wo + 2;
            std::vector<float> ap(static_cast<size_t>(tokens * sa), 7.0f), bp(static_cast<size_t>(tokens * sb), 7.0f);
            std::vector<float> op(static_cast<size_t>(tokens * so), 7.0f), rows(static_cast<size_t>(tokens * wo));
            for (long t = 0; t < tokens; ++t)
            {
                std::copy(a.begin() + t * wa, a.begin() + (t + 1) * wa, ap.begin() + t * sa);
                std::copy(b.begin() + t * wb, b.begin() + (t + 1) * wb, bp.begin() + t * sb);
            }
            tpa_combine_rows(ap.data(), sa, bp.data(), sb, tokens, heads, ranks, head_dim, op.data(), so);
            bool kept = true;
            for (long t = 0; t < tokens; ++t)
            {
                std::copy(op.begin() + t * so, op.begin() + t * so + wo, rows.begin() + t * wo);
                const float* between = op.data() + t * so + wo;
                kept = kept && between[0] == 7.0f && between[1] == 7.0f;
            }
            const double d = distance(rows, get("tpa.", which[i]));
            worst = worse(worst, d);
            check(d <= 1e-5,
                std::string("the ") + which[i] + " of tensor product attention, distance " + std::to_string(d));
            check(kept, std::string("the floats between the rows of the ") + which[i]);
        }
        std::printf("  tensor product attention: queries of rank %ld, keys and values of rank %ld: %.1e\n",
            q_rank, rank, worst);
    }

    void test_differential(const std::string& name)
    {
        const std::string p = "diff." + name + ".";
        const array& g = get(p, "geometry");
        const long depth = g.at(1), heads = g.at(2), kv_heads = g.at(3), hd = g.at(4), tokens = g.at(5);
        const bool paired = g.at(6) != 0;
        const long rep = heads / kv_heads, width = 2 * hd;
        const array& x = get(p, "x");
        const std::vector<float> q = product(get(p, "q_proj"), x.data.data(), tokens);     // 2 heads * hd per token
        const std::vector<float> k = product(get(p, "k_proj"), x.data.data(), tokens);     // 2 kv_heads * hd
        const std::vector<float> v = product(get(p, "v_proj"), x.data.data(), tokens);     // kv_heads * 2 hd

        const float lambda_init = differential_lambda_init(depth);
        check(std::fabs(lambda_init - get(p, "lambda_init").data[0]) <= 1e-6, name + ": lambda_init of the layer");
        const float lambda = differential_lambda(get(p, "lambda_q1").data.data(), get(p, "lambda_k1").data.data(),
            get(p, "lambda_q2").data.data(), get(p, "lambda_k2").data.data(), hd, lambda_init);
        check(std::fabs(lambda - get(p, "lambda").data[0]) <= 1e-5, name + ": lambda of the layer");

        /*
            The maps of an attention with twice the heads.  Map m reads the values (m / 2) / rep, and the
            keys m / rep, or the keys 2 * ((m / 2) / rep) + m % 2 when the case pairs the two maps of a head
            with the two key sub-heads of its group.
        */
        std::vector<float> y(static_cast<size_t>(tokens * heads * width)), y_in_place;
        std::vector<float> maps(static_cast<size_t>(2 * heads * width));
        std::vector<double> scores(static_cast<size_t>(tokens));
        const double scaling = 1.0 / std::sqrt(static_cast<double>(hd));
        for (long t = 0; t < tokens; ++t)
        {
            for (long m = 0; m < 2 * heads; ++m)
            {
                const float* qm = q.data() + (t * 2 * heads + m) * hd;
                const long key = paired ? 2 * ((m / 2) / rep) + m % 2 : m / rep;
                double top = -1e300;
                for (long s = 0; s <= t; ++s)
                {
                    const float* ks = k.data() + (s * 2 * kv_heads + key) * hd;
                    double dot = 0.0;
                    for (long i = 0; i < hd; ++i) dot += static_cast<double>(qm[i]) * ks[i];
                    scores[static_cast<size_t>(s)] = dot * scaling;
                    top = std::max(top, scores[static_cast<size_t>(s)]);
                }
                double sum = 0.0;
                for (long s = 0; s <= t; ++s)
                {
                    scores[static_cast<size_t>(s)] = std::exp(scores[static_cast<size_t>(s)] - top);
                    sum += scores[static_cast<size_t>(s)];
                }
                float* out = maps.data() + m * width;
                for (long i = 0; i < width; ++i)
                {
                    double value = 0.0;
                    for (long s = 0; s <= t; ++s)
                    {
                        const size_t at = static_cast<size_t>((s * kv_heads + (m / 2) / rep) * width + i);
                        value += scores[static_cast<size_t>(s)] / sum * v[at];
                    }
                    out[i] = static_cast<float>(value);
                }
            }
            differential_combine(maps.data(), heads, width, lambda, get(p, "subln").data.data(), 1e-5f,
                1.0f - lambda_init, y.data() + t * heads * width);
            // The outputs may take the place of the maps.
            differential_combine(maps.data(), heads, width, lambda, get(p, "subln").data.data(), 1e-5f,
                1.0f - lambda_init, maps.data());
            y_in_place.insert(y_in_place.end(), maps.begin(), maps.begin() + heads * width);
        }
        const double d = distance(y, get(p, "y"));
        check(d <= 2e-5, name + ": differential attention against the reference, distance " + std::to_string(d));
        check(same(y, y_in_place), name + ": the outputs written over the maps");
        std::printf("  differential attention %-5s %ld heads of 2 x %ld, %ld key heads%s, lambda %.4f: %.1e\n",
            name.c_str(), heads, hd, 2 * kv_heads, paired ? " paired with the maps" : "", lambda, d);
    }

    void test_head_norm()
    {
        const long heads = 3, width = 10;
        std::vector<float> x(static_cast<size_t>(heads * width)), weight(static_cast<size_t>(width));
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = std::sin(0.7f * static_cast<float>(i)) * (1.0f + static_cast<float>(i % 7));
        // The values of the second head are small against eps, which then decides its scale.
        for (long i = 0; i < width; ++i) x[static_cast<size_t>(width + i)] *= 1e-3f;
        for (size_t i = 0; i < weight.size(); ++i) weight[i] = 0.5f + 0.1f * static_cast<float>(i);
        std::vector<float> scaled = x, plain = x;
        head_rms_norm(scaled.data(), heads, width, weight.data(), 1e-5f);
        head_rms_norm(plain.data(), heads, width, nullptr, 1e-5f);
        double worst = 0.0;
        for (long h = 0; h < heads; ++h)
        {
            double mean = 0.0;
            for (long i = 0; i < width; ++i)
            {
                const double value = x[static_cast<size_t>(h * width + i)];
                mean += value * value;
            }
            const double scale = 1.0 / std::sqrt(mean / width + 1e-5);
            for (long i = 0; i < width; ++i)
            {
                const size_t at = static_cast<size_t>(h * width + i);
                const double without = std::fabs(plain[at] - x[at] * scale);
                const double with = std::fabs(scaled[at] - x[at] * scale * weight[static_cast<size_t>(i)]);
                worst = worse(worse(worst, without), with);
            }
        }
        check(worst <= 1e-5, "the normalization of the outputs of each head, distance " + std::to_string(worst));
    }

    // The time of a token in a layer of the size of a language model, which the products around it dwarf.
    void measure()
    {
        const mamba3_geometry g = mamba3_layer(2048, 2.0, 64, 128, 1, 0.5, false, 1, false);
        mamba3_parameters w;
        w.dt_bias.assign(static_cast<size_t>(g.n_heads), -2.0f);
        w.b_bias.assign(static_cast<size_t>(g.n_heads * g.d_state), 1.0f);
        w.c_bias.assign(static_cast<size_t>(g.n_heads * g.d_state), 1.0f);
        w.b_norm.assign(static_cast<size_t>(g.d_state), 1.0f);
        w.c_norm.assign(static_cast<size_t>(g.d_state), 1.0f);
        w.d.assign(static_cast<size_t>(g.n_heads), 1.0f);
        const mamba3_mixer layer(g, w);
        const long tokens = 64;
        std::vector<float> proj(static_cast<size_t>(tokens * g.projection_width()));
        std::vector<float> y(static_cast<size_t>(tokens * g.d_inner));
        for (size_t i = 0; i < proj.size(); ++i) proj[i] = std::sin(0.37f * static_cast<float>(i % 1009));
        mamba3_state s = layer.initial_state();
        mamba3_work work;
        const auto start = std::chrono::steady_clock::now();
        layer.forward(s, proj.data(), tokens, y.data(), work);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        bool finite = true;
        for (const float v : y) finite = finite && std::isfinite(v);
        check(finite, "a layer of 4096 channels gives finite numbers");
        std::printf("  mamba-3, %ld channels in %ld heads, state %ld: %.0f microseconds per token on one thread, "
            "%.1f MB of state per sequence\n", g.d_inner, g.n_heads, g.d_state, 1e6 * seconds / tokens,
            static_cast<double>(s.bytes()) / (1024.0 * 1024.0));
    }
}

int main(int argc, char** argv)
{
    const unsigned lacks = qk_missing_isa(read_cpu_info());
    if (lacks)
    {
        std::printf("albireo mixers (%s): skipped, the processor lacks %s\n", qk_isa(), qk_isa_text(lacks));
        return 0;
    }
    std::printf("albireo mixers (%s)\n", qk_isa());
    if (argc < 2 || !load(argv[1]))
    {
        std::printf("usage: test_mixers mixers.golden (tests/data)\n");
        return 2;
    }
    try
    {
        test_geometry();
        const char* cases[] = { "siso_gate", "siso_norm", "siso_groups", "siso_wide", "mimo_gate", "mimo_norm",
            "mimo_pairwise", "siso_scaled", "mimo_scaled" };
        for (const char* name : cases) test_mamba3(name);
        test_rotation();
        test_tpa();
        test_differential("mha");
        test_differential("gqa");
        test_differential("flash");
        test_head_norm();
        measure();
    }
    catch (const std::exception& e)
    {
        std::printf("FAILED: %s\n", e.what());
        ++failures;
    }
    if (failures)
    {
        std::printf("%d checks FAILED\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
