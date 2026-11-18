// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "mamba3.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// MSVC names no macro for FMA, which /arch:AVX2 enables with AVX2.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#define ALBIREO_MAMBA3_AVX2 1
#include <immintrin.h>
#endif

namespace albireo
{

    namespace
    {
        const double pi = 3.14159265358979323846;
        const double two_pi = 6.28318530717958647692;

        /*
            The scalars of a head (decay, step, trapezoid weight, angles) are computed in double
            precision: a step of several units multiplies the rounding of single precision into
            angles off by 1e-5, and they cost a few operations per head and per token.
        */
        inline double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
        inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
        // The activation of the decay: 1 + x from 0 on, 1 / (1 - x) below, positive everywhere.
        inline double heavy_tail(double x) { return x >= 0.0 ? 1.0 + x : 1.0 / (1.0 - x); }

        // y = x / sqrt(mean(x^2) + eps) * weight
        void rms_norm(const float* x, const float* weight, float eps, long n, float* y)
        {
            const float scale = 1.0f / std::sqrt(qk_dot_f32(x, x, n) / static_cast<float>(n) + eps);
            for (long i = 0; i < n; ++i) y[i] = x[i] * scale * weight[i];
        }

#if ALBIREO_MAMBA3_AVX2
        inline float sum8(__m256 v)
        {
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            return _mm_cvtss_f32(s);
        }
#endif

        // s = alpha s + c1 k1 + c2 k2
        void update_first(float* s, float alpha, float c1, const float* k1, float c2, const float* k2, long n)
        {
            long i = 0;
#if ALBIREO_MAMBA3_AVX2
            const __m256 va = _mm256_set1_ps(alpha), v1 = _mm256_set1_ps(c1), v2 = _mm256_set1_ps(c2);
            for (; i + 8 <= n; i += 8)
            {
                __m256 v = _mm256_mul_ps(va, _mm256_loadu_ps(s + i));
                v = _mm256_fmadd_ps(v1, _mm256_loadu_ps(k1 + i), v);
                v = _mm256_fmadd_ps(v2, _mm256_loadu_ps(k2 + i), v);
                _mm256_storeu_ps(s + i, v);
            }
#endif
            for (; i < n; ++i) s[i] = alpha * s[i] + c1 * k1[i] + c2 * k2[i];
        }

        // s += c1 k1 + c2 k2
        void update_more(float* s, float c1, const float* k1, float c2, const float* k2, long n)
        {
            long i = 0;
#if ALBIREO_MAMBA3_AVX2
            const __m256 v1 = _mm256_set1_ps(c1), v2 = _mm256_set1_ps(c2);
            for (; i + 8 <= n; i += 8)
            {
                __m256 v = _mm256_fmadd_ps(v1, _mm256_loadu_ps(k1 + i), _mm256_loadu_ps(s + i));
                v = _mm256_fmadd_ps(v2, _mm256_loadu_ps(k2 + i), v);
                _mm256_storeu_ps(s + i, v);
            }
#endif
            for (; i < n; ++i) s[i] += c1 * k1[i] + c2 * k2[i];
        }

        // s = alpha s + c1 k1 + c2 k2, and returns s . q: the state of a channel and its output in one reading.
        float update_and_read(float* s, float alpha, float c1, const float* k1, float c2, const float* k2,
            const float* q, long n)
        {
            long i = 0;
            float sum = 0.0f;
#if ALBIREO_MAMBA3_AVX2
            const __m256 va = _mm256_set1_ps(alpha), v1 = _mm256_set1_ps(c1), v2 = _mm256_set1_ps(c2);
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= n; i += 8)
            {
                __m256 v = _mm256_mul_ps(va, _mm256_loadu_ps(s + i));
                v = _mm256_fmadd_ps(v1, _mm256_loadu_ps(k1 + i), v);
                v = _mm256_fmadd_ps(v2, _mm256_loadu_ps(k2 + i), v);
                _mm256_storeu_ps(s + i, v);
                acc = _mm256_fmadd_ps(v, _mm256_loadu_ps(q + i), acc);
            }
            sum = sum8(acc);
#endif
            for (; i < n; ++i)
            {
                const float v = alpha * s[i] + c1 * k1[i] + c2 * k2[i];
                s[i] = v;
                sum += v * q[i];
            }
            return sum;
        }

        void expect(const std::vector<float>& v, long n, const char* name)
        {
            if (static_cast<long>(v.size()) != n)
                throw std::runtime_error(std::string("mamba3: ") + name + " holds " + std::to_string(v.size())
                    + " values, " + std::to_string(n) + " expected");
        }

        // The values a parameter keeps head by head then rank by rank, rank by rank then head by head.
        template <typename out_vector>
        void by_rank(const std::vector<float>& in, long heads, long rank, long width, out_vector& out)
        {
            out.resize(in.size());
            // A parameter the layer does not have.
            if (in.empty()) return;
            for (long h = 0; h < heads; ++h)
                for (long r = 0; r < rank; ++r)
                    std::memcpy(out.data() + (r * heads + h) * width, in.data() + (h * rank + r) * width,
                        sizeof(float) * static_cast<size_t>(width));
        }
    }

    void check_mamba3(const mamba3_geometry& g)
    {
        const auto fail = [](const std::string& why) { throw std::runtime_error("mamba3: " + why); };
        // The sizes one by one first: their products below then fit 64 bits.
        if (g.d_inner <= 0 || g.head_dim <= 0 || g.n_heads <= 0 || g.d_inner > mamba3_max_channels
            || g.head_dim > g.d_inner || g.n_heads > g.d_inner
            || static_cast<long long>(g.n_heads) * g.head_dim != g.d_inner)
            fail("the channels are not heads of head_dim channels, or are beyond those of a layer");
        if (g.d_state < 2 || g.d_state > mamba3_max_state) fail("d_state is below 2 or beyond the state of a layer");
        if (g.groups < 1 || g.n_heads % g.groups != 0) fail("the heads are not a multiple of the groups of B and C");
        if (g.rank < 1 || g.rank > mamba3_max_rank || (!g.mimo && g.rank != 1))
            fail("the rank is not 1 without MIMO, is below 1 or beyond the ranks of a layer");
        // What a state and a row hold, within the floats a layer may keep: the sizes then fit a long of 32 bits.
        const long long state = static_cast<long long>(g.d_inner) * g.d_state;
        const long long keys = static_cast<long long>(g.rank) * g.n_heads * g.d_state;
        const long long bc = static_cast<long long>(g.d_state) * g.groups * g.rank;
        if (state > mamba3_max_floats || keys > mamba3_max_floats || bc > mamba3_max_floats)
            fail("the state or the row of a token is beyond the floats of a layer");
        if (g.n_angles < 1) fail("the layer rotates no pair");
        if (g.pairwise ? 2 * g.n_angles > g.d_state : (g.d_state % 2 != 0 || g.n_angles > g.d_state / 2))
            fail("d_state does not hold the rotated pairs");
        if (!(g.a_floor > 0.0f) || !(g.norm_eps > 0.0f)) fail("a_floor and norm_eps must be positive");
    }

    mamba3_geometry mamba3_layer(long d_model, double expand, long head_dim, long d_state, long groups,
        double rope_fraction, bool mimo, long rank, bool norm_before_gate)
    {
        if (d_model <= 0 || !(expand > 0.0) || head_dim <= 0)
            throw std::runtime_error("mamba3: d_model, expand and head_dim must be positive");
        if (rope_fraction != 0.5 && rope_fraction != 1.0)
            throw std::runtime_error("mamba3: rope_fraction is neither 0.5 nor 1");
        // The channels as a double first: a product beyond a long must not reach the conversion.
        const double inner = expand * static_cast<double>(d_model);
        if (!(inner >= 1.0 && inner <= static_cast<double>(mamba3_max_channels)))
            throw std::runtime_error("mamba3: expand * d_model is below 1 or beyond the channels of a layer");
        mamba3_geometry g;
        g.d_inner = static_cast<long>(inner);
        g.head_dim = head_dim;
        if (g.d_inner % head_dim != 0)
            throw std::runtime_error("mamba3: the channels are not heads of head_dim channels");
        g.n_heads = g.d_inner / head_dim;
        g.d_state = d_state;
        g.groups = groups;
        g.mimo = mimo;
        g.rank = mimo ? rank : 1;
        g.pairwise = !mimo;
        g.norm_before_gate = norm_before_gate;
        long rotated = static_cast<long>(static_cast<double>(d_state) * rope_fraction);
        rotated -= rotated % 2;
        g.n_angles = rotated / 2;
        check_mamba3(g);
        return g;
    }

    void mamba3_rotate(float* row, const float* cos, const float* sin, long d_state, long n_angles, bool pairwise)
    {
        if (pairwise)
        {
            for (long i = 0; i < n_angles; ++i)
            {
                const float a = row[2 * i], b = row[2 * i + 1];
                row[2 * i] = a * cos[i] - b * sin[i];
                row[2 * i + 1] = a * sin[i] + b * cos[i];
            }
        }
        else
        {
            float* second = row + d_state / 2;
            for (long i = 0; i < n_angles; ++i)
            {
                const float a = row[i], b = second[i];
                row[i] = a * cos[i] - b * sin[i];
                second[i] = a * sin[i] + b * cos[i];
            }
        }
    }

    mamba3_mixer::mamba3_mixer(const mamba3_geometry& g, const mamba3_parameters& p) : g_(g)
    {
        check_mamba3(g_);
        const long H = g_.n_heads, R = g_.rank, N = g_.d_state, P = g_.head_dim;
        expect(p.dt_bias, H, "dt_bias");
        expect(p.b_bias, H * R * N, "b_bias");
        expect(p.c_bias, H * R * N, "c_bias");
        expect(p.b_norm, N, "b_norm");
        expect(p.c_norm, N, "c_norm");
        expect(p.d, H, "d");
        expect(p.mimo_x, g_.mimo ? H * R * P : 0, "mimo_x");
        expect(p.mimo_z, g_.mimo ? H * R * P : 0, "mimo_z");
        expect(p.mimo_o, g_.mimo ? H * R * P : 0, "mimo_o");
        expect(p.out_norm, g_.norm_before_gate ? g_.d_inner : 0, "out_norm");
        dt_bias_ = p.dt_bias;
        b_norm_ = p.b_norm;
        c_norm_ = p.c_norm;
        d_ = p.d;
        out_norm_ = p.out_norm;
        by_rank(p.b_bias, H, R, N, b_bias_);
        by_rank(p.c_bias, H, R, N, c_bias_);
        by_rank(p.mimo_x, H, R, P, mimo_x_);
        by_rank(p.mimo_z, H, R, P, mimo_z_);
        by_rank(p.mimo_o, H, R, P, mimo_o_);
    }

    mamba3_state mamba3_mixer::initial_state() const
    {
        mamba3_state s;
        s.angle.assign(static_cast<size_t>(g_.n_heads * g_.n_angles), 0.0f);
        s.ssm.assign(static_cast<size_t>(g_.d_inner * g_.d_state), 0.0f);
        s.k.assign(static_cast<size_t>(g_.rank * g_.n_heads * g_.d_state), 0.0f);
        s.v.assign(static_cast<size_t>(g_.d_inner), 0.0f);
        return s;
    }

    void mamba3_mixer::forward(mamba3_state& s, const float* projections, long count, float* outputs,
        mamba3_work& w, long head_begin, long head_end) const
    {
        const long H = g_.n_heads, P = g_.head_dim, N = g_.d_state, G = g_.groups, R = g_.rank, S = g_.n_angles;
        const long bc = g_.bc_width();
        const long per_group = H / G;
        const long width = g_.projection_width();
        if (head_begin < 0 || head_end > H)
            throw std::runtime_error("mamba3: the range of heads is not one of this layer");
        if (static_cast<long>(s.angle.size()) != H * S || static_cast<long>(s.ssm.size()) != g_.d_inner * N
            || static_cast<long>(s.k.size()) != R * H * N || static_cast<long>(s.v.size()) != g_.d_inner)
            throw std::runtime_error("mamba3: the state is not one of this layer");
        if (head_begin >= head_end) return;
        w.b.resize(static_cast<size_t>(bc));
        w.c.resize(static_cast<size_t>(bc));
        w.k.resize(static_cast<size_t>(R * N));
        w.q.resize(static_cast<size_t>(R * N));
        w.turn.resize(static_cast<size_t>(S));
        w.cos.resize(static_cast<size_t>(S));
        w.sin.resize(static_cast<size_t>(S));
        w.y.resize(static_cast<size_t>(R * P));
        w.gate.resize(static_cast<size_t>(R * P));
        const size_t row_bytes = sizeof(float) * static_cast<size_t>(N);
        const size_t head_bytes = sizeof(float) * static_cast<size_t>(P);

        for (long t = 0; t < count; ++t)
        {
            const float* z = projections + t * width;
            const float* x = z + g_.d_inner;
            const float* b_row = x + g_.d_inner;
            const float* c_row = b_row + bc;
            const float* dt_row = c_row + bc;
            const float* a_row = dt_row + H;
            const float* trap_row = a_row + H;
            const float* angle_row = trap_row + H;
            float* out = outputs + t * g_.d_inner;

            // The B and C of the groups of these heads, normalized.
            for (long r = 0; r < R; ++r)
            {
                for (long g = head_begin / per_group; g <= (head_end - 1) / per_group; ++g)
                {
                    const long at = (r * G + g) * N;
                    rms_norm(b_row + at, b_norm_.data(), g_.norm_eps, N, w.b.data() + at);
                    rms_norm(c_row + at, c_norm_.data(), g_.norm_eps, N, w.c.data() + at);
                }
            }
            for (long i = 0; i < S; ++i)
                w.turn[static_cast<size_t>(i)] = std::tanh(static_cast<double>(angle_row[i])) * pi;

            for (long h = head_begin; h < head_end; ++h)
            {
                const long g = h / per_group;
                double decay = -heavy_tail(a_row[h]);
                if (decay > -static_cast<double>(g_.a_floor)) decay = -static_cast<double>(g_.a_floor);
                const double step = softplus(static_cast<double>(dt_row[h]) + dt_bias_[static_cast<size_t>(h)]);
                const double weight = sigmoid(trap_row[h]);
                const double kept = std::exp(decay * step);
                const float alpha = static_cast<float>(kept);
                const float beta = static_cast<float>((1.0 - weight) * step * kept);
                const float gamma = static_cast<float>(weight * step);

                /*
                    The angles of the head, kept within one turn, as the kernels of the reference keep
                    them.  The cosines and the sines come from the angle as the state holds it: a
                    sequence that takes back a copy of its state computes the same numbers.
                */
                float* angle = s.angle.data() + h * S;
                for (long i = 0; i < S; ++i)
                {
                    double sum = angle[i] + w.turn[static_cast<size_t>(i)] * step;
                    sum -= two_pi * std::floor(sum / two_pi);
                    float v = static_cast<float>(sum);
                    // An angle just below a whole turn rounds to it.
                    if (v >= static_cast<float>(two_pi)) v = 0.0f;
                    angle[i] = v;
                    w.cos[static_cast<size_t>(i)] = std::cos(v);
                    w.sin[static_cast<size_t>(i)] = std::sin(v);
                }

                // The B and C of the head with their biases, rotated.
                for (long r = 0; r < R; ++r)
                {
                    float* k = w.k.data() + r * N;
                    float* q = w.q.data() + r * N;
                    const float* bn = w.b.data() + (r * G + g) * N;
                    const float* cn = w.c.data() + (r * G + g) * N;
                    const float* bias_k = b_bias_.data() + (r * H + h) * N;
                    const float* bias_q = c_bias_.data() + (r * H + h) * N;
                    for (long n = 0; n < N; ++n)
                    {
                        k[n] = bn[n] + bias_k[n];
                        q[n] = cn[n] + bias_q[n];
                    }
                    mamba3_rotate(k, w.cos.data(), w.sin.data(), N, S, g_.pairwise);
                    mamba3_rotate(q, w.cos.data(), w.sin.data(), N, S, g_.pairwise);
                }

                const float* xh = x + h * P;
                const float* zh = z + h * P;
                const float* x_before = s.v.data() + h * P;
                float* state = s.ssm.data() + h * P * N;
                const float d = d_[static_cast<size_t>(h)];
                if (!g_.mimo)
                {
                    const float* k_before = s.k.data() + h * N;
                    for (long p = 0; p < P; ++p)
                        w.y[static_cast<size_t>(p)] = update_and_read(state + p * N, alpha, beta * x_before[p],
                            k_before, gamma * xh[p], w.k.data(), w.q.data(), N) + d * xh[p];
                }
                else
                {
                    for (long p = 0; p < P; ++p)
                    {
                        float* channel = state + p * N;
                        for (long r = 0; r < R; ++r)
                        {
                            const float mx = mimo_x_[static_cast<size_t>((r * H + h) * P + p)];
                            const float* k_before = s.k.data() + (r * H + h) * N;
                            if (r == 0)
                                update_first(channel, alpha, beta * x_before[p] * mx, k_before, gamma * xh[p] * mx,
                                    w.k.data(), N);
                            else
                                update_more(channel, beta * x_before[p] * mx, k_before, gamma * xh[p] * mx,
                                    w.k.data() + r * N, N);
                        }
                        for (long r = 0; r < R; ++r)
                            w.y[static_cast<size_t>(r * P + p)] = qk_dot_f32(channel, w.q.data() + r * N, N)
                                + d * xh[p] * mimo_x_[static_cast<size_t>((r * H + h) * P + p)];
                    }
                }

                // The normalization of the outputs of the head, when asked, then the gate.
                for (long r = 0; r < R; ++r)
                {
                    float* y = w.y.data() + r * P;
                    float* gate = w.gate.data() + r * P;
                    if (g_.norm_before_gate)
                    {
                        const float scale = 1.0f / std::sqrt(qk_dot_f32(y, y, P) / static_cast<float>(P) + g_.norm_eps);
                        const float* weight = out_norm_.data() + h * P;
                        for (long p = 0; p < P; ++p) y[p] *= scale * weight[p];
                    }
                    if (g_.mimo)
                    {
                        const float* mz = mimo_z_.data() + (r * H + h) * P;
                        for (long p = 0; p < P; ++p) gate[p] = zh[p] * mz[p];
                    }
                    else
                        std::memcpy(gate, zh, head_bytes);
                    qk_silu_mul(gate, y, P);
                }
                float* o = out + h * P;
                if (g_.mimo)
                {
                    for (long p = 0; p < P; ++p) o[p] = 0.0f;
                    for (long r = 0; r < R; ++r)
                    {
                        const float* gate = w.gate.data() + r * P;
                        const float* mo = mimo_o_.data() + (r * H + h) * P;
                        for (long p = 0; p < P; ++p) o[p] += gate[p] * mo[p];
                    }
                }
                else
                    std::memcpy(o, w.gate.data(), head_bytes);

                // What the next token reads of this one.
                for (long r = 0; r < R; ++r)
                    std::memcpy(s.k.data() + (r * H + h) * N, w.k.data() + r * N, row_bytes);
                std::memcpy(s.v.data() + h * P, xh, head_bytes);
            }
        }
    }

}
