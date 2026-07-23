// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Low-rank adaptation of a frozen projection, shared by LoRA and DoRA.
//
// A fine-tuning method of this family replaces a frozen weight W of shape (in, out) by
//
//     LoRA   W' = W + s.A.B
//     DoRA   W' = m * (W + s.A.B) / ||W + s.A.B||       (norm taken column by column)
//
// with A of shape (in, rank), B of shape (rank, out), m a vector of out magnitudes and
// s = alpha / rank. Both are therefore the same object seen twice: DoRA is LoRA followed
// by a per-column rescale. That is why one core covers them, and why a third variant of
// the family costs a branch rather than an implementation.
//
// Nothing here materializes W'. The distributive law gives the output directly,
//
//     y = (x.W + s.(x.A).B) * c        with c_j = m_j / ||v_j||, v_j the j-th column of W'
//
// and the column norms follow from an identity that never forms v_j:
//
//     ||v_j||^2 = ||w_j||^2 + 2s.<w_j, (AB)_j> + s^2.||(AB)_j||^2
//               = ||w_j||^2 + 2s.(W'A)[j,:].B[:,j] + s^2.B[:,j]'.(A'A).B[:,j]
//
// where ||w_j||^2 is computed once, W being frozen, A'A is a rank x rank matrix and W'A
// costs a fraction of the projection it adapts. The same identity differentiates without
// materialization, so this implementation carries the exact gradient of the magnitude
// path rather than the detached approximation the reference implementations settle for.
//
// Both methods are inert at step zero: B starts at zero, and m starts at the column
// norms of W, so c is exactly one. A freshly adapted model reproduces its base bit for
// bit, which is what makes an adapted run comparable to the run it started from.

#ifndef DLIB_DNN_LORA_ADAPTER_H_
#define DLIB_DNN_LORA_ADAPTER_H_

#include "lora_adapter_abstract.h"

#include <cmath>
#include <string>
#include <vector>

#include "../cuda/tensor_tools.h"
#include "../rand.h"
#include "../serialize.h"

namespace dlib
{
    enum class adapter_method { none = 0, lora = 1, dora = 2 };

    inline std::string adapter_method_name(adapter_method m)
    {
        switch (m)
        {
        case adapter_method::lora: return "lora";
        case adapter_method::dora: return "dora";
        default:                   return "none";
        }
    }

    inline adapter_method adapter_method_from_name(const std::string& s)
    {
        if (s == "lora") return adapter_method::lora;
        if (s == "dora") return adapter_method::dora;
        return adapter_method::none;
    }

    /* Where the adapter blocks sit in a packed parameter blob and how many floats they
       take. The layout is [A | B | m], m being absent outside DoRA, so switching methods
       changes the size of the blob and is a configuration-time decision, not a runtime
       one. A rank of zero yields an empty layout and an inactive adapter. */
    struct adapter_geometry
    {
        long in_dim = 0;
        long out_dim = 0;
        long rank = 0;
        adapter_method method = adapter_method::none;

        bool active() const { return rank > 0 && method != adapter_method::none; }
        size_t a_count() const { return active() ? static_cast<size_t>(in_dim) * rank : 0; }
        size_t b_count() const { return active() ? static_cast<size_t>(rank) * out_dim : 0; }
        size_t m_count() const
        {
            return (active() && method == adapter_method::dora)
                ? static_cast<size_t>(out_dim) : 0;
        }
        size_t total() const { return a_count() + b_count() + m_count(); }

        size_t a_offset(size_t base) const { return base; }
        size_t b_offset(size_t base) const { return base + a_count(); }
        size_t m_offset(size_t base) const { return base + a_count() + b_count(); }
    };

    /* Adapter of one projection. Holds the geometry, the scale and the scratch buffers of
       the low-rank passes; the parameters themselves live in the host layer's packed blob
       and arrive as aliased views, which is what lets the trainer update them like any
       other weight.

       An instance is not thread safe and belongs to one layer, like the rest of the
       layer's scratch state. */
    class low_rank_adapter
    {
    public:
        void configure(long in_dim, long out_dim, long rank, adapter_method method, double alpha)
        {
            DLIB_CASSERT(in_dim > 0 && out_dim > 0, "adapter dimensions must be positive");
            DLIB_CASSERT(rank >= 0, "adapter rank must be non-negative");
            geom_.in_dim = in_dim;
            geom_.out_dim = out_dim;
            geom_.rank = rank;
            geom_.method = method;
            /* alpha / rank rather than alpha: it keeps the effective step of the adapter
               comparable when the rank changes, so a rank sweep does not silently become
               a learning-rate sweep. */
            scale_ = (rank > 0) ? static_cast<float>(alpha / rank) : 0.0f;
            alpha_ = alpha;
            base_sqnorm_.clear();
            if (geom_.active())
            {
                a_alias_ = alias_tensor(in_dim, rank);
                b_alias_ = alias_tensor(rank, out_dim);
                m_alias_ = alias_tensor(1, out_dim);
            }
        }

        const adapter_geometry& geometry() const { return geom_; }
        bool active() const { return geom_.active(); }
        size_t parameter_count() const { return geom_.total(); }
        float scale() const { return scale_; }
        double alpha() const { return alpha_; }

        alias_tensor a_view() const { return a_alias_; }
        alias_tensor b_view() const { return b_alias_; }
        alias_tensor m_view() const { return m_alias_; }

        /* A drawn from N(0, 1/rank), B set to zero, m set to the column norms of the base
           weight. The zero on B is what makes the adapter inert at step zero; putting it
           on A instead would leave the pair with no gradient at all, B being multiplied
           by a zero input. */
        void initialize(const tensor& base_w, tensor& a, tensor& b, tensor& m, dlib::rand& rnd)
        {
            if (!geom_.active()) return;
            DLIB_CASSERT(a.size() == geom_.a_count() && b.size() == geom_.b_count(),
                "adapter parameter views do not match the configured geometry");

            const float sd = 1.0f / std::sqrt(static_cast<float>(geom_.rank));
            float* pa = a.host_write_only();
            for (size_t i = 0; i < a.size(); ++i)
                pa[i] = sd * static_cast<float>(rnd.get_random_gaussian());
            b = 0;

            if (geom_.method == adapter_method::dora)
            {
                DLIB_CASSERT(m.size() == geom_.m_count(),
                    "adapter magnitude view does not match the configured geometry");
                refresh_base_sqnorm(base_w);
                float* pm = m.host_write_only();
                for (long j = 0; j < geom_.out_dim; ++j)
                    pm[j] = std::sqrt(base_sqnorm_[static_cast<size_t>(j)]);
            }
        }

        /* Column norms of the base weight, cached because the base is frozen. Call again
           after merging an adapter into the base, which is the only event that changes
           them. */
        void refresh_base_sqnorm(const tensor& base_w)
        {
            DLIB_CASSERT(base_w.size() == static_cast<size_t>(geom_.in_dim) * geom_.out_dim,
                "base weight does not match the configured geometry");
            base_sqnorm_.assign(static_cast<size_t>(geom_.out_dim), 0.0f);
            const float* w = base_w.host();
            for (long i = 0; i < geom_.in_dim; ++i)
            {
                const float* row = w + static_cast<size_t>(i) * geom_.out_dim;
                for (long j = 0; j < geom_.out_dim; ++j)
                    base_sqnorm_[static_cast<size_t>(j)] += row[j] * row[j];
            }
        }

        /* Adds the adapter contribution to a projection output.

           y arrives holding x.W and leaves holding the adapted output. x is (rows, in_dim)
           and y is (rows, out_dim), both as plain matrix views. The intermediate x.A and,
           for DoRA, the pre-rescale output are kept for the backward pass, so a forward
           must precede the backward of the same batch. */
        void forward(const tensor& x, const tensor& base_w,
            const tensor& a, const tensor& b, const tensor& m, tensor& y)
        {
            if (!geom_.active()) return;
            const long rows = static_cast<long>(x.num_samples());

            xa_.set_size(rows, geom_.rank);
            tt::gemm(0.0f, xa_, 1.0f, x, false, a, false);
            tt::gemm(1.0f, y, scale_, xa_, false, b, false);

            if (geom_.method != adapter_method::dora) return;

            compute_column_scales(base_w, a, b, m);
            pre_scale_.copy_size(y);
            memcpy(pre_scale_, y);
            tt::scale_columns(y, pre_scale_, col_scale_);
        }

        /* Gradients of the adapter parameters and the input gradient of the whole
           adapted projection, base term included.

           The base term belongs here rather than to the caller because it is not
           dy . W' with the frozen weight: DoRA rescales the entire projection output,
           base part and low-rank part alike, so the gradient that reaches W carries the
           column factors, which live inside the adapter. A caller applying dy . W' on
           its own would be wrong by exactly the deviation of those factors from one, a
           quiet error that trains a slightly wrong model rather than failing.

           Every output is accumulated into, never overwritten. */
        void backward(const tensor& x, const tensor& base_w,
            const tensor& a, const tensor& b, const tensor& m,
            const tensor& dy, tensor& dx, tensor& da, tensor& db, tensor& dm)
        {
            if (!geom_.active()) return;
            const long rows = static_cast<long>(x.num_samples());

            const tensor* du = &dy;
            if (geom_.method == adapter_method::dora)
            {
                /* Rescale the incoming gradient by the column factors, then differentiate
                   the factors themselves. dL/dm follows directly; dL/d||v|| feeds back
                   into A and B through the identity that gave the norms, which keeps the
                   whole path exact without ever forming the merged weight. */
                scaled_dy_.copy_size(dy);
                tt::scale_columns(scaled_dy_, dy, col_scale_);
                du = &scaled_dy_;

                prod_.copy_size(dy);
                tt::multiply(false, prod_, dy, pre_scale_);
                col_grad_.set_size(1, geom_.out_dim);
                tt::assign_conv_bias_gradient(col_grad_, prod_);

                const float* t = col_grad_.host();
                const float* pm = m.host();
                float* pdm = dm.host();
                k_.assign(static_cast<size_t>(geom_.out_dim), 0.0f);
                for (long j = 0; j < geom_.out_dim; ++j)
                {
                    const size_t u = static_cast<size_t>(j);
                    const float n = col_norm_[u];
                    pdm[j] += t[j] / n;
                    k_[u] = -pm[j] * t[j] / (n * n * n);
                }
                accumulate_magnitude_path(base_w, a, b, da, db);
            }

            // dB = s.(x.A)' . du     and     dA = s.x' . (du.B')
            tt::gemm(1.0f, db, scale_, xa_, true, *du, false);
            dub_.set_size(rows, geom_.rank);
            tt::gemm(0.0f, dub_, 1.0f, *du, false, b, true);
            tt::gemm(1.0f, da, scale_, x, true, dub_, false);

            // dx = du . W' + s.(du.B') . A'
            tt::gemm(1.0f, dx, 1.0f, *du, false, base_w, true);
            tt::gemm(1.0f, dx, scale_, dub_, false, a, true);
        }

        /* Folds the adapter into the base weight and returns it to its inert state.

           Sequential fine-tuning needs this between stages: resuming a second stage on
           saved adapters trains the adapters alone and leaves the base exactly where the
           first stage found it, so the knowledge of that stage is never carried forward.
           Merging is also how an adapted model is exported for inference at no cost. */
        void merge_into_base(const tensor& a, const tensor& b, const tensor& m, tensor& base_w)
        {
            if (!geom_.active()) return;
            const long in = geom_.in_dim, out = geom_.out_dim, r = geom_.rank;

            const float* pa = a.host();
            const float* pb = b.host();
            float* w = base_w.host();

            for (long i = 0; i < in; ++i)
            {
                float* row = w + static_cast<size_t>(i) * out;
                const float* arow = pa + static_cast<size_t>(i) * r;
                for (long j = 0; j < out; ++j)
                {
                    float acc = 0.0f;
                    for (long q = 0; q < r; ++q) acc += arow[q] * pb[static_cast<size_t>(q) * out + j];
                    row[j] += scale_ * acc;
                }
            }

            if (geom_.method == adapter_method::dora)
            {
                refresh_base_sqnorm(base_w);
                const float* pm = m.host();
                for (long i = 0; i < in; ++i)
                {
                    float* row = w + static_cast<size_t>(i) * out;
                    for (long j = 0; j < out; ++j)
                    {
                        const float n = std::sqrt(base_sqnorm_[static_cast<size_t>(j)]);
                        row[j] *= (n > 0.0f) ? (pm[j] / n) : 1.0f;
                    }
                }
            }
            refresh_base_sqnorm(base_w);
        }

        friend void serialize(const low_rank_adapter& item, std::ostream& out)
        {
            serialize("low_rank_adapter_", out);
            serialize(item.geom_.in_dim, out);
            serialize(item.geom_.out_dim, out);
            serialize(item.geom_.rank, out);
            serialize(static_cast<int>(item.geom_.method), out);
            serialize(item.alpha_, out);
        }

        friend void deserialize(low_rank_adapter& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "low_rank_adapter_")
                throw serialization_error(
                    "Unexpected version found while deserializing dlib::low_rank_adapter.");
            long in_dim, out_dim, rank;
            int method;
            double alpha;
            deserialize(in_dim, in);
            deserialize(out_dim, in);
            deserialize(rank, in);
            deserialize(method, in);
            deserialize(alpha, in);
            item.configure(in_dim, out_dim, rank, static_cast<adapter_method>(method), alpha);
        }

    private:

        /* c_j = m_j / ||v_j||, from the identity that avoids forming the merged weight.
           The two matrix products run on the device; the reduction over out x rank and
           the square root run on the host, which costs one readback of that size per
           adapted projection and per forward. */
        void compute_column_scales(const tensor& base_w,
            const tensor& a, const tensor& b, const tensor& m)
        {
            const long out = geom_.out_dim, r = geom_.rank;
            if (base_sqnorm_.size() != static_cast<size_t>(out)) refresh_base_sqnorm(base_w);

            gram_.set_size(r, r);
            tt::gemm(0.0f, gram_, 1.0f, a, true, a, false);      // A'A
            proj_.set_size(out, r);
            tt::gemm(0.0f, proj_, 1.0f, base_w, true, a, false); // W'A

            const float* pg = gram_.host();
            const float* pp = proj_.host();
            const float* pb = b.host();
            const float* pm = m.host();

            col_norm_.assign(static_cast<size_t>(out), 0.0f);
            col_scale_.set_size(1, out);
            float* pc = col_scale_.host_write_only();
            std::vector<float> bj(static_cast<size_t>(r));

            for (long j = 0; j < out; ++j)
            {
                for (long q = 0; q < r; ++q) bj[static_cast<size_t>(q)] = pb[static_cast<size_t>(q) * out + j];

                float cross = 0.0f;
                for (long q = 0; q < r; ++q) cross += pp[static_cast<size_t>(j) * r + q] * bj[static_cast<size_t>(q)];

                float quad = 0.0f;
                for (long q = 0; q < r; ++q)
                {
                    float row = 0.0f;
                    for (long p = 0; p < r; ++p) row += pg[static_cast<size_t>(q) * r + p] * bj[static_cast<size_t>(p)];
                    quad += bj[static_cast<size_t>(q)] * row;
                }

                float n2 = base_sqnorm_[static_cast<size_t>(j)] + 2.0f * scale_ * cross
                    + scale_ * scale_ * quad;
                if (n2 < 1e-12f) n2 = 1e-12f;
                const float n = std::sqrt(n2);
                col_norm_[static_cast<size_t>(j)] = n;
                pc[j] = pm[j] / n;
            }
        }

        /* Contribution of the column norms to dA and dB. Both reduce to products with the
           already available A'A and W'A, so the exact gradient costs a pair of small
           matrix products rather than the merged weight the naive derivation asks for. */
        void accumulate_magnitude_path(const tensor& base_w,
            const tensor& a, const tensor& b, tensor& da, tensor& db)
        {
            const long in = geom_.in_dim, out = geom_.out_dim, r = geom_.rank;
            const float* pb = b.host();
            const float* pg = gram_.host();
            const float* pp = proj_.host();
            const float* w = base_w.host();
            float* pda = da.host();
            float* pdb = db.host();
            const float* pa = a.host();

            /* dA += s.( W.diag(k).B' + s.A.(B.diag(k).B') ) */
            bk_.assign(static_cast<size_t>(r) * out, 0.0f);          // B.diag(k), r x out
            for (long q = 0; q < r; ++q)
                for (long j = 0; j < out; ++j)
                    bk_[static_cast<size_t>(q) * out + j] = pb[static_cast<size_t>(q) * out + j] * k_[static_cast<size_t>(j)];

            bkb_.assign(static_cast<size_t>(r) * r, 0.0f);           // B.diag(k).B', r x r
            for (long q = 0; q < r; ++q)
                for (long p = 0; p < r; ++p)
                {
                    float acc = 0.0f;
                    for (long j = 0; j < out; ++j)
                        acc += bk_[static_cast<size_t>(q) * out + j] * pb[static_cast<size_t>(p) * out + j];
                    bkb_[static_cast<size_t>(q) * r + p] = acc;
                }

            for (long i = 0; i < in; ++i)
            {
                const float* wrow = w + static_cast<size_t>(i) * out;
                const float* arow = pa + static_cast<size_t>(i) * r;
                float* drow = pda + static_cast<size_t>(i) * r;
                for (long q = 0; q < r; ++q)
                {
                    float acc = 0.0f;
                    for (long j = 0; j < out; ++j) acc += wrow[j] * bk_[static_cast<size_t>(q) * out + j];
                    float second = 0.0f;
                    for (long p = 0; p < r; ++p) second += arow[p] * bkb_[static_cast<size_t>(p) * r + q];
                    drow[q] += scale_ * (acc + scale_ * second);
                }
            }

            /* dB += s.( (W'A)' + s.(A'A).B ).diag(k) */
            for (long q = 0; q < r; ++q)
                for (long j = 0; j < out; ++j)
                {
                    float second = 0.0f;
                    for (long p = 0; p < r; ++p)
                        second += pg[static_cast<size_t>(q) * r + p] * pb[static_cast<size_t>(p) * out + j];
                    const float first = pp[static_cast<size_t>(j) * r + q];
                    pdb[static_cast<size_t>(q) * out + j] +=
                        scale_ * (first + scale_ * second) * k_[static_cast<size_t>(j)];
                }
        }

        adapter_geometry geom_;
        float scale_ = 0.0f;
        double alpha_ = 0.0;

        alias_tensor a_alias_, b_alias_, m_alias_;

        resizable_tensor xa_, dub_, pre_scale_, scaled_dy_, prod_;
        resizable_tensor gram_, proj_, col_scale_, col_grad_;

        std::vector<float> base_sqnorm_, col_norm_, k_, bk_, bkb_;
    };
}

#endif // DLIB_DNN_LORA_ADAPTER_H_
