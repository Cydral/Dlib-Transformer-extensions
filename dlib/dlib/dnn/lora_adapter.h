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
// B is stored transposed, as (out, rank). Every use of it then lands on an existing
// tensor operation: the low-rank product becomes a gemm with a transposed right operand,
// the two norm reductions become dot_prods over rows, and the magnitude gradients become
// gemms around a scale_rows. In the natural (rank, out) layout those same quantities pair
// a row of one matrix with a column of another and have to be walked on the host, which
// on a decoder-sized projection is tens of millions of host operations per layer and per
// optimizer step. The closing step, turning squared norms into per-column factors, is
// tt::divide_by_sqrt, so no part of a forward or a backward leaves the device. Only the
// squared column norms of the frozen base are walked on the host, once per adapter.
//
// That closing step is a division and not a multiplication by a reciprocal, because the
// two are not interchangeable here: x / x is exactly one for every finite non-zero float,
// x * (1 / x) is not, and the inertness below rests on the first identity.
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
       take. The layout is [A | B | m], B in its transposed (out, rank) form and m absent
       outside DoRA, so switching methods changes the size of the blob and is a
       configuration-time decision, not a runtime one. A rank of zero yields an empty
       layout and an inactive adapter. */
    struct adapter_geometry
    {
        long in_dim = 0;
        long out_dim = 0;
        long rank = 0;
        adapter_method method = adapter_method::none;

        bool active() const { return rank > 0 && method != adapter_method::none; }
        size_t a_count() const { return active() ? static_cast<size_t>(in_dim) * rank : 0; }
        size_t b_count() const { return active() ? static_cast<size_t>(out_dim) * rank : 0; }
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
            DLIB_CASSERT(rank >= 0, "adapter rank must be non-negative");
            /* Zero dimensions describe an adapter that was never configured, which is what
               a default-constructed one serializes as and what a layer carrying none reads
               back. Only an active adapter needs a shape. */
            DLIB_CASSERT(rank == 0 || (in_dim > 0 && out_dim > 0),
                "an active adapter needs positive dimensions");
            geom_.in_dim = in_dim;
            geom_.out_dim = out_dim;
            geom_.rank = rank;
            geom_.method = method;
            /* alpha / rank rather than alpha: it keeps the effective step of the adapter
               comparable when the rank changes, so a rank sweep does not silently become
               a learning-rate sweep. */
            scale_ = (rank > 0) ? static_cast<float>(alpha / rank) : 0.0f;
            alpha_ = alpha;
            base_sq_.clear();
            if (geom_.active())
            {
                a_alias_ = alias_tensor(in_dim, rank);
                b_alias_ = alias_tensor(out_dim, rank);   // B transposed
                m_alias_ = alias_tensor(out_dim, 1);
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
                const float* sq = base_sq_.host();
                float* pm = m.host_write_only();
                for (long j = 0; j < geom_.out_dim; ++j) pm[j] = std::sqrt(sq[j]);
            }
        }

        /* Squared column norms of the frozen base. Walked on the host on purpose: the base
           does not change, so this runs once per adapter and once more after a merge, and
           a tensor path would cost a full-size temporary to save nothing measurable. */
        void refresh_base_sqnorm(const tensor& base_w)
        {
            const long in = geom_.in_dim, out = geom_.out_dim;
            DLIB_CASSERT(base_w.size() == static_cast<size_t>(in) * out,
                "base weight does not match the configured geometry");
            base_sq_.set_size(out, 1);
            float* dst = base_sq_.host_write_only();
            for (long j = 0; j < out; ++j) dst[j] = 0.0f;
            const float* w = base_w.host();
            for (long i = 0; i < in; ++i)
            {
                const float* row = w + static_cast<size_t>(i) * out;
                for (long j = 0; j < out; ++j) dst[j] += row[j] * row[j];
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
            tt::gemm(1.0f, y, scale_, xa_, false, b, true);   // B is stored transposed

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
                accumulate_magnitude_path(base_w, a, b, m, dy, da, db, dm);
            }

            // dB' = s.du'.(x.A)   and   dA = s.x'.(du.B')
            tt::gemm(1.0f, db, scale_, *du, true, xa_, false);
            dub_.set_size(rows, geom_.rank);
            tt::gemm(0.0f, dub_, 1.0f, *du, false, b, false);
            tt::gemm(1.0f, da, scale_, x, true, dub_, false);

            // dx = du.W' + s.(du.B').A'
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

            tt::gemm(1.0f, base_w, scale_, a, false, b, true);   // W <- W + s.A.B

            if (geom_.method == adapter_method::dora)
            {
                refresh_base_sqnorm(base_w);
                col_scale_.set_size(geom_.out_dim, 1);
                tt::divide_by_sqrt(col_scale_, m, base_sq_, norm_floor);
                /* scale_columns has no aliasing guarantee, so the merged weight is read
                   from a copy rather than from the tensor being written. */
                merged_.copy_size(base_w);
                memcpy(merged_, base_w);
                tt::scale_columns(base_w, merged_, col_scale_);
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
           Everything but the closing square root and division is a tensor operation; what
           remains is one host pass over out values. */
        void compute_column_scales(const tensor& base_w,
            const tensor& a, const tensor& b, const tensor& m)
        {
            const long out = geom_.out_dim, r = geom_.rank;
            if (base_sq_.size() != static_cast<size_t>(out)) refresh_base_sqnorm(base_w);

            gram_.set_size(r, r);
            tt::gemm(0.0f, gram_, 1.0f, a, true, a, false);      // A'A
            proj_.set_size(out, r);
            tt::gemm(0.0f, proj_, 1.0f, base_w, true, a, false); // W'A
            bg_.set_size(out, r);
            tt::gemm(0.0f, bg_, 1.0f, b, false, gram_, false);   // B'.(A'A)

            tt::dot_prods(cross_, proj_, b);                     // <w_j, (AB)_j>
            tt::dot_prods(quad_, bg_, b);                        // ||(AB)_j||^2

            nsq_.set_size(out, 1);
            tt::affine_transform(nsq_, base_sq_, cross_, quad_,
                1.0f, 2.0f * scale_, scale_ * scale_);

            /* c = m / ||v||, in one division. Splitting it into a reciprocal and a
               product would cost the exact one that an untrained adapter must produce. */
            col_scale_.set_size(out, 1);
            tt::divide_by_sqrt(col_scale_, m, nsq_, norm_floor);
        }

        /* Contribution of the column norms to dA, dB and dm. Both parameter terms reduce
           to products with the already available A'A and W'A, so the exact gradient costs
           a pair of small matrix products rather than the merged weight the naive
           derivation asks for. */
        void accumulate_magnitude_path(const tensor& base_w,
            const tensor& a, const tensor& b, const tensor& m,
            const tensor& dy, tensor& da, tensor& db, tensor& dm)
        {
            const long out = geom_.out_dim, r = geom_.rank;

            prod_.copy_size(dy);
            tt::multiply(false, prod_, dy, pre_scale_);
            tsum_.set_size(out, 1);
            {
                /* assign_conv_bias_gradient sums over samples and wants a (1, out)
                   destination, which is the same storage seen with the other shape. */
                auto tsum_row = alias_tensor(1, out)(tsum_, 0);
                tt::assign_conv_bias_gradient(tsum_row, prod_);
            }

            /* Reciprocal norms, obtained from the same primitive with a unit numerator.
               Nothing downstream of here needs an exact one, so the extra rounding of a
               reciprocal is harmless in the gradients. */
            if (ones_.size() != static_cast<size_t>(out))
            {
                ones_.set_size(out, 1);
                ones_ = 1;
            }
            inv_norm_.set_size(out, 1);
            tt::divide_by_sqrt(inv_norm_, ones_, nsq_, norm_floor);

            // dm += t / ||v||
            tt::multiply(true, dm, tsum_, inv_norm_);

            /* k = -m.t / ||v||^3, written as -(c.t) / ||v||^2 so that the column factors
               computed in the forward are reused instead of recomputed. */
            ct_.set_size(out, 1);
            tt::multiply(false, ct_, col_scale_, tsum_);
            inv2_.set_size(out, 1);
            tt::multiply(false, inv2_, inv_norm_, inv_norm_);
            kvec_.set_size(out, 1);
            tt::multiply(false, kvec_, ct_, inv2_);
            tt::affine_transform(kvec_, kvec_, -1.0f, 0.0f);

            // dA += s.( W.(diag(k).B') + s.A.(B.diag(k).B') )
            bk_.set_size(out, r);
            tt::scale_rows(bk_, b, kvec_);
            tt::gemm(1.0f, da, scale_, base_w, false, bk_, false);
            bkb_.set_size(r, r);
            tt::gemm(0.0f, bkb_, 1.0f, b, true, bk_, false);
            tt::gemm(1.0f, da, scale_ * scale_, a, false, bkb_, false);

            // dB' += s.diag(k).( W'A + s.B'.(A'A) )
            mix_.set_size(out, r);
            tt::affine_transform(mix_, proj_, bg_, 1.0f, scale_);
            mixk_.set_size(out, r);
            tt::scale_rows(mixk_, mix_, kvec_);
            tt::add(1.0f, db, scale_, mixk_);
        }

        adapter_geometry geom_;
        float scale_ = 0.0f;
        double alpha_ = 0.0;

        alias_tensor a_alias_, b_alias_, m_alias_;

        resizable_tensor xa_, dub_, pre_scale_, scaled_dy_, prod_, merged_;
        resizable_tensor gram_, proj_, bg_, cross_, quad_, nsq_;
        /* Lower bound on a squared norm, applied as a clamp rather than as an added term
           so that a norm already equal to its numerator divides out to exactly one. A
           plain enumeration-free constant rather than a static constexpr member: the
           primitive takes it by const reference, which would odr-use the member and
           require an out-of-class definition under C++14. */
        double norm_floor = 1e-12;

        resizable_tensor base_sq_, col_scale_, inv_norm_, inv2_, ct_, ones_;
        resizable_tensor tsum_, kvec_, bk_, bkb_, mix_, mixk_;
    };
}

#endif // DLIB_DNN_LORA_ADAPTER_H_
