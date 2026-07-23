// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNN_LORA_ADAPTER_ABSTRACT_H_
#ifdef DLIB_DNN_LORA_ADAPTER_ABSTRACT_H_

#include "../cuda/tensor_abstract.h"

namespace dlib
{
    /*!
        WHAT THIS FILE REPRESENTS
            The shared core of the low-rank fine-tuning family, for a single frozen
            projection y = x.W with W of shape (in_dim, out_dim).

                LoRA   W' = W + s.A.B
                DoRA   W' = m * (W + s.A.B) / ||W + s.A.B||   (column by column)

            with A of shape (in_dim, rank), B of shape (rank, out_dim), m a vector of
            out_dim magnitudes and s = alpha / rank. DoRA is LoRA followed by a per-column
            rescale, which is why one implementation serves both and why another variant
            of the family costs a branch rather than a rewrite.

            The merged weight is never formed. The output follows from the distributive
            law, and the column norms from an identity built on A'A and W'A, both small
            next to the projection they adapt. The same identity differentiates, so the
            magnitude path carries its exact gradient rather than the detached
            approximation the published implementations settle for.

        INERTNESS
            B starts at zero and m at the column norms of W, so an adapter added to a
            trained model reproduces it bit for bit until the first optimizer step. A run
            is therefore comparable to the run it started from, and a non-regression check
            on a freshly adapted network is exact rather than approximate.

        THREAD SAFETY
            An instance holds the scratch buffers of one projection and is not thread
            safe. It belongs to its layer, like the rest of that layer's scratch state.

        TYPICAL USAGE
            low_rank_adapter ad;
            ad.configure(EMBEDDING_DIM, Q_PROJ_DIM, 8, adapter_method::dora, 16.0);
            // reserve ad.parameter_count() floats in the layer's packed blob, then
            ad.initialize(wq, a, b, m, rnd);
            ...
            tt::gemm(0.0f, y, 1.0f, x, false, wq, false);   // frozen base projection
            ad.forward(x, wq, a, b, m, y);                  // adapted output
            ...
            ad.backward(x, wq, a, b, m, dy, dx, da, db, dm);
    !*/

// ----------------------------------------------------------------------------------------

    enum class adapter_method
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                Which member of the family an adapter implements.

            VALUES
                none - inactive, the projection is used as is
                lora - additive low-rank update
                dora - low-rank update with a trained per-column magnitude
        !*/
    };

    std::string adapter_method_name(adapter_method m);
    adapter_method adapter_method_from_name(const std::string& s);
    /*!
        ensures
            - Convert between the enumeration and the names "none", "lora" and "dora".
            - An unrecognized name yields adapter_method::none.
    !*/

// ----------------------------------------------------------------------------------------

    struct adapter_geometry
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The shape of one adapter and its layout inside a packed parameter blob,
                which is [A | B | m] with m absent outside DoRA.

                Because the method decides the size of the blob, changing it is a
                configuration-time decision rather than a runtime one.

            FIELDS
                in_dim, out_dim - shape of the adapted projection
                rank            - inner dimension of A and B; zero deactivates the adapter
                method          - which member of the family

            METHODS
                active()   - true when the adapter contributes anything
                a_count(), b_count(), m_count(), total() - block sizes, in floats
                a_offset(base), b_offset(base), m_offset(base) - block positions,
                                 relative to the offset of the adapter in the blob
        !*/
    };

// ----------------------------------------------------------------------------------------

    class low_rank_adapter
    {
    public:

        void configure(
            long in_dim,
            long out_dim,
            long rank,
            adapter_method method,
            double alpha
        );
        /*!
            requires
                - in_dim > 0 && out_dim > 0 && rank >= 0
            ensures
                - #geometry() describes the configured adapter.
                - #scale() == alpha / rank, zero when rank is zero. The division by the
                  rank keeps the effective step comparable when the rank changes, so a
                  rank sweep does not silently become a learning-rate sweep.
                - Invalidates any cached base column norms.
        !*/

        const adapter_geometry& geometry() const;
        bool active() const;
        size_t parameter_count() const;
        float scale() const;
        double alpha() const;

        alias_tensor a_view() const;
        alias_tensor b_view() const;
        alias_tensor m_view() const;
        /*!
            ensures
                - Return the views of the three blocks, to be applied to the host layer's
                  packed parameter tensor at the offsets given by the geometry.
        !*/

        void initialize(
            const tensor& base_w,
            tensor& a,
            tensor& b,
            tensor& m,
            dlib::rand& rnd
        );
        /*!
            requires
                - a, b and m are views matching the configured geometry
                - base_w holds in_dim * out_dim values
            ensures
                - #a is drawn from N(0, 1/rank), #b is zero, and for DoRA #m holds the
                  column norms of base_w.
                - The adapter is inert: the adapted output equals the base output exactly.
                - The zero sits on B rather than on A because a pair with both blocks zero
                  would have no gradient at all.
        !*/

        void refresh_base_sqnorm(
            const tensor& base_w
        );
        /*!
            ensures
                - Recomputes the cached squared column norms of the base weight. The base
                  is frozen, so this is needed once, and again only after merge_into_base()
                  which is the single event that changes them.
        !*/

        void forward(
            const tensor& x,
            const tensor& base_w,
            const tensor& a,
            const tensor& b,
            const tensor& m,
            tensor& y
        );
        /*!
            requires
                - x is a (rows, in_dim) matrix view and y a (rows, out_dim) one
                - y holds x.W on entry
            ensures
                - #y holds the adapted output.
                - Retains x.A, and for DoRA the pre-rescale output and the column factors,
                  for the backward pass of the same batch. A forward must therefore precede
                  the matching backward.
                - Does nothing when the adapter is inactive, so a rank of zero leaves the
                  host layer bit for bit unchanged.
        !*/

        void backward(
            const tensor& x,
            const tensor& base_w,
            const tensor& a,
            const tensor& b,
            const tensor& m,
            const tensor& dy,
            tensor& dx,
            tensor& da,
            tensor& db,
            tensor& dm
        );
        /*!
            requires
                - forward() was called on the same batch
                - dy is the gradient of the adapted output
            ensures
                - Accumulates into #dx, #da, #db and #dm rather than overwriting them, so
                  the caller may add the frozen projection's own contribution to dx before
                  or after, in any order.
                - The magnitude path is exact: the dependency of the column norms on A and
                  B is differentiated, not detached.
        !*/

        void merge_into_base(
            const tensor& a,
            const tensor& b,
            const tensor& m,
            tensor& base_w
        );
        /*!
            ensures
                - #base_w becomes the merged weight W', and the cached column norms are
                  refreshed accordingly.
                - Sequential fine-tuning requires this between stages: resuming a second
                  stage on saved adapters trains the adapters alone and leaves the base
                  exactly where the first stage found it, so the knowledge of that stage
                  is never carried forward.
                - Merging is also how an adapted model is exported for inference, the
                  adapter then costing nothing.
                - The adapter blocks are left untouched; reset them through initialize()
                  before starting another stage.
        !*/
    };

    void serialize(const low_rank_adapter& item, std::ostream& out);
    void deserialize(low_rank_adapter& item, std::istream& in);
    /*!
        ensures
            - Only the configuration travels: dimensions, rank, method and alpha. The
              parameters themselves live in the host layer's packed blob and are
              serialized with it.
    !*/
}

#endif // DLIB_DNN_LORA_ADAPTER_ABSTRACT_H_
