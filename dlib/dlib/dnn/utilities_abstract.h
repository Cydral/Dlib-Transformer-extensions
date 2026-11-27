// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_UTILITIES_ABSTRACT_H_
#ifdef DLIB_DNn_UTILITIES_ABSTRACT_H_

#include "core_abstract.h"
#include "../geometry/vector_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    double log1pexp(
        double x
    );
    /*!
        ensures
            - returns log(1+exp(x))
              (except computes it using a numerically accurate method)
    !*/

// ----------------------------------------------------------------------------------------

    void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    );
    /*!
        ensures
            - This function assigns random values into params based on the given random
              number generator.  In particular, it uses the parameter initialization method
              of formula 16 from the paper "Understanding the difficulty of training deep
              feedforward neural networks" by Xavier Glorot and Yoshua Bengio.
            - It is assumed that the total number of inputs and outputs from the layer is
              num_inputs_and_outputs.  That is, you should set num_inputs_and_outputs to
              the sum of the dimensionalities of the vectors going into and out of the
              layer that uses params as its parameters.
    !*/

    void randomize_parameters_truncated_normal (
        tensor& params,
        double deviation,
        dlib::rand& rnd
    );
    /*!
        requires
            - deviation >= 0
        ensures
            - Assigns to every element of params a value drawn from a normal law of mean
              zero truncated at two deviations, scaled so that the values have the given
              deviation. This is the truncated normal initialization of JAX and Flax,
              which the reference implementation of the Hierarchical Reasoning Model uses.
    !*/

    enum class weight_init
    {
        glorot_uniform,
        truncated_lecun_normal
    };
    /*!
        WHAT THIS ENUM REPRESENTS
            How a projection layer draws its weights when it is set up.
              - glorot_uniform:         randomize_parameters(), the default of the
                                        library.
              - truncated_lecun_normal: randomize_parameters_truncated_normal() with a
                                        deviation of one over the square root of the
                                        input width.
            Biases start at zero in both cases.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename LayerType>
    class layer_helper
    {
        /*!
            REQUIREMENTS ON LayerType
                - LayerType is default-constructible.
                - LayerType implements the layer interface used by add_layer<>:
                  template <typename SUBNET> void setup(const SUBNET& sub);
                  template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
                  template <typename SUBNET> void backward(const tensor& gradient_input,
                                                           SUBNET& sub,
                                                           tensor& params_grad);
                - In setup() and forward(), LayerType reads only sub.get_output() from the
                  SUBNET argument; it does not require access to deeper layers via layer<>().
                - In backward(), LayerType writes to sub.get_gradient_input(); the
                  params_grad argument is ignored (LayerType is parameter-free or its
                  parameters are managed internally without participating in the host
                  layer's parameter optimization).

            WHAT THIS OBJECT REPRESENTS
                This object is a generic adapter that lets you reuse any parameter-free
                Dlib layer as a sub-component inside another layer's forward/backward
                implementation, without having to hand-roll a SUBNET-like wrapper for
                each call site.

                The motivating use case is a fused composite layer (e.g. a
                grouped-query attention block consolidating linear projections,
                rotary positional embedding, causal masking, softmax, and matmul
                into a single Dlib layer). Such a layer typically wants to delegate
                some of its sub-computations to existing parameter-free Dlib layers
                like rotary_positional_embedding_ or tril_ rather than reimplement
                their internal state and edge cases.

                Examples of compatible LayerTypes:
                    - tril_<...>                       (causal masking)
                    - rotary_positional_embedding_     (RoPE with optional YaRN)

                Examples of NOT compatible LayerTypes (have trainable parameters
                that would need to participate in the host layer's parameter
                optimization):
                    - linear_, fc_, con_, rms_norm_, layer_norm_

                The wrapped layer's internal state (caches, masks, position-
                dependent precomputations, etc.) persists across forward/backward
                calls on the same layer_helper instance. The wrapped layer is
                serialized and deserialized along with the helper so that
                save/load round-trips preserve any precomputed state.

            THREAD SAFETY
                This object is not thread-safe. The wrapped layer maintains internal
                state that is mutated by setup/forward/backward. Concurrent calls on
                the same instance from different threads are not supported.
        !*/

    public:

        layer_helper(
        );
        /*!
            ensures
                - The wrapped layer is default-constructed.
        !*/

        LayerType& get(
        );
        /*!
            ensures
                - returns a non-const reference to the wrapped layer instance, so
                  callers can configure it (e.g. set hyperparameters) before the
                  first forward call.
        !*/

        const LayerType& get(
        ) const;
        /*!
            ensures
                - returns a const reference to the wrapped layer instance.
        !*/

        void setup(
            const tensor& input_proxy
        );
        /*!
            ensures
                - Invokes the wrapped layer's setup() method, with input_proxy
                  standing in for sub.get_output() of a normal SUBNET.
                - Useful when the host layer wants to pre-allocate caches or
                  internal state that depend on the expected input shape, before
                  the first forward call.
                - input_proxy does not need to contain meaningful data; only its
                  shape is read by typical layer setup() implementations.
        !*/

        void forward(
            const tensor& input,
            resizable_tensor& output
        );
        /*!
            ensures
                - Applies the wrapped layer's forward computation to input and
                  writes the result to output.
                - output is resized as needed by the wrapped layer's forward()
                  implementation.
                - The wrapped layer's internal state may be updated (e.g. caches
                  rebuilt if the input shape changed since the previous call).
        !*/

        void backward(
            const tensor& gradient_input,
            const tensor& original_input,
            resizable_tensor& input_grad
        );
        /*!
            requires
                - gradient_input has the shape of the output produced by the
                  most recent forward() call (i.e. dL/d(output)).
                - original_input is the same tensor (or numerically equivalent)
                  that was passed to the most recent forward() call.
            ensures
                - Computes the gradient of the loss with respect to the input of
                  the wrapped layer and writes it into input_grad.
                - input_grad is resized to match original_input and zero-
                  initialized before the wrapped layer's backward() is called.
                  This ensures that wrapped layers using add-to semantics for
                  their input gradient (the Dlib convention) produce a clean
                  stand-alone gradient rather than accumulating into stale data.
                - The wrapped layer is treated as parameter-free: any params_grad
                  that the wrapped layer's backward() tries to populate is
                  discarded.
        !*/

        friend void serialize(
            const layer_helper& item,
            std::ostream& out
        );
        /*!
            ensures
                - Serializes item to out by delegating to the wrapped layer's
                  own serialize() function. A version tag is prepended so future
                  changes to the helper format can be detected at load time.
        !*/

        friend void deserialize(
            layer_helper& item,
            std::istream& in
        );
        /*!
            ensures
                - Deserializes item from in.
            throws
                - serialization_error if the data on the stream was not produced
                  by a compatible serialize() call (in particular, if the
                  version tag does not match).
        !*/
    };

// ----------------------------------------------------------------------------------------
// Parameter vectors and their consolidation across replicas
// ----------------------------------------------------------------------------------------

    template <typename T>
    void deserialize_bounded (
        std::vector<T>& item,
        std::istream& in
    );
    /*!
        ensures
            - reads a vector written by serialize(), as deserialize() does, after checking
              that its declared length does not exceed the bytes left in the stream, every
              element taking one byte at least. A vector of vectors is read the same way at
              every level.
            - The memory grows with the elements actually read, so a stream that cannot tell
              how many bytes it has left (a socket) reserves no more than it delivers.
            - Meant for payloads received from other hosts: a length forged in one would
              otherwise have deserialize() reserve memory without limit before the stream
              runs out. The objects of this section read their vectors this way.
        throws
            - serialization_error when the length exceeds the stream or the stream ends early.
    !*/

    struct parameter_block
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                One tensor of a parameter vector: where its values start, how many there
                are, and the matrix they form, rows by cols in row-major order. A tensor
                of shape (n, k, nr, nc) is read as n rows of k*nr*nc values, which is the
                (in_dim, out_dim) matrix of a fully connected projection.

                is_matrix() is true when both dimensions exceed one: the operations on
                singular values and the factored curvature apply to those blocks, and the
                others are vectors.
        !*/

        size_t offset = 0;
        size_t size = 0;
        long rows = 0;
        long cols = 0;

        bool is_matrix () const;
    };

    class parameter_layout
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The structure of a flat vector of parameters: consecutive blocks that tile
                it from its first value to its last. Every replica of a model derives the
                same layout from the same network, so two vectors of the same layout hold
                the same parameters at the same positions.

                The layout of a network comes from get_parameter_layout(). A tensor that
                packs several matrices, the fused attention of a decoder for instance, is
                described matrix by matrix with split_block().
        !*/

    public:
        void clear ();

        void add_block (long rows, long cols);
        /*!
            requires
                - rows > 0 && cols > 0
            ensures
                - appends a block of rows*cols values at the end of the vector.
        !*/

        void split_block (size_t index, const std::vector<std::pair<long,long>>& shapes);
        /*!
            requires
                - index < num_blocks()
                - the shapes are positive and their sizes sum to at most block(index).size
            ensures
                - replaces block index by blocks of the given shapes, in order, followed
                  by a one-row block for the remaining values when they do not fill it.
                - the blocks after it keep their offsets and move up in the list.
        !*/

        size_t num_blocks () const;
        const parameter_block& block (size_t i) const;
        const std::vector<parameter_block>& blocks () const;
        size_t size () const;
        /*!
            ensures
                - returns the number of values the blocks cover.
        !*/

        uint64 fingerprint () const;
        /*!
            ensures
                - returns a hash of the shapes of the blocks, equal for two equal layouts.
        !*/

        bool operator== (const parameter_layout& other) const;
        bool operator!= (const parameter_layout& other) const;
    };

    void serialize (const parameter_layout& item, std::ostream& out);
    void deserialize (parameter_layout& item, std::istream& in);
    /*!
        provides serialization support.  deserialize() refuses blocks that do not tile
        the vector or whose size differs from their shape.
    !*/

    uint64 parameter_digest (
        const std::vector<float>& values
    );
    /*!
        ensures
            - returns a 64-bit hash of the bytes of values. Two replicas holding the same
              state compute the same digest, which is how a federation checks that every
              replica applied the same update to the same state.
    !*/

    double scale_adam_beta2 (
        double beta2,
        double reference_replicas,
        double replicas
    );
    /*!
        requires
            - 0 <= beta2 < 1
            - reference_replicas > 0 && replicas > 0
        ensures
            - returns beta2^(reference_replicas / replicas): the second-moment decay that
              keeps the half-life of the moving average constant in tokens when a fixed
              global batch is shared by replicas instead of reference_replicas. Used for
              the inner optimizer of local training, whose estimates of the second
              moment otherwise lose precision as the batch of each replica shrinks.
    !*/

    double privatize_delta (
        std::vector<float>& delta,
        double clip_norm,
        double noise_multiplier,
        dlib::rand& rnd
    );
    /*!
        requires
            - clip_norm > 0
            - noise_multiplier >= 0
        ensures
            - scales delta down to a Euclidean norm of at most clip_norm, then adds to
              every value a Gaussian noise of standard deviation
              noise_multiplier*clip_norm: the Gaussian mechanism a replica applies on its
              own update before it leaves the host.
            - returns the norm of delta before clipping.
    !*/

// ----------------------------------------------------------------------------------------

    struct delta_codec_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                How a delta_encoder compresses an update.
                  keep_fraction - share of the values of each block sent, those of the
                                  largest magnitude, in (0, 1]
                  bits          - 8 or 4 for values quantized in groups of 32 sharing one
                                  scale, 32 for values sent as floats
        !*/

        double keep_fraction = 1.0;
        int bits = 8;
    };

    class compressed_delta
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An update in the compact form delta_encoder produces: for each block of a
                layout, its kept positions coded as gaps and its kept values quantized.
                It is what a replica sends and what decode_delta() expands.

                deserialize() checks every count against every other, and every scale and
                value against the range of a float, before anything is decoded, and
                decode_delta() checks every position against its block, so a damaged or
                forged payload is refused rather than read out of bounds. Every value a
                compressed_delta decodes to is finite.
        !*/

    public:
        size_t dimension () const;
        bool empty () const;
        int bits () const;
        size_t kept () const;
        /*!
            ensures
                - returns the number of values the update carries.
        !*/

        size_t byte_size () const;
        /*!
            ensures
                - returns the size of the payload, in bytes, near that of its serialization.
        !*/
    };

    void serialize (const compressed_delta& item, std::ostream& out);
    void deserialize (compressed_delta& item, std::istream& in);
    /*!
        provides serialization support.
        throws
            - serialization_error when the payload is inconsistent.
    !*/

    void decode_delta (
        const compressed_delta& in,
        std::vector<float>& out
    );
    /*!
        requires
            - in.dimension() is the dimension the caller expects: the vector is allocated
              before the values are read, so a caller checks dimension() first.
        ensures
            - #out.size() == in.dimension()
            - #out holds the values in carries at their positions and zeros elsewhere.
        throws
            - serialization_error when a position of in is invalid.
    !*/

    class delta_encoder
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The compressor of one replica, with error feedback: what an encoding leaves
                out, by quantization or by sparsification, is kept in a residual and added
                to the next update before it is encoded. Over successive rounds every part
                of the updates is eventually sent, which is what keeps a compressed
                federation converging where plain compression stalls.

                An encoder belongs to one replica and one layout. serialize() saves the
                residual with the options, so a replica that restarts loses nothing.
        !*/

    public:
        delta_encoder ();
        explicit delta_encoder (const delta_codec_options& options);
        const delta_codec_options& options () const;
        void set_options (const delta_codec_options& options);
        void reset ();
        /*!
            ensures
                - clears the residual.
        !*/

        const std::vector<float>& residual () const;

        void encode (
            const parameter_layout& layout,
            const std::vector<float>& delta,
            compressed_delta& out
        );
        /*!
            requires
                - delta.size() == layout.size()
            ensures
                - #out holds delta plus the residual, compressed block by block according
                  to options().
                - #residual() holds what #out leaves out of delta plus the previous
                  residual, so that decoding #out and adding #residual() restores it.
            throws
                - dlib::error when delta plus the residual holds a value that is not finite
                  or, for quantized values, larger than half the largest float; the encoder
                  is then left as it was.
        !*/
    };

    void serialize (const delta_encoder& item, std::ostream& out);
    void deserialize (delta_encoder& item, std::istream& in);
    void serialize (const delta_codec_options& item, std::ostream& out);
    void deserialize (delta_codec_options& item, std::istream& in);

// ----------------------------------------------------------------------------------------

    class curvature_summary
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A compact estimate of a non-negative statistic over a parameter vector,
                typically the mean of the squared gradients a replica measured, which is
                the diagonal of its empirical Fisher information. A matrix block keeps the
                sums of its rows and of its columns, whose product divided by the total
                estimates each entry, as Adafactor factors its second moments; a vector
                block keeps its values. The summary of a decoder thus weighs about as much
                as its norms and biases.

                delta_consolidator reads it to weigh each replica, coordinate by
                coordinate, by how sharply its loss responds there.
        !*/

    public:
        bool empty () const;
        size_t num_blocks () const;
        void clear ();

        void build (
            const parameter_layout& layout,
            const std::vector<float>& diagonal
        );
        /*!
            requires
                - diagonal.size() == layout.size()
            ensures
                - #*this summarizes diagonal block by block, negative values read as zero.
        !*/

        bool matches (const parameter_layout& layout) const;
        /*!
            ensures
                - returns true when the summary was built on a layout of the same blocks.
        !*/

        void expand_normalized (
            size_t index,
            float* dst
        ) const;
        /*!
            requires
                - index < num_blocks()
                - dst has room for the values of block index
            ensures
                - writes the estimate of every value of the block, divided by its mean
                  over the block, so that the values average to one; a block whose
                  statistic is zero everywhere reads as ones.
        !*/
    };

    void serialize (const curvature_summary& item, std::ostream& out);
    void deserialize (curvature_summary& item, std::istream& in);
    /*!
        provides serialization support. deserialize() refuses negative or non-finite values.
    !*/

// ----------------------------------------------------------------------------------------

    enum class consolidation_method
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                How delta_consolidator turns the updates of the replicas into one.

            VALUES
                average             - the mean of the updates weighted by the replicas,
                                      applied with the step size given (FedAvg);
                outer_momentum      - the same mean through a Nesterov momentum and the
                                      outer learning rate (DiLoCo);
                curvature_consensus - the procedure described in delta_consolidator.
        !*/
        average,
        outer_momentum,
        curvature_consensus
    };

    std::string consolidation_method_name (consolidation_method m);
    consolidation_method consolidation_method_from_name (const std::string& name);
    /*!
        ensures
            - convert between the enumeration and its names.
        throws
            - dlib::error for an unknown name.
    !*/

    struct consolidation_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The settings of a delta_consolidator. A value of zero disables the stage it
                controls, which leaves each stage open to an ablation.

                method              - see consolidation_method
                outer_learning_rate - step size of the outer update
                outer_momentum      - Nesterov momentum of the outer update, in [0, 1)
                curvature_mix       - weight of the curvature in the weight of a
                                      coordinate, in [0, 1]; 0 weighs every coordinate of
                                      a replica alike
                curvature_ceiling   - bound on the normalized curvature of a coordinate, >= 1
                consensus_power     - exponent of the agreement of the replicas applied to
                                      each coordinate of the mean; 0 disables the shrinkage
                spectral_clip       - singular values of a matrix block above this multiple
                                      of their root mean square are brought down to it; 0
                                      disables the clipping
                spectral_rank       - number of leading singular values examined per block
                norm_bound          - a block of an update larger than this multiple of the
                                      median norm of that block across three replicas or
                                      more is scaled down to it; 0 disables the bound
                trust_ratio         - a step larger than this fraction of the norm of a
                                      block is scaled down to it; 0 disables the bound
                staleness_decay     - weight factor per round of age of an update
                max_staleness       - age beyond which an update is ignored
                step_multipliers    - multiples of the outer learning rate that the replicas
                                      evaluate on held-out data before a step is chosen;
                                      empty when the step is applied without evaluation

                The defaults are those of curvature_consensus as measured by the example
                slm_federated_ex: no outer momentum, a step of one, curvature and consensus
                at one half, the step chosen among half, one and twice the direction. The
                outer momentum and the spectral clipping are left at zero there, where they
                cost held-out loss with few replicas and few rounds; the literature measures
                their benefit with many replicas and long runs (DiLoCo, IsoLoCo).
        !*/

        consolidation_method method = consolidation_method::curvature_consensus;
        double outer_learning_rate = 1.0;
        double outer_momentum = 0.0;
        double curvature_mix = 0.5;
        double curvature_ceiling = 10;
        double consensus_power = 0.5;
        double spectral_clip = 0.0;
        long spectral_rank = 8;
        double norm_bound = 2.5;
        double trust_ratio = 0;
        double staleness_decay = 0.5;
        long max_staleness = 1;
        std::vector<double> step_multipliers = {0.5, 1.0, 2.0};
    };

    void serialize (const consolidation_options& item, std::ostream& out);
    void deserialize (consolidation_options& item, std::istream& in);

    void apply_consolidated_step (
        const parameter_layout& layout,
        std::vector<float>& params,
        const std::vector<float>& direction,
        double step,
        double trust_ratio = 0
    );
    /*!
        requires
            - params.size() == layout.size() && direction.size() == layout.size()
        ensures
            - #params == params - step*direction, block by block, where a block whose step
              would exceed trust_ratio times its norm (with a floor for blocks near zero)
              is scaled down to that bound, when trust_ratio > 0.
            - The result depends only on its arguments: replicas of the same build applying
              the same direction to the same state reach the same state, bit for bit (builds
              that contract products into fused operations differently may differ in the
              last bit, which parameter_digest() reveals).
    !*/

    class delta_consolidator
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The outer optimizer of a federation: it gathers the updates replicas
                computed from a shared state, each one the difference between that state
                and the state its local training reached, and turns them into the direction
                every replica then steps along.

                With curvature_consensus, a round runs these stages:
                  1. an update older than max_staleness rounds is ignored, a younger one
                     loses staleness_decay of its weight per round of age;
                  2. with three updates or more, each block of each update is bounded by
                     norm_bound times the median of the norms of that block across the
                     replicas, one vote each, which limits what a faulty or hostile replica
                     moves. The median is the lower one: a block that half the replicas or
                     more leave unchanged stays unchanged, which also holds against half of
                     them pushing it, and which a federation where only some replicas train
                     a block (routed experts) has to take into account;
                  3. each coordinate is the mean of the updates, each weighted by its
                     replica and by the normalized curvature the replica measured there,
                     mixed with uniform weights by curvature_mix and capped at
                     curvature_ceiling: a replica whose loss is sharp along a coordinate
                     speaks for it with more weight, as in Fisher merging;
                  4. each coordinate is shrunk by the agreement of the replicas raised to
                     consensus_power, the agreement being the ratio of the magnitude of the
                     weighted sum to the weighted sum of magnitudes, one when they all move
                     it the same way, zero when they cancel out: a signal-to-noise ratio
                     that damps the directions the replicas dispute;
                  5. the singular values of each matrix block above spectral_clip times
                     their root mean square are brought down to that level, which keeps a
                     few dominant directions from carrying the whole step;
                  6. the result goes through a Nesterov momentum.
                proposal() is then the direction, and the replicas may measure the
                held-out loss of the states step_size(m) along it for each multiplier m
                of the options; choose_multiplier() picks the best of them, or none when
                the current state is better, and the round is committed or rejected.

                With average and outer_momentum, stages 1 and 6 apply (6 for
                outer_momentum only) to the plain weighted mean.

                Stage 2 and the medians of choose_multiplier() tolerate a minority of faulty
                or hostile replicas when there are three of them or more. With average, with
                outer_momentum, or with fewer than three replicas, one replica can move the
                direction as far as the range of a float allows, or make consolidate() return
                false, and so keep every round from being committed.

                The momentum persists across rounds and is serialized with the layout,
                the options and the round counter, so a master can stop and resume.
        !*/

    public:
        delta_consolidator ();
        delta_consolidator (const parameter_layout& layout, const consolidation_options& options);
        void setup (const parameter_layout& layout, const consolidation_options& options);
        /*!
            requires
                - layout.size() > 0
                - the options lie in the ranges given in consolidation_options
            ensures
                - starts a consolidation at round 0, with a zero momentum.
        !*/

        const parameter_layout& layout () const;
        const consolidation_options& options () const;
        long round () const;
        size_t num_contributions () const;
        void clear_contributions ();
        const std::vector<float>& momentum () const;

        void add (
            const compressed_delta& delta,
            double weight,
            const curvature_summary& curvature = curvature_summary(),
            long base_round = -1
        );
        void add (
            const std::vector<float>& delta,
            double weight,
            const curvature_summary& curvature = curvature_summary(),
            long base_round = -1
        );
        /*!
            ensures
                - records the update of a replica, computed from the state of base_round
                  (the current round when negative), with its weight and its curvature. A
                  compressed update stays compressed until consolidate() reads it.
            throws
                - dlib::error when the update does not match the layout, when the weight is
                  not positive and finite, when a value of a dense update is not finite, or
                  when the curvature does not match the layout. The consolidator is then left
                  as it was. A compressed update holds only finite values, which
                  deserialize() and delta_encoder ensure.
        !*/

        bool consolidate ();
        /*!
            ensures
                - turns the recorded updates into #proposal() and returns true, or returns
                  false when none of them is recent enough or when the sums of the updates
                  overflow the range of a float, which only updates of that order of
                  magnitude cause.
                - The weights of the replicas count relative to the heaviest of them, and the
                  norm bound takes the median of the replicas with one vote each, so that a
                  replica declaring a large weight moves the mean and never the bound.
                - the momentum changes only at commit().
            throws
                - serialization_error when a compressed contribution cannot be decoded.
        !*/

        bool has_proposal () const;
        const std::vector<float>& proposal () const;

        double step_size (double multiplier) const;
        /*!
            ensures
                - returns multiplier for average, and outer_learning_rate*multiplier for
                  the other methods.
        !*/

        void apply (std::vector<float>& params, double multiplier) const;
        /*!
            requires
                - has_proposal()
            ensures
                - apply_consolidated_step(layout(), params, proposal(),
                  step_size(multiplier), options().trust_ratio)
        !*/

        static size_t choose_multiplier (
            const std::vector<std::vector<double>>& losses,
            const std::vector<double>& weights
        );
        /*!
            requires
                - losses.size() == weights.size() > 0
                - all the rows of losses have the same length > 0
                - weights are >= 0 with a positive sum
            ensures
                - losses[k][j] being the validation loss replica k measured for candidate j,
                  candidate 0 the current state and candidate j the multiplier j-1, returns
                  the candidate whose change of loss over the current state is the lowest and
                  negative; 0 when no candidate lowers it.
                - A replica of zero weight, or whose loss of the current state is not finite,
                  is left out of every candidate. A candidate loss that is not finite, over a
                  finite loss of the current state, counts as an infinite increase.
                - With three replicas or more, the change of a candidate is the weighted
                  median of the changes the replicas measured, the midpoint of the lower and
                  the upper weighted medians (the mean of the two middle changes for an even
                  count of equal weights). Replicas holding less than half of the weight keep
                  it within the range of the changes of the others: they can neither have a
                  step taken that every other replica finds worse, nor have the round
                  rejected when every other replica finds the same step better. With one or
                  two replicas, it is their weighted mean: a step that helps one replica and
                  harms the other as much is not taken, and either replica can still steer
                  the choice by the losses it reports.
        !*/

        void commit ();
        /*!
            requires
                - has_proposal()
            ensures
                - the momentum takes the value the proposal was computed with, the
                  contributions are cleared and #round() == round() + 1.
        !*/

        void reject ();
        /*!
            ensures
                - the proposal is discarded, the momentum is multiplied by the outer
                  momentum, the contributions are cleared and #round() == round() + 1.
        !*/
    };

    void serialize (const delta_consolidator& item, std::ostream& out);
    void deserialize (delta_consolidator& item, std::istream& in);
    /*!
        provides serialization support for the layout, the options, the momentum and the
        round. The contributions and the proposal of a round in progress are not saved.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_ABSTRACT_H_ 


