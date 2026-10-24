// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_NPU_DLIB_H_
#define DLIB_DNN_NPU_DLIB_H_

/*
    Run-time routing of tensor operations to the inference devices of the processor: its
    neural processing unit and its integrated GPU.

    dlib is compiled for one primary target, the host or CUDA. When it is also built with
    DLIB_USE_NPU, against the OpenVINO runtime, every tensor operation of tensor_tools.h that
    runs on the host first offers itself to this module. The module looks once, at the first
    offer, for the devices DLIB_NPU_DEVICE lists, the NPU and the Intel GPU by default, and then
    decides call by call whether one of them or the host computes the result. Nothing changes
    for a program: the same calls, the same tensors, the same results up to the arithmetic of
    the device. Without a device, or with routing disabled, the host path runs exactly as it
    does in a build without DLIB_USE_NPU.

    How a call is routed
        - An operation qualifies when its size reaches a threshold of its own, so that small
          operations never pay the cost of a round trip.
        - The first qualifying calls of a given shape keep running on the host while a graph
          for that shape is compiled in the background, for every device, each by a compiler
          thread of its own. Compiled graphs are cached on disk by OpenVINO, so a later run
          starts from them.
        - The host and every device whose graph is ready are timed on a few calls of that
          shape, and the fastest is kept for the calls that follow. A device whose graph is
          ready later, as that of an NPU whose compiler is slower, is timed in turn and takes
          the shape if it is faster. No call is timed while a graph compiles, since the
          compiler loads the processor and would make the host look slower than it is: a shape
          not measured yet stays on the host until the compilers are idle.
        - The choice is checked after 64 calls of the shape, then after twice as many calls at
          each check, up to every 65536 calls: one call is timed on the host and on each ready
          device, and only a check that finds another of them faster times them again on a
          few calls before choosing. A choice that the load of the machine distorted does not
          last.
        - An NPU is offered neither the gradients of the convolutions nor those of the pooling,
          which only training needs, even when DLIB_NPU_OPS names them: its compiler stopped
          the program on one of them. The GPU or the host computes them.
        - Shapes that change from call to call, such as a growing sequence, are padded up to a
          small set of sizes, so that the number of graphs stays bounded.
        - The operands of the products and of the convolutions that keep their values from one
          call to the next, the weights of a model above all, are converted once: from the
          second call that finds them unchanged, their device copy is kept and every later
          call binds it instead of converting them again. Any access to a tensor that may
          change its values (non-const host(), host_write_only(), set_size(), see
          gpu_data::copy_mark()) makes that copy outdated, and the next call writes it again.
          This is what lets a product with few rows, such as the token by token inference of a
          language model, gain from the device, whose cost is then that of the activations.
          As with the device copy of a CUDA build, a write through a pointer obtained before
          the call that took the copy goes unseen: code asks the tensor for its pointer again
          before writing, as dlib does everywhere.

    Arithmetic
        An NPU computes in half precision, and so does a GPU by default. The module converts
        the operands on the host, where it scales those that enter the result linearly by a
        power of two chosen from their largest magnitude, so that neither overflow nor
        underflow occurs, and it undoes the scaling, in single precision, when it writes the
        result. The rows of a softmax argument are brought to a maximum of zero before their
        conversion, which leaves the result as it is and keeps the resolution of the values
        near the maximum, which carry the probability. A result the device returns with an
        infinity or a NaN that the host would not have produced is discarded and the host
        computes the operation instead. Accumulations into an existing tensor (beta and add_to
        arguments) are always done on the host in single precision.

        Operations whose result must not round run only on a device that computes in single
        precision, which an NPU never does: those that update a persistent state in place,
        such as the optimizer moments or an embedding table, threshold(), whose comparison would
        move with the rounding, divide_by_sqrt(), whose documented identity is exact, and the
        matrix inversion. A linear combination that assigns one of its own operands, as a solver
        step does with the parameters, is held to the same rule. Operations that only move
        data, such as copies, permutations and gathers, follow it as well and moreover stay on
        the host unless routing is forced, since the device would add two copies to them and
        no arithmetic. Random number generation always stays on the host, so that a seeded
        sequence does not depend on the hardware present.

    Environment variables, read once at initialization
        DLIB_NPU            0, off or false disables the module; nothing is initialized.
        DLIB_NPU_DEVICE     comma separated OpenVINO devices to use, "NPU,GPU" by default:
                            those found are used, in that order of preference for forced
                            routing. GPU is the first Intel GPU, GPU.1 the second one, and CPU
                            runs the device paths on the OpenVINO CPU plugin, which validates
                            them on a machine without an NPU and can beat the host code on some
                            operations. A name may end with :f16 or :f32, which sets the
                            precision of that device, e.g. "NPU,GPU:f32".
        DLIB_NPU_ROUTING    auto (default), always or never. See set_routing().
        DLIB_NPU_PRECISION  f16 or f32: the element type exchanged with the CPU and GPU devices
                            and the inference precision requested from them. f16 by default,
                            f32 on CPU.
        DLIB_NPU_CACHE_DIR  directory of the compiled graph cache, dlib/npu_cache under the user
                            cache folder by default (LOCALAPPDATA on Windows, XDG_CACHE_HOME or
                            ~/.cache elsewhere). An empty value disables it.
        DLIB_NPU_OPS        comma separated operation names to enable in auto routing, or to
                            disable when prefixed with '-', e.g. "split_heads,-softmax". A name
                            listed here is enabled whatever the precision of the device, for the
                            operations that must not round too, but a linear combination that
                            assigns one of its operands still stays on the host in half
                            precision. print_status() gives the names.
        DLIB_NPU_SYNC       1 compiles a graph on the call that needs it instead of in the
                            background.
        DLIB_NPU_MEMORY_MB  budget of the buffers held by compiled graphs, 2048 by default.
        DLIB_NPU_RESIDENT_MB
                            budget of the device copies of unchanged operands, 2048 by
                            default; 0 converts every operand at every call.
        DLIB_NPU_VERBOSE    1 reports the devices and, at exit, the routing summary; 2 also
                            reports every compilation; 3 also writes a line before each stage
                            of every compilation and every device call, so that the last line
                            names the graph and the stage at which a plugin or a driver stopped
                            the program.
*/

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>
#include "operation_mode.h"

namespace dlib
{
    class tensor;
    class resizable_tensor;
    class rectangle;

    namespace npu
    {

    // ----------------------------------------------------------------------------------------
    //                         Run-time control, present in every build
    // ----------------------------------------------------------------------------------------

        bool is_compiled_in (
        );
        /*!
            ensures
                - returns true if dlib was built with DLIB_USE_NPU, and false otherwise.
        !*/

        bool is_available (
        );
        /*!
            ensures
                - returns true if an inference device was found and routing is not disabled.
                - The first call initializes the runtime. It is otherwise done by the first
                  tensor operation that qualifies for the device.
        !*/

        std::string device_name (
        );
        /*!
            ensures
                - returns the OpenVINO name and the full name of each device in use, separated
                  by " + ", or an empty string if no device is in use.
        !*/

        std::vector<std::string> device_names (
        );
        /*!
            ensures
                - returns the OpenVINO names of the devices in use, in the order of
                  DLIB_NPU_DEVICE, which is also the order of op_statistics::calls_by_device. A
                  device listed twice, with two precisions, carries its precision as well, e.g.
                  CPU:f16 and CPU:f32.
        !*/

        bool uses_half_precision (
        );
        /*!
            ensures
                - returns true if a device in use computes in half precision, and false
                  otherwise. An NPU always does; the CPU and GPU devices do unless their
                  precision is f32, the default on CPU.
        !*/

        bool has_single_precision_device (
        );
        /*!
            ensures
                - returns true if a device in use computes in single precision, and false
                  otherwise. The operations that must not round may run on such a device.
        !*/

        void restrict_to_device (
            int index
        );
        /*!
            ensures
                - if (index >= 0) then
                    - only the device device_names()[index] receives operations, in automatic
                      and in forced routing, and an index past the last device leaves them all
                      to the host. With routing::always, this times a given device, as the
                      offload example does.
                - else
                    - every device in use receives operations.
        !*/

        std::string runtime_version (
        );
        /*!
            ensures
                - returns the version of the OpenVINO runtime in use, or an empty string in a
                  build without DLIB_USE_NPU.
        !*/

        enum class routing
        {
            automatic, // each shape runs where it was measured to run fastest
            always,    // every operation runs on the first device that supports it, once compiled
            never      // every operation runs on the host
        };

        void set_routing (
            routing mode
        );
        /*!
            ensures
                - #get_routing() == mode
                - always also lifts the size thresholds and includes the data movement
                  operations. It is meant for tests and measurements.
                - The operations that must not round, data movement included, are never
                  routed to a device that computes in half precision, whatever the mode,
                  unless DLIB_NPU_OPS names them.
        !*/

        routing get_routing (
        );
        /*!
            ensures
                - returns the routing mode in force.
        !*/

        void set_synchronous_compilation (
            bool synchronous
        );
        /*!
            ensures
                - if (synchronous) then
                    - a graph is compiled on the call that first needs it, which then runs on
                      the device. Results then depend on nothing but the routing mode, which is
                      what tests need.
                - else
                    - graphs are compiled by a background thread while the host computes.
        !*/

        class exact_scope
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    While an exact_scope exists, the operations called by the thread that
                    created it run on the host, with single precision arithmetic, as in a build
                    without DLIB_USE_NPU. Scopes nest. Code whose decisions must not depend on
                    the hardware present, such as an arithmetic coder or the detection of
                    personal data, runs its inference inside one.
            !*/
        public:
            exact_scope();
            ~exact_scope();
            exact_scope(const exact_scope&) = delete;
            exact_scope& operator=(const exact_scope&) = delete;
        };

        struct op_statistics
        {
            std::string name;
            unsigned long long device_calls = 0;     // calls computed by a device
            std::vector<unsigned long long> calls_by_device; // the same, per device_names()
            unsigned long long host_calls = 0;       // qualifying calls left to the host
            unsigned long long rejected = 0;         // device results discarded by the guards
            unsigned long long graphs = 0;           // graphs compiled for this operation
            unsigned long long compile_failures = 0; // graphs the device could not compile
            unsigned long long reused_calls = 0;     // device calls that reused operand copies
            double device_seconds = 0;               // time spent in device calls
            double compile_seconds = 0;              // time spent compiling
        };

        std::vector<op_statistics> get_statistics (
        );
        /*!
            ensures
                - returns one entry per operation that qualified for the device at least once.
        !*/

        void reset_statistics (
        );
        /*!
            ensures
                - clears the counters and the routing decisions taken so far, which the next
                  calls take again.
        !*/

        void print_status (
            std::ostream& out
        );
        /*!
            ensures
                - writes the build configuration, the devices in use, the routing settings and
                  the statistics to out.
        !*/

    // ----------------------------------------------------------------------------------------
    //                    Device paths, present in builds with DLIB_USE_NPU
    // ----------------------------------------------------------------------------------------

#ifdef DLIB_USE_NPU

        /*
            Each function below takes the arguments of the tt:: function of the same name and
            has its semantics. It returns true if it computed the operation and false if the
            caller has to run the host path, which then must follow immediately in the same
            thread, timed by a cpu_measure, as DLIB_NPU_TRY in tensor_tools.cpp does.
        */

        class cpu_measure
        {
            /*!
                WHAT THIS OBJECT REPRESENTS
                    Times the host path that follows a refused device call, when the routing of
                    that call asked for a host measurement. Costs nothing otherwise.
            !*/
        public:
            cpu_measure();
            ~cpu_measure();
            cpu_measure(const cpu_measure&) = delete;
            cpu_measure& operator=(const cpu_measure&) = delete;
        private:
            void* entry;
            long long start;
        };

        bool inverse_norms (resizable_tensor& invnorms, const tensor& data, const double eps);
        bool inverse_sqrt (tensor& dest, const tensor& src, const double eps);
        bool divide_by_sqrt (tensor& dest, const tensor& num, const tensor& sqnorm, const double eps);
        bool dot_prods (resizable_tensor& out, const tensor& lhs, const tensor& rhs);
        bool dot_prods (bool add_to, tensor& out, const tensor& lhs, const tensor& rhs);
        bool scale_columns (tensor& out, const tensor& m, const tensor& v);
        bool scale_rows (tensor& out, const tensor& m, const tensor& v);
        bool scale_rows2 (float beta, tensor& out, const tensor& m1, const tensor& m2,
            const tensor& v1, const tensor& v2);
        bool exp (tensor& dest, const tensor& src);
        bool log (tensor& dest, const tensor& src);
        bool log10 (tensor& dest, const tensor& src);

        bool gemm (float beta, tensor& dest, float alpha, const tensor& lhs, bool trans_lhs,
            const tensor& rhs, bool trans_rhs, operation_mode mode);

        bool multiply (bool add_to, tensor& dest, const tensor& src1, const tensor& src2);
        bool scale_channels (bool add_to, tensor& dest, const tensor& src, const tensor& scales);
        bool multiply_conv (bool add_to, tensor& dest, const tensor& src1, const tensor& src2);
        bool multiply_zero_padded (bool add_to, tensor& dest, const tensor& src1, const tensor& src2);

        bool affine_transform (tensor& dest, const tensor& src, const float A, const float B);
        bool affine_transform (tensor& dest, const tensor& src, const float A);
        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const float A, const float B, const float C);
        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const float A, const float B);
        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const tensor& src3, const float A, const float B, const float C, const float D);
        bool affine_transform_range (size_t begin, size_t end, tensor& dest, const tensor& src1,
            const tensor& src2, const tensor& src3, const float A, const float B, const float C);
        bool affine_transform (const rectangle& rect, tensor& dest, const tensor& src1,
            const tensor& src2, const tensor& src3, float A, float B, float C);
        bool affine_transform (tensor& dest, const tensor& src1, const tensor& src2,
            const tensor& src3, const float A, const float B, const float C);
        bool affine_transform (tensor& dest, const tensor& src, const tensor& A, const tensor& B);
        bool affine_transform_conv (tensor& dest, const tensor& src, const tensor& A, const tensor& B);

        bool compute_adam_update (size_t begin, size_t end, tensor& s, tensor& m, tensor& v,
            const float t, const float learning_rate, const float weight_decay,
            const float momentum1, const float momentum2, const tensor& params,
            const tensor& params_grad);
        bool compute_adam_atan2_update (size_t begin, size_t end, tensor& s, tensor& m, tensor& v,
            const float t, const float learning_rate, const float weight_decay,
            const float momentum1, const float momentum2, const tensor& params,
            const tensor& params_grad);

        bool batch_normalize_inference (const double eps, resizable_tensor& dest, const tensor& src,
            const tensor& gamma, const tensor& beta, const tensor& running_means,
            const tensor& running_variances);
        bool batch_normalize (const double eps, resizable_tensor& dest, resizable_tensor& means,
            resizable_tensor& invstds, const double averaging_factor,
            resizable_tensor& running_means, resizable_tensor& running_variances,
            const tensor& src, const tensor& gamma, const tensor& beta);
        bool batch_normalize_gradient (const double eps, const tensor& gradient_input,
            const tensor& means, const tensor& invstds, const tensor& src, const tensor& gamma,
            tensor& src_grad, tensor& gamma_grad, tensor& beta_grad);
        bool batch_normalize_conv_inference (const double eps, resizable_tensor& dest,
            const tensor& src, const tensor& gamma, const tensor& beta,
            const tensor& running_means, const tensor& running_variances);
        bool batch_normalize_conv (const double eps, resizable_tensor& dest, resizable_tensor& means,
            resizable_tensor& invstds, const double averaging_factor,
            resizable_tensor& running_means, resizable_tensor& running_variances,
            const tensor& src, const tensor& gamma, const tensor& beta);
        bool batch_normalize_conv_gradient (const double eps, const tensor& gradient_input,
            const tensor& means, const tensor& invstds, const tensor& src, const tensor& gamma,
            tensor& src_grad, tensor& gamma_grad, tensor& beta_grad);

        bool layer_normalize (const double eps, resizable_tensor& dest, resizable_tensor& means,
            resizable_tensor& invstds, const tensor& src, const tensor& gamma, const tensor& beta);
        bool layer_normalize_gradient (const double eps, const tensor& gradient_input,
            const tensor& means, const tensor& invstds, const tensor& src, const tensor& gamma,
            tensor& src_grad, tensor& gamma_grad, tensor& beta_grad,
            resizable_tensor& dmeans, resizable_tensor& dvars);
        bool rms_normalize (const double eps, resizable_tensor& dest, resizable_tensor& scale,
            const tensor& src, const tensor& gamma, bool axis_is_nc);
        bool rms_normalize_gradient (const tensor& gradient_input, const tensor& scale,
            const tensor& src, const tensor& gamma, tensor& src_grad, tensor& gamma_grad,
            resizable_tensor& dscale, bool axis_is_nc);

        bool threshold (tensor& data, float thresh);
        bool dot (const tensor& a, const tensor& b, tensor& result, size_t idx);
        bool add (float beta, tensor& dest, float alpha, const tensor& src);
        bool add (tensor& dest, const tensor& src1, const tensor& src2);
        bool assign_conv_bias_gradient (tensor& grad, const tensor& gradient_input);
        bool assign_bias_gradient (tensor& grad, const tensor& gradient_input);

        bool softmax (tensor& dest, const tensor& src, operation_mode mode);
        bool softmax_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input,
            operation_mode mode);
        bool softmax_all (tensor& dest, const tensor& src);
        bool softmax_all_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input);

        bool sigmoid (tensor& dest, const tensor& src);
        bool sigmoid_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input);
        bool mish (tensor& dest, const tensor& src);
        bool mish_gradient (tensor& grad, const tensor& src, const tensor& gradient_input);
        bool relu (tensor& dest, const tensor& src);
        bool relu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input);
        bool prelu (tensor& dest, const tensor& src, const tensor& param);
        bool prelu_gradient (tensor& grad, const tensor& src, const tensor& gradient_input,
            const tensor& param, tensor& params_grad);
        bool leaky_relu (tensor& dest, const tensor& src, const float alpha);
        bool leaky_relu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input,
            const float alpha);
        bool tanh (tensor& dest, const tensor& src);
        bool tanh_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input);
        bool clipped_relu (tensor& dest, const tensor& src, const float ceiling);
        bool clipped_relu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input,
            const float ceiling);
        bool elu (tensor& dest, const tensor& src, const float alpha);
        bool elu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input,
            const float alpha);
        bool gelu (tensor& dest, const tensor& src);
        bool gelu_gradient (tensor& grad, const tensor& src, const tensor& gradient_input);
        bool smelu (tensor& dest, const tensor& src, const float beta);
        bool smelu_gradient (tensor& grad, const tensor& dest, const tensor& gradient_input,
            const float beta);
        bool silu (tensor& dest, const tensor& src);
        bool silu_gradient (tensor& grad, const tensor& src, const tensor& gradient_input);

        bool resize_bilinear (tensor& dest, long dest_row_stride, long dest_channel_stride,
            const tensor& src, long src_row_stride, long src_channel_stride);
        bool resize_bilinear_gradient (tensor& grad, long grad_row_stride, long grad_channel_stride,
            const tensor& gradient_input, long gradient_input_row_stride,
            long gradient_input_channel_stride);
        bool reorg (bool add_to, tensor& dest, const int row_stride, const int col_stride,
            const tensor& src);
        bool reorg_gradient (bool add_to, tensor& grad, const int row_stride, const int col_stride,
            const tensor& gradient_input);

        bool copy_tensor (bool add_to, tensor& dest, size_t dest_k_offset, const tensor& src,
            size_t src_k_offset, size_t count_k);
        bool copy_tensor (bool add_to, tensor& dest, size_t dk, size_t dnr, size_t dnc,
            const tensor& src, size_t sk, size_t snr, size_t snc, size_t k, size_t nr, size_t nc);

        bool inv (const tensor& m, resizable_tensor& out);
        bool transpose (bool add_to, tensor& dest, const tensor& src);

        bool embeddings (resizable_tensor& dest, const tensor& src, const tensor& embs);
        bool embeddings_gradient (const tensor& prev, const tensor& gradient_input, tensor& grads,
            const tensor& freqs, float learning_rate, bool scale);
        bool puzzle_embeddings (resizable_tensor& dest, const tensor& tokens,
            const tensor& token_table, const tensor& puzzle_table, float scale);
        bool puzzle_embeddings_gradient (const tensor& tokens, const tensor& gradient_input,
            tensor& token_grad, tensor& puzzle_table, float scale, float puzzle_rate,
            float puzzle_decay);

        bool repeat_channels (tensor& dest, const tensor& src, long repeat_factor);
        bool accumulate_repeated_channels (tensor& dest, const tensor& src, long repeat_factor);

        bool compute_halt_logits (tensor& logits, tensor& halt_probs, const tensor& state,
            const tensor& halt_params, const tensor& active_mask, long batch_size, long seq_len,
            long num_channels, long feature_dim);
        bool compute_halt_gradient (tensor& grad_params, tensor& grad_state,
            const tensor& gradient_input, const tensor& state, const tensor& halt_probs,
            const tensor& halt_params, const tensor& active_mask, const tensor& n_steps,
            const tensor& final_state, long current_step, long batch_size, long seq_len,
            long num_channels, long feature_dim, float ponder_penalty, bool gradient_check_mode);
        bool accumulate_weighted_state (tensor& output, const tensor& state, const tensor& weights,
            const tensor& active_mask, long batch_size, long seq_len, long num_channels,
            long feature_dim);
        bool apply_act_depth_scaling (tensor& gradients, const tensor& n_steps, long batch_size,
            long seq_len, long num_channels, long feature_dim, long max_steps);
        bool broadcast_row (tensor& dest, const tensor& row);
        bool act_mark_active (tensor& active_mask, tensor& active_count,
            const tensor& cumulative_halting, float halt_threshold);
        bool act_update_halting (tensor& cumulative_halting, tensor& remainders, tensor& n_steps,
            tensor& step_weights, const tensor& halt_probs, const tensor& active_mask, long step,
            long max_steps, float halt_threshold);
        bool act_gather_final_state (tensor& final_state, const tensor& step_state,
            const tensor& n_steps, long step, long batch_size, long seq_len, long num_channels,
            long feature_dim);
        bool act_accumulate_output_grad (tensor& grad_state, const tensor& gradient_input,
            const tensor& step_weights, const tensor& active_mask, const tensor& n_steps, long step,
            long batch_size, long seq_len, long num_channels, long feature_dim);

        bool apply_rotary_positional_embedding (bool is_backward, resizable_tensor& data,
            const resizable_tensor& cos_cache, const resizable_tensor& sin_cache);
        bool split_heads (bool add_to, tensor& dst, const tensor& src);
        bool merge_heads (bool add_to, tensor& dst, const tensor& src);

        /*
            Convolution and pooling, with the geometry the tt::tensor_conv and tt::pooling
            objects hold after setup.
        */
        bool conv (bool add_to_output, tensor& output, const tensor& data, const tensor& filters,
            int stride_y, int stride_x, int padding_y, int padding_x);
        bool conv_gradient_for_data (bool add_to_output, const tensor& gradient_input,
            const tensor& filters, tensor& data_gradient, int stride_y, int stride_x,
            int padding_y, int padding_x);
        bool conv_gradient_for_filters (bool add_to_output, const tensor& gradient_input,
            const tensor& data, tensor& filters_gradient, int stride_y, int stride_x,
            int padding_y, int padding_x);
        bool pooling (bool max_pooling, int window_height, int window_width, int stride_y,
            int stride_x, int padding_y, int padding_x, tensor& dest, const tensor& src);
        bool pooling_gradient (bool max_pooling, int window_height, int window_width,
            int stride_y, int stride_x, int padding_y, int padding_x,
            const tensor& gradient_input, const tensor& dest, const tensor& src, tensor& grad);

        /*
            Cores of the loss layers. The host classes of cpu_dlib.h select the positions to
            score from their labels and hand them over here.
        */
        bool cross_entropy_rows (const tensor& logits, tensor& grad,
            const std::vector<long>& rows, const std::vector<unsigned long>& targets,
            double label_smoothing, double z_loss_weight, double& loss);
        /*!
            requires
                - logits.k() == 1, logits holds num_samples()*nr() rows of nc() logits.
                - rows.size() == targets.size(), every row < num_samples()*nr(), every target
                  < nc(), and grad holds zeros at the rows not listed.
            ensures
                - if (it returns true) then
                    - computes what compute_loss_cross_entropy_per_token computes for the
                      positions listed: the mean over them of the cross entropy with label
                      smoothing and z-loss, into loss, and its gradient into grad.
        !*/

        bool mean_squared_per_pixel (const tensor& output,
            const std::function<void(float*)>& fill_targets, tensor& grad, double& loss);
        /*!
            ensures
                - fill_targets(t) is called at most once, when the device is chosen, to write
                  output.size() targets in the layout of output.
                - if (it returns true) then
                    - computes what compute_loss_mean_squared_per_channel_and_pixel computes
                      from those targets.
        !*/

#endif // DLIB_USE_NPU

    }
}

#endif // DLIB_DNN_NPU_DLIB_H_
