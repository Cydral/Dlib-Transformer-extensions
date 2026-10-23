// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/dnn.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "tester.h"

/*
    Tests of the routing of tensor operations to an NPU (dlib/cuda/npu_dlib.h).

    The control functions are checked in every build. The device paths are checked when dlib
    is built with DLIB_USE_NPU, without CUDA, and a device is found: every operation is run
    once on the host and once forced on the device from the same operands, and the results
    are compared with a tolerance set by the precision the device computes in.

    A machine without an NPU runs the device paths on the OpenVINO CPU plugin:
        DLIB_NPU_DEVICE=CPU ./dtest --test_npu                         (single precision)
        DLIB_NPU_DEVICE=CPU DLIB_NPU_PRECISION=f16 ./dtest --test_npu  (half precision)

    On an NPU or a GPU, an operation whose graph the device compiler refuses is left to the
    host and reported in the log (dtest -d writes it to debug.txt); with DLIB_NPU_VERBOSE=1,
    the summary printed at exit counts these graphs in its failed column.
*/

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.npu");

// ----------------------------------------------------------------------------------------

    void test_control_api (
    )
    {
        print_spinner();
#ifdef DLIB_USE_NPU
        DLIB_TEST(npu::is_compiled_in());
        const npu::routing previous = npu::get_routing();
        npu::set_routing(npu::routing::never);
        DLIB_TEST(npu::get_routing() == npu::routing::never);
        npu::set_routing(npu::routing::always);
        DLIB_TEST(npu::get_routing() == npu::routing::always);
        npu::set_routing(previous);
#else
        DLIB_TEST(!npu::is_compiled_in());
        DLIB_TEST(!npu::is_available());
        DLIB_TEST(npu::runtime_version().empty());
        DLIB_TEST(npu::get_routing() == npu::routing::never);
        npu::set_routing(npu::routing::always);
        DLIB_TEST(npu::get_routing() == npu::routing::never);
        DLIB_TEST(npu::get_statistics().empty());
#endif
        if (npu::is_available())
        {
            DLIB_TEST(!npu::device_name().empty());
        }
        else
        {
            DLIB_TEST(npu::device_name().empty());
            DLIB_TEST(!npu::uses_half_precision());
        }

        {
            npu::exact_scope outer;
            npu::exact_scope inner;
        }
        npu::reset_statistics();

        std::ostringstream sout;
        npu::print_status(sout);
        DLIB_TEST(sout.str().find("dlib NPU routing") == 0);
        dlog << LINFO << "\n" << sout.str();
    }

// ----------------------------------------------------------------------------------------

#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)

    unsigned long long device_calls (
        const std::string& op
    )
    {
        for (const auto& s : npu::get_statistics())
            if (s.name == op)
                return s.device_calls;
        return 0;
    }

    unsigned long long rejected_calls (
        const std::string& op
    )
    {
        for (const auto& s : npu::get_statistics())
            if (s.name == op)
                return s.rejected;
        return 0;
    }

    unsigned long long refused_graphs (
        const std::string& op
    )
    {
        for (const auto& s : npu::get_statistics())
            if (s.name == op)
                return s.compile_failures;
        return 0;
    }

    unsigned long long reused_calls (
        const std::string& op
    )
    {
        for (const auto& s : npu::get_statistics())
            if (s.name == op)
                return s.reused_calls;
        return 0;
    }

    /*
        The compiler of an NPU or of a GPU may refuse a graph that the CPU plugin accepts. The
        host then computes the operation, as the routing does in every mode, and print_status()
        counts the graph among the failed ones. On the CPU plugin every graph must compile.
    */
    bool refused_by_device (
        const std::string& op
    )
    {
        return npu::device_name().compare(0, 3, "CPU") != 0 && refused_graphs(op) > 0;
    }

    // The largest difference relative to the largest magnitude of the expected values. NaN and
    // infinities must sit at the same places with the same values.
    double relative_error (
        const tensor& expected,
        const tensor& actual
    )
    {
        const double inf = std::numeric_limits<double>::infinity();
        if (expected.size() != actual.size())
            return inf;
        const float* a = expected.host();
        const float* b = actual.host();
        double scale = 0, err = 0;
        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (std::isnan(a[i]) || std::isnan(b[i]))
            {
                if (std::isnan(a[i]) != std::isnan(b[i]))
                    return inf;
                continue;
            }
            if (std::isinf(a[i]) || std::isinf(b[i]))
            {
                if (a[i] != b[i])
                    return inf;
                continue;
            }
            scale = std::max(scale, static_cast<double>(std::abs(a[i])));
            err = std::max(err, static_cast<double>(std::abs(a[i] - b[i])));
        }
        return err == 0 ? 0 : err / std::max(scale, 1e-30);
    }

    void randomize (
        tensor& t,
        dlib::rand& rnd,
        float lo = -2,
        float hi = 2
    )
    {
        float* p = t.host();
        for (size_t i = 0; i < t.size(); ++i)
            p[i] = lo + (hi - lo) * rnd.get_random_float();
    }

    resizable_tensor random_tensor (
        dlib::rand& rnd,
        long n,
        long k,
        long nr,
        long nc,
        float lo = -2,
        float hi = 2
    )
    {
        resizable_tensor t(n, k, nr, nc);
        randomize(t, rnd, lo, hi);
        return t;
    }

    // Integer values below limit, as token ids and step counts are stored.
    void fill_integers (
        tensor& t,
        dlib::rand& rnd,
        unsigned long limit,
        unsigned long first = 0
    )
    {
        float* p = t.host();
        for (size_t i = 0; i < t.size(); ++i)
            p[i] = static_cast<float>(first + rnd.get_random_32bit_number() % (limit - first));
    }

    // Distinct values, exact in half precision, so that a maximum never depends on rounding.
    void fill_distinct (
        tensor& t,
        dlib::rand& rnd
    )
    {
        std::vector<long> order(t.size());
        for (size_t i = 0; i < order.size(); ++i)
            order[i] = static_cast<long>(i);
        for (size_t i = order.size(); i > 1; --i)
            std::swap(order[i - 1], order[rnd.get_random_32bit_number() % i]);
        float* p = t.host();
        const long half = static_cast<long>(t.size() / 2);
        for (size_t i = 0; i < t.size(); ++i)
            p[i] = static_cast<float>(order[i] - half) / 64.0f;
    }

    // Whether the operations that must not round may run on the device in use.
    bool exact_on_device (
    )
    {
        return !npu::uses_half_precision();
    }

    /*
        Runs compute twice from the same values of the tensors in io, which it reads or writes:
        on the host, then with every operation forced on the device. Checks that op ran on the
        device the second time, or did not if on_device is false, and that every tensor of io
        ends with the same values both times, up to scale times the tolerance of the
        precision the device computes in.
    */
    template <typename F>
    void check (
        const std::string& op,
        const std::vector<resizable_tensor*>& io,
        F compute,
        double scale = 1,
        bool on_device = true
    )
    {
        print_spinner();
        std::vector<resizable_tensor> start(io.size()), host(io.size());
        for (size_t i = 0; i < io.size(); ++i)
            start[i] = *io[i];

        npu::set_routing(npu::routing::never);
        compute();
        for (size_t i = 0; i < io.size(); ++i)
        {
            host[i] = *io[i];
            *io[i] = start[i];
        }

        const unsigned long long before = device_calls(op);
        npu::set_routing(npu::routing::always);
        compute();
        npu::set_routing(npu::routing::never);
        const bool ran = device_calls(op) > before;
        if (on_device && !ran && refused_by_device(op))
            dlog << LWARN << op << ": the device refused the graph, the host computed it";
        else if (on_device)
            DLIB_TEST_MSG(ran, op << " did not run on the device");
        else
            DLIB_TEST_MSG(!ran, op << " ran on the device although it must not round");

        const double tolerance = scale * (npu::uses_half_precision() ? 2e-2 : 1e-4);
        for (size_t i = 0; i < io.size(); ++i)
        {
            const double err = relative_error(host[i], *io[i]);
            DLIB_TEST_MSG(err <= tolerance, op << ": tensor " << i << " differs by " << err
                << " (tolerance " << tolerance << ")");
        }
    }

// ----------------------------------------------------------------------------------------

    void test_elementwise (
        dlib::rand& rnd
    )
    {
        // Sizes that are not multiples of anything the device prefers.
        resizable_tensor x = random_tensor(rnd, 3, 5, 7, 11, -3, 3);
        resizable_tensor positive = random_tensor(rnd, 3, 5, 7, 11, 0.01f, 4);
        resizable_tensor gi = random_tensor(rnd, 3, 5, 7, 11);
        resizable_tensor y(3, 5, 7, 11), g = random_tensor(rnd, 3, 5, 7, 11), d(3, 5, 7, 11);

        check("sigmoid", {&y}, [&]{ tt::sigmoid(y, x); });
        check("tanh", {&y}, [&]{ tt::tanh(y, x); });
        check("relu", {&y}, [&]{ tt::relu(y, x); });
        check("mish", {&y}, [&]{ tt::mish(y, x); });
        check("gelu", {&y}, [&]{ tt::gelu(y, x); });
        check("silu", {&y}, [&]{ tt::silu(y, x); });
        check("elu", {&y}, [&]{ tt::elu(y, x, 0.7f); });
        check("leaky_relu", {&y}, [&]{ tt::leaky_relu(y, x, 0.1f); });
        check("clipped_relu", {&y}, [&]{ tt::clipped_relu(y, x, 1.5f); });
        check("smelu", {&y}, [&]{ tt::smelu(y, x, 1.0f); });
        check("exp", {&y}, [&]{ tt::exp(y, x); });
        check("log", {&y}, [&]{ tt::log(y, positive); });
        check("log10", {&y}, [&]{ tt::log10(y, positive); });
        check("inverse_sqrt", {&y}, [&]{ tt::inverse_sqrt(y, positive, 1e-5); });

        // Gradients added to an existing gradient, then assigned when grad is gradient_input.
        npu::set_routing(npu::routing::never);
        tt::sigmoid(d, x);
        check("sigmoid_gradient", {&g}, [&]{ tt::sigmoid_gradient(g, d, gi); });
        tt::tanh(d, x);
        check("tanh_gradient", {&g}, [&]{ tt::tanh_gradient(g, d, gi); });
        tt::relu(d, x);
        check("relu_gradient", {&g}, [&]{ tt::relu_gradient(g, d, gi); });
        resizable_tensor same = gi;
        check("relu_gradient", {&same}, [&]{ tt::relu_gradient(same, d, same); });
        check("mish_gradient", {&g}, [&]{ tt::mish_gradient(g, x, gi); });
        check("gelu_gradient", {&g}, [&]{ tt::gelu_gradient(g, x, gi); });
        check("silu_gradient", {&g}, [&]{ tt::silu_gradient(g, x, gi); });
        tt::elu(d, x, 0.7f);
        check("elu_gradient", {&g}, [&]{ tt::elu_gradient(g, d, gi, 0.7f); });
        tt::leaky_relu(d, x, 0.1f);
        check("leaky_relu_gradient", {&g}, [&]{ tt::leaky_relu_gradient(g, d, gi, 0.1f); });
        tt::clipped_relu(d, x, 1.5f);
        check("clipped_relu_gradient", {&g}, [&]{ tt::clipped_relu_gradient(g, d, gi, 1.5f); });
        tt::smelu(d, x, 1.0f);
        check("smelu_gradient", {&g}, [&]{ tt::smelu_gradient(g, d, gi, 1.0f); });

        resizable_tensor param(1);
        param.host()[0] = 0.2f;
        resizable_tensor param_grad(1);
        check("prelu", {&y}, [&]{ tt::prelu(y, x, param); });
        check("prelu_gradient", {&g, &param_grad}, [&]{ tt::prelu_gradient(g, x, gi, param, param_grad); }, 2);

        resizable_tensor mask = x;
        check("threshold", {&mask}, [&]{ tt::threshold(mask, 0.3f); }, 1, exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_arithmetic (
        dlib::rand& rnd
    )
    {
        resizable_tensor a = random_tensor(rnd, 4, 3, 9, 13);
        resizable_tensor b = random_tensor(rnd, 4, 3, 9, 13);
        resizable_tensor c = random_tensor(rnd, 4, 3, 9, 13);
        resizable_tensor d = random_tensor(rnd, 4, 3, 9, 13);

        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, 1.5f, -0.25f); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, -0.5f); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, b, 0.5f, 2.0f, 0.1f); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, b, 0.5f, -2.0f); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, b, c, 1, 2, 3, 0.5f); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, b, c, 1, -2, 3); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform_range(17, d.size() - 31, d, a, b, c, 1, 2, 3); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform(rectangle(5, 1, 70, 2), d, a, b, c, 1, 2, 3); });
        resizable_tensor A1 = random_tensor(rnd, 1, 3, 9, 13), B1 = random_tensor(rnd, 1, 3, 9, 13);
        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, A1, B1); });
        check("affine_transform", {&d}, [&]{ tt::affine_transform(d, a, b, c); });
        resizable_tensor Ak = random_tensor(rnd, 1, 3, 1, 1), Bk = random_tensor(rnd, 1, 3, 1, 1);
        check("affine_transform_conv", {&d}, [&]{ tt::affine_transform_conv(d, a, Ak, Bk); });

        check("multiply", {&d}, [&]{ tt::multiply(false, d, a, b); });
        check("multiply", {&d}, [&]{ tt::multiply(true, d, a, b); });
        resizable_tensor b1 = random_tensor(rnd, 1, 3, 9, 13), d1 = random_tensor(rnd, 1, 3, 9, 13);
        check("multiply", {&d}, [&]{ tt::multiply(false, d, a, b1); });
        check("multiply", {&d1}, [&]{ tt::multiply(true, d1, a, b); }, 2);
        resizable_tensor scales = random_tensor(rnd, 4, 3, 1, 1);
        check("scale_channels", {&d}, [&]{ tt::scale_channels(false, d, a, scales); });
        check("scale_channels", {&d}, [&]{ tt::scale_channels(true, d, a, scales); });
        resizable_tensor dk = random_tensor(rnd, 1, 3, 1, 1);
        check("multiply_conv", {&d}, [&]{ tt::multiply_conv(false, d, a, Ak); });
        check("multiply_conv", {&dk}, [&]{ tt::multiply_conv(true, dk, a, b); }, 3);
        resizable_tensor small = random_tensor(rnd, 4, 2, 9, 10);
        check("multiply_zero_padded", {&d}, [&]{ tt::multiply_zero_padded(false, d, a, small); });
        check("multiply_zero_padded", {&d}, [&]{ tt::multiply_zero_padded(true, d, a, small); });

        check("add", {&d}, [&]{ tt::add(0.5f, d, 2.0f, b); });
        resizable_tensor bias_k = random_tensor(rnd, 1, 3, 1, 1), bias_krc = random_tensor(rnd, 1, 3, 9, 13);
        resizable_tensor bias_rc = random_tensor(rnd, 1, 1, 9, 13), bias_n = random_tensor(rnd, 4, 1, 1, 1);
        check("add", {&d}, [&]{ tt::add(1.0f, d, 1.0f, bias_k); });
        check("add", {&d}, [&]{ tt::add(0.0f, d, 1.0f, bias_krc); });
        check("add", {&d}, [&]{ tt::add(1.0f, d, -1.0f, bias_rc); });
        check("add", {&d}, [&]{ tt::add(2.0f, d, 1.0f, bias_n); });
        check("add", {&d}, [&]{ tt::add(d, a, small); });

        // A tensor updated in place, as a parameter by its solver step, keeps single precision.
        resizable_tensor p = random_tensor(rnd, 4, 3, 9, 13), step = random_tensor(rnd, 4, 3, 9, 13, -1e-3f, 1e-3f);
        check("add", {&p}, [&]{ tt::add(p, p, step); }, 1, exact_on_device());
        check("affine_transform", {&p}, [&]{ tt::affine_transform(p, p, step, 0.9f, -0.1f); }, 1, exact_on_device());

        resizable_tensor result = random_tensor(rnd, 5, 1, 1, 1);
        check("dot", {&result}, [&]{ tt::dot(a, b, result, 3); }, 3);

        resizable_tensor m1 = random_tensor(rnd, 37, 1, 1, 50), m2 = random_tensor(rnd, 37, 1, 1, 50);
        resizable_tensor out = random_tensor(rnd, 37, 1, 1, 50), dots, dots2 = random_tensor(rnd, 37, 1, 1, 1);
        check("dot_prods", {&dots}, [&]{ tt::dot_prods(dots, m1, m2); }, 3);
        check("dot_prods", {&dots2}, [&]{ tt::dot_prods(true, dots2, m1, m2); }, 3);
        resizable_tensor invnorms;
        check("inverse_norms", {&invnorms}, [&]{ tt::inverse_norms(invnorms, m1, 1e-6); });
        resizable_tensor vc = random_tensor(rnd, 1, 1, 1, 50), vr = random_tensor(rnd, 37, 1, 1, 1);
        resizable_tensor v2 = random_tensor(rnd, 37, 1, 1, 1);
        check("scale_columns", {&out}, [&]{ tt::scale_columns(out, m1, vc); });
        check("scale_rows", {&out}, [&]{ tt::scale_rows(out, m1, vr); });
        check("scale_rows2", {&out}, [&]{ tt::scale_rows2(0.5f, out, m1, m2, vr, v2); }, 2);
        check("scale_rows2", {&out}, [&]{ tt::scale_rows2(0.0f, out, m1, m2, vr, v2); }, 2);
        resizable_tensor sqnorm = random_tensor(rnd, 37, 1, 1, 50, 0.1f, 4);
        check("divide_by_sqrt", {&out}, [&]{ tt::divide_by_sqrt(out, m1, sqnorm, 1e-6); }, 1, exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_matrix (
        dlib::rand& rnd
    )
    {
        resizable_tensor L = random_tensor(rnd, 19, 1, 1, 37), Lt = random_tensor(rnd, 37, 1, 1, 19);
        resizable_tensor R = random_tensor(rnd, 37, 1, 1, 23), Rt = random_tensor(rnd, 23, 1, 1, 37);
        resizable_tensor D = random_tensor(rnd, 19, 1, 1, 23);
        check("gemm", {&D}, [&]{ tt::gemm(0, D, 1, L, false, R, false); }, 2);
        check("gemm", {&D}, [&]{ tt::gemm(0.5f, D, 2, Lt, true, R, false); }, 2);
        check("gemm", {&D}, [&]{ tt::gemm(0, D, 1, L, false, Rt, true); }, 2);
        check("gemm", {&D}, [&]{ tt::gemm(1, D, -1, Lt, true, Rt, true); }, 2);

        // The products of an attention head.
        resizable_tensor Q = random_tensor(rnd, 2, 3, 11, 16), K = random_tensor(rnd, 2, 3, 13, 16);
        resizable_tensor V = random_tensor(rnd, 2, 3, 13, 16), S(2, 3, 11, 13), O(2, 3, 11, 16), dV(2, 3, 13, 16);
        check("gemm", {&S}, [&]{ tt::gemm(0, S, 0.25f, Q, false, K, true, operation_mode::PLANE_WISE); }, 2);
        check("gemm", {&O}, [&]{ tt::gemm(0, O, 1, S, false, V, false, operation_mode::PLANE_WISE); }, 2);
        check("gemm", {&dV}, [&]{ tt::gemm(0, dV, 1, S, true, O, false, operation_mode::PLANE_WISE); }, 2);

        // One matrix shared by every plane, on either side.
        resizable_tensor W = random_tensor(rnd, 1, 1, 16, 7), QW(2, 3, 11, 7);
        resizable_tensor P = random_tensor(rnd, 1, 1, 5, 11), PQ(2, 3, 5, 16);
        check("gemm", {&QW}, [&]{ tt::gemm(0, QW, 1, Q, false, W, false, operation_mode::PLANE_WISE); }, 2);
        check("gemm", {&PQ}, [&]{ tt::gemm(0, PQ, 1, P, false, Q, false, operation_mode::PLANE_WISE); }, 2);

        // A long reduction, and operands far below the half precision range.
        resizable_tensor Llong = random_tensor(rnd, 9, 1, 1, 3000), Rlong = random_tensor(rnd, 3000, 1, 1, 5);
        resizable_tensor Dlong(9, 1, 1, 5);
        check("gemm", {&Dlong}, [&]{ tt::gemm(0, Dlong, 1, Llong, false, Rlong, false); }, 3);
        resizable_tensor tiny = L;
        tt::affine_transform(tiny, L, 1e-6f);
        check("gemm", {&D}, [&]{ tt::gemm(0, D, 1, tiny, false, R, false); }, 2);

        resizable_tensor M = random_tensor(rnd, 24, 24, 1, 1, -0.1f, 0.1f), Minv;
        for (long i = 0; i < 24; ++i)
            M.host()[i * 24 + i] += 2;
        tt::inv inverse;
        check("inv", {&Minv}, [&]{ inverse(M, Minv); }, 2, exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_softmax (
        dlib::rand& rnd
    )
    {
        resizable_tensor x = random_tensor(rnd, 3, 7, 5, 4, -4, 4), y(3, 7, 5, 4), d(3, 7, 5, 4);
        resizable_tensor gi = random_tensor(rnd, 3, 7, 5, 4), g = random_tensor(rnd, 3, 7, 5, 4);
        check("softmax", {&y}, [&]{ tt::softmax(y, x); });
        npu::set_routing(npu::routing::never);
        tt::softmax(d, x);
        check("softmax_gradient", {&g}, [&]{ tt::softmax_gradient(g, d, gi); }, 2);

        // Attention scores with a causal mask, a large negative mask and a fully masked row.
        resizable_tensor s = random_tensor(rnd, 2, 3, 9, 13, -4, 4), ys(2, 3, 9, 13), ds(2, 3, 9, 13);
        float* p = s.host();
        for (long plane = 0; plane < 6; ++plane)
            for (long r = 0; r < 9; ++r)
                for (long c = r + 5; c < 13; ++c)
                    p[(plane * 9 + r) * 13 + c] = -std::numeric_limits<float>::infinity();
        for (long c = 0; c < 13; ++c)
            p[c] = -std::numeric_limits<float>::infinity();
        p[13 * 9 + 2] = -1e9f;
        resizable_tensor gis = random_tensor(rnd, 2, 3, 9, 13), gs = random_tensor(rnd, 2, 3, 9, 13);
        check("softmax", {&ys}, [&]{ tt::softmax(ys, s, operation_mode::PLANE_WISE); });
        npu::set_routing(npu::routing::never);
        tt::softmax(ds, s, operation_mode::PLANE_WISE);
        check("softmax_gradient", {&gs}, [&]{ tt::softmax_gradient(gs, ds, gis, operation_mode::PLANE_WISE); }, 2);

        // Logits far from zero, whose differences half precision resolves only once each row is
        // brought to a maximum of zero.
        resizable_tensor distant = random_tensor(rnd, 4, 1, 16, 33, 195, 205), yf(4, 1, 16, 33);
        check("softmax", {&yf}, [&]{ tt::softmax(yf, distant, operation_mode::PLANE_WISE); });

        resizable_tensor ya(3, 7, 5, 4), da(3, 7, 5, 4);
        check("softmax_all", {&ya}, [&]{ tt::softmax_all(ya, x); });
        npu::set_routing(npu::routing::never);
        tt::softmax_all(da, x);
        check("softmax_all_gradient", {&g}, [&]{ tt::softmax_all_gradient(g, da, gi); }, 2);
    }

// ----------------------------------------------------------------------------------------

    void test_normalization (
        dlib::rand& rnd
    )
    {
        const double eps = 1e-5;
        resizable_tensor src = random_tensor(rnd, 6, 3, 5, 7);
        resizable_tensor gi = random_tensor(rnd, 6, 3, 5, 7), sg = random_tensor(rnd, 6, 3, 5, 7);

        // Batch normalization, fully connected layout.
        resizable_tensor gamma = random_tensor(rnd, 1, 3, 5, 7, 0.5f, 1.5f), beta = random_tensor(rnd, 1, 3, 5, 7);
        resizable_tensor rm = random_tensor(rnd, 1, 3, 5, 7), rv = random_tensor(rnd, 1, 3, 5, 7, 0.5f, 2);
        // Averaging into the running statistics needs means and invstds of their size.
        resizable_tensor dest, means(1, 3, 5, 7), invstds(1, 3, 5, 7), gg(1, 3, 5, 7), bg(1, 3, 5, 7);
        check("batch_normalize_inference", {&dest},
            [&]{ tt::batch_normalize_inference(eps, dest, src, gamma, beta, rm, rv); });
        check("batch_normalize", {&dest, &means, &invstds, &rm, &rv},
            [&]{ tt::batch_normalize(eps, dest, means, invstds, 0.1, rm, rv, src, gamma, beta); }, 2);
        npu::set_routing(npu::routing::never);
        tt::batch_normalize(eps, dest, means, invstds, 1, rm, rv, src, gamma, beta);
        check("batch_normalize_gradient", {&sg, &gg, &bg},
            [&]{ tt::batch_normalize_gradient(eps, gi, means, invstds, src, gamma, sg, gg, bg); }, 3);

        // Batch normalization, convolutional layout.
        resizable_tensor cgamma = random_tensor(rnd, 1, 3, 1, 1, 0.5f, 1.5f), cbeta = random_tensor(rnd, 1, 3, 1, 1);
        resizable_tensor crm = random_tensor(rnd, 1, 3, 1, 1), crv = random_tensor(rnd, 1, 3, 1, 1, 0.5f, 2);
        resizable_tensor cmeans(1, 3, 1, 1), cinvstds(1, 3, 1, 1), cgg(1, 3, 1, 1), cbg(1, 3, 1, 1);
        check("batch_normalize_conv_inference", {&dest},
            [&]{ tt::batch_normalize_conv_inference(eps, dest, src, cgamma, cbeta, crm, crv); });
        check("batch_normalize_conv", {&dest, &cmeans, &cinvstds, &crm, &crv},
            [&]{ tt::batch_normalize_conv(eps, dest, cmeans, cinvstds, 0.1, crm, crv, src, cgamma, cbeta); }, 2);
        npu::set_routing(npu::routing::never);
        tt::batch_normalize_conv(eps, dest, cmeans, cinvstds, 1, crm, crv, src, cgamma, cbeta);
        check("batch_normalize_conv_gradient", {&sg, &cgg, &cbg},
            [&]{ tt::batch_normalize_conv_gradient(eps, gi, cmeans, cinvstds, src, cgamma, sg, cgg, cbg); }, 3);

        // Layer normalization.
        resizable_tensor lgamma = random_tensor(rnd, 1, 3, 1, 1, 0.5f, 1.5f), lbeta = random_tensor(rnd, 1, 3, 1, 1);
        resizable_tensor lgg(1, 3, 1, 1), lbg(1, 3, 1, 1), dmeans, dvars;
        check("layer_normalize", {&dest, &means, &invstds},
            [&]{ tt::layer_normalize(eps, dest, means, invstds, src, lgamma, lbeta); });
        npu::set_routing(npu::routing::never);
        tt::layer_normalize(eps, dest, means, invstds, src, lgamma, lbeta);
        check("layer_normalize_gradient", {&sg, &lgg, &lbg},
            [&]{ tt::layer_normalize_gradient(eps, gi, means, invstds, src, lgamma, sg, lgg, lbg, dmeans, dvars); }, 3);

        // RMS normalization along nc, as in a transformer, then along k.
        resizable_tensor rsrc = random_tensor(rnd, 2, 3, 5, 64), rgamma = random_tensor(rnd, 1, 1, 1, 64, 0.5f, 1.5f);
        resizable_tensor rgi = random_tensor(rnd, 2, 3, 5, 64), rsg = random_tensor(rnd, 2, 3, 5, 64);
        resizable_tensor rgg = random_tensor(rnd, 1, 1, 1, 64), rdest, rscale, dscale;
        check("rms_normalize", {&rdest, &rscale}, [&]{ tt::rms_normalize(1e-6, rdest, rscale, rsrc, rgamma, true); });
        npu::set_routing(npu::routing::never);
        tt::rms_normalize(1e-6, rdest, rscale, rsrc, rgamma, true);
        check("rms_normalize_gradient", {&rsg, &rgg},
            [&]{ tt::rms_normalize_gradient(rgi, rscale, rsrc, rgamma, rsg, rgg, dscale, true); }, 3);

        resizable_tensor ksrc = random_tensor(rnd, 4, 48, 3, 5), kgamma = random_tensor(rnd, 1, 1, 1, 48, 0.5f, 1.5f);
        resizable_tensor kgi = random_tensor(rnd, 4, 48, 3, 5), ksg = random_tensor(rnd, 4, 48, 3, 5);
        resizable_tensor kgg = random_tensor(rnd, 1, 1, 1, 48), kdest, kscale;
        check("rms_normalize", {&kdest, &kscale}, [&]{ tt::rms_normalize(1e-6, kdest, kscale, ksrc, kgamma, false); });
        npu::set_routing(npu::routing::never);
        tt::rms_normalize(1e-6, kdest, kscale, ksrc, kgamma, false);
        check("rms_normalize_gradient", {&ksg, &kgg},
            [&]{ tt::rms_normalize_gradient(kgi, kscale, ksrc, kgamma, ksg, kgg, dscale, false); }, 3);

        // Bias gradients.
        resizable_tensor bias_conv(1, 3, 1, 1), bias_fc(1, 3, 5, 7);
        check("assign_conv_bias_gradient", {&bias_conv}, [&]{ tt::assign_conv_bias_gradient(bias_conv, gi); }, 2);
        check("assign_bias_gradient", {&bias_fc}, [&]{ tt::assign_bias_gradient(bias_fc, gi); }, 2);
    }

// ----------------------------------------------------------------------------------------

    void test_spatial (
        dlib::rand& rnd
    )
    {
        resizable_tensor img = random_tensor(rnd, 2, 3, 6, 6), up(2, 3, 11, 11), down(2, 3, 3, 4);
        check("resize_bilinear", {&up}, [&]{ tt::resize_bilinear(up, img); });
        check("resize_bilinear", {&down}, [&]{ tt::resize_bilinear(down, img); });
        resizable_tensor img_grad = random_tensor(rnd, 2, 3, 6, 6), up_grad = random_tensor(rnd, 2, 3, 11, 11);
        check("resize_bilinear_gradient", {&img_grad}, [&]{ tt::resize_bilinear_gradient(img_grad, up_grad); }, 2);

        // Data movement.
        resizable_tensor rsrc = random_tensor(rnd, 2, 3, 8, 6), rdst = random_tensor(rnd, 2, 12, 4, 3);
        resizable_tensor rgrad = random_tensor(rnd, 2, 3, 8, 6);
        check("reorg", {&rdst}, [&]{ tt::reorg(false, rdst, 2, 2, rsrc); }, 1, exact_on_device());
        check("reorg_gradient", {&rgrad}, [&]{ tt::reorg_gradient(true, rgrad, 2, 2, rdst); }, 1, exact_on_device());
        resizable_tensor big = random_tensor(rnd, 3, 7, 4, 5), part = random_tensor(rnd, 3, 4, 4, 5);
        check("copy_tensor", {&big}, [&]{ tt::copy_tensor(false, big, 2, part, 1, 3); }, 1, exact_on_device());
        check("copy_tensor", {&big}, [&]{ tt::copy_tensor(true, big, 1, 1, 2, part, 0, 2, 1, 3, 2, 3); }, 1,
            exact_on_device());
        resizable_tensor tsrc = random_tensor(rnd, 2, 3, 5, 9), tdst = random_tensor(rnd, 2, 3, 9, 5);
        check("transpose", {&tdst}, [&]{ tt::transpose(false, tdst, tsrc); }, 1, exact_on_device());
        check("transpose", {&tdst}, [&]{ tt::transpose(true, tdst, tsrc); }, 1, exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_convolution_and_pooling (
        dlib::rand& rnd
    )
    {
        resizable_tensor data = random_tensor(rnd, 2, 3, 13, 11), filters = random_tensor(rnd, 4, 3, 3, 3), out;
        tt::tensor_conv conv;
        conv.setup(data, filters, 2, 1, 1, 1);
        check("conv", {&out}, [&]{ conv(false, out, data, filters); }, 2);
        npu::set_routing(npu::routing::never);
        conv(false, out, data, filters);
        check("conv", {&out}, [&]{ conv(true, out, data, filters); }, 2);
        resizable_tensor biases = random_tensor(rnd, 1, 4, 1, 1), fused;
        check("conv", {&fused}, [&]{ conv(false, fused, data, filters, biases, true); }, 2);

        resizable_tensor gi = random_tensor(rnd, out.num_samples(), out.k(), out.nr(), out.nc());
        resizable_tensor data_grad = random_tensor(rnd, 2, 3, 13, 11), filters_grad = random_tensor(rnd, 4, 3, 3, 3);
        check("conv_gradient_for_data", {&data_grad},
            [&]{ conv.get_gradient_for_data(true, gi, filters, data_grad); }, 2);
        check("conv_gradient_for_filters", {&filters_grad},
            [&]{ conv.get_gradient_for_filters(false, gi, data, filters_grad); }, 3);

        // A stride that does not divide the padded width leaves the last columns of the input
        // out of every window, which the gradient for the data gives back as zeros.
        resizable_tensor wide = random_tensor(rnd, 4, 3, 5, 5), wout;
        tt::tensor_conv strided;
        strided.setup(data, wide, 3, 3, 2, 2);
        check("conv", {&wout}, [&]{ strided(false, wout, data, wide); }, 2);
        npu::set_routing(npu::routing::never);
        strided(false, wout, data, wide);
        resizable_tensor wgi = random_tensor(rnd, wout.num_samples(), wout.k(), wout.nr(), wout.nc());
        check("conv_gradient_for_data", {&data_grad},
            [&]{ strided.get_gradient_for_data(false, wgi, wide, data_grad); }, 2);
        resizable_tensor wide_grad = random_tensor(rnd, 4, 3, 5, 5);
        check("conv_gradient_for_filters", {&wide_grad},
            [&]{ strided.get_gradient_for_filters(true, wgi, data, wide_grad); }, 3);

        resizable_tensor pool_src(2, 3, 13, 11);
        fill_distinct(pool_src, rnd);
        tt::pooling maxp, avgp;
        maxp.setup_max_pooling(3, 3, 2, 2, 1, 1);
        avgp.setup_avg_pooling(2, 3, 2, 1, 0, 1);
        resizable_tensor pmax, pavg;
        check("pooling", {&pmax}, [&]{ maxp(pmax, pool_src); });
        check("pooling", {&pavg}, [&]{ avgp(pavg, pool_src); });
        npu::set_routing(npu::routing::never);
        maxp(pmax, pool_src);
        avgp(pavg, pool_src);
        resizable_tensor gmax = random_tensor(rnd, pmax.num_samples(), pmax.k(), pmax.nr(), pmax.nc());
        resizable_tensor gavg = random_tensor(rnd, pavg.num_samples(), pavg.k(), pavg.nr(), pavg.nc());
        resizable_tensor pgrad = random_tensor(rnd, 2, 3, 13, 11);
        check("pooling_gradient", {&pgrad}, [&]{ maxp.get_gradient(gmax, pmax, pool_src, pgrad); });
        check("pooling_gradient", {&pgrad}, [&]{ avgp.get_gradient(gavg, pavg, pool_src, pgrad); });
    }

// ----------------------------------------------------------------------------------------

    void test_embeddings (
        dlib::rand& rnd
    )
    {
        const long V = 50, P = 5, D = 24;
        resizable_tensor table = random_tensor(rnd, V, D, 1, 1), puzzles = random_tensor(rnd, P, D, 1, 1);
        resizable_tensor tokens(3, 1, 7, 1), mixed(3, 1, 7, 1);
        fill_integers(tokens, rnd, V + 3);
        fill_integers(mixed, rnd, V + P + 2);
        resizable_tensor emb(3, 1, 7, D), pemb(3, 1, 7, D);
        check("embeddings", {&emb}, [&]{ tt::embeddings(emb, tokens, table); }, 1, exact_on_device());
        check("puzzle_embeddings", {&pemb}, [&]{ tt::puzzle_embeddings(pemb, mixed, table, puzzles, 1.5f); }, 1,
            exact_on_device());

        resizable_tensor egi = random_tensor(rnd, 3, 1, 7, D), grads = table, freqs(V, 1, 1, 1);
        fill_integers(freqs, rnd, 10, 1);
        check("embeddings_gradient", {&grads},
            [&]{ tt::embeddings_gradient(tokens, egi, grads, freqs, 0.1f, true); }, 1, exact_on_device());
        resizable_tensor token_grad(V, D, 1, 1), state = puzzles;
        check("puzzle_embeddings_gradient", {&token_grad, &state},
            [&]{ tt::puzzle_embeddings_gradient(mixed, egi, token_grad, state, 1.5f, 0.01f, 0.1f); }, 1,
            exact_on_device());

        resizable_tensor rsrc = random_tensor(rnd, 2, 3, 4, 5), rdst(2, 6, 4, 5);
        resizable_tensor acc = random_tensor(rnd, 2, 3, 4, 5), wide = random_tensor(rnd, 2, 6, 4, 5);
        check("repeat_channels", {&rdst}, [&]{ tt::repeat_channels(rdst, rsrc, 2); }, 1, exact_on_device());
        check("accumulate_repeated_channels", {&acc}, [&]{ tt::accumulate_repeated_channels(acc, wide, 2); }, 1,
            exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_optimizers (
        dlib::rand& rnd
    )
    {
        resizable_tensor params = random_tensor(rnd, 1, 1, 37, 41), grad = random_tensor(rnd, 1, 1, 37, 41);
        resizable_tensor s = random_tensor(rnd, 1, 1, 37, 41), m = random_tensor(rnd, 1, 1, 37, 41, -0.1f, 0.1f);
        resizable_tensor v = random_tensor(rnd, 1, 1, 37, 41, 0, 0.01f);
        check("adam", {&s, &m, &v},
            [&]{ tt::compute_adam_update(0, params.size(), s, m, v, 3, 0.01f, 0.001f, 0.9f, 0.999f, params, grad); }, 1,
            exact_on_device());
        check("adam", {&s, &m, &v},
            [&]{
                tt::compute_adam_update(100, params.size() - 50, s, m, v, 3, 0.01f, 0.0f, 0.9f, 0.999f, params, grad);
            }, 1, exact_on_device());
        check("adam_atan2", {&s, &m, &v},
            [&]{
                tt::compute_adam_atan2_update(0, params.size(), s, m, v, 3, 0.01f, 0.001f, 0.9f, 0.999f, params, grad);
            }, 1, exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_adaptive_computation (
        dlib::rand& rnd
    )
    {
        const long B = 2, T = 5, C = 3, F = 7;
        resizable_tensor state = random_tensor(rnd, B, C, T, F), halt_params = random_tensor(rnd, F + 1, 1, 1, 1);
        resizable_tensor mask(B * T, 1, 1, 1), steps(B * T, 1, 1, 1);
        fill_integers(mask, rnd, 2);
        fill_integers(steps, rnd, 5, 1);
        resizable_tensor logits(B * T, 1, 1, 1), probs(B * T, 1, 1, 1);
        check("compute_halt_logits", {&logits, &probs},
            [&]{ tt::compute_halt_logits(logits, probs, state, halt_params, mask, B, T, C, F); }, 2);
        npu::set_routing(npu::routing::never);
        tt::compute_halt_logits(logits, probs, state, halt_params, mask, B, T, C, F);

        resizable_tensor grad_params = random_tensor(rnd, F + 1, 1, 1, 1), grad_state = random_tensor(rnd, B, C, T, F);
        resizable_tensor gi = random_tensor(rnd, B, C, T, F), final_state = random_tensor(rnd, B, C, T, F);
        check("compute_halt_gradient", {&grad_params, &grad_state},
            [&]{ tt::compute_halt_gradient(grad_params, grad_state, gi, state, probs, halt_params, mask, steps,
                final_state, 1, B, T, C, F, 0.01f, false); }, 3);

        resizable_tensor output = random_tensor(rnd, B, C, T, F), weights = random_tensor(rnd, B * T, 1, 1, 1, 0, 1);
        check("accumulate_weighted_state", {&output},
            [&]{ tt::accumulate_weighted_state(output, state, weights, mask, B, T, C, F); });
        resizable_tensor scaled = random_tensor(rnd, B, C, T, F);
        check("apply_act_depth_scaling", {&scaled}, [&]{ tt::apply_act_depth_scaling(scaled, steps, B, T, C, F, 4); });
        check("act_accumulate_output_grad", {&grad_state},
            [&]{ tt::act_accumulate_output_grad(grad_state, gi, weights, mask, steps, 1, B, T, C, F); });

        resizable_tensor row = random_tensor(rnd, 1, 1, 1, F), filled = random_tensor(rnd, B, C, T, F);
        check("broadcast_row", {&filled}, [&]{ tt::broadcast_row(filled, row); }, 1, exact_on_device());
        resizable_tensor active(B * T, 1, 1, 1), count(1), cumulative = random_tensor(rnd, B * T, 1, 1, 1, 0, 1);
        check("act_mark_active", {&active, &count}, [&]{ tt::act_mark_active(active, count, cumulative, 0.9f); }, 1,
            exact_on_device());
        resizable_tensor remainders = random_tensor(rnd, B * T, 1, 1, 1, 0, 1), counts(B * T, 1, 1, 1);
        resizable_tensor step_weights(B * T, 1, 1, 1);
        counts = 0;
        check("act_update_halting", {&cumulative, &remainders, &counts, &step_weights},
            [&]{ tt::act_update_halting(cumulative, remainders, counts, step_weights, probs, mask, 1, 4, 0.9f); }, 1,
            exact_on_device());
        resizable_tensor step_state = random_tensor(rnd, B, C, T, F);
        check("act_gather_final_state", {&final_state},
            [&]{ tt::act_gather_final_state(final_state, step_state, steps, 1, B, T, C, F); }, 1, exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_attention_helpers (
        dlib::rand& rnd
    )
    {
        for (long nc : {16L, 17L})
        {
            resizable_tensor data = random_tensor(rnd, 2, 3, 9, nc);
            resizable_tensor cos_cache(1, 1, 9, nc / 2), sin_cache(1, 1, 9, nc / 2);
            for (long r = 0; r < 9; ++r)
                for (long i = 0; i < nc / 2; ++i)
                {
                    const double angle = r * std::pow(10000.0, -2.0 * i / nc);
                    cos_cache.host()[r * (nc / 2) + i] = static_cast<float>(std::cos(angle));
                    sin_cache.host()[r * (nc / 2) + i] = static_cast<float>(std::sin(angle));
                }
            check("rotary_embedding", {&data},
                [&]{ tt::apply_rotary_positional_embedding(false, data, cos_cache, sin_cache); });
            check("rotary_embedding", {&data},
                [&]{ tt::apply_rotary_positional_embedding(true, data, cos_cache, sin_cache); });
        }

        resizable_tensor flat = random_tensor(rnd, 2, 1, 9, 24), heads = random_tensor(rnd, 2, 3, 9, 8);
        resizable_tensor merged = random_tensor(rnd, 2, 1, 9, 24);
        check("split_heads", {&heads}, [&]{ tt::split_heads(false, heads, flat); }, 1, exact_on_device());
        check("merge_heads", {&merged}, [&]{ tt::merge_heads(true, merged, heads); }, 1, exact_on_device());
    }

// ----------------------------------------------------------------------------------------

    void test_losses (
        dlib::rand& rnd
    )
    {
        resizable_tensor loss(1);
        resizable_tensor out = random_tensor(rnd, 2, 1, 5, 7, -3, 3), grad(2, 1, 5, 7);
        std::vector<matrix<float>> binary(2, matrix<float>(5, 7));
        for (auto& t : binary)
            for (auto& v : t)
                v = static_cast<float>(static_cast<long>(rnd.get_random_32bit_number() % 3) - 1);
        cpu::compute_loss_binary_log_per_pixel binary_loss;
        check("sigmoid", {&grad, &loss}, [&]{
            double l;
            binary_loss(binary.begin(), out, grad, l);
            loss.host()[0] = static_cast<float>(l);
        });

        resizable_tensor logits = random_tensor(rnd, 2, 4, 5, 7, -3, 3), cgrad(2, 4, 5, 7);
        std::vector<matrix<uint16_t>> classes(2, matrix<uint16_t>(5, 7));
        std::vector<matrix<weighted_label<uint16_t>>> weighted(2, matrix<weighted_label<uint16_t>>(5, 7));
        for (size_t s = 0; s < classes.size(); ++s)
            for (long r = 0; r < 5; ++r)
                for (long c = 0; c < 7; ++c)
                {
                    const uint16_t y = static_cast<uint16_t>(rnd.get_random_32bit_number() % 5);
                    classes[s](r, c) = y == 4 ? std::numeric_limits<uint16_t>::max() : y;
                    const float weight = y == 4 ? 0.0f : rnd.get_random_float();
                    weighted[s](r, c) = weighted_label<uint16_t>(static_cast<uint16_t>(y % 4), weight);
                }
        cpu::compute_loss_multiclass_log_per_pixel multiclass_loss;
        check("softmax", {&cgrad, &loss}, [&]{
            double l;
            multiclass_loss(classes.begin(), logits, cgrad, l);
            loss.host()[0] = static_cast<float>(l);
        });
        cpu::compute_loss_multiclass_log_per_pixel_weighted weighted_loss;
        check("softmax", {&cgrad, &loss}, [&]{
            double l;
            weighted_loss(weighted.begin(), logits, cgrad, l);
            loss.host()[0] = static_cast<float>(l);
        });

        resizable_tensor values = random_tensor(rnd, 2, 3, 5, 7), vgrad(2, 3, 5, 7);
        std::vector<std::vector<matrix<float>>> targets(2, std::vector<matrix<float>>(3, matrix<float>(5, 7)));
        for (auto& sample : targets)
            for (auto& t : sample)
                for (auto& v : t)
                    v = 4 * rnd.get_random_float() - 2;
        cpu::compute_loss_mean_squared_per_channel_and_pixel squared_loss;
        check("mean_squared", {&vgrad, &loss}, [&]{
            double l;
            squared_loss(targets.begin(), values, vgrad, l);
            loss.host()[0] = static_cast<float>(l);
        });

        // Cross entropy per token, with smoothing, z-loss, an ignored target and a padded query.
        const long vocab = 50, len = 9;
        resizable_tensor token_logits = random_tensor(rnd, 2, 1, len, vocab, -3, 3), tgrad(2, 1, len, vocab);
        resizable_tensor input(2, 1, len, 1);
        fill_integers(input, rnd, vocab);
        input.host()[3] = 0;
        std::vector<matrix<unsigned long, 0, 1>> next(2, matrix<unsigned long, 0, 1>(len));
        for (auto& t : next)
            for (long i = 0; i < len; ++i)
                t(i) = rnd.get_random_32bit_number() % vocab;
        next[1](4) = 7;
        cpu::compute_loss_cross_entropy_per_token token_loss;
        check("cross_entropy", {&tgrad, &loss}, [&]{
            tgrad = 0;
            double l;
            token_loss(next.begin(), input, token_logits, tgrad, l, 7, 0.1, 0, 1e-4);
            loss.host()[0] = static_cast<float>(l);
        });
        cpu::compute_loss_cross_entropy_per_token_multi multi_loss;
        const std::vector<long> ignored = {3, 7};
        check("cross_entropy", {&tgrad, &loss}, [&]{
            tgrad = 0;
            double l;
            multi_loss(next.begin(), input, token_logits, tgrad, l, ignored, 0.0, -1, 0.0);
            loss.host()[0] = static_cast<float>(l);
        });
    }

// ----------------------------------------------------------------------------------------

    /*
        A weight read by successive products is converted once and its device copy reused;
        every way of writing to it must make the next product see the new values.
    */
    void test_residency (
        dlib::rand& rnd
    )
    {
        const double tolerance = npu::uses_half_precision() ? 2e-2 : 1e-4;
        resizable_tensor x = random_tensor(rnd, 64, 1, 1, 96);
        resizable_tensor w = random_tensor(rnd, 96, 1, 1, 80);
        resizable_tensor y(64, 1, 1, 80), expected(64, 1, 1, 80);

        // On the device, then on the host, both from the current values of x and w.
        auto products_agree = [&]()
        {
            print_spinner();
            x.host();  // activations change from call to call: this write keeps x from being kept
            npu::set_routing(npu::routing::always);
            tt::gemm(0, y, 1, x, false, w, false);
            npu::set_routing(npu::routing::never);
            tt::gemm(0, expected, 1, x, false, w, false);
            return relative_error(expected, y) <= tolerance;
        };

        const unsigned long long calls = device_calls("gemm");
        const unsigned long long reused = reused_calls("gemm");
        for (int i = 0; i < 4; ++i)
            DLIB_TEST(products_agree());
        if (device_calls("gemm") == calls && refused_by_device("gemm"))
            return;
        // The first call meets w, the second keeps a copy of it, the next ones reuse that copy.
        DLIB_TEST_MSG(reused_calls("gemm") >= reused + 2, "reused " << reused_calls("gemm") - reused);

        w.host()[5] = 7;
        DLIB_TEST(products_agree());

        w = 0.25f;
        DLIB_TEST(products_agree());
        DLIB_TEST(products_agree());

        tt::affine_transform(w, w, -3.0f, 1.0f);
        DLIB_TEST(products_agree());
        DLIB_TEST(products_agree());

        resizable_tensor other = random_tensor(rnd, 96, 1, 1, 80);
        memcpy(w, other);
        DLIB_TEST(products_agree());
        DLIB_TEST(products_agree());

        // Through another view of the same block.
        alias_tensor whole(96 * 80);
        auto view = whole(w, 0);
        view.host()[100] = -9;
        DLIB_TEST(products_agree());
        DLIB_TEST(products_agree());

        std::ostringstream sout;
        serialize(other, sout);
        randomize(other, rnd);
        DLIB_TEST(products_agree());
        std::istringstream sin(sout.str());
        deserialize(w, sin);
        DLIB_TEST(products_agree());
        DLIB_TEST(products_agree());

        // The buffers of two tensors exchanged: the key follows the values.
        resizable_tensor w2 = random_tensor(rnd, 96, 1, 1, 80);
        DLIB_TEST(products_agree());
        w.swap(w2);
        DLIB_TEST(products_agree());
        DLIB_TEST(products_agree());

        // A weight on the left of the product, kept as well.
        resizable_tensor wl = random_tensor(rnd, 80, 1, 1, 96), z(80, 1, 1, 64), zexp(80, 1, 1, 64);
        resizable_tensor xt = random_tensor(rnd, 96, 1, 1, 64);
        const unsigned long long before_left = reused_calls("gemm");
        for (int i = 0; i < 4; ++i)
        {
            print_spinner();
            xt.host();
            npu::set_routing(npu::routing::always);
            tt::gemm(0, z, 1, wl, false, xt, false);
            npu::set_routing(npu::routing::never);
        }
        tt::gemm(0, zexp, 1, wl, false, xt, false);
        DLIB_TEST(relative_error(zexp, z) <= tolerance);
        DLIB_TEST(reused_calls("gemm") >= before_left + 2);

        // The filters of a convolution.
        resizable_tensor image = random_tensor(rnd, 2, 8, 12, 12), filters = random_tensor(rnd, 16, 8, 3, 3);
        resizable_tensor maps, maps_host;
        tt::tensor_conv conv;
        conv.setup(image, filters, 1, 1, 1, 1);
        auto convolutions_agree = [&]()
        {
            print_spinner();
            image.host();
            npu::set_routing(npu::routing::always);
            conv(false, maps, image, filters);
            npu::set_routing(npu::routing::never);
            conv(false, maps_host, image, filters);
            return relative_error(maps_host, maps) <= tolerance;
        };
        const unsigned long long before_conv = reused_calls("conv");
        for (int i = 0; i < 4; ++i)
            DLIB_TEST(convolutions_agree());
        if (device_calls("conv") > 0)
            DLIB_TEST(reused_calls("conv") >= before_conv + 2);
        filters.host()[0] = 5;
        DLIB_TEST(convolutions_agree());
        filters = -0.5f;
        DLIB_TEST(convolutions_agree());
        DLIB_TEST(convolutions_agree());
    }

// ----------------------------------------------------------------------------------------

    void test_guards (
        dlib::rand& rnd
    )
    {
        print_spinner();
        // A non-finite operand sends the call back to the host, which produces the result.
        resizable_tensor x = random_tensor(rnd, 3, 5, 7, 11), y_host(3, 5, 7, 11), y(3, 5, 7, 11);
        x.host()[17] = std::numeric_limits<float>::quiet_NaN();
        x.host()[18] = std::numeric_limits<float>::infinity();
        npu::set_routing(npu::routing::never);
        tt::tanh(y_host, x);
        const unsigned long long rejected = rejected_calls("tanh");
        npu::set_routing(npu::routing::always);
        tt::tanh(y, x);
        npu::set_routing(npu::routing::never);
        DLIB_TEST(rejected_calls("tanh") > rejected || refused_by_device("tanh"));
        DLIB_TEST(relative_error(y_host, y) == 0);

        // Nothing reaches the device inside an exact_scope.
        resizable_tensor a = random_tensor(rnd, 3, 5, 7, 11);
        const unsigned long long before = device_calls("sigmoid");
        npu::set_routing(npu::routing::always);
        {
            npu::exact_scope host_only;
            tt::sigmoid(y, a);
        }
        npu::set_routing(npu::routing::never);
        DLIB_TEST(device_calls("sigmoid") == before);

        // Automatic routing: a product large enough to qualify, timed on both sides until one
        // is chosen, gives the same result whichever computes it.
        resizable_tensor L = random_tensor(rnd, 256, 1, 1, 192), R = random_tensor(rnd, 192, 1, 1, 160);
        resizable_tensor expected(256, 1, 1, 160), D(256, 1, 1, 160);
        tt::gemm(0, expected, 1, L, false, R, false);
        npu::set_routing(npu::routing::automatic);
        for (int i = 0; i < 10; ++i)
        {
            print_spinner();
            tt::gemm(0, D, 1, L, false, R, false);
            DLIB_TEST(relative_error(expected, D) <= 2 * (npu::uses_half_precision() ? 2e-2 : 1e-4));
        }
        npu::set_routing(npu::routing::never);
    }

// ----------------------------------------------------------------------------------------

    using small_net = loss_multiclass_log<fc<4, relu<fc<32, input<matrix<float>>>>>>;

    void train_steps (
        small_net& net,
        const std::vector<matrix<float>>& samples,
        const std::vector<unsigned long>& labels,
        npu::routing mode
    )
    {
        dnn_trainer<small_net> trainer(net, sgd(0.0005, 0.9));
        trainer.be_quiet();
        trainer.set_learning_rate(0.05);
        npu::set_routing(mode);
        for (size_t first = 0; first + 32 <= samples.size(); first += 32)
        {
            print_spinner();
            const std::vector<matrix<float>> x(samples.begin() + first, samples.begin() + first + 32);
            const std::vector<unsigned long> y(labels.begin() + first, labels.begin() + first + 32);
            trainer.train_one_step(x, y);
        }
        trainer.get_net();
        npu::set_routing(npu::routing::never);
    }

    void test_training (
        dlib::rand& rnd
    )
    {
        std::vector<matrix<float>> samples;
        std::vector<unsigned long> labels;
        for (int i = 0; i < 320; ++i)
        {
            matrix<float> x(16, 1);
            for (long j = 0; j < x.size(); ++j)
                x(j) = 2 * rnd.get_random_float() - 1;
            samples.push_back(x);
            labels.push_back((sum(rowm(x, range(0, 7))) > 0 ? 1UL : 0UL) + (x(15) > 0 ? 2UL : 0UL));
        }

        npu::set_routing(npu::routing::never);
        small_net host_net;
        host_net(samples[0]);
        small_net device_net = host_net;
        train_steps(host_net, samples, labels, npu::routing::never);
        train_steps(device_net, samples, labels, npu::routing::always);

        const double tolerance = npu::uses_half_precision() ? 5e-2 : 1e-3;
        const double e1 = relative_error(layer<1>(host_net).layer_details().get_layer_params(),
            layer<1>(device_net).layer_details().get_layer_params());
        const double e3 = relative_error(layer<3>(host_net).layer_details().get_layer_params(),
            layer<3>(device_net).layer_details().get_layer_params());
        dlog << LINFO << "trained parameters differ by " << e1 << " and " << e3;
        DLIB_TEST_MSG(e1 <= tolerance && e3 <= tolerance, "trained parameters differ by " << e1 << " and " << e3);
    }

#endif // DLIB_USE_NPU && !DLIB_USE_CUDA

// ----------------------------------------------------------------------------------------

    class npu_tester : public tester
    {
    public:
        npu_tester (
        ) :
            tester ("test_npu",
                    "Runs tests on the routing of tensor operations to an NPU.")
        {}

        void perform_test (
        )
        {
            test_control_api();
#if defined(DLIB_USE_NPU) && !defined(DLIB_USE_CUDA)
            if (!npu::is_available())
            {
                dlog << LINFO << "No inference device, the device paths are not tested. "
                    "DLIB_NPU_DEVICE=CPU runs them on the OpenVINO CPU plugin.";
                return;
            }
            const npu::routing previous = npu::get_routing();
            npu::set_synchronous_compilation(true);
            npu::reset_statistics();

            dlib::rand rnd(1234);
            test_elementwise(rnd);
            test_arithmetic(rnd);
            test_matrix(rnd);
            test_softmax(rnd);
            test_normalization(rnd);
            test_spatial(rnd);
            test_convolution_and_pooling(rnd);
            test_embeddings(rnd);
            test_optimizers(rnd);
            test_adaptive_computation(rnd);
            test_attention_helpers(rnd);
            test_losses(rnd);
            test_residency(rnd);
            test_guards(rnd);
            test_training(rnd);

            std::ostringstream sout;
            npu::print_status(sout);
            dlog << LINFO << "\n" << sout.str();
            npu::set_synchronous_compilation(false);
            npu::set_routing(previous);
#endif
        }
    } a;

}
