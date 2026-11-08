// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <cmath>
#include <sstream>
#include <vector>
#include <dlib/dnn.h>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.parameter_average");

    // A hierarchical layer holds its two modules as networks of their own.
    using module = rms_norm<linear<8, tag10<input_tensor>>>;
    using net_type = loss_halting_cross_entropy_per_token<linear<5,
        hrm<module, module, 2, 2, embeddings<12, 8, input<matrix<int, 0, 1>>>>>>;
    using flat_net_type = loss_halting_cross_entropy_per_token<linear<5,
        embeddings<12, 8, input<matrix<int, 0, 1>>>>>;

    template <typename network>
    void set_up (network& net)
    {
        dlib::rand rnd(5);
        std::vector<matrix<int, 0, 1>> xs(2, matrix<int, 0, 1>(5));
        for (auto& x : xs)
            for (long t = 0; t < x.size(); ++t)
                x(t) = rnd.get_integer_in_range(0, 12);
        resizable_tensor x;
        net.to_tensor(xs.begin(), xs.end(), x);
        net.forward(x);
    }

    template <typename network>
    std::vector<std::vector<float>> values_of (network& net)
    {
        std::vector<std::vector<float>> values;
        visit_learnable_parameters(net, [&values](tensor& t) {
            values.emplace_back(t.host(), t.host() + t.size());
        });
        return values;
    }

    double largest_gap (const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b,
                        float shift)
    {
        double gap = a.size() == b.size() ? 0 : 1e30;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        {
            if (a[i].size() != b[i].size())
                return 1e30;
            for (size_t j = 0; j < a[i].size(); ++j)
                gap = std::max(gap, (double)std::abs(a[i][j] - (b[i][j] + shift)));
        }
        return gap;
    }

    void test_visit_learnable_parameters ()
    {
        print_spinner();
        net_type net;
        set_up(net);

        size_t visited = 0, tensors = 0;
        visit_learnable_parameters(net, [&](tensor& t) { visited += t.size(); ++tensors; });
        size_t outer = 0, outer_tensors = 0;
        visit_layer_parameters(net, [&](tensor& t) { outer += t.size(); ++outer_tensors; });

        // The modules are reached, and nothing the counter counts is left out.
        DLIB_TEST_MSG(visited == count_parameters(net).total,
            visited << " values visited, " << count_parameters(net).total << " counted");
        DLIB_TEST(tensors > outer_tensors);
        DLIB_TEST(visited > outer);

        // Two networks of one type yield matching tensors at matching positions.
        net_type copy = net;
        const auto mine = values_of(net);
        const auto theirs = values_of(copy);
        DLIB_TEST(largest_gap(mine, theirs, 0) == 0);
    }

    void test_parameter_average ()
    {
        print_spinner();
        net_type net;
        set_up(net);

        parameter_average average;
        DLIB_TEST(average.get_decay() == 0.999);
        DLIB_TEST(average.empty());
        average.set_decay(0.75);
        DLIB_TEST(average.get_decay() == 0.75);

        // The first update starts the average at the parameters.
        average.update(net);
        DLIB_TEST(!average.empty());
        const auto start = values_of(net);

        // Every parameter moves by one, and the average moves by a quarter of that.
        visit_learnable_parameters(net, [](tensor& t) { tt::affine_transform(t, t, 1, 1); });
        average.update(net);
        DLIB_TEST(largest_gap(values_of(net), start, 1) < 1e-6);

        average.swap(net);
        const double averaged = largest_gap(values_of(net), start, 0.25f);
        DLIB_TEST_MSG(averaged < 1e-5, "the network does not hold the average: " << averaged);
        average.swap(net);
        DLIB_TEST(largest_gap(values_of(net), start, 1) < 1e-6);

        // The average reads back as it was written.
        std::ostringstream sout;
        serialize(average, sout);
        parameter_average restored;
        std::istringstream sin(sout.str());
        deserialize(restored, sin);
        DLIB_TEST(restored.get_decay() == 0.75);
        std::ostringstream again;
        serialize(restored, again);
        DLIB_TEST(again.str() == sout.str());

        restored.swap(net);
        DLIB_TEST(largest_gap(values_of(net), start, 0.25f) < 1e-5);
        restored.swap(net);

        // A network with other parameter tensors is refused.
        flat_net_type other;
        set_up(other);
        bool refused = false;
        try { average.update(other); }
        catch (const dlib::error&) { refused = true; }
        DLIB_TEST(refused);

        average.clear();
        DLIB_TEST(average.empty());
        average.update(other);
        DLIB_TEST(!average.empty());

        bool rejected = false;
        try { parameter_average outside(1.0); }
        catch (const dlib::fatal_error&) { rejected = true; }
        DLIB_TEST(rejected);
    }

    class parameter_average_tester : public tester
    {
    public:
        parameter_average_tester (
        ) :
            tester ("test_parameter_average",
                    "Runs tests on visit_learnable_parameters() and parameter_average.")
        {}

        void perform_test (
        )
        {
            test_visit_learnable_parameters();
            test_parameter_average();
        }
    } a;
}
