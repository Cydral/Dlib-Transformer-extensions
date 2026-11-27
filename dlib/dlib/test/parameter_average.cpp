// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <cmath>
#include <cstring>
#include <limits>
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

// ----------------------------------------------------------------------------------------

    using vector_net_type = loss_cross_entropy_per_token<linear<5, rms_norm<linear<7,
        embeddings<12, 8, input<matrix<int, 0, 1>>>>>>>;

    void test_parameter_vectors ()
    {
        vector_net_type net;
        set_up(net);
        const parameter_layout layout = get_parameter_layout(net);
        std::vector<float> v;
        get_parameter_vector(net, v);
        DLIB_TEST(v.size() == layout.size());
        size_t covered = 0;
        for (const auto& b : layout.blocks())
        {
            DLIB_TEST(b.offset == covered);
            DLIB_TEST(b.size == static_cast<size_t>(b.rows) * b.cols);
            covered += b.size;
        }
        DLIB_TEST(covered == v.size());

        // Writing back a modified vector changes the network, and reading it gives it back.
        const uint64 before = parameter_digest(v);
        std::vector<float> w = v;
        for (float& x : w) x = x * 0.5f + 0.25f;
        set_parameter_vector(net, w);
        std::vector<float> back;
        get_parameter_vector(net, back);
        DLIB_TEST(back == w);
        DLIB_TEST(parameter_digest(back) != before);
        set_parameter_vector(net, v);
        get_parameter_vector(net, back);
        DLIB_TEST(parameter_digest(back) == before);

        bool refused = false;
        try { set_parameter_vector(net, std::vector<float>(v.size() + 1)); }
        catch (const dlib::error&) { refused = true; }
        DLIB_TEST(refused);

        // Squared gradients land at the positions of their parameters.
        std::vector<matrix<int, 0, 1>> xs(2, matrix<int, 0, 1>(5));
        std::vector<matrix<unsigned long, 0, 1>> ys(2, matrix<unsigned long, 0, 1>(5));
        dlib::rand rnd(3);
        for (size_t i = 0; i < xs.size(); ++i)
            for (long t = 0; t < 5; ++t)
            {
                xs[i](t) = rnd.get_integer_in_range(0, 12);
                ys[i](t) = static_cast<unsigned long>(rnd.get_integer_in_range(0, 5));
            }
        resizable_tensor x;
        net.to_tensor(xs.begin(), xs.end(), x);
        net.compute_parameter_gradients(x, ys.begin());
        std::vector<float> squares;
        accumulate_squared_gradients(net, squares, 2.0f);
        DLIB_TEST(squares.size() == v.size());
        std::vector<float> grads;
        visit_layer_parameter_gradients(net, [&grads](size_t, tensor& t) {
            grads.insert(grads.end(), t.begin(), t.end()); });
        DLIB_TEST(grads.size() == squares.size());
        double err = 0, total = 0;
        for (size_t i = 0; i < grads.size(); ++i)
        {
            err += std::abs(squares[i] - 2 * grads[i] * grads[i]);
            total += squares[i];
        }
        DLIB_TEST(total > 0);
        DLIB_TEST(err <= 1e-6 * total);

        // A network holding networks of its own keeps gradients out of reach.
        net_type nested;
        set_up(nested);
        refused = false;
        try { accumulate_squared_gradients(nested, squares); }
        catch (const dlib::error&) { refused = true; }
        DLIB_TEST(refused);

        // Splitting a block keeps the offsets and the size.
        parameter_layout split = layout;
        const size_t index = split.num_blocks() - 1;
        const parameter_block whole = split.block(index);
        split.split_block(index, { { 2, whole.cols } });
        DLIB_TEST(split.size() == layout.size());
        DLIB_TEST(split.block(index).offset == whole.offset);
        DLIB_TEST(split.block(index + 1).offset == whole.offset + 2 * static_cast<size_t>(whole.cols));
        DLIB_TEST(split.fingerprint() != layout.fingerprint());

        std::ostringstream sout;
        serialize(split, sout);
        std::istringstream sin(sout.str());
        parameter_layout read;
        deserialize(read, sin);
        DLIB_TEST(read == split);
    }

    void test_weighted_median ()
    {
        DLIB_TEST(weighted_median(std::vector<double>{ 5, 1, 3, 9 }, std::vector<double>{ 1, 1, 1, 1 }) == 3);
        DLIB_TEST(weighted_median(std::vector<double>{ 5, 1, 3, 9 }, std::vector<double>{ 0, 0, 0, 1 }) == 9);
        DLIB_TEST(weighted_median(std::vector<double>{ 5, 1, 3, 9 }, std::vector<double>{ 1, 10, 1, 1 }) == 1);
        DLIB_TEST(weighted_median(std::vector<float>{ 2 }, std::vector<double>{ 0.5 }) == 2);
    }

    void test_delta_codec ()
    {
        parameter_layout layout;
        layout.add_block(40, 30);
        layout.add_block(1, 77);
        layout.add_block(13, 5);
        dlib::rand rnd(11);
        std::vector<float> delta(layout.size());
        for (float& x : delta) x = static_cast<float>(rnd.get_random_gaussian());
        double norm = 0;
        for (float x : delta) norm += x * x;

        for (int bits : { 32, 8, 4 })
            for (double keep : { 1.0, 0.25, 0.01 })
            {
                delta_codec_options options;
                options.bits = bits;
                options.keep_fraction = keep;
                delta_encoder encoder(options);
                compressed_delta packed;
                encoder.encode(layout, delta, packed);
                std::ostringstream sout;
                serialize(packed, sout);
                std::istringstream sin(sout.str());
                compressed_delta read;
                deserialize(read, sin);
                std::vector<float> decoded;
                decode_delta(read, decoded);
                DLIB_TEST(decoded.size() == delta.size());
                // What was sent plus what was kept is the update, exactly for error feedback.
                double gap = 0, err = 0;
                for (size_t i = 0; i < delta.size(); ++i)
                {
                    gap = std::max(gap, static_cast<double>(std::abs(decoded[i] + encoder.residual()[i] - delta[i])));
                    err += (decoded[i] - delta[i]) * (decoded[i] - delta[i]);
                }
                DLIB_TEST(gap < 1e-5);
                if (keep == 1.0)
                    DLIB_TEST(std::sqrt(err / norm) < (bits == 32 ? 1e-7 : bits == 8 ? 0.01 : 0.15));
                if (keep < 1.0)
                    DLIB_TEST(read.kept() < delta.size());

                // The residual reaches the next encoding.
                compressed_delta second;
                std::vector<float> zero(delta.size(), 0.0f), next;
                encoder.encode(layout, zero, second);
                decode_delta(second, next);
                double carried = 0;
                for (size_t i = 0; i < next.size(); ++i) carried += std::abs(next[i]);
                DLIB_TEST(keep == 1.0 && bits == 32 ? carried == 0 : carried > 0);
            }

        // A payload whose tables disagree, or whose positions leave their block, is refused.
        delta_codec_options sparse;
        sparse.keep_fraction = 0.1;
        delta_encoder encoder(sparse);
        compressed_delta packed;
        encoder.encode(layout, delta, packed);
        std::ostringstream sout;
        serialize(packed, sout);
        const std::string good = sout.str();
        int refusals = 0;
        for (size_t at = 20; at < good.size(); at += 7)
        {
            std::string bad = good;
            bad[at] = static_cast<char>(bad[at] ^ 0x5a);
            try
            {
                std::istringstream sin(bad);
                compressed_delta read;
                deserialize(read, sin);
                std::vector<float> out;
                decode_delta(read, out);
                DLIB_TEST(out.size() == layout.size());
            }
            catch (const serialization_error&) { ++refusals; }
        }
        DLIB_TEST(refusals > 0);
    }

    void test_consolidator ()
    {
        parameter_layout layout;
        layout.add_block(24, 16);
        layout.add_block(1, 16);
        dlib::rand rnd(7);
        std::vector<float> signal(layout.size());
        for (float& x : signal) x = static_cast<float>(rnd.get_random_gaussian());

        // The plain average of identical updates is the update.
        consolidation_options options;
        options.method = consolidation_method::average;
        delta_consolidator average(layout, options);
        average.add(signal, 1.0);
        average.add(signal, 3.0);
        DLIB_TEST(average.consolidate());
        for (size_t i = 0; i < signal.size(); ++i)
            DLIB_TEST(std::abs(average.proposal()[i] - signal[i]) < 1e-6);
        DLIB_TEST(average.step_size(1.0) == 1.0);

        // Nesterov: the second identical round carries the momentum of the first.
        options.method = consolidation_method::outer_momentum;
        options.outer_momentum = 0.9;
        delta_consolidator outer(layout, options);
        outer.add(signal, 1.0);
        DLIB_TEST(outer.consolidate());
        outer.commit();
        outer.add(signal, 1.0);
        DLIB_TEST(outer.consolidate());
        const float mu = static_cast<float>(options.outer_momentum);
        for (size_t i = 0; i < signal.size(); ++i)
        {
            const float b = mu * signal[i] + signal[i];
            DLIB_TEST(std::abs(outer.proposal()[i] - (signal[i] + mu * b)) < 1e-5);
        }
        DLIB_TEST(outer.round() == 1);

        // Replicas that cancel out leave no direction; replicas that agree keep theirs.
        options = consolidation_options();
        options.spectral_clip = 1.5;
        delta_consolidator full(layout, options);
        std::vector<float> opposite = signal;
        for (float& x : opposite) x = -x;
        full.add(signal, 1.0);
        full.add(opposite, 1.0);
        DLIB_TEST(full.consolidate());
        double left = 0;
        for (float x : full.proposal()) left += std::abs(x);
        DLIB_TEST(left < 1e-4);
        full.reject();
        DLIB_TEST(full.round() == 1);

        options.spectral_clip = 0;
        options.outer_momentum = 0.9;
        delta_consolidator agree(layout, options);
        agree.add(signal, 1.0);
        agree.add(signal, 1.0);
        DLIB_TEST(agree.consolidate());
        // The momentum is zero at the first round: the proposal is (1 + mu) times the update.
        for (size_t i = 0; i < signal.size(); ++i)
            DLIB_TEST(std::abs(agree.proposal()[i] - (1 + mu) * signal[i]) < 1e-4);

        // A replica far larger than the others is brought back within the norm bound.
        options.consensus_power = 0;
        options.curvature_mix = 0;
        delta_consolidator bounded(layout, options);
        std::vector<float> huge = signal;
        for (float& x : huge) x *= 1000;
        bounded.add(signal, 1.0);
        bounded.add(signal, 1.0);
        bounded.add(huge, 1.0);
        DLIB_TEST(bounded.consolidate());
        double pn = 0, sn = 0;
        for (size_t i = 0; i < signal.size(); ++i)
        {
            pn += bounded.proposal()[i] * bounded.proposal()[i];
            sn += signal[i] * signal[i];
        }
        DLIB_TEST(std::sqrt(pn) < (1 + mu) * 2 * std::sqrt(sn));

        // Curvature moves the mean towards the replica that measured more of it.
        options = consolidation_options();
        options.consensus_power = 0;
        options.spectral_clip = 0;
        options.curvature_mix = 1;
        delta_consolidator curved(layout, options);
        std::vector<float> sharp(layout.size(), 1.0f), flat(layout.size(), 1.0f);
        for (size_t i = 0; i < layout.block(1).size; ++i)
        {
            sharp[layout.block(1).offset + i] = i == 0 ? 9.0f : 1.0f;
            flat[layout.block(1).offset + i] = 1.0f;
        }
        curvature_summary cs, cf;
        cs.build(layout, sharp);
        cf.build(layout, flat);
        std::vector<float> one(layout.size(), 1.0f), zero(layout.size(), 0.0f);
        curved.add(one, 1.0, cs);
        curved.add(zero, 1.0, cf);
        DLIB_TEST(curved.consolidate());
        const size_t at = layout.block(1).offset;
        DLIB_TEST(curved.proposal()[at] > curved.proposal()[at + 1]);

        // Stale contributions are ignored; malformed ones are refused without effect.
        options = consolidation_options();
        delta_consolidator stale(layout, options);
        stale.reject();
        stale.reject();
        stale.add(signal, 1.0, curvature_summary(), 0);
        DLIB_TEST(!stale.consolidate());
        bool refused = false;
        try { stale.add(std::vector<float>(3), 1.0); }
        catch (const dlib::error&) { refused = true; }
        DLIB_TEST(refused && stale.num_contributions() == 1);
        refused = false;
        try { stale.add(signal, -1.0); }
        catch (const dlib::error&) { refused = true; }
        DLIB_TEST(refused && stale.num_contributions() == 1);

        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 0.9, 0.8, 0.95 }, { 1.0, 0.95, 0.9, 1.1 } },
            { 1, 1 }) == 2);
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 1.2, 1.3, 1.4 } }, { 1 }) == 0);
        /* Two replicas: the mean of the changes, so a step that helps one replica and harms the
           other almost as much loses to a step that helps both, and one that harms the other
           as much is not taken. */
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 0.5, 0.9 }, { 1.0, 1.4, 0.9 } }, { 1, 1 }) == 2);
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 0.5 }, { 1.0, 1.5 } }, { 1, 1 }) == 0);
        // Three replicas: the median, so a replica reporting an extreme loss is outvoted.
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 0.9, 0.95 }, { 1.0, 0.92, 0.96 }, { 1.0, 1.0, -100.0 } },
            { 1, 1, 1 }) == 1);
        /* Four replicas: the mean of the two middle changes. A step two replicas find slightly
           better and two much worse is not taken; a replica reporting an extreme gain does not
           have a step taken that two of the three others find worse, and one reporting an
           extreme loss does not have the round rejected when the three others find the step
           better. */
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 0.7 }, { 1.0, 0.99 }, { 1.0, 1.5 }, { 1.0, 1.6 } },
            { 1, 1, 1, 1 }) == 0);
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, -1e9 }, { 1.0, 0.99 }, { 1.0, 1.4 }, { 1.0, 1.5 } },
            { 1, 1, 1, 1 }) == 0);
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 1e9 }, { 1.0, 0.8 }, { 1.0, 0.9 }, { 1.0, 0.95 } },
            { 1, 1, 1, 1 }) == 1);
        /* A candidate loss that is not finite counts as an infinite increase: the median of the
           first candidate is +0.2 and the second one, whose median is -0.04, is chosen. A
           replica whose current loss is not finite is left out of every candidate. */
        const double infinite = std::numeric_limits<double>::infinity();
        const double not_a_number = std::numeric_limits<double>::quiet_NaN();
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, 0.1, 0.96 }, { 1.0, 1.2, 0.95 },
            { 1.0, infinite, 0.97 } }, { 1, 1, 1 }) == 2);
        DLIB_TEST(delta_consolidator::choose_multiplier({ { 1.0, not_a_number }, { 1.0, 0.9 } }, { 1, 1 }) == 0);
        DLIB_TEST(delta_consolidator::choose_multiplier({ { not_a_number, 2.0 }, { 1.0, 0.9 }, { 1.0, 0.95 } },
            { 1, 1, 1 }) == 1);

        // The state of a consolidator survives serialization.
        agree.commit();
        std::ostringstream sout;
        serialize(agree, sout);
        std::istringstream sin(sout.str());
        delta_consolidator read;
        deserialize(read, sin);
        DLIB_TEST(read.round() == agree.round());
        DLIB_TEST(read.momentum() == agree.momentum());
        DLIB_TEST(read.layout() == layout);

        // The step is deterministic and stays within the trust ratio.
        std::vector<float> params(layout.size(), 0.5f), again(layout.size(), 0.5f);
        apply_consolidated_step(layout, params, signal, 10.0, 0.1);
        apply_consolidated_step(layout, again, signal, 10.0, 0.1);
        DLIB_TEST(parameter_digest(params) == parameter_digest(again));
        for (const parameter_block& b : layout.blocks())
        {
            double moved = 0, base = 0;
            for (size_t i = 0; i < b.size; ++i)
            {
                moved += (params[b.offset + i] - 0.5) * (params[b.offset + i] - 0.5);
                base += 0.25;
            }
            DLIB_TEST(std::sqrt(moved) <= 0.1 * std::sqrt(base) * 1.0001);
        }

        DLIB_TEST(std::abs(scale_adam_beta2(0.95, 1, 8) - std::pow(0.95, 1.0 / 8)) < 1e-12);
        std::vector<float> d = signal;
        dlib::rand nr(1);
        privatize_delta(d, 1.0, 0.0, nr);
        double dn = 0;
        for (float x : d) dn += x * x;
        DLIB_TEST(std::abs(std::sqrt(dn) - 1.0) < 1e-4);
    }


    // Payloads and statistics that come from another host, forged or degenerate.
    void test_untrusted_inputs ()
    {
        compressed_delta last;
        auto forged = [&last](uint64 dimension, int32 bits, std::vector<uint64> sizes, std::vector<uint64> kept,
                              std::vector<float> scales, std::vector<unsigned char> codes)
        {
            std::ostringstream sout;
            serialize("compressed_delta", sout);
            serialize(dimension, sout);
            serialize(bits, sout);
            serialize(sizes, sout);
            serialize(kept, sout);
            serialize(std::vector<unsigned char>(), sout);
            serialize(scales, sout);
            serialize(codes, sout);
            std::istringstream sin(sout.str());
            try { deserialize(last, sin); }
            catch (const serialization_error&) { return false; }
            return true;
        };
        const float fmax = std::numeric_limits<float>::max();
        // Blocks larger than the dimension they claim, and scales whose codes overflow a float.
        DLIB_TEST(!forged(100, 8, { 150 }, { 150 }, std::vector<float>(5, 1.0f), std::vector<unsigned char>(150)));
        DLIB_TEST(!forged(32, 8, { 32 }, { 32 }, { fmax }, std::vector<unsigned char>(32)));
        DLIB_TEST(forged(32, 8, { 32 }, { 32 }, { 1.0f }, std::vector<unsigned char>(32)));
        // The codes -128 and -8 are the largest magnitudes a payload can carry.
        DLIB_TEST(!forged(32, 8, { 32 }, { 32 }, { fmax / 127 }, std::vector<unsigned char>(32, 0x80)));
        DLIB_TEST(!forged(32, 4, { 32 }, { 32 }, { fmax / 7 }, std::vector<unsigned char>(16, 0x00)));
        DLIB_TEST(forged(32, 4, { 32 }, { 32 }, { fmax / 8 }, std::vector<unsigned char>(16, 0x00)));
        DLIB_TEST(forged(32, 8, { 32 }, { 32 }, { fmax / 128 }, std::vector<unsigned char>(32, 0x80)));
        std::vector<float> edge;
        decode_delta(last, edge);
        for (float v : edge)
            DLIB_TEST(v == -fmax);
        const compressed_delta extreme = last;
        std::vector<unsigned char> raw(4 * 32, 0);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(&raw[8], &nan, sizeof(float));
        DLIB_TEST(!forged(32, 32, { 32 }, { 32 }, {}, raw));

        parameter_layout layout;
        layout.add_block(8, 6);
        // A replica declaring an enormous weight moves the mean, never past the norm bound.
        delta_consolidator c(layout, consolidation_options());
        std::vector<float> honest(layout.size(), 0.01f), hostile(layout.size(), 50.0f);
        for (int k = 0; k < 3; ++k)
            c.add(honest, 1.0);
        c.add(hostile, 1e30);
        DLIB_TEST(c.consolidate());
        double n = 0;
        for (float v : c.proposal())
        {
            DLIB_TEST(std::abs(v) <= std::numeric_limits<float>::max());
            n += v * v;
        }
        DLIB_TEST(std::sqrt(n) <= 2.5 * 0.01 * std::sqrt(static_cast<double>(layout.size())) * 1.001);

        /* A replica at the edge of the range of a float neither vetoes the round nor moves a
           block the others leave unchanged. */
        parameter_layout pair;
        pair.add_block(1, 16);
        pair.add_block(1, 16);
        delta_consolidator edged(pair, consolidation_options());
        std::vector<float> first_only(pair.size(), 0.0f);
        for (size_t i = 0; i < 16; ++i)
            first_only[i] = 0.01f;
        for (int k = 0; k < 3; ++k)
            edged.add(first_only, 1.0);
        edged.add(extreme, 1.0);
        DLIB_TEST(edged.consolidate());
        for (size_t i = 0; i < pair.size(); ++i)
        {
            const float v = edged.proposal()[i];
            DLIB_TEST(i < 16 ? std::abs(v) <= 0.025f * 1.001f : v == 0);
        }

        // Statistics too small for a float expand to finite values.
        std::vector<float> tiny(layout.size(), std::numeric_limits<float>::denorm_min());
        tiny[3] = 1e-30f;
        curvature_summary cs;
        cs.build(layout, tiny);
        std::vector<float> expanded(layout.size());
        cs.expand_normalized(0, expanded.data());
        for (float v : expanded)
            DLIB_TEST(v >= 0 && v <= 1e6f);

        DLIB_TEST(parameter_digest({ -0.0f, 1.0f }) == parameter_digest({ 0.0f, 1.0f }));
        delta_encoder enc;
        std::vector<float> bad(layout.size(), 0.0f);
        bad[2] = std::numeric_limits<float>::quiet_NaN();
        compressed_delta out;
        bool refused = false;
        try { enc.encode(layout, bad, out); }
        catch (const dlib::error&) { refused = true; }
        DLIB_TEST(refused && enc.residual().empty());
        // A quantized value too large for a scale the decoders accept is refused as well.
        bad[2] = 0.75f * fmax;
        refused = false;
        try { enc.encode(layout, bad, out); }
        catch (const dlib::error&) { refused = true; }
        DLIB_TEST(refused && enc.residual().empty());
    }

// ----------------------------------------------------------------------------------------

    class parameter_average_tester : public tester
    {
    public:
        parameter_average_tester (
        ) :
            tester ("test_parameter_average",
                    "Runs tests on visit_learnable_parameters(), parameter_average, the parameter "
                    "vectors of a network and their consolidation across replicas.")
        {}

        void perform_test (
        )
        {
            test_visit_learnable_parameters();
            test_parameter_average();
            test_parameter_vectors();
            test_weighted_median();
            test_delta_codec();
            test_consolidator();
            test_untrusted_inputs();
        }
    } a;
}
