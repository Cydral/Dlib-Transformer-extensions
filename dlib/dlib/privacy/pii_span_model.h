// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PII_SPAN_MODEL_H_
#define DLIB_PII_SPAN_MODEL_H_

#include "pii_span_model_abstract.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <vector>
#include "../dnn.h"
#include "../serialize.h"
#include "pii_types.h"
#include "pii_dataset.h"
#include "unicode_text.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    enum class pii_span_optimizer
    {
        adamw,
        adam_atan2
    };

    struct pii_span_model_training_options
    {
        unsigned long steps = 20000;
        unsigned long batch_size = 32;
        std::vector<long> window_lengths = { 32, 64, 128, 256 };
        std::vector<double> window_weights = { 1, 2, 3, 4 };
        double learning_rate = 1e-3;
        double min_learning_rate = 1e-5;
        unsigned long warmup_steps = 500;
        double weight_decay = 0.01;
        pii_span_optimizer optimizer = pii_span_optimizer::adamw;
        bool continue_training = false;
        unsigned long report_every = 200;
        unsigned long evaluate_every = 2000;
        unsigned long evaluate_examples = 500;
        unsigned long seed = 1;
        std::vector<int> gpus = { 0 };
    };

// ----------------------------------------------------------------------------------------

    namespace privacy_impl
    {
        /* The bytes of a training example and their BIO classes, one byte each, so that a
           corpus stays in memory at about twice the size of its text. */
        struct byte_example
        {
            std::vector<unsigned char> bytes;
            std::vector<unsigned char> classes;
        };

        inline byte_example to_byte_example(const pii_example& ex)
        {
            byte_example b;
            std::vector<std::pair<std::size_t, std::size_t>> units;
            units.reserve(ex.text.size());
            b.bytes.reserve(ex.text.size());
            for (std::size_t i = 0; i < ex.text.size(); ++i)
            {
                units.push_back(std::make_pair(i, i + 1));
                b.bytes.push_back(static_cast<unsigned char>(ex.text[i]));
            }
            const std::vector<unsigned long> classes = pii_spans_to_bio(units, ex.spans);
            b.classes.reserve(classes.size());
            for (unsigned long c : classes)
            {
                DLIB_ASSERT(c < 256);
                b.classes.push_back(static_cast<unsigned char>(c));
            }
            return b;
        }

        /* Cuts windows of exactly the requested length from the training examples laid
           end to end, in a shuffled order renewed at each pass. A window continues where
           the previous one stopped, so that windows also start and end inside examples. */
        class window_sampler
        {
        public:
            window_sampler(const std::vector<byte_example>& data, std::mt19937& rng)
                : data_(data), rng_(rng)
            {
                order_.resize(data_.size());
                for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = i;
                std::shuffle(order_.begin(), order_.end(), rng_);
            }

            void next(long length, matrix<int,0,1>& x, matrix<unsigned long,0,1>& y)
            {
                x.set_size(length);
                y.set_size(length);
                for (long k = 0; k < length; ++k)
                {
                    if (offset_ > data_[order_[current_]].bytes.size())
                        advance();
                    const byte_example& ex = data_[order_[current_]];
                    if (offset_ == ex.bytes.size())
                    {
                        // The line feed that separates two examples.
                        x(k) = '\n';
                        y(k) = 0;
                        advance();
                        continue;
                    }
                    x(k) = ex.bytes[offset_];
                    y(k) = ex.classes[offset_];
                    ++offset_;
                }
                // A window that starts inside a span starts it.
                if (pii_bio_class_inside(y(0))) y(0) = y(0) - 1;
            }

        private:
            void advance()
            {
                offset_ = 0;
                if (++current_ == order_.size())
                {
                    current_ = 0;
                    std::shuffle(order_.begin(), order_.end(), rng_);
                }
            }

            const std::vector<byte_example>& data_;
            std::mt19937& rng_;
            std::vector<std::size_t> order_;
            std::size_t current_ = 0;
            std::size_t offset_ = 0;
        };

        /* The most probable class sequence under the BIO constraint, from per position log
           probabilities stored row by row (n rows of c values). Positions flagged in
           outside can only take the O class: line breaks and tabs, which no span crosses. */
        inline std::vector<unsigned long> viterbi_bio(const std::vector<float>& logp, long n, long c,
                                                      const std::vector<bool>& outside = std::vector<bool>())
        {
            auto allowed = [&](long t, long k) {
                return k == 0 || outside.empty() || !outside[static_cast<std::size_t>(t)];
            };
            std::vector<unsigned long> out(static_cast<std::size_t>(n), 0);
            if (n == 0) return out;
            const float neg = -std::numeric_limits<float>::infinity();
            std::vector<float> score(static_cast<std::size_t>(c)), next(static_cast<std::size_t>(c));
            std::vector<unsigned char> back(static_cast<std::size_t>(n * c), 0);
            for (long k = 0; k < c; ++k)
                score[k] = pii_bio_transition_allowed(0, static_cast<unsigned long>(k)) && allowed(0, k) ? logp[k] : neg;
            for (long t = 1; t < n; ++t)
            {
                for (long k = 0; k < c; ++k)
                {
                    float best = neg;
                    long arg = 0;
                    for (long p = 0; p < c; ++p)
                    {
                        if (score[p] == neg || !pii_bio_transition_allowed(static_cast<unsigned long>(p),
                                                                           static_cast<unsigned long>(k)))
                            continue;
                        if (score[p] > best) { best = score[p]; arg = p; }
                    }
                    next[k] = best == neg || !allowed(t, k) ? neg : best + logp[t * c + k];
                    back[t * c + k] = static_cast<unsigned char>(arg);
                }
                score.swap(next);
            }
            long arg = 0;
            for (long k = 1; k < c; ++k)
                if (score[k] > score[arg]) arg = k;
            for (long t = n - 1; t >= 0; --t)
            {
                out[static_cast<std::size_t>(t)] = static_cast<unsigned long>(arg);
                if (t > 0) arg = back[t * c + arg];
            }
            return out;
        }

        inline bool trim_byte(const std::string& s, std::size_t i)
        {
            const char c = s[i];
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',' || c == ';' || c == ':';
        }

        /* Moves the ends of the spans to UTF-8 character boundaries and leaves spaces,
           no-break spaces and separators outside them. */
        inline void tidy_spans(const std::string& text, std::vector<pii_span>& spans)
        {
            auto continuation = [&](std::size_t i) {
                return i < text.size() && (static_cast<unsigned char>(text[i]) & 0xC0) == 0x80;
            };
            auto nbsp_at = [&](std::size_t i) {
                return i + 1 < text.size() && static_cast<unsigned char>(text[i]) == 0xC2 &&
                       static_cast<unsigned char>(text[i + 1]) == 0xA0;
            };
            std::vector<pii_span> out;
            for (pii_span s : spans)
            {
                s.end = std::min(s.end, text.size());
                while (s.begin > 0 && continuation(s.begin)) --s.begin;
                while (s.end < text.size() && continuation(s.end)) ++s.end;
                for (bool moved = true; moved && s.begin < s.end;)
                {
                    moved = false;
                    if (trim_byte(text, s.begin)) { ++s.begin; moved = true; }
                    else if (s.end >= s.begin + 2 && nbsp_at(s.begin)) { s.begin += 2; moved = true; }
                }
                for (bool moved = true; moved && s.begin < s.end;)
                {
                    moved = false;
                    if (trim_byte(text, s.end - 1)) { --s.end; moved = true; }
                    else if (s.end >= s.begin + 2 && nbsp_at(s.end - 2)) { s.end -= 2; moved = true; }
                }
                if (!out.empty() && s.begin < out.back().end) s.begin = out.back().end;
                if (s.end > s.begin) out.push_back(s);
            }
            spans.swap(out);
        }

        /* The text without its invisible marks (soft hyphens, zero width spaces and
           joiners, byte order marks), which the model reads as it was trained, and for
           each byte of it the offset of the same byte in text, plus text.size(). */
        inline std::string visible_bytes(const std::string& text, std::vector<std::size_t>& origin)
        {
            std::string view;
            origin.clear();
            std::size_t i = 0;
            while (i < text.size())
            {
                const std::size_t at = i;
                const char32_t c = utf8_next(text, i);
                if (is_invisible_mark(c)) continue;
                for (std::size_t k = at; k < i; ++k)
                {
                    view.push_back(text[k]);
                    origin.push_back(k);
                }
            }
            origin.push_back(text.size());
            return view;
        }

        inline bool glued_word_start(const utf8_text& t, std::size_t i)
        {
            // Code point i starts a word glued to the previous one: "DupontMarie", "DUPONTMarie".
            const char32_t prev = t.before(i), next = t.at(i);
            if (is_lower(prev) && is_upper(next)) return true;
            return is_upper(prev) && is_upper(next) && is_lower(t.at(i + 1));
        }

        inline void snap_to_words(const std::string& text, std::vector<pii_span>& spans)
        {
            /* Reading bytes, the model sometimes stops a letter or two short of a word or
               starts inside one, which would leave part of the word readable once the value
               is replaced. A boundary between two word characters moves to the word limit,
               unless a change of case marks a word start there, and never across the next
               or the previous span. */
            const utf8_text t(text);
            auto index_of = [&](std::size_t byte) {
                return static_cast<std::size_t>(std::lower_bound(t.offsets.begin(), t.offsets.end(), byte) -
                                                t.offsets.begin());
            };
            auto inside_word = [&](std::size_t i) {
                return i > 0 && i < t.size() && is_word_char(t.before(i)) && is_word_char(t.at(i)) &&
                       !glued_word_start(t, i);
            };
            for (std::size_t k = 0; k < spans.size(); ++k)
            {
                pii_span& s = spans[k];
                const std::size_t lowest = k > 0 ? spans[k - 1].end : 0;
                const std::size_t highest = k + 1 < spans.size() ? spans[k + 1].begin : text.size();
                std::size_t b = index_of(s.begin), e = index_of(s.end);
                while (inside_word(e) && t.offsets[e + 1] <= highest) ++e;
                while (inside_word(b) && t.offsets[b - 1] >= lowest) --b;
                s.begin = t.offsets[b];
                s.end = t.offsets[e];
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <long EMBEDDING_DIM, long NUM_LAYERS, long NUM_HEADS>
    class pii_span_model
    {
        static_assert(EMBEDDING_DIM % NUM_HEADS == 0 && (EMBEDDING_DIM / NUM_HEADS) % 2 == 0,
                      "the head dimension must be an even divisor of the embedding dimension");
    public:
        static constexpr long num_classes = 15;
        static constexpr long vocabulary = 260;

        typedef loss_cross_entropy_per_token<linear<num_classes, rms_norm<
            gqa_transformer_unified::transformer_stack<NUM_LAYERS, EMBEDDING_DIM, NUM_HEADS, NUM_HEADS,
                embeddings<vocabulary, EMBEDDING_DIM, input<matrix<int, 0, 1>>>,
                false, 8, 3, linear, EMBEDDING_DIM / NUM_HEADS, false, false>>>> net_type;

        pii_span_model() = default;
        pii_span_model(const pii_span_model&) = delete;
        pii_span_model& operator=(const pii_span_model&) = delete;

        long window() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return window_;
        }

        void set_window(long length)
        {
            DLIB_CASSERT(16 <= length && length <= max_window);
            std::lock_guard<std::mutex> lock(mutex_);
            window_ = length;
        }

        bool is_trained() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return trained_;
        }

        size_t parameter_count() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return trained_ ? static_cast<size_t>(count_parameters(net_)) : 0;
        }

        void train(
            const std::vector<pii_example>& training_set,
            const std::vector<pii_example>& dev_set,
            const pii_span_model_training_options& options = pii_span_model_training_options(),
            std::ostream* log = nullptr
        )
        {
            DLIB_CASSERT(!training_set.empty());
            DLIB_CASSERT(pii_bio_class_count() == static_cast<unsigned long>(num_classes));
            DLIB_CASSERT(!options.window_lengths.empty() &&
                         options.window_lengths.size() == options.window_weights.size());
            DLIB_CASSERT(options.batch_size > 0 && options.learning_rate > 0 && options.min_learning_rate >= 0 &&
                         options.weight_decay >= 0);
            double weight_sum = 0;
            for (std::size_t k = 0; k < options.window_lengths.size(); ++k)
            {
                DLIB_CASSERT(16 <= options.window_lengths[k] && options.window_lengths[k] <= max_window,
                             "pii_span_model::train(): window lengths lie in [16, " << max_window << "]");
                DLIB_CASSERT(options.window_weights[k] >= 0);
                weight_sum += options.window_weights[k];
            }
            DLIB_CASSERT(weight_sum > 0, "pii_span_model::train(): every window weight is 0");
            std::vector<privacy_impl::byte_example> data;
            data.reserve(training_set.size());
            for (const pii_example& ex : training_set)
                if (!ex.text.empty()) data.push_back(privacy_impl::to_byte_example(ex));
            DLIB_CASSERT(!data.empty());

            std::lock_guard<std::mutex> lock(mutex_);
            const bool resume = options.continue_training && trained_;
            trained_ = false;
            // find() reads long texts through windows of the longest length trained on.
            window_ = *std::max_element(options.window_lengths.begin(), options.window_lengths.end());
            if (!resume)
            {
                // The layers draw their initial weights from std::rand().
                std::srand(static_cast<unsigned int>(options.seed));
                net_ = net_type();
            }
            network_context::clear_padding();
            const float decay = static_cast<float>(options.weight_decay);
            if (options.optimizer == pii_span_optimizer::adam_atan2)
            {
                dnn_trainer<net_type, adam_atan2> trainer(net_, adam_atan2(decay, 0.9f, 0.98f), options.gpus);
                run_training(trainer, data, dev_set, options, resume, log);
            }
            else
            {
                dnn_trainer<net_type, adamw> trainer(net_, adamw(decay, 0.9f, 0.98f), options.gpus);
                run_training(trainer, data, dev_set, options, resume, log);
            }
            net_.clean();
            trained_ = true;
        }

        std::vector<pii_span> find(
            const std::string& text
        ) const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            DLIB_CASSERT(trained_, "pii_span_model::find(): the model is not trained");
            return find_unlocked(text);
        }

        static constexpr long max_window = 4096;

        template <long D, long L, long H>
        friend void serialize(const pii_span_model<D, L, H>& item, std::ostream& out);
        template <long D, long L, long H>
        friend void deserialize(pii_span_model<D, L, H>& item, std::istream& in);

    private:
        template <typename trainer_type>
        void run_training(
            trainer_type& trainer,
            const std::vector<privacy_impl::byte_example>& data,
            const std::vector<pii_example>& dev_set,
            const pii_span_model_training_options& options,
            bool resumed,
            std::ostream* log
        )
        {
            trainer.set_mini_batch_size(options.batch_size);
            trainer.set_iterations_without_progress_threshold(std::numeric_limits<unsigned long>::max() / 4);
            trainer.be_quiet();
            const unsigned long steps = std::max<unsigned long>(1, options.steps);
            matrix<double,0,1> schedule(static_cast<long>(steps));
            for (unsigned long s = 0; s < steps; ++s)
            {
                double lr;
                if (s < options.warmup_steps)
                {
                    lr = options.learning_rate * static_cast<double>(s + 1) / static_cast<double>(options.warmup_steps);
                }
                else
                {
                    const double progress = static_cast<double>(s - options.warmup_steps) /
                                            std::max<double>(1.0, static_cast<double>(steps - options.warmup_steps));
                    lr = options.min_learning_rate + 0.5 * (options.learning_rate - options.min_learning_rate) *
                                                         (1 + std::cos(3.14159265358979323846 * progress));
                }
                schedule(static_cast<long>(s)) = std::max(lr, 1e-9);
            }
            trainer.set_learning_rate_schedule(schedule);

            std::mt19937 rng(static_cast<std::mt19937::result_type>(options.seed));
            privacy_impl::window_sampler sampler(data, rng);
            std::discrete_distribution<std::size_t> pick_length(options.window_weights.begin(),
                                                                options.window_weights.end());
            std::vector<matrix<int,0,1>> x(options.batch_size);
            std::vector<matrix<unsigned long,0,1>> y(options.batch_size);

            /* The evaluations on dev: the loss on windows that cover its first examples
               once, drawn the same way at every evaluation so that two values compare, and
               the exact span F1 on those examples. */
            std::vector<privacy_impl::byte_example> dev_data;
            for (std::size_t i = 0; i < dev_set.size() && i < options.evaluate_examples; ++i)
                if (!dev_set[i].text.empty()) dev_data.push_back(privacy_impl::to_byte_example(dev_set[i]));
            auto evaluate_dev = [&](unsigned long step) {
                trainer.get_net(force_flush_to_disk::no);
                double bytes = 0, mean_length = 0, weights = 0;
                for (const privacy_impl::byte_example& ex : dev_data) bytes += ex.bytes.size() + 1;
                for (std::size_t k = 0; k < options.window_lengths.size(); ++k)
                {
                    mean_length += options.window_weights[k] * options.window_lengths[k];
                    weights += options.window_weights[k];
                }
                mean_length /= weights;
                const unsigned long batches = std::max<unsigned long>(1, static_cast<unsigned long>(
                    std::ceil(bytes / (static_cast<double>(options.batch_size) * mean_length))));
                std::mt19937 dev_rng(static_cast<std::mt19937::result_type>(options.seed + 1));
                privacy_impl::window_sampler dev_sampler(dev_data, dev_rng);
                std::discrete_distribution<std::size_t> dev_length(options.window_weights.begin(),
                                                                   options.window_weights.end());
                std::vector<matrix<int,0,1>> dx(options.batch_size);
                std::vector<matrix<unsigned long,0,1>> dy(options.batch_size);
                running_stats<double> loss;
                for (unsigned long b = 0; b < batches; ++b)
                {
                    const long length = options.window_lengths[dev_length(dev_rng)];
                    for (std::size_t k = 0; k < dx.size(); ++k)
                        dev_sampler.next(length, dx[k], dy[k]);
                    loss.add(net_.compute_loss(dx.begin(), dx.end(), dy.begin()));
                }
                std::vector<std::vector<pii_span>> truth, predicted;
                for (std::size_t i = 0; i < dev_set.size() && i < options.evaluate_examples; ++i)
                {
                    truth.push_back(dev_set[i].spans);
                    predicted.push_back(find_unlocked(dev_set[i].text));
                }
                const pii_evaluation ev = evaluate_pii_spans(truth, predicted, span_match::exact, pii_fine_labels());
                *log << "step " << step << "  dev loss " << loss.mean() << "  precision " << ev.overall.precision()
                     << "  recall " << ev.overall.recall() << "  f1 " << ev.overall.f1() << std::endl;
            };
            const bool evaluations = log && !dev_data.empty() && options.evaluate_every;
            // The starting point of a complementary training.
            if (evaluations && resumed) evaluate_dev(0);

            const auto start = std::chrono::steady_clock::now();
            for (unsigned long s = 0; s < steps; ++s)
            {
                const long length = options.window_lengths[pick_length(rng)];
                for (std::size_t k = 0; k < x.size(); ++k)
                    sampler.next(length, x[k], y[k]);
                trainer.train_one_step(x, y);
                if (log && options.report_every && (s + 1) % options.report_every == 0)
                {
                    // The mean loss of the steps since the previous line.
                    const double seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                    *log << "step " << (s + 1) << "/" << steps << "  loss " << trainer.get_average_loss()
                         << "  lr " << trainer.get_learning_rate() << "  " << static_cast<long>(seconds) << " s"
                         << std::endl;
                    trainer.clear_average_loss();
                }
                if (evaluations && ((s + 1) % options.evaluate_every == 0 || s + 1 == steps))
                    evaluate_dev(s + 1);
            }
            trainer.get_net();
        }

        std::vector<pii_span> find_unlocked(const std::string& original) const
        {
            /* What is reported as personal data must not depend on the hardware present, so
               the network runs here with the arithmetic of the host. */
            npu::exact_scope single_precision;
            std::vector<std::size_t> origin;
            const std::string text = privacy_impl::visible_bytes(original, origin);
            const long n = static_cast<long>(text.size());
            if (n == 0) return std::vector<pii_span>();
            const long w = std::min<long>(window_, n);
            std::vector<long> starts;
            if (n <= window_)
            {
                starts.push_back(0);
            }
            else
            {
                const long stride = std::max<long>(1, window_ * 3 / 4);
                for (long s = 0; s + w < n; s += stride) starts.push_back(s);
                if (starts.empty() || starts.back() != n - w) starts.push_back(n - w);
            }

            const long c = num_classes;
            std::vector<float> logp(static_cast<std::size_t>(n * c), 0.0f);
            std::vector<long> owner(static_cast<std::size_t>(n), -1), centrality(static_cast<std::size_t>(n), -1);
            for (std::size_t k = 0; k < starts.size(); ++k)
                for (long q = starts[k]; q < starts[k] + w; ++q)
                {
                    const long central = std::min(q - starts[k], starts[k] + w - 1 - q);
                    if (central > centrality[q]) { centrality[q] = central; owner[q] = static_cast<long>(k); }
                }

            /* Windows go through the network a few at a time: the attention scores of a
               batch grow with the square of the window. */
            const std::size_t chunk = 16;
            network_context::clear_padding();
            for (std::size_t first = 0; first < starts.size(); first += chunk)
            {
                const std::size_t last = std::min(starts.size(), first + chunk);
                std::vector<matrix<int,0,1>> batch(last - first);
                for (std::size_t k = first; k < last; ++k)
                {
                    batch[k - first].set_size(w);
                    for (long q = 0; q < w; ++q)
                        batch[k - first](q) = static_cast<unsigned char>(text[static_cast<std::size_t>(starts[k] + q)]);
                }
                resizable_tensor x;
                net_.to_tensor(batch.begin(), batch.end(), x);
                net_.subnet().forward(x);
                const tensor& out = net_.subnet().get_output();
                const float* p = out.host();
                for (std::size_t k = first; k < last; ++k)
                    for (long q = 0; q < w; ++q)
                    {
                        const long pos = starts[k] + q;
                        if (owner[pos] != static_cast<long>(k)) continue;
                        const float* row = p + tensor_index(out, static_cast<long>(k - first), 0, q, 0);
                        float m = row[0];
                        for (long j = 1; j < c; ++j) m = std::max(m, row[j]);
                        double sum = 0;
                        for (long j = 0; j < c; ++j) sum += std::exp(static_cast<double>(row[j] - m));
                        const float lse = m + static_cast<float>(std::log(sum));
                        for (long j = 0; j < c; ++j) logp[static_cast<std::size_t>(pos * c + j)] = row[j] - lse;
                    }
            }

            std::vector<bool> outside(static_cast<std::size_t>(n), false);
            for (long q = 0; q < n; ++q)
            {
                const char b = text[static_cast<std::size_t>(q)];
                outside[static_cast<std::size_t>(q)] = b == '\n' || b == '\r' || b == '\t';
            }
            const std::vector<unsigned long> classes = privacy_impl::viterbi_bio(logp, n, c, outside);
            std::vector<double> confidence(static_cast<std::size_t>(n));
            std::vector<std::pair<std::size_t, std::size_t>> units(static_cast<std::size_t>(n));
            for (long q = 0; q < n; ++q)
            {
                units[q] = std::make_pair(static_cast<std::size_t>(q), static_cast<std::size_t>(q + 1));
                confidence[q] = std::exp(static_cast<double>(logp[static_cast<std::size_t>(q * c) + classes[q]]));
            }
            std::vector<pii_span> spans = pii_bio_to_spans(units, classes, confidence);
            privacy_impl::tidy_spans(text, spans);
            privacy_impl::snap_to_words(text, spans);
            // Back to the offsets of the original text.
            if (text.size() != original.size())
                for (pii_span& s : spans)
                {
                    s.begin = origin[s.begin];
                    s.end = origin[s.end - 1] + 1;
                }
            return spans;
        }

        bool trained_ = false;
        long window_ = 256;
        mutable net_type net_;
        mutable std::mutex mutex_;
    };

    template <long D, long L, long H>
    void serialize(const pii_span_model<D, L, H>& item, std::ostream& out)
    {
        std::lock_guard<std::mutex> lock(item.mutex_);
        serialize("pii_span_model", out);
        serialize(D, out);
        serialize(L, out);
        serialize(H, out);
        serialize(item.trained_, out);
        serialize(item.window_, out);
        if (item.trained_) serialize(item.net_, out);
    }

    template <long D, long L, long H>
    void deserialize(pii_span_model<D, L, H>& item, std::istream& in)
    {
        std::lock_guard<std::mutex> lock(item.mutex_);
        std::string name;
        deserialize(name, in);
        if (name != "pii_span_model")
            throw serialization_error("Unexpected version found while deserializing dlib::pii_span_model.");
        long d = 0, l = 0, h = 0;
        deserialize(d, in);
        deserialize(l, in);
        deserialize(h, in);
        if (d != D || l != L || h != H)
            throw serialization_error("This pii_span_model was trained with embedding dimension " + std::to_string(d) +
                                      ", " + std::to_string(l) + " layers and " + std::to_string(h) +
                                      " heads, not " + std::to_string(D) + ", " + std::to_string(L) + " and " +
                                      std::to_string(H) + ".");
        bool trained = false;
        long window = 0;
        deserialize(trained, in);
        deserialize(window, in);
        if (window < 16 || window > pii_span_model<D, L, H>::max_window)
            throw serialization_error("Invalid window length found while deserializing dlib::pii_span_model.");
        typename pii_span_model<D, L, H>::net_type net;
        if (trained) deserialize(net, in);
        item.net_ = std::move(net);
        item.window_ = window;
        item.trained_ = trained;
    }

    template <long D, long L, long H> constexpr long pii_span_model<D, L, H>::num_classes;
    template <long D, long L, long H> constexpr long pii_span_model<D, L, H>::vocabulary;
    template <long D, long L, long H> constexpr long pii_span_model<D, L, H>::max_window;

    typedef pii_span_model<64, 2, 2> pii_span_model_tiny;
    typedef pii_span_model<128, 4, 4> pii_span_model_small;
    typedef pii_span_model<256, 6, 8> pii_span_model_base;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PII_SPAN_MODEL_H_
