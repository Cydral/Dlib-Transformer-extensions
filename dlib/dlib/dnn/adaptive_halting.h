// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ADAPTIVE_HALTING_H_
#define DLIB_ADAPTIVE_HALTING_H_

#include "adaptive_halting_abstract.h"
#include "core.h"
#include "layers.h"
#include "loss.h"
#include "trainer.h"
#include "solvers.h"
#include "../rand.h"

#include <algorithm>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    /*!
        DECIDING WHEN A MODEL HAS THOUGHT ENOUGH

        A model whose computation can be repeated has to decide how many times to repeat
        it. One way is to learn the value of stopping: a small head reads a summary of
        where the computation has got to and predicts two numbers, what halting now is
        worth and what going on is worth. Stopping when the first exceeds the second turns
        the question into a two-action decision problem, and the head is trained by
        Q-learning against whether the answer came out right.

        This is deliberately not a layer. The signal that trains it is a reward, which is
        a fact about the task rather than about the network, and no layer can know whether
        an answer was correct. So the reward comes from the caller and this object owns
        only what is generic: the head, its training, the targets, the decision rule and
        the exploration that keeps a young head from stopping at the first step forever.

        The contrast with the adaptive computation layer is worth keeping in mind. That
        layer decides per position, inside a module, and is trained by a ponder cost it
        can compute for itself. This decides over whole passes, around the model, and
        cannot be trained without being told the outcome. They are not alternatives and a
        model may use both.

        HOW A CALLER DRIVES IT

            q_halting_controller ctl(opts);
            for (each example)
            {
                ctl.begin_episode();
                for (long s = 0; s < ctl.max_steps(); ++s)
                {
                    advance_the_model_by_one_pass();
                    if (ctl.decide(summarise(state_of_the_model), s)) break;
                }
                train_the_model_on_the_last_pass_only();
                ctl.finish(answer_was_correct ? 1.0f : 0.0f);
            }

        The order matters and the object checks it: decide() before finish(), and
        begin_episode() before either.

        AN INVARIANT THE CALLER MUST KEEP

        The passes before the last are context, not credit. A model whose recurrence
        carries state from one pass to the next cannot be trained on every pass: the
        gradient of pass n would run through a forward whose starting state pass n-1 has
        already overwritten. Advance with a plain forward and train on the final pass
        alone. This object cannot enforce that, since it never touches the model, but
        getting it wrong is the first thing to check when a run fails oddly.
    !*/

    struct q_halting_options
    {
        long   max_steps   = 8;
        long   min_steps    = 1;
        long   summary_dim = 64;
        double discount    = 1.0;
        double exploration = 0.1;
        double head_learning_rate = 1e-3;
        unsigned long seed = 1;
    };

// ----------------------------------------------------------------------------------------

    using q_halting_head_type = loss_mean_squared_multioutput<
        fc<2, relu<fc<64, input<matrix<float, 0, 1>>>>>>;

// ----------------------------------------------------------------------------------------

    inline matrix<float, 0, 1> summarise_state (
        const tensor& state,
        long summary_dim
    )
    {
        DLIB_CASSERT(summary_dim > 0);

        /* A fixed-width summary, so the head stays the same size whatever the model is.
           Folding the state onto that width and averaging is enough for what the head has
           to tell apart, a computation that has settled from one still moving; a summary
           that kept every position would be as large as the model it is judging. */
        matrix<float, 0, 1> v(summary_dim);
        v = 0;
        const size_t n = state.size();
        if (n == 0) return v;

        const float* p = state.host();
        for (size_t i = 0; i < n; ++i)
            v((long)(i % (size_t)summary_dim)) += p[i];

        const float per_bin = (float)((n + (size_t)summary_dim - 1) / (size_t)summary_dim);
        if (per_bin > 0) v /= per_bin;
        return v;
    }

// ----------------------------------------------------------------------------------------

    /*!
        Builds the targets of one finished episode.

        The value of halting at a step is the reward that step would have earned, which is
        known only for the step the episode actually stopped at; for the others it is what
        the head already believed, so that a step never taken teaches nothing. The value of
        continuing is the discounted best of the next step, and at the last step there is
        no next one, so it is the reward as well.

        Separated out because it is the part worth testing on its own: everything else in
        this file is bookkeeping around it.
    !*/
    inline std::vector<matrix<float, 0, 1>> q_halting_targets (
        const std::vector<matrix<float, 0, 1>>& predicted,
        long stopped_at,
        float reward,
        double discount
    )
    {
        DLIB_CASSERT(!predicted.empty());
        DLIB_CASSERT(stopped_at >= 0 && stopped_at < (long)predicted.size());
        for (const auto& q : predicted)
            DLIB_CASSERT(q.size() == 2,
                "q_halting: a step holds " << q.size() << " values, expected two");

        std::vector<matrix<float, 0, 1>> targets = predicted;
        const long last = (long)predicted.size() - 1;

        for (long i = last; i >= 0; --i)
        {
            const bool stopped_here = (i == stopped_at);
            targets[(size_t)i](0) = stopped_here ? reward : predicted[(size_t)i](0);

            if (i == last || stopped_here)
                targets[(size_t)i](1) = reward;
            else
                targets[(size_t)i](1) = (float)(discount *
                    std::max(targets[(size_t)(i + 1)](0), targets[(size_t)(i + 1)](1)));
        }
        return targets;
    }

// ----------------------------------------------------------------------------------------

    class q_halting_controller
    {
    public:
        explicit q_halting_controller (
            const q_halting_options& opts = q_halting_options()
        ) : opt(opts),
            trainer(head, adam(1e-4, 0.9, 0.999)),
            rnd(opts.seed)
        {
            DLIB_CASSERT(opt.max_steps > 0 && opt.min_steps >= 0 &&
                         opt.min_steps <= opt.max_steps);
            DLIB_CASSERT(opt.summary_dim > 0);
            trainer.set_learning_rate(opt.head_learning_rate);
            trainer.set_mini_batch_size(32);
            trainer.be_quiet();
        }

        long max_steps () const { return opt.max_steps; }

        void begin_episode ()
        {
            summaries.clear();
            predicted.clear();
            stopped_at = -1;

            /* Exploration is a floor on the number of passes, not a random action. A head
               that has learned nothing predicts nearly the same value for both choices, and
               the tie would resolve the same way every time; forcing a longer episode now
               and then is what gives it anything to learn from. */
            forced_min = opt.min_steps;
            if (opt.exploration > 0 && rnd.get_double_in_range(0, 1) < opt.exploration)
                forced_min = (long)rnd.get_integer_in_range(opt.min_steps, opt.max_steps);
        }

        /*!
            Records where the computation has got to and says whether to stop there.
        !*/
        bool decide (
            const matrix<float, 0, 1>& summary,
            long step
        )
        {
            DLIB_CASSERT(summary.size() == opt.summary_dim,
                "q_halting: the summary is " << summary.size() << " wide, the head expects "
                << opt.summary_dim);
            DLIB_CASSERT(stopped_at < 0, "q_halting: this episode has already stopped");

            summaries.push_back(summary);

            matrix<float, 0, 1> q(2);
            if (head_ready)
            {
                const auto out = head(summary);
                q(0) = out(0);
                q(1) = out(1);
            }
            else
            {
                q = 0;                       // nothing learned yet, so no opinion either way
            }
            predicted.push_back(q);

            const bool last  = (step + 1 >= opt.max_steps);
            const bool early = (step + 1 < forced_min);
            const bool stop  = last || (!early && q(0) >= q(1));
            if (stop) stopped_at = step;
            return stop;
        }

        /*!
            Closes the episode with what the answer was worth, and trains the head on it.
        !*/
        void finish (
            float reward
        )
        {
            DLIB_CASSERT(!summaries.empty(),
                "q_halting: finish() was called on an episode that never decided anything");
            if (stopped_at < 0) stopped_at = (long)summaries.size() - 1;

            const auto targets = q_halting_targets(predicted, stopped_at, reward,
                                                   opt.discount);
            trainer.train_one_step(summaries.begin(), summaries.end(),
                                   targets.begin());
            head_ready = true;

            ++episodes;
            steps_taken += stopped_at + 1;
            reward_sum  += reward;
        }

        // What the runs have looked like so far, for a caller that wants to report it.
        long   episodes_run   () const { return episodes; }
        double average_steps  () const { return episodes ? (double)steps_taken / episodes : 0; }
        double average_reward () const { return episodes ? reward_sum / episodes : 0; }
        double head_loss      () const { return trainer.get_average_loss(); }
        void   clear_average  ()       { trainer.clear_average_loss(); }

        const q_halting_head_type& get_head () const { return head; }
        q_halting_head_type&       get_head ()       { return head; }

    private:
        q_halting_options   opt;
        q_halting_head_type head;
        dnn_trainer<q_halting_head_type, adam> trainer;
        dlib::rand          rnd;

        std::vector<matrix<float, 0, 1>> summaries;
        std::vector<matrix<float, 0, 1>> predicted;
        long   stopped_at  = -1;
        long   forced_min  = 0;
        bool   head_ready  = false;

        long   episodes    = 0;
        long   steps_taken = 0;
        double reward_sum  = 0;
    };

// ----------------------------------------------------------------------------------------

    inline void serialize (const q_halting_controller& item, std::ostream& out)
    {
        serialize("q_halting_controller_1", out);
        serialize(item.get_head(), out);
    }

    inline void deserialize (q_halting_controller& item, std::istream& in)
    {
        std::string version;
        deserialize(version, in);
        if (version != "q_halting_controller_1")
            throw serialization_error("Unexpected version '" + version +
                                      "' while deserializing q_halting_controller.");
        deserialize(item.get_head(), in);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ADAPTIVE_HALTING_H_
