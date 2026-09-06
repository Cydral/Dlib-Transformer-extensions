// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_ADAPTIVE_HALTING_ABSTRACT_H_
#ifdef DLIB_ADAPTIVE_HALTING_ABSTRACT_H_

#include "core_abstract.h"
#include "layers_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct q_halting_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object holds the settings of a q_halting_controller.

                max_steps is the most passes an episode may take, and the loop stops there
                whatever the head believes. min_steps is the fewest, which keeps a head
                that has learned nothing from stopping immediately every time.

                summary_dim is the width of the vector the state is folded into before the
                head reads it. It fixes the size of the head, so the same head serves any
                model.

                discount weights what a later pass is worth against an earlier one. One
                means a correct answer is worth the same however long it took, which is
                what a task with no cost per pass wants; below one, the head prefers to
                arrive sooner.

                exploration is the probability that an episode is given a longer floor than
                min_steps, drawn uniformly up to max_steps. Without it a head whose two
                outputs start equal resolves the tie the same way every time and never sees
                what a longer episode would have earned.

                head_learning_rate is the rate the head's own trainer uses. It is separate
                from the model's, since the head is small and learns a much simpler thing.

            CONVENTION
                - max_steps > 0
                - 0 <= min_steps <= max_steps
                - summary_dim > 0
                - 0 <= exploration <= 1
        !*/

        long   max_steps          = 8;
        long   min_steps          = 1;
        long   summary_dim        = 64;
        double discount           = 1.0;
        double exploration        = 0.1;
        double head_learning_rate = 1e-3;
        unsigned long seed        = 1;
    };

// ----------------------------------------------------------------------------------------

    using q_halting_head_type = /* a two-output regression network */;
    /*!
        The head a q_halting_controller trains: it reads a summary of the model's state and
        predicts two values, what halting now is worth and what continuing is worth.
    !*/

// ----------------------------------------------------------------------------------------

    matrix<float, 0, 1> summarise_state (
        const tensor& state,
        long summary_dim
    );
    /*!
        requires
            - summary_dim > 0
        ensures
            - returns a vector of length summary_dim summarising state, by folding it onto
              that width and averaging.
            - returns a vector of zeros when state is empty.
            - The width does not depend on the model, so a head trained through this
              function fits any model without being resized.
    !*/

// ----------------------------------------------------------------------------------------

    std::vector<matrix<float, 0, 1>> q_halting_targets (
        const std::vector<matrix<float, 0, 1>>& predicted,
        long stopped_at,
        float reward,
        double discount
    );
    /*!
        requires
            - predicted.size() > 0
            - 0 <= stopped_at < predicted.size()
            - every element of predicted has size 2
        ensures
            - returns the training targets of one finished episode.
            - The value of halting is the reward at the step the episode stopped at, and
              what the head already believed at every other step, so that a step never
              taken teaches nothing.
            - The value of continuing is the discounted better of the two values at the
              next step, and the reward itself at the last step and at the step where the
              episode stopped, since neither has a next step.
    !*/

// ----------------------------------------------------------------------------------------

    class q_halting_controller
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object decides how many passes a model whose computation can be
                repeated should take, and learns that decision from whether the answers
                came out right.

                A small head reads a summary of where the computation has got to and
                predicts what halting is worth against what continuing is worth. Stopping
                when the first exceeds the second makes this a two-action decision problem,
                and the head is trained by Q-learning against a reward the caller supplies.

                It is deliberately not a layer. The reward is a fact about the task rather
                than about the network, and no layer can know whether an answer was
                correct. This object therefore owns only what is generic: the head, its
                training, the targets, the decision rule and the exploration. The model,
                and what counts as a correct answer, stay with the caller.

                The contrast with adaptive_computation_time_ is worth keeping in mind. That
                layer decides per position, inside a module, and is trained by a ponder cost
                it computes for itself. This decides over whole passes, around the model,
                and cannot be trained without being told the outcome. They are not
                alternatives, and a model may use both.

            HOW A CALLER DRIVES IT

                q_halting_controller ctl(opts);
                for (each example)
                {
                    ctl.begin_episode();
                    for (long s = 0; s < ctl.max_steps(); ++s)
                    {
                        advance_the_model_by_one_pass();
                        if (ctl.decide(summarise_state(state, opts.summary_dim), s))
                            break;
                    }
                    train_the_model_on_the_last_pass_only();
                    ctl.finish(answer_was_correct ? 1.0f : 0.0f);
                }

            AN INVARIANT THE CALLER MUST KEEP
                The passes before the last are context, not credit. A model whose
                recurrence carries state from one pass to the next cannot be trained on
                every pass: the gradient of pass n would run through a forward whose
                starting state pass n-1 has already overwritten. Advance with a plain
                forward and train on the final pass alone.

                This object cannot enforce that, since it never touches the model. It is
                the first thing to check when a run behaves oddly.

            THREAD SAFETY
                An instance is not thread safe. One episode is in flight at a time.
        !*/

    public:

        explicit q_halting_controller (
            const q_halting_options& opts = q_halting_options()
        );
        /*!
            requires
                - opts satisfies the convention of q_halting_options
            ensures
                - #max_steps() == opts.max_steps
                - #episodes_run() == 0
                - the head is untrained, so until the first finish() it has no opinion and
                  decide() stops only when the step budget runs out or the floor is met.
        !*/

        long max_steps (
        ) const;
        /*!
            ensures
                - returns the most passes an episode may take.
        !*/

        void begin_episode (
        );
        /*!
            ensures
                - discards anything left from a previous episode and draws this one's floor
                  on the number of passes, which is min_steps, or a longer draw with
                  probability exploration.
        !*/

        bool decide (
            const matrix<float, 0, 1>& summary,
            long step
        );
        /*!
            requires
                - begin_episode() has been called and finish() has not
                - summary.size() == the summary_dim this object was constructed with
                - step is the index of the pass just completed, counting from zero
            ensures
                - records where the computation has got to, together with what the head
                  believes it is worth.
                - returns whether the episode should stop here, which it does when the step
                  budget is exhausted, or when the floor has been met and halting is worth
                  at least as much as continuing.
        !*/

        void finish (
            float reward
        );
        /*!
            requires
                - decide() has been called at least once since begin_episode()
            ensures
                - builds the targets of this episode and trains the head on them.
                - #episodes_run() == episodes_run() + 1
                - the head now has an opinion, so later episodes may stop early.
        !*/

        long   episodes_run   () const;
        double average_steps  () const;
        double average_reward () const;
        double head_loss      () const;
        void   clear_average  ();
        /*!
            ensures
                - report what the episodes have looked like so far. average_steps() is the
                  figure that says whether the head has learned anything: a head that never
                  stops early sits at max_steps().
        !*/

        const q_halting_head_type& get_head () const;
        q_halting_head_type&       get_head ();
        /*!
            ensures
                - returns the head, so that it can be inspected or serialized.
        !*/
    };

    void serialize   (const q_halting_controller& item, std::ostream& out);
    void deserialize (q_halting_controller& item, std::istream& in);
    /*!
        provides serialization support for the head. The settings and the running counts
        are not serialized: they describe how a caller is driving the object rather than
        anything it has learned.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_ADAPTIVE_HALTING_ABSTRACT_H_
