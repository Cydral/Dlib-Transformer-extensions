// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Hierarchical reasoning on ARC-AGI, with three ways of deciding when to stop.

    The recurrent core of this network can be run for a variable number of passes, and
    there is more than one place to decide how many. This program puts three of those
    decisions in competition on the same data, the same architecture and the same
    budget, because the question of where adaptive computation belongs is open and the
    published results do not settle it.

        --halting external
            The recurrence is driven from outside the network. Each segment resumes the
            state the previous one left, so several forward passes compose into one
            longer chain of reasoning, and a small head predicts the value of halting
            against the value of continuing. That head is trained by Q-learning, its
            target built from whether the answer came out right. This is the arrangement
            the published work uses.

        --halting internal
            The adaptive computation layer is placed inside the modules the recurrence
            encapsulates, and it decides for itself, per position, when it has done
            enough. It is trained by a ponder cost rather than by a value estimate: the
            network pays for every step it takes and learns to take fewer. This is the
            mechanism of Graves, and it lives in the graph rather than around it.

        --halting none
            A fixed number of passes and no decision at all. Neither of the above can be
            credited with anything until this baseline is known.

    The comparison is between two complete devices, not between two placements: the
    signal that trains them differs as much as the place they sit. That is deliberate
    and it is what the results should be read as saying.

    THE CONDITIONING

    A model here is shown one input grid and no worked examples of the same task. What
    tells it which transformation to apply is an identifier, carried as the first token
    and looked up in the same embedding table as the colours. Which means an identifier
    the model never trained on carries no meaning at all, and a puzzle can only be asked
    about by name if it was named during training. Build the dataset with
    --protocol reference to allow that, or evaluate with --blank-puzzle-id to ask the
    model to work without being told, which is the harder setting and the honest one.

    Prepare the data first:
        slm_tools/build_arc_dataset.py --download --data-dir arc-agi --out-dir arc-data \
            --protocol reference --augmentations 50

    Then, for instance:
        ./slm_hrm_arc_puzzle_ex --train --data arc-data --halting external
        ./slm_hrm_arc_puzzle_ex --train --data arc-data --halting internal
        ./slm_hrm_arc_puzzle_ex --train --data arc-data --halting none
        ./slm_hrm_arc_puzzle_ex --eval  --data arc-data --halting external
*/

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace dlib;
using std::cout;
using std::endl;

// ----------------------------------------------------------------------------------------

/*
    The window is the largest ARC grid plus the identifier that precedes it. The output
    is only as wide as the colours: a model is never asked to predict an identifier, and
    a head sized for the whole table would carry a weight column and an optimizer state
    for every puzzle in the dataset.
*/
const long GRID_SIDE   = 30;
const long WINDOW      = GRID_SIDE * GRID_SIDE;
const long SEQ_LEN     = WINDOW + 1;
const long COLOUR_VOCAB = 11;

/*
    The embedding table has to hold every identifier, so its size is a property of the
    dataset rather than of the model. C++ needs it at compile time, so a few sizes are
    provided and the program picks the smallest that fits, refusing rather than
    truncating if none does. Raising the ceiling is a matter of adding one line here.
*/
/*
    One table size rather than two. Each is a template parameter and therefore a whole
    instantiation of the network, and the dense control doubles those already. An oversized
    table occupies device memory but no optimizer state and no arithmetic, since the
    embedding updates only the rows a batch touched, so the waste is bounded where the
    build time would not have been.
*/
const long TABLE_ROWS = 262144;

/*
    The network follows the reference configuration in its shape: four layers in each of
    the two modules, two cycles of each, eight query heads, and a feed-forward expansion of
    four. Two things depart from it, both deliberately.

    Attention is grouped-query with two key-value heads rather than eight. And the width is
    320 rather than 512, which takes the count from 22.1 million to 8.6 million against the
    27 million reported there.

    The width is where the reasoning happens and it costs the square, so cutting it is what
    makes a four-way comparison affordable: the step runs 2.6 times faster. What it does
    not cut is the capacity to hold a puzzle, which lives in the identifier table, 320 wide
    by half a million rows and outside the parameter count entirely. Whether reasoning
    survives the cut better than memorisation would is the assumption this configuration
    rests on, and the comparison against the published figure is what tests it.

    The table sizes are template parameters, so every one of them costs an instantiation of
    a network this size. Two are provided rather than four: an oversized table occupies
    device memory but no optimizer state and no arithmetic, since the embedding updates
    only the rows a batch touched, so the waste is bounded and the build time is not.
*/
const long NUM_H_LAYERS = 4;
const long NUM_L_LAYERS = 4;
const long NUM_HEADS    = 8;
const long NUM_KV_HEADS = 2;
const long EMBED_DIM    = 320;
const long HRM_N        = 2;
const long HRM_T        = 2;

template <long TABLE, bool USE_ACT>
using arc_config = hrm_transformer_config<
    TABLE, NUM_H_LAYERS, NUM_L_LAYERS, NUM_HEADS, EMBED_DIM, HRM_N, HRM_T,
    gelu, dropout_10, attention_impl::unified, NUM_KV_HEADS, USE_ACT, COLOUR_VOCAB>;

/*
    The control: the same blocks, stacked, with nothing of the hierarchical layer.

    Three things separate the two networks and only a straight stack removes all three.
    The recurrent layer starts its state from vectors it has learned, combines its inputs
    by addition rather than by stacking, and carries gradient through its last pass alone,
    which is the one-step approximation the architecture rests on. Shortening the
    recurrence leaves all three in place, so a run that plateaus with it and also without
    it says nothing.

    This stack holds eight blocks, four and four as the two modules do, at the same width
    and over the same embedding table, and the gradient reaches every one of them. If it
    learns where the recurrent network stops, the recurrence is what to look at; if it
    stops in the same place, the cause is elsewhere and an expensive avenue is closed.
*/
template <long TABLE, bool USE_ACT>
using dense_stack = typename impl::hrm_stack_selector<
    attention_impl::unified, NUM_H_LAYERS + NUM_L_LAYERS, EMBED_DIM, NUM_HEADS,
    NUM_KV_HEADS, gelu, dropout_10,
    embeddings<TABLE, EMBED_DIM, input<matrix<int, 0, 1>>>, USE_ACT>::type;

template <long TABLE, bool USE_ACT>
using dense_net = classification_head<COLOUR_VOCAB, dense_stack<TABLE, USE_ACT>>;

// ----------------------------------------------------------------------------------------

/* Raises the rate of an embedding layer and leaves every other layer alone. */
template <typename layer_type>
auto set_embedding_rate (layer_type& l, double m, int)
    -> decltype(l.get_scale_by_freq(), void())
{
    l.set_learning_rate_multiplier(m);
}
template <typename layer_type>
void set_embedding_rate (layer_type&, double, long) {}

template <typename layer_type>
void set_embedding_rate (layer_type& l, double m) { set_embedding_rate(l, m, 0); }

enum class halting_mode { external, internal, both, none };

/* The two mechanisms are not alternatives. The adaptive layer decides, per position and
   inside a module, how many refinement steps that position needs; the outer loop decides
   how many whole segments the recurrence runs. Nothing makes them exclusive, so the pair
   is a fourth arrangement and not a contradiction. */
inline bool uses_act  (halting_mode m)
{ return m == halting_mode::internal || m == halting_mode::both; }
inline bool uses_loop (halting_mode m)
{ return m == halting_mode::external || m == halting_mode::both; }

inline std::string describe (halting_mode m)
{
    switch (m)
    {
        case halting_mode::external:
            return "external, a value head trained by Q-learning around whole segments";
        case halting_mode::internal:
            return "internal, an adaptive layer trained by a ponder cost";
        case halting_mode::both:
            return "both, the adaptive layer inside the modules and the outer loop around them";
        default:
            return "none, a single pass and no decision";
    }
}

// ----------------------------------------------------------------------------------------

enum class reward_kind { exact, cells };

/*
    What a predicted window was worth. This is the reward the halting controller is trained
    against, and it is the part a library cannot supply: only the caller knows what a right
    answer looks like.

    Two ways of saying it, and the choice is not cosmetic. The published work scores a grid
    exactly right or not at all, which is the measure anyone reporting ARC results has to
    use. But a reward that is zero on every episode teaches a value head nothing: every
    target becomes zero, the head converges on zero, and its two outputs then differ only
    by noise, so the decision to stop becomes a coin flip. That is what happens here at the
    start, when no grid is yet exact.

    Scoring the fraction of cells that came out right gives the head a gradient from the
    first episode. It is not the published measure and no reported number should be built
    on it; it is what makes the mechanism trainable at all before the model is good enough
    for the published one to say anything.
*/
inline float grid_reward (const std::vector<unsigned long>& predicted,
                          const matrix<unsigned long, 0, 1>& label,
                          reward_kind kind)
{
    const long n = std::min<long>((long)predicted.size(), label.size());
    if (n <= 0) return 0.0f;

    long right = 0;
    for (long i = 0; i < n; ++i)
        if (predicted[(size_t)i] == label(i)) ++right;

    if (kind == reward_kind::exact)
        return right == n ? 1.0f : 0.0f;
    return (float)right / (float)n;
}

// ----------------------------------------------------------------------------------------

/*
    Finds the hierarchical layer inside a network so the external loop can tell it to
    resume rather than restart. The index is a property of the network type and is
    resolved at compile time by the layer visitor.
*/
/* The overloads have to be declared before the visitor that calls them, since a name
   found only by argument-dependent lookup would not be found here: the layer types live
   in dlib and these functions do not. */
template <typename layer_type>
auto set_carry_if_hierarchical (layer_type& l, bool on, int)
    -> decltype(l.layer_details().set_carry_state(on), void())
{
    l.layer_details().set_carry_state(on);
}
template <typename layer_type>
void set_carry_if_hierarchical (layer_type&, bool, long) {}

template <typename layer_type>
auto reset_if_hierarchical (layer_type& l, int)
    -> decltype(l.layer_details().reset_state(), void())
{
    l.layer_details().reset_state();
}
template <typename layer_type>
void reset_if_hierarchical (layer_type&, long) {}

template <typename net_type>
void set_carry (net_type& net, bool on)
{
    visit_layers(net, [on](size_t, auto& l) { set_carry_if_hierarchical(l, on, 0); });
}

template <typename net_type>
void reset_state (net_type& net)
{
    visit_layers(net, [](size_t, auto& l) { reset_if_hierarchical(l, 0); });
}

// ----------------------------------------------------------------------------------------

struct run_options
{
    std::string  data_dir;
    std::string  model_file;
    halting_mode halting     = halting_mode::none;
    long         max_steps   = 16;
    long         batch_size  = 8;
    long         max_epochs  = 50;
    double       learning_rate = 1e-4;
    double       ponder_cost = 0.01;
    bool         blank_id    = false;
    long         summary_dim = 64;
    long         eval_every  = 5;
    long         eval_samples = 2000;
    long         patience    = 1000000;
    bool         anneal      = false;
    double       embedding_rate = 100.0;
    bool         dense       = false;
    long         report_every = 200;
    long         steps_per_epoch = 0;
    bool         all_examples = false;
    reward_kind  reward      = reward_kind::cells;
    bool         verbose     = false;
};

// ----------------------------------------------------------------------------------------

template <typename net_type>
double evaluate (net_type& net, const arc_dataset& d, const run_options& o,
                 bool quiet, q_halting_controller* halting, long limit = 0)
{
    long exact = 0, cells_right = 0, cells_total = 0;

    /*
        A periodic measurement reads a sample, the final one reads everything.

        The held-out split holds twenty thousand examples and an arrangement that composes
        sixteen segments runs the network sixteen times for each of them, which is two
        hours per measurement. Taken every ten epochs that is more time spent measuring
        than training. A few thousand examples settle a cell accuracy to well within the
        differences being looked for, and the figure that gets reported is the full one.
    */
    const long n = (limit > 0 && limit < d.num_examples()) ? limit : d.num_examples();
    const long stride = std::max(1L, d.num_examples() / std::max(1L, n));

    for (long k = 0; k < n && !signal_handler::is_triggered(); ++k)
    {
        const long i = (k * stride) % d.num_examples();
        const int pid = o.blank_id ? arc_blank_puzzle_id : d.identifier_of(i);
        const auto seq = arc_make_input(d, i, pid);

        /* Only the arrangements that compose segments run the network more than once,
           and only they need the state to survive between those runs. */
        std::vector<matrix<int, 0, 1>> one(1, seq);
        resizable_tensor batch;
        net.to_tensor(one.begin(), one.end(), batch);

        if (uses_loop(o.halting))
        {
            /* The head decides here too. Running a fixed number of segments would measure
               the budget rather than the decision, and the thing that was trained would
               never be used. A null controller falls back to the budget, which is all an
               evaluation of a model loaded from disk with no head beside it can do. */
            reset_state(net);
            set_carry(net, true);
            if (halting) halting->begin_episode();
            for (long s = 0; s < o.max_steps; ++s)
            {
                net.subnet().forward(batch);
                if (halting && halting->decide(
                        summarise_state(net.subnet().get_output(), o.summary_dim), s))
                    break;
            }
            set_carry(net, false);
        }
        else
        {
            net.subnet().forward(batch);
        }
        const tensor& logits = net.subnet().get_output();
        const auto pred = arc_predict_sequence(logits.host(), logits.nc(),
                                               SEQ_LEN, COLOUR_VOCAB);

        const auto lab = arc_make_label(d, i);
        std::vector<unsigned long> want_seq(lab.begin(), lab.end());
        const auto got  = arc_decode_grid(pred, WINDOW, GRID_SIDE);
        const auto want = arc_decode_grid(want_seq, WINDOW, GRID_SIDE);

        if (got.nr() == want.nr() && got.nc() == want.nc())
        {
            long wrong = 0;
            for (long r = 0; r < got.nr(); ++r)
                for (long c = 0; c < got.nc(); ++c)
                {
                    ++cells_total;
                    if (got(r, c) == want(r, c)) ++cells_right; else ++wrong;
                }
            if (wrong == 0) ++exact;
        }
        else
        {
            cells_total += want.nr() * want.nc();
        }
    }

    const double grids = n ? 100.0 * exact / n : 0.0;
    const double cells = cells_total ? 100.0 * cells_right / cells_total : 0.0;
    if (!quiet)
    {
        cout << "  exact grids   : " << exact << " of " << n
             << "  (" << grids << " %)\n";
        cout << "  cells correct : " << cells << " %\n";
    }
    /* Cells rather than grids, because on a model that has yet to produce one exact grid
       the grid count is zero for every arrangement and orders nothing. */
    return cells;
}

// ----------------------------------------------------------------------------------------

template <typename net_type>
int run (const arc_dataset& train, const arc_dataset& eval_set, const run_options& o,
         bool do_train, bool do_eval)
{
    net_type net;
    double best_accuracy = -1;
    long   best_epoch    = 0;
    const bool quiet     = true;
    q_halting_options qopt;
    qopt.max_steps   = o.max_steps;
    qopt.min_steps   = 1;
    qopt.summary_dim = o.summary_dim;
    q_halting_controller halting(qopt);

    if (file_exists(o.model_file))
    {
        /* A checkpoint carries the shape it was trained at. Loading one written by a
           different width leaves the weights mismatched and the failure surfaces deep
           inside an attention layer, where nothing names the cause. Saying it here costs
           one try block. */
        cout << "Loading " << o.model_file << "\n";
        try
        {
            deserialize(o.model_file) >> net;
        }
        catch (const std::exception& e)
        {
            cout << "\n" << o.model_file << " does not fit this network. It was most likely "
                    "written by a\nrun of a different width or depth; the current one is "
                 << NUM_H_LAYERS << " and " << NUM_L_LAYERS << " layers at width "
                 << EMBED_DIM << ".\nRemove it, or point --model-file elsewhere.\n\n  "
                 << e.what() << "\n";
            return 1;
        }
    }
    cout << "Parameters: " << count_network_parameters(net, SEQ_LEN) << "\n";

    if (do_train)
    {
        dnn_trainer<net_type, adam> trainer(net, adam(1e-4, 0.9, 0.999));
        trainer.set_learning_rate(o.learning_rate);
        trainer.set_mini_batch_size((size_t)o.batch_size);
        trainer.set_synchronization_file("chkpt-" + o.model_file, std::chrono::minutes(15));
        trainer.be_quiet();

        /* The synchronization file carries the trainer's state, the learning rate among it,
           and restoring it overwrites the rate set above. A run resumed after the rate had
           been lowered would otherwise continue at the lowered one, silently. */
        trainer.set_learning_rate(o.learning_rate);

        /* The trainer lowers its rate when the loss stops falling, which assumes the loss
           is measured on the same thing from one step to the next. Here an epoch draws a
           different variant of each task, so the loss jumps whenever the draw changes and
           the heuristic reads that as a plateau: the rate reached one in a million by the
           fourth epoch on a corpus it had barely begun to learn. The schedule is therefore
           explicit, held flat and stepped down by the caller. */
        trainer.set_iterations_without_progress_threshold(o.patience);
        trainer.set_min_learning_rate(o.learning_rate * 1e-3);

        /* Raising the patience was not enough: the rate still fell at the third epoch,
           after five thousand steps against a threshold of two hundred thousand, so
           something other than that counter lowered it. Rather than guess which, the
           shrink itself is neutralised. A factor of one leaves the multiplication that
           performs it without effect, whichever counter calls for it.

           This is the right default here for a reason beyond the bug. The loss of one
           epoch is measured on a different draw from the loss of the next, so a schedule
           that reads a plateau in it is reading noise. --anneal restores the usual
           behaviour for a run whose draw is fixed. */
        if (!o.anneal)
            trainer.set_learning_rate_shrink_factor(1);

        /* The identifier embedding needs a rate of its own.

           It receives gradient through one position of nine hundred and one, and only by
           way of attention from the rest, so at the network's rate it barely moves. The
           reference gives the puzzle embedding a rate a hundred times the base one, which
           is not incidental: it is what makes the conditioning learnable at all. Here the
           colours share the table, and the layer already scales each row's update by how
           often it was seen, which keeps the frequent rows from running away. */
        visit_computational_layers(trainer.get_net(force_flush_to_disk::no),
                                   [&](auto& l) { set_embedding_rate(l, o.embedding_rate); });

        std::mt19937 rng(1);

        /*
            An epoch draws one variant of each task rather than every variant of every one.

            A corpus augmented three hundred times holds two million examples, and running
            all of them before measuring anything would put a held-out figure fourteen
            hours away. The group index says which puzzles are variants of the same task,
            so drawing one per group gives an epoch of a few thousand examples in which
            every task appears exactly once. A heavily augmented task then carries no more
            weight than any other, which is the reason the index exists.

            --all-examples runs the whole set instead, which is what a final pass wants.
        */
        auto draw_epoch = [&](std::vector<long>& order) {
            order.clear();
            if (o.all_examples || train.num_groups() <= 0)
            {
                order.resize((size_t)train.num_examples());
                std::iota(order.begin(), order.end(), 0);
            }
            else
            {
                for (long g = 0; g + 1 < (long)train.group_indices.size(); ++g)
                {
                    const long first = train.group_indices[(size_t)g];
                    const long last  = train.group_indices[(size_t)(g + 1)];
                    if (last <= first) continue;
                    const long p = first + (long)(rng() % (unsigned long)(last - first));
                    for (long e = train.puzzle_indices[(size_t)p];
                         e < train.puzzle_indices[(size_t)(p + 1)]; ++e)
                        order.push_back(e);
                }
            }
            std::shuffle(order.begin(), order.end(), rng);

            /* An epoch is the unit on which the rate, the held-out measurement and the
               best model all hang. On a corpus this size it runs to fifty thousand steps,
               which puts all three out of reach for a day, so the caller may cut it. */
            const size_t cap = (size_t)o.steps_per_epoch * (size_t)o.batch_size;
            if (o.steps_per_epoch > 0 && order.size() > cap)
                order.resize(cap);
        };

        std::vector<long> order;

        cout << "\nTraining, halting " << describe(o.halting) << "\n";
        for (long epoch = 0; epoch < o.max_epochs && !signal_handler::is_triggered(); ++epoch)
        {
            draw_epoch(order);
            long steps_done = 0;
            const long steps_this_epoch = (long)(order.size() / (size_t)o.batch_size);
            auto last_report = std::chrono::steady_clock::now();
            std::shuffle(order.begin(), order.end(), rng);

            std::vector<matrix<int, 0, 1>>           xs;
            std::vector<matrix<unsigned long, 0, 1>> ys;

            for (size_t k = 0; k < order.size() && !signal_handler::is_triggered(); ++k)
            {
                const long i = order[k];
                const int pid = o.blank_id ? arc_blank_puzzle_id : train.identifier_of(i);
                xs.push_back(arc_make_input(train, i, pid));
                ys.push_back(arc_make_label(train, i));

                if ((long)xs.size() < o.batch_size) continue;

                if (uses_loop(o.halting))
                {
                    /* Only the segment that produced the answer carries the gradient. The
                       earlier ones advance the state and nothing else, which is the
                       one-step approximation the recurrence is built around: running a
                       training step per segment would ask the trainer to back-propagate
                       through a forward whose starting state its own previous step had
                       already overwritten. */
                    net_type& live = trainer.get_net(force_flush_to_disk::no);
                    reset_state(live);
                    set_carry(live, true);

                    resizable_tensor batch;
                    live.to_tensor(xs.begin(), xs.end(), batch);

                    /* The reward is read from the pass the decision was taken on, not
                       from the training step that follows. The trainer runs its own copy
                       of the network on its own thread and cleans its tensors when it
                       pleases, so what it holds after a step is not something a caller
                       may read: asking anyway returns an output of width zero. The
                       question the head is being asked is in any case about the state it
                       decided on, before the extra pass. */
                    halting.begin_episode();
                    std::vector<unsigned long> got;
                    bool decided = false;

                    for (long s = 0; s + 1 < o.max_steps; ++s)
                    {
                        live.subnet().forward(batch);
                        const tensor& out = live.subnet().get_output();
                        if (out.nc() >= COLOUR_VOCAB && out.size() > 0)
                            got = arc_predict_sequence(out.host(), out.nc(),
                                                       SEQ_LEN, COLOUR_VOCAB);
                        decided = true;
                        if (halting.decide(summarise_state(out, qopt.summary_dim), s))
                            break;
                    }

                    trainer.train_one_step(xs, ys);

                    if (decided && !got.empty())
                        halting.finish(grid_reward(got, ys.front(), o.reward));
                    set_carry(live, false);
                }
                else
                {
                    trainer.train_one_step(xs, ys);
                }

                xs.clear(); ys.clear();
                ++steps_done;

                /* A line every so often, because an epoch on an augmented corpus is tens
                   of thousands of steps and a program that says nothing for ten hours
                   cannot be told from one that has hung. The rate is measured over the
                   interval rather than the run, so it reflects what the machine is doing
                   now. */
                if (o.report_every > 0 && steps_done % o.report_every == 0)
                {
                    const auto now = std::chrono::steady_clock::now();
                    const double dt = std::chrono::duration<double>(now - last_report).count();
                    const double per = dt > 0 ? o.report_every / dt : 0;
                    const long   left = steps_this_epoch - steps_done;
                    cout << "  step " << steps_done << "/" << steps_this_epoch
                         << "  loss " << trainer.get_average_loss()
                         << "  " << per << " steps/s";
                    if (per > 0 && left > 0)
                        cout << "  " << (long)(left / per / 60) << " min left in the epoch";
                    cout << "\n" << std::flush;
                    last_report = now;
                }
            }

            cout << "epoch " << (epoch + 1) << "/" << o.max_epochs
                 << "  average loss " << trainer.get_average_loss()
                 << "  lr " << trainer.get_learning_rate();
            if (uses_loop(o.halting))
                cout << "  |  segments " << halting.average_steps()
                     << "  reward " << halting.average_reward()
                     << "  head loss " << halting.head_loss();
            cout << "\n";
            trainer.clear_average_loss();
            halting.clear_average();

            /* A training loss that falls while held-out accuracy falls with it is what
               overfitting looks like, and on a dataset of a few thousand examples it
               arrives early. Measuring on the held-out split as the run goes, and keeping
               the model that scored best rather than the one the run ended on, is what
               makes the final number mean something. Without it a comparison between
               arrangements compares how fast each one overfits. */
            if (eval_set.num_examples() > 0 && o.eval_every > 0 &&
                (epoch + 1) % o.eval_every == 0)
            {
                /* The measurement runs on a copy. Running it on the trainer's own
                   network resized that network's tensors from the training batch to a
                   single example and back, and cleaned a network still being trained;
                   under the memory extension that churn left a block stale on the device
                   with no host copy to refresh it from, and the next prefetch failed
                   inside the copy. A copy costs the parameters and the embedding table
                   once per measurement and leaves the trainer's state untouched. */
                net_type probe = trainer.get_net(force_flush_to_disk::no);
                const double acc = evaluate(probe, eval_set, o, quiet,
                                            uses_loop(o.halting) ? &halting : nullptr,
                                            o.eval_samples);
                cout << "         held out " << acc << " % of cells";
                if (acc > best_accuracy)
                {
                    best_accuracy = acc;
                    best_epoch    = epoch + 1;
                    probe.clean();
                    serialize(o.model_file + ".best") << probe;
                    cout << ", best so far";
                }
                cout << "\n";
            }

            if (trainer.get_learning_rate() < 1e-7) break;
        }

        trainer.get_net();
        net.clean();
        serialize(o.model_file) << net;
        cout << "Model saved to " << o.model_file << "\n";

        /* The memory report belongs to a run that is measuring the extension, not to one
           that is training a model. It was added while the working set of this example was
           being sized and it has no business in a training trace. */
        if (o.verbose && extended_memory_enabled())
        {
            cout << "\n";
            print_extended_memory_stats(cout);
        }
    }

    if (do_eval && eval_set.num_examples() > 0)
    {
        cout << "\nEvaluating on " << eval_set.num_examples() << " held-out examples, "
             << (o.blank_id ? "without the identifier" : "with the identifier") << "\n";
        if (best_accuracy >= 0 && file_exists(o.model_file + ".best"))
        {
            cout << "  using the model of epoch " << best_epoch
                 << ", which scored best while training\n";
            deserialize(o.model_file + ".best") >> net;
        }
        evaluate(net, eval_set, o, false, uses_loop(o.halting) ? &halting : nullptr);
        if (uses_loop(o.halting))
            cout << "  segments used : " << halting.average_steps() << " of "
                 << o.max_steps << "\n";
    }
    return 0;
}

// ----------------------------------------------------------------------------------------

/*
    Picks the smallest embedding table that holds the dataset's identifiers, and refuses
    rather than truncating when none does: a puzzle whose identifier does not fit would
    silently collide with another one's embedding.
*/
template <bool USE_ACT>
int dispatch_table (const arc_dataset& train, const arc_dataset& eval_set,
                    const run_options& o, bool do_train, bool do_eval)
{
    const long need = std::max(train.colour_vocab + train.largest_identifier() + 1,
                               eval_set.num_examples() > 0
                                   ? eval_set.colour_vocab + eval_set.largest_identifier() + 1
                                   : 0L);
    if (need > TABLE_ROWS)
    {
        cout << "The dataset needs an embedding table of " << need << " rows, above the "
             << TABLE_ROWS << " this program is built for.\nRebuild the dataset with fewer "
                "augmentations, or raise TABLE_ROWS and rebuild.\n";
        return 1;
    }
    return o.dense
        ? run<dense_net<TABLE_ROWS, USE_ACT>>(train, eval_set, o, do_train, do_eval)
        : run<typename arc_config<TABLE_ROWS, USE_ACT>::template network_type<true>>(
              train, eval_set, o, do_train, do_eval);
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("h", "Display this help message");
        parser.add_option("train", "Train the model");
        parser.add_option("eval", "Evaluate on the held-out split");
        parser.add_option("data", "Directory holding the prepared dataset "
                                  "(default: arc-data)", 1);
        parser.add_option("model-file", "Model file path (default: hrm_arc_model.dat)", 1);
        parser.add_option("halting", "Where the decision to stop is taken: external, "
                                     "internal, both or none. The two are not alternatives: "
                                     "one works per position inside a module, the other over "
                                     "whole segments (default: none)", 1);
        parser.add_option("max-steps", "Segments the external loop may run, which the "
                                       "reference sets to sixteen (default: 16)", 1);
        parser.add_option("batch-size", "Mini-batch size (default: 8)", 1);
        parser.add_option("max-epochs", "Maximum number of epochs (default: 50)", 1);
        parser.add_option("learning-rate", "Base learning rate (default: 1e-4)", 1);
        parser.add_option("ponder-cost", "What the internal layer pays per step "
                                         "(default: 0.01)", 1);
        parser.add_option("reward", "What an episode is worth to the halting head: exact, "
                                    "the published measure, one when the whole grid is "
                                    "right and zero otherwise; or cells, the fraction of "
                                    "positions that came out right, which is what gives "
                                    "the head anything to learn from before any grid is "
                                    "exact (default: cells)", 1);
        parser.add_option("blank-puzzle-id", "Withhold the identifier, so the model is "
                                             "asked to work without being told the rule");
        parser.add_option("steps-per-epoch", "Cut an epoch to N steps, so that the rate "
                                             "schedule, the held-out measurement and the "
                                             "best model come round at a reachable "
                                             "interval, 0 for the whole draw (default: 0)", 1);
        parser.add_option("report-every", "Print a line every N steps, 0 to print only at "
                                          "the end of an epoch. An epoch on an augmented "
                                          "corpus runs to tens of thousands of steps "
                                          "(default: 200)", 1);
        parser.add_option("dense", "Replace the hierarchical layer by a straight stack of "
                                   "the same blocks, at the same width and over the same "
                                   "embedding table, with gradient reaching all of them");
        parser.add_option("embedding-rate", "Learning rate multiplier for the identifier "
                                            "embedding, which receives gradient through one "
                                            "position of nine hundred and one. The reference "
                                            "gives it a hundred times the base rate "
                                            "(default: 100)", 1);
        parser.add_option("anneal", "Let the trainer lower its rate when it reads a plateau. "
                                    "Off by default: an epoch here draws a different variant "
                                    "of each task, so the loss is not comparable from one to "
                                    "the next and a plateau in it is noise");
        parser.add_option("patience", "Steps the trainer will accept without apparent "
                                      "progress before lowering its rate. The default is "
                                      "large because an epoch here draws a different "
                                      "variant of each task, so the loss is not comparable "
                                      "from one epoch to the next (default: 1000000)", 1);
        parser.add_option("eval-samples", "Examples read by a periodic measurement, spread "
                                          "across the split, 0 for all of them. The final "
                                          "measurement always reads everything (default: 2000)", 1);
        parser.add_option("all-examples", "Run every variant of every task in an epoch "
                                          "instead of drawing one variant per task, which "
                                          "on an augmented corpus is two million examples");
        parser.add_option("eval-every", "Measure on the held-out split every N epochs and "
                                        "keep the model that scores best, 0 to switch off "
                                        "(default: 5)", 1);
        parser.add_option("verbose", "Report more as the run goes, including what the "
                                     "memory extension is doing");

        /* Extended device memory. Off unless asked for, so a run that does not mention it
           behaves exactly as before, and inert in a build without CUDA. */
        parser.add_option("no-extended-memory", "Do not stream tensors through the device; "
                                                "every allocation then has to fit at once");
        parser.add_option("vram-budget", "Device memory the extension may use, in MiB "
                                         "(default: what the device reports free at "
                                         "startup, less fifteen percent)", 1);
        parser.add_option("vram-store", "Directory holding the store, or \"none\" to keep "
                                        "evicted blocks in host memory only", 1);
        parser.add_option("host-limit", "Pinned host memory the extension may take for "
                                        "evicted blocks, in MiB", 1);

        parser.parse(argc, argv);

        /* Before any tensor exists, which is why this sits at the top of main. */
        /* On unless refused. A budget left unset is taken from what the device reports
           free at startup, less the share the context and the libraries occupy outside it,
           so a run that says nothing about memory still gets the extension sized for the
           card it is on. */
        if (!parser.option("no-extended-memory"))
        {
            extended_memory_options xopts;
            if (parser.option("vram-budget"))
                xopts.vram_budget = (size_t)get_option(parser, "vram-budget", 0) << 20;
            xopts.store_path  = get_option(parser, "vram-store",
                                           default_extended_memory_store_path());
            if (xopts.store_path == "none")
                xopts.store_path.clear();
            xopts.max_pinned_bytes = (size_t)get_option(parser, "host-limit",
                                        (unsigned long)(xopts.vram_budget >> 20)) << 20;
            /* Silent unless asked. What the extension does is not the subject of a run,
               and its startup notes crowd out the figures that are. Failures and the
               warnings that precede a bad run are printed either way, since those are
               not commentary. */
            xopts.verbose = parser.option("verbose");
            if (!enable_extended_memory(xopts))
                cout << "Extended memory was requested but is unavailable in this build; "
                        "continuing without it.\n";
        }

        run_options o;
        o.data_dir      = get_option(parser, "data", "arc-data");
        o.model_file    = get_option(parser, "model-file", "hrm_arc_model.dat");
        o.max_steps     = get_option(parser, "max-steps", 16);
        o.batch_size    = get_option(parser, "batch-size", 8);
        o.max_epochs    = get_option(parser, "max-epochs", 50);
        o.learning_rate = get_option(parser, "learning-rate", 1e-4);
        o.ponder_cost   = get_option(parser, "ponder-cost", 0.01);
        o.blank_id      = parser.option("blank-puzzle-id");
        o.eval_every    = get_option(parser, "eval-every", 5);
        o.all_examples  = parser.option("all-examples");
        o.eval_samples  = get_option(parser, "eval-samples", 2000);
        o.patience      = get_option(parser, "patience", 1000000);
        o.anneal        = parser.option("anneal");
        o.embedding_rate = get_option(parser, "embedding-rate", 100.0);
        o.dense         = parser.option("dense");
        o.report_every  = get_option(parser, "report-every", 200);
        o.steps_per_epoch = get_option(parser, "steps-per-epoch", 0);
        const std::string rw = get_option(parser, "reward", "cells");
        if      (rw == "exact") o.reward = reward_kind::exact;
        else if (rw == "cells") o.reward = reward_kind::cells;
        else
        {
            cout << "Unknown reward: " << rw << "\nExpected one of: exact, cells\n";
            return 1;
        }
        o.verbose       = parser.option("verbose");

        const std::string h = get_option(parser, "halting", "none");
        if      (h == "external") o.halting = halting_mode::external;
        else if (h == "internal") o.halting = halting_mode::internal;
        else if (h == "both")     o.halting = halting_mode::both;
        else if (h == "none")     o.halting = halting_mode::none;
        else
        {
            cout << "Unknown halting mode: " << h
                 << "\nExpected one of: none, internal, external, both\n";
            return 1;
        }

        const bool do_train = parser.option("train");
        const bool do_eval  = parser.option("eval");

        if (parser.option("h") || (!do_train && !do_eval))
        {
            parser.print_options();
            cout << "\nHierarchical reasoning on ARC-AGI, three ways of deciding when to stop\n"
                 << "Example usage:\n"
                 << "  Prepare  : slm_tools/build_arc_dataset.py --download "
                    "--data-dir arc-agi --out-dir arc-data --protocol reference\n"
                 << "  Train    : " << argv[0] << " --train --data arc-data --halting external\n"
                 << "  Evaluate : " << argv[0] << " --eval --data arc-data --halting external\n"
                 << "\n  --halting internal places the adaptive layer inside the modules, "
                    "external composes\n  whole segments around them, both does the two at "
                    "once since they are not\n  alternatives, and none is the baseline no "
                    "other can be credited against until\n  it is known.\n"
                 << "\n  Add --extended-memory when the batch or the table no longer fits "
                    "the card.\n";
            return 0;
        }

        signal_handler::setup();

        const arc_dataset train = load_arc_dataset(o.data_dir + "/training.bin");
        arc_dataset eval_set;
        if (do_eval)
        {
            try { eval_set = load_arc_dataset(o.data_dir + "/evaluation.bin"); }
            catch (const std::exception& e)
            { cout << "No evaluation split: " << e.what() << "\n"; }
        }

        cout << "=== ARC-AGI hierarchical reasoning ===\n"
             << "  data          : " << o.data_dir << "\n"
             << "  halting       : " << describe(o.halting) << "\n"
             << "  window        : " << SEQ_LEN << " (" << WINDOW << " cells and one identifier)\n"
             << "  colours       : " << COLOUR_VOCAB << "\n"
             << "  network       : width " << EMBED_DIM << ", " << NUM_HEADS
             << " query heads over " << NUM_KV_HEADS << " key-value\n"
             << (parser.option("dense")
                 ? "  structure     : a straight stack of "
                   + std::to_string(NUM_H_LAYERS + NUM_L_LAYERS)
                   + " blocks, gradient through all of them\n"
                 : "  H module      : " + std::to_string(NUM_H_LAYERS) + " blocks, run "
                   + std::to_string(HRM_N) + " times per pass\n"
                   "  L module      : " + std::to_string(NUM_L_LAYERS) + " blocks, run "
                   + std::to_string(HRM_T) + " times per H cycle, so "
                   + std::to_string(HRM_N * HRM_T) + " times per pass\n")
             << "  puzzles       : " << train.num_puzzles() << "\n"
             << "  examples      : " << train.num_examples()
             << " in " << train.num_groups() << " groups\n"
             << "  an epoch      : " << (parser.option("all-examples")
                    ? "every example"
                    : "one variant of each task, about "
                      + std::to_string(train.num_groups() ?
                            train.num_examples() / std::max(1L, train.num_puzzles() /
                                                                 std::max(1L, train.num_groups()))
                          : train.num_examples()) + " examples") << "\n"
             << "  identifier    : " << (o.blank_id ? "withheld" : "given") << "\n"
             << "  reward        : " << (o.reward == reward_kind::exact
                                             ? "exact grids, the published measure"
                                             : "fraction of cells right") << "\n\n";

        if (train.colour_vocab != COLOUR_VOCAB)
        {
            cout << "The dataset uses " << train.colour_vocab << " colour tokens, this "
                    "program is built for " << COLOUR_VOCAB << ".\n";
            return 1;
        }

        /* The adaptive layer is present or absent, not merely switched off, so the two
           cases are different network types and each driver has to be instantiated for
           the one it will build. */
        return uses_act(o.halting)
            ? dispatch_table<true >(train, eval_set, o, do_train, do_eval)
            : dispatch_table<false>(train, eval_set, o, do_train, do_eval);
    }
    catch (std::exception& e)
    {
        cout << "Exception: " << e.what() << endl;
        return 1;
    }
}
