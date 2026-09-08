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
const long TABLE_SMALL  = 1024;      // no augmentation, 400 puzzles
const long TABLE_MEDIUM = 32768;     // up to about 80 augmentations
const long TABLE_LARGE  = 131072;    // up to about 320 augmentations

const long NUM_H_LAYERS = 1;
const long NUM_L_LAYERS = 2;
const long NUM_HEADS    = 6;
const long NUM_KV_HEADS = 2;
const long EMBED_DIM    = 192;
const long HRM_N        = 1;
const long HRM_T        = 2;

template <long TABLE, bool USE_ACT>
using arc_config = hrm_transformer_config<
    TABLE, NUM_H_LAYERS, NUM_L_LAYERS, NUM_HEADS, EMBED_DIM, HRM_N, HRM_T,
    gelu, dropout_10, attention_impl::unified, NUM_KV_HEADS, USE_ACT, COLOUR_VOCAB>;

// ----------------------------------------------------------------------------------------

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
    long         max_steps   = 8;
    long         batch_size  = 8;
    long         max_epochs  = 50;
    double       learning_rate = 1e-4;
    double       ponder_cost = 0.01;
    bool         blank_id    = false;
    long         summary_dim = 64;
    reward_kind  reward      = reward_kind::cells;
    bool         verbose     = false;
};

// ----------------------------------------------------------------------------------------

template <typename net_type>
double evaluate (net_type& net, const arc_dataset& d, const run_options& o,
                 q_halting_controller* halting)
{
    long exact = 0, cells_right = 0, cells_total = 0;

    for (long i = 0; i < d.num_examples() && !signal_handler::is_triggered(); ++i)
    {
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

    const double acc = d.num_examples() ? 100.0 * exact / d.num_examples() : 0.0;
    cout << "  exact grids   : " << exact << " of " << d.num_examples()
         << "  (" << acc << " %)\n";
    cout << "  cells correct : "
         << (cells_total ? 100.0 * cells_right / cells_total : 0.0) << " %\n";
    return acc;
}

// ----------------------------------------------------------------------------------------

template <typename net_type>
int run (const arc_dataset& train, const arc_dataset& eval_set, const run_options& o,
         bool do_train, bool do_eval)
{
    net_type net;
    q_halting_options qopt;
    qopt.max_steps   = o.max_steps;
    qopt.min_steps   = 1;
    qopt.summary_dim = o.summary_dim;
    q_halting_controller halting(qopt);

    if (file_exists(o.model_file))
    {
        cout << "Loading " << o.model_file << "\n";
        deserialize(o.model_file) >> net;
    }
    cout << "Parameters: " << count_network_parameters(net, SEQ_LEN) << "\n";

    if (do_train)
    {
        dnn_trainer<net_type, adam> trainer(net, adam(1e-4, 0.9, 0.999));
        trainer.set_learning_rate(o.learning_rate);
        trainer.set_mini_batch_size((size_t)o.batch_size);
        trainer.set_synchronization_file("chkpt-" + o.model_file, std::chrono::minutes(15));
        trainer.be_quiet();

        std::vector<long> order((size_t)train.num_examples());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(1);

        cout << "\nTraining, halting " << describe(o.halting) << "\n";
        for (long epoch = 0; epoch < o.max_epochs && !signal_handler::is_triggered(); ++epoch)
        {
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

            if (trainer.get_learning_rate() < 1e-7) break;
        }

        trainer.get_net();
        net.clean();
        serialize(o.model_file) << net;
        cout << "Model saved to " << o.model_file << "\n";

        if (extended_memory_enabled())
        {
            cout << "\n";
            print_extended_memory_stats(cout);
        }
    }

    if (do_eval && eval_set.num_examples() > 0)
    {
        cout << "\nEvaluating on " << eval_set.num_examples() << " held-out examples, "
             << (o.blank_id ? "without the identifier" : "with the identifier") << "\n";
        evaluate(net, eval_set, o, uses_loop(o.halting) ? &halting : nullptr);
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
    if (need <= TABLE_SMALL)
        return run<typename arc_config<TABLE_SMALL, USE_ACT>::template network_type<true>>(
            train, eval_set, o, do_train, do_eval);
    if (need <= TABLE_MEDIUM)
        return run<typename arc_config<TABLE_MEDIUM, USE_ACT>::template network_type<true>>(
            train, eval_set, o, do_train, do_eval);
    if (need <= TABLE_LARGE)
        return run<typename arc_config<TABLE_LARGE, USE_ACT>::template network_type<true>>(
            train, eval_set, o, do_train, do_eval);

    cout << "The dataset needs an embedding table of " << need << " rows, above the "
         << TABLE_LARGE << " this program is built for.\nRebuild the dataset with fewer "
            "augmentations, or add a larger size to the table list.\n";
    return 1;
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
        parser.add_option("max-steps", "Segments the external loop may run (default: 8)", 1);
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
        parser.add_option("verbose", "Report more as the run goes");

        /* Extended device memory. Off unless asked for, so a run that does not mention it
           behaves exactly as before, and inert in a build without CUDA. */
        parser.add_option("extended-memory", "Stream tensors through the device under a "
                                             "budget when the working set does not fit");
        parser.add_option("vram-budget", "Device memory the extension may use, in MiB "
                                         "(default: 8192)", 1);
        parser.add_option("vram-store", "Directory holding the store, or \"none\" to keep "
                                        "evicted blocks in host memory only", 1);
        parser.add_option("host-limit", "Pinned host memory the extension may take for "
                                        "evicted blocks, in MiB", 1);

        parser.parse(argc, argv);

        /* Before any tensor exists, which is why this sits at the top of main. */
        if (parser.option("extended-memory"))
        {
            extended_memory_options xopts;
            xopts.vram_budget = (size_t)get_option(parser, "vram-budget", 8192) << 20;
            xopts.store_path  = get_option(parser, "vram-store",
                                           default_extended_memory_store_path());
            if (xopts.store_path == "none")
                xopts.store_path.clear();
            xopts.max_pinned_bytes = (size_t)get_option(parser, "host-limit",
                                        (unsigned long)(xopts.vram_budget >> 20)) << 20;
            xopts.verbose = true;
            if (!enable_extended_memory(xopts))
                cout << "Extended memory was requested but is unavailable in this build; "
                        "continuing without it.\n";
        }

        run_options o;
        o.data_dir      = get_option(parser, "data", "arc-data");
        o.model_file    = get_option(parser, "model-file", "hrm_arc_model.dat");
        o.max_steps     = get_option(parser, "max-steps", 8);
        o.batch_size    = get_option(parser, "batch-size", 8);
        o.max_epochs    = get_option(parser, "max-epochs", 50);
        o.learning_rate = get_option(parser, "learning-rate", 1e-4);
        o.ponder_cost   = get_option(parser, "ponder-cost", 0.01);
        o.blank_id      = parser.option("blank-puzzle-id");
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
             << "  puzzles       : " << train.num_puzzles() << "\n"
             << "  examples      : " << train.num_examples()
             << " in " << train.num_groups() << " groups\n"
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
