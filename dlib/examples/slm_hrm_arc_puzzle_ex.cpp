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
            target built from whether the answer came out right. The segments before the
            decision only advance the state; the training step that follows runs one more
            and is the only one supervised.

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

    DEEP SUPERVISION

        --segments N
            Each batch runs through N segments, every one a training step of its own that
            resumes the recurrence where the previous one left it. Every segment is then
            supervised, so the network learns to improve an answer from any point of the
            recurrence, which is how the published work trains; the external loop above
            supervises only its last segment. Evaluation runs as many segments, and the
            final one is also reported after a single segment, to separate what the
            training brought from what the extra passes bring. Applies to --halting none
            and internal, and only to the hierarchical network.

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
        ./slm_hrm_arc_puzzle_ex --train --data arc-data --halting none --segments 4
        ./slm_hrm_arc_puzzle_ex --eval  --data arc-data --halting external
*/

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include <algorithm>
#include <chrono>
#include <fstream>
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
// Padding, the end marker and the ten colours. The output layer predicts all twelve: the
// marker has to be predicted too, since it is what says how large the answer is.
const long COLOUR_VOCAB = 12;

/*
    The embedding table has to hold every identifier, so its size is a property of the
    dataset rather than of the model, and C++ needs it at compile time. Each size is a
    template parameter and therefore a whole instantiation of the network, which the dense
    control doubles already, so one size serves every dataset and the program refuses a
    dataset that exceeds it rather than truncating. An oversized table occupies device
    memory but no optimizer state and no arithmetic, since the embedding updates only the
    rows a batch touched.
*/
const long TABLE_ROWS = 262144;

/*
    The network follows the reference configuration in its shape: four layers in each of
    the two modules, two cycles of each, eight query heads, and a feed-forward expansion of
    four. Two things depart from it, both deliberately.

    Attention is grouped-query with two key-value heads rather than eight. And the width is
    320 rather than 512.

    The width is where the reasoning happens and it costs the square, so cutting it is what
    makes a four-way comparison affordable. The capacity to hold a puzzle lives in the
    identifier table, outside the parameter count, and keeps its size. Whether reasoning
    survives the cut better than memorisation would is the assumption this configuration
    rests on, and the comparison against the published figure is what tests it.
*/
const long NUM_H_LAYERS = 4;
const long NUM_L_LAYERS = 4;
const long NUM_HEADS    = 8;
const long NUM_KV_HEADS = 2;
const long EMBED_DIM    = 320;
const long HRM_N        = 2;
const long HRM_T        = 2;

/*
    Attention is bidirectional. The model answers a whole grid in one pass, and a cell of
    the answer can depend on any cell of the question, including those after it in reading
    order: a vertical flip needs the top-left cell to see the bottom-left one. Under the
    causal mask the library uses for next-token prediction, every such transformation was
    out of reach, for this network and for the dense control alike.
*/
const bool ATTENTION_CAUSAL = false;

template <long TABLE, bool USE_ACT>
using arc_config = hrm_transformer_config<
    TABLE, NUM_H_LAYERS, NUM_L_LAYERS, NUM_HEADS, EMBED_DIM, HRM_N, HRM_T,
    gelu, dropout_10, attention_impl::unified, NUM_KV_HEADS, USE_ACT, COLOUR_VOCAB,
    ATTENTION_CAUSAL>;

/*
    The control: the same blocks, stacked, with nothing of the hierarchical layer.

    Three things separate the two networks and only a straight stack removes all three.
    The recurrent layer starts its state from fixed vectors, combines its inputs by
    addition rather than by stacking, and carries gradient through its last pass alone,
    which is the one-step approximation the architecture rests on. Shortening the
    recurrence leaves all three in place, so a run that plateaus with it and also without
    it says nothing.

    This stack holds eight blocks, four and four as the two modules do, at the same width
    and over the same embedding table, and the gradient reaches every one of them. If it
    learns where the recurrent network stops, the recurrence is what to look at; if it
    stops in the same place, the cause is elsewhere and an expensive avenue is closed.
*/
template <long TABLE, bool USE_ACT, template <typename> class DO>
using dense_stack = typename impl::hrm_stack_selector<
    attention_impl::unified, NUM_H_LAYERS + NUM_L_LAYERS, EMBED_DIM, NUM_HEADS,
    NUM_KV_HEADS, gelu, DO,
    embeddings<TABLE, EMBED_DIM, input<matrix<int, 0, 1>>>, USE_ACT, ATTENTION_CAUSAL>::type;

/*
    Each network comes in two types, as every network with dropout does in dlib: the one
    trained, and the one measured, where each dropout becomes the scaling it stands for.
    The measured type is obtained by deserializing the trained one into it.
*/
template <long TABLE, bool USE_ACT>
using dense_net = classification_head<COLOUR_VOCAB, dense_stack<TABLE, USE_ACT, dropout_10>>;

template <long TABLE, bool USE_ACT>
using dense_infer_net = classification_head<COLOUR_VOCAB, dense_stack<TABLE, USE_ACT, multiply>>;

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

/*
    Sets what every adaptive layer pays per step, wherever it sits: in the stack of the
    dense control, or inside the two modules of the hierarchical layer, which are networks
    of their own that the outer visitor does not enter. The fallback comes first so the
    visitors below find it.
*/
template <typename layer_type>
void set_ponder_if_adaptive (layer_type&, float, ...) {}

template <typename layer_type>
auto set_ponder_if_adaptive (layer_type& l, float c, int)
    -> decltype(l.layer_details().set_ponder_penalty(c), void())
{
    l.layer_details().set_ponder_penalty(c);
}

template <typename layer_type>
auto set_ponder_if_adaptive (layer_type& l, float c, long)
    -> decltype(l.layer_details().get_h_net(), void())
{
    visit_layers(l.layer_details().get_h_net(), [c](size_t, auto& m) { set_ponder_if_adaptive(m, c, 0); });
    visit_layers(l.layer_details().get_l_net(), [c](size_t, auto& m) { set_ponder_if_adaptive(m, c, 0); });
}

template <typename net_type>
void set_ponder_cost (net_type& net, float c)
{
    visit_layers(net, [c](size_t, auto& l) { set_ponder_if_adaptive(l, c, 0); });
}

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

    // Only the positions the loss scores count, the cells of the answer and its end
    // markers. Rewarding a correct guess of padding would reward emptiness.
    long scored = 0, right = 0;
    for (long i = 0; i < n; ++i)
    {
        if (label(i) == arc_ignore_label) continue;
        ++scored;
        if (predicted[(size_t)i] == label(i)) ++right;
    }
    if (scored == 0) return 0.0f;

    if (kind == reward_kind::exact)
        return right == scored ? 1.0f : 0.0f;
    return (float)right / (float)scored;
}

// ----------------------------------------------------------------------------------------

/*
    Reaches the hierarchical layer inside a network so the external loop can tell it to
    resume rather than restart. The overloads are declared before the visitor that calls
    them: the layer types live in dlib and these functions do not, so argument-dependent
    lookup would not find them from inside it.
*/
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

/*
    The rate the hierarchical layer last updated its two modules at, or a negative value
    for a network without one. It is read from the layer rather than from the trainer:
    the layer updates the modules itself, so a rate that fails to reach them shows here
    and nowhere else.
*/
template <typename layer_type>
auto read_rate_if_hierarchical (layer_type& l, double& rate, int)
    -> decltype(l.layer_details().get_carry_state(), l.layer_details().get_learning_rate(), void())
{
    rate = l.layer_details().get_learning_rate();
}
template <typename layer_type>
void read_rate_if_hierarchical (layer_type&, double&, long) {}

template <typename net_type>
double module_rate (net_type& net)
{
    double rate = -1;
    visit_layers(net, [&rate](size_t, auto& l) { read_rate_if_hierarchical(l, rate, 0); });
    return rate;
}

// ----------------------------------------------------------------------------------------

struct run_options
{
    std::string  data_dir;
    std::string  model_file;
    halting_mode halting     = halting_mode::none;
    long         max_steps   = 16;
    long         segments    = 1;
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

/*
    Runs the held-out split and scores it. segments is how many segments a network trained
    with deep supervision runs before its answer is read; the external loop decides its own
    number and reports how many it used, through segments_used when the caller asks.
*/
template <typename net_type>
double evaluate (net_type& net, const arc_dataset& d, const run_options& o,
                 bool quiet, q_halting_controller* halting, long limit = 0,
                 long segments = 1, double* segments_used = nullptr)
{
    long exact = 0, cells_right = 0, cells_total = 0;
    long near90 = 0, near95 = 0, near99 = 0;
    long miss_marker = 0, miss_colour = 0;
    long segments_run = 0;

    /*
        A periodic measurement reads a sample, the final one reads everything.

        A held-out split of augmented puzzles can hold tens of thousands of examples, and
        an arrangement that composes sixteen segments runs the network sixteen times for
        each of them. Taken every few epochs that can cost more than the training. A few
        thousand examples settle a cell accuracy to well within the differences being
        looked for, and the figure that gets reported is the full one.
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
            if (halting) halting->begin_episode(false);
            for (long s = 0; s < o.max_steps; ++s)
            {
                net.subnet().forward(batch);
                ++segments_run;
                if (halting && halting->decide(
                        summarise_state(net.subnet().get_output(), o.summary_dim), s))
                    break;
            }
            set_carry(net, false);
        }
        else if (segments > 1)
        {
            reset_state(net);
            set_carry(net, true);
            for (long s = 0; s < segments; ++s)
                net.subnet().forward(batch);
            segments_run += segments;
            set_carry(net, false);
        }
        else
        {
            net.subnet().forward(batch);
            ++segments_run;
        }
        const tensor& logits = net.subnet().get_output();
        const auto pred = arc_predict_sequence(logits.host(), logits.nc(),
                                               SEQ_LEN, COLOUR_VOCAB);

        /* Scored over the positions the loss scores: the cells of the expected answer
           and its end markers, never the padding around it. An answer is exact only if
           every one of them is right, its size included, since the markers are among
           them. This is the measure the reference reports. */
        const auto lab = arc_make_label(d, i);
        long scored = 0, right = 0, wrong_marker = 0, wrong_colour = 0;
        for (long j = 0; j < SEQ_LEN && j < (long)pred.size(); ++j)
        {
            if (lab(j) == arc_ignore_label) continue;
            ++scored;
            if (pred[(size_t)j] == lab(j)) ++right;
            else if (lab(j) == arc_eos_token) ++wrong_marker;
            else ++wrong_colour;
        }
        cells_total += scored;
        cells_right += right;
        if (scored > 0 && right == scored) ++exact;

        /* How close each answer came. An exact grid needs every cell right, so at eighty
           five percent per cell the exact count stays at zero long after the model starts
           to approach it; these thresholds show the approach while the exact count cannot. */
        if (scored > 0)
        {
            const double frac = (double)right / (double)scored;
            if (frac >= 0.90) ++near90;
            if (frac >= 0.95) ++near95;
            if (frac >= 0.99)
            {
                ++near99;
                /* For the answers that came within one percent and still missed, which kind
                   of position was wrong. A wrong end marker means the size was misjudged by
                   a row or a column, a matter of structure; a wrong colour means a cell was
                   misjudged, a matter of precision. */
                miss_marker += wrong_marker;
                miss_colour += wrong_colour;
            }
        }
    }

    const double grids = n ? 100.0 * exact / n : 0.0;
    const double cells = cells_total ? 100.0 * cells_right / cells_total : 0.0;
    const double per_example = n ? (double)segments_run / (double)n : 0.0;
    if (segments_used)
        *segments_used = per_example;
    if (!quiet)
    {
        cout << "  grids right at 90, 95 and 99 % of their cells : "
             << near90 << ", " << near95 << ", " << near99 << " of " << n << "\n";
        if (near99 > 0)
            cout << "  within 1 %, positions still wrong : " << miss_marker << " end markers, "
                 << miss_colour << " colours\n";
        cout << "  exact grids   : " << exact << " of " << n
             << "  (" << grids << " %)\n";
        cout << "  cells correct : " << cells << " %\n";
        if (uses_loop(o.halting))
            cout << "  segments used : " << per_example << " of " << o.max_steps << "\n";
    }
    /* Cells rather than grids, because on a model that has yet to produce one exact grid
       the grid count is zero for every arrangement and orders nothing. */
    return cells;
}

// ----------------------------------------------------------------------------------------

/*
    A checkpoint carries the shape it was trained at. Loading one written by a different
    width leaves the weights mismatched and the failure surfaces deep inside an attention
    layer, where nothing names the cause. Saying it here costs one try block.
*/
template <typename network>
bool load_model (network& net, const std::string& file)
{
    try
    {
        deserialize(file) >> net;
        return true;
    }
    catch (const std::exception& e)
    {
        cout << "\n" << file << " does not fit this network. It was most likely written "
                "by a\nrun of a different width or depth; the current one is "
             << NUM_H_LAYERS << " and " << NUM_L_LAYERS << " layers at width "
             << EMBED_DIM << ".\nRemove it, or point --model-file elsewhere.\n\n  "
             << e.what() << "\n";
        return false;
    }
}

/*
    net_type is the network trained, infer_type the one measured.

    Every measurement runs on infer_type. dlib's dropout draws a fresh mask on every
    forward, at evaluation as well, so the trained type answers with a random share of its
    activations removed at every dropout site, a different share each time. The recurrent
    network crosses those sites at every cycle of each module, several times as often per
    pass as the dense control does, and again with every segment, so a measurement taken
    on the trained type penalizes it more the more it recurs. infer_type replaces each
    dropout by the scaling it stands for, and its answers no longer depend on a draw.
*/
template <typename net_type, typename infer_type>
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

    // A run that only evaluates reads the file into infer_type, further down.
    if (do_train && file_exists(o.model_file))
    {
        cout << "Loading " << o.model_file << "\n";
        if (!load_model(net, o.model_file))
            return 1;
    }
    cout << "Parameters: " << count_network_parameters(net, SEQ_LEN) << "\n";

    if (do_train)
    {
        /* The loss scores the answer and nothing else. Positions labelled with the ignored
           value, the padding and the identifier, are left out.

           The padding index is set to the same value explicitly. The loss also skips a
           position whose own input token equals that index, and left unset it would fall
           back to the ignored value anyway; naming it makes the intent survive a change
           to that fallback. No input ever holds this value, so the mask stays inert and a
           cell is never skipped merely because the question was empty there, which is
           the case for every cell an answer has beyond the size of its question. */
        net.loss_details().set_ignore_index((long)arc_ignore_label);
        net.loss_details().set_pad_index((long)arc_ignore_label);

        /* The trainer's optimizer settings, named once because the context below hands them
           to the hierarchical layer as well, which updates its two modules itself. */
        const double weight_decay = 1e-4, beta1 = 0.9, beta2 = 0.999;
        dnn_trainer<net_type, adam> trainer(net, adam(weight_decay, beta1, beta2));
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
           the heuristic reads that as a plateau long before the corpus is learned. The
           patience is therefore the caller's, and large by default. */
        trainer.set_iterations_without_progress_threshold(o.patience);
        trainer.set_min_learning_rate(o.learning_rate * 1e-3);

        /* With a synchronization file the rate has a second way down: the trainer reverts
           to its last saved state when the losses recorded since that save trend upward,
           and after repeated reverts it lowers the rate as well, and a draw that changes
           between epochs produces exactly that reading. A factor of one leaves every such
           reduction without effect, whichever path calls for it.

           The loss of one epoch is measured on a different draw from the loss of the next,
           so a schedule that reads a plateau in it is reading noise. --anneal restores the
           usual behaviour for a run whose draw is fixed. */
        if (!o.anneal)
            trainer.set_learning_rate_shrink_factor(1);

        /* The recurrent modules sit inside the hierarchical layer, and the transition
           networks inside the adaptive layer, and both layers update those inner networks
           with solvers of their own, out of the trainer's reach. Left alone, they train at
           the rate those layers default to, whatever --learning-rate says, and no schedule
           reaches them. The context hands both layers the trainer's rate, and the
           hierarchical layer its optimizer settings too, so that its modules train at the
           rate, and with the settings, the trainer applies to the dense control's blocks. */
        network_context::set_optimizer_params(weight_decay, beta1, beta2);
        double context_rate = trainer.get_learning_rate();
        network_context::set_learning_rate(context_rate);

        /* The identifier embedding needs a rate of its own.

           It receives gradient through one position of nine hundred and one, and only by
           way of attention from the rest, so at the network's rate it barely moves. The
           reference gives the puzzle embedding a rate a hundred times the base one, which
           is not incidental: it is what makes the conditioning learnable at all. Here the
           colours share the table, and the layer already scales each row's update by how
           often it was seen, which keeps the frequent rows from running away. */
        visit_computational_layers(trainer.get_net(force_flush_to_disk::no),
                                   [&](auto& l) { set_embedding_rate(l, o.embedding_rate); });
        set_ponder_cost(trainer.get_net(force_flush_to_disk::no), (float)o.ponder_cost);

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

                /* The trainer changes its rate on its own thread, when it reads a plateau or
                   reloads its synchronization file. The context follows at a barrier, taken
                   only when the rate has moved, since the barrier keeps the trainer from
                   overlapping the next batch with the current step. */
                if (trainer.get_learning_rate() != context_rate)
                {
                    trainer.get_net(force_flush_to_disk::no);
                    context_rate = trainer.get_learning_rate();
                    network_context::set_learning_rate(context_rate);
                }

                if (uses_loop(o.halting))
                {
                    /* The segments before the decision advance the state with plain
                       forwards, and the training step that follows runs one more from
                       there and is the only one supervised. */
                    net_type& live = trainer.get_net(force_flush_to_disk::no);
                    reset_state(live);
                    set_carry(live, true);

                    resizable_tensor batch;
                    live.to_tensor(xs.begin(), xs.end(), batch);

                    /* The reward is read from the pass the decision was taken on, not
                       from the training step that follows. The trainer runs the network on
                       its own thread and cleans its tensors when it pleases, so what it
                       holds after a step is not something a caller may read: asking anyway
                       returns an output of width zero. The question the head is being asked
                       is in any case about the state it decided on, before the extra pass.

                       The decision covers the whole batch, so the reward is the batch's:
                       every example is scored, where the first alone would train the head
                       on part of what it decided about. */
                    halting.begin_episode();
                    std::vector<std::vector<unsigned long>> got;
                    bool decided = false;

                    for (long s = 0; s + 1 < o.max_steps; ++s)
                    {
                        live.subnet().forward(batch);
                        const tensor& out = live.subnet().get_output();
                        got.clear();
                        if (out.nc() >= COLOUR_VOCAB && out.size() > 0 && out.num_samples() > 0)
                        {
                            const size_t per_sample = out.size() / (size_t)out.num_samples();
                            for (long b = 0; b < out.num_samples(); ++b)
                                got.push_back(arc_predict_sequence(out.host() + b * per_sample,
                                                                   out.nc(), SEQ_LEN, COLOUR_VOCAB));
                        }
                        decided = true;
                        if (halting.decide(summarise_state(out, qopt.summary_dim), s))
                            break;
                    }

                    /* The carry flag stays set. The trainer reads it when its forward starts,
                       on its own thread, after this call has returned; cleared here, it would
                       race that read, and the supervised segment would restart the recurrence
                       instead of resuming it. The next batch resets the state, with the
                       trainer idle, before anything runs. */
                    trainer.train_one_step(xs, ys);

                    if (decided && got.size() == ys.size())
                    {
                        float reward = 0;
                        for (size_t b = 0; b < ys.size(); ++b)
                            reward += grid_reward(got[b], ys[b], o.reward);
                        halting.finish(reward / (float)ys.size());
                    }
                }
                else if (o.segments > 1)
                {
                    /* Deep supervision. The batch runs through several segments, each a
                       training step of its own that resumes the recurrence where the step
                       before left it, so every segment is supervised and the network learns
                       to improve an answer from any point of the recurrence. Each step
                       back-propagates through the pass it has just run, which is the only
                       one a backward may go through. The state is reset between batches,
                       with the trainer idle; the flag, once set, stays set. */
                    net_type& live = trainer.get_net(force_flush_to_disk::no);
                    reset_state(live);
                    set_carry(live, true);
                    for (long m = 0; m < o.segments; ++m)
                        trainer.train_one_step(xs, ys);
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
            const double modules = module_rate(trainer.get_net(force_flush_to_disk::no));
            if (modules >= 0)
                cout << ", modules " << modules;
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
                /* The measurement runs on a snapshot. Run on the trainer's own network, it
                   would resize that network's tensors from the training batch to a single
                   example and back, and clean a network still being trained. The snapshot
                   is the trained network serialized, which is also the form in which it is
                   read into the measured type, and the form saved when the measurement is
                   the best so far. The copy it is taken from is released before the
                   measured network is built. */
                std::vector<char> snapshot;
                {
                    net_type copy = trainer.get_net(force_flush_to_disk::no);
                    copy.clean();
                    serialize(snapshot) << copy;
                }
                infer_type measured;
                deserialize(snapshot) >> measured;

                double used = 0;
                const double acc = evaluate(measured, eval_set, o, quiet,
                                            uses_loop(o.halting) ? &halting : nullptr,
                                            o.eval_samples, o.segments, &used);
                cout << "         held out " << acc << " % of cells";
                if (uses_loop(o.halting))
                    cout << " in " << used << " segments";
                if (acc > best_accuracy)
                {
                    best_accuracy = acc;
                    best_epoch    = epoch + 1;
                    std::ofstream best(o.model_file + ".best", std::ios::binary);
                    best.write(snapshot.data(), (std::streamsize)snapshot.size());
                    if (!best)
                        throw std::runtime_error("unable to write " + o.model_file + ".best");
                    cout << ", best so far";
                }
                cout << "\n";
            }

            if (trainer.get_learning_rate() < 1e-7) break;
        }

        trainer.get_net();
        set_carry(net, false);
        network_context::reset();
        net.clean();
        serialize(o.model_file) << net;
        cout << "Model saved to " << o.model_file << "\n";

        // The memory report describes the extension rather than the model: --verbose only.
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

        /* The model is read from the file the run wrote, the best one when the run kept
           one, or from the file named when there was no training. */
        std::string source = o.model_file;
        if (best_accuracy >= 0 && file_exists(o.model_file + ".best"))
        {
            cout << "  using the model of epoch " << best_epoch
                 << ", which scored best while training\n";
            source = o.model_file + ".best";
        }
        if (!file_exists(source))
        {
            cout << "  there is no model to evaluate at " << source << "\n";
            return 1;
        }
        infer_type measured;
        if (!load_model(measured, source))
            return 1;

        /* The head is not saved with the model, so a run that only evaluates has none
           trained beside it and runs the budget. */
        if (o.segments > 1)
            cout << "  after " << o.segments << " segments\n";
        evaluate(measured, eval_set, o, false,
                 uses_loop(o.halting) && do_train ? &halting : nullptr, 0, o.segments);
        if (o.segments > 1)
        {
            /* The same network read after one segment. Against the line above it says how
               much the extra passes add at inference; against a network trained on single
               passes, how much training on several segments changed the first one. */
            cout << "  after a single segment\n";
            evaluate(measured, eval_set, o, false, nullptr, 0, 1);
        }
    }
    return 0;
}

// ----------------------------------------------------------------------------------------

/*
    Checks that the embedding table holds the dataset's identifiers, and refuses rather
    than truncating when it does not: a puzzle whose identifier does not fit would
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
        ? run<dense_net<TABLE_ROWS, USE_ACT>, dense_infer_net<TABLE_ROWS, USE_ACT>>(
              train, eval_set, o, do_train, do_eval)
        : run<typename arc_config<TABLE_ROWS, USE_ACT>::template network_type<true>,
              typename arc_config<TABLE_ROWS, USE_ACT>::template network_type<false>>(
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
        parser.add_option("segments", "Run each batch through N segments, each a training "
                                      "step resuming the recurrence where the previous one "
                                      "left it, so that every segment is supervised; "
                                      "evaluation runs as many. With --halting none or "
                                      "internal, on the hierarchical network (default: 1)", 1);
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

        /* Extended device memory. On unless refused, and inert in a build without CUDA. */
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

        /* Before any tensor exists, which is why this sits at the top of main. A budget left
           unset is taken from what the device reports free at startup, less the share the
           context and the libraries occupy outside it, so a run that says nothing about
           memory still gets the extension sized for the card it is on. */
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
        o.segments      = get_option(parser, "segments", 1);
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

        /* Deep supervision drives the recurrence itself, segment by segment. The external
           loop drives it too and supervises only its last segment, so the two cannot be
           combined, and the dense control carries no state between passes. */
        if (o.segments < 1)
        {
            cout << "--segments must be at least 1\n";
            return 1;
        }
        if (o.segments > 1 && uses_loop(o.halting))
        {
            cout << "--segments applies to --halting none and internal; the external loop "
                    "decides its own segments.\n";
            return 1;
        }
        if (o.segments > 1 && parser.option("dense"))
        {
            cout << "--segments needs the hierarchical network; the dense control carries "
                    "no state from one pass to the next.\n";
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
                 << "\n  The memory extension lets the batch and the table exceed the card; "
                    "--no-extended-memory\n  turns it off.\n";
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
             << (o.segments > 1
                 ? "  segments      : " + std::to_string(o.segments)
                   + " per batch, each a supervised training step\n"
                 : std::string())
             << "  window        : " << SEQ_LEN << " (" << WINDOW << " cells and one identifier)\n"
             << "  colours       : " << COLOUR_VOCAB << "\n"
             << "  network       : width " << EMBED_DIM << ", " << NUM_HEADS
             << " query heads over " << NUM_KV_HEADS << " key-value, attention "
             << (ATTENTION_CAUSAL ? "causal" : "bidirectional") << "\n"
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
