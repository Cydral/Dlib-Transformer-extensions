// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Hierarchical reasoning on ARC-AGI, trained the way the reference trains it.

    The reference work reports its ARC result from a model conditioned on a puzzle
    identifier and trained on the demonstration pairs of every task, evaluation tasks
    included, each task augmented a thousand times. This program trains that model in
    dlib with the procedure the Sudoku example reproduces: the network of the reference
    implementation, block for block, the same supervision of every pass, the same halting
    head trained jointly with the network, the same loss and the same optimizer, and the
    sparse sign descent the reference applies to the identifier embedding.

    The network is the one of the Sudoku example. Two modules of four post-norm blocks
    normalized without a gain, a bias-free SwiGLU of width 1536, bidirectional attention
    with rotary positions, initial states drawn once from a truncated normal law,
    projections drawn as the reference draws them, and an output layer reading the H
    state as the last block leaves it. This is the reference structure of
    hrm_transformer_config.

    HOW A BATCH IS TRAINED

    A batch is a set of slots, each holding one example and the recurrent state the
    network has reached on it. A training step runs one pass of the recurrence on every
    slot, from the state the previous step left, and supervises every slot's answer
    whatever pass it is at. After the step, a slot that is done takes a new example and
    starts it from the initial state, while the others continue.

    Examples are drawn as the reference draws them: the tasks in a random order, one
    augmented variant of each at random, and every example of that variant.

    WHEN A SLOT IS DONE

        --halting external
            The reference arrangement, and the default. A halting logit, read at the
            position of the identifier, is trained by the loss towards whether the answer
            of the pass was exact, and a slot halts when the logit is positive or when it
            has run --max-passes passes. With probability 0.1 a slot is first held to a
            random minimum of passes, as in the reference, so that the head also sees the
            passes it would have skipped. The reference also trains a second, "continue",
            logit towards a target bootstrapped from one more pass of the network; this
            program trains the halting logit alone, as the Sudoku example does.

        --halting none
            Every example runs --max-passes passes, each one supervised, and no head is
            trained. --max-passes 1 gives a single pass.

        --halting internal
            As none, with the adaptive computation layer of Graves placed around the
            feed-forward of every block of the two modules, where it decides per position
            how many steps to take. It is trained by a ponder cost, --ponder-cost, paid for
            every step.

        --halting both
            The adaptive layer inside the modules and the halting head outside them. The
            one decides per position inside a module and the other per example over whole
            passes, so the pair is an arrangement of its own.

    THE CONDITIONING

    A model here is shown one input grid and no worked example of the same task. What
    tells it which transformation to apply is the identifier carried at the first
    position, which the puzzle_embeddings layer maps to its own row of a table trained by
    sign descent on the rows each batch names. An identifier the model never trained on
    keeps the zero row it started with, so a puzzle can only be asked about by name if it
    was named during training. Build the dataset with --protocol reference to allow that.
    A dataset built with --protocol held-out asks about its evaluation puzzles under the
    blank identifier, and --blank-puzzle-id withholds the identifier from any dataset; both
    ask the model to work without being told, which is the harder setting.

    HOW IT IS MEASURED

    Evaluation runs every held-out example through --max-passes passes from the initial
    state and reads the last answer, as in the reference. An answer counts only if every
    cell of the expected grid and every end marker around it is right, which also checks
    its size.

    The variants of a task then vote, as in the reference. An answer is read from the
    top-left corner of the window, cropped to the largest rectangle of colours anchored
    there, and brought back to its task by undoing the symmetry and the colours its
    puzzle's name records. Each answer is a vote for a grid on its test input. The grids a
    test input receives are ranked by their votes, then by the mean probability the halting
    head gave them, and a task counts as solved with k attempts when every one of its test
    inputs has its expected answer among the first k grids. The program reports one attempt
    and two, two being the published measure.

    A measurement limited by --eval-samples reads whole variants spread evenly across the
    split, each answering all the test inputs of its task, so a task votes with the
    variants read. The published measure reads them all, which --eval-samples 0 does.

    WHAT DIFFERS FROM THE REFERENCE

    The halting head trains the halting logit alone, as described above. One linear layer
    carries the cell logits and the halting logit, so the cell logits have a bias where
    the reference's output layer has none. The computation is in single precision, where
    the reference runs its forward pass in bfloat16.

    Prepare the data first, with the tasks of ConceptARC beside those of ARC-AGI-1, as the
    reference trains:
        git clone https://github.com/victorvikram/ConceptARC
        slm_tools/build_arc_dataset.py --download --data-dir arc-agi --out-dir arc-data \
            --protocol reference --extra-corpus ConceptARC/corpus

    Then:
        ./slm_hrm_arc_puzzle_ex --train --data arc-data
        ./slm_hrm_arc_puzzle_ex --eval  --data arc-data --eval-samples 0

    The reference trains with a batch of 768, the batch every default of this program is
    set for. A run with another --batch-size gets the settings the square-root rule of
    Malladi et al. (2022) derives for adaptive optimizers, as in the Sudoku example. With
    kappa the ratio of the two batches, the rates are multiplied by sqrt(kappa), each
    moment decay becomes 1 - kappa * (1 - beta), the decoupled decays are multiplied by
    sqrt(kappa), and the warmup is divided by kappa. The number of steps follows from
    --epochs, so the amount of data seen stays the same. A setting given on the command
    line is taken as it is, and --no-batch-scaling turns the rule off.

    A slot here holds eleven times the positions of a Sudoku slot, and the attention of
    every block grows with the square of that number. The memory extension, on unless
    --no-extended-memory is given, carries what does not fit the card at a cost in speed.
*/

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/misc_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
    The window is the largest ARC grid, 30 by 30, preceded by the position that carries
    the identifier. The cells use twelve tokens, the padding, the end marker and the ten
    colours. The output predicts all twelve at every position, the end markers included,
    since they say how large the answer is, and adds one channel for the halting logit.
*/
const long GRID_SIDE    = 30;
const long WINDOW       = GRID_SIDE * GRID_SIDE;
const long SEQ_LEN      = WINDOW + 1;
const long COLOUR_VOCAB = 12;
const long OUTPUTS      = COLOUR_VOCAB + 1;    // and the halting logit

/*
    The reference configuration has two modules of four blocks each, eight heads, a width
    of 512, and two high-level cycles of two low-level steps each per pass.
*/
const long NUM_H_LAYERS = 4;
const long NUM_L_LAYERS = 4;
const long NUM_HEADS    = 8;
const long NUM_KV_HEADS = 8;
const long EMBED_DIM    = 512;
const long HRM_N        = 2;
const long HRM_T        = 2;

/*
    Attention is bidirectional. The model answers a whole grid in one pass, and a cell of
    the answer can depend on any cell of the question, those after it in reading order
    included: a vertical flip needs the top-left cell to see the bottom-left one.
*/
const bool ATTENTION_CAUSAL = false;

template <bool USE_ACT>
using arc_config = hrm_transformer_config<
    COLOUR_VOCAB, NUM_H_LAYERS, NUM_L_LAYERS, NUM_HEADS, EMBED_DIM, HRM_N, HRM_T,
    gelu, dropout_10, attention_impl::unified, NUM_KV_HEADS, USE_ACT, COLOUR_VOCAB,
    ATTENTION_CAUSAL, hrm_structure::reference>;

/*
    The input holds one index per position, the identifier and then the window. The
    puzzle_embeddings layer reads the colours from a table the solver trains with the rest
    of the network, and the identifier from a table of one row per puzzle, which it trains
    itself at --puzzle-rate. Both are scaled by the square root of the width, as in the
    reference.

    The unified blocks carry no dropout, as the reference has none, so one network type
    serves training and measurement. USE_ACT places the adaptive computation layer around
    the feed-forward of every block of the two modules, which makes it a different type.
    The output layer reads the H state with no normalization before it, since the last
    block of the module has just normalized it.
*/
template <bool USE_ACT>
using arc_net = loss_halting_cross_entropy_per_token<linear<OUTPUTS,
    typename arc_config<USE_ACT>::template train_hrm<
    puzzle_embeddings<COLOUR_VOCAB, EMBED_DIM, input<matrix<int, 0, 1>>>>>>;

// The layers the program reaches into, counted from the loss.
template <typename net> auto& halting_head (net& n) { return layer<1>(n).layer_details(); }
template <typename net> auto& recurrence (net& n) { return layer<2>(n).layer_details(); }
template <typename net> auto& identifier_table (net& n) { return layer<3>(n).layer_details(); }

// ----------------------------------------------------------------------------------------

enum class halting_mode { external, none, internal, both };

inline bool uses_head (halting_mode m) { return m == halting_mode::external || m == halting_mode::both; }
inline bool uses_act  (halting_mode m) { return m == halting_mode::internal || m == halting_mode::both; }

inline std::string describe (halting_mode m)
{
    switch (m)
    {
        case halting_mode::external: return "external, the halting head trained by the loss";
        case halting_mode::internal: return "internal, the adaptive layer inside the modules";
        case halting_mode::both:     return "both, the adaptive layer and the halting head";
        default:                     return "none, every example runs all its passes";
    }
}

/*
    The defaults are the reference's settings for its batch of 768: the rates, the
    decays, the moments and the warmup.
*/
const long REFERENCE_BATCH = 768;

struct run_options
{
    std::string   data_dir            = "arc-data";
    std::string   model_file          = "hrm_arc_model.dat";
    halting_mode  halting             = halting_mode::external;
    long          batch_size          = REFERENCE_BATCH;
    long          epochs              = 100000;
    long          eval_interval       = 10000;
    long          eval_samples        = 2000;
    double        learning_rate       = 1e-4;
    long          warmup_steps        = 2000;
    double        lr_min_ratio        = 1.0;
    double        weight_decay        = 0.1;
    double        beta1               = 0.9;
    double        beta2               = 0.95;
    double        puzzle_rate         = 1e-2;
    double        puzzle_weight_decay = 0.1;
    bool          transposed          = false;
    long          max_passes          = 16;
    double        exploration         = 0.1;
    double        halting_weight      = 0.5;
    double        ponder_cost         = 0.01;
    bool          softmax             = false;
    bool          blank_id            = false;
    long          report_every        = 100;
    unsigned long seed                = 0;
    bool          verbose             = false;
};

/*
    Carries the reference's settings over to the batch of this run by the square-root
    rule, leaving alone every setting the command line gives.
*/
void transpose_to_batch (run_options& o, const command_line_parser& parser)
{
    if (o.batch_size == REFERENCE_BATCH || parser.option("no-batch-scaling"))
        return;
    const double kappa = (double)o.batch_size / (double)REFERENCE_BATCH;
    const double root = std::sqrt(kappa);
    auto moment = [kappa](double beta) { return std::max(0.0, 1 - kappa * (1 - beta)); };
    if (!parser.option("learning-rate"))       o.learning_rate *= root;
    if (!parser.option("puzzle-rate"))         o.puzzle_rate *= root;
    if (!parser.option("weight-decay"))        o.weight_decay *= root;
    if (!parser.option("puzzle-weight-decay")) o.puzzle_weight_decay *= root;
    if (!parser.option("beta1"))               o.beta1 = moment(o.beta1);
    if (!parser.option("beta2"))               o.beta2 = moment(o.beta2);
    if (!parser.option("warmup-steps"))        o.warmup_steps = std::lround(o.warmup_steps / kappa);
    o.transposed = true;
}

// ----------------------------------------------------------------------------------------

/*
    The adaptive layers sit inside the two modules of the recurrence, which are networks
    of their own that a visitor of the outer network does not enter, so these visit the
    modules. The fallbacks come first so that the visitors find them.
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
void add_steps_if_adaptive (layer_type&, double&, long&, ...) {}

template <typename layer_type>
auto add_steps_if_adaptive (layer_type& l, double& sum, long& count, int)
    -> decltype(l.layer_details().get_average_steps(), void())
{
    sum += l.layer_details().get_average_steps();
    ++count;
}

template <typename network>
void set_ponder_cost (network& net, float c)
{
    auto& core = recurrence(net);
    visit_layers(core.get_h_net(), [c](size_t, auto& l) { set_ponder_if_adaptive(l, c, 0); });
    visit_layers(core.get_l_net(), [c](size_t, auto& l) { set_ponder_if_adaptive(l, c, 0); });
}

/* The steps the adaptive layers took on their last forward, averaged over all of them. */
template <typename network>
double adaptive_steps (network& net)
{
    double sum = 0;
    long count = 0;
    auto& core = recurrence(net);
    visit_layers(core.get_h_net(), [&](size_t, auto& l) { add_steps_if_adaptive(l, sum, count, 0); });
    visit_layers(core.get_l_net(), [&](size_t, auto& l) { add_steps_if_adaptive(l, sum, count, 0); });
    return count ? sum / count : 0.0;
}

/*
    The halting head starts silent, with a zero weight and a bias of minus five as in the
    reference, so that no slot halts on its own before the head has learned anything and
    every early example runs its full number of passes.
*/
template <typename network>
void initialize_halting_head (network& net)
{
    auto& head = halting_head(net);
    auto w = head.get_weights();
    float* pw = w.host();
    const long inputs = static_cast<long>(head.get_num_inputs());
    for (long i = 0; i < inputs; ++i)
        pw[i * OUTPUTS + (OUTPUTS - 1)] = 0;
    auto b = head.get_biases();
    b.host()[OUTPUTS - 1] = -5;
}

/* The network's input for one example, with its identifier or the blank one. */
inline matrix<int, 0, 1> make_input (const arc_dataset& d, long example, const run_options& o)
{
    return arc_make_input(d, example, o.blank_id ? arc_blank_puzzle_id : d.identifier_of(example));
}

// ----------------------------------------------------------------------------------------

/*
    Draws training examples as the reference does. The groups come in a random order, a
    group being one task and its augmentations, and each gives one of its puzzles at
    random, every example of which is drawn before the next group, in a random order. A
    task is then seen once per pass over the groups, however many variants it has.
*/
class example_stream
{
public:
    example_stream (const arc_dataset& d, unsigned long seed) : data(d), rng(seed) {}

    long next ()
    {
        while (pending.empty())
        {
            if (position >= order.size())
            {
                order.resize((size_t)data.num_groups());
                std::iota(order.begin(), order.end(), 0L);
                std::shuffle(order.begin(), order.end(), rng);
                position = 0;
            }
            const long g = order[position++];
            const long first = data.group_indices[(size_t)g];
            const long last  = data.group_indices[(size_t)g + 1];
            if (last <= first)
                continue;
            std::uniform_int_distribution<long> pick(first, last - 1);
            const long puzzle = pick(rng);
            for (long e = data.puzzle_indices[(size_t)puzzle]; e < data.puzzle_indices[(size_t)puzzle + 1]; ++e)
                pending.push_back(e);
            std::shuffle(pending.begin(), pending.end(), rng);
        }
        const long e = pending.back();
        pending.pop_back();
        return e;
    }

private:
    const arc_dataset& data;
    std::mt19937_64 rng;
    std::vector<long> order;
    size_t position = 0;
    std::vector<long> pending;
};

/* The examples a puzzle holds on average, which the reference's step count multiplies by. */
inline double mean_puzzle_examples (const arc_dataset& d)
{
    const long puzzles = std::max(1L, (long)d.puzzle_indices.size() - 1);
    return (double)d.num_examples() / (double)puzzles;
}

// ----------------------------------------------------------------------------------------

struct measurement
{
    long   examples = 0;
    long   exact    = 0;
    double cells    = 0;
    // Answers within 90, 95 and 99 % of their cells, and what the ones within 1 % missed.
    long   near90 = 0, near95 = 0, near99 = 0;
    long   wrong_markers = 0, wrong_colours = 0;
    // The vote: the variants read, their tasks and test inputs, and the tasks solved with
    // one attempt and with two.
    long   variants = 0, tasks = 0, questions = 0, solved1 = 0, solved2 = 0;
};

// The longest a run stays silent, during training as during a measurement.
const auto REPORT_PERIOD = std::chrono::minutes(5);

/*
    The examples a measurement reads. A limit takes whole variants spread evenly across the
    split, as many as hold about that many examples. Every variant read then answers all the
    test inputs of its task, and the same variants are read every time, so successive
    measurements during a run compare like with like.
*/
std::vector<long> examples_to_read (const arc_dataset& d, long limit)
{
    const long puzzles = d.num_puzzles() - 1;              // the blank one aside
    long wanted = puzzles;
    if (limit > 0 && limit < d.num_examples() && puzzles > 0)
    {
        const double per_puzzle = (double)d.num_examples() / (double)puzzles;
        wanted = std::min(puzzles, std::max(1L, (long)std::ceil(limit / per_puzzle)));
    }
    std::vector<long> chosen;
    for (long k = 0; k < wanted; ++k)
    {
        const long p = 1 + (long)((double)k * (double)puzzles / (double)wanted);
        for (long e = d.first_example(p); e < d.end_example(p); ++e)
            chosen.push_back(e);
    }
    return chosen;
}

/*
    Runs held-out examples through the full number of passes from the initial state and
    scores the last answer, then lets the variants of each task vote.

    Only the positions the loss scores count, the cells of the expected answer and its end
    markers, never the padding around it. An exact grid needs every cell right, so the
    exact count stays at zero long after the cells start to come right; the thresholds
    show that approach. For the answers within 1 % that still missed, a wrong end marker
    means the size was misjudged by a row or a column, a wrong colour that a cell was.

    The network's carried state is replaced by the measurement's, so the caller restarts
    its slots afterwards.
*/
template <typename network>
measurement evaluate (network& net, const arc_dataset& test, const run_options& o, long limit)
{
    measurement m;
    const std::vector<long> chosen = examples_to_read(test, limit);
    const long n = (long)chosen.size();

    auto& core = recurrence(net);
    std::vector<matrix<int, 0, 1>> xs;
    std::vector<long> ids;
    resizable_tensor x;
    arc_vote vote;
    long cells_right = 0, cells_total = 0;
    const auto started = std::chrono::steady_clock::now();
    auto last_line = started;

    for (long k = 0; k < n && !signal_handler::is_triggered(); )
    {
        xs.clear();
        ids.clear();
        for (; k < n && (long)xs.size() < o.batch_size; ++k)
        {
            ids.push_back(chosen[(size_t)k]);
            xs.push_back(make_input(test, chosen[(size_t)k], o));
        }
        net.to_tensor(xs.begin(), xs.end(), x);

        core.reset_state();
        core.set_carry_state(true);
        for (long pass = 0; pass < o.max_passes; ++pass)
            net.forward(x);

        const tensor& out = net.subnet().get_output();
        const float* logits = out.host();
        for (size_t b = 0; b < ids.size(); ++b)
        {
            const long e = ids[b];
            const float* rows = logits + (long)b * SEQ_LEN * OUTPUTS;
            const std::vector<unsigned long> predicted =
                arc_predict_sequence(rows, OUTPUTS, SEQ_LEN, COLOUR_VOCAB);
            const auto label = arc_make_label(test, e);
            long right = 0, scored = 0, markers = 0, colours = 0;
            for (long t = 0; t < SEQ_LEN; ++t)
            {
                if (label(t) == arc_ignore_label)
                    continue;
                ++scored;
                if (predicted[(size_t)t] == label(t)) ++right;
                else if (label(t) == arc_eos_token) ++markers;
                else ++colours;
            }

            /* The answer and the expected grid are read from the top-left corner, where
               the held-out split keeps its grids, and brought back to the task by undoing
               the variant the puzzle's name records. The confidence of the vote is the
               probability the halting head gives the answer after the last pass. */
            const long puzzle = test.puzzle_of(e);
            const std::string& name = test.puzzle_names[(size_t)puzzle];
            const arc_variant variant = arc_variant_of(name);
            const long question = e - test.first_example(puzzle);
            const double halting = rows[OUTPUTS - 1];
            vote.add(arc_task_of(name), question,
                     arc_undo_variant(arc_label_grid(test, e), variant),
                     arc_undo_variant(arc_decode_grid(predicted, WINDOW, GRID_SIDE), variant),
                     1 / (1 + std::exp(-halting)));
            if (question == 0)
                ++m.variants;
            cells_right += right;
            cells_total += scored;
            ++m.examples;
            if (scored == 0)
                continue;
            if (right == scored) ++m.exact;
            const double frac = (double)right / (double)scored;
            if (frac >= 0.90) ++m.near90;
            if (frac >= 0.95) ++m.near95;
            if (frac >= 0.99)
            {
                ++m.near99;
                m.wrong_markers += markers;
                m.wrong_colours += colours;
            }
        }

        // A measurement over the whole split runs for hours, so it says where it stands.
        const auto now = std::chrono::steady_clock::now();
        if (k < n && now - last_line >= REPORT_PERIOD)
        {
            const double elapsed = std::chrono::duration<double>(now - started).count();
            cout << "  measuring: " << m.examples << " of " << n << " examples, "
                 << 100.0 * m.exact / m.examples << " % exact so far, "
                 << (long)(elapsed * (n - m.examples) / m.examples / 60) << " min left\n" << std::flush;
            last_line = now;
        }
    }
    core.reset_state();
    m.cells = cells_total ? 100.0 * cells_right / cells_total : 0.0;
    m.tasks = vote.num_tasks();
    m.questions = vote.num_questions();
    m.solved1 = vote.solved(1);
    m.solved2 = vote.solved(2);
    return m;
}

void report (const measurement& m, const char* what)
{
    cout << "  " << what << ": " << m.exact << " of " << m.examples << " grids exact ("
         << (m.examples ? 100.0 * m.exact / m.examples : 0.0) << " %), "
         << m.cells << " % of cells\n"
         << "    within 90, 95 and 99 % of their cells: " << m.near90 << ", " << m.near95
         << " and " << m.near99 << "\n";
    if (m.near99 > m.exact)
        cout << "    the answers within 1 % that missed hold " << m.wrong_markers
             << " wrong end markers and " << m.wrong_colours << " wrong colours\n";
    if (m.tasks > 0)
        cout << "    vote of " << m.variants << " variants over " << m.tasks << " tasks and "
             << m.questions << " test inputs: " << m.solved1 << " tasks solved with one attempt ("
             << 100.0 * m.solved1 / m.tasks << " %), " << m.solved2 << " with two ("
             << 100.0 * m.solved2 / m.tasks << " %)\n";
    cout << std::flush;
}

// ----------------------------------------------------------------------------------------

/*
    The rate at a given step. A linear warmup is followed by a cosine down to lr_min_ratio
    of the base rate, and the reference's ARC setting keeps the ratio at one, which leaves
    the rate constant once warm. The identifier table follows the same schedule, at its
    own base rate.
*/
double rate_at (const run_options& o, long step, long total)
{
    if (step < o.warmup_steps)
        return o.learning_rate * (double)step / (double)std::max(1L, o.warmup_steps);
    const double progress = (double)(step - o.warmup_steps) /
                            (double)std::max(1L, total - o.warmup_steps);
    const double pi = 3.14159265358979323846;
    return o.learning_rate * (o.lr_min_ratio + std::max(0.0,
        (1 - o.lr_min_ratio) * 0.5 * (1 + std::cos(pi * progress))));
}

/*
    A checkpoint holds the network, the identifier table with it, the solvers, the step,
    the best held-out rate measured so far and the batch. The rate lets a resumed run keep
    the best model it already saved, since its first measurement then replaces that model
    only by doing better. The slots are not kept, so a resumed run starts every slot on a
    new example, which costs the examples in flight and nothing else.

    The solvers keep the moment decays and the weight decay they were built with, and a
    step means a batch of examples, so a run resumes at the batch it was written with.

    The network is cleaned first, since a serialized network carries its activations
    otherwise. The next step allocates them again, and the recurrent state the slots carry
    is kept, as cleaning leaves it alone.
*/
template <typename network>
void save_checkpoint (const std::string& file, network& net,
                      const std::vector<adam_atan2>& solvers, long step, double best, long batch)
{
    net.clean();
    const std::string tmp = file + "_";
    {
        std::ofstream out(tmp, std::ios::binary);
        serialize(net, out);
        serialize(solvers, out);
        serialize(step, out);
        serialize(best, out);
        serialize(batch, out);
        if (!out)
            throw std::runtime_error("unable to write " + tmp);
    }
    std::rename(tmp.c_str(), file.c_str());
}

template <typename network>
bool load_checkpoint (const std::string& file, network& net,
                      std::vector<adam_atan2>& solvers, long& step, double& best, long& batch)
{
    if (!file_exists(file))
        return false;
    std::ifstream in(file, std::ios::binary);
    deserialize(net, in);
    deserialize(solvers, in);
    deserialize(step, in);
    deserialize(best, in);
    deserialize(batch, in);
    return true;
}

/*
    The model files hold what a measurement reads. The recurrent layer, and the adaptive
    layers inside it, keep the state of the solvers that train their networks, three times
    the size of those weights, which a copy of the network leaves out and the checkpoint
    keeps.
*/
template <typename network>
void save_model (const std::string& file, network& net)
{
    net.clean();
    const network copy(net);
    serialize(file) << copy;
}

/* A file written for another network type, or by a run of another arrangement, does not
   read into this network. */
void explain_mismatch (const std::string& file, const run_options& o, const std::exception& e)
{
    cout << "\n" << file << " does not fit this network. The adaptive layer of --halting "
            "internal and both makes\nthe modules a different type, so a file written under "
            "one arrangement reads under the\nsame one only, and this run uses --halting "
         << describe(o.halting) << ".\nRemove the file, or point --model-file elsewhere.\n\n  "
         << e.what() << "\n";
}

// ----------------------------------------------------------------------------------------

template <bool USE_ACT>
int train (const arc_dataset& train_set, const arc_dataset& test, long table_rows, const run_options& o)
{
    using network = arc_net<USE_ACT>;
    network net;

    /* The total follows the reference's accounting, in which an epoch is one pass over the
       groups, each giving every example of one of its puzzles, and a step consumes a batch
       of them, whether or not every slot takes a new example. */
    const double per_epoch = train_set.num_groups() * mean_puzzle_examples(train_set);
    const long total_steps = std::max(1L, (long)((double)o.epochs * per_epoch / o.batch_size));
    const long eval_every = std::max(1L, (long)((double)o.eval_interval * per_epoch / o.batch_size));

    std::vector<adam_atan2> solvers(network::num_computational_layers,
                                    adam_atan2(o.weight_decay, o.beta1, o.beta2));
    long step = 0, written_batch = o.batch_size;
    double best_exact = -1;
    const std::string checkpoint = "chkpt-" + o.model_file;
    bool resumed = false;
    try
    {
        resumed = load_checkpoint(checkpoint, net, solvers, step, best_exact, written_batch);
    }
    catch (const std::exception& e)
    {
        explain_mismatch(checkpoint, o, e);
        return 1;
    }
    if (resumed)
    {
        if (written_batch != o.batch_size)
        {
            cout << checkpoint << " was written by a run of batch " << written_batch
                 << ", and this one asks for " << o.batch_size << ".\nResume with --batch-size "
                 << written_batch << ", or remove the file to start afresh.\n";
            return 1;
        }
        cout << "Resuming from " << checkpoint << " at step " << step;
        if (best_exact >= 0)
            cout << ", best held-out rate so far " << 100.0 * best_exact << " %";
        cout << "\n";
    }

    // Set after loading, so that a resumed run takes the options it was given.
    auto& loss = net.loss_details();
    loss.set_ignore_index((long)arc_ignore_label);
    loss.set_normalization(o.softmax ? token_normalization::softmax : token_normalization::stablemax);
    loss.set_halting_position(0);
    loss.set_halting_weight(uses_head(o.halting) ? o.halting_weight : 0.0);

    /* The identifier table needs a row for every identifier either split names. A table
       loaded from a checkpoint keeps the size it was built with, and one too small would
       read the zero row for the identifiers beyond it. */
    auto& table = identifier_table(net);
    if (table.get_num_puzzles() == 0)
        table.set_num_puzzles(table_rows);
    else if (table.get_num_puzzles() < table_rows)
    {
        cout << checkpoint << " holds an identifier table of " << table.get_num_puzzles()
             << " rows, and this dataset names " << table_rows << ".\nIt was written for "
                "another dataset. Remove it, or point --model-file elsewhere.\n";
        return 1;
    }
    /* The table moves at its own rate, expressed against the base rate so that it follows
       the warmup and the schedule the context carries. */
    table.set_puzzle_rate_multiplier(o.puzzle_rate / o.learning_rate);
    table.set_puzzle_weight_decay(o.puzzle_weight_decay);
    if (uses_act(o.halting))
        set_ponder_cost(net, (float)o.ponder_cost);

    /* The modules of the recurrence, and the transition networks of the adaptive layers in
       them, are updated by their layers, out of the solvers' reach. The context hands those
       layers the algorithm, the settings and, every step, the rate, so that they train
       exactly as the rest of the network does. */
    network_context::set_optimizer_params(o.weight_decay, o.beta1, o.beta2,
                                          network_context::optimizer_type::adam_atan2);

    example_stream stream(train_set, o.seed);
    std::mt19937_64 rng(o.seed + 1);
    std::uniform_real_distribution<double> coin(0, 1);
    std::uniform_int_distribution<long> min_passes(std::min(2L, o.max_passes), o.max_passes);

    const long B = o.batch_size;
    std::vector<long> slot_example(B, -1), slot_passes(B, 0);
    std::vector<bool> slot_restart(B, true);
    std::vector<matrix<int, 0, 1>> xs(B);
    std::vector<matrix<unsigned long, 0, 1>> ys(B);
    resizable_tensor x;

    auto& core = recurrence(net);
    core.set_carry_state(true);

    /* The layers draw their parameters on the first forward. The output layer draws its
       weights as the reference draws its own, from a truncated normal law of deviation one
       over the square root of the width, and the halting head is set right after. */
    if (!resumed)
    {
        halting_head(net).set_weight_init(weight_init::truncated_lecun_normal);
        for (long b = 0; b < B; ++b)
            xs[b] = make_input(train_set, stream.next(), o);
        net.to_tensor(xs.begin(), xs.end(), x);
        net.forward(x);
        initialize_halting_head(net);
        core.reset_state();
    }
    cout << "Parameters: " << count_parameters(net).total << ", and an identifier table of "
         << table.get_num_puzzles() << " rows of " << EMBED_DIM << " beside them ("
         << (table.get_num_puzzles() * EMBED_DIM * sizeof(float) >> 20) << " MiB)\n";

    // Figures gathered over the halted slots since the last report.
    long   halted = 0, halted_exact = 0, halted_passes = 0, head_right = 0;
    double halted_cells = 0, token_loss = 0, halting_loss = 0;
    long   steps_reported = 0;
    auto last_report = std::chrono::steady_clock::now();
    auto last_save   = std::chrono::steady_clock::now();

    cout << "\nTraining for " << total_steps << " steps of " << B << " slots, measuring every "
         << eval_every << " steps\n";

    while (step < total_steps && !signal_handler::is_triggered())
    {
        ++step;
        const double lr = rate_at(o, step, total_steps);
        network_context::set_learning_rate(lr);

        // Every slot that is done takes a new example and starts it from the initial state.
        for (long b = 0; b < B; ++b)
        {
            if (!slot_restart[b] && slot_example[b] >= 0)
                continue;
            slot_example[b] = stream.next();
            slot_passes[b] = 0;
            xs[b] = make_input(train_set, slot_example[b], o);
            ys[b] = arc_make_label(train_set, slot_example[b]);
            slot_restart[b] = true;
        }
        core.reset_state(slot_restart);
        net.to_tensor(xs.begin(), xs.end(), x);

        net.compute_parameter_gradients(x, ys.begin());
        net.update_parameters(solvers, lr);

        token_loss += loss.get_token_loss();
        halting_loss += loss.get_halting_loss();
        ++steps_reported;

        const auto& logits = loss.get_halting_logits();
        const auto& exact = loss.get_exact();
        const auto& accuracy = loss.get_token_accuracy();
        for (long b = 0; b < B; ++b)
        {
            const long passes = ++slot_passes[b];
            bool stop = passes >= o.max_passes;
            if (uses_head(o.halting))
            {
                stop = stop || logits[b] > 0;
                // Now and then a slot is held to a random minimum of passes first.
                const long floor = coin(rng) < o.exploration ? min_passes(rng) : 0;
                stop = stop && passes >= floor;
            }
            slot_restart[b] = stop;
            if (stop)
            {
                ++halted;
                halted_exact += exact[b];
                halted_passes += passes;
                halted_cells += accuracy[b];
                head_right += ((logits[b] >= 0) == (exact[b] != 0)) ? 1 : 0;
            }
        }

        /* A line every report_every steps, and at the latest every few minutes, so that a
           configuration whose steps are slow still shows its pace early. */
        const auto now = std::chrono::steady_clock::now();
        if (o.report_every > 0 && (step % o.report_every == 0 || now - last_report >= REPORT_PERIOD))
        {
            const double dt = std::chrono::duration<double>(now - last_report).count();
            const double per = dt > 0 ? steps_reported / dt : 0;
            cout << "step " << step << "/" << total_steps
                 << "  loss " << token_loss / steps_reported;
            if (uses_head(o.halting))
                cout << "  halting " << halting_loss / steps_reported;
            cout << "  lr " << lr;
            if (halted > 0)
            {
                cout << "  |  done " << halted
                     << ", exact " << 100.0 * halted_exact / halted << " %"
                     << ", cells " << 100.0 * halted_cells / halted << " %"
                     << ", passes " << (double)halted_passes / halted;
                if (uses_head(o.halting))
                    cout << ", head right " << 100.0 * head_right / halted << " %";
            }
            if (uses_act(o.halting))
                cout << "  |  adaptive steps " << adaptive_steps(net);
            cout << "  |  " << per << " steps/s";
            if (per > 0)
                cout << ", " << (long)((total_steps - step) / per / 60) << " min left";
            cout << "\n" << std::flush;
            halted = halted_exact = halted_passes = head_right = 0;
            halted_cells = token_loss = halting_loss = 0;
            steps_reported = 0;
            last_report = now;
        }

        if (std::chrono::steady_clock::now() - last_save > std::chrono::minutes(15))
        {
            save_checkpoint(checkpoint, net, solvers, step, best_exact, B);
            last_save = std::chrono::steady_clock::now();
        }

        if (test.num_examples() > 0 && (step % eval_every == 0 || step == total_steps))
        {
            const auto measured_at = std::chrono::steady_clock::now();
            const measurement m = evaluate(net, test, o, o.eval_samples);
            report(m, "held out");
            const double rate = m.examples ? (double)m.exact / m.examples : 0;
            if (rate > best_exact)
            {
                best_exact = rate;
                save_model(o.model_file + ".best", net);
                cout << "  best so far, saved to " << o.model_file << ".best\n";
            }
            // The measurement replaced the carried state, so every slot starts a new example.
            std::fill(slot_restart.begin(), slot_restart.end(), true);
            // Its time is left out of the training pace the next line reports.
            last_report += std::chrono::steady_clock::now() - measured_at;
        }
    }

    /* The memory report describes the extension, so it comes with --verbose only. It is
       printed before the network is cleaned, while the activations of the last step are
       still allocated, so that what the device reports free is what a larger budget could
       still take. */
    if (o.verbose && extended_memory_enabled())
    {
        cout << "\n";
        print_extended_memory_stats(cout);
        cout << "\n";
    }

    save_checkpoint(checkpoint, net, solvers, step, best_exact, B);
    core.set_carry_state(false);
    core.reset_state();
    save_model(o.model_file, net);
    cout << "Model saved to " << o.model_file << "\n";
    network_context::reset();
    return 0;
}

// ----------------------------------------------------------------------------------------

template <bool USE_ACT>
int run (const arc_dataset& train_set, const arc_dataset& test, long table_rows,
         const run_options& o, bool do_train, bool do_eval)
{
    if (do_train)
    {
        const int rc = train<USE_ACT>(train_set, test, table_rows, o);
        if (rc != 0)
            return rc;
    }

    if (do_eval)
    {
        std::string source = o.model_file;
        if (file_exists(o.model_file + ".best"))
            source = o.model_file + ".best";
        if (!file_exists(source))
        {
            cout << "There is no model to evaluate at " << source << "\n";
            return 1;
        }
        arc_net<USE_ACT> net;
        try
        {
            deserialize(source) >> net;
        }
        catch (const std::exception& e)
        {
            explain_mismatch(source, o, e);
            return 1;
        }
        const long rows = identifier_table(net).get_num_puzzles();
        if (!o.blank_id && test.largest_identifier() >= rows)
            cout << "The model's identifier table holds " << rows << " rows and the held-out "
                    "split names identifiers up to " << test.largest_identifier()
                 << ";\nthose beyond the table read the zero row.\n";
        cout << "Evaluating " << source << " after " << o.max_passes << " passes, "
             << (o.blank_id ? "without the identifier" : "with the identifier") << "\n";
        report(evaluate(net, test, o, o.eval_samples), "held out");
    }
    return 0;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        command_line_parser parser;
        parser.add_option("h", "Display this help message");
        parser.add_option("train", "Train the model");
        parser.add_option("eval", "Evaluate a trained model on the held-out split");
        parser.add_option("data", "Directory holding the prepared dataset (default: arc-data)", 1);
        parser.add_option("model-file", "Model file path (default: hrm_arc_model.dat)", 1);
        parser.add_option("halting", "When a slot is done: external, when the halting head says "
                                     "so; none, after --max-passes passes; internal, as none "
                                     "with the adaptive layer inside the modules; both, the "
                                     "adaptive layer and the halting head (default: external)", 1);
        parser.add_option("batch-size", "Slots trained together; the defaults are the reference's "
                                        "settings at 768 and follow another batch by the "
                                        "square-root rule (default: 768)", 1);
        parser.add_option("epochs", "Passes over the training groups, in the reference's "
                                    "accounting (default: 100000)", 1);
        parser.add_option("eval-interval", "Measure every N epochs (default: 10000)", 1);
        parser.add_option("eval-samples", "Held-out examples read by a measurement, as whole "
                                          "variants spread across the split, 0 for all of "
                                          "them (default: 2000)", 1);
        parser.add_option("learning-rate", "Base learning rate (default: 1e-4 at a batch of 768)", 1);
        parser.add_option("warmup-steps", "Steps of linear warmup (default: 2000 at a batch of 768)", 1);
        parser.add_option("lr-min-ratio", "Where the cosine ends, as a fraction of the base rate; "
                                          "1 keeps the rate constant after the warmup (default: 1)", 1);
        parser.add_option("weight-decay", "Decoupled weight decay (default: 0.1 at a batch of 768)", 1);
        parser.add_option("beta1", "First moment decay (default: 0.9 at a batch of 768)", 1);
        parser.add_option("beta2", "Second moment decay (default: 0.95 at a batch of 768)", 1);
        parser.add_option("puzzle-rate", "Rate of the sign descent that trains the identifier "
                                         "table, following the schedule of the base rate "
                                         "(default: 1e-2 at a batch of 768)", 1);
        parser.add_option("puzzle-weight-decay", "Decoupled decay of the identifier rows a "
                                                 "batch names (default: 0.1 at a batch of 768)", 1);
        parser.add_option("no-batch-scaling", "Take the defaults as they are at any batch, "
                                              "without the square-root rule");
        parser.add_option("max-passes", "Passes an example may run, and the passes a measurement "
                                        "runs (default: 16)", 1);
        parser.add_option("exploration", "Probability that a slot is held to a random minimum "
                                         "of passes, with the halting head (default: 0.1)", 1);
        parser.add_option("halting-weight", "Weight of the halting term in the loss (default: 0.5)", 1);
        parser.add_option("ponder-cost", "What the adaptive layer pays per step, with --halting "
                                         "internal or both (default: 0.01)", 1);
        parser.add_option("softmax", "Score the cells under softmax, where the reference uses stablemax");
        parser.add_option("blank-puzzle-id", "Withhold the identifier, so that the model is asked "
                                             "to work without being told the rule");
        parser.add_option("report-every", "Print a line every N steps, and at least every five "
                                          "minutes (default: 100)", 1);
        parser.add_option("seed", "Seed of the draws (default: 0)", 1);
        parser.add_option("verbose", "Report what the memory extension is doing, and its figures "
                                     "when training stops");
        parser.add_option("no-extended-memory", "Do not stream tensors through the device; "
                                                "every allocation then has to fit at once");
        parser.add_option("vram-budget", "Device memory the extension may use, in MiB", 1);
        parser.add_option("vram-store", "Directory holding the store, or \"none\" to keep "
                                        "evicted blocks in host memory only", 1);
        parser.add_option("host-limit", "Pinned host memory the extension may take, in MiB", 1);
        parser.parse(argc, argv);

        const bool do_train = parser.option("train");
        const bool do_eval  = parser.option("eval");
        if (parser.option("h") || (!do_train && !do_eval))
        {
            parser.print_options();
            cout << "\nHierarchical reasoning on ARC-AGI, trained as the reference trains it\n"
                 << "  Prepare  : slm_tools/build_arc_dataset.py --download --data-dir arc-agi "
                    "--out-dir arc-data --protocol reference --extra-corpus ConceptARC/corpus\n"
                 << "  Train    : " << argv[0] << " --train --data arc-data\n"
                 << "  Evaluate : " << argv[0] << " --eval --data arc-data --eval-samples 0\n";
            return 0;
        }

        if (!parser.option("no-extended-memory"))
        {
            extended_memory_options xopts;
            if (parser.option("vram-budget"))
                xopts.vram_budget = (size_t)get_option(parser, "vram-budget", 0) << 20;
            xopts.store_path = get_option(parser, "vram-store", default_extended_memory_store_path());
            if (xopts.store_path == "none")
                xopts.store_path.clear();
            xopts.max_pinned_bytes = (size_t)get_option(parser, "host-limit",
                                        (unsigned long)(xopts.vram_budget >> 20)) << 20;
            xopts.verbose = parser.option("verbose");
            if (!enable_extended_memory(xopts))
                cout << "Extended memory was requested but is unavailable in this build; "
                        "continuing without it.\n";
        }

        run_options o;
        o.data_dir            = get_option(parser, "data", o.data_dir);
        o.model_file          = get_option(parser, "model-file", o.model_file);
        o.batch_size          = get_option(parser, "batch-size", o.batch_size);
        o.epochs              = get_option(parser, "epochs", o.epochs);
        o.eval_interval       = get_option(parser, "eval-interval", o.eval_interval);
        o.eval_samples        = get_option(parser, "eval-samples", o.eval_samples);
        o.learning_rate       = get_option(parser, "learning-rate", o.learning_rate);
        o.warmup_steps        = get_option(parser, "warmup-steps", o.warmup_steps);
        o.lr_min_ratio        = get_option(parser, "lr-min-ratio", o.lr_min_ratio);
        o.weight_decay        = get_option(parser, "weight-decay", o.weight_decay);
        o.beta1               = get_option(parser, "beta1", o.beta1);
        o.beta2               = get_option(parser, "beta2", o.beta2);
        o.puzzle_rate         = get_option(parser, "puzzle-rate", o.puzzle_rate);
        o.puzzle_weight_decay = get_option(parser, "puzzle-weight-decay", o.puzzle_weight_decay);
        o.max_passes          = get_option(parser, "max-passes", o.max_passes);
        o.exploration         = get_option(parser, "exploration", o.exploration);
        o.halting_weight      = get_option(parser, "halting-weight", o.halting_weight);
        o.ponder_cost         = get_option(parser, "ponder-cost", o.ponder_cost);
        o.softmax             = parser.option("softmax");
        o.blank_id            = parser.option("blank-puzzle-id");
        o.report_every        = get_option(parser, "report-every", o.report_every);
        o.seed                = get_option(parser, "seed", o.seed);
        o.verbose             = parser.option("verbose");

        const std::string h = get_option(parser, "halting", "external");
        if      (h == "external") o.halting = halting_mode::external;
        else if (h == "none")     o.halting = halting_mode::none;
        else if (h == "internal") o.halting = halting_mode::internal;
        else if (h == "both")     o.halting = halting_mode::both;
        else
        {
            cout << "Unknown halting mode: " << h << "\nExpected one of: external, none, internal, both\n";
            return 1;
        }

        if (o.batch_size < 1 || o.max_passes < 1 || o.epochs < 1)
        {
            cout << "--batch-size, --max-passes and --epochs must be at least 1\n";
            return 1;
        }
        if (!(o.learning_rate > 0) || o.puzzle_rate < 0 || o.puzzle_weight_decay < 0)
        {
            cout << "--learning-rate must be positive, --puzzle-rate and --puzzle-weight-decay "
                    "at least 0\n";
            return 1;
        }
        transpose_to_batch(o, parser);

        signal_handler::setup();

        arc_dataset train_set, test;
        if (do_train)
            train_set = load_arc_dataset(o.data_dir + "/training.bin");
        try
        {
            test = load_arc_dataset(o.data_dir + "/evaluation.bin");
        }
        catch (const std::exception& e)
        {
            if (do_eval)
                throw;
            cout << "No held-out split, so the run measures nothing as it goes: " << e.what() << "\n";
        }

        const arc_dataset* splits[] = {&train_set, &test};
        for (const arc_dataset* d : splits)
        {
            if (d->num_examples() == 0)
                continue;
            if (d->seq_len != WINDOW || d->colour_vocab != COLOUR_VOCAB)
            {
                cout << "The dataset holds sequences of " << d->seq_len << " cells over "
                     << d->colour_vocab << " tokens; this program reads ARC files written by "
                        "build_arc_dataset.py.\n";
                return 1;
            }
            /* The names record the variant of every puzzle, which the vote undoes. A split
               without them was written by an earlier version of the builder. */
            if (d->puzzle_names.empty())
            {
                cout << "The splits in " << o.data_dir << " have no .names file beside them. "
                        "They were written by an\nearlier build_arc_dataset.py; rebuild them "
                        "with the current one.\n";
                return 1;
            }
        }
        if (do_train && (train_set.num_examples() == 0 || train_set.num_groups() == 0))
        {
            cout << "The training split holds no examples.\n";
            return 1;
        }

        /* One row per identifier either split names, and a single row, the blank one, when
           the identifier is withheld. */
        const long table_rows = o.blank_id ? 1L
            : 1L + std::max<long>(train_set.largest_identifier(), test.largest_identifier());

        cout << "=== ARC-AGI hierarchical reasoning ===\n"
             << "  data          : " << o.data_dir << "\n";
        if (do_train)
            cout << "  training      : " << train_set.num_examples() << " examples of "
                 << (long)train_set.puzzle_indices.size() - 1 << " puzzles in "
                 << train_set.num_groups() << " groups\n";
        cout << "  held out      : " << test.num_examples() << " examples of "
             << std::max(0L, test.num_puzzles() - 1) << " puzzles in "
             << std::max(0L, test.num_groups()) << " groups\n"
             << "  network       : width " << EMBED_DIM << ", " << NUM_HEADS << " heads, "
             << NUM_H_LAYERS << " and " << NUM_L_LAYERS << " blocks, " << HRM_N << " cycles of "
             << HRM_T << " steps per pass, attention "
             << (ATTENTION_CAUSAL ? "causal" : "bidirectional") << "\n"
             << "  structure     : post-norm blocks without gain, SwiGLU of "
             << impl::hrm_reference_ffn_width(EMBED_DIM) << ", initial states drawn once\n"
             << "  identifier    : " << (o.blank_id ? "withheld" : "given") << "\n"
             << "  loss          : " << (o.softmax ? "softmax" : "stablemax") << " cross-entropy"
             << (uses_head(o.halting) ? ", and the halting head" : "") << "\n"
             << "  batch         : " << o.batch_size << " slots";
        if (o.transposed)
            cout << ", the reference's settings at " << REFERENCE_BATCH
                 << " carried over by the square-root rule";
        cout << "\n"
             << "  optimizer     : Adam-atan2, rate " << o.learning_rate << ", decay "
             << o.weight_decay << ", betas " << o.beta1 << " and " << o.beta2
             << ", warmup " << o.warmup_steps << " steps\n"
             << "  identifiers   : sign descent, rate " << o.puzzle_rate << ", decay "
             << o.puzzle_weight_decay << "\n"
             << "  halting       : " << describe(o.halting) << "\n"
             << "  passes        : up to " << o.max_passes << " per example"
             << (uses_head(o.halting) ? ", halting when the head says so" : ", all of them") << "\n";
        if (uses_act(o.halting))
            cout << "  ponder cost   : " << o.ponder_cost << " per adaptive step\n";
        cout << "\n";

        /* The adaptive layer changes the type of the modules, so each arrangement
           instantiates the network it builds. */
        return uses_act(o.halting)
            ? run<true >(train_set, test, table_rows, o, do_train, do_eval)
            : run<false>(train_set, test, table_rows, o, do_train, do_eval);
    }
    catch (std::exception& e)
    {
        cout << "Exception: " << e.what() << endl;
        return 1;
    }
}
