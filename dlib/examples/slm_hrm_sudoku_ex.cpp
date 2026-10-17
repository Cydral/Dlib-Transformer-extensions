// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    Hierarchical reasoning on Sudoku-Extreme, trained the way the reference trains it.

    The reference work reports its Sudoku result from a model trained on a thousand
    puzzles, each augmented a thousand times, and evaluated on the 423 thousand puzzles of
    the test split. This program reproduces that procedure in dlib, with the same network
    shape, the same halting head trained jointly with the network, the same supervision of
    every pass, the same loss and the same optimizer. Sudoku is the cheap place to test
    the reproduction, since the reference trains it on a single GPU, where its ARC run
    needs several.

    HOW A BATCH IS TRAINED

    A batch is a set of slots, each holding one puzzle and the recurrent state the network
    has reached on it. A training step runs one pass of the recurrence on every slot, from
    the state the previous step left, and is supervised in full, every slot's answer being
    scored whatever pass it is at. This is deep supervision, the part of the training to
    which the independent analysis of the reference attributes most of the result.

    After each step every slot decides whether it is done. A halting logit, read from the
    first position of the sequence, says whether the answer is believed exact, and the
    slot halts when the logit is positive or when it has run the maximum number of passes.
    A halted slot takes a new puzzle and starts it from the initial state, while the
    others continue. Hard puzzles therefore get more passes and solved ones free their
    place, which is what the halting buys during training.

    The halting head is trained with the network, by the loss, towards whether the answer
    of the pass was exact. Only the loss sees the labels, and exactness is a question about
    the labels. The reference also trains a second, "continue", logit towards a target
    bootstrapped from one more pass of the network; like the tiny recursive model that
    followed it, this program trains the halting logit alone, which removes that extra
    pass.

    With probability 0.1 a slot is held to a random minimum number of passes before it may
    halt, as in the reference, so that the head also sees the passes it would have skipped.

    HOW IT IS MEASURED

    Evaluation runs every test puzzle through the maximum number of passes, from the
    initial state, and reads the last answer, so the halting decision plays no part in the
    measurement, as in the reference. An answer counts only if all 81 cells are right.

    WHAT DIFFERS FROM THE REFERENCE

    The blocks are pre-norm, and each module's output is normalised, which bounds the
    carried state as the reference's post-norm blocks do. The initial states are zero,
    where the reference draws them once at random. The computation is in single
    precision. Whether these matter is for the results to say.

    Prepare the data first:
        slm_tools/build_sudoku_dataset.py --download --out-dir sudoku-data

    Then:
        ./slm_hrm_sudoku_ex --train --data sudoku-data
        ./slm_hrm_sudoku_ex --eval  --data sudoku-data --eval-samples 0

    The reference trains with a batch of 384. That batch holds the activations of two
    modules of four blocks over 82 positions, a large working set in single precision. The
    memory extension, on unless --no-extended-memory is given, carries what does not fit
    the card at a cost in speed. A smaller --batch-size is the other way, and the number
    of steps follows from --epochs, so the amount of data seen stays the same.
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
    A puzzle is 81 cells, preceded by one position that carries the puzzle identifier.
    Every Sudoku puzzle carries the same one, as in the reference, so that position is a
    learned vector in front of every grid. It is also where the halting logit is read.

    The cells use eleven tokens. Token 0 is the padding, which Sudoku never needs, token 1
    the blank cell and tokens 2 to 10 the digits. The output predicts all eleven at every
    position and adds one channel for the halting logit.
*/
const long CELLS        = 81;
const long SEQ_LEN      = CELLS + 1;
const long CELL_TOKENS  = 11;
const long INPUT_TOKENS = CELL_TOKENS + 1;     // and the identifier
const long OUTPUTS      = CELL_TOKENS + 1;     // and the halting logit

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
    The input is a one-hot vector per position, scaled by the square root of the width,
    and the embedding is a projection of it.

    The reference's embedding computes the same thing, a lookup in a table initialised
    small and multiplied by the square root of the width. Written as a projection, the
    table is an ordinary parameter, so the solver trains it as it trains the rest of the
    network, with the same algorithm, rate and decay, as the reference does.
*/
const float EMBED_SCALE = 22.627417f;          // sqrt(EMBED_DIM)

using sudoku_config = hrm_transformer_config<
    INPUT_TOKENS, NUM_H_LAYERS, NUM_L_LAYERS, NUM_HEADS, EMBED_DIM, HRM_N, HRM_T,
    gelu, dropout_10, attention_impl::unified, NUM_KV_HEADS, false, CELL_TOKENS, false>;

/*
    The unified blocks carry no dropout, as the reference has none, so one network type
    serves training and measurement. Attention is bidirectional, since a cell depends on
    cells on both sides of it.
*/
using net_type = loss_halting_cross_entropy_per_token<linear<OUTPUTS, rms_norm<
    hrm<sudoku_config::train_h_net_type, sudoku_config::train_l_net_type, HRM_N, HRM_T,
    linear_no_bias<EMBED_DIM, input<matrix<float>>>>>>>;

// The layers the program reaches into, counted from the loss.
template <typename net> auto& halting_head (net& n) { return layer<1>(n).layer_details(); }
template <typename net> auto& recurrence (net& n) { return layer<3>(n).layer_details(); }

// ----------------------------------------------------------------------------------------

struct run_options
{
    std::string data_dir       = "sudoku-data";
    std::string model_file     = "hrm_sudoku_model.dat";
    long        batch_size     = 384;
    long        epochs         = 20000;
    long        eval_interval  = 2000;
    long        eval_samples   = 4096;
    double      learning_rate  = 7e-5;
    long        warmup_steps   = 2000;
    double      lr_min_ratio   = 1.0;
    double      weight_decay   = 1.0;
    double      beta1          = 0.9;
    double      beta2          = 0.95;
    long        max_passes     = 16;
    double      exploration    = 0.1;
    double      halting_weight = 0.5;
    bool        fixed_passes   = false;
    bool        softmax        = false;
    long        report_every   = 100;
    unsigned long seed         = 0;
    bool        verbose        = false;
};

// ----------------------------------------------------------------------------------------

/*
    The network's input for one example, one scaled one-hot row per position over the
    identifier token and the eleven cell tokens.
*/
inline matrix<float> make_input (const arc_dataset& d, long example)
{
    const matrix<int, 0, 1> tokens = arc_make_input(d, example, d.identifier_of(example));
    matrix<float> x = zeros_matrix<float>(SEQ_LEN, INPUT_TOKENS);
    for (long t = 0; t < SEQ_LEN; ++t)
    {
        DLIB_CASSERT(tokens(t) >= 0 && tokens(t) < INPUT_TOKENS,
            "token " << tokens(t) << " at position " << t << " is outside the vocabulary");
        x(t, tokens(t)) = EMBED_SCALE;
    }
    return x;
}

/*
    The halting head starts silent, with a zero weight and a bias of minus five as in the
    reference, so that no slot halts on its own before the head has learned anything and
    every early puzzle runs its full number of passes.
*/
void initialize_halting_head (net_type& net)
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

// ----------------------------------------------------------------------------------------

/*
    Draws training examples as the reference does, taking the groups in a random order and
    one variant of each group at random. A group is one original puzzle and its
    augmentations, so every original is seen once per pass over the groups, however many
    variants it has.
*/
class example_stream
{
public:
    example_stream (const arc_dataset& d, unsigned long seed) : data(d), rng(seed) {}

    long next ()
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
        std::uniform_int_distribution<long> pick(first, last - 1);
        const long puzzle = pick(rng);
        return data.puzzle_indices[(size_t)puzzle];
    }

private:
    const arc_dataset& data;
    std::mt19937_64 rng;
    std::vector<long> order;
    size_t position = 0;
};

// ----------------------------------------------------------------------------------------

struct measurement
{
    long   puzzles = 0;
    long   exact   = 0;
    double cells   = 0;
};

/*
    Runs test puzzles through the full number of passes from the initial state and scores
    the last answer. A limit reads that many puzzles spread evenly across the split, the
    same ones every time, so successive measurements during a run compare like with like.

    The network's carried state is replaced by the measurement's, so the caller restarts
    its slots afterwards.
*/
measurement evaluate (net_type& net, const arc_dataset& test, const run_options& o, long limit)
{
    measurement m;
    const long available = test.num_examples();
    const long n = (limit > 0 && limit < available) ? limit : available;
    const double stride = (double)available / (double)std::max(1L, n);

    auto& core = recurrence(net);
    std::vector<matrix<float>> xs;
    std::vector<long> ids;
    resizable_tensor x;
    long cells_right = 0, cells_total = 0;

    for (long k = 0; k < n && !signal_handler::is_triggered(); )
    {
        xs.clear();
        ids.clear();
        for (; k < n && (long)xs.size() < o.batch_size; ++k)
        {
            const long e = std::min(available - 1, (long)(k * stride));
            ids.push_back(e);
            xs.push_back(make_input(test, e));
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
            const auto label = arc_make_label(test, ids[b]);
            long right = 0, scored = 0;
            for (long t = 0; t < SEQ_LEN; ++t)
            {
                if (label(t) == arc_ignore_label)
                    continue;
                const float* row = logits + ((long)b * SEQ_LEN + t) * OUTPUTS;
                long best = 0;
                for (long c = 1; c < CELL_TOKENS; ++c)
                    if (row[c] > row[best]) best = c;
                ++scored;
                if ((unsigned long)best == label(t)) ++right;
            }
            cells_right += right;
            cells_total += scored;
            ++m.puzzles;
            if (scored > 0 && right == scored) ++m.exact;
        }
    }
    core.reset_state();
    m.cells = cells_total ? 100.0 * cells_right / cells_total : 0.0;
    return m;
}

void report (const measurement& m, const char* what)
{
    cout << "  " << what << ": " << m.exact << " of " << m.puzzles << " puzzles exact ("
         << (m.puzzles ? 100.0 * m.exact / m.puzzles : 0.0) << " %), "
         << m.cells << " % of cells\n" << std::flush;
}

// ----------------------------------------------------------------------------------------

/*
    The rate at a given step. A linear warmup is followed by a cosine down to lr_min_ratio
    of the base rate, and the reference's Sudoku setting keeps the ratio at one, which
    leaves the rate constant once warm.
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
    A checkpoint holds the network, the solvers and the step. The slots are not kept, so a
    resumed run starts every slot on a new puzzle, which costs the puzzles in flight and
    nothing else.

    The network is cleaned first, since a serialized network carries its activations
    otherwise. The next step allocates them again, and the recurrent state the slots carry
    is kept, as cleaning leaves it alone.
*/
void save_checkpoint (const std::string& file, net_type& net,
                      const std::vector<adam_atan2>& solvers, long step)
{
    net.clean();
    const std::string tmp = file + "_";
    {
        std::ofstream out(tmp, std::ios::binary);
        serialize(net, out);
        serialize(solvers, out);
        serialize(step, out);
        if (!out)
            throw std::runtime_error("unable to write " + tmp);
    }
    std::rename(tmp.c_str(), file.c_str());
}

bool load_checkpoint (const std::string& file, net_type& net,
                      std::vector<adam_atan2>& solvers, long& step)
{
    if (!file_exists(file))
        return false;
    std::ifstream in(file, std::ios::binary);
    deserialize(net, in);
    deserialize(solvers, in);
    deserialize(step, in);
    return true;
}

// ----------------------------------------------------------------------------------------

int train (const arc_dataset& train_set, const arc_dataset& test, const run_options& o)
{
    net_type net;
    auto& loss = net.loss_details();
    loss.set_ignore_index((long)arc_ignore_label);
    loss.set_normalization(o.softmax ? token_normalization::softmax : token_normalization::stablemax);
    loss.set_halting_position(0);
    loss.set_halting_weight(o.fixed_passes ? 0.0 : o.halting_weight);

    /* The total follows the reference's accounting, in which an epoch is one pass over the
       groups and a step consumes a batch of them, whether or not every slot takes a new
       puzzle. */
    const long total_steps = std::max(1L, (long)((double)o.epochs * train_set.num_groups() / o.batch_size));
    const long eval_every = std::max(1L, (long)((double)o.eval_interval * train_set.num_groups() / o.batch_size));

    std::vector<adam_atan2> solvers(net_type::num_computational_layers,
                                    adam_atan2(o.weight_decay, o.beta1, o.beta2));
    long step = 0;
    const std::string checkpoint = "chkpt-" + o.model_file;
    const bool resumed = load_checkpoint(checkpoint, net, solvers, step);
    if (resumed)
        cout << "Resuming from " << checkpoint << " at step " << step << "\n";

    /* The modules of the recurrence are updated by the layer itself, out of the solvers'
       reach. The context hands it the algorithm, the settings and, every step, the rate,
       so that the modules train exactly as the rest of the network does. */
    network_context::set_optimizer_params(o.weight_decay, o.beta1, o.beta2,
                                          network_context::optimizer_type::adam_atan2);

    example_stream stream(train_set, o.seed);
    std::mt19937_64 rng(o.seed + 1);
    std::uniform_real_distribution<double> coin(0, 1);
    std::uniform_int_distribution<long> min_passes(std::min(2L, o.max_passes), o.max_passes);

    const long B = o.batch_size;
    std::vector<long> slot_example(B, -1), slot_passes(B, 0);
    std::vector<bool> slot_restart(B, true);
    std::vector<matrix<float>> xs(B);
    std::vector<matrix<unsigned long, 0, 1>> ys(B);
    resizable_tensor x;

    auto& core = recurrence(net);
    core.set_carry_state(true);

    // The halting head is set before the first step, once the layers have their parameters.
    if (!resumed)
    {
        for (long b = 0; b < B; ++b)
        {
            slot_example[b] = stream.next();
            xs[b] = make_input(train_set, slot_example[b]);
        }
        net.to_tensor(xs.begin(), xs.end(), x);
        net.forward(x);
        initialize_halting_head(net);
        core.reset_state();
        std::fill(slot_example.begin(), slot_example.end(), -1);
    }
    cout << "Parameters: " << count_parameters(net).total << "\n";

    // Figures gathered over the halted slots since the last report.
    long   halted = 0, halted_exact = 0, halted_passes = 0, head_right = 0;
    double halted_cells = 0, token_loss = 0, halting_loss = 0;
    long   steps_reported = 0;
    double best_exact = -1;
    auto last_report = std::chrono::steady_clock::now();
    auto last_save   = std::chrono::steady_clock::now();

    cout << "\nTraining for " << total_steps << " steps of " << B << " slots, measuring every "
         << eval_every << " steps\n";

    while (step < total_steps && !signal_handler::is_triggered())
    {
        ++step;
        const double lr = rate_at(o, step, total_steps);
        network_context::set_learning_rate(lr);

        // Every slot that halted takes a new puzzle and starts it from the initial state.
        for (long b = 0; b < B; ++b)
        {
            if (!slot_restart[b] && slot_example[b] >= 0)
                continue;
            slot_example[b] = stream.next();
            slot_passes[b] = 0;
            xs[b] = make_input(train_set, slot_example[b]);
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
            if (!o.fixed_passes)
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

        if (o.report_every > 0 && step % o.report_every == 0)
        {
            const auto now = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(now - last_report).count();
            const double per = dt > 0 ? steps_reported / dt : 0;
            cout << "step " << step << "/" << total_steps
                 << "  loss " << token_loss / steps_reported
                 << "  halting " << halting_loss / steps_reported
                 << "  lr " << lr;
            if (halted > 0)
                cout << "  |  halted " << halted
                     << ", exact " << 100.0 * halted_exact / halted << " %"
                     << ", cells " << 100.0 * halted_cells / halted << " %"
                     << ", passes " << (double)halted_passes / halted
                     << ", head right " << 100.0 * head_right / halted << " %";
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
            save_checkpoint(checkpoint, net, solvers, step);
            last_save = std::chrono::steady_clock::now();
        }

        if (test.num_examples() > 0 && (step % eval_every == 0 || step == total_steps))
        {
            const measurement m = evaluate(net, test, o, o.eval_samples);
            report(m, "held out");
            const double rate = m.puzzles ? (double)m.exact / m.puzzles : 0;
            if (rate > best_exact)
            {
                best_exact = rate;
                net.clean();
                serialize(o.model_file + ".best") << net;
                cout << "  best so far, saved to " << o.model_file << ".best\n";
            }
            // The measurement replaced the carried state, so every slot starts a new puzzle.
            std::fill(slot_restart.begin(), slot_restart.end(), true);
            last_report = std::chrono::steady_clock::now();
        }
    }

    save_checkpoint(checkpoint, net, solvers, step);
    core.set_carry_state(false);
    core.reset_state();
    net.clean();
    serialize(o.model_file) << net;
    cout << "Model saved to " << o.model_file << "\n";
    network_context::reset();
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
        parser.add_option("eval", "Evaluate a trained model on the test split");
        parser.add_option("data", "Directory holding the prepared dataset (default: sudoku-data)", 1);
        parser.add_option("model-file", "Model file path (default: hrm_sudoku_model.dat)", 1);
        parser.add_option("batch-size", "Slots trained together (default: 384, the reference's)", 1);
        parser.add_option("epochs", "Passes over the training groups, in the reference's "
                                    "accounting (default: 20000)", 1);
        parser.add_option("eval-interval", "Measure every N epochs (default: 2000)", 1);
        parser.add_option("eval-samples", "Test puzzles read by a measurement, spread across the "
                                          "split, 0 for all of them (default: 4096)", 1);
        parser.add_option("learning-rate", "Base learning rate (default: 7e-5)", 1);
        parser.add_option("warmup-steps", "Steps of linear warmup (default: 2000)", 1);
        parser.add_option("lr-min-ratio", "Where the cosine ends, as a fraction of the base rate; "
                                          "1 keeps the rate constant after the warmup (default: 1)", 1);
        parser.add_option("weight-decay", "Decoupled weight decay (default: 1.0)", 1);
        parser.add_option("beta1", "First moment decay (default: 0.9)", 1);
        parser.add_option("beta2", "Second moment decay (default: 0.95)", 1);
        parser.add_option("max-passes", "Passes a puzzle may run, and the passes a measurement "
                                        "runs (default: 16)", 1);
        parser.add_option("exploration", "Probability that a slot is held to a random minimum "
                                         "of passes (default: 0.1)", 1);
        parser.add_option("halting-weight", "Weight of the halting term in the loss (default: 0.5)", 1);
        parser.add_option("fixed-passes", "No learned halting: every puzzle runs the maximum "
                                          "number of passes, each one supervised");
        parser.add_option("softmax", "Score the cells under softmax, where the reference uses stablemax");
        parser.add_option("report-every", "Print a line every N steps (default: 100)", 1);
        parser.add_option("seed", "Seed of the draws (default: 0)", 1);
        parser.add_option("verbose", "Report what the memory extension is doing");
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
            cout << "\nHierarchical reasoning on Sudoku-Extreme, trained as the reference trains it\n"
                 << "  Prepare  : slm_tools/build_sudoku_dataset.py --download --out-dir sudoku-data\n"
                 << "  Train    : " << argv[0] << " --train --data sudoku-data\n"
                 << "  Evaluate : " << argv[0] << " --eval --data sudoku-data --eval-samples 0\n";
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
        o.data_dir       = get_option(parser, "data", o.data_dir);
        o.model_file     = get_option(parser, "model-file", o.model_file);
        o.batch_size     = get_option(parser, "batch-size", o.batch_size);
        o.epochs         = get_option(parser, "epochs", o.epochs);
        o.eval_interval  = get_option(parser, "eval-interval", o.eval_interval);
        o.eval_samples   = get_option(parser, "eval-samples", o.eval_samples);
        o.learning_rate  = get_option(parser, "learning-rate", o.learning_rate);
        o.warmup_steps   = get_option(parser, "warmup-steps", o.warmup_steps);
        o.lr_min_ratio   = get_option(parser, "lr-min-ratio", o.lr_min_ratio);
        o.weight_decay   = get_option(parser, "weight-decay", o.weight_decay);
        o.beta1          = get_option(parser, "beta1", o.beta1);
        o.beta2          = get_option(parser, "beta2", o.beta2);
        o.max_passes     = get_option(parser, "max-passes", o.max_passes);
        o.exploration    = get_option(parser, "exploration", o.exploration);
        o.halting_weight = get_option(parser, "halting-weight", o.halting_weight);
        o.fixed_passes   = parser.option("fixed-passes");
        o.softmax        = parser.option("softmax");
        o.report_every   = get_option(parser, "report-every", o.report_every);
        o.seed           = get_option(parser, "seed", o.seed);
        o.verbose        = parser.option("verbose");

        if (o.batch_size < 1 || o.max_passes < 1 || o.epochs < 1)
        {
            cout << "--batch-size, --max-passes and --epochs must be at least 1\n";
            return 1;
        }

        signal_handler::setup();

        arc_dataset train_set;
        if (do_train)
            train_set = load_arc_dataset(o.data_dir + "/train.bin");
        const arc_dataset test = load_arc_dataset(o.data_dir + "/test.bin");

        const arc_dataset* splits[] = {&train_set, &test};
        for (const arc_dataset* d : splits)
        {
            if (d->num_examples() == 0)
                continue;
            if (d->seq_len != CELLS || d->colour_vocab != CELL_TOKENS || d->largest_identifier() != 0)
            {
                cout << "The dataset holds sequences of " << d->seq_len << " cells over "
                     << d->colour_vocab << " tokens; this program reads Sudoku files written by "
                        "build_sudoku_dataset.py.\n";
                return 1;
            }
        }

        cout << "=== Sudoku-Extreme hierarchical reasoning ===\n"
             << "  data          : " << o.data_dir << "\n";
        if (do_train)
            cout << "  training      : " << train_set.num_examples() << " examples in "
                 << train_set.num_groups() << " groups\n";
        cout << "  test          : " << test.num_examples() << " puzzles\n"
             << "  network       : width " << EMBED_DIM << ", " << NUM_HEADS << " heads, "
             << NUM_H_LAYERS << " and " << NUM_L_LAYERS << " blocks, " << HRM_N << " cycles of "
             << HRM_T << " steps per pass\n"
             << "  loss          : " << (o.softmax ? "softmax" : "stablemax") << " cross-entropy, "
             << (o.fixed_passes ? "no halting head" : "and the halting head") << "\n"
             << "  optimizer     : Adam-atan2, rate " << o.learning_rate << ", decay "
             << o.weight_decay << ", betas " << o.beta1 << " and " << o.beta2 << "\n"
             << "  passes        : up to " << o.max_passes << " per puzzle"
             << (o.fixed_passes ? ", all of them" : ", halting when the head says so") << "\n\n";

        if (do_train)
        {
            const int rc = train(train_set, test, o);
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
            net_type net;
            deserialize(source) >> net;
            cout << "Evaluating " << source << " after " << o.max_passes << " passes\n";
            report(evaluate(net, test, o, o.eval_samples), "test");
        }
        return 0;
    }
    catch (std::exception& e)
    {
        cout << "Exception: " << e.what() << endl;
        return 1;
    }
}
