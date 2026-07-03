/*!
    @file slm_hrm_arc_agi.cpp
    @brief Transformer-based training for ARC-AGI reasoning tasks

    This program implements a complete training and evaluation pipeline for
    solving ARC-AGI (Abstraction and Reasoning Corpus) tasks using a
    Transformer-based architecture.

    Key capabilities:
    - Learning visual transformation patterns from demonstrations
    - Generating output grids token-by-token autoregressively
    - Handling non-square grids through implicit dimension encoding

    Usage modes:
    --train       Train model on ARC-AGI training set
    --eval        Evaluate model on test pairs

    References:
    [1] Chollet, "On the Measure of Intelligence" (ARC-AGI) arXiv:1911.01547
!*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>

using namespace std;
using namespace dlib;

// ARC-AGI grid bounds
constexpr long MAX_ROWS = 30;
constexpr long MAX_COLS = 30;

// Training context window. Sized to comfortably hold 2-3 demonstration pairs
// while keeping attention compute tractable for HRM (no KV cache during
// inference: each generated token reprocesses the full window). With grids
// up to ~150-400 tokens including row separators, 768 is a pragmatic
// compromise; raise once initial convergence is established.
constexpr long WINDOW_LEN = 768;
constexpr long MAX_OUTPUT_TOKENS = (MAX_ROWS * (MAX_COLS + 1) + 1) * 110 / 100;

/* YaRN context extension. Evaluation can feed contexts longer than WINDOW_LEN (full
   forward per generated token, no KV cache), so the RoPE frequencies must be rescaled
   for lengths beyond the training window. YaRN is opt-in in the library (classical
   RoPE by default); enabling it here with original_len pinned to WINDOW_LEN keeps the
   scaling stable across trig-cache rebuilds and independent of the first forward size.
   The setter targets every layer exposing set_yarn_params (the fused attention and the
   standalone rotary embedding), and is a no-op for all other layers. */
struct yarn_enabler
{
    template <typename T>
    auto set(T& layer, int) -> decltype((void)layer.set_yarn_params(0.0f, 0.0f, 0, false))
    {
        layer.set_yarn_params(1.0f, 0.5f, WINDOW_LEN, true);
    }
    template <typename T> void set(T&, long) {}
    template <typename T> void operator()(T& layer) { set(layer, 0); }
};

template <typename net_type>
void enable_yarn_context_extension(net_type& net)
{
    yarn_enabler v;
    visit_computational_layers(net, v);
}

// Utility function to validate token sequence before detokenization
bool validate_token_sequence(const std::vector<int>& tokens, bool verbose = false)
{
    if (tokens.empty()) return false;

    std::vector<long> current_row_lengths;
    long current_row_length = 0;

    for (int token : tokens) {
        if (token == TOKEN_ROW_END) {
            if (current_row_length > 0) {
                current_row_lengths.push_back(current_row_length);
                current_row_length = 0;
            }
        }
        else if (token >= COLOR_0 && token <= COLOR_9) {
            current_row_length++;
        }
        else if (token == TOKEN_END_OF_OUTPUT || token == TOKEN_SEP_IO || token == TOKEN_SEP_PAIR) {
            break;
        }
    }

    // Check if all rows have the same length
    if (current_row_lengths.empty()) {
        if (verbose) cout << "      Validation: No complete rows found\n";
        return false;
    }

    long expected_length = current_row_lengths[0];
    for (size_t i = 1; i < current_row_lengths.size(); ++i) {
        if (current_row_lengths[i] != expected_length) {
            if (verbose) {
                cout << "      Validation: Inconsistent row lengths detected\n";
                cout << "        Row 0 has " << expected_length << " columns\n";
                cout << "        Row " << i << " has " << current_row_lengths[i] << " columns\n";
            }
            return false;
        }
    }

    return true;
}

/*!
    WHAT THIS OBJECT REPRESENTS
        Tracks the generation state of an ARC-AGI output grid during autoregressive
        token generation. Monitors row consistency and detects invalid patterns early.

        ARC-AGI constraints:
        - Maximum grid size: 30×30
        - All rows must have the same length
        - Valid tokens: 0-9 (colors), TOKEN_ROW_END
!*/
struct generation_state
{
    std::vector<long> row_lengths;      // Length of each completed row
    long current_row_length = 0;        // Length of current incomplete row
    bool is_valid = true;               // Whether generation is valid so far
    bool is_complete = false;           // Whether a complete grid has been generated

    void add_token(int token)
    {
        if (token == TOKEN_ROW_END)
        {
            if (current_row_length > 0)
            {
                // Check consistency with previous rows
                if (!row_lengths.empty() && row_lengths[0] != current_row_length)
                    is_valid = false;  // Inconsistent row lengths

                row_lengths.push_back(current_row_length);
                current_row_length = 0;

                // Stop if too many rows (ARC-AGI max is 30)
                if (row_lengths.size() > 30) is_valid = false;
            }
        }
        else if (token >= COLOR_0 && token <= COLOR_9)
        {
            current_row_length++;

            // Stop if row is too long (ARC-AGI max is 30)
            if (current_row_length > 30) is_valid = false;
        }
    }

    bool should_stop() const
    {
        // Stop if generation has become invalid
        if (!is_valid) return true;

        // Continue if we have a valid grid in progress
        // We could add more sophisticated stopping conditions here
        // (e.g., if we detect the expected output size)

        return false;
    }

    size_t num_rows() const { return row_lengths.size(); }

    long grid_width() const
    {
        return row_lengths.empty() ? 0 : row_lengths[0];
    }
};

struct generation_result
{
    arc_grid_t grid;
    long context_size;
    long window_size;
    bool context_fits;
    long tokens_truncated;

    generation_result(long ctx_size, long win_size)
        : context_size(ctx_size),
        window_size(win_size),
        context_fits(ctx_size <= win_size),
        tokens_truncated(std::max(0L, ctx_size - win_size))
    {
    }
};

template <typename TASK_TYPE, typename PAIR_TYPE>
long compute_context_size(const TASK_TYPE& task, const PAIR_TYPE& test_pair)
{
    auto input_context = arc_agi_manager::tokenize_input_context(task, test_pair);
    return input_context.size();
}

/*!
    ensures
        - Generates the output grid for a given ARC-AGI test pair
        - Uses autoregressive token generation with smart early stopping
        - Stops generation when:
          * TOKEN_END_OF_OUTPUT is generated
          * Invalid pattern detected (inconsistent row lengths)
          * Maximum grid size exceeded (30×30)
          * Maximum token limit reached (2048 tokens)
        - Throws std::runtime_error if generation produces invalid grid
        - Returns the predicted output grid if successful
!*/
template <typename NET_TYPE>
generation_result generate_output_for_test_pair_with_info(
    NET_TYPE& net,
    const arc_task& task,
    const arc_task_pair& test_pair,
    bool verbose = false)
{
    // Tokenize the full input context (all training demonstrations + test input)
    auto input_context = arc_agi_manager::tokenize_input_context(task, test_pair);
    const long actual_context_size = input_context.size();
    const long inference_window_size = actual_context_size;

    generation_result result(actual_context_size, WINDOW_LEN);
    result.context_fits = true;
    result.tokens_truncated = 0;

    if (verbose) {
        cout << "  Input context: " << actual_context_size << " tokens\n";
        cout << "  Training window size: " << WINDOW_LEN << " tokens\n";
        cout << "  Inference window size: " << actual_context_size << " tokens\n";
        if (actual_context_size > WINDOW_LEN) {
            cout << "  Extended context: YES (+" << (actual_context_size - WINDOW_LEN)
                << " tokens beyond training window)\n";
        }
        else {
            cout << "  Context fits in training window: YES (margin: "
                << (WINDOW_LEN - actual_context_size) << " tokens)\n";
        }
    }

    // Important: the hrm_ layer re-initializes its recurrence states (z_H, z_L)
    // from learned init vectors at every forward call and runs N*T internal
    // sub-net forwards over the *current* sequence length. This makes KV cache
    // semantics (prefill + 1-token incremental) inapplicable: the cached K/V
    // would be associated with stale recurrence states from a prior call, and
    // a single-token incremental forward would re-run the full N*T recurrence
    // over that one position, which cannot reproduce the multi-position
    // recurrence trajectory the model was trained with.
    //
    // For HRM, autoregressive generation must therefore use a full forward at
    // every step. The sliding window keeps the input length bounded; each token
    // generation recomputes attention from scratch. attention_impl::unified is
    // still safe here because we never call set_inference_mode(prefill /
    // incremental) and never call set_kv_cache_capacity(), so gqa_attention_
    // stays on its standard (training-mode) code path.

    // Initialize context window with the actual input context
    std::vector<int> context_window(inference_window_size);
    for (long i = 0; i < inference_window_size; ++i)
        context_window[i] = input_context(i);

    network_context::clear_padding();

    std::vector<int> generated_tokens;
    generation_state state;
    long generated_count = 0;

    while (generated_count < MAX_OUTPUT_TOKENS && !signal_handler::is_triggered())
    {
        const long current_window_size = static_cast<long>(context_window.size());
        arc_token_sequence_t input_seq(current_window_size);
        for (long i = 0; i < current_window_size; ++i)
            input_seq(i) = context_window[i];

        const int next_token = static_cast<int>(net(input_seq));

        if (next_token == TOKEN_END_OF_OUTPUT) {
            if (verbose) cout << "  Stopping: TOKEN_END_OF_OUTPUT generated\n";
            break;
        }

        generated_tokens.push_back(next_token);
        generated_count++;
        state.add_token(next_token);

        if (state.should_stop()) {
            if (verbose) {
                cout << "  Early stopping: invalid generation detected\n";
                cout << "    Rows generated: " << state.num_rows() << "\n";
            }
            if (!state.is_valid && state.num_rows() < 2)
                throw std::runtime_error("generation failed, invalid grid structure");
            break;
        }
        if (state.num_rows() >= MAX_ROWS) {
            if (verbose) cout << "  Stopping: maximum rows reached\n";
            break;
        }

        // Slide window: drop oldest, append newest
        for (long i = 0; i < current_window_size - 1; ++i)
            context_window[i] = context_window[i + 1];
        context_window[current_window_size - 1] = next_token;
    }

    if (verbose) {
        cout << "  Generated " << generated_count << " tokens\n";
        cout << "  Detected " << state.num_rows() << " complete rows\n";
    }

    if (!validate_token_sequence(generated_tokens, verbose))
        throw std::runtime_error("Invalid token sequence");

    arc_token_sequence_t output_seq(generated_tokens.size());
    for (size_t i = 0; i < generated_tokens.size(); ++i)
        output_seq(i) = generated_tokens[i];

    result.grid = arc_agi_manager::detokenize_to_grid(output_seq, 0);
    return result;
}

int main(int argc, char** argv)
{
    try
    {
        // Setup interrupt handling for clean termination
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("train", "Train transformer model on ARC-AGI tasks");
        parser.add_option("eval", "Evaluate model on test pairs");
        parser.add_option("training-path", "Path to training JSON files", 1);
        parser.add_option("eval-path", "Path to evaluation JSON files", 1);
        parser.add_option("model-file", "Path for model file", 1);
        parser.add_option("learning-rate", "Learning rate (default: 2e-4)", 1);
        parser.add_option("batch-size", "Mini-batch size (default: 8)", 1);
        parser.add_option("max-epochs", "Maximum training epochs (default: 500)", 1);
        parser.add_option("patience", "Early stopping patience (default: 10000)", 1);
        parser.add_option("weight-decay", "Set the weight decay for AdamW (default: 0.004)", 1);
        parser.add_option("beta1", "Set AdamW's beta1 coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set AdamW's beta2 coefficient (default: 0.997)", 1);
        parser.add_option("task-id", "Specific task ID to evaluate", 1);
        parser.add_option("verbose", "Show detailed output during evaluation");
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("train") &&
            !parser.option("eval"))
        {
            parser.print_options();
            cout << "\nExample usage:\n"
                << "  Training:\t" << argv[0] << " --train --training-path data/training --eval-path data/evaluation\n"
                << "  Evaluation:\t" << argv[0] << " --eval --eval-path data/evaluation\n"
                << "  Single task:\t" << argv[0] << " --eval --task-id 007bbfb7 --verbose\n";
            return 0;
        }

        // Configuration
        const std::string training_path = get_option(parser, "training-path", "data/training");
        const std::string eval_path = get_option(parser, "eval-path", "data/evaluation");
        const std::string model_file = get_option(parser, "model-file", "dlib_lm_arc_agi_model.dat");
        const double learning_rate = get_option(parser, "learning-rate", 2e-4);
        const size_t batch_size = get_option(parser, "batch-size", 8);
        const size_t max_epochs = get_option(parser, "max-epochs", 500);
        const long patience = get_option(parser, "patience", 10000);
        const double weight_decay = get_option(parser, "weight-decay", 0.004);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.997);

        // Model configuration tuned for a 10 GB GPU and ARC-AGI reasoning.
        // HRM splits its layer budget between a thin H module (slow loop,
        // 1 layer) and a deeper L module (fast loop, 2 layers); the recurrence
        // factors (hrm_N=1, hrm_T=2) match the canonical HRM setup. Attention
        // uses the unified gqa_attention_ layer with 2 KV heads (GQA, 3x KV
        // memory reduction vs MHA); embedding_dim=192 yields head_dim=32 with
        // num_heads=6.
        //
        // KV cache caveat: the hrm_ layer re-initializes its recurrence state
        // (z_H, z_L) at every forward call and runs N*T internal sub-net
        // forwards over the current sequence. KV cache semantics (prefill +
        // 1-token incremental) are therefore not applicable to HRM, even with
        // unified attention. The inference path below always does a full
        // sliding-window forward and never calls set_inference_mode(prefill)
        // or set_kv_cache_capacity(); attention_impl::unified is still safe
        // since gqa_attention_ stays on its standard (training-mode) code path.
        using my_transformer = hrm_transformer_config<
            ARC_VOCAB_SIZE_TOTAL,        // vocab_size
            1,                           // num_h_layers
            2,                           // num_l_layers
            6,                           // num_heads
            192,                         // embedding_dim
            1,                           // hrm_N (outer recurrence)
            2,                           // hrm_T (inner recurrence)
            gelu,                        // activation
            dropout_10,                  // dropout policy
            attention_impl::unified,     // gqa_attention_ (cache never engaged for HRM)
            2>;                          // num_kv_heads (GQA)
        cout << my_transformer::model_info::describe() << "\n\n";

        // Load ARC-AGI data
        arc_agi_manager data_mgr;
        data_mgr.load_data(training_path, eval_path);

        // Check size of the tokenized contexts
        long max_L_in = 0, max_L_full = 0;
        for (size_t i = 0; i < data_mgr.num_training_tasks(); ++i)
        {
            const auto& task = data_mgr.get_training_task(i);
            for (size_t h = 0; h < task.train_pairs.size(); ++h)
            {
                arc_task synthetic;
                synthetic.task_id = task.task_id;
                for (size_t j = 0; j < task.train_pairs.size(); ++j)
                    if (j != h) synthetic.train_pairs.push_back(task.train_pairs[j]);

                auto ctx = arc_agi_manager::tokenize_input_context(
                    synthetic, task.train_pairs[h]);
                auto out = arc_agi_manager::tokenize_target_output(
                    task.train_pairs[h]);

                max_L_in = std::max(max_L_in, ctx.size());
                max_L_full = std::max(max_L_full, ctx.size() + out.size());
            }
        }
        cout << "Dataset context size analysis:\n";
        cout << "  max L_in   = " << max_L_in
            << " (minimum recommended WINDOW_LEN)\n";
        cout << "  max L_full = " << max_L_full
            << " (context + longest output)\n";
        if (max_L_in > WINDOW_LEN)
            cout << "  WARNING: WINDOW_LEN=" << WINDOW_LEN
            << " is too small! Many samples will have truncated context.\n";
        else
            cout << "  OK: WINDOW_LEN=" << WINDOW_LEN << " covers all contexts.\n";

        // Training mode
        if (parser.option("train"))
        {
            cout << "=== TRAINING MODE ===\n";

            if (data_mgr.num_training_tasks() == 0) {
                cerr << "Error: No training tasks loaded\n";
                return 1;
            }

            // Prepare training data from all tasks
            cout << "Preparing training data...\n";
            std::vector<arc_token_sequence_t> all_X;
            std::vector<unsigned long> all_Y;

            for (size_t task_idx = 0; task_idx < data_mgr.num_training_tasks(); ++task_idx)
            {
                const auto& task = data_mgr.get_training_task(task_idx);

                std::vector<arc_token_sequence_t> task_X;
                std::vector<long> task_Y;

                // Held-out few-shot strategy: each train_pair is treated in turn
                // as the target while the others act as demonstrations. This is
                // the canonical ARC-AGI training scheme; the alternative
                // prepare_training_data_pair_only (no demos) and
                // prepare_training_data_sliding_window (uses test outputs too)
                // are available for warmup or different setups.
                arc_agi_manager::prepare_training_data_batch(task, WINDOW_LEN, task_X, task_Y, false);

                all_X.insert(all_X.end(), task_X.begin(), task_X.end());

                // Convert long to unsigned long for dlib classification
                for (auto y : task_Y)
                    all_Y.push_back(static_cast<unsigned long>(y));

                if ((task_idx + 1) % 10 == 0) {
                    cout << "Processed " << (task_idx + 1) << "/"
                        << data_mgr.num_training_tasks() << " tasks...\r" << flush;
                }
            }
            cout << "\nTotal training samples: " << all_X.size() << endl;
            //size_t removed = filter_training_samples(all_X, all_Y);
            //cout << "\nTotal training samples: " << all_X.size()
            //    << " (removed samples: " << removed << ")" ccc

            // Build network
            using net_type = my_transformer::network_type<true>;
            net_type net;
            if (file_exists(model_file) && !file_exists("chkpt-" + model_file)) {
                cout << "Loading existing model from " << model_file << endl;
                deserialize(model_file) >> net;
            }
            layer<0>(net).loss_details().set_ignore_index(TOKEN_PADDING);
            // Disable label smoothing during initial training. With the
            // expected uniform-softmax baseline at init (~log(17) = 2.83 for
            // a 17-token vocabulary), a clean cross-entropy loss is easier
            // to monitor than the smoothed variant whose interactions with
            // the floor in the loss kernel can mask early convergence
            // issues. Re-enable (set to 0.1) once convergence is solid.
            layer<0>(net).loss_details().set_label_smoothing(0.0);
            /* Enable YaRN before the first forward so the serialized model carries the
               extended-context RoPE configuration, whatever the library default. */
            enable_yarn_context_extension(net);
            network_context::set_optimizer_params(weight_decay, beta1, beta2);
            cout << net << endl << endl; // Show the model architecture

            arc_token_sequence_t dummy(WINDOW_LEN);
            for (long i = 0; i < WINDOW_LEN; ++i) dummy(i) = TOKEN_PADDING;
            net(dummy);
            cout << "Number of model parameters: " << count_parameters(net) << "\n";

            // Setup trainer
            std::vector<int> gpus{ 0 };
            dnn_trainer<net_type, adamw> trainer(net, adamw(weight_decay, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-8);
            trainer.set_learning_rate_shrink_factor(0.1);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_synchronization_file("chkpt-" + model_file, std::chrono::minutes(10));
            trainer.be_quiet();

            // Training loop
            cout << "Starting training...\n";
            size_t epoch = 0;
            auto start_time = std::chrono::steady_clock::now();

            size_t batches_count = 0;
            while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
                && epoch < max_epochs && !signal_handler::is_triggered())
            {
                // Shuffle the dataset
                shuffle_training_dataset(all_X, all_Y);

                // Train epoch
                size_t batches_seen = 0;
                for (size_t i = 0; i < all_X.size() && !signal_handler::is_triggered(); i += batch_size)
                {
                    std::vector<matrix<int, 0, 1>> batch_X;
                    std::vector<unsigned long> batch_Y;
                    batch_X.reserve(batch_size);
                    batch_Y.reserve(batch_size);

                    for (size_t j = 0; j < batch_size && (i + j) < all_X.size(); ++j) {
                        size_t idx = i + j;
                        batch_X.push_back(all_X[idx]);
                        batch_Y.push_back(all_Y[idx]);
                    }

                    // Synchronize: ensure trainer has finished processing previous batch
                    // before modifying the shared network_context singleton
                    trainer.get_net(force_flush_to_disk::no);

                    network_context::set_learning_rate(trainer.get_learning_rate());
                    std::vector<long> pad_lengths(batch_X.size());
                    for (size_t j = 0; j < batch_X.size(); ++j)
                        pad_lengths[j] = count_leading_padding(batch_X[j], static_cast<int>(TOKEN_PADDING));
                    network_context::set_padding_from_lengths(pad_lengths);

                    trainer.train_one_step(batch_X, batch_Y);
                    batches_seen++;

                    // Progress reporting
                    if (batches_count++ % 50 == 0) {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

                        cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
                            << " (batches: " << batches_seen << ")"
                            << " \t loss: " << trainer.get_average_loss()
                            << " \t patience: " << trainer.get_steps_without_progress()
                            << " \t time: " << elapsed << "s\n";
                        cout.flush();
                    }
                }
                epoch++;
            }

            // Save model
            trainer.get_net();
            network_context::reset();
            net.clean();
            serialize(model_file) << net;
            cout << "Model saved to " << model_file << "\n";
        }

        // Evaluation mode (with generation)
        if (parser.option("eval"))
        {
            cout << "=== EVALUATION MODE ===\n";

            // Load model
            my_transformer::network_type<false> net;
            if (!file_exists(model_file)) {
                cerr << "Error: Model file not found: " << model_file << "\n";
                return 1;
            }

            deserialize(model_file) >> net;
            /* Re-apply the YaRN configuration after loading: this overrides whatever
               state the model file carries, so models serialized before or after a
               library-default change behave identically in extended-context runs. */
            enable_yarn_context_extension(net);
            cout << "Model loaded.\n";
            cout << "Number of model parameters: " << count_parameters(net) << "\n";

            const bool verbose = parser.option("verbose");
            const bool single_task = parser.option("task-id");

            // Statistics
            size_t total_tasks = 0;
            size_t tasks_with_correct_dims = 0;
            size_t tasks_fully_correct = 0;
            size_t generation_failures = 0;
            double total_pixel_accuracy = 0.0;

            // Get task list to evaluate
            std::vector<const arc_task*> tasks_to_eval;

            if (single_task) {
                std::string task_id = parser.option("task-id").argument();
                cout << "Evaluating single task: " << task_id << "\n\n";

                try {
                    tasks_to_eval.push_back(&data_mgr.get_evaluation_task_by_id(task_id));
                }
                catch (...) {
                    try {
                        tasks_to_eval.push_back(&data_mgr.get_training_task_by_id(task_id));
                    }
                    catch (...) {
                        cerr << "Error: Task not found: " << task_id << "\n";
                        return 1;
                    }
                }
            }
            else {
                // Evaluate all evaluation tasks
                for (size_t i = 0; i < data_mgr.num_evaluation_tasks(); ++i) {
                    tasks_to_eval.push_back(&data_mgr.get_evaluation_task(i));
                }
            }

            // Evaluate each task. Inference mode and KV cache are managed
            // per-test-pair inside generate_output_for_test_pair_with_info(),
            // so the global network_context is only cleared here as a safety
            // measure before the first call.
            network_context::reset();
            for (const arc_task* task_ptr : tasks_to_eval)
            {
                const arc_task& task = *task_ptr;
                if (task.test_pairs.empty()) {
                    if (verbose) cout << "Task " << task.task_id << ": No test pairs\n";
                    continue;
                }

                cout << "Task " << task.task_id << " (" << task.train_pairs.size()
                    << " train, " << task.test_pairs.size() << " test):\n";

                // Evaluate each test pair
                for (size_t pair_idx = 0; pair_idx < task.test_pairs.size(); ++pair_idx)
                {
                    const auto& test_pair = task.test_pairs[pair_idx];

                    if (verbose)
                        cout << "  Test pair " << (pair_idx + 1) << "/" << task.test_pairs.size() << ":\n";

                    // Calculate context size before attempting generation
                    long context_size = compute_context_size(task, test_pair);

                    // Now attempt generation
                    generation_result gen_result(context_size, WINDOW_LEN);
                    bool generation_failed = false;

                    try {
                        gen_result = generate_output_for_test_pair_with_info<my_transformer::network_type<false>>(
                            net, task, test_pair, verbose);
                    }
                    catch (const std::exception& e) {
                        if (verbose) {
                            cout << "    Generation error: " << e.what() << "\n";
                        }
                        generation_failed = true;
                        generation_failures++;
                    }

                    if (generation_failed) {
                        cout << "    Generation: FAILED\n";
                        cout << "    Dimensions: KO\n";
                        cout << "    Pixel accuracy: 0.0% (0/0)\n";
                        cout << "    Fully correct: NO\n";
                        total_tasks++;
                        continue;
                    }

                    const arc_grid_t& generated = gen_result.grid;

                    // Rest of the evaluation code...
                    bool dims_correct = (generated.nr() == test_pair.output_rows &&
                        generated.nc() == test_pair.output_cols);

                    if (verbose) {
                        cout << "    Expected: " << test_pair.output_rows << "x" << test_pair.output_cols << "\n";
                        cout << "    Generated: " << generated.nr() << "x" << generated.nc() << "\n";
                    }

                    long correct_pixels = 0;
                    long total_pixels = 0;

                    if (dims_correct) {
                        total_pixels = generated.nr() * generated.nc();
                        for (long r = 0; r < generated.nr(); ++r) {
                            for (long c = 0; c < generated.nc(); ++c) {
                                if (generated(r, c) == test_pair.output(r, c)) {
                                    correct_pixels++;
                                }
                            }
                        }
                    }

                    bool fully_correct = (dims_correct && correct_pixels == total_pixels);
                    double pixel_acc = (total_pixels > 0) ?
                        (100.0 * correct_pixels / total_pixels) : 0.0;

                    total_tasks++;
                    if (dims_correct) tasks_with_correct_dims++;
                    if (fully_correct) tasks_fully_correct++;
                    if (total_pixels > 0) total_pixel_accuracy += pixel_acc;

                    cout << "    Dimensions: " << (dims_correct ? "OK" : "KO") << "\n";
                    cout << "    Pixel accuracy: " << std::fixed << std::setprecision(1)
                        << pixel_acc << "% (" << correct_pixels << "/" << total_pixels << ")\n";
                    cout << "    Fully correct: " << (fully_correct ? "YES" : "NO") << "\n";

                    if (verbose || !fully_correct) {
                        cout << "    Generated grid:\n";
                        for (long r = 0; r < std::min(10L, generated.nr()); ++r) {
                            cout << "      ";
                            for (long c = 0; c < std::min(15L, generated.nc()); ++c) {
                                cout << static_cast<int>(generated(r, c)) << " ";
                            }
                            if (generated.nc() > 15) cout << "...";
                            cout << "\n";
                        }
                        if (generated.nr() > 10) cout << "      ...\n";
                    }
                }
                cout << "\n";

                if (signal_handler::is_triggered()) break;
            }
            network_context::reset();

            // Final statistics
            cout << "=== EVALUATION SUMMARY ===\n";
            cout << "Tasks evaluated: " << total_tasks << "\n";
            if (generation_failures > 0) {
                cout << "Generation failures: " << generation_failures << "/" << total_tasks << "\n";
            }
            if (total_tasks > 0) {
                cout << "Correct dimensions: " << tasks_with_correct_dims << "/" << total_tasks
                    << " (" << (100.0 * tasks_with_correct_dims / total_tasks) << "%)\n";
                cout << "Fully correct outputs: " << tasks_fully_correct << "/" << total_tasks
                    << " (" << (100.0 * tasks_fully_correct / total_tasks) << "%)\n";
                cout << "Average pixel accuracy: "
                    << (total_pixel_accuracy / total_tasks) << "%\n";
            }
            else {
                cout << "No tasks evaluated.\n";
            }
        }

        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}