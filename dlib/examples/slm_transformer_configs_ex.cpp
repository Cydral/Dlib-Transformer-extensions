/*!
    @file slm_transformer_configs_ex.cpp
    @brief Training and generation pipeline for the pre-configured transformer architectures built into Dlib.

    This example demonstrates how to train and evaluate two network topologies available as
    ready-to-use configurations:

      --arch moe   Grouped Query Attention + Mixture-of-Experts feed-forward.
                   Sparse top-k expert activation with load-balancing auxiliary loss.
                   Best for exploring conditional computation and expert specialization.

      --arch hrm   Hierarchical Recurrent Model.
                   Two-level recurrent structure (H+L modules) for multi-scale sequence modeling.
                   Suited for tasks requiring both local and global context.

    Both configurations share the same training and generation pipeline: BPE tokenization,
    sliding-window dataset construction, AdamW optimization with reduce-on-plateau learning rate
    control, checkpoint support, and autoregressive text generation with prompt/reference split
    validation.

    The architecture is selected at runtime via --arch; the compiler instantiates the full
    pipeline independently for each configuration so there is no runtime overhead from the
    dispatch. Model files are automatically named per architecture (dlib_lm_<arch>_model.dat)
    to prevent accidental cross-architecture checkpoint loading.

    Usage:
      Training  : slm_transformer_configs_ex --train --arch moe
      Generation: slm_transformer_configs_ex --generate --arch hrm
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
#include <dlib/tokenizer.h>

#include "slm_data.h"

using namespace std;
using namespace dlib;

// Utility functions
std::string read_file_content(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        cerr << "Warning: Cannot open file: " << filepath << "\n";
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string normalize_paragraph_delimiters(const std::string& text)
{
    std::string result;
    result.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (i + 1 < text.size() && text[i] == '\n' && text[i + 1] == '\n') {
            result += "@@";
            i += 2;
            while (i < text.size() && text[i] == '\n') ++i;
        }
        else {
            result += text[i];
            ++i;
        }
    }
    return result;
}

void collect_text_files_recursive(const directory& dir, std::vector<std::string>& text_files, size_t max_files = 0)
{
    for (const auto& file : dir.get_files()) {
        if (max_files > 0 && text_files.size() >= max_files) return;
        file_content_type content_type;
        if (detect_file_type(file.full_name(), content_type))
            text_files.push_back(file.full_name());
    }
    for (const auto& subdir : dir.get_dirs()) {
        if (max_files > 0 && text_files.size() >= max_files) return;
        collect_text_files_recursive(subdir, text_files, max_files);
    }
}

std::string load_external_data(const std::string& path, bool normalize_delimiters = true)
{
    std::string combined_text;

    try {
        directory dir(path);
        cout << "Scanning directory recursively: " << path << "\n";

        std::vector<std::string> text_files;
        collect_text_files_recursive(dir, text_files);
        cout << "Found " << text_files.size() << " text file(s)\n";

        if (text_files.empty()) {
            cerr << "Warning: No text files found in directory\n";
            return "";
        }

        std::sort(text_files.begin(), text_files.end());
        size_t total_bytes = 0;
        for (const auto& filepath : text_files) {
            std::string content = read_file_content(filepath);
            if (!content.empty()) {
                combined_text += content;
                if (combined_text.size() >= 2 && combined_text.substr(combined_text.size() - 2) != "@@")
                    combined_text += "@@";
                total_bytes += content.size();
            }
        }
        cout << "Total loaded: " << total_bytes << " bytes from " << text_files.size() << " file(s)\n";
    }
    catch (const directory::dir_not_found&) {
        cout << "Loading single text file: " << path << "\n";
        file_content_type content_type;
        if (!detect_file_type(path, content_type)) {
            cerr << "Error: File does not appear to be text: " << path << "\n";
            return "";
        }
        combined_text = read_file_content(path);
        if (combined_text.empty()) {
            cerr << "Warning: File is empty or could not be read\n";
            return "";
        }
        cout << "Loaded " << combined_text.size() << " bytes from file\n";
    }
    catch (const std::exception& e) {
        cerr << "Error loading external data: " << e.what() << "\n";
        return "";
    }

    if (normalize_delimiters && !combined_text.empty())
        combined_text = normalize_paragraph_delimiters(combined_text);

    return combined_text;
}

std::vector<std::string> parse_delimited_segments(const std::string& text)
{
    std::vector<std::string> segments;
    const std::string delimiter = "@@";
    size_t start = 0;
    size_t end = text.find(delimiter);

    while (end != std::string::npos) {
        std::string segment = text.substr(start, end - start);
        size_t first = segment.find_first_not_of(" \t\n\r");
        if (first != std::string::npos) {
            size_t last = segment.find_last_not_of(" \t\n\r");
            segment = segment.substr(first, last - first + 1);
            if (!segment.empty()) segments.push_back(segment);
        }
        start = end + delimiter.length();
        end = text.find(delimiter, start);
    }

    if (start < text.size()) {
        std::string segment = text.substr(start);
        size_t first = segment.find_first_not_of(" \t\n\r");
        if (first != std::string::npos) {
            size_t last = segment.find_last_not_of(" \t\n\r");
            segment = segment.substr(first, last - first + 1);
            if (!segment.empty()) segments.push_back(segment);
        }
    }

    return segments;
}

// MoE parameter and expert usage analysis
struct moe_param_info
{
    size_t single_expert_params;
    size_t total_params;
    size_t inference_params;
    long   num_experts;
    long   num_moe_layers;
    long   top_k;
    float  efficiency_ratio;
    std::vector<float> expert_usage;

    void print() const
    {
        cout << "=== MoE network parameter analysis ===\n"
            << "Architecture:\n"
            << "  MoE layers          : " << num_moe_layers << "\n"
            << "  Experts per layer   : " << num_experts << "\n"
            << "  Active experts (k)  : " << top_k << "\n\n"
            << "Parameters per expert : " << single_expert_params << "\n"
            << "Total (training)      : " << total_params << "\n"
            << "Active (inference)    : " << inference_params << "\n\n"
            << "Efficiency:\n"
            << "  Inference uses " << (efficiency_ratio * 100.0f) << "% of training params\n"
            << "  Savings: " << ((1.0f - efficiency_ratio) * 100.0f) << "% fewer active params\n\n";

        if (!expert_usage.empty()) {
            cout << "Expert usage statistics (EMA):\n";

            float total_usage = 0.0f, min_usage = expert_usage[0], max_usage = expert_usage[0];
            for (float u : expert_usage) {
                total_usage += u;
                min_usage = std::min(min_usage, u);
                max_usage = std::max(max_usage, u);
            }

            float mean_usage = total_usage / num_experts;
            float ideal_usage = 1.0f / num_experts;
            float variance = 0.0f;
            for (float u : expert_usage) { float diff = u - mean_usage; variance += diff * diff; }
            variance /= num_experts;
            float std_dev = std::sqrt(variance);
            float cv = (mean_usage > 1e-8f) ? (std_dev / mean_usage) : 0.0f;

            cout << "  Mean usage : " << std::fixed << std::setprecision(4) << mean_usage
                << " (ideal: " << ideal_usage << ")\n"
                << "  Range      : [" << min_usage << ", " << max_usage << "]\n"
                << "  Std dev    : " << std_dev << "\n"
                << "  CV         : " << cv << "\n"
                << "  Balance    : ";

            if (cv < 0.3f)      cout << "excellent (CV < 0.3)\n";
            else if (cv < 0.5f) cout << "good (CV < 0.5)\n";
            else if (cv < 0.8f) cout << "fair (CV < 0.8)\n";
            else                cout << "poor (CV >= 0.8) -- possible expert collapse\n";

            cout << "\n  Per-expert usage:\n";
            for (long e = 0; e < num_experts; ++e) {
                cout << "    expert " << e << ": " << std::fixed << std::setprecision(4) << expert_usage[e];
                int bar_len = (max_usage > 0) ? static_cast<int>(expert_usage[e] * 20.0f / max_usage) : 0;
                cout << " [";
                for (int i = 0; i < bar_len; ++i) cout << "=";
                for (int i = bar_len; i < 20; ++i) cout << " ";
                cout << "]";
                float ratio = expert_usage[e] / ideal_usage;
                if (ratio < 0.5f) cout << " (underutilized)";
                else if (ratio > 2.0f) cout << " (overutilized)";
                cout << "\n";
            }
        }
        else {
            cout << "Expert usage statistics: not available (inference mode or no training yet)\n";
        }
        cout << "\n";
    }
};

// Retrieve expert statistics from the first MoE layer.
// Layer index in our topology: loss=0, linear=1, rms_norm=2, add_prev=3, moe=4.
template <typename net_type>
moe_param_info get_moe_param_info(const net_type& net, long num_layers)
{
    moe_param_info info;
    const auto& moe_layer = layer<4>(net).layer_details();

    info.num_experts = moe_layer.num_experts();
    info.top_k = moe_layer.num_active_experts();
    info.num_moe_layers = num_layers;
    info.single_expert_params = (info.num_experts > 0) ? count_parameters(moe_layer.get_expert(0)) : 0;
    info.total_params = count_parameters(net);

    size_t inactive_per_layer = (info.num_experts - info.top_k) * info.single_expert_params;
    info.inference_params = info.total_params - static_cast<size_t>(num_layers) * inactive_per_layer;
    info.efficiency_ratio = (info.total_params > 0)
        ? static_cast<float>(info.inference_params) / static_cast<float>(info.total_params) : 1.0f;
    info.expert_usage = moe_layer.get_expert_usage();

    return info;
}

// SFINAE helper: prints MoE analysis when the network has a MoE layer at position 4,
// and silently does nothing for other architectures (e.g. HRM).
template <typename net_type>
auto try_print_moe_info(const net_type& net, long num_layers)
-> decltype(layer<4>(net).layer_details().num_experts(), void())
{
    auto info = get_moe_param_info(net, num_layers);
    if (info.num_experts > 0) info.print();
}

inline void try_print_moe_info(...) {}

// Type-dependent training and generation logic, called once from main() with the concrete
// transformer config selected by --arch.  Each call site in main() compiles a completely
// independent version of this function with its own concrete network types.
template <typename TRANSFORMER_CONFIG>
int run_pipeline(
    bool do_train, bool do_generate,
    const double learning_rate, const size_t batch_size, const long patience, const size_t max_epochs,
    const double weight_decay, const double beta1, const double beta2,
    const std::string& model_file, const std::string& tokenizer_file,
    std::vector<std::string>& text_segments, std::string& external_corpus_for_tokenizer,
    bpe_tokenizer& tokenizer, std::vector<int>& gpus)
{
    using my_transformer = TRANSFORMER_CONFIG;

    constexpr long num_tokens = 2000;
    constexpr long num_layers = 4;
    constexpr long max_seq_len = 100;

    const std::string tokens_file = "dlib_configs_datasets_tokens.bin";
    cout << my_transformer::model_info::describe() << "\n";

    // TRAINING MODE
    if (do_train)
    {
        cout << "TRAINING MODE\n\n";

        // Try to load pre-tokenized tokens
        std::vector<std::vector<int>> full_tokens;
        bool tokens_loaded = false;

        if (file_exists(tokens_file)) {
            cout << "Found pre-tokenized tokens file: " << tokens_file << "\n";
            try {
                dlib::deserialize(tokens_file) >> full_tokens;
                size_t total_tokens = 0;
                for (const auto& seg : full_tokens) total_tokens += seg.size();
                cout << "Loaded " << full_tokens.size() << " segments (" << total_tokens << " tokens)\n";
                tokens_loaded = true;
            }
            catch (const std::exception& e) {
                cerr << "Failed to load tokens: " << e.what() << "\nWill tokenize again.\n";
                full_tokens.clear();
            }
        }

        if (!tokens_loaded) {
            // Train a new tokenizer if needed
            if (!file_exists(tokenizer_file)) {
                cout << "Training new BPE tokenizer (vocab size " << num_tokens << ")...\n";

                const std::string delimiter = "@@";
                std::string tokenizer_corpus =
                    get_dataset_as_text(dataset_id::BLACK_HOLE_ARTICLE) + delimiter
                    + get_dataset_as_text(dataset_id::PHYSICS_PARAGRAPHS) + delimiter
                    + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTA) + delimiter
                    + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTB) + delimiter
                    + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTC) + delimiter
                    + get_dataset_as_text(dataset_id::GENERAL_KNOWLEDGE);

                if (!external_corpus_for_tokenizer.empty())
                    tokenizer_corpus += delimiter + external_corpus_for_tokenizer;

                size_t pos = 0;
                while ((pos = tokenizer_corpus.find(delimiter, pos)) != std::string::npos) {
                    tokenizer_corpus.replace(pos, delimiter.length(), " ");
                    pos += 1;
                }

                tokenizer.train(tokenizer_corpus, num_tokens, 1e6, true);
                serialize(tokenizer_file) << tokenizer;
                cout << "Tokenizer saved to " << tokenizer_file << "\n";
            }

            // Validate required special tokens
            long text_start_id = tokenizer.get_special_token_id("<text>");
            long text_end_id = tokenizer.get_special_token_id("</text>");
            if (text_start_id < 0 || text_end_id < 0) {
                cerr << "ERROR: Required special tokens not found in tokenizer!\n"
                    << "The tokenizer must include: <text>, </text>\n";
                return 1;
            }

            // Tokenize all segments with <text>...</text> wrapping
            cout << "Tokenizing " << text_segments.size() << " segments...\n";
            auto t_start = std::chrono::high_resolution_clock::now();
            size_t total_tokens = 0;

            for (const auto& segment : text_segments) {
                std::vector<int> seg_tokens;
                seg_tokens.push_back(static_cast<int>(text_start_id));
                auto encoded = tokenizer.encode(segment);
                seg_tokens.insert(seg_tokens.end(), encoded.begin(), encoded.end());
                seg_tokens.push_back(static_cast<int>(text_end_id));
                total_tokens += seg_tokens.size();
                full_tokens.push_back(std::move(seg_tokens));
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            cout << "Tokenization complete: " << total_tokens << " tokens in "
                << std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count() << "s\n";
            text_segments.clear();

            cout << "Saving tokens to: " << tokens_file << "\n";
            try { serialize(tokens_file) << full_tokens; cout << "Tokens saved.\n"; }
            catch (const std::exception& e) { cerr << "Warning: Failed to save tokens: " << e.what() << "\n"; }
        }

        // Build training sequences via sliding window
        cout << "Preparing training sequences (window=" << max_seq_len << ")...\n";
        std::vector<matrix<int, 0, 1>> samples;
        std::vector<unsigned long> labels;

        const int pad_token = static_cast<int>(tokenizer.get_special_token_id("<pad>"));
        build_single_token_prediction_dataset(full_tokens, max_seq_len, pad_token, false, samples, labels);
        cout << "Created " << samples.size() << " training samples\n";
        full_tokens.clear();

        // Augment dataset with noisy copies for robustness
        augment_training_dataset(samples, labels,
            static_cast<int>(tokenizer.get_special_token_id("<unk>")), pad_token, 0.03);
        cout << "Augmented dataset size: " << samples.size() << "\n";

        // Build and initialize network
        using train_net_type = typename my_transformer::template network_type<true>;
        train_net_type net;
        layer<0>(net).loss_details().set_ignore_index(pad_token);

        if (file_exists(model_file) && !file_exists("chkpt-" + model_file)) {
            cout << "Loading existing model from " << model_file << "\n";
            deserialize(model_file) >> net >> tokenizer;
        }

        // Propagate optimizer hyperparameters to internal sub-network solvers (HRM H/L, MoE experts/gate)
        network_context::set_optimizer_params(weight_decay, beta1, beta2);

        cout << net << "\n\n";
        cout << "Number of model parameters: " << count_parameters(net) << "\n";

        // Create trainer with reduce-on-plateau learning rate control
        dnn_trainer<train_net_type, adamw> trainer(net, adamw(weight_decay, beta1, beta2), gpus);
        trainer.set_learning_rate(learning_rate);
        trainer.set_min_learning_rate(1e-8);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_mini_batch_size(batch_size);
        trainer.set_iterations_without_progress_threshold(patience);
        trainer.set_synchronization_file("chkpt-" + model_file, std::chrono::minutes(10));
        trainer.be_quiet();

        // Training loop
        cout << "Starting training...\n";
        size_t epoch = 0, batches_count = 0;

        while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
            && epoch < max_epochs && !signal_handler::is_triggered())
        {
            double total_loss = 0.0;
            size_t batches_seen = 0, samples_seen = 0;
            auto epoch_start = std::chrono::high_resolution_clock::now();

            shuffle_training_dataset(samples, labels);

            for (size_t i = 0; i < samples.size() && !signal_handler::is_triggered(); i += batch_size)
            {
                size_t batch_end = std::min(i + batch_size, samples.size());
                std::vector<matrix<int, 0, 1>> batch_X(samples.begin() + i, samples.begin() + batch_end);
                std::vector<unsigned long> batch_Y(labels.begin() + i, labels.begin() + batch_end);

                std::vector<long> pad_lengths(batch_X.size());
                for (size_t j = 0; j < batch_X.size(); ++j)
                    pad_lengths[j] = count_leading_padding(batch_X[j], pad_token);

                // Synchronize: ensure previous batch has completed before modifying network_context
                trainer.get_net(force_flush_to_disk::no);

                // Propagate the current learning rate to internal sub-networks
                network_context::set_learning_rate(trainer.get_learning_rate());
                network_context::set_padding_from_lengths(pad_lengths);

                trainer.train_one_step(batch_X, batch_Y);

                total_loss += trainer.get_average_loss();
                batches_seen++;
                samples_seen += batch_X.size();

                if (batches_count++ % 50 == 0) {
                    double avg_loss = total_loss / batches_seen;
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - epoch_start).count();
                    double spd = samples_seen / (elapsed > 0 ? elapsed : 1);

                    std::ios_base::fmtflags old_flags = cout.flags();
                    std::streamsize old_prec = cout.precision();
                    cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
                        << " \t loss: " << std::fixed << std::setprecision(3) << avg_loss
                        << " \t lr: " << std::scientific << std::setprecision(2) << trainer.get_learning_rate()
                        << " \t speed: " << std::fixed << std::setprecision(1) << spd << " samples/sec\n";
                    cout.flush();
                    cout.flags(old_flags);
                    cout.precision(old_prec);
                }
            }
            epoch++;
        }

        // Save model
        trainer.get_net();
        net.clean();
        serialize(model_file) << net << tokenizer;
        cout << "Model saved to " << model_file << "\n";

        // Evaluate accuracy on training set
        if (!signal_handler::is_triggered())
        {
            cout << "Evaluating model accuracy on training set...\n";

            typename my_transformer::template network_type<false> g_infer;
            deserialize(model_file) >> g_infer >> tokenizer;

            std::vector<long> eval_pad_lengths(samples.size());
            for (size_t i = 0; i < samples.size(); ++i)
                eval_pad_lengths[i] = count_leading_padding(samples[i], pad_token);
            network_context::set_padding_from_lengths(eval_pad_lengths);

            auto predicted = g_infer(samples);
            network_context::clear_padding();

            size_t correct = 0;
            for (size_t i = 0; i < labels.size(); ++i)
                if (predicted[i] == labels[i]) correct++;

            double accuracy = static_cast<double>(correct) / labels.size();
            cout << "Training accuracy: " << (accuracy * 100.0) << "%\n";
            if (accuracy < 0.999)
                cout << "WARNING: Accuracy below 99.9% -- model may not fully memorize the corpus.\n";
        }
        network_context::reset();
    }

    // GENERATION MODE
    if (do_generate)
    {
        cout << "GENERATION MODE\n\n";

        typename my_transformer::template network_type<false> net;
        if (!file_exists(model_file)) {
            cerr << "Error: model file not found. Please run --train first.\n";
            return 0;
        }

        deserialize(model_file) >> net >> tokenizer;
        cout << "Loaded model from " << model_file << "\n";
        cout << "Number of model parameters: " << count_parameters(net) << "\n";

        // Display MoE expert usage and balance statistics (no-op for HRM)
        try_print_moe_info(net, num_layers);

        if (tokenizer.get_vocab_size() == 0) {
            cerr << "Error: Tokenizer not loaded.\n";
            return 0;
        }

        const std::string tokens_file_gen = "dlib_configs_datasets_tokens.bin";

        std::vector<std::vector<int>> tokenized_segments;
        if (!file_exists(tokens_file_gen)) {
            cerr << "Error: Tokenized file not found. Please run --train first.\n";
            return 0;
        }

        cout << "Loading tokenized segments from: " << tokens_file_gen << "\n";
        try {
            deserialize(tokens_file_gen) >> tokenized_segments;
            cout << "Loaded " << tokenized_segments.size() << " tokenized segments\n";
        }
        catch (const std::exception& e) {
            cerr << "Error loading tokens: " << e.what() << "\n";
            return 0;
        }

        if (tokenized_segments.empty()) {
            cerr << "Error: No segments found in tokens file.\n";
            return 0;
        }

        // Select a random segment with at least 2 tokens for splitting
        dlib::rand rng(std::chrono::system_clock::now().time_since_epoch().count());

        std::vector<size_t> valid_indices;
        for (size_t i = 0; i < tokenized_segments.size(); ++i)
            if (tokenized_segments[i].size() >= 2) valid_indices.push_back(i);

        if (valid_indices.empty()) {
            cerr << "Error: No segments with at least 2 tokens.\n";
            return 1;
        }

        size_t pick = rng.get_random_32bit_number() % valid_indices.size();
        size_t segment_idx = valid_indices[pick];
        cout << "Randomly selected segment #" << segment_idx << " (out of "
            << tokenized_segments.size() << ", " << valid_indices.size() << " valid)\n";

        const auto& selected_segment = tokenized_segments[segment_idx];
        const size_t seg_len = selected_segment.size();
        const int pad_token = static_cast<int>(tokenizer.get_special_token_id("<pad>"));

        // Prompt / reference split
        //
        // Case A: segment fits within the context window (seg_len < max_seq_len)
        //   Split 50/50: first half is the prompt (left-padded), second half is ground-truth reference.
        //
        // Case B: segment >= max_seq_len
        //   First max_seq_len tokens become the prompt, remaining tokens are the reference.

        std::vector<int> input_half;
        std::vector<int> verify_half;

        if (seg_len < static_cast<size_t>(max_seq_len)) {
            size_t input_count = (seg_len + 1) / 2;
            size_t verify_count = seg_len - input_count;
            if (verify_count == 0) {
                cerr << "Error: Segment too short for verification split (" << seg_len << " token(s)).\n";
                return 1;
            }
            input_half.assign(selected_segment.begin(), selected_segment.begin() + input_count);
            verify_half.assign(selected_segment.begin() + input_count, selected_segment.end());
            cout << "Short segment (" << seg_len << " tokens < window " << max_seq_len << "): "
                << "50/50 split - input=" << input_count << " tokens (left-padded), verify="
                << verify_count << " tokens\n";
        }
        else {
            input_half.assign(selected_segment.begin(), selected_segment.begin() + max_seq_len);
            verify_half.assign(selected_segment.begin() + max_seq_len, selected_segment.end());
            cout << "Full segment (" << seg_len << " tokens): first " << max_seq_len
                << " as input, " << verify_half.size() << " for verification\n";
        }

        // Build inference window with left-padding
        inference_context llm_context(max_seq_len, 4, pad_token);
        llm_context.add_tokens(input_half);
        auto input_seq = llm_context.get_input_window();

        cout << "\n--- Prompt (input to model) ---\n"
            << tokenizer.decode(input_half, false) << "\n"
            << "-------------------------------\n\n";

        cout << "Starting autoregressive generation...\n";
        const size_t tokens_to_generate = verify_half.size();
        std::vector<int> generated_tokens;
        generated_tokens.reserve(tokens_to_generate);

        const int end_of_text_id = static_cast<int>(tokenizer.get_special_token_id("</text>"));
        auto gen_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < tokens_to_generate && !signal_handler::is_triggered(); ++i)
        {
            long pad_len = count_leading_padding(input_seq, pad_token);
            network_context::set_padding_uniform(pad_len, 1);
            int next_token = net(input_seq);
            generated_tokens.push_back(next_token);

            llm_context.add_token(next_token);
            input_seq = llm_context.get_input_window();

            if ((i + 1) % 50 == 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - gen_start).count();
                double tps = (i + 1) / (elapsed > 0 ? elapsed : 1);
                cout << "Generated " << (i + 1) << "/" << tokens_to_generate
                    << " tokens - " << tps << " tokens/sec\r";
                cout.flush();
            }
            if (next_token == end_of_text_id) break;
        }
        network_context::clear_padding();

        auto gen_end = std::chrono::high_resolution_clock::now();
        long gen_time = std::chrono::duration_cast<std::chrono::seconds>(gen_end - gen_start).count();
        cout << "\nGeneration complete in " << gen_time << " seconds ("
            << generated_tokens.size() << " tokens generated)\n";

        cout << "\n--- Generated text ---\n" << tokenizer.decode(generated_tokens, false) << "\n"
            << "----------------------\n\n";
        cout << "--- Reference text ---\n" << tokenizer.decode(verify_half, false) << "\n"
            << "----------------------\n\n";

        // Validate generated tokens against the reference
        cout << "=== Validation: generated vs. reference ===\n";

        size_t compare_len = std::min(verify_half.size(), generated_tokens.size());
        std::vector<int> ref_subset(verify_half.begin(), verify_half.begin() + compare_len);
        std::vector<int> gen_subset(generated_tokens.begin(), generated_tokens.begin() + compare_len);

        cout << "Comparing " << compare_len << " tokens\n";
        auto similarity = compute_text_similarity(ref_subset, gen_subset);
        similarity.print();

        if (similarity.edit_similarity < 0.95) {
            cout << "Sample comparison (first 100 tokens):\n";
            size_t sample_len = std::min(size_t(100), compare_len);
            size_t diff_count = 0;
            for (size_t i = 0; i < sample_len; ++i) {
                if (ref_subset[i] != gen_subset[i]) {
                    if (diff_count < 10) {
                        cout << "  Position " << i << ": '"
                            << tokenizer.decode({ ref_subset[i] }, false) << "' -> '"
                            << tokenizer.decode({ gen_subset[i] }, false) << "'\n";
                    }
                    diff_count++;
                }
            }
            cout << "Total differences in sample: " << diff_count << "/" << sample_len << "\n";
        }
        else {
            cout << "Excellent match! Generated text closely follows the original.\n";
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("train", "Train a transformer on internal datasets");
        parser.add_option("generate", "Generate text from a previously trained model");
        parser.add_option("arch", "Network architecture: moe | hrm (default: moe)", 1);
        parser.add_option("learning-rate", "Set the learning rate (default: 2e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size (default: 64)", 1);
        parser.add_option("patience", "Steps without progress before LR reduction (default: 8000)", 1);
        parser.add_option("max-epochs", "Maximum number of training epochs (default: 100)", 1);
        parser.add_option("weight-decay", "AdamW weight decay (default: 0.004)", 1);
        parser.add_option("beta1", "AdamW beta1 coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "AdamW beta2 coefficient (default: 0.997)", 1);
        parser.add_option("model-file", "Override model file path (default: auto-named by arch)", 1);
        parser.add_option("tokenizer-file", "Path for tokenizer (default: dlib_lm_tokenizer.vocab)", 1);
        parser.add_option("external-data", "Path to external text data for training", 1);
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 && !parser.option("train") && !parser.option("generate"))
        {
            parser.print_options();
            cout << "\nExample usage:\n"
                << "  Training  : " << argv[0] << " --train --arch moe\n"
                << "  Generation: " << argv[0] << " --generate --arch moe\n"
                << "  Other     : " << argv[0] << " --train --arch hrm --learning-rate 5e-5\n";
            return 0;
        }

        const std::string arch = get_option(parser, "arch", std::string("moe"));
        if (arch != "moe" && arch != "hrm") {
            cerr << "Error: unknown architecture '" << arch << "'. Valid choices: moe, hrm\n";
            return 1;
        }
        cout << "Architecture: " << arch << "\n";

        const double learning_rate = get_option(parser, "learning-rate", 2e-4);
        const size_t batch_size = get_option(parser, "batch-size", 64);
        const long patience = get_option(parser, "patience", 8000);
        const size_t max_epochs = get_option(parser, "max-epochs", 100);
        const double weight_decay = get_option(parser, "weight-decay", 0.004);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.997);
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", "dlib_lm_tokenizer.vocab");

        const std::string model_file = parser.option("model-file")
            ? parser.option("model-file").argument()
            : "dlib_lm_" + arch + "_model.dat";
        cout << "Model file : " << model_file << "\n\n";

        constexpr long num_tokens = 2000;
        constexpr long num_layers = 4;
        constexpr long num_heads = 6;
        constexpr long num_kv_heads = 2;
        constexpr long embedding_dim = 228;
        constexpr long num_experts = 4;
        constexpr long top_k = 0;

        cout << "Loading internal training datasets...\n";
        std::vector<dataset_id> text_datasets = {
            dataset_id::BLACK_HOLE_ARTICLE, dataset_id::PHYSICS_PARAGRAPHS, dataset_id::GENERAL_KNOWLEDGE
        };
        auto text_segments = get_dataset_as_segments(text_datasets);

        std::string external_corpus_for_tokenizer;
        if (parser.option("external-data")) {
            std::string external_path = parser.option("external-data").argument();
            std::string external_text = load_external_data(external_path, true);

            if (!external_text.empty()) {
                external_corpus_for_tokenizer = external_text;
                cout << "Parsing external data into segments...\n";
                auto external_segments = parse_delimited_segments(external_text);
                cout << "Parsed " << external_segments.size() << " external segments\n";
                if (!external_segments.empty()) {
                    size_t original_count = text_segments.size();
                    text_segments.insert(text_segments.end(), external_segments.begin(), external_segments.end());
                    cout << "Training segments: " << original_count << " (internal) + "
                        << external_segments.size() << " (external) = " << text_segments.size() << " (total)\n";
                }
            }
            else {
                cerr << "Warning: No valid external data loaded, continuing with internal datasets only\n";
            }
        }

        bpe_tokenizer tokenizer;
        if (file_exists(tokenizer_file)) {
            cout << "Loading pre-trained tokenizer from: " << tokenizer_file << "\n";
            deserialize(tokenizer_file) >> tokenizer;
            cout << "Tokenizer loaded - vocabulary size: " << tokenizer.get_vocab_size() << "\n";
        }

        std::vector<int> gpus{ 0 };

        if (arch == "moe") {
            using selected = gqa_moe_transformer_config<
                num_tokens, num_layers, num_heads, num_kv_heads, embedding_dim, num_experts, top_k>;
            return run_pipeline<selected>(
                parser.option("train"), parser.option("generate"),
                learning_rate, batch_size, patience, max_epochs, weight_decay, beta1, beta2,
                model_file, tokenizer_file, text_segments, external_corpus_for_tokenizer, tokenizer, gpus);
        }
        else if (arch == "hrm") {
            using selected = hrm_transformer_config<num_tokens, num_layers, num_layers, num_heads, embedding_dim, 1, 2>;
            return run_pipeline<selected>(
                parser.option("train"), parser.option("generate"),
                learning_rate, batch_size, patience, max_epochs, weight_decay, beta1, beta2,
                model_file, tokenizer_file, text_segments, external_corpus_for_tokenizer, tokenizer, gpus);
        }
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}