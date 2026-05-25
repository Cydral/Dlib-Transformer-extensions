/*!
    @file slm_chatbot_ex.cpp
    @brief Transformer-based chatbot with staged fine-tuning.

    This program demonstrates how to build a specialized chatbot using the
    Transformer Mixture-of-Experts (MoE) architecture configured in Dlib.
    The fine-tuning process specializes the model for conversational Q&A tasks
    using formatted prompt-response pairs with special tags.

    The chatbot is designed to answer questions about black holes and astrophysics,
    demonstrating how proper data formatting can specialize a Small Language Model
    (SLM) for specific domains.

    Configuration is kept strictly aligned with slm_transformer_configs_ex.cpp
    (the pre-training program) via a shared pipeline_constants struct so that
    the fine-tuned model architecture matches the pre-trained one bit for bit.

    Key fine-tuning techniques applied here:

    1. Multi-index loss masking. The cross-entropy loss is configured to ignore
       prompt tokens (question text + role markers) so gradients only flow
       through response tokens. This is implemented via the multi-index
       extension of loss_cross_entropy_per_logit_ added to Dlib.

    2. Selective layer freezing. Pre-trained representations are largely
       preserved by lowering the global learning rate multiplier to 0.3 while
       keeping the final projection (1.0) and final RMSNorm (1.0) at full
       learning rate so the model can adapt its output distribution to the Q&A
       format without forgetting general language structure.

    3. Conservative optimizer hyperparameters aligned with the pre-training
       program (weight_decay=0.004, beta1=0.9, beta2=0.98) and a learning rate
       of 5e-5 (~4x below the pre-training LR).

    Usage modes:
    --fine-tune          Fine-tune the pre-trained model on Q&A pairs
    --prompt             Interactive prompting mode

    Data format for fine-tuning:
    <question><text>What is a black hole?</text>
    <answer><text>A black hole is a region of spacetime...</text>
!*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <atomic>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/tokenizer/bpe_tokenizer.h>
#include <dlib/misc_api.h>

// Include internal datasets (Q&A) and utility functions of the library
#include "slm_data.h"

using namespace std;
using namespace dlib;

// Single source of truth for model configuration. MUST match the values used
// when pre-training the base model in slm_euro_moe_ex.cpp, otherwise
// deserialization will fail or produce silently incorrect results.

struct pipeline_constants
{
    static constexpr long num_tokens = 5850;
    static constexpr long num_layers = 4;
    static constexpr long num_heads = 6;
    static constexpr long num_kv_heads = 2;
    static constexpr long embedding_dim = 228;
    static constexpr long num_experts = 4;
    static constexpr long top_k = 2;
    static constexpr long max_seq_len = 300;
};

// ----------------------------------------------------------------------------------------

void display_random_qa_samples(size_t num_samples = 3)
{
    try {
        auto qa_pairs = get_dataset_as_pairs({ dataset_id::BLACK_HOLE_QA_PARTA });
        if (qa_pairs.empty()) {
            cout << "Warning: No Q&A pairs found.\n";
            return;
        }

        cout << "=== QUESTION EXAMPLES (TRAINING DATA) ===\n";
        cout << "Total Q&A pairs in <part.a> dataset: " << qa_pairs.size() << "\n\n";

        dlib::rand rng(std::time(0));
        std::vector<size_t> indices(qa_pairs.size());
        std::iota(indices.begin(), indices.end(), 0);

        for (size_t i = indices.size() - 1; i > 0; --i) {
            size_t j = rng.get_random_32bit_number() % (i + 1);
            std::swap(indices[i], indices[j]);
        }

        num_samples = std::min(num_samples, qa_pairs.size());
        for (size_t i = 0; i < num_samples; ++i) {
            cout << "Example " << (i + 1) << " - Q: " << qa_pairs[indices[i]].first << "\n";
        }
        cout << "======================================================\n\n";
    }
    catch (const std::exception& e) {
        cerr << "Error loading examples: " << e.what() << "\n";
    }
}

// Helper to append a special token, or its text encoding if not present in the vocabulary.
// Returns the number of tokens appended (always >= 1 for valid input).
size_t append_special_or_text(std::vector<int>& dest, bpe_tokenizer& tokenizer, const std::string& tag)
{
    try {
        int id = tokenizer.get_special_token_id(tag);
        dest.push_back(id);
        return 1;
    }
    catch (const std::runtime_error&) {
        auto enc = tokenizer.encode(tag);
        dest.insert(dest.end(), enc.begin(), enc.end());
        return enc.size();
    }
}

// Build the ignore-index set used during fine-tuning. The model should not
// be penalized for failing to predict prompt tokens (questions and role
// markers) since those are user-provided context, not generated content.
//
// We ignore:
//   - <pad>            : padding positions
//   - <question>       : role marker introducing the question
//   - <answer>         : role marker introducing the response
//   - <text>, </text>  : content boundary markers
//
// We deliberately do NOT ignore the question's content tokens themselves at
// the loss level. Instead, we mask them out during sample construction by
// replacing their labels with <pad> (see build_finetune_samples below).
// This keeps the multi-index ignore set small and architecture-agnostic.
std::vector<long> build_default_ignore_indices(bpe_tokenizer& tokenizer)
{
    std::vector<long> ids;
    auto try_add = [&](const std::string& tag) {
        try { ids.push_back(static_cast<long>(tokenizer.get_special_token_id(tag))); }
        catch (...) { /* tag not in vocab, skip */ }
        };

    try_add("<pad>");
    try_add("<question>");
    try_add("<answer>");
    try_add("<text>");
    try_add("</text>");

    return ids;
}

// Construct fine-tuning samples from Q/A pairs using a sliding window over each
// formatted pair. For each window, the label is the token immediately following
// the window. We mask labels falling inside the prompt portion (everything up
// to and including the <answer><text> opening) by replacing them with pad_token,
// which the loss layer will skip via its ignore mechanism.
//
// This approach combines per-position loss masking with the existing
// build_single_token_prediction_dataset infrastructure without requiring
// changes to that function.
void build_finetune_samples(
    const std::vector<std::pair<std::string, std::string>>& qa_pairs,
    bpe_tokenizer& tokenizer,
    long max_seq_len,
    int pad_token,
    std::vector<matrix<int, 0, 1>>& samples,
    std::vector<unsigned long>& labels)
{
    samples.clear();
    labels.clear();

    int question_tag_id = -1, answer_tag_id = -1, text_open_id = -1, text_close_id = -1;
    try { question_tag_id = tokenizer.get_special_token_id("<question>"); }
    catch (...) {}
    try { answer_tag_id = tokenizer.get_special_token_id("<answer>"); }
    catch (...) {}
    try { text_open_id = tokenizer.get_special_token_id("<text>"); }
    catch (...) {}
    try { text_close_id = tokenizer.get_special_token_id("</text>"); }
    catch (...) {}

    // First pass: tokenize all pairs and record where the response starts
    // (right after the <answer><text> opening). Tokens at indices >= response_start
    // are response tokens whose labels should drive the gradient.
    struct tokenized_pair {
        std::vector<int> tokens;
        size_t response_start;   // index in tokens where the response content begins
    };
    std::vector<tokenized_pair> tokenized;
    tokenized.reserve(qa_pairs.size());

    for (const auto& qa : qa_pairs)
    {
        tokenized_pair tp;
        tp.tokens.reserve(64);

        append_special_or_text(tp.tokens, tokenizer, "<question>");
        append_special_or_text(tp.tokens, tokenizer, "<text>");
        auto q_tokens = tokenizer.encode(qa.first);
        tp.tokens.insert(tp.tokens.end(), q_tokens.begin(), q_tokens.end());
        append_special_or_text(tp.tokens, tokenizer, "</text>");

        append_special_or_text(tp.tokens, tokenizer, "<answer>");
        append_special_or_text(tp.tokens, tokenizer, "<text>");

        // Mark the boundary: response content starts here
        tp.response_start = tp.tokens.size();

        auto a_tokens = tokenizer.encode(qa.second);
        tp.tokens.insert(tp.tokens.end(), a_tokens.begin(), a_tokens.end());
        append_special_or_text(tp.tokens, tokenizer, "</text>");

        tokenized.push_back(std::move(tp));
    }

    // Second pass: sliding window with label masking. For each window ending
    // at position `pos`, the label is tokens[pos+1]. If pos+1 < response_start,
    // this label is a prompt token and should be masked (set to pad_token so
    // the loss layer ignores it).
    size_t total_emitted = 0, total_masked = 0;

    for (const auto& tp : tokenized)
    {
        const long L = static_cast<long>(tp.tokens.size());
        if (L < 2) continue;

        // Generate left-padded windows for the early positions so the model
        // learns to start generating responses from short contexts. We slide
        // the window over [0, L-1) and emit a sample whose label is tokens[pos+1].
        for (long pos = 0; pos < L - 1; ++pos)
        {
            matrix<int, 0, 1> window(max_seq_len, 1);
            const long window_start = pos + 1 - max_seq_len;   // can be negative

            for (long i = 0; i < max_seq_len; ++i)
            {
                const long src_idx = window_start + i;
                window(i) = (src_idx >= 0 && src_idx < L)
                    ? tp.tokens[src_idx]
                    : pad_token;
            }

            const size_t target_idx = static_cast<size_t>(pos + 1);
            int target_token = tp.tokens[target_idx];

            // Mask labels for positions inside the prompt portion. The loss
            // layer's ignore mechanism will skip them, so they contribute
            // neither to the loss nor to the gradient.
            if (target_idx < tp.response_start)
            {
                target_token = pad_token;
                ++total_masked;
            }

            samples.push_back(std::move(window));
            labels.push_back(static_cast<unsigned long>(target_token));
            ++total_emitted;
        }
    }

    cout << "Generated " << total_emitted << " fine-tuning samples ("
        << total_masked << " prompt-positions masked, "
        << (total_emitted - total_masked) << " response-positions kept)\n";
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("fine-tune", "Fine-tune the model on Q&A pairs for the chatbot");
        parser.add_option("prompt", "Interactive mode");
        parser.add_option("learning-rate", "Learning rate (default: 5e-5)", 1);
        parser.add_option("batch-size", "Batch size (default: 64)", 1);
        parser.add_option("max-epochs", "Max number of epochs (default: 500)", 1);
        parser.add_option("patience", "Iterations without progress before LR shrink (default: 8000)", 1);
        parser.add_option("model-file", "Base model path (default: dlib_euro_moe_model.dat)", 1);
        parser.add_option("tokenizer-file", "Tokenizer path (default: dlib_euro_moe_tokenizer.vocab)", 1);
        parser.add_option("temperature", "Sampling temperature (default: 0.8)", 1);
        parser.add_option("top-k", "Top-k filter (default: 50)", 1);
        parser.add_option("top-p", "Nucleus sampling (default: 0.9)", 1);
        parser.add_option("repeat-penalty", "Repetition penalty (default: 1.2)", 1);
        parser.add_option("min-p", "Relative min-p threshold (default: 0.05)", 1);
        parser.add_option("deterministic", "Deterministic mode (strict argmax)");
        parser.parse(argc, argv);

        if (!parser.option("fine-tune") && !parser.option("prompt")) {
            cout << "Transformer-based Chatbot (MoE) - Fine-tuning & Inference\n\n";
            parser.print_options();
            return 0;
        }

        // Hyperparameters aligned with pre-training conventions, except for the
        // learning rate which is dialed down for fine-tuning stability.
        const double learning_rate = get_option(parser, "learning-rate", 5e-5);
        const long batch_size = get_option(parser, "batch-size", 64);
        const long max_epochs = get_option(parser, "max-epochs", 500);
        const long patience = get_option(parser, "patience", 8000);
        const double weight_decay = 0.004;
        const double beta1 = 0.9;
        const double beta2 = 0.98;

        const std::string model_file = get_option(parser, "model-file", std::string("dlib_euro_moe_model.dat"));
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", std::string("dlib_euro_moe_tokenizer.vocab"));

        // Bind the architecture using the shared constants. Any drift between
        // these values and the pre-training program will cause deserialization
        // mismatches at load time, surfacing the issue early.
        using my_transformer = gqa_moe_transformer_config<
            pipeline_constants::num_tokens,
            pipeline_constants::num_layers,
            pipeline_constants::num_heads,
            pipeline_constants::num_kv_heads,
            pipeline_constants::embedding_dim,
            pipeline_constants::num_experts,
            pipeline_constants::top_k>;
        using train_net = my_transformer::network_type<true>;
        using infer_net = my_transformer::network_type<false>;

        cout << my_transformer::model_info::describe() << "\n\n";
        std::vector<int> gpus{ 0 };

        // ============================================================================
        // FINE-TUNING MODE
        // ============================================================================
        if (parser.option("fine-tune"))
        {
            cout << "=== FINE-TUNING MODE ===\n";
            cout << "Goal: Specialize the model (Instruct) for conversational Q&A\n\n";

            std::string finetuned_model = model_file.substr(0, model_file.find_last_of('.')) + "_instruct.dat";
            train_net net;
            bpe_tokenizer tokenizer;

            // Load priority: fine-tuned checkpoint > base model > standalone tokenizer.
            // The tokenizer is always read from the same file as the network when
            // possible to guarantee vocab/model consistency.
            if (file_exists(finetuned_model) && !file_exists("chkpt-" + finetuned_model)) {
                deserialize(finetuned_model) >> net >> tokenizer;
                cout << "Resuming from fine-tuned model: " << finetuned_model << "\n";
            }
            else if (file_exists(model_file)) {
                deserialize(model_file) >> net >> tokenizer;
                cout << "Base model loaded: " << model_file << "\n";
            }
            else if (file_exists(tokenizer_file)) {
                cerr << "Error: Pre-trained model not found (" << model_file << ").\n"
                    << "       Run the pre-training program first.\n";
                return 1;
            }
            else {
                cerr << "Error: Neither pre-trained model nor tokenizer found.\n";
                return 1;
            }

            // Sanity check: tokenizer vocab size must match the model's vocab dimension
            if (tokenizer.get_vocab_size() != static_cast<size_t>(pipeline_constants::num_tokens))
            {
                cerr << "Error: Tokenizer vocab size (" << tokenizer.get_vocab_size()
                    << ") does not match model num_tokens (" << pipeline_constants::num_tokens << ").\n"
                    << "       The base model was likely trained with a different tokenizer.\n";
                return 1;
            }

            // Configure loss masking: pad + role markers are ignored. This is the
            // multi-index variant of the loss, which became available after the
            // recent Dlib extension.
            const std::vector<long> ignore_ids = build_default_ignore_indices(tokenizer);
            int pad_token = -1;
            try { pad_token = tokenizer.get_special_token_id("<pad>"); }
            catch (...) {
                cerr << "Error: <pad> token missing from tokenizer.\n";
                return 1;
            }

            layer<0>(net).loss_details().set_ignore_indices(ignore_ids);
            cout << "Configured loss with " << ignore_ids.size() << " ignored token IDs: [";
            for (size_t i = 0; i < ignore_ids.size(); ++i) {
                if (i > 0) cout << ", ";
                cout << ignore_ids[i];
            }
            cout << "]\n";

            // Apply the relaxed freeze strategy (option B):
            //   - global multiplier 0.3: lets MoE experts and intermediate layers
            //     adapt moderately to the Q/A format
            //   - layer<1> (linear final, projects to vocab): 1.0
            //   - layer<2> (rms_norm final): 1.0
            // This preserves general language structure while allowing the output
            // distribution to specialize.
            cout << "Applying relaxed freeze strategy...\n";
            set_all_learning_rate_multipliers(net, 0.3);
            layer<1>(net).layer_details().set_learning_rate_multiplier(1.0);
            layer<2>(net).layer_details().set_learning_rate_multiplier(1.0);

            // Build training samples from Q/A pairs with prompt masking
            cout << "Loading Q&A datasets...\n";
            auto all_qa_pairs = get_dataset_as_pairs({
                dataset_id::BLACK_HOLE_QA_PARTA,
                dataset_id::BLACK_HOLE_QA_PARTB,
                dataset_id::BLACK_HOLE_QA_PARTC });
            cout << "Loaded " << all_qa_pairs.size() << " Q/A pairs\n";

            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;
            build_finetune_samples(all_qa_pairs, tokenizer, pipeline_constants::max_seq_len, pad_token, samples, labels);

            if (samples.empty()) {
                cerr << "Error: No fine-tuning samples produced.\n";
                return 1;
            }

            // Configure trainer
            dnn_trainer<train_net, adamw> trainer(net, adamw(weight_decay, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-7);
            trainer.set_learning_rate_shrink_factor(0.1);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_synchronization_file("chkpt-" + finetuned_model, std::chrono::minutes(5));
            trainer.be_quiet();

            network_context::set_optimizer_params(weight_decay, beta1, beta2);

            cout << "Starting fine-tuning (lr=" << learning_rate
                << ", batch_size=" << batch_size
                << ", max_epochs=" << max_epochs << ")...\n";

            size_t epoch = 0, batches_count = 0;

            while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
                && epoch < static_cast<size_t>(max_epochs)
                && !signal_handler::is_triggered())
            {
                double total_loss = 0.0;
                size_t batches_seen = 0, samples_seen = 0;
                auto epoch_start = std::chrono::high_resolution_clock::now();

                shuffle_training_dataset(samples, labels);

                for (size_t i = 0; i < samples.size() && !signal_handler::is_triggered(); i += batch_size)
                {
                    size_t batch_end = std::min(i + static_cast<size_t>(batch_size), samples.size());
                    std::vector<matrix<int, 0, 1>> batch_samples(samples.begin() + i, samples.begin() + batch_end);
                    std::vector<unsigned long> batch_labels(labels.begin() + i, labels.begin() + batch_end);

                    std::vector<long> pad_lengths(batch_samples.size());
                    for (size_t j = 0; j < batch_samples.size(); ++j)
                        pad_lengths[j] = count_leading_padding(batch_samples[j], pad_token);

                    trainer.get_net(force_flush_to_disk::no);
                    network_context::set_learning_rate(trainer.get_learning_rate());
                    network_context::set_padding_from_lengths(pad_lengths);

                    trainer.train_one_step(batch_samples, batch_labels);

                    total_loss += trainer.get_average_loss();
                    batches_seen++;
                    samples_seen += batch_samples.size();

                    if (batches_count++ % 25 == 0) {
                        double avg_loss = total_loss / batches_seen;
                        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - epoch_start).count();
                        double samples_per_sec = (elapsed_ms > 0)
                            ? (samples_seen * 1000.0 / elapsed_ms)
                            : 0.0;

                        std::ostringstream line;
                        line << "epoch#: " << std::setw(4) << std::right << (epoch + 1)
                            << "/" << std::left << std::setw(4) << max_epochs
                            << "  loss: " << std::fixed << std::setprecision(4)
                            << std::setw(8) << std::right << avg_loss
                            << "  lr: " << std::scientific << std::setprecision(2)
                            << std::setw(9) << trainer.get_learning_rate()
                            << "  speed: " << std::fixed << std::setprecision(0)
                            << std::setw(6) << std::right << samples_per_sec << " samples/sec";

                        std::string s = line.str();
                        if (s.size() < 90) s.append(90 - s.size(), ' ');
                        cout << "\r" << s << std::flush;
                    }
                }
                epoch++;
            }
            cout << "\n";

            // Restore unit learning rate multipliers before saving so subsequent
            // training (e.g. another fine-tune pass) starts from a clean state.
            trainer.get_net();
            set_all_learning_rate_multipliers(net, 1.0);
            net.clean();
            network_context::reset();

            serialize(finetuned_model) << net << tokenizer;
            cout << "Fine-tuning complete. Model saved to: " << finetuned_model << "\n";
        }

        // ============================================================================
        // PROMPT MODE (INFERENCE)
        // ============================================================================
        else if (parser.option("prompt"))
        {
            cout << "=== INTERACTIVE MODE (PROMPTING) ===\n";
            display_random_qa_samples(3);
            cout << "Type 'quit' or 'exit' to stop.\n\n";

            const size_t top_k = get_option(parser, "top-k", 50);
            const float  top_p = get_option(parser, "top-p", 0.9f);
            const float  repeat_penalty = get_option(parser, "repeat-penalty", 1.2f);
            const float  min_p = get_option(parser, "min-p", 0.05f);
            const bool   deterministic_mode = parser.option("deterministic");
            const float  temperature = deterministic_mode ? 1.0f : get_option(parser, "temperature", 0.8f);
            dlib::rand rng(std::time(0));

            std::string finetuned_model = model_file.substr(0, model_file.find_last_of('.')) + "_instruct.dat";
            bpe_tokenizer tokenizer;
            infer_net net;

            if (!file_exists(finetuned_model)) {
                cerr << "Error: Fine-tuned model not found: " << finetuned_model << "\n"
                    << "Please run the training with --fine-tune first.\n";
                return 1;
            }
            deserialize(finetuned_model) >> net >> tokenizer;
            cout << "Model loaded: " << finetuned_model << "\n\n";

            if (tokenizer.get_vocab_size() != static_cast<size_t>(pipeline_constants::num_tokens))
            {
                cerr << "Error: Tokenizer vocab size mismatch with model.\n";
                return 1;
            }

            softmaxm<multiply<infer_net::subnet_type>> generator(multiply_(1.0 / temperature));
            generator.subnet().subnet() = net.subnet();

            int text_start_id = -1, text_end_id = -1, pad_token = -1;
            try { text_start_id = tokenizer.get_special_token_id("<text>"); }
            catch (...) {}
            try { text_end_id = tokenizer.get_special_token_id("</text>"); }
            catch (...) {}
            try { pad_token = tokenizer.get_special_token_id("<pad>"); }
            catch (...) {}

            // Inference context with capacity 3x the window for multi-turn dialogue.
            inference_context ctx(pipeline_constants::max_seq_len, 3, pad_token);

            // Activate KV cache mode for the whole dialogue. At each user
            // turn we re-prefill the windowed dialogue (cache cleared between
            // turns) and then generate the response incrementally.
            network_context::reset();
            network_context::set_kv_cache_capacity(pipeline_constants::max_seq_len);

            while (!signal_handler::is_triggered())
            {
                cout << "You: ";
                cout.flush();

                std::string user_input;
                if (!std::getline(std::cin, user_input)) break;

                user_input.erase(0, user_input.find_first_not_of(" \t\n\r"));
                user_input.erase(user_input.find_last_not_of(" \t\n\r") + 1);
                if (user_input.empty()) continue;

                if (user_input == "quit" || user_input == "exit") {
                    cout << "Goodbye!\n";
                    break;
                }

                std::vector<int> input_tokens;
                append_special_or_text(input_tokens, tokenizer, "<question>");
                if (text_start_id >= 0) input_tokens.push_back(text_start_id);
                auto q_tokens = tokenizer.encode(user_input);
                input_tokens.insert(input_tokens.end(), q_tokens.begin(), q_tokens.end());
                if (text_end_id >= 0) input_tokens.push_back(text_end_id);

                ctx.add_tokens(input_tokens);

                std::vector<int> answer_prompt;
                append_special_or_text(answer_prompt, tokenizer, "<answer>");
                if (text_start_id >= 0) answer_prompt.push_back(text_start_id);
                ctx.add_tokens(answer_prompt);

                cout << "CHATBOT: ";
                cout.flush();

                auto top_k_p_sample = [&rng, &ctx, text_end_id](
                    const float* probs, size_t N, size_t k, float p, float rp, float mp) -> size_t
                    {
                        std::vector<float> p_copy(probs, probs + N);

                        if (rp > 1.0f) {
                            const auto& context_tokens = ctx.get_full_context();
                            size_t recent_size = std::max(size_t(1), static_cast<size_t>(context_tokens.size() * 0.3));
                            size_t start_idx = (context_tokens.size() > recent_size) ? context_tokens.size() - recent_size : 0;

                            for (size_t i = start_idx; i < context_tokens.size(); ++i) {
                                int t_id = context_tokens[i];
                                if (t_id >= 0 && static_cast<size_t>(t_id) < N) p_copy[t_id] /= rp;
                            }
                            float sum_rp = 0.0f;
                            for (size_t i = 0; i < N; ++i) sum_rp += p_copy[i];
                            if (sum_rp > 1e-8f) for (size_t i = 0; i < N; ++i) p_copy[i] /= sum_rp;
                        }

                        float max_prob = *std::max_element(p_copy.begin(), p_copy.end());
                        float min_p_thresh = max_prob * mp;
                        std::vector<std::pair<size_t, float>> candidates;
                        candidates.reserve(N);
                        for (size_t i = 0; i < N; ++i) {
                            if (p_copy[i] >= min_p_thresh) candidates.push_back({ i, p_copy[i] });
                        }
                        if (candidates.empty()) return text_end_id >= 0 ? static_cast<size_t>(text_end_id) : 0;

                        k = std::min(k, candidates.size());
                        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                            [](const auto& a, const auto& b) { return a.second > b.second; });

                        float cumsum = 0.0f;
                        size_t cutoff = 0;
                        for (size_t i = 0; i < k; ++i) {
                            cumsum += candidates[i].second;
                            cutoff = i;
                            if (cumsum >= p) break;
                        }

                        float final_sum = 0.0f;
                        for (size_t i = 0; i <= cutoff; ++i) final_sum += candidates[i].second;
                        if (final_sum < 1e-8f) return candidates[0].first;

                        float r = rng.get_random_float() * final_sum;
                        float cs = 0.0f;
                        for (size_t i = 0; i <= cutoff; ++i) {
                            cs += candidates[i].second;
                            if (r <= cs) return candidates[i].first;
                        }
                        return candidates[0].first;
                    };

                int next_token = -1;
                const int max_response_tokens = 3 * pipeline_constants::max_seq_len;

                // Helper extracting next-token probabilities from the
                // softmaxm output tensor at the last sequence position.
                auto pick_next_token = [&](const tensor& probs_tensor) -> int
                    {
                        const long seq_len = probs_tensor.nr();
                        const long v_size = probs_tensor.nc();
                        const long last_pos = seq_len - 1;
                        const long offset = tensor_index(probs_tensor, 0, 0, last_pos, 0);
                        const float* probs = probs_tensor.host() + offset;

                        if (deterministic_mode) {
                            const float* max_ptr = std::max_element(probs, probs + v_size);
                            return static_cast<int>(std::distance(probs, max_ptr));
                        }
                        return static_cast<int>(
                            top_k_p_sample(probs, v_size, top_k, top_p, repeat_penalty, min_p));
                    };

                // Prefill the windowed dialogue. The cache from any previous
                // turn is discarded via request_kv_cache_clear(); each
                // gqa_attention_ layer observes the flag and resets its cache
                // on its first forward call of this turn. Padding (leading
                // pad tokens in the windowed view) is stripped from the
                // cache by the attention layer itself.
                {
                    auto input_seq = ctx.get_input_window();
                    long pad_len = count_leading_padding(input_seq, pad_token);

                    network_context::request_kv_cache_clear();
                    network_context::set_padding_uniform(pad_len, 1);
                    network_context::set_inference_mode(network_context::inference_mode::prefill);

                    auto& probs_tensor = generator(input_seq);
                    next_token = pick_next_token(probs_tensor);

                    network_context::clear_kv_cache_request();
                    network_context::clear_padding();
                    network_context::set_inference_mode(network_context::inference_mode::incremental);
                }

                ctx.add_token(next_token);
                if (text_end_id < 0 || next_token != text_end_id) {
                    std::string token_text = tokenizer.decode(next_token, false);
                    cout << token_text;
                    cout.flush();
                }

                // Incremental generation: one token at a time, the cache
                // supplies the rest. Sliding-window shifts happen
                // automatically inside gqa_attention_ when the cache is
                // full.
                for (int i = 1; i < max_response_tokens && !signal_handler::is_triggered(); ++i)
                {
                    if (text_end_id >= 0 && next_token == text_end_id) break;

                    matrix<int, 0, 1> incr_input(1, 1);
                    incr_input(0) = next_token;

                    auto& probs_tensor = generator(incr_input);
                    next_token = pick_next_token(probs_tensor);
                    ctx.add_token(next_token);

                    if (text_end_id >= 0 && next_token == text_end_id) break;

                    std::string token_text = tokenizer.decode(next_token, false);
                    cout << token_text;
                    cout.flush();
                }

                cout << "\n\n";
            }

            // End of dialogue: release the KV cache and any other context state.
            network_context::reset();
        }
        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}