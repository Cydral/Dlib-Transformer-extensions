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

    Usage modes:
    --fine-tune          Fine-tune the pre-trained model on Q&A pairs
    --prompt             Interactive prompting mode

    Data format for fine-tuning:
    <question><text>What is a black hole?</text>
    <answer><text>A black hole is a region of spacetime...</text>
!*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <sstream>
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

// Helper to append a special token, or its text encoding if not present in the vocabulary
void append_special_or_text(std::vector<int>& dest, bpe_tokenizer& tokenizer, const std::string& tag)
{
    try {
        int id = tokenizer.get_special_token_id(tag);
        dest.push_back(id);
    }
    catch (const std::runtime_error&) {
        auto enc = tokenizer.encode(tag);
        dest.insert(dest.end(), enc.begin(), enc.end());
    }
}

int main(int argc, char** argv)
{
    try
    {
        // Call internal interrupt handler
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("fine-tune", "Fine-tune the model on Q&A pairs for the chatbot");
        parser.add_option("prompt", "Interactive mode");
        parser.add_option("learning-rate", "Learning rate (default: 1e-5)", 1);
        parser.add_option("batch-size", "Batch size (default: 64)", 1);
        parser.add_option("max-epochs", "Max number of epochs (default: 500)", 1);
        parser.add_option("patience", "Iterations without progress before stop/reduce (default: 8000)", 1);
        parser.add_option("model-file", "Base model path (default: dlib_lm_moe_model.dat)", 1);
        parser.add_option("tokenizer-file", "Tokenizer path (default: dlib_lm_tokenizer.vocab)", 1);
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

        // Training hyperparameters
        const double learning_rate = get_option(parser, "learning-rate", 1e-5);
        const long batch_size = get_option(parser, "batch-size", 64);
        const long max_epochs = get_option(parser, "max-epochs", 500);
        const long patience = get_option(parser, "patience", 8000);
        const double weight_decay = 0.01;
        const double beta1 = 0.9;
        const double beta2 = 0.999;

        const std::string model_file = get_option(parser, "model-file", std::string("dlib_lm_moe_model.dat"));
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", std::string("dlib_lm_tokenizer.vocab"));

        // Strict alignment with MoE configuration
        constexpr long num_tokens = 2000;
        constexpr long max_seq_len = 100;
        constexpr long num_layers = 4;
        constexpr long num_heads = 6;
        constexpr long num_kv_heads = 2;
        constexpr long embedding_dim = 228;
        constexpr long num_experts = 4;
        constexpr long top_k_experts = 0;

        using my_transformer = gqa_moe_transformer_config<
            num_tokens, num_layers, num_heads, num_kv_heads, embedding_dim, num_experts, top_k_experts>;
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
            if (file_exists(model_file) && !file_exists("chkpt-" + finetuned_model)) {
                deserialize(model_file) >> net >> tokenizer;
                cout << "Base model loaded: " << model_file << "\n";
            }
            else if (file_exists(finetuned_model) && !file_exists("chkpt-" + finetuned_model)) {
                deserialize(finetuned_model) >> net >> tokenizer;
                cout << "Resuming from fine-tuned model: " << finetuned_model << "\n";
            }
            else if (file_exists(tokenizer_file)) {
                deserialize(tokenizer_file) >> tokenizer;
            }
            else {
                cerr << "Error: Pre-trained tokenizer not found (" << tokenizer_file << ")\n";
                return 1;
            }

            int pad_token = 0;
            try { pad_token = tokenizer.get_special_token_id("<pad>"); } catch (...) {}
            layer<0>(net).loss_details().set_ignore_index(pad_token);

            dnn_trainer<train_net, adam> trainer(net, adam(weight_decay, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-7);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_max_num_epochs(max_epochs);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_synchronization_file("chkpt-" + finetuned_model, std::chrono::minutes(5));
            trainer.be_quiet();

            cout << "Loading Q&A datasets...\n";
            auto all_qa_pairs = get_dataset_as_pairs({
                dataset_id::BLACK_HOLE_QA_PARTA, dataset_id::BLACK_HOLE_QA_PARTB, dataset_id::BLACK_HOLE_QA_PARTC
                });

            cout << "Tokenizing " << all_qa_pairs.size() << " pairs (Instruct format)...\n";
            std::vector<std::vector<int>> qa_tokens;
            size_t total_tokens = 0;

            for (const auto& qa_pair : all_qa_pairs) {
                std::vector<int> pair_tokens;

                append_special_or_text(pair_tokens, tokenizer, "<question>");
                append_special_or_text(pair_tokens, tokenizer, "<text>");
                auto q_tokens = tokenizer.encode(qa_pair.first);
                pair_tokens.insert(pair_tokens.end(), q_tokens.begin(), q_tokens.end());
                append_special_or_text(pair_tokens, tokenizer, "</text>");

                append_special_or_text(pair_tokens, tokenizer, "<answer>");
                append_special_or_text(pair_tokens, tokenizer, "<text>");
                auto a_tokens = tokenizer.encode(qa_pair.second);
                pair_tokens.insert(pair_tokens.end(), a_tokens.begin(), a_tokens.end());
                append_special_or_text(pair_tokens, tokenizer, "</text>");

                total_tokens += pair_tokens.size();
                qa_tokens.push_back(std::move(pair_tokens));
            }

            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;
            build_single_token_prediction_dataset(qa_tokens, max_seq_len, pad_token, true, samples, labels);
            cout << "Fine-tuning samples created: " << samples.size() << "\n";
            qa_tokens.clear();

            cout << "Applying partial freeze (fine-tuning strategy)...\n";
            set_all_learning_rate_multipliers(net, 0.1);
            layer<1>(net).layer_details().set_learning_rate_multiplier(1.0);
            layer<2>(net).layer_details().set_learning_rate_multiplier(0.5);

            network_context::set_optimizer_params(weight_decay, beta1, beta2);

            size_t epoch = 0, steps = 0, batches_count = 0;
            cout << "Starting training...\n";

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
                    steps += batch_samples.size();

                    if (batches_count++ % 50 == 0) {
                        double avg_loss = total_loss / batches_seen;
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::high_resolution_clock::now() - epoch_start).count();
                        double samples_per_sec = samples_seen / (elapsed > 0 ? elapsed : 1);

                        cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
                            << " \t loss: " << avg_loss
                            << " \t lr: " << trainer.get_learning_rate()
                            << " \t speed: " << samples_per_sec << " samples/sec\r";
                        cout.flush();
                    }
                }
                epoch++;
                cout << "\n";
            }

            set_all_learning_rate_multipliers(net, 1.0);
            net.clean();
            network_context::reset();

            serialize(finetuned_model) << net << tokenizer;
            cout << "\nFine-tuning completed! Model saved to: " << finetuned_model << "\n";
        }

        // ============================================================================
        // PROMPT MODE (INFERENCE)
        // ============================================================================
        else if (parser.option("prompt"))
        {
            cout << "=== INTERACTIVE MODE (PROMPTING) ===\n";
            display_random_qa_samples(3);
            cout << "Type 'quit' or 'exit' to stop.\n\n";

            size_t top_k = get_option(parser, "top-k", 50);
            float top_p = get_option(parser, "top-p", 0.9f);
            float repeat_penalty = get_option(parser, "repeat-penalty", 1.2f);
            float min_p = get_option(parser, "min-p", 0.05f);
            bool deterministic_mode = parser.option("deterministic");
            float temperature = deterministic_mode ? 1.0f : get_option(parser, "temperature", 0.8f);
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

            softmaxm<multiply<infer_net::subnet_type>> generator(multiply_(1.0 / temperature));
            generator.subnet().subnet() = net.subnet();

            int text_start_id = 0;
            int text_end_id = 0;
            int pad_token = 0;

            try { text_start_id = tokenizer.get_special_token_id("<text>"); } catch (...) {}
            try { text_end_id = tokenizer.get_special_token_id("</text>"); } catch (...) {}
            try { pad_token = tokenizer.get_special_token_id("<pad>"); } catch (...) {}
            inference_context ctx(max_seq_len, 3, pad_token);

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
                input_tokens.push_back(text_start_id);
                auto q_tokens = tokenizer.encode(user_input);
                input_tokens.insert(input_tokens.end(), q_tokens.begin(), q_tokens.end());
                input_tokens.push_back(text_end_id);

                ctx.add_tokens(input_tokens);

                std::vector<int> answer_prompt;
                append_special_or_text(answer_prompt, tokenizer, "<answer>");
                answer_prompt.push_back(text_start_id);
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
                        if (candidates.empty()) return text_end_id;

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

                int next_token, max_response_tokens = 3 * max_seq_len;

                for (int i = 0; i < max_response_tokens && !signal_handler::is_triggered(); ++i)
                {
                    auto input_seq = ctx.get_input_window();
                    long pad_len = count_leading_padding(input_seq, pad_token);
                    network_context::set_padding_uniform(pad_len, 1);

                    auto& probs_tensor = generator(input_seq);

                    const long seq_len = probs_tensor.nr();
                    const long v_size = probs_tensor.nc();
                    const long last_pos = seq_len - 1;
                    const long offset = tensor_index(probs_tensor, 0, 0, last_pos, 0);
                    const float* probs = probs_tensor.host() + offset;

                    if (deterministic_mode) {
                        const float* max_ptr = std::max_element(probs, probs + v_size);
                        next_token = static_cast<int>(std::distance(probs, max_ptr));
                    }
                    else {
                        next_token = top_k_p_sample(probs, v_size, top_k, top_p, repeat_penalty, min_p);
                    }

                    ctx.add_token(next_token);

                    if (next_token == text_end_id) break;

                    std::string token_text = tokenizer.decode(next_token, false);
                    cout << token_text;
                    cout.flush();
                }

                network_context::clear_padding();
                cout << "\n\n";
            }
        }
        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}