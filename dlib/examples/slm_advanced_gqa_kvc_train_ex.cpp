/*!
    @file slm_advanced_gqa_kvc_train_ex.cpp
    @brief Variant of slm_advanced_gqa_train_ex.cpp using the unified GQA attention
           with a KV cache, validating end-to-end byte-accurate generation.

    This program shares its dataset, tokenizer, training pipeline, prompting workflow
    and command-line options with slm_advanced_gqa_train_ex.cpp; please refer to that
    example for the general description. The two differences are:

    1. Attention implementation
       This program selects the *unified* attention implementation, in which the
       full attention sub-graph (Q/K/V projections, RoPE on Q and K, optional GQA
       repeat, scaled dot-product, output projection) is fused into a single Dlib
       layer (gqa_attention_). The fused layer maintains a per-instance KV cache
       and exposes prefill / incremental inference modes via network_context.

    2. Inference loop
       Instead of running a full forward pass on the entire context window for
       every new token, the generation loop drives the network through three modes:

         - prefill     : a single forward on the initial prompt populates each
                         attention layer's KV cache with the new tokens' K (before
                         RoPE) and V tensors, restricted to the effective
                         (non-padded) positions.

         - incremental : each subsequent step feeds a 1-token input to the network.
                         Each attention layer projects Q/K/V for the new position,
                         appends K (pre-RoPE) and V to its cache, applies RoPE on
                         the entire cached K window with positions [0, L-1] and on
                         the new Q with position L-1, then computes attention
                         against the cached window.

         - sliding window: when an incremental step would push cache_filled_len_
                         past cache_capacity_ (= max_seq_len), the attention layer
                         automatically shifts its cache one position to the left,
                         dropping the oldest entry and freeing room for the new
                         one. RoPE positions therefore always remain inside
                         [0, max_seq_len), the range observed during training.
                         The generation loop has nothing to manage at this level.

    The KV cache itself is held inside each gqa_attention_ instance and is runtime-
    only (not serialized with the model). Mode and cache capacity are configured
    via the network_context singleton; the attention layer consults them at the
    start of every forward call.
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

// Include internal dataset
#include "slm_data.h"

using namespace std;
using namespace dlib;

// Utility functions
std::string generate_tokens_filename(size_t max_bytes)
{
    if (max_bytes > 0) {
        return "dlib_dataset_" + std::to_string(max_bytes) + "_tokens.bin";
    }
    return "dlib_dataset_tokens.bin";
}

bool save_tokens_to_file(const std::vector<int>& tokens, const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    uint64_t num_tokens = tokens.size();
    file.write(reinterpret_cast<const char*>(&num_tokens), sizeof(num_tokens));

    for (int token : tokens) {
        uint32_t t = static_cast<uint32_t>(token);
        file.write(reinterpret_cast<const char*>(&t), sizeof(t));
    }

    return file.good();
}

bool load_tokens_from_file(std::vector<int>& tokens, const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    uint64_t num_tokens;
    file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
    if (!file.good()) return false;

    tokens.clear();
    tokens.reserve(num_tokens);

    for (uint64_t i = 0; i < num_tokens; ++i) {
        uint32_t t;
        file.read(reinterpret_cast<char*>(&t), sizeof(t));
        if (!file.good()) return false;
        tokens.push_back(static_cast<int>(t));
    }

    return true;
}

std::string read_file_content(const std::string& filename, size_t max_bytes = 0)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string content;
    if (max_bytes > 0) {
        content.resize(max_bytes);
        file.read(&content[0], max_bytes);
        content.resize(file.gcount());
    }
    else {
        content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

    return content;
}

bool verify_match(const std::string& original, const std::string& generated)
{
    if (original.size() != generated.size()) {
        cout << "Size mismatch: original=" << original.size()
            << ", generated=" << generated.size() << "\n";
        return false;
    }

    size_t mismatch_count = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        if (original[i] != generated[i]) {
            if (mismatch_count < 10) {
                cout << "Mismatch at byte " << i << ": expected='" << original[i]
                    << "' (0x" << std::hex << (int)(unsigned char)original[i] << std::dec
                    << "), got='" << generated[i]
                    << "' (0x" << std::hex << (int)(unsigned char)generated[i] << std::dec << ")\n";
            }
            mismatch_count++;
        }
    }

    if (mismatch_count > 0) {
        cout << "Total mismatches: " << mismatch_count << "\n";
        return false;
    }

    cout << "Files match perfectly. All " << original.size() << " bytes are identical.\n";
    return true;
}

// ----------------------------------------------------------------------------------------

/*
    Returns the current KV cache fill length from the first gqa_attention
    layer in the network (layer<8> in the standard 4-layer GQA stack).
    All attention layers in the stack maintain caches of identical length
    since they all observe the same prefill/incremental dispatching, so
    reading any one of them suffices.
*/
template <typename net_type>
long gqa_cache_full_len(net_type& net)
{
    return dlib::layer<8>(net).layer_details().get_kv_cache_filled_len();
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        // Setup interrupt handling for clean termination
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("train", "Train a transformer model on internal dataset");
        parser.add_option("generate", "Generate text from a previously trained model");
        parser.add_option("verify", "Verify generated output against original dataset");
        parser.add_option("learning-rate", "Set the learning rate (default: 2e-4)", 1);
        parser.add_option("batch-size", "Set the mini-batch size (default: 64)", 1);
        parser.add_option("patience", "Iterations without progress before early stopping (default: 8000)", 1);
        parser.add_option("max-epochs", "Maximum number of training epochs (default: 400)", 1);
        parser.add_option("alpha", "Set the weight decay for AdamW (default: 0.004)", 1);
        parser.add_option("beta1", "Set AdamW's first moment coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "Set AdamW's second moment coefficient (default: 0.998)", 1);
        parser.add_option("model-file", "Path for model (default: dlib_lm_tokens_gqa_kvc_model.dat)", 1);
        parser.add_option("tokenizer-file", "Path for tokenizer (default: dlib_lm_tokenizer.vocab)", 1);
        parser.add_option("output-file", "Path for generated output (default: generated_text.txt)", 1);
        parser.add_option("max-tokens", "Maximum number of tokens to process (default: all)", 1);
        parser.add_option("max-bytes", "Maximum number of bytes to process (default: all)", 1);
        parser.add_option("percent", "Percentage of bytes to process (0-100 - default: all)", 1);
        parser.parse(argc, argv);

        if (parser.number_of_arguments() == 0 &&
            !parser.option("train") && !parser.option("generate") &&
            !parser.option("verify"))
        {
            parser.print_options();
            return 0;
        }

        // Default values
        const double learning_rate = get_option(parser, "learning-rate", 2e-4);
        const size_t batch_size = get_option(parser, "batch-size", 64);
        const long patience = get_option(parser, "patience", 8000);
        const size_t max_epochs = get_option(parser, "max-epochs", 400);
        const double alpha = get_option(parser, "alpha", 0.004);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.998);
        const std::string model_file = get_option(parser, "model-file", "dlib_lm_tokens_gqa_kvc_model.dat");
        const std::string tokenizer_file = get_option(parser, "tokenizer-file", "dlib_lm_tokenizer.vocab");
        const std::string output_file = get_option(parser, "output-file", "generated_text.txt");
        
        // Model architecture parameters
        const long num_tokens = 1400;
        const long num_layers = 3;
        const long num_heads = 6;
        const long num_kv_heads = 2;
        const long embedding_dim = 228;
        const long max_seq_len = 200;

        // Define transformer configuration
        using my_transformer = gqa_transformer_config<
            num_tokens,     // vocab
            num_layers,     // layers
            num_heads,      // heads
            num_kv_heads,   // kv_heads
            embedding_dim,  // dim
			attention_impl::unified // attention implementation
        >;        

        // Load internal dataset
        cout << "Loading internal training dataset...\n";
        std::string training_text = get_dataset_as_text(dataset_id::BLACK_HOLE_ARTICLE);
        size_t original_size = training_text.size();
        cout << "Loaded " << original_size << " bytes from internal dataset\n";

        // Calculate max bytes to process
        size_t max_bytes = 0, max_tokens_limit = 0;
        if (parser.option("max-tokens"))
            max_tokens_limit = std::stoul(parser.option("max-tokens").argument());
        if (parser.option("max-bytes")) {
            max_bytes = std::stoul(parser.option("max-bytes").argument());
        }
        else if (parser.option("percent")) {
            double percent = std::stod(parser.option("percent").argument());
            max_bytes = static_cast<size_t>(original_size * percent / 100.0);
            cout << "Processing " << percent << "% of dataset = " << max_bytes << " bytes\n";
        }

        // Apply size limits to dataset
        if (max_bytes > 0 && max_bytes < training_text.size()) {
            training_text.resize(max_bytes);
            cout << "Limited to " << training_text.size() << " bytes\n";
        }

        // Determine tokens filename
        const std::string tokens_file = generate_tokens_filename(max_bytes);

        // Tokenizer BPE
        bpe_tokenizer tokenizer;

        // Load pre-trained tokenizer if it exists
        if (file_exists(tokenizer_file)) {
            cout << "Loading pre-trained tokenizer from: " << tokenizer_file << endl;
            deserialize(tokenizer_file) >> tokenizer;
            cout << "Tokenizer loaded successfully with vocabulary size: " << tokenizer.get_vocab_size() << endl;
        }
        else {
            cout << "Pre-trained tokenizer not found at: " << tokenizer_file << endl;
            cout << "Will train a new tokenizer if needed." << endl;
        }

        // For GPU usage (if available)
        std::vector<int> gpus{ 0 };

        // Variables to store tokens
        std::vector<int> full_tokens;

        // Training mode
        if (parser.option("train"))
        {
            cout << "=== TRAINING MODE ===\n";

            bool tokens_loaded = false;

            // Check if we should load pre-tokenized tokens
            if (file_exists(tokens_file)) {
                cout << "Found pre-tokenized tokens file: " << tokens_file << endl;
                cout << "Loading tokens from file...\n";
                if (load_tokens_from_file(full_tokens, tokens_file)) {
                    cout << "Loaded " << full_tokens.size() << " tokens from file.\n";
                    if (max_tokens_limit > 0 && max_tokens_limit < full_tokens.size()) {
                        full_tokens.resize(max_tokens_limit);
                        cout << "Limited to " << full_tokens.size() << " tokens for training.\n";
                    }
                    tokens_loaded = true;
                }
                else {
                    cerr << "Failed to load tokens from file. Will tokenize again.\n";
                }
            }

            if (!tokens_loaded) {
                // Train a new tokenizer if needed
                if (!file_exists(tokenizer_file)) {
                    cout << "Training new BPE tokenizer with vocabulary size " << num_tokens << "...\n";

                    // Compose training corpus from multiple datasets
                    std::string delimiter = "@@";
                    std::string tokenizer_corpus =
                        get_dataset_as_text(dataset_id::BLACK_HOLE_ARTICLE) + delimiter
                        + get_dataset_as_text(dataset_id::PHYSICS_PARAGRAPHS) + delimiter
                        + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTA) + delimiter
                        + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTB) + delimiter
                        + get_dataset_as_text(dataset_id::BLACK_HOLE_QA_PARTC) + delimiter
                        + get_dataset_as_text(dataset_id::GENERAL_KNOWLEDGE);

                    // Replace all "@@" delimiters with spaces                    
                    size_t pos = 0;
                    while ((pos = tokenizer_corpus.find(delimiter, pos)) != std::string::npos) {
                        tokenizer_corpus.replace(pos, delimiter.length(), " ");
                        pos += 1; // Move past the replacement space
                    }

                    tokenizer.train(tokenizer_corpus, num_tokens, 1e6, true);
                    serialize(tokenizer_file) << tokenizer;
                    cout << "Tokenizer saved to " << tokenizer_file << endl;
                }

                // Tokenize the full text
                cout << "Tokenizing input text...\n";
                int text_start_id = tokenizer.get_special_token_id("<text>"),
                    text_end_id = tokenizer.get_special_token_id("</text>");
                if (text_start_id < 0 || text_end_id < 0)
                    cout << "Warning: Special tokens not found in tokenizer vocabulary.\n";
                auto start_time = std::chrono::high_resolution_clock::now();
                full_tokens.clear();
                full_tokens.push_back(text_start_id);
                auto encoded_tokens = tokenizer.encode(training_text);
                full_tokens.insert(full_tokens.end(), encoded_tokens.begin(), encoded_tokens.end());
                full_tokens.push_back(text_end_id);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto tokenize_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

                cout << "Tokenization completed in " << tokenize_time << " seconds.\n";
                cout << "Number of tokens: " << full_tokens.size() << endl;

                // Save tokens for future use
                cout << "Saving tokens to file: " << tokens_file << endl;
                if (save_tokens_to_file(full_tokens, tokens_file)) {
                    cout << "Tokens successfully saved for future use.\n";
                }
                else {
                    cerr << "Warning: Failed to save tokens for future use.\n";
                }
            }

            // Prepare training sequences (sliding window)
            cout << "Preparing training sequences...\n";
            std::vector<matrix<int, 0, 1>> samples;
            std::vector<unsigned long> labels;

            const int pad_token = tokenizer.get_special_token_id("<pad>");
            build_single_token_prediction_dataset({ full_tokens }, max_seq_len,
                pad_token, false, samples, labels);
            full_tokens.clear();
            cout << "Created " << samples.size() << " training samples\n";

            // Build and train the network
            using net_type = my_transformer::network_type<true>;
            net_type net;            
            layer<0>(net).loss_details().set_ignore_index(pad_token);
            cout << my_transformer::model_info::describe() << endl;

            // Tokenizer stored with model for simplified inference
            if (file_exists(model_file) &&
                !file_exists("chkpt-" + model_file)) deserialize(model_file) >> net >> tokenizer;

            // Create trainer
            dnn_trainer<net_type, adamw> trainer(net, adamw(alpha, beta1, beta2), gpus);
            trainer.set_learning_rate(learning_rate);
            trainer.set_min_learning_rate(1e-6);
            trainer.set_mini_batch_size(batch_size);
            trainer.set_iterations_without_progress_threshold(patience);
            trainer.set_synchronization_file("chkpt-" + model_file, std::chrono::minutes(10));
            trainer.be_quiet();

            cout << "Number of model parameters: " << count_network_parameters(net, max_seq_len) << endl;
            cout << "Starting training...\n";

            size_t epoch = 0, steps = 0, batches_count = 0, batches_seen, samples_seen;
            double total_loss;
            auto epoch_start = std::chrono::high_resolution_clock::now();

            // Training loop
            while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
                && epoch < max_epochs && !signal_handler::is_triggered())
            {
                total_loss = 0.0;
                batches_seen = samples_seen = 0;
                epoch_start = std::chrono::high_resolution_clock::now();

                // Shuffle the dataset
                shuffle_training_dataset(samples, labels);

                for (size_t i = 0; i < samples.size() && !signal_handler::is_triggered(); i += batch_size)
                {
                    size_t batch_end = std::min(i + batch_size, samples.size());
                    std::vector<matrix<int, 0, 1>> batch_samples(
                        samples.begin() + i, samples.begin() + batch_end);
                    std::vector<unsigned long> batch_labels(
                        labels.begin() + i, labels.begin() + batch_end);

                    std::vector<long> pad_lengths(batch_samples.size());
                    for (size_t j = 0; j < batch_samples.size(); ++j)
                        pad_lengths[j] = count_leading_padding(batch_samples[j], pad_token);

                    // Synchronize: ensure trainer has finished processing previous batch
                    // before modifying the shared network_context singleton
                    trainer.get_net(force_flush_to_disk::no);

                    network_context::set_padding_from_lengths(pad_lengths);
                    network_context::set_learning_rate(trainer.get_learning_rate());

                    trainer.train_one_step(batch_samples, batch_labels);
                    total_loss += trainer.get_average_loss();
                    batches_seen++;
                    samples_seen += batch_samples.size();
					steps += batch_samples.size();

                    // Progress reporting
                    if (batches_count++ % 50 == 0) {
                        double avg_loss = total_loss / batches_seen;
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::high_resolution_clock::now() - epoch_start).count();
                        double samples_per_sec = samples_seen / (elapsed > 0 ? elapsed : 1);

                        cout << "epoch#: " << (epoch + 1) << "/" << max_epochs
							<< " (ksteps: " << (steps / 1000) << ")"
                            << " \t loss: " << avg_loss
                            << " \t patience: " << trainer.get_steps_without_progress()
                            << " \t speed: " << samples_per_sec << " samples/sec\n";
                        cout.flush();
                    }
                }
                epoch++;
            }

            // Save model
            trainer.get_net();
            net.clean();
            serialize(model_file) << net << tokenizer;
            cout << "Model saved to " << model_file << "\n";

            // Evaluate on training set
            {
                if (!signal_handler::is_triggered()) {
                    cout << "Evaluating model accuracy...\n";
                    my_transformer::network_type<false> g_infer;
                    deserialize(model_file) >> g_infer >> tokenizer;

                    // Feed padding context for consistency with training
                    std::vector<long> eval_pad_lengths(samples.size());
                    for (size_t i = 0; i < samples.size(); ++i)
                        eval_pad_lengths[i] = count_leading_padding(samples[i], pad_token);
                    network_context::set_padding_from_lengths(eval_pad_lengths);

                    auto predicted = g_infer(samples);
                    size_t correct = 0;
                    for (size_t i = 0; i < labels.size(); ++i)
                        if (predicted[i] == labels[i]) correct++;
                    double accuracy = (double)correct / labels.size();
                    cout << "Training accuracy: " << (accuracy * 100.0) << "%\n";

                    // We need perfect accuracy to reconstruct the internal dataset
                    if (accuracy < 0.999) {
                        cout << "WARNING: Model accuracy is less than 99.90%. The model may not "
                            << "perfectly reconstruct the input text.\n";
                    }
                }
            }
            network_context::reset();
        }

        // Generation mode
        if (parser.option("generate"))
        {
            cout << "=== GENERATION MODE ===\n";

            // Load the model
            my_transformer::network_type<false> net;
            if (file_exists(model_file)) {
                deserialize(model_file) >> net >> tokenizer;
                cout << "Loaded model from " << model_file << "\n";
                cout << "Number of model parameters: " << count_network_parameters(net, max_seq_len) << endl;
            }
            else {
                cerr << "Error: model file not found. Please run --train first.\n";
                return 0;
            }

            // Check that tokenizer is loaded
            if (tokenizer.get_vocab_size() == 0) {
                cerr << "Error: Tokenizer not loaded. Please provide a valid tokenizer file.\n";
                return 0;
            }

            std::vector<int> prompt_tokens;
            if (file_exists(tokens_file)) {
                cout << "Found pre-tokenized tokens file: " << tokens_file << endl;
                cout << "Loading tokens for prompt...\n";

                std::ifstream file(tokens_file, std::ios::binary);
                if (!file) {
                    cerr << "Failed to open tokens file: " << tokens_file << endl;
                }
                else {
                    uint64_t num_tokens_in_file;
                    file.read(reinterpret_cast<char*>(&num_tokens_in_file), sizeof(num_tokens_in_file));

                    size_t tokens_to_read = std::min(static_cast<size_t>(max_seq_len),
                        static_cast<size_t>(num_tokens_in_file));
                    prompt_tokens.resize(tokens_to_read);

                    for (size_t i = 0; i < tokens_to_read; ++i) {
                        uint32_t t;
                        file.read(reinterpret_cast<char*>(&t), sizeof(t));
                        prompt_tokens[i] = static_cast<int>(t);
                    }

                    cout << "Loaded " << prompt_tokens.size() << " tokens for prompt from file.\n";
                }
            }

            if (prompt_tokens.empty()) {
                cout << "Tokenizing initial prompt from internal dataset...\n";

                std::string prompt_text = training_text.substr(0, std::min(training_text.size(),
                    static_cast<size_t>(max_seq_len * 10)));

                int text_start_id = tokenizer.get_special_token_id("<text>");
                prompt_tokens.clear();
                prompt_tokens.push_back(text_start_id);
                auto encoded_tokens = tokenizer.encode(prompt_text);
                prompt_tokens.insert(prompt_tokens.end(), encoded_tokens.begin(), encoded_tokens.end());
            }

            if (prompt_tokens.size() > (size_t)max_seq_len) {
                prompt_tokens.resize(max_seq_len);
            }
            else if (prompt_tokens.size() < (size_t)max_seq_len) {
                cerr << "Warning: Not enough tokens in prompt. Got " << prompt_tokens.size()
                    << ", needed " << max_seq_len << ".\n";
                return 0;
            }
            cout << "Using " << prompt_tokens.size() << " tokens for initial prompt\n";

            size_t target_size = (max_bytes > 0) ? max_bytes : training_text.size();
            cout << "Will generate approximately " << target_size << " bytes\n";

            std::ofstream outfile(output_file, std::ios::binary);
            if (!outfile) {
                cerr << "Error: Cannot open output file: " << output_file << "\n";
                return 0;
            }

            // Write initial text (corresponding to prompt tokens)
            std::string initial_text = tokenizer.decode(prompt_tokens, false);
            outfile.write(initial_text.c_str(), initial_text.size());

            // The cache capacity equals max_seq_len : positions[0, max_seq_len) are
            // the only ones the model has seen during training. The attention layer
            // automatically slides the window left when the cache is full, so the
            // generation loop has nothing to manage beyond feeding the next token.
            cout << "Starting autoregressive generation (KV-cache mode)...\n";

            std::vector<int> token_buffer;
            const size_t buffer_size = 100;
            auto start_time = std::chrono::high_resolution_clock::now();
            size_t total_bytes = initial_text.size();
            size_t token_count = prompt_tokens.size();

            const int end_of_text = tokenizer.get_special_token_id("</text>");
            int next_token = 0;

            network_context::reset();
            network_context::set_kv_cache_capacity(max_seq_len);

            // Prefill on the prompt (no padding)
            {
                network_context::set_inference_mode(network_context::inference_mode::prefill);
                network_context::clear_padding();

                matrix<int, 0, 1> prefill_input(prompt_tokens.size(), 1);
                for (size_t i = 0; i < prompt_tokens.size(); ++i)
                    prefill_input(i) = prompt_tokens[i];

                next_token = net(prefill_input);
                token_buffer.push_back(next_token);
                token_count++;
                cout << "[Prefill done] cache_filled_len=" << gqa_cache_full_len(net)
                    << " next_token=" << next_token << "\n";
            }

            // Incremental generation
            network_context::set_inference_mode(network_context::inference_mode::incremental);
            network_context::clear_padding();

            while (total_bytes < target_size && next_token != end_of_text
                && !signal_handler::is_triggered())
            {
                matrix<int, 0, 1> incr_input(1, 1);
                incr_input(0) = next_token;
                next_token = net(incr_input);
                token_buffer.push_back(next_token);
                token_count++;

                if (token_buffer.size() >= buffer_size)
                {
                    std::string chunk = tokenizer.decode(token_buffer, false);
                    outfile.write(chunk.c_str(), chunk.size());
                    total_bytes += chunk.size();
                    token_buffer.clear();

                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - start_time).count();
                    const long generated = static_cast<long>(token_count) - static_cast<long>(prompt_tokens.size());
                    double tokens_per_second = generated / (elapsed > 0 ? elapsed : 1);
                    const double bytes_per_token = (chunk.size() > 0) ? chunk.size() / (double)buffer_size : 1.0;
                    cout << "Generated " << generated << " tokens, "
                        << total_bytes << " bytes ("
                        << (total_bytes * 100.0 / target_size) << "%) - "
                        << tokens_per_second << " tokens/sec - "
                        << "Est. completion: "
                        << (int)((target_size - total_bytes) / (tokens_per_second * bytes_per_token))
                        << " seconds\r";
                }
                if (max_tokens_limit > 0 && token_count >= max_tokens_limit) break;
            }
            network_context::reset();

            // Flush remaining buffer
            if (!token_buffer.empty()) {
                std::string chunk = tokenizer.decode(token_buffer, false);
                outfile.write(chunk.c_str(), chunk.size());
                total_bytes += chunk.size();
            }
            outfile.flush();
            outfile.close();

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
                end_time - start_time).count();

            cout << "\nGeneration complete in " << total_time << " seconds! (100%)\n";
            cout << "Generated " << (token_count - prompt_tokens.size()) << " tokens after prompt, "
                << total_bytes << " bytes total\n";
            cout << "Output saved to " << output_file << "\n";
        }

        // Verification mode - Compare original and generated file
        if (parser.option("verify"))
        {
            cout << "=== VERIFICATION MODE ===\n";

            if (!file_exists(output_file)) {
                cerr << "Error: Generated file not found at " << output_file << "\n";
                return 0;
            }

            // Read generated file
            cout << "Reading generated file...\n";
            std::string generated = read_file_content(output_file);

            // Read the same portion of original dataset
            cout << "Reading original dataset (set to same size as generated)...\n";
            std::string original = training_text.substr(0, std::min(training_text.size(), generated.size()));

            cout << "Verifying byte-for-byte match...\n";
            bool verify = verify_match(original, generated);

            if (verify)
                cout << "SUCCESS: The generated file matches the original text perfectly!\n";
            else
                cout << "FAILED: The generated file does not match the original text.\n";
        }

        return 0;
    }
    catch (exception& e)
    {
        cerr << "Exception thrown: " << e.what() << endl;
        return 1;
    }
}