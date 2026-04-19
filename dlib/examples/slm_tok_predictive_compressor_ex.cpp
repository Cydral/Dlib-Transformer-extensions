/*!
    @file slm_predictive_compressor_ex.cpp
    @brief Token-level predictive compression using a Transformer.

    This example demonstrates lossless data compression using BPE tokenization
    and a small Transformer model to predict the next token.  The compression
    scheme uses a single bit to indicate prediction success (argmax match),
    followed by a multi-bit literal token ID on mismatch.  The raw bitstream
    is then entropy-coded using Dlib's built-in codec for additional reduction.

    This approach is robust to GPU floating-point non-determinism because it
    only relies on the argmax of the prediction (invariant to tiny logit
    perturbations), not the full softmax distribution.

    The pipeline has three stages:

    1. BPE tokenization: reduces the byte stream to a shorter token sequence.

    2. Transformer training: a next-token predictor is trained on the token
       sequence using Dlib's dnn_trainer with Adam.

    3. Predictive compression: for each token beyond the seed window, the
       model predicts via argmax.  If correct, one bit is written.  If wrong,
       one bit + the actual token ID (in TOKEN_BITS bits) is written.  The
       resulting bitstream is entropy-coded with Dlib's compress_stream.

    Compressed file format:
    [MAGIC 3B "DLC"] [original_size 8B] [CRC32 4B] [num_tokens 8B] [flags 1B]
    [serialized bpe_tokenizer + model (if embedded)]
    [literal_seed tokens] [entropy-coded bitstream]

    Decompression reverses the process: the same Transformer produces the same
    softmax sequence (deterministic forward pass), the range decoder recovers
    each token, and the BPE tokenizer decodes the token sequence back to the
    original byte stream.

    Usage:
    --compress   --input <file> [--output <file>]
    --compress   --input <file> --no-train            (skip training, reuse saved model)
    --compress   --input <file> --no-embed-model       (do not embed model in output)
    --decompress --input <file> [--output <file>]

    Recommended workflow:
    1. First run on a large representative corpus:
         ./compressor --compress --input corpus.txt --no-embed-model
       This trains both the BPE tokenizer and the Transformer model, saving
       them together in MODEL_SAVE_FILE.  The tokenizer vocabulary is frozen
       from this point forward.

    2. Subsequent runs on any file (fine-tune the model, keep the tokenizer):
         ./compressor --compress --input data.txt
       The tokenizer is reloaded unchanged; only the Transformer is fine-tuned.

    3. Compress without retraining (reuse everything as-is):
         ./compressor --compress --input data.txt --no-train
!*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <thread>
#include <atomic>
#include <mutex>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/crc32.h>
#include <dlib/tokenizer.h>
#include <dlib/compress_stream.h>

using namespace std;
using namespace dlib;

// Thread-safe progress display: workers use try_lock to avoid blocking.
// If the mutex is already held, the progress update is simply skipped.
// This guarantees clean output: at most one writer at any time, and no
// thread ever blocks on console I/O.
static std::mutex progress_mutex;

// Default number of worker threads: keep 4 cores for OS/system, cap at 10
// to avoid diminishing returns and excessive memory usage from network copies.
long default_num_threads()
{
    long hw = static_cast<long>(std::thread::hardware_concurrency());
    return std::max(1L, std::min(10L, hw - 4));
}
const char     MAGIC[3] = { 'D', 'L', 'C' };        // 3-byte file signature
const long     MAX_VOCAB_SIZE = 1400;               // BPE vocabulary target
const int      WINDOW_SIZE = 16;                    // Prediction context in tokens
const long     NUM_LAYERS = 2;
const long     NUM_HEADS = 4;
const long     EMBEDDING_DIM = 32;
const int      BATCH_SIZE = 128;                    // Training mini-batch
const size_t   MAX_TRAINING_BYTES = 500UL * 1024 * 1024;    // Cap for BPE tokenizer training
const size_t   MAX_TRAINING_TOKENS = 50000000UL;            // 50M token cap for transformer training
const uint8_t  FLAG_MODEL_EMBEDDED = 0x01;

const std::string MODEL_SAVE_FILE = "dlib_tok_predictive_compressor.dat";

using compressor_config = fused_transformer_config<MAX_VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, EMBEDDING_DIM>;
using train_net = compressor_config::network_type<true>;
using infer_net = compressor_config::network_type<false>;

typedef dlib::compress_stream::kernel_1ec entropy_codec;

// Bit stream classes (same as byte-level compressor, extended for multi-bit tokens)
class out_bit_stream {
    std::vector<uint8_t>& buffer;
    uint8_t current_byte = 0;
    int bits_count = 0;
public:
    out_bit_stream(std::vector<uint8_t>& buf) : buffer(buf) {}
    void write_bit(bool b) {
        if (b) current_byte |= (1 << bits_count);
        bits_count++;
        if (bits_count == 8) { buffer.push_back(current_byte); current_byte = 0; bits_count = 0; }
    }
    void write_token(int token, int nbits) {
        for (int i = 0; i < nbits; ++i) write_bit((token >> i) & 1);
    }
    void flush() {
        if (bits_count > 0) { buffer.push_back(current_byte); current_byte = 0; bits_count = 0; }
    }
};

class in_bit_stream {
    const std::vector<uint8_t>& buffer;
    size_t byte_idx = 0;
    int bits_count = 0;
public:
    in_bit_stream(const std::vector<uint8_t>& buf) : buffer(buf) {}
    bool read_bit() {
        if (byte_idx >= buffer.size()) return false;
        bool b = (buffer[byte_idx] >> bits_count) & 1;
        bits_count++;
        if (bits_count == 8) { byte_idx++; bits_count = 0; }
        return b;
    }
    int read_token(int nbits) {
        int token = 0;
        for (int i = 0; i < nbits; ++i) { if (read_bit()) token |= (1 << i); }
        return token;
    }
};

std::string format_duration(double s)
{
    int h = (int)(s / 3600), m = (int)((s - h * 3600) / 60), sec = (int)(s - h * 3600 - m * 60);
    std::ostringstream o;
    if (h > 0) o << h << "h " << m << "m " << sec << "s";
    else if (m > 0) o << m << "m " << sec << "s";
    else o << sec << "s";
    return o.str();
}

std::string format_size(size_t bytes)
{
    std::ostringstream o;
    if (bytes >= 1048576) o << std::fixed << std::setprecision(2) << bytes / 1048576.0 << " MB";
    else if (bytes >= 1024) o << std::fixed << std::setprecision(1) << bytes / 1024.0 << " KB";
    else o << bytes << " B";
    return o.str();
}

std::string format_ratio(size_t compressed, size_t original)
{
    if (original == 0) return "N/A";
    double r = (1.0 - (double)compressed / original) * 100.0;
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << (r >= 0 ? "+" : "") << r << "%";
    return o.str();
}

uint32_t compute_crc32(const std::vector<uint8_t>& data)
{
    return dlib::crc32(std::string(data.begin(), data.end()));
}

std::vector<uint8_t> read_file_bytes(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    return std::vector<uint8_t>(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Thread-safe progress bar helper: prints only if the mutex is available,
// silently skips otherwise.  This avoids interleaved output without blocking.
void try_show_progress(const std::string& label, size_t done, size_t total,
    std::chrono::steady_clock::time_point t0, const std::string& unit = "tok/s")
{
    if (!progress_mutex.try_lock()) return;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double pct = (double)done / std::max(total, (size_t)1) * 100.0;
    double speed = (elapsed > 0) ? done / elapsed : 0;
    std::string eta = (done > 0 && done < total && speed > 0) ? format_duration((total - done) / speed) : "---";
    int bar_w = 30, filled = (int)(pct * bar_w / 100.0);
    cout << "\r  " << label << " [";
    for (int j = 0; j < bar_w; ++j) cout << (j < filled ? '#' : '.');
    cout << "] " << std::fixed << std::setprecision(1) << pct << "% " << (long)speed << " " << unit << "  ETA: " << eta << "     " << std::flush;
    progress_mutex.unlock();
}

// Parallel BPE tokenization: split text at newline boundaries, encode each
// chunk in parallel via dlib::parallel_for, concatenate in order.  Correct
// because BPE encode() operates on whitespace-delimited pre-tokens and does
// not inject special tokens.
std::vector<int> parallel_encode(const bpe_tokenizer& tokenizer, const std::string& text, long num_threads = 0)
{
    if (num_threads <= 0) num_threads = default_num_threads();
    if (num_threads == 1 || text.size() < 100000) return tokenizer.encode(text);

    // Split text at newline boundaries into approximately equal chunks
    std::vector<std::pair<size_t, size_t>> chunks;  // (start, length)
    size_t chunk_target = text.size() / num_threads;
    size_t start = 0;
    for (long i = 0; i < num_threads; ++i) {
        if (start >= text.size()) break;
        size_t end = (i == num_threads - 1) ? text.size() : std::min(start + chunk_target, text.size());
        while (end < text.size() && text[end] != '\n') ++end;
        if (end < text.size()) ++end;
        chunks.push_back({ start, end - start });
        start = end;
    }
    if (start < text.size()) {
        if (!chunks.empty()) chunks.back().second = text.size() - chunks.back().first;
        else chunks.push_back({ 0, text.size() });
    }

    long n_chunks = static_cast<long>(chunks.size());
    std::vector<std::vector<int>> chunk_tokens(n_chunks);

    std::atomic<long> chunks_done(0);
    std::atomic<size_t> bytes_done(0);
    auto tok_start = std::chrono::steady_clock::now();
    size_t total_bytes = text.size();

    cout << "  Parallel tokenization: " << n_chunks << " chunks, " << num_threads
        << " threads, " << format_size(total_bytes) << endl;

    dlib::parallel_for(num_threads, 0L, n_chunks, [&](long i) {
        chunk_tokens[i] = tokenizer.encode(text.substr(chunks[i].first, chunks[i].second));
        bytes_done += chunks[i].second;
        chunks_done++;
        try_show_progress("Tokenizing", bytes_done.load(), total_bytes, tok_start, "B/s");
        });

    double total_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tok_start).count();
    cout << "\r  Tokenizing [##############################] 100.0%  "
        << format_size((size_t)(total_bytes / std::max(total_secs, 0.001))) << "/s  " << format_duration(total_secs)
        << "                    " << endl;

    size_t total = 0;
    for (auto& ct : chunk_tokens) total += ct.size();
    std::vector<int> result;
    result.reserve(total);
    for (auto& ct : chunk_tokens) result.insert(result.end(), ct.begin(), ct.end());

    cout << "  Tokenization complete: " << result.size() << " tokens from " << n_chunks << " chunks" << endl;
    return result;
}

// Parallel BPE detokenization: each token decodes independently (simple byte
// pattern lookup), so we split the token vector and decode chunks in parallel.
std::string parallel_decode(const bpe_tokenizer& tokenizer, const std::vector<int>& tokens, long num_threads = 0)
{
    if (num_threads <= 0) num_threads = default_num_threads();
    if (num_threads == 1 || tokens.size() < 10000) return tokenizer.decode(tokens, false);

    long n_chunks = std::min(num_threads, static_cast<long>(tokens.size() / 1000));
    if (n_chunks < 1) n_chunks = 1;
    size_t chunk_size = tokens.size() / n_chunks;

    std::vector<std::string> chunk_strings(n_chunks);

    dlib::parallel_for(num_threads, 0L, n_chunks, [&](long i) {
        size_t start = i * chunk_size;
        size_t end = (i == n_chunks - 1) ? tokens.size() : start + chunk_size;
        std::vector<int> chunk(tokens.begin() + start, tokens.begin() + end);
        chunk_strings[i] = tokenizer.decode(chunk, false);
        });

    size_t total_len = 0;
    for (auto& s : chunk_strings) total_len += s.size();
    std::string result;
    result.reserve(total_len);
    for (auto& s : chunk_strings) result += s;
    return result;
}

void sample_batch(
    const std::vector<int>& tokens,
    size_t training_size,
    std::mt19937& rng,
    std::vector<matrix<int, 0, 1>>& batch_samples,
    std::vector<unsigned long>& batch_labels
)
{
    batch_samples.clear();
    batch_labels.clear();
    batch_samples.reserve(BATCH_SIZE);
    batch_labels.reserve(BATCH_SIZE);

    std::uniform_int_distribution<size_t> dist(WINDOW_SIZE, training_size - 1);
    matrix<int, 0, 1> window(WINDOW_SIZE, 1);

    for (int b = 0; b < BATCH_SIZE; ++b) {
        size_t target = dist(rng);
        for (int j = 0; j < WINDOW_SIZE; ++j)
            window(j) = tokens[target - WINDOW_SIZE + j];
        batch_samples.push_back(window);
        batch_labels.push_back(static_cast<unsigned long>(tokens[target]));
    }
}

void train_predictor(
    train_net& net,
    const std::vector<int>& tokens,
    const std::string& input_path
)
{
    size_t training_size = std::min(tokens.size(), MAX_TRAINING_TOKENS);
    if (training_size <= (size_t)WINDOW_SIZE) return;

    if (training_size < tokens.size()) {
        cout << "Token sequence exceeds training limit (" << MAX_TRAINING_TOKENS / 1000000
            << "M tokens). Training on first " << training_size << " tokens." << endl;
    }

    size_t num_transitions = training_size - WINDOW_SIZE;
    cout << "Training transitions available: " << num_transitions << endl;
    cout << "Mini-batch size: " << BATCH_SIZE << endl;

    dnn_trainer<train_net, adam> trainer(net, adam(0.004, 0.9, 0.999));
    trainer.set_learning_rate(1e-4);
    trainer.set_min_learning_rate(1e-8);
    trainer.set_iterations_without_progress_threshold(8000);

    std::string chkpt = input_path + ".chkpt";
    trainer.set_synchronization_file(chkpt, std::chrono::minutes(10));
    trainer.be_quiet();

    cout << "Checkpoint: " << chkpt << endl;
    cout << "Press CTRL+C to interrupt safely and proceed to compression.\n" << endl;

    std::mt19937 rng(42);
    std::vector<matrix<int, 0, 1>> batch_samples;
    std::vector<unsigned long> batch_labels;

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
        && !signal_handler::is_triggered())
    {
        sample_batch(tokens, training_size, rng, batch_samples, batch_labels);
        trainer.train_one_step(batch_samples, batch_labels);

        if (trainer.get_train_one_step_calls() % 50 == 0) {
            cout << "Step: " << trainer.get_train_one_step_calls()
                << " | Loss: " << trainer.get_average_loss()
                << " | LR: " << trainer.get_learning_rate() << "\r";
            cout.flush();
        }
    }

    if (signal_handler::is_triggered()) {
        cout << "\n[!] Training interrupted. Proceeding with current weights." << endl;
        signal_handler::reset();
    }
    else {
        cout << "\nTraining complete." << endl;
    }

    trainer.get_net();
    net.clean();
}

void compress_file(const std::string& input_path, const std::string& output_path, bool do_train, bool embed_model)
{
    cout << "=== COMPRESSION MODE ===" << endl;
    cout << "Input: " << input_path << "  Output: " << output_path << endl;
    cout << "Training: " << (do_train ? "Yes" : "No (reuse)") << "  Embed model: " << (embed_model ? "Yes" : "No") << endl << endl;

    std::vector<uint8_t> data = read_file_bytes(input_path);
    cout << "Input size: " << format_size(data.size()) << " (" << data.size() << " bytes)" << endl;

    if (data.size() < 256)
        throw std::runtime_error("File too small to compress (minimum 256 bytes).");

    uint32_t crc = compute_crc32(data);
    cout << "CRC32: " << std::hex << std::setfill('0') << std::setw(8) << crc
        << std::dec << endl << endl;

    // Stage 1: BPE tokenization
    // The tokenizer is trained once (first run) and persisted with the model.
    // Subsequent runs reload it unchanged; only the Transformer is fine-tuned.
    bpe_tokenizer tokenizer;
    std::vector<int> tokens;
    bool model_exists = file_exists(MODEL_SAVE_FILE);

    cout << "--- Stage 1: BPE tokenization ---" << endl;
    std::string text(data.begin(), data.end());

    if (model_exists) {
        // Tokenizer already trained: reload it from the saved model file
        cout << "Loading existing tokenizer from " << MODEL_SAVE_FILE << "..." << endl;
        std::ifstream model_in(MODEL_SAVE_FILE, std::ios::binary);
        deserialize(tokenizer, model_in);
        // model_in is left positioned after the tokenizer for later model loading
    }
    else {
        // First run: train the tokenizer from scratch on the input corpus
        if (!do_train)
            throw std::runtime_error(
                "No saved model found (" + MODEL_SAVE_FILE +
                "). Run without --no-train first to train the tokenizer and model.");

        size_t bpe_bytes = std::min(data.size(), MAX_TRAINING_BYTES);
        cout << "Training BPE tokenizer (vocab=" << MAX_VOCAB_SIZE
            << ", bytes=" << format_size(bpe_bytes) << ")..." << endl;
        tokenizer.train(text, MAX_VOCAB_SIZE, bpe_bytes, true);
    }

    cout << "Vocab: " << tokenizer.get_vocab_size() << " (specials: " << tokenizer.get_specials_size() << ")" << endl;

    tokens = parallel_encode(tokenizer, text);
    cout << "Tokens: " << tokens.size() << " (" << std::fixed << std::setprecision(2)
        << (double)data.size() / tokens.size() << " bytes/token)" << endl;

    // Verify lossless round-trip
    std::string decoded = parallel_decode(tokenizer, tokens);
    if (decoded != text) { decoded = tokenizer.decode(tokens, true); }
    if (decoded != text) {
        throw std::runtime_error("BPE round-trip failure (decoded " + std::to_string(decoded.size()) +
            " vs original " + std::to_string(text.size()) + " bytes). Cannot proceed.");
    }
    cout << "BPE round-trip: OK" << endl;

    // Ensure the token sequence is long enough for the prediction window
    if (tokens.size() <= (size_t)WINDOW_SIZE) {
        throw std::runtime_error(
            "Token sequence too short (" + std::to_string(tokens.size()) +
            " tokens) for WINDOW_SIZE=" + std::to_string(WINDOW_SIZE) +
            ". Use a larger input file or reduce WINDOW_SIZE.");
    }

    // Validate all token IDs are within the compile-time vocabulary range
    long effective_vocab = static_cast<long>(tokenizer.get_vocab_size());
    if (effective_vocab > MAX_VOCAB_SIZE) {
        cerr << "WARNING: Effective vocab (" << effective_vocab
            << ") exceeds compile-time MAX_VOCAB_SIZE (" << MAX_VOCAB_SIZE
            << "). Clamping." << endl;
        effective_vocab = MAX_VOCAB_SIZE;
    }

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 0 || tokens[i] >= effective_vocab) {
            throw std::runtime_error("Token ID " + std::to_string(tokens[i]) + " at position " + std::to_string(i)
                + " is out of vocabulary range [0, " + std::to_string(effective_vocab) + ").");
        }
    }

    // Vocabulary utilization analysis: count which tokens are actually used
    // in the encoded stream.  The vocabulary layout is:
    //   [0..255]                          base byte tokens
    //   [256..vocab_without_specials-1]   BPE merge tokens
    //   [vocab_without_specials..vocab-1] special tokens (never emitted by encode)
    {
        std::vector<size_t> token_freq(effective_vocab, 0);
        for (int t : tokens) token_freq[t]++;

        long merge_start = 256;
        long specials_start = static_cast<long>(tokenizer.get_vocab_without_specials_size());
        long active_base = 0, active_merge = 0, unused_merge = 0;

        for (long i = 0; i < effective_vocab; ++i) {
            if (i < merge_start) { if (token_freq[i] > 0) active_base++; }
            else if (i < specials_start) { if (token_freq[i] > 0) active_merge++; else unused_merge++; }
            // IDs >= specials_start are special tokens — never in encode() output, not counted
        }

        long total_merges = specials_start - merge_start;
        long total_active = active_base + active_merge;
        double utilization = total_merges > 0 ? (double)active_merge / total_merges * 100.0 : 0;

        cout << "Vocab utilization: " << total_active << "/" << specials_start << " content tokens active ("
            << active_base << "/256 base + " << active_merge << "/" << total_merges << " merges"
            << "), " << unused_merge << " unused merges (" << std::fixed << std::setprecision(1) << utilization << "%)" << endl;

        if (unused_merge > total_merges / 4) {
            cout << "  NOTE: " << unused_merge << " unused merge tokens. Consider reducing --target-vocab "
                << "or training BPE on the actual corpus with max_bytes." << endl;
        }
    }

    // Stage 2: Transformer training
    cout << endl << "--- Stage 2: Transformer training ---" << endl;
    train_net t_net;

    if (do_train) {
        if (model_exists) {
            cout << "Loading existing model for fine-tuning..." << endl;
            std::ifstream model_in(MODEL_SAVE_FILE, std::ios::binary);
            // Skip past the tokenizer to reach the model
            bpe_tokenizer skip_tok;
            deserialize(skip_tok, model_in);
            deserialize(t_net, model_in);
        }
        else {
            cout << "Training new model from scratch." << endl;
        }

        cout << "Model parameters: " << count_network_parameters(t_net, WINDOW_SIZE) << endl;
        train_predictor(t_net, tokens, input_path);

        // Save tokenizer + model together (tokenizer is always re-persisted unchanged)
        std::ofstream model_out(MODEL_SAVE_FILE, std::ios::binary);
        serialize(tokenizer, model_out);
        serialize(t_net, model_out);
        cout << "Model saved to: " << MODEL_SAVE_FILE << endl << endl;
    }
    else {
        if (!model_exists)
            throw std::runtime_error(
                "No saved model found (" + MODEL_SAVE_FILE +
                "). Run without --no-train first.");

        std::ifstream model_in(MODEL_SAVE_FILE, std::ios::binary);
        bpe_tokenizer skip_tok;
        deserialize(skip_tok, model_in);
        deserialize(t_net, model_in);
        t_net.clean();
        cout << "Model loaded from: " << MODEL_SAVE_FILE << endl << endl;
    }

    // Build inference network
    infer_net net(t_net);

    // Stage 3: Predict-or-literal compression (same scheme as byte-level compressor)
    // Uses argmax prediction (robust to GPU non-determinism) + Dlib entropy codec.
    cout << "--- Stage 3: Predictive compression ---" << endl;

    int token_bits = 0;
    { long v = effective_vocab - 1; while (v > 0) { token_bits++; v >>= 1; } }
    cout << "Token encoding: " << token_bits << " bits/literal (vocab=" << effective_vocab << ")" << endl;

    std::vector<int> literal_seed(tokens.begin(), tokens.begin() + WINDOW_SIZE);
    std::vector<uint8_t> compressed_data;
    out_bit_stream bit_writer(compressed_data);
    matrix<int, 0, 1> window(WINDOW_SIZE, 1);

    size_t tokens_to_compress = tokens.size() - WINDOW_SIZE;
    long correct_predictions = 0;
    auto comp_start = std::chrono::steady_clock::now();

    for (size_t i = WINDOW_SIZE; i < tokens.size() && !signal_handler::is_triggered(); ++i) {
        for (int j = 0; j < WINDOW_SIZE; ++j) window(j) = tokens[i - WINDOW_SIZE + j];
        int predicted = net(window);

        if (predicted == tokens[i]) {
            bit_writer.write_bit(true);
            correct_predictions++;
        }
        else {
            bit_writer.write_bit(false);
            bit_writer.write_token(tokens[i], token_bits);
        }

        size_t done = i - WINDOW_SIZE + 1;
        if (done % 2000 == 0) try_show_progress("Compressing", done, tokens_to_compress, comp_start);
    }
    bit_writer.flush();

    if (signal_handler::is_triggered()) {
        cout << "\n[!] Compression interrupted. Output file NOT saved." << endl;
        signal_handler::reset();
        return;
    }

    // Entropy-encode the bitstream (same as byte-level compressor)
    std::string raw_bits(compressed_data.begin(), compressed_data.end());
    std::istringstream raw_input(raw_bits);
    std::ostringstream encoded_output;
    entropy_codec codec;
    codec.compress(raw_input, encoded_output);
    std::string encoded_str = encoded_output.str();
    std::vector<uint8_t> entropy_data(encoded_str.begin(), encoded_str.end());

    cout << "\r  Compressing [##############################] 100.0%  "
        << format_duration(std::chrono::duration<double>(std::chrono::steady_clock::now() - comp_start).count())
        << "                    " << endl;
    double accuracy = tokens_to_compress > 0 ? (double)correct_predictions / tokens_to_compress * 100.0 : 0;
    cout << "Bitstream before entropy: " << format_size(compressed_data.size()) << endl;
    cout << "Bitstream after entropy:  " << format_size(entropy_data.size())
        << "  " << format_ratio(entropy_data.size(), compressed_data.size()) << endl;
    cout << "Prediction accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << endl << endl;
    compressed_data.swap(entropy_data);

    // Write output file (same layout as byte-level: header + model + seed + data)
    uint64_t original_size = data.size();
    uint64_t num_tokens_total = tokens.size();
    uint8_t flags = embed_model ? FLAG_MODEL_EMBEDDED : 0;

    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) throw std::runtime_error("Cannot create output file: " + output_path);

    out_file.write(MAGIC, sizeof(MAGIC));
    serialize(original_size, out_file);  serialize(crc, out_file);
    serialize(num_tokens_total, out_file); serialize(flags, out_file);

    std::streampos pos_before_model = out_file.tellp();
    if (embed_model) { serialize(tokenizer, out_file); serialize(net, out_file); }
    std::streampos pos_after_model = out_file.tellp();
    size_t model_size = static_cast<size_t>(pos_after_model - pos_before_model);

    serialize(literal_seed, out_file);
    serialize(compressed_data, out_file);

    size_t final_size = static_cast<size_t>(out_file.tellp());
    out_file.close();

    size_t data_payload = final_size - model_size;
    cout << "=== COMPRESSION SUMMARY ===" << endl;
    cout << "Original: " << format_size(data.size()) << "  Tokens: " << tokens.size()
        << " (" << std::fixed << std::setprecision(1) << (double)data.size() / tokens.size() << " B/tok)" << endl;
    cout << "Data payload: " << format_size(data_payload) << "  " << format_ratio(data_payload, data.size()) << endl;
    if (embed_model) {
        cout << "Embedded model+tok: " << format_size(model_size) << "  Total: " << format_size(final_size)
            << "  vs original: " << format_ratio(final_size, data.size()) << endl;
    }
    else {
        cout << "Model: external (" << MODEL_SAVE_FILE << ")  Total: " << format_size(final_size)
            << "  " << format_ratio(final_size, data.size()) << endl;
    }
    cout << "===========================" << endl;
    cout << "Compressed to: " << output_path << endl;
}

void decompress_file(const std::string& input_path, const std::string& output_path)
{
    cout << "=== DECOMPRESSION MODE ===" << endl;
    cout << "Input: " << input_path << "  Output: " << output_path << endl << endl;

    std::ifstream in_file(input_path, std::ios::binary);
    if (!in_file) throw std::runtime_error("Cannot open file: " + input_path);

    char magic[3];
    in_file.read(magic, sizeof(magic));
    if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0)
        throw std::runtime_error("Invalid file format (magic mismatch, expected 'DLC').");

    uint64_t original_size;  deserialize(original_size, in_file);
    uint32_t stored_crc;     deserialize(stored_crc, in_file);
    uint64_t num_tokens;     deserialize(num_tokens, in_file);
    uint8_t flags;           deserialize(flags, in_file);
    bool model_embedded = (flags & FLAG_MODEL_EMBEDDED) != 0;

    cout << "Original: " << format_size(static_cast<size_t>(original_size))
        << "  CRC32: " << std::hex << std::setfill('0') << std::setw(8) << stored_crc << std::dec
        << "  Tokens: " << num_tokens << "  Model: " << (model_embedded ? "embedded" : "external") << endl;

    bpe_tokenizer tokenizer;
    infer_net net;
    if (model_embedded) {
        cout << "Loading embedded tokenizer + model..." << endl;
        deserialize(tokenizer, in_file); deserialize(net, in_file);
    }
    else {
        if (!file_exists(MODEL_SAVE_FILE)) throw std::runtime_error("External model not found: " + MODEL_SAVE_FILE);
        cout << "Loading external model: " << MODEL_SAVE_FILE << endl;
        std::ifstream model_in(MODEL_SAVE_FILE, std::ios::binary);
        deserialize(tokenizer, model_in);
        train_net t_net; deserialize(t_net, model_in);
        net = infer_net(t_net);
    }

    long effective_vocab = static_cast<long>(tokenizer.get_vocab_size());
    if (effective_vocab > MAX_VOCAB_SIZE) effective_vocab = MAX_VOCAB_SIZE;

    int token_bits = 0;
    { long v = effective_vocab - 1; while (v > 0) { token_bits++; v >>= 1; } }

    std::vector<int> literal_seed;
    deserialize(literal_seed, in_file);
    std::vector<uint8_t> compressed_data;
    deserialize(compressed_data, in_file);
    in_file.close();

    cout << "Tokenizer: vocab=" << tokenizer.get_vocab_size() << "  Token bits: " << token_bits << endl;
    cout << "Seed: " << literal_seed.size() << " tokens  Compressed: " << format_size(compressed_data.size()) << endl;

    // Entropy-decode the bitstream (reverse of compression)
    std::string encoded_bits(compressed_data.begin(), compressed_data.end());
    std::istringstream encoded_input(encoded_bits);
    std::ostringstream decoded_output;
    entropy_codec codec;
    codec.decompress(encoded_input, decoded_output);
    std::string decoded_str = decoded_output.str();
    compressed_data.assign(decoded_str.begin(), decoded_str.end());
    cout << "Entropy-decoded bitstream: " << format_size(compressed_data.size()) << endl << endl;

    // Reconstruct token stream (same logic as byte-level, but with token IDs)
    std::vector<int> decoded_tokens = literal_seed;
    decoded_tokens.reserve(static_cast<size_t>(num_tokens));
    matrix<int, 0, 1> window(WINDOW_SIZE, 1);
    in_bit_stream bit_reader(compressed_data);

    size_t tokens_to_decode = static_cast<size_t>(num_tokens) - literal_seed.size();
    auto decomp_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < tokens_to_decode && !signal_handler::is_triggered(); ++i) {
        size_t pos = literal_seed.size() + i;
        for (int j = 0; j < WINDOW_SIZE; ++j) window(j) = decoded_tokens[pos - WINDOW_SIZE + j];

        int predicted = net(window);
        bool success = bit_reader.read_bit();

        if (success) {
            decoded_tokens.push_back(predicted);
        }
        else {
            int actual = bit_reader.read_token(token_bits);
            decoded_tokens.push_back(actual);
        }

        if ((i + 1) % 2000 == 0) try_show_progress("Decompressing", i + 1, tokens_to_decode, decomp_start);
    }

    if (signal_handler::is_triggered()) {
        cout << "\n[!] Decompression interrupted. Output file NOT saved." << endl;
        signal_handler::reset();
        return;
    }

    auto decomp_end = std::chrono::steady_clock::now();
    double decomp_secs = std::chrono::duration<double>(decomp_end - decomp_start).count();
    cout << "\r  Decompressing [##############################] 100.0%  " << format_duration(decomp_secs) << "                    " << endl;

    cout << "Detokenizing " << decoded_tokens.size() << " tokens..." << endl;
    std::string restored_text = parallel_decode(tokenizer, decoded_tokens);

    std::vector<uint8_t> restored_bytes(restored_text.begin(), restored_text.end());
    uint32_t computed_crc = compute_crc32(restored_bytes);
    if (computed_crc != stored_crc) {
        cerr << "WARNING: CRC32 mismatch! Expected: " << std::hex << std::setw(8) << std::setfill('0')
            << stored_crc << "  Computed: " << std::setw(8) << computed_crc << std::dec << endl;
    }
    else {
        cout << "CRC32 verification: OK" << endl;
    }

    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) throw std::runtime_error("Cannot create output file: " + output_path);
    out_file.write(restored_text.data(), restored_text.size());
    out_file.close();

    cout << "\n=== DECOMPRESSION SUMMARY ===" << endl;
    cout << "Restored: " << format_size(restored_bytes.size()) << "  Time: " << format_duration(decomp_secs) << endl;
    cout << "Output: " << output_path << endl;
}

int main(int argc, char** argv)
{
    try
    {
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("compress", "Compress a file");
        parser.add_option("decompress", "Decompress a file");
        parser.add_option("input", "Input file path", 1);
        parser.add_option("output", "Output file path (optional)", 1);
        parser.add_option("no-train", "Skip training, reuse previously saved model");
        parser.add_option("no-embed-model", "Do not embed the model in the compressed file");
        parser.parse(argc, argv);

        if (!parser.option("compress") && !parser.option("decompress")) {
            cout << "Dlib predictive compressor (BPE + Transformer + Arithmetic Coding)\n\n";
            parser.print_options();
            cout << "\nExamples:\n"
                << "  Compress:   " << argv[0] << " --compress --input data.txt\n"
                << "  Reuse:      " << argv[0] << " --compress --input data.txt --no-train --no-embed-model\n"
                << "  Decompress: " << argv[0] << " --decompress --input data.txt.dlc\n";
            return 0;
        }

        if (parser.option("compress") && parser.option("decompress"))
            throw std::runtime_error("Cannot specify both --compress and --decompress");
        if (!parser.option("input"))
            throw std::runtime_error("Missing --input parameter");

        std::string input_path = parser.option("input").argument();
        std::string output_path;
        if (parser.option("output")) output_path = parser.option("output").argument();
        else output_path = parser.option("compress") ? input_path + ".dlc" : input_path + ".restored";

        if (parser.option("compress")) {
            bool do_train = !parser.option("no-train");
            bool embed_model = !parser.option("no-embed-model");
            compress_file(input_path, output_path, do_train, embed_model);
        }
        else {
            decompress_file(input_path, output_path);
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "\nERROR: " << e.what() << endl;
        return 1;
    }
}