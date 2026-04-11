/*!
    @file slm_predictive_compressor_ex.cpp
    @brief Byte-level predictive compression using a Transformer.

    This example demonstrates an advanced application of AI generative models beyond
    traditional chatbot use cases. It implements a compression/decompression system
    for any file type using a small Transformer model to predict the next byte.

    The compression scheme uses a single bit to indicate prediction success, reducing
    data size when the predictor is accurate. Actual bit-packing is used to ensure
    physical file size reduction.

    Key features:
    - True bit-packing stream to physically shrink the serialized file.
    - Eliminates padding entirely: the model only operates on fully populated context windows.
    - The first WINDOW_SIZE bytes are stored uncompressed to prime the prediction engine.
    - Graceful interruption (CTRL+C) via dlib::signal_handler with training checkpoints.
    - Early-stopping: interrupting training continues directly to compression.
    - CRC32 integrity verification on decompression.
    - Model persistence: the trained model is saved to disk and optionally embedded in the
      compressed file so that decompression is fully self-contained.
    - Skip-training option: reuse a previously saved model without retraining.
    - No-embed option: produce a smaller compressed file without the model (requires the
      external model file for decompression).
    - Text-mode progress bars with throughput statistics.
    - Training data capped at MAX_TRAINING_BYTES (350 MB) to limit memory and time;
      compression and decompression always process the full file.
    - Memory-efficient training: mini-batches are sampled on-the-fly from raw byte data,
      avoiding pre-building the full dataset in memory.

    Compressed file format:
    [MAGIC_NUMBER 4B] [original_size 8B] [CRC32 4B] [flags 1B]
    [serialized model if embedded] [literal_seed] [compressed_data]

    Flags byte:
    - bit 0: model is embedded (1) or external (0)

    Usage:
    --compress   --input <file> [--output <file>]
    --compress   --input <file> --no-train            (skip training, reuse saved model)
    --compress   --input <file> --no-embed-model       (do not embed model in output)
    --decompress --input <file> [--output <file>]
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

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/crc32.h>
#include <dlib/compress_stream.h>

using namespace std;
using namespace dlib;

// Constants & configuration
const uint32_t MAGIC_NUMBER = 0x444C4943;              // "DLIC" in big-endian
const int WINDOW_SIZE = 16;                            // Fixed prediction window size
const long MAX_VOCAB_SIZE = 256;                       // Exact byte range (0-255)
const std::string MODEL_SAVE_FILE = "dlib_predictive_compressor.dat";
const size_t MAX_TRAINING_BYTES = 350 * 1024 * 1024;   // 350 MB cap for training data
const int BATCH_SIZE = 128;                            // Mini-batch size for training

// Flags byte layout
const uint8_t FLAG_MODEL_EMBEDDED = 0x01;              // bit 0: model is embedded

// Network architecture parameters
const long NUM_LAYERS = 2;
const long NUM_HEADS = 4;
const long EMBEDDING_DIM = 32;

// Transformer configuration: single network type for both training and inference
using compressor_transformer = fused_transformer_config<
    MAX_VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, EMBEDDING_DIM>;

using train_net = compressor_transformer::network_type<true>;
using infer_net = compressor_transformer::network_type<false>;

// Utility functions
typedef dlib::compress_stream::kernel_1ec entropy_codec;

std::string format_duration(double seconds)
{
    int hours = static_cast<int>(seconds / 3600);
    int minutes = static_cast<int>((seconds - hours * 3600) / 60);
    int secs = static_cast<int>(seconds - hours * 3600 - minutes * 60);

    std::ostringstream oss;
    if (hours > 0)
        oss << hours << "h " << minutes << "m " << secs << "s";
    else if (minutes > 0)
        oss << minutes << "m " << secs << "s";
    else
        oss << secs << "s";
    return oss.str();
}

std::string format_throughput(double bytes_per_sec)
{
    if (bytes_per_sec >= 1048576)
        return std::to_string(static_cast<int>(bytes_per_sec / 1048576)) + " MB/s";
    else if (bytes_per_sec >= 1024)
        return std::to_string(static_cast<int>(bytes_per_sec / 1024)) + " KB/s";
    else
        return std::to_string(static_cast<int>(bytes_per_sec)) + " B/s";
}

std::string format_size(size_t bytes)
{
    std::ostringstream oss;
    if (bytes >= 1048576)
        oss << std::fixed << std::setprecision(2) << (bytes / 1048576.0) << " MB";
    else if (bytes >= 1024)
        oss << std::fixed << std::setprecision(1) << (bytes / 1024.0) << " KB";
    else
        oss << bytes << " B";
    return oss.str();
}

// Format a compression ratio: positive = reduction, negative = expansion.
// Example: 23000 -> 7000 yields "+69.57%", 23000 -> 30000 yields "-30.43%"
std::string format_ratio(size_t compressed, size_t original)
{
    if (original == 0) return "N/A";
    double reduction = (1.0 - static_cast<double>(compressed) / original) * 100.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (reduction >= 0)
        oss << "+" << reduction << "%";
    else
        oss << reduction << "%";
    return oss.str();
}

void show_progress(const std::string& label, size_t current, size_t total,
    std::chrono::steady_clock::time_point start_time)
{
    double percent = (static_cast<double>(current) / total) * 100.0;
    int bar_width = 30;
    int filled = static_cast<int>(percent * bar_width / 100.0);

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    double speed = (elapsed > 0) ? current / elapsed : 0;

    std::string eta_str = "---";
    if (current > 0 && current < total) {
        double eta = (total - current) / speed;
        eta_str = format_duration(eta);
    }

    cout << "\r" << label << " [";
    for (int i = 0; i < bar_width; ++i)
        cout << (i < filled ? '#' : '.');
    cout << "] " << std::fixed << std::setprecision(1) << percent << "% "
        << format_throughput(speed)
        << " ETA: " << eta_str << "     " << std::flush;

    if (current == total) cout << "\n";
}

// Bit stream classes
class out_bit_stream {
    std::vector<uint8_t>& buffer;
    uint8_t current_byte = 0;
    int bits_count = 0;
public:
    out_bit_stream(std::vector<uint8_t>& buf) : buffer(buf) {}

    void write_bit(bool b) {
        if (b) current_byte |= (1 << bits_count);
        bits_count++;
        if (bits_count == 8) {
            buffer.push_back(current_byte);
            current_byte = 0;
            bits_count = 0;
        }
    }

    void write_byte(uint8_t b) {
        for (int i = 0; i < 8; ++i)
            write_bit((b >> i) & 1);
    }

    void flush() {
        if (bits_count > 0) {
            buffer.push_back(current_byte);
            current_byte = 0;
            bits_count = 0;
        }
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
        if (bits_count == 8) {
            byte_idx++;
            bits_count = 0;
        }
        return b;
    }

    uint8_t read_byte() {
        uint8_t b = 0;
        for (int i = 0; i < 8; ++i) {
            if (read_bit())
                b |= (1 << i);
        }
        return b;
    }
};

// File I/O
std::vector<uint8_t> read_file_bytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
}

uint32_t compute_crc32(const std::vector<uint8_t>& data)
{
    return dlib::crc32(std::string(data.begin(), data.end()));
}

// Training with on-the-fly batch sampling
// Builds a single mini-batch by sampling random positions from the raw byte stream.
// No full dataset is ever materialized: memory usage is O(batch_size), not O(file_size).
void sample_batch(
    const std::vector<uint8_t>& data,
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
    for (int b = 0; b < BATCH_SIZE; ++b)
    {
        size_t target_pos = dist(rng);
        long pos = 0;
        for (size_t j = target_pos - WINDOW_SIZE; j < target_pos; ++j)
            window(pos++) = data[j];

        batch_samples.push_back(window);
        batch_labels.push_back(data[target_pos]);
    }
}

void train_predictor_model(train_net& net, const std::vector<uint8_t>& data,
    const std::string& input_path)
{
    size_t training_size = data.size();
    if (training_size > MAX_TRAINING_BYTES) {
        training_size = MAX_TRAINING_BYTES;
        cout << "File exceeds training limit (" << format_size(MAX_TRAINING_BYTES)
            << "). Training on first " << format_size(training_size) << " only." << endl;
    }

    size_t num_transitions = training_size - WINDOW_SIZE;
    if (num_transitions == 0) return;

    cout << "Training transitions available: " << num_transitions << endl;
    cout << "Mini-batch size: " << BATCH_SIZE << endl;

    dnn_trainer<train_net, adam> trainer(net, adam(0.004, 0.9, 0.999));
    trainer.set_learning_rate(1e-4);
    trainer.set_min_learning_rate(1e-8);
    trainer.set_iterations_without_progress_threshold(8000);

    std::string checkpoint_file = input_path + ".chkpt";
    trainer.set_synchronization_file(checkpoint_file, std::chrono::minutes(1));
    trainer.be_quiet();

    cout << "Checkpoint file: " << checkpoint_file << endl;
    cout << "Press CTRL+C to interrupt safely and proceed to compression.\n" << endl;

    std::mt19937 rng(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<matrix<int, 0, 1>> batch_samples;
    std::vector<unsigned long> batch_labels;

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate() && !signal_handler::is_triggered())
    {
        sample_batch(data, training_size, rng, batch_samples, batch_labels);
        trainer.train_one_step(batch_samples, batch_labels);

        if (trainer.get_train_one_step_calls() % 50 == 0) {
            cout << "Step: " << trainer.get_train_one_step_calls()
                << " | Loss: " << trainer.get_average_loss()
                << " | LR: " << trainer.get_learning_rate() << "\r";
            cout.flush();
        }
    }

    if (signal_handler::is_triggered()) {
        cout << "\n\n[!] Training interrupted by user. State saved to: " << checkpoint_file << endl;
    }
    else {
        cout << "\nTraining complete." << endl;
    }

    trainer.get_net();
    net.clean();
}

// Compression
void compress_file(const std::string& input_path, const std::string& output_path,
    bool do_train, bool embed_model)
{
    cout << "=== COMPRESSION MODE ===" << endl;
    cout << "Input file   : " << input_path << endl;
    cout << "Output file  : " << output_path << endl;
    cout << "Training     : " << (do_train ? "Yes" : "No (reuse saved model)") << endl;
    cout << "Embed model  : " << (embed_model ? "Yes" : "No (external model required for decompression)") << endl;
    cout << endl;

    // Read input
    std::vector<uint8_t> data = read_file_bytes(input_path);
    cout << "Input file size: " << format_size(data.size()) << " (" << data.size() << " bytes)" << endl;

    if (data.size() <= static_cast<size_t>(WINDOW_SIZE))
        throw std::runtime_error("File is too small to compress (must be > WINDOW_SIZE).");

    // CRC32 of original data
    uint32_t crc = compute_crc32(data);
    cout << "CRC32: " << std::hex << std::setfill('0') << std::setw(8) << crc << std::dec << endl;
    cout << endl;

    // Train or load model
    train_net t_net;
    bool model_exists = file_exists(MODEL_SAVE_FILE);

    if (do_train) {
        if (model_exists) {
            cout << "Loading existing model for fine-tuning: " << MODEL_SAVE_FILE << endl;
            deserialize(MODEL_SAVE_FILE) >> t_net;
        }
        else {
            cout << "Training new model from scratch." << endl;
        }

        train_predictor_model(t_net, data, input_path);

        if (signal_handler::is_triggered()) {
            cout << "Training interrupted. Proceeding to compression with partial model..." << endl;
            signal_handler::reset();
        }

        serialize(MODEL_SAVE_FILE) << t_net;
        cout << "Model saved to: " << MODEL_SAVE_FILE << endl;
    }
    else {
        if (!model_exists)
            throw std::runtime_error("No saved model found (" + MODEL_SAVE_FILE + "). Run without --no-train first.");

        cout << "Loading saved model: " << MODEL_SAVE_FILE << endl;
        deserialize(MODEL_SAVE_FILE) >> t_net;
        t_net.clean();
    }

    // Build inference network
    infer_net net(t_net);

    // Compress data stream (always processes the full file)
    cout << "\nCompressing data stream..." << endl;

    std::vector<uint8_t> literal_seed(data.begin(), data.begin() + WINDOW_SIZE);
    std::vector<uint8_t> compressed_data;
    out_bit_stream bit_writer(compressed_data);

    matrix<int, 0, 1> window(WINDOW_SIZE, 1);
    long bits_saved = 0;
    long total_predictions = 0;
    long correct_predictions = 0;

    size_t tokens_to_compress = data.size() - WINDOW_SIZE;
    auto comp_start = std::chrono::steady_clock::now();

    for (size_t i = WINDOW_SIZE; i < data.size() && !signal_handler::is_triggered(); ++i)
    {
        long pos = 0;
        for (long j = i - WINDOW_SIZE; j < static_cast<long>(i); ++j)
            window(pos++) = data[j];

        int predicted_byte = net(window);
        total_predictions++;

        if (predicted_byte == data[i]) {
            bit_writer.write_bit(true);
            bits_saved += 7;
            correct_predictions++;
        }
        else {
            bit_writer.write_bit(false);
            bit_writer.write_byte(data[i]);
            bits_saved -= 1;
        }

        if ((i - WINDOW_SIZE) % 500 == 0 || i == data.size() - 1)
            show_progress("Compressing", i - WINDOW_SIZE + 1, tokens_to_compress, comp_start);
    }

    bit_writer.flush();

    // Entropy-encode the bitstream for additional compression
    std::string raw_bits(compressed_data.begin(), compressed_data.end());
    std::istringstream raw_input(raw_bits);
    std::ostringstream encoded_output;
    
    entropy_codec codec;
    codec.compress(raw_input, encoded_output);

    std::string encoded_str = encoded_output.str();
    std::vector<uint8_t> entropy_data(encoded_str.begin(), encoded_str.end());

    // Report entropy gain
    cout << "Bitstream before entropy coding: " << format_size(compressed_data.size()) << endl;
    cout << "Bitstream after entropy coding : " << format_size(entropy_data.size())
        << "  " << format_ratio(entropy_data.size(), compressed_data.size()) << endl;

    // Replace compressed_data with entropy-coded version for serialization
    compressed_data.swap(entropy_data);

    auto comp_end = std::chrono::steady_clock::now();
    double comp_seconds = std::chrono::duration<double>(comp_end - comp_start).count();

    if (signal_handler::is_triggered()) {
        cout << "\n[!] Compression interrupted. Output file will NOT be saved to avoid corruption." << endl;
        return;
    }

    // Prediction stats
    double accuracy = (total_predictions > 0)
        ? (correct_predictions * 100.0 / total_predictions) : 0.0;
    cout << "Prediction accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << endl;
    cout << "Encoding time: " << format_duration(comp_seconds)
        << " (" << format_throughput(data.size() / comp_seconds) << ")" << endl;

    // Write output file
    // Format: [MAGIC 4B] [original_size 8B] [CRC32 4B] [flags 1B]
    //         [model if embedded] [literal_seed] [compressed_data]
    uint64_t original_size = data.size();
    uint8_t flags = embed_model ? FLAG_MODEL_EMBEDDED : 0;

    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) throw std::runtime_error("Cannot create output file: " + output_path);

    serialize(MAGIC_NUMBER, out_file);
    serialize(original_size, out_file);
    serialize(crc, out_file);
    serialize(flags, out_file);

    // Record position before model to measure its serialized size
    std::streampos pos_before_model = out_file.tellp();
    if (embed_model) {
        serialize(net, out_file);
    }
    std::streampos pos_after_model = out_file.tellp();
    size_t model_size_in_file = static_cast<size_t>(pos_after_model - pos_before_model);

    serialize(literal_seed, out_file);
    serialize(compressed_data, out_file);

    std::streampos final_pos = out_file.tellp();
    size_t final_size = static_cast<size_t>(final_pos);
    out_file.close();

    // Compute sizes for summary
    // Header = magic(4) + original_size(8) + crc(4) + flags(1) = 17 bytes
    const size_t header_size = 17;
    size_t data_payload_size = final_size - model_size_in_file;

    cout << "\n=== COMPRESSION SUMMARY ===" << endl;
    cout << "Original size          : " << format_size(data.size())
        << " (" << data.size() << " bytes)" << endl;
    cout << "Prediction accuracy    : " << std::fixed << std::setprecision(2) << accuracy << "%" << endl;
    cout << "Data payload           : " << format_size(data_payload_size)
        << "  compression: " << format_ratio(data_payload_size, data.size()) << endl;
    if (embed_model) {
        cout << "Embedded model         : " << format_size(model_size_in_file) << endl;
        cout << "Total file (w/ model)  : " << format_size(final_size)
            << "  vs original: " << format_ratio(final_size, data.size()) << endl;
    }
    else {
        cout << "Model                  : external (" << MODEL_SAVE_FILE << ")" << endl;
        cout << "Total file (no model)  : " << format_size(final_size)
            << "  compression: " << format_ratio(final_size, data.size()) << endl;
    }
    cout << "===========================" << endl;
    cout << "File compressed to: " << output_path << endl;
}

// Decompression
void decompress_file(const std::string& input_path, const std::string& output_path)
{
    cout << "=== DECOMPRESSION MODE ===" << endl;
    cout << "Input file : " << input_path << endl;
    cout << "Output file: " << output_path << endl;
    cout << endl;

    std::ifstream in_file(input_path, std::ios::binary);
    if (!in_file) throw std::runtime_error("Cannot open file: " + input_path);

    // Read header
    uint32_t magic;
    deserialize(magic, in_file);
    if (magic != MAGIC_NUMBER)
        throw std::runtime_error("Invalid file format (Magic Number mismatch).");

    uint64_t original_size;
    deserialize(original_size, in_file);

    uint32_t stored_crc;
    deserialize(stored_crc, in_file);

    uint8_t flags;
    deserialize(flags, in_file);

    bool model_embedded = (flags & FLAG_MODEL_EMBEDDED) != 0;

    cout << "Original file size: " << format_size(static_cast<size_t>(original_size)) << endl;
    cout << "Stored CRC32: " << std::hex << std::setfill('0') << std::setw(8) << stored_crc
        << std::dec << endl;
    cout << "Model: " << (model_embedded ? "embedded" : "external") << endl;

    // Load model
    infer_net net;
    if (model_embedded) {
        cout << "Loading embedded model..." << endl;
        deserialize(net, in_file);
    }
    else {
        cout << "Loading external model: " << MODEL_SAVE_FILE << endl;
        if (!file_exists(MODEL_SAVE_FILE))
            throw std::runtime_error("External model file not found: " + MODEL_SAVE_FILE);
        deserialize(MODEL_SAVE_FILE) >> net;
    }

    std::vector<uint8_t> literal_seed;
    deserialize(literal_seed, in_file);

    std::vector<uint8_t> compressed_data;
    deserialize(compressed_data, in_file);
    in_file.close();

    cout << "Compressed data: " << format_size(compressed_data.size()) << endl;

    // Entropy-decode the bitstream
    std::string encoded_bits(compressed_data.begin(), compressed_data.end());
    std::istringstream encoded_input(encoded_bits);
    std::ostringstream decoded_output;

    entropy_codec codec;
    codec.decompress(encoded_input, decoded_output);

    std::string decoded_str = decoded_output.str();
    compressed_data.assign(decoded_str.begin(), decoded_str.end());

    cout << "Entropy-decoded bitstream: " << format_size(compressed_data.size()) << endl;

    // Reconstruct byte stream
    cout << "\nRestoring byte stream..." << endl;

    std::vector<uint8_t> decompressed_data = literal_seed;
    decompressed_data.reserve(original_size);
    matrix<int, 0, 1> window(WINDOW_SIZE, 1);
    in_bit_stream bit_reader(compressed_data);

    size_t tokens_to_decode = original_size - literal_seed.size();
    auto decomp_start = std::chrono::steady_clock::now();

    while (decompressed_data.size() < original_size && !signal_handler::is_triggered())
    {
        long pos = 0;
        for (long j = decompressed_data.size() - WINDOW_SIZE;
            j < static_cast<long>(decompressed_data.size()); ++j)
            window(pos++) = decompressed_data[j];

        int predicted_byte = net(window);

        bool success = bit_reader.read_bit();
        if (success) {
            decompressed_data.push_back(predicted_byte);
        }
        else {
            uint8_t actual_byte = bit_reader.read_byte();
            decompressed_data.push_back(actual_byte);
        }

        size_t decoded_so_far = decompressed_data.size() - literal_seed.size();
        if (decoded_so_far % 500 == 0 || decompressed_data.size() == original_size)
            show_progress("Decompressing", decoded_so_far, tokens_to_decode, decomp_start);
    }

    auto decomp_end = std::chrono::steady_clock::now();
    double decomp_seconds = std::chrono::duration<double>(decomp_end - decomp_start).count();

    if (signal_handler::is_triggered()) {
        cout << "\n[!] Decompression interrupted. Output file will NOT be saved to avoid corruption." << endl;
        return;
    }

    // CRC32 verification
    uint32_t computed_crc = compute_crc32(decompressed_data);
    if (computed_crc != stored_crc) {
        cerr << "\nWARNING: CRC32 mismatch! Data may be corrupted." << endl;
        cerr << "  Expected: " << std::hex << std::setw(8) << std::setfill('0') << stored_crc << endl;
        cerr << "  Computed: " << std::hex << std::setw(8) << std::setfill('0') << computed_crc << endl;
        cerr << std::dec;
    }
    else {
        cout << "CRC32 verification: OK" << endl;
    }

    // Write output
    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) throw std::runtime_error("Cannot create output file: " + output_path);
    out_file.write(reinterpret_cast<const char*>(decompressed_data.data()), decompressed_data.size());
    out_file.close();

    cout << "\n=== DECOMPRESSION SUMMARY ===" << endl;
    cout << "Restored: " << format_size(decompressed_data.size())
        << " (" << decompressed_data.size() << " bytes)" << endl;
    cout << "Time: " << format_duration(decomp_seconds)
        << " (" << format_throughput(decompressed_data.size() / decomp_seconds) << ")" << endl;
    cout << "=============================" << endl;
    cout << "Output file: " << output_path << endl;
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
            cout << "Dlib predictive compressor\n\n";
            parser.print_options();
            cout << "\nExamples:\n";
            cout << "  Compress with training (model embedded):\n";
            cout << "    " << argv[0] << " --compress --input data.txt --output data.dpc\n\n";
            cout << "  Compress without embedding model (smaller output):\n";
            cout << "    " << argv[0] << " --compress --input data.txt --no-embed-model\n\n";
            cout << "  Compress without training (reuse model):\n";
            cout << "    " << argv[0] << " --compress --input data.txt --no-train\n\n";
            cout << "  Decompress:\n";
            cout << "    " << argv[0] << " --decompress --input data.dpc --output data_restored.txt\n";
            return 0;
        }

        if (parser.option("compress") && parser.option("decompress"))
            throw std::runtime_error("Cannot specify both --compress and --decompress");

        if (!parser.option("input"))
            throw std::runtime_error("Missing --input parameter");

        std::string input_path = parser.option("input").argument();
        std::string output_path;

        if (parser.option("output")) {
            output_path = parser.option("output").argument();
        }
        else {
            output_path = parser.option("compress")
                ? input_path + ".dpc"
                : input_path + ".restored";
            cout << "No output specified. Defaulting to: " << output_path << "\n\n";
        }

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