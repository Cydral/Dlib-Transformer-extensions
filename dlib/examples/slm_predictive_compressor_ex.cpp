/*!
    @file slm_predictive_compressor_ex.cpp
    @brief Byte-level predictive compression using a GQA Transformer.

    This example demonstrates an advanced application of AI generative models beyond
    traditional chatbot use cases. It implements a compression/decompression system
    for any file type using a small Transformer model to predict the next byte.

    The compression scheme uses a single bit to indicate prediction success, reducing
    data size when the predictor is accurate. Actual bit-packing is used to ensure
    physical file size reduction.

    Key features:
    - True bit-packing stream to physically shrink the serialized file.
    - Uses Grouped Query Attention (GQA) for reduced parameter count and faster inference.
    - Eliminates padding entirely: the model only operates on fully populated context windows.
    - The first WINDOW_SIZE bytes are stored uncompressed to prime the prediction engine.
    - Graceful interruption (CTRL+C) via dlib::signal_handler with training checkpoints.
    - Early-stopping: interrupting training continues directly to compression.
    - CRC32 integrity verification on decompression.
    - Model persistence: the trained model is saved to disk and embedded in the compressed
      file so that decompression is fully self-contained.
    - Skip-training option: reuse a previously saved model without retraining.
    - Text-mode progress bars with throughput statistics.
    - Training data capped at MAX_TRAINING_BYTES (150 MB) to limit memory and time;
      compression and decompression always process the full file.

    Compressed file format:
    [MAGIC_NUMBER  4B] [original_size  8B] [CRC32  4B]
    [serialized model (dlib serialize)] [literal_seed] [compressed_data]

    Usage:
    --compress   --input <file> [--output <file>]
    --compress   --input <file> --no-train          (skip training, reuse saved model)
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

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/crc32.h>

#include <dlib/dnn/transformer_config.h>
#include <dlib/dnn/transformer.h>

using namespace std;
using namespace dlib;

// Constants & Configuration
const uint32_t MAGIC_NUMBER = 0x444C4943;           // "DLIC" in big-endian
const int WINDOW_SIZE = 10;                         // Fixed prediction window size
const long MAX_VOCAB_SIZE = 256;                    // Exact byte range (0-255)
const std::string MODEL_SAVE_FILE = "dlib_predictive_compressor.dat";
const size_t MAX_TRAINING_BYTES = 150 * 1024 * 1024;  // 150 MB cap for training data

// GQA Network architecture parameters
const long NUM_LAYERS = 2;
const long NUM_HEADS = 4;
const long NUM_KV_HEADS = NUM_HEADS;                // Full MHA (set < NUM_HEADS for true GQA)
const long EMBEDDING_DIM = 16;

// GQA Transformer configuration: single network type for both training and inference
using compressor_transformer = gqa_transformer_config<
    MAX_VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, EMBEDDING_DIM>;

using train_net = compressor_transformer::network_type<true>;
using infer_net = compressor_transformer::network_type<false>;

// Utility Functions
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

// Bit Stream Classes
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

// Training
void train_predictor_model(train_net& net, const std::vector<uint8_t>& data,
    const std::string& input_path)
{
    // Cap training data to MAX_TRAINING_BYTES to limit memory and training time.
    // The full file is always used for compression/decompression regardless.
    size_t training_size = data.size();
    if (training_size > MAX_TRAINING_BYTES) {
        training_size = MAX_TRAINING_BYTES;
        cout << "File exceeds training limit (" << format_size(MAX_TRAINING_BYTES)
            << "). Training on first " << format_size(training_size) << " only." << endl;
    }

    cout << "Preparing dataset for training (no padding)..." << endl;

    std::vector<matrix<int, 0, 1>> samples;
    std::vector<unsigned long> labels;

    matrix<int, 0, 1> window(WINDOW_SIZE, 1);
    for (size_t i = WINDOW_SIZE; i < training_size; ++i)
    {
        long pos = 0;
        for (long j = i - WINDOW_SIZE; j < static_cast<long>(i); ++j)
            window(pos++) = data[j];

        samples.push_back(window);
        labels.push_back(data[i]);
    }

    if (samples.empty()) return;

    dnn_trainer<train_net, adam> trainer(net, adam(0.01, 0.9, 0.999));
    trainer.set_learning_rate(1e-3);
    trainer.set_min_learning_rate(1e-5);
    trainer.set_mini_batch_size(64);

    std::string checkpoint_file = input_path + ".chkpt";
    trainer.set_synchronization_file(checkpoint_file, std::chrono::minutes(1));
    trainer.be_quiet();

    cout << "Training predictor on " << samples.size() << " full-context transitions..." << endl;
    cout << "Checkpoint file: " << checkpoint_file << endl;
    cout << "Press CTRL+C to interrupt safely and proceed to compression.\n" << endl;

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate() && !signal_handler::is_triggered())
    {
        trainer.train_one_step(samples, labels);
        if (trainer.get_train_one_step_calls() % 50 == 0) {
            cout << "  Step: " << trainer.get_train_one_step_calls()
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
void compress_file(const std::string& input_path, const std::string& output_path, bool do_train)
{
    cout << "=== COMPRESSION MODE ===" << endl;
    cout << "Input file : " << input_path << endl;
    cout << "Output file: " << output_path << endl;
    cout << "Training   : " << (do_train ? "Yes" : "No (reuse saved model)") << endl;
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

        // Continue compression even if training was interrupted
        if (signal_handler::is_triggered()) {
            cout << "Training interrupted. Proceeding to compression with partial model..." << endl;
            signal_handler::reset();
        }

        // Save trained model
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
    uint64_t original_size = data.size();

    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file) throw std::runtime_error("Cannot create output file: " + output_path);

    serialize(MAGIC_NUMBER, out_file);
    serialize(original_size, out_file);
    serialize(crc, out_file);
    serialize(net, out_file);
    serialize(literal_seed, out_file);
    serialize(compressed_data, out_file);

    std::streampos final_size = out_file.tellp();
    out_file.close();

    double theoretical_size = WINDOW_SIZE + ((total_predictions * 8.0) - bits_saved) / 8.0;
    double bitstream_ratio = (theoretical_size / data.size()) * 100.0;
    double overall_ratio = (static_cast<double>(final_size) / data.size()) * 100.0;

    cout << "\n=== COMPRESSION SUMMARY ===" << endl;
    cout << "Original size            : " << format_size(data.size()) << endl;
    cout << "Bitstream (theoretical)  : " << format_size(static_cast<size_t>(theoretical_size))
        << " (" << std::fixed << std::setprecision(2) << bitstream_ratio << "%)" << endl;
    cout << "Final file (w/ model)    : " << format_size(static_cast<size_t>(final_size))
        << " (" << std::fixed << std::setprecision(2) << overall_ratio << "%)" << endl;
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

    cout << "Original file size: " << format_size(static_cast<size_t>(original_size)) << endl;
    cout << "Stored CRC32: " << std::hex << std::setfill('0') << std::setw(8) << stored_crc
        << std::dec << endl;

    // Load embedded model
    cout << "Loading embedded model..." << endl;
    infer_net net;
    deserialize(net, in_file);

    std::vector<uint8_t> literal_seed;
    deserialize(literal_seed, in_file);

    std::vector<uint8_t> compressed_data;
    deserialize(compressed_data, in_file);
    in_file.close();

    cout << "Compressed data: " << format_size(compressed_data.size()) << endl;

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

        parser.parse(argc, argv);

        if (!parser.option("compress") && !parser.option("decompress")) {
            cout << "Dlib Predictive Compressor (GQA Edition)\n\n";
            parser.print_options();
            cout << "\nExamples:\n";
            cout << "  Compress with training:\n";
            cout << "    " << argv[0] << " --compress --input data.txt --output data.dpc\n\n";
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
            compress_file(input_path, output_path, do_train);
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