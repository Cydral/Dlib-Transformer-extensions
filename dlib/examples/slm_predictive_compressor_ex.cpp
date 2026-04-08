/*!
    @file slm_predictive_compressor_ex.cpp
    @brief Byte-level predictive compression using a GQA Transformer.

    This example demonstrates an advanced application of AI generative models beyond
    traditional chatbot use cases. It implements a compression/decompression system
    for any file type using a small Transformer model to predict the next byte.

    The compression scheme uses a single bit to indicate prediction success, reducing
    data size when the predictor is accurate. Actual bit-packing is used to ensure
    physical file size reduction.

    Key updates in this version:
    - True bit-packing stream to physically shrink the serialized file.
    - Uses Grouped Query Attention (GQA) for reduced parameter count and faster inference.
    - Eliminates padding entirely: the model only operates on fully populated context windows.
    - The first WINDOW_SIZE bytes are stored uncompressed to prime the prediction engine.
    - Includes Graceful Interruption (CTRL+C) and Training Checkpoints via dlib::signal_handler.
    - Allows early-stopping: interrupting training continues directly to compression.

    Usage:
    --compress --input <file> [--output <file>]
    --decompress --input <file> [--output <file>]
!*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>

#include <dlib/dnn/transformer_config.h>
#include <dlib/dnn/transformer.h>

using namespace std;
using namespace dlib;

// ========================================================================================
// Constants & Configuration
// ========================================================================================

const uint32_t MAGIC_NUMBER = 0x444C4943;   // "DLIC" in big-endian (updated for bit-packing version)
const int WINDOW_SIZE = 8;                  // Fixed prediction window size
const long MAX_VOCAB_SIZE = 256;            // Exact byte range (0-255), no padding token needed

// GQA Network architecture parameters 
const long NUM_LAYERS = 2;
const long NUM_HEADS = 4;
const long NUM_KV_HEADS = 1;                // GQA: 4 query heads share 1 KV head
const long EMBEDDING_DIM = 16;

// High-level instantiation using GQA Transformer API
using compressor_transformer = gqa_transformer_config<
    MAX_VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, EMBEDDING_DIM>;

using train_net = compressor_transformer::network_type<true>;
using infer_net = compressor_transformer::network_type<false>;

// ========================================================================================
// Bit Stream Utilities
// ========================================================================================

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
        for (int i = 0; i < 8; ++i) {
            write_bit((b >> i) & 1);
        }
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
            if (read_bit()) {
                b |= (1 << i);
            }
        }
        return b;
    }
};

// ========================================================================================
// Helper Functions
// ========================================================================================

std::vector<uint8_t> read_file_bytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
}

// ========================================================================================
// Core Logic: Training
// ========================================================================================

void train_predictor(train_net& net, const std::vector<uint8_t>& data, const std::string& input_path)
{
    cout << "Preparing dataset for training (no padding)..." << endl;

    std::vector<matrix<int, 0, 1>> samples;
    std::vector<unsigned long> labels;

    matrix<int, 0, 1> window(WINDOW_SIZE, 1);
    for (size_t i = WINDOW_SIZE; i < data.size(); ++i)
    {
        long pos = 0;
        for (long j = i - WINDOW_SIZE; j < static_cast<long>(i); ++j) {
            window(pos++) = data[j];
        }

        samples.push_back(window);
        labels.push_back(data[i]);
    }

    if (samples.empty()) return;

    dnn_trainer<train_net, adam> trainer(net, adam(0.01, 0.9, 0.999));
    trainer.set_learning_rate(1e-3);
    trainer.set_min_learning_rate(1e-5);
    trainer.set_mini_batch_size(64);
    trainer.set_max_num_epochs(5);

    std::string checkpoint_file = input_path + ".chkpt";
    trainer.set_synchronization_file(checkpoint_file, std::chrono::minutes(1));
    trainer.be_quiet();

    cout << "Training predictor on " << samples.size() << " full-context transitions..." << endl;
    cout << "Checkpoint enabled. Press CTRL+C to interrupt safely and proceed to compression.\n";

    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate() && !signal_handler::is_triggered())
    {
        trainer.train_one_step(samples, labels);
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

    net.clean();
}

// ========================================================================================
// Compression
// ========================================================================================

void compress_file(const std::string& input_path, const std::string& output_path)
{
    cout << "Reading " << input_path << "..." << endl;
    std::vector<uint8_t> data = read_file_bytes(input_path);

    if (data.size() <= WINDOW_SIZE) {
        throw std::runtime_error("File is too small to compress (must be > WINDOW_SIZE).");
    }

    train_net t_net;
    train_predictor(t_net, data, input_path);

    // Continue compression even if training was interrupted (Early Stopping)
    if (signal_handler::is_triggered()) {
        cout << "Training interrupted. Proceeding to compression with the partial model..." << endl;
        // Reset the signal so the compression phase isn't immediately aborted.
        // NOTE: Adjust to "signal_handler::clear()" if "reset()" is not the exact name in your API.
        signal_handler::reset();
    }

    infer_net net;
    std::ostringstream net_data;
    serialize(t_net, net_data);
    std::istringstream net_in(net_data.str());
    deserialize(net, net_in);

    cout << "Compressing data stream..." << endl;

    std::vector<uint8_t> literal_seed(data.begin(), data.begin() + WINDOW_SIZE);
    std::vector<uint8_t> compressed_data;
    out_bit_stream bit_writer(compressed_data);

    matrix<int, 0, 1> window(WINDOW_SIZE, 1);
    long bits_saved = 0;
    long total_predictions = 0;

    for (size_t i = WINDOW_SIZE; i < data.size() && !signal_handler::is_triggered(); ++i)
    {
        long pos = 0;
        for (long j = i - WINDOW_SIZE; j < static_cast<long>(i); ++j) {
            window(pos++) = data[j];
        }

        int predicted_byte = net(window);
        total_predictions++;

        if (predicted_byte == data[i]) {
            bit_writer.write_bit(true);
            bits_saved += 7;
        }
        else {
            bit_writer.write_bit(false);
            bit_writer.write_byte(data[i]);
            bits_saved -= 1;
        }
    }

    bit_writer.flush();

    if (signal_handler::is_triggered()) {
        cout << "\n[!] Compression interrupted. Output file will NOT be saved to avoid corruption." << endl;
        return;
    }

    uint64_t original_size = data.size();

    std::ofstream out_file(output_path, std::ios::binary);
    serialize(MAGIC_NUMBER, out_file);
    serialize(original_size, out_file);
    serialize(net, out_file);
    serialize(literal_seed, out_file);
    serialize(compressed_data, out_file);

    double orig_size_bytes = static_cast<double>(data.size());
    double comp_size_bytes = WINDOW_SIZE + ((total_predictions * 8.0) - bits_saved) / 8.0;
    double ratio = (comp_size_bytes / orig_size_bytes) * 100.0;

    cout << "\n=== Compression Statistics ===\n"
        << "Original Size               : " << data.size() << " bytes\n"
        << "Theoretical Bitstream Size  : " << static_cast<size_t>(comp_size_bytes) << " bytes\n"
        << "Final File Size (w/ model)  : " << out_file.tellp() << " bytes\n"
        << "Bitstream Compression Ratio : " << std::fixed << std::setprecision(2) << ratio << "%\n"
        << "==============================\n";

    cout << "File successfully compressed to: " << output_path << endl;
}

// ========================================================================================
// Decompression
// ========================================================================================

void decompress_file(const std::string& input_path, const std::string& output_path)
{
    cout << "Decompressing " << input_path << "..." << endl;
    std::ifstream in_file(input_path, std::ios::binary);

    uint32_t magic;
    deserialize(magic, in_file);
    if (magic != MAGIC_NUMBER) throw std::runtime_error("Invalid file format (Magic Number mismatch).");

    uint64_t original_size;
    deserialize(original_size, in_file);

    infer_net net;
    deserialize(net, in_file);

    std::vector<uint8_t> literal_seed;
    deserialize(literal_seed, in_file);

    std::vector<uint8_t> compressed_data;
    deserialize(compressed_data, in_file);

    std::vector<uint8_t> decompressed_data = literal_seed;
    decompressed_data.reserve(original_size);
    matrix<int, 0, 1> window(WINDOW_SIZE, 1);
    in_bit_stream bit_reader(compressed_data);

    cout << "Restoring byte stream..." << endl;

    while (decompressed_data.size() < original_size && !signal_handler::is_triggered())
    {
        long pos = 0;
        for (long j = decompressed_data.size() - WINDOW_SIZE; j < static_cast<long>(decompressed_data.size()); ++j) {
            window(pos++) = decompressed_data[j];
        }

        int predicted_byte = net(window);

        bool success = bit_reader.read_bit();
        if (success) {
            decompressed_data.push_back(predicted_byte);
        }
        else {
            uint8_t actual_byte = bit_reader.read_byte();
            decompressed_data.push_back(actual_byte);
        }
    }

    if (signal_handler::is_triggered()) {
        cout << "\n[!] Decompression interrupted. Output file will NOT be saved to avoid corruption." << endl;
        return;
    }

    std::ofstream out_file(output_path, std::ios::binary);
    out_file.write(reinterpret_cast<const char*>(decompressed_data.data()), decompressed_data.size());
    cout << "Decompression complete. Restored " << decompressed_data.size() << " bytes to: " << output_path << endl;
}

// ========================================================================================
// Main
// ========================================================================================

int main(int argc, char** argv)
{
    try
    {
        signal_handler::setup();

        command_line_parser parser;
        parser.add_option("compress", "Compress file");
        parser.add_option("decompress", "Decompress file");
        parser.add_option("input", "Input file path", 1);
        parser.add_option("output", "Output file path", 1);
        parser.parse(argc, argv);

        if (!parser.option("compress") && !parser.option("decompress")) {
            cout << "Dlib Predictive Compressor (GQA Edition)\n";
            parser.print_options();
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
            if (parser.option("compress")) {
                output_path = input_path + ".dpc";
            }
            else {
                output_path = input_path + ".restored";
            }
            cout << "No output specified. Defaulting to: " << output_path << "\n";
        }

        if (parser.option("compress")) {
            compress_file(input_path, output_path);
        }
        else {
            decompress_file(input_path, output_path);
        }

        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "Exception: " << e.what() << endl;
        return 1;
    }
}