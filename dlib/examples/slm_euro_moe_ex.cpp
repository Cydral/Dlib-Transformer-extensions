/*!
    @file slm_euro_moe_ex.cpp
    @brief Training pipeline for a multilingual SLM based on GQA + MoE.

    This program trains a compact small language model using Grouped Query Attention (GQA) combined
    with Mixture-of-Experts (MoE) feed-forward blocks for conditional computation. At inference,
    only a fraction of the total parameters are activated per token thanks to top-k sparse routing.

    Five modes are exposed:

      --build-tokenizer   Train a BPE tokenizer on the corpus (run once).
      --train             Pre-train the model (resumes from checkpoint if any).
      --fine-tune         Instruction fine-tuning (placeholder at this stage).
      --prompt            Interactive generation loop (placeholder at this stage).
      --generate          Autoregressive generation sanity check.

    Typical usage:
      Build tokenizer : slm_euro_moe_ex --build-tokenizer --external-data corpus.txt
      Pre-train       : slm_euro_moe_ex --train --external-data corpus.txt
      Generate sample : slm_euro_moe_ex --generate

    Architecture note (ACT extension):
    Replacing the MoE feed-forward by an ACT-wrapped FFN in the upper layers would add adaptive-depth
    reasoning while keeping MoE specialization in the lower layers. This requires either extending
    gqa_moe_transformer_config or building the hybrid network manually.
!*/

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/misc_api.h>
#include <dlib/tokenizer.h>

#include "slm_data.h"

using namespace std;
using namespace dlib;

/*
 * Static architecture parameters. These are compile-time template arguments for the network type
 * and must match the tokenizer vocabulary size. To change the architecture, edit the constants
 * here and recompile.
 */
constexpr long NUM_TOKENS = 5850;
constexpr long NUM_LAYERS = 4;
constexpr long NUM_HEADS = 6;
constexpr long NUM_KV_HEADS = 2;
constexpr long EMBEDDING_DIM = 228;
constexpr long NUM_EXPERTS = 4;
constexpr long TOP_K = 2;
constexpr long MAX_SEQ_LEN = 300;

// Default sampling budget for tokenizer training (in bytes)
constexpr size_t DEFAULT_MAX_TOKENIZER_BYTES = 200ull * 1024 * 1024;

/*
 * Derive the tokens cache filename from the source corpus path. When the same corpus is reused
 * across runs, the cached tokens are reloaded instead of re-tokenizing.
 */
std::string derive_tokens_filename(const std::string& external_path)
{
    if (external_path.empty()) return "dlib_euro_moe_tokens.bin";
    size_t sep = external_path.find_last_of("/\\");
    std::string base = (sep == std::string::npos) ? external_path : external_path.substr(sep + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return "dlib_euro_moe_tokens_" + base + ".bin";
}

/*
 * Read file content, optionally applying stratified sampling. When sample is false, returns the
 * full file content verbatim.
 *
 * When sample is true and the file exceeds max_bytes, the file is divided into N equal windows and
 * a chunk is taken from each one, ensuring uniform coverage across the entire file. Each chunk is
 * aligned on whitespace boundaries to avoid truncating words.
 */
std::string read_file_content(const std::string& filepath, bool sample = false, size_t max_bytes = 0)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file) { cerr << "Warning: Cannot open file: " << filepath << "\n"; return ""; }

    file.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    if (!sample || max_bytes == 0 || file_size <= max_bytes) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    // Stratified sampling path
    constexpr size_t num_samples = 1024;
    constexpr size_t alignment_margin = 1024;
    const size_t chunk_size = max_bytes / num_samples;
    const size_t stride = file_size / num_samples;
    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };

    std::string result;
    result.reserve(max_bytes + num_samples);
    std::vector<char> buffer(chunk_size + 2 * alignment_margin);

    for (size_t i = 0; i < num_samples; ++i) {
        const size_t offset = i * stride;
        if (offset >= file_size) break;
        const size_t to_read = std::min(buffer.size(), file_size - offset);
        file.seekg(offset);
        file.read(buffer.data(), to_read);
        const size_t got = static_cast<size_t>(file.gcount());
        if (got == 0) continue;

        size_t start = 0;
        if (i > 0) {
            while (start < got && !is_ws(static_cast<unsigned char>(buffer[start]))) ++start;
            while (start < got && is_ws(static_cast<unsigned char>(buffer[start])))  ++start;
        }
        size_t end = std::min(start + chunk_size, got);
        while (end > start && !is_ws(static_cast<unsigned char>(buffer[end - 1]))) --end;
        while (end > start && is_ws(static_cast<unsigned char>(buffer[end - 1])))  --end;

        if (end > start) { result.append(buffer.data() + start, end - start); result.push_back('\n'); }
    }

    cout << "Stratified sampling: " << (result.size() / (1024.0 * 1024.0)) << " MB extracted from "
        << (file_size / (1024.0 * 1024.0)) << " MB in " << num_samples << " aligned chunks\n";
    return result;
}

/*
 * Load a text corpus and group consecutive non-blank lines into coherent training segments. Each
 * segment contains lines_per_segment source lines joined by '\n'.
 */
std::vector<std::string> load_external_corpus_grouped(const std::string& path, size_t lines_per_segment)
{
    std::vector<std::string> segments;
    std::ifstream file(path);
    if (!file) { cerr << "Error: Cannot open corpus file: " << path << "\n"; return segments; }

    auto flush_buffer = [&](std::vector<std::string>& buffer) {
        if (buffer.empty()) return;
        std::string segment;
        size_t total = 0;
        for (const auto& l : buffer) total += l.size() + 1;
        segment.reserve(total);
        for (size_t i = 0; i < buffer.size(); ++i) { segment += buffer[i]; if (i + 1 < buffer.size()) segment += '\n'; }
        segments.push_back(std::move(segment));
        buffer.clear();
        };

    std::string line;
    std::vector<std::string> buffer;
    size_t total_lines = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        buffer.push_back(std::move(line));
        ++total_lines;
        if (buffer.size() == lines_per_segment) flush_buffer(buffer);
    }
    flush_buffer(buffer);

    cout << "Loaded " << total_lines << " non-blank lines grouped into " << segments.size()
        << " segments (" << lines_per_segment << " lines/segment)\n";
    return segments;
}

// MoE parameter and expert usage analysis

struct moe_param_info
{
    size_t single_expert_params, total_params, inference_params;
    long num_experts, num_moe_layers, top_k;
    float efficiency_ratio;
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
            float total = 0.0f, mn = expert_usage[0], mx = expert_usage[0];
            for (float u : expert_usage) { total += u; mn = std::min(mn, u); mx = std::max(mx, u); }
            float mean = total / num_experts, ideal = 1.0f / num_experts;
            float var = 0.0f;
            for (float u : expert_usage) { float d = u - mean; var += d * d; }
            var /= num_experts;
            float std_dev = std::sqrt(var), cv = (mean > 1e-8f) ? (std_dev / mean) : 0.0f;

            cout << "  Mean usage : " << std::fixed << std::setprecision(4) << mean << " (ideal: " << ideal << ")\n"
                << "  Range      : [" << mn << ", " << mx << "]\n"
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
                int bar = (mx > 0) ? static_cast<int>(expert_usage[e] * 20.0f / mx) : 0;
                cout << " [";
                for (int i = 0; i < bar; ++i) cout << "=";
                for (int i = bar; i < 20; ++i) cout << " ";
                cout << "]";
                float r = expert_usage[e] / ideal;
                if (r < 0.5f)      cout << " (underutilized)";
                else if (r > 2.0f) cout << " (overutilized)";
                cout << "\n";
            }
        }
        cout << "\n";
    }
};

// Layer index in topology: loss=0, linear=1, rms_norm=2, add_prev=3, moe=4
template <typename net_type>
moe_param_info get_moe_param_info(const net_type& net, long num_layers)
{
    moe_param_info info;
    const auto& moe = layer<4>(net).layer_details();
    info.num_experts = moe.num_experts();
    info.top_k = moe.num_active_experts();
    info.num_moe_layers = num_layers;
    info.single_expert_params = (info.num_experts > 0) ? count_parameters(moe.get_expert(0)) : 0;
    info.total_params = count_parameters(net);
    size_t inactive = (info.num_experts - info.top_k) * info.single_expert_params;
    info.inference_params = info.total_params - static_cast<size_t>(num_layers) * inactive;
    info.efficiency_ratio = (info.total_params > 0) ? float(info.inference_params) / float(info.total_params) : 1.0f;
    info.expert_usage = moe.get_expert_usage();
    return info;
}

template <typename net_type>
auto try_print_moe_info(const net_type& net, long nl) -> decltype(layer<4>(net).layer_details().num_experts(), void())
{
    auto info = get_moe_param_info(net, nl); if (info.num_experts > 0) info.print();
}
inline void try_print_moe_info(...) {}

/*
 * Tokenizer building (standalone mode). The bpe_tokenizer class pre-registers special tokens
 * in its constructor, so no manual registration is needed. Large corpora are sampled down to
 * max_tokenizer_bytes with stratified sampling to keep BPE training tractable.
 */
 // ---------------------------------------------------------------------------
 // Chunked on-disk tokens storage
 //
 // For very large corpora the tokenized stream and the resulting training
 // samples no longer fit in RAM. Beyond TOKEN_STREAMING_THRESHOLD total tokens,
 // the tokens cache file switches from the legacy single-vector format to a
 // chunked format described below, and the training loop iterates chunks
 // sequentially:
 //   - one chunk is loaded from disk;
 //   - its main samples (flat sliding windows) and cold-start auxiliary
 //     samples are rebuilt from the chunk's segments;
 //   - the trainer processes those samples;
 //   - the chunk is dropped before the next one is loaded.
 //
 // File layout (little-endian):
 //   [8 bytes]  Magic "DLEMTKNS"
 //   [4 bytes]  Version (uint32)
 //   [8 bytes]  Number of chunks (uint64)
 //   [8 bytes]  Total token count (uint64)
 //   [N*8 bytes] Per-chunk byte offsets (uint64)
 //   [N*8 bytes] Per-chunk token counts (uint64)
 //   [...]      Chunk 0 ... Chunk N-1: each a dlib::serialize'd
 //              std::vector<std::vector<int>> (segments contained in the chunk)
 //
 // The reader caches the most recently fetched chunk so consecutive
 // get_chunk(i) calls do not trigger a disk read.
 // ---------------------------------------------------------------------------

constexpr size_t TOKEN_STREAMING_THRESHOLD = 100000;     // total tokens above which streaming kicks in
constexpr size_t CHUNK_TOKEN_TARGET = 500000;            // approximate tokens per disk chunk
constexpr char CHUNKED_TOKENS_MAGIC[8] = { 'D','L','E','M','T','K','N','S' };
constexpr uint32_t CHUNKED_TOKENS_VERSION = 1;

bool is_chunked_tokens_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    char magic[8];
    f.read(magic, 8);
    return f.gcount() == 8 && std::memcmp(magic, CHUNKED_TOKENS_MAGIC, 8) == 0;
}

void write_chunked_tokens(const std::string& path,
    std::vector<std::vector<int>>& full_tokens,
    size_t chunk_token_target)
{
    // Partition segments greedily into chunks of >= chunk_token_target tokens
    std::vector<std::pair<size_t, size_t>> ranges; // [start, end) over full_tokens
    {
        size_t start = 0, accum = 0;
        for (size_t i = 0; i < full_tokens.size(); ++i) {
            accum += full_tokens[i].size();
            if (accum >= chunk_token_target) {
                ranges.emplace_back(start, i + 1);
                start = i + 1;
                accum = 0;
            }
        }
        if (start < full_tokens.size()) ranges.emplace_back(start, full_tokens.size());
    }
    const uint64_t num_chunks = ranges.size();
    uint64_t total_tokens = 0;
    for (const auto& seg : full_tokens) total_tokens += seg.size();

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("cannot write " + path);

    out.write(CHUNKED_TOKENS_MAGIC, 8);
    uint32_t version = CHUNKED_TOKENS_VERSION;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&num_chunks), sizeof(num_chunks));
    out.write(reinterpret_cast<const char*>(&total_tokens), sizeof(total_tokens));

    // Reserve space for offset and count tables, to be backfilled.
    const std::streampos table_pos = out.tellp();
    std::vector<uint64_t> offsets(num_chunks, 0), counts(num_chunks, 0);
    out.write(reinterpret_cast<const char*>(offsets.data()), num_chunks * sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(counts.data()), num_chunks * sizeof(uint64_t));

    for (size_t c = 0; c < num_chunks; ++c) {
        offsets[c] = static_cast<uint64_t>(out.tellp());
        std::vector<std::vector<int>> chunk;
        chunk.reserve(ranges[c].second - ranges[c].first);
        uint64_t toks = 0;
        for (size_t i = ranges[c].first; i < ranges[c].second; ++i) {
            toks += full_tokens[i].size();
            chunk.push_back(std::move(full_tokens[i]));
        }
        counts[c] = toks;
        dlib::serialize(chunk, out);
    }

    // Backfill the offset and count tables now that we know them.
    out.seekp(table_pos);
    out.write(reinterpret_cast<const char*>(offsets.data()), num_chunks * sizeof(uint64_t));
    out.write(reinterpret_cast<const char*>(counts.data()), num_chunks * sizeof(uint64_t));
}

class chunked_tokens_reader
{
public:
    explicit chunked_tokens_reader(const std::string& path) :
        cached_idx_(static_cast<size_t>(-1)), cache_valid_(false)
    {
        in_.open(path, std::ios::binary);
        if (!in_.is_open()) throw std::runtime_error("cannot open " + path);

        char magic[8];
        in_.read(magic, 8);
        if (in_.gcount() != 8 || std::memcmp(magic, CHUNKED_TOKENS_MAGIC, 8) != 0)
            throw std::runtime_error(path + " is not a chunked tokens file");

        uint32_t version = 0;
        in_.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != CHUNKED_TOKENS_VERSION)
            throw std::runtime_error("unsupported chunked tokens version");

        in_.read(reinterpret_cast<char*>(&num_chunks_), sizeof(num_chunks_));
        in_.read(reinterpret_cast<char*>(&total_tokens_), sizeof(total_tokens_));

        offsets_.resize(num_chunks_);
        counts_.resize(num_chunks_);
        in_.read(reinterpret_cast<char*>(offsets_.data()), num_chunks_ * sizeof(uint64_t));
        in_.read(reinterpret_cast<char*>(counts_.data()), num_chunks_ * sizeof(uint64_t));
    }

    size_t num_chunks() const { return static_cast<size_t>(num_chunks_); }
    uint64_t total_tokens() const { return total_tokens_; }
    uint64_t chunk_token_count(size_t i) const { return counts_[i]; }

    const std::vector<std::vector<int>>& get_chunk(size_t idx)
    {
        if (cache_valid_ && cached_idx_ == idx) return cached_chunk_;
        if (idx >= num_chunks_) throw std::out_of_range("chunk index out of range");

        in_.clear();
        in_.seekg(static_cast<std::streamoff>(offsets_[idx]));
        cached_chunk_.clear();
        dlib::deserialize(cached_chunk_, in_);
        cached_idx_ = idx;
        cache_valid_ = true;
        return cached_chunk_;
    }

private:
    std::ifstream in_;
    uint64_t num_chunks_ = 0;
    uint64_t total_tokens_ = 0;
    std::vector<uint64_t> offsets_;
    std::vector<uint64_t> counts_;

    std::vector<std::vector<int>> cached_chunk_;
    size_t cached_idx_;
    bool cache_valid_;
};

// Build the per-chunk (or full-corpus, in-memory mode) training samples.
// Produces: flat-corpus main samples + filtered cold-start auxiliary samples
// + random-token augmentation on main. Cold-start is capped at 5% of main.
void build_samples_from_segments(
    const std::vector<std::vector<int>>& segments,
    long max_seq_len, int pad_token, const bpe_tokenizer& tokenizer,
    std::vector<matrix<int, 0, 1>>& out_samples,
    std::vector<unsigned long>& out_labels,
    size_t& out_main_size, size_t& out_aux_kept)
{
    out_samples.clear();
    out_labels.clear();

    // Flat-corpus main samples
    {
        std::vector<std::vector<int>> flat_corpus(1);
        size_t flat_total = 0;
        for (const auto& seg : segments) flat_total += seg.size();
        flat_corpus[0].reserve(flat_total);
        for (const auto& seg : segments)
            flat_corpus[0].insert(flat_corpus[0].end(), seg.begin(), seg.end());
        build_single_token_prediction_dataset(flat_corpus, max_seq_len, pad_token, false, out_samples, out_labels);
    }
    out_main_size = out_samples.size();

    // Cold-start: per-segment with progressive left padding, then keep only padded ones
    std::vector<matrix<int, 0, 1>> aux_samples;
    std::vector<unsigned long> aux_labels;
    build_single_token_prediction_dataset(segments, max_seq_len, pad_token, true, aux_samples, aux_labels);
    {
        std::vector<matrix<int, 0, 1>> fx;
        std::vector<unsigned long> fy;
        fx.reserve(aux_samples.size());
        fy.reserve(aux_samples.size());
        for (size_t i = 0; i < aux_samples.size(); ++i) {
            if (count_leading_padding(aux_samples[i], pad_token) > 0) {
                fx.push_back(std::move(aux_samples[i]));
                fy.push_back(aux_labels[i]);
            }
        }
        aux_samples = std::move(fx);
        aux_labels = std::move(fy);
    }

    const size_t aux_target = out_main_size / 20;
    if (aux_samples.size() > aux_target) {
        dlib::rand rng(std::chrono::system_clock::now().time_since_epoch().count());
        for (size_t i = aux_samples.size(); i > aux_target; --i) {
            const size_t idx = rng.get_random_64bit_number() % i;
            aux_samples[idx] = std::move(aux_samples.back());
            aux_samples.pop_back();
            aux_labels[idx] = aux_labels.back();
            aux_labels.pop_back();
        }
    }
    out_aux_kept = aux_samples.size();

    // Augment ONLY the main samples (cold-start teaches short contexts and must not be perturbed)
    const size_t num_specials = tokenizer.get_specials_size();
    const int random_max = static_cast<int>(tokenizer.get_vocab_size() - num_specials - 1);
    const int random_min = 256;
    augment_training_dataset(out_samples, out_labels,
        static_cast<int>(tokenizer.get_special_token_id("<unk>")), pad_token, 0.05,
        1, 3, 0, augmentation_mode::random_token, random_min, random_max);

    // Combine main+aux
    out_samples.insert(out_samples.end(),
        std::make_move_iterator(aux_samples.begin()),
        std::make_move_iterator(aux_samples.end()));
    out_labels.insert(out_labels.end(), aux_labels.begin(), aux_labels.end());
}

int run_build_tokenizer(const std::string& external_path, const std::string& tokenizer_file, size_t max_tokenizer_bytes)
{
    cout << "=== BUILDING BPE TOKENIZER ===\n"
        << "Target vocabulary size  : " << NUM_TOKENS << "\n"
        << "Max corpus size (sample): " << (max_tokenizer_bytes / (1024.0 * 1024.0)) << " MB\n";

    std::string corpus;
    if (!external_path.empty()) {
        cout << "Loading corpus from: " << external_path << "\n";
        corpus = read_file_content(external_path, /*sample=*/true, max_tokenizer_bytes);
        cout << "Corpus size after sampling: " << (corpus.size() / (1024.0 * 1024.0)) << " MB\n";
    }
    corpus += "\n" + get_dataset_as_text(dataset_id::GENERAL_KNOWLEDGE);
    if (corpus.empty()) { cerr << "Error: Corpus is empty. Provide --external-data.\n"; return 1; }

    cout << "Total corpus size for tokenizer training: " << (corpus.size() / (1024.0 * 1024.0)) << " MB\n";

    bpe_tokenizer tokenizer;
    cout << "Training BPE...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    tokenizer.train(corpus, NUM_TOKENS, /*max_bytes=*/0, /*verbose=*/true);
    auto t1 = std::chrono::high_resolution_clock::now();
    cout << "Training complete in " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << " seconds\n"
        << "Final vocabulary size: " << tokenizer.get_vocab_size() << "\n";

    serialize(tokenizer_file) << tokenizer;
    cout << "Tokenizer saved to: " << tokenizer_file << "\n";
    return 0;
}

// Pre-training and generation pipeline

template <typename TRANSFORMER_CONFIG>
int run_pipeline(bool do_train, bool do_generate, const double learning_rate, const size_t batch_size,
    const long patience, const size_t max_epochs, const double weight_decay, const double beta1, const double beta2,
    const std::string& model_file, const std::string& tokenizer_file, const std::string& tokens_file,
    std::vector<std::string>& text_segments, bpe_tokenizer& tokenizer, std::vector<int>& gpus)
{
    using my_transformer = TRANSFORMER_CONFIG;
    cout << my_transformer::model_info::describe() << "\n";
    cout << "Tokens cache file: " << tokens_file << "\n";

    // Training mode
    if (do_train)
    {
        if (tokenizer.get_vocab_size() == 0) { cerr << "Error: tokenizer is empty. Run --build-tokenizer first.\n"; return 1; }

        std::vector<std::vector<int>> full_tokens;
        bool tokens_loaded = false;
        bool use_streaming = false;
        size_t total_token_count = 0;

        if (file_exists(tokens_file)) {
            cout << "Found pre-tokenized tokens file: " << tokens_file << "\n";
            if (is_chunked_tokens_file(tokens_file)) {
                try {
                    chunked_tokens_reader probe(tokens_file);
                    total_token_count = static_cast<size_t>(probe.total_tokens());
                    cout << "Format: chunked (" << probe.num_chunks() << " chunks, "
                        << total_token_count << " tokens). Streaming mode enabled.\n";
                    use_streaming = true;
                    tokens_loaded = true;
                }
                catch (const std::exception& e) {
                    cerr << "Failed to open chunked tokens file: " << e.what() << "\nWill tokenize again.\n";
                }
            }
            else {
                try {
                    dlib::deserialize(tokens_file) >> full_tokens;
                    for (const auto& seg : full_tokens) total_token_count += seg.size();
                    cout << "Loaded " << full_tokens.size() << " segments ("
                        << total_token_count << " tokens)\n";
                    tokens_loaded = true;
                    // Migrate legacy tokens cache to chunked if it crosses the streaming threshold,
                    // so the next run streams from disk instead of reloading everything.
                    if (total_token_count > TOKEN_STREAMING_THRESHOLD) {
                        cout << "Token count exceeds streaming threshold (" << TOKEN_STREAMING_THRESHOLD
                            << "); migrating cache to chunked format...\n";
                        write_chunked_tokens(tokens_file, full_tokens, CHUNK_TOKEN_TARGET);
                        std::vector<std::vector<int>>().swap(full_tokens);
                        use_streaming = true;
                    }
                }
                catch (const std::exception& e) {
                    cerr << "Failed to load tokens: " << e.what() << "\nWill tokenize again.\n";
                    full_tokens.clear();
                }
            }
        }

        if (!tokens_loaded) {
            long text_start_id = tokenizer.get_special_token_id("<text>");
            long text_end_id = tokenizer.get_special_token_id("</text>");
            if (text_start_id < 0 || text_end_id < 0) { cerr << "ERROR: Required special tokens missing.\n"; return 1; }

            const size_t total_segs = text_segments.size();
            cout << "Tokenizing " << total_segs << " segments...\n";
            auto t_start = std::chrono::high_resolution_clock::now();
            size_t total_tokens = 0;
            constexpr size_t progress_every = 5000;

            for (size_t s = 0; s < total_segs; ++s) {
                std::vector<int> seg_tokens;
                seg_tokens.push_back(static_cast<int>(text_start_id));
                auto encoded = tokenizer.encode(text_segments[s]);
                seg_tokens.insert(seg_tokens.end(), encoded.begin(), encoded.end());
                seg_tokens.push_back(static_cast<int>(text_end_id));
                total_tokens += seg_tokens.size();
                full_tokens.push_back(std::move(seg_tokens));

                if ((s + 1) % progress_every == 0 || s + 1 == total_segs) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto secs = std::chrono::duration_cast<std::chrono::seconds>(now - t_start).count();
                    double rate = (secs > 0) ? static_cast<double>(s + 1) / secs : 0.0;
                    cout << "  " << (s + 1) << "/" << total_segs << " segments ("
                        << std::fixed << std::setprecision(0) << rate << " seg/s)\r";
                    cout.flush();
                }
            }
            cout << "\n";
            auto t_end = std::chrono::high_resolution_clock::now();
            cout << "Tokenization complete: " << total_tokens << " tokens in "
                << std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count() << "s\n";
            text_segments.clear();
            total_token_count = total_tokens;

            if (total_token_count > TOKEN_STREAMING_THRESHOLD) {
                cout << "Token count exceeds streaming threshold (" << TOKEN_STREAMING_THRESHOLD
                    << "); writing chunked tokens to: " << tokens_file << "\n";
                try {
                    write_chunked_tokens(tokens_file, full_tokens, CHUNK_TOKEN_TARGET);
                    cout << "Chunked tokens saved.\n";
                    std::vector<std::vector<int>>().swap(full_tokens);
                    use_streaming = true;
                }
                catch (const std::exception& e) {
                    cerr << "Warning: failed to write chunked tokens: " << e.what()
                        << "\nFalling back to in-memory training.\n";
                }
            }
            else {
                cout << "Saving tokens to: " << tokens_file << "\n";
                try { serialize(tokens_file) << full_tokens; cout << "Tokens saved.\n"; }
                catch (const std::exception& e) { cerr << "Warning: Failed to save tokens: " << e.what() << "\n"; }
            }
        }

        std::vector<matrix<int, 0, 1>> samples;
        std::vector<unsigned long> labels;
        const int pad_token = static_cast<int>(tokenizer.get_special_token_id("<pad>"));
        std::unique_ptr<chunked_tokens_reader> stream_reader;
        size_t in_memory_main_size = 0, in_memory_aux_kept = 0;

        if (use_streaming) {
            cout << "Preparing chunked streaming training (window=" << MAX_SEQ_LEN << ")...\n";
            stream_reader.reset(new chunked_tokens_reader(tokens_file));
            cout << "Streaming dataset: " << stream_reader->num_chunks() << " chunks, "
                << stream_reader->total_tokens() << " tokens total.\n";
        }
        else {
            cout << "Preparing training sequences in memory (window=" << MAX_SEQ_LEN << ")...\n";
            build_samples_from_segments(full_tokens, MAX_SEQ_LEN, pad_token, tokenizer,
                samples, labels, in_memory_main_size, in_memory_aux_kept);
            full_tokens.clear();
            cout << "Final dataset size: main=" << in_memory_main_size
                << " (+" << (samples.size() - in_memory_main_size - in_memory_aux_kept) << " augmented)"
                << ", cold-start=" << in_memory_aux_kept
                << ", total=" << samples.size() << "\n";
        }

        using train_net_type = typename my_transformer::template network_type<true>;
        train_net_type net;
        layer<0>(net).loss_details().set_ignore_index(pad_token);

        if (file_exists(model_file) && !file_exists("chkpt-" + model_file)) {
            cout << "Loading existing model from " << model_file << "\n";
            deserialize(model_file) >> net >> tokenizer;
        }

        network_context::set_optimizer_params(weight_decay, beta1, beta2);
        cout << net << "\n";
        parameter_counts param_count = count_network_parameters(net, MAX_SEQ_LEN);
        cout << "Model parameters: " << param_count.total << " (active: " << param_count.active << ")\n";

        dnn_trainer<train_net_type, adamw> trainer(net, adamw(weight_decay, beta1, beta2), gpus);
        trainer.set_learning_rate(learning_rate);
        trainer.set_min_learning_rate(1e-8);
        trainer.set_learning_rate_shrink_factor(0.1);
        trainer.set_mini_batch_size(batch_size);
        trainer.set_iterations_without_progress_threshold(patience);
        trainer.set_synchronization_file("chkpt-" + model_file, std::chrono::minutes(15));
        trainer.be_quiet();

        cout << "Starting training...\n";
        size_t epoch = 0, batches_count = 0;

        // One pass over a (batch, label) pair: synchronize trainer, propagate
        // learning rate and padding to the context, then train_one_step. Used
        // by both in-memory and streaming modes.
        auto run_one_step = [&](std::vector<matrix<int, 0, 1>>& batch_X,
            std::vector<unsigned long>& batch_Y,
            double& total_loss, size_t& batches_seen, size_t& samples_seen,
            std::chrono::high_resolution_clock::time_point epoch_start)
            {
                std::vector<long> pad_lengths(batch_X.size());
                for (size_t j = 0; j < batch_X.size(); ++j)
                    pad_lengths[j] = count_leading_padding(batch_X[j], pad_token);

                trainer.get_net(force_flush_to_disk::no);
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
                    double sps = samples_seen / (elapsed > 0 ? elapsed : 1);
                    cout << "epoch#: " << (epoch + 1) << "/" << max_epochs << " \t loss: " << avg_loss
                        << " \t lr: " << trainer.get_learning_rate() << " \t speed: " << sps << " samples/sec\r";
                    cout.flush();
                }
            };

        auto run_one_dataset_pass = [&](std::vector<matrix<int, 0, 1>>& X,
            std::vector<unsigned long>& Y,
            double& total_loss, size_t& batches_seen, size_t& samples_seen,
            std::chrono::high_resolution_clock::time_point epoch_start)
            {
                shuffle_training_dataset(X, Y);
                for (size_t i = 0; i < X.size() && !signal_handler::is_triggered(); i += batch_size) {
                    size_t batch_end = std::min(i + batch_size, X.size());
                    std::vector<matrix<int, 0, 1>> batch_X(X.begin() + i, X.begin() + batch_end);
                    std::vector<unsigned long> batch_Y(Y.begin() + i, Y.begin() + batch_end);
                    run_one_step(batch_X, batch_Y, total_loss, batches_seen, samples_seen, epoch_start);
                }
            };

        if (!use_streaming)
        {
            // In-memory mode: a full epoch is one shuffle + one pass over all samples.
            while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
                && epoch < max_epochs && !signal_handler::is_triggered())
            {
                double total_loss = 0.0;
                size_t batches_seen = 0, samples_seen = 0;
                auto epoch_start = std::chrono::high_resolution_clock::now();
                run_one_dataset_pass(samples, labels, total_loss, batches_seen, samples_seen, epoch_start);
                epoch++;
            }
        }
        else
        {
            // Streaming mode: an epoch is one pass over all chunks (in shuffled
            // order). Each chunk is loaded from disk, its samples are rebuilt,
            // the trainer runs through them, then the samples are freed before
            // the next chunk is loaded. The reader caches the most recently
            // loaded chunk so the case where the same chunk index is requested
            // twice in a row never triggers a redundant disk read.
            std::vector<size_t> chunk_order(stream_reader->num_chunks());
            std::iota(chunk_order.begin(), chunk_order.end(), 0);
            dlib::rand chunk_rng(std::chrono::system_clock::now().time_since_epoch().count());

            while (trainer.get_learning_rate() >= trainer.get_min_learning_rate()
                && epoch < max_epochs && !signal_handler::is_triggered())
            {
                // Shuffle chunk order each outer epoch to avoid presenting the
                // corpus in the same disk order every time.
                for (size_t i = chunk_order.size(); i > 1; --i) {
                    size_t j = static_cast<size_t>(chunk_rng.get_random_64bit_number() % i);
                    std::swap(chunk_order[i - 1], chunk_order[j]);
                }

                double total_loss = 0.0;
                size_t batches_seen = 0, samples_seen = 0;
                auto epoch_start = std::chrono::high_resolution_clock::now();

                for (size_t c = 0; c < chunk_order.size() && !signal_handler::is_triggered()
                    && trainer.get_learning_rate() >= trainer.get_min_learning_rate(); ++c)
                {
                    const size_t chunk_idx = chunk_order[c];
                    const auto& chunk_segments = stream_reader->get_chunk(chunk_idx);

                    std::vector<matrix<int, 0, 1>> chunk_samples;
                    std::vector<unsigned long> chunk_labels;
                    size_t main_size = 0, aux_kept = 0;
                    build_samples_from_segments(chunk_segments, MAX_SEQ_LEN, pad_token, tokenizer,
                        chunk_samples, chunk_labels, main_size, aux_kept);

                    run_one_dataset_pass(chunk_samples, chunk_labels,
                        total_loss, batches_seen, samples_seen, epoch_start);

                    // Release per-chunk samples before moving on.
                    std::vector<matrix<int, 0, 1>>().swap(chunk_samples);
                    std::vector<unsigned long>().swap(chunk_labels);
                }
                epoch++;
            }
        }
        cout << "\n";

        trainer.get_net();
        net.clean();
        serialize(model_file) << net << tokenizer;
        cout << "Model saved to " << model_file << "\n";
        network_context::reset();
    }

    // Generation mode
    if (do_generate)
    {
        typename my_transformer::template network_type<false> net;
        if (!file_exists(model_file)) { cerr << "Error: model file not found. Run --train first.\n"; return 0; }

        deserialize(model_file) >> net >> tokenizer;
        cout << "Loaded model from " << model_file << "\n";
        cout << "Number of model parameters: " << count_parameters(net) << "\n";
        try_print_moe_info(net, NUM_LAYERS);

        if (tokenizer.get_vocab_size() == 0) { cerr << "Error: Tokenizer not loaded.\n"; return 0; }
        if (!file_exists(tokens_file)) { cerr << "Error: Tokenized file not found. Run --train first.\n"; return 0; }

        cout << "Loading tokenized segments from: " << tokens_file << "\n";
        std::vector<std::vector<int>> tokenized_segments;
        deserialize(tokens_file) >> tokenized_segments;
        cout << "Loaded " << tokenized_segments.size() << " segments\n";

        std::vector<size_t> valid;
        for (size_t i = 0; i < tokenized_segments.size(); ++i)
            if (tokenized_segments[i].size() >= 2) valid.push_back(i);
        if (valid.empty()) { cerr << "Error: No segments with at least 2 tokens.\n"; return 1; }

        dlib::rand rng(std::chrono::system_clock::now().time_since_epoch().count());
        size_t segment_idx = valid[rng.get_random_32bit_number() % valid.size()];
        cout << "Randomly selected segment #" << segment_idx << "\n";

        const auto& seg = tokenized_segments[segment_idx];
        const size_t seg_len = seg.size();
        const int pad_token = static_cast<int>(tokenizer.get_special_token_id("<pad>"));

        std::vector<int> input_half, verify_half;
        if (seg_len < static_cast<size_t>(MAX_SEQ_LEN)) {
            size_t input_count = (seg_len + 1) / 2;
            if (seg_len - input_count == 0) { cerr << "Error: Segment too short for verification.\n"; return 1; }
            input_half.assign(seg.begin(), seg.begin() + input_count);
            verify_half.assign(seg.begin() + input_count, seg.end());
        }
        else {
            input_half.assign(seg.begin(), seg.begin() + MAX_SEQ_LEN);
            verify_half.assign(seg.begin() + MAX_SEQ_LEN, seg.end());
        }

        inference_context ctx(MAX_SEQ_LEN, 3, pad_token);
        ctx.add_tokens(input_half);

        cout << "\n--- Prompt ---\n" << tokenizer.decode(input_half, false) << "\n--------------\n\n";
        cout << "Generating...\n";
        const size_t target = verify_half.size();
        std::vector<int> generated;
        generated.reserve(target);
        const int eot_id = static_cast<int>(tokenizer.get_special_token_id("</text>"));
        auto t0 = std::chrono::high_resolution_clock::now();

        // KV cache strategy: one prefill forward over the (windowed, possibly
        // padded) prompt populates each gqa_attention_ layer's K/V cache, then
        // each subsequent token is produced by a single-token incremental
        // forward. The attention layer slides its cache automatically once it
        // reaches MAX_SEQ_LEN, so the generation loop only feeds the next token.
        network_context::reset();
        network_context::set_kv_cache_capacity(MAX_SEQ_LEN);

        // Prefill on the windowed prompt; padding (if any) is stripped from
        // the cache by gqa_attention_ via network_context::get_padding_length.
        auto input_seq = ctx.get_input_window();
        long pad_len = count_leading_padding(input_seq, pad_token);
        network_context::set_padding_uniform(pad_len, 1);
        network_context::set_inference_mode(network_context::inference_mode::prefill);

        int next_tok = net(input_seq);
        generated.push_back(next_tok);
        ctx.add_token(next_tok);

        // Switch to incremental: no padding info needed (single-token inputs).
        network_context::clear_padding();
        network_context::set_inference_mode(network_context::inference_mode::incremental);

        for (size_t i = 1; i < target && !signal_handler::is_triggered(); ++i) {
            if (next_tok == eot_id) break;
            matrix<int, 0, 1> incr_input(1, 1);
            incr_input(0) = next_tok;
            next_tok = net(incr_input);
            generated.push_back(next_tok);
            ctx.add_token(next_tok);
        }
        network_context::reset();

        auto t1 = std::chrono::high_resolution_clock::now();
        cout << "Generated " << generated.size() << " tokens in "
            << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "s\n";

        cout << "\n--- Generated ---\n" << tokenizer.decode(generated, false) << "\n-----------------\n\n";
        cout << "--- Reference ---\n" << tokenizer.decode(verify_half, false) << "\n-----------------\n\n";

        size_t compare_len = std::min(verify_half.size(), generated.size());
        std::vector<int> ref(verify_half.begin(), verify_half.begin() + compare_len);
        std::vector<int> gen(generated.begin(), generated.begin() + compare_len);
        cout << "Comparing " << compare_len << " tokens\n";
        auto sim = compute_text_similarity(ref, gen);
        sim.print();
    }

    return 0;
}

// Fine-tuning mode (instruction tuning) -- placeholder
int run_fine_tune(const std::string& model_file, const std::string& /*tokenizer_file*/)
{
    cout << "=== FINE-TUNING MODE (placeholder) ===\n"
        << "This mode will load " << model_file << " and apply instruction fine-tuning.\n"
        << "Not yet implemented in this iteration.\n";
    return 0;
}

// Interactive prompt mode -- placeholder
int run_prompt(const std::string& model_file, const std::string& /*tokenizer_file*/)
{
    cout << "=== INTERACTIVE PROMPT MODE (placeholder) ===\n"
        << "This mode will load " << model_file << " for interactive generation.\n"
        << "Not yet implemented in this iteration.\n";
    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        signal_handler::setup();
        command_line_parser parser;

        parser.add_option("build-tokenizer", "Train the BPE tokenizer on the corpus and exit");
        parser.add_option("train", "Pre-train the model (requires a previously built tokenizer)");
        parser.add_option("fine-tune", "Instruction fine-tuning of a pre-trained model (placeholder)");
        parser.add_option("prompt", "Interactive generation loop (placeholder)");
        parser.add_option("generate", "Autoregressive generation sanity check");

        parser.add_option("external-data", "Path to external corpus file or directory", 1);
        parser.add_option("lines-per-segment", "Number of consecutive lines grouped as one segment (default: 10)", 1);
        parser.add_option("max-tokenizer-bytes", "Max corpus size in MB for BPE training (default: 200)", 1);

        parser.add_option("learning-rate", "Base learning rate (default: 1e-4)", 1);
        parser.add_option("batch-size", "Mini-batch size (default: 16)", 1);
        parser.add_option("patience", "Steps without progress before LR reduction (default: 20000)", 1);
        parser.add_option("max-epochs", "Maximum number of training epochs (default: 3)", 1);
        parser.add_option("weight-decay", "AdamW weight decay (default: 0.01)", 1);
        parser.add_option("beta1", "AdamW beta1 coefficient (default: 0.9)", 1);
        parser.add_option("beta2", "AdamW beta2 coefficient (default: 0.98)", 1);

        parser.add_option("model-file", "Model file path (default: dlib_euro_moe_model.dat)", 1);
        parser.add_option("tokenizer-file", "Tokenizer file path (default: dlib_euro_moe_tokenizer.vocab)", 1);

        parser.parse(argc, argv);

        const bool do_build_tok = parser.option("build-tokenizer");
        const bool do_train = parser.option("train");
        const bool do_fine_tune = parser.option("fine-tune");
        const bool do_prompt = parser.option("prompt");
        const bool do_generate = parser.option("generate");

        if (!do_build_tok && !do_train && !do_fine_tune && !do_prompt && !do_generate) {
            parser.print_options();
            cout << "\nExample usage:\n"
                << "  Build tokenizer : " << argv[0] << " --build-tokenizer --external-data corpus.txt\n"
                << "  Pre-train       : " << argv[0] << " --train --external-data corpus.txt\n"
                << "  Fine-tune       : " << argv[0] << " --fine-tune\n"
                << "  Interactive     : " << argv[0] << " --prompt\n"
                << "  Generate sample : " << argv[0] << " --generate\n";
            return 0;
        }

        const double learning_rate = get_option(parser, "learning-rate", 1e-4);
        const size_t batch_size = get_option(parser, "batch-size", 16);
        const long   patience = get_option(parser, "patience", 20000);
        const size_t max_epochs = get_option(parser, "max-epochs", 3);
        const double weight_decay = get_option(parser, "weight-decay", 0.01);
        const double beta1 = get_option(parser, "beta1", 0.9);
        const double beta2 = get_option(parser, "beta2", 0.98);
        const size_t lines_per_segment = get_option(parser, "lines-per-segment", 10);
        const size_t max_tok_mb = get_option(parser, "max-tokenizer-bytes", DEFAULT_MAX_TOKENIZER_BYTES / (1024 * 1024));
        const size_t max_tok_bytes = max_tok_mb * 1024 * 1024;

        const std::string tokenizer_file = get_option(parser, "tokenizer-file", std::string("dlib_euro_moe_tokenizer.vocab"));
        const std::string model_file = get_option(parser, "model-file", std::string("dlib_euro_moe_model.dat"));
        const std::string external_path = parser.option("external-data") ? parser.option("external-data").argument() : "";

        cout << "=== Configuration ===\n"
            << "  Model file       : " << model_file << "\n"
            << "  Tokenizer file   : " << tokenizer_file << "\n"
            << "  Architecture     : " << NUM_TOKENS << "/" << NUM_LAYERS << "/" << NUM_HEADS << "/" << NUM_KV_HEADS
            << "/" << EMBEDDING_DIM << "/" << NUM_EXPERTS << "/" << TOP_K << "\n"
            << "  Max seq len      : " << MAX_SEQ_LEN << "\n\n";

        if (do_build_tok) return run_build_tokenizer(external_path, tokenizer_file, max_tok_bytes);
        if (do_fine_tune) return run_fine_tune(model_file, tokenizer_file);
        if (do_prompt)    return run_prompt(model_file, tokenizer_file);

        bpe_tokenizer tokenizer;
        if (!file_exists(tokenizer_file)) {
            cerr << "Error: Tokenizer file not found: " << tokenizer_file << "\nRun --build-tokenizer first.\n";
            return 1;
        }
        cout << "Loading tokenizer from: " << tokenizer_file << "\n";
        deserialize(tokenizer_file) >> tokenizer;
        cout << "Tokenizer vocabulary size: " << tokenizer.get_vocab_size() << "\n";

        std::vector<std::string> text_segments;
        if (do_train) {
            cout << "Loading internal datasets...\n";
            std::vector<dataset_id> internal_ids = { dataset_id::GENERAL_KNOWLEDGE, dataset_id::PHYSICS_PARAGRAPHS };
            auto internal_segs = get_dataset_as_segments(internal_ids);
            text_segments.insert(text_segments.end(), internal_segs.begin(), internal_segs.end());
            cout << "Internal segments: " << internal_segs.size() << "\n";

            if (!external_path.empty()) {
                auto ext_segs = load_external_corpus_grouped(external_path, lines_per_segment);
                cout << "External segments: " << ext_segs.size() << "\n";
                text_segments.insert(text_segments.end(),
                    std::make_move_iterator(ext_segs.begin()), std::make_move_iterator(ext_segs.end()));
            }
            cout << "Total training segments: " << text_segments.size() << "\n";
        }

        std::vector<int> gpus{ 0 };
        using selected = gqa_moe_transformer_config<NUM_TOKENS, NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS,
            EMBEDDING_DIM, NUM_EXPERTS, TOP_K>;
        const std::string tokens_file = derive_tokens_filename(external_path);

        return run_pipeline<selected>(do_train, do_generate, learning_rate, batch_size, patience, max_epochs,
            weight_decay, beta1, beta2, model_file, tokenizer_file, tokens_file, text_segments, tokenizer, gpus);
    }
    catch (exception& e) { cerr << "Exception thrown: " << e.what() << endl; return 1; }
}