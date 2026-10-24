// Copyright (C) 2025 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_BPE_TOKENIZER_H
#define DLIB_BPE_TOKENIZER_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <thread>
#include <algorithm>
#include <sstream>
#include <list>
#include <queue>
#include <chrono>
#include <cstdint>
#include <climits>
#include <stdexcept>

#include "../base64.h"
#include "../serialize.h"
#include "bpe_tokenizer_abstract.h"

namespace dlib
{

    // How training and encoding cut a text into the segments merges are learned within.
    enum class bpe_segmentation
    {
        // Every whitespace byte is a segment of its own (the historical behaviour).
        whitespace_split = 0,
        // As above, except that a space directly followed by a non-whitespace byte opens
        // the segment that follows it, so that a word and the space before it can merge.
        space_prefixed = 1,
        // As space_prefixed, with every run of non-whitespace bytes further cut where the
        // byte class changes: letters (bytes 0x80 and above included), digits, and the rest.
        character_classes = 2
    };

    class bpe_tokenizer
    {
    public:
        bpe_tokenizer()
        {
            vocab_size = BPE_BASE_VOCAB_SIZE + special_token_list.size();

            // Initialize base vocabulary (bytes 0-255)
            reset_to_base_vocabulary();

            // Initialize special tokens
            initialize_special_tokens();
            rebuild_merge_index();
        }

        // Train the tokenizer on input data
        void train(const std::string& text, size_t max_vocab_size, size_t max_bytes = 0, bool verbose = false)
        {
            if (text.empty()) return;

            const auto t_start = std::chrono::steady_clock::now();
            const size_t data_size = (max_bytes > 0 && text.size() > max_bytes) ? max_bytes : text.size();

            // Calculate available merges (reserving space for special tokens)
            size_t num_merges = max_vocab_size - BPE_BASE_VOCAB_SIZE - special_token_list.size();

            if (num_merges <= 0) {
                if (verbose) {
                    std::cout << "Warning: max_vocab_size too small for any merges. Need at least "
                        << (BPE_BASE_VOCAB_SIZE + special_token_list.size() + 1) << " tokens." << std::endl;
                }
                return;
            }

            if (verbose) {
                std::cout << "Training BPE tokenizer on " << data_size << " bytes..." << std::endl;
                std::cout << "Target vocabulary size: " << max_vocab_size << std::endl;
                std::cout << "Base vocabulary: " << BPE_BASE_VOCAB_SIZE << " tokens" << std::endl;
                std::cout << "Special tokens: " << special_token_list.size() << " tokens" << std::endl;
                std::cout << "Available merges: " << num_merges << std::endl;
            }

            // Reset merges beyond base vocabulary
            reset_to_base_vocabulary();

            /*
                The merge table depends on the segments only through their frequencies, so each
                distinct segment is held once with its count. The pair counts after merge k+1 are
                then obtained from those after merge k by touching only the segments that held the
                merged pair, and the most frequent pair comes out of a priority queue. The result
                is the merge table of the plain algorithm, with the same pairs, order and
                tie-breaking.
            */
            segment_set segments;
            count_segments(text.data(), data_size, segmentation, segments);

            if (verbose) {
                std::cout << "Created " << segments.total << " segments for training ("
                    << segments.count.size() << " distinct)" << std::endl;
            }

            std::vector<std::pair<uint32_t, uint32_t>> learned;
            learn_merges(segments, num_merges, BPE_BASE_VOCAB_SIZE, learned, verbose);

            for (size_t merge_idx = 0; merge_idx < learned.size(); ++merge_idx) {
                Merge m;
                m.token_id = static_cast<int>(BPE_BASE_VOCAB_SIZE + merge_idx);
                m.left = static_cast<int>(learned[merge_idx].first);
                m.right = static_cast<int>(learned[merge_idx].second);
                m.pattern = merges[m.left].pattern;
                const auto& right_pattern = merges[m.right].pattern;
                m.pattern.insert(m.pattern.end(), right_pattern.begin(), right_pattern.end());
                merges.push_back(m);
            }

            // Update vocabulary size: base + special tokens + actual merges performed
            vocab_size = merges.size() + special_token_list.size();
            initialize_special_tokens();
            rebuild_merge_index();

            if (verbose) {
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
                std::cout << "\nTraining completed!" << std::endl;
                std::cout << "Base vocabulary: " << BPE_BASE_VOCAB_SIZE << " tokens" << std::endl;
                std::cout << "Merges performed: " << learned.size() << std::endl;
                std::cout << "Special tokens: " << special_token_list.size() << " tokens" << std::endl;
                std::cout << "Total vocabulary size: " << vocab_size << std::endl;
                std::cout << "Training time: " << std::fixed << std::setprecision(1) << secs << " s"
                    << std::defaultfloat << std::endl;
            }
        }

        // Encode text into tokens (parallelized for large inputs)
        std::vector<int> encode(const std::string& text) const
        {
            if (text.empty()) return {};

            long num_threads = compute_num_threads(text.size());
            if (num_threads <= 1) return encode_sequential(text);

            // Split text at newline boundaries into approximately equal chunks.
            // Newline splitting preserves pre-token boundaries so each chunk
            // produces the same token IDs as a single-pass sequential encode.
            std::vector<std::pair<size_t, size_t>> chunks;
            size_t chunk_target = text.size() / num_threads;
            size_t start = 0;
            for (long i = 0; i < num_threads; ++i) {
                if (start >= text.size()) break;
                size_t end = (i == num_threads - 1)
                    ? text.size()
                    : std::min(start + chunk_target, text.size());
                while (end < text.size() && text[end] != '\n') ++end;
                if (end < text.size()) ++end;
                chunks.push_back({ start, end - start });
                start = end;
            }
            if (start < text.size()) {
                if (!chunks.empty())
                    chunks.back().second = text.size() - chunks.back().first;
                else
                    chunks.push_back({ 0, text.size() });
            }

            long n_chunks = static_cast<long>(chunks.size());
            std::vector<std::vector<int>> chunk_tokens(n_chunks);
            std::vector<std::thread> workers;
            workers.reserve(n_chunks);

            for (long i = 0; i < n_chunks; ++i) {
                workers.emplace_back([this, &text, &chunks, &chunk_tokens, i]() {
                    chunk_tokens[i] = encode_range(text.data() + chunks[i].first, chunks[i].second);
                    });
            }
            for (auto& w : workers) w.join();

            size_t total = 0;
            for (auto& ct : chunk_tokens) total += ct.size();
            std::vector<int> result;
            result.reserve(total);
            for (auto& ct : chunk_tokens)
                result.insert(result.end(), ct.begin(), ct.end());
            return result;
        }

        // Decode tokens back to text (parallelized for large inputs)
        std::string decode(const std::vector<int>& tokens, bool display_special_tokens = true) const
        {
            if (tokens.empty()) return {};

            long num_threads = compute_num_threads(tokens.size());
            if (num_threads <= 1) return decode_sequential(tokens, display_special_tokens);

            long n_chunks = num_threads;
            size_t chunk_size = tokens.size() / n_chunks;
            std::vector<std::string> chunk_strings(n_chunks);
            std::vector<std::thread> workers;
            workers.reserve(n_chunks);

            for (long i = 0; i < n_chunks; ++i) {
                workers.emplace_back([this, &tokens, &chunk_strings, i, n_chunks,
                    chunk_size, display_special_tokens]() {
                        size_t start = i * chunk_size;
                        size_t end = (i == n_chunks - 1) ? tokens.size() : start + chunk_size;
                        std::vector<int> chunk(tokens.begin() + start, tokens.begin() + end);
                        chunk_strings[i] = decode_sequential(chunk, display_special_tokens);
                    });
            }
            for (auto& w : workers) w.join();

            size_t total_len = 0;
            for (auto& s : chunk_strings) total_len += s.size();
            std::string result;
            result.reserve(total_len);
            for (auto& s : chunk_strings) result += s;
            return result;
        }

        // Decode single token (for compatibility)
        std::string decode(int token, bool display_special_tokens = true) const
        {
            std::vector<int> tokens = { token };
            return decode(tokens, display_special_tokens);
        }

        // Get special token ID
        int get_special_token_id(const std::string& token) const
        {
            auto it = special_tokens.find(token);
            if (it != special_tokens.end()) {
                // Special tokens come after base vocab and merges
                return it->second;
            }
            throw std::runtime_error("Special token not found: " + token);
        }

        // Get vocabulary size
        size_t get_specials_size() const { return special_token_list.size(); }
        size_t get_vocab_size() const { return vocab_size; }
        size_t get_vocab_without_specials_size() const { return (vocab_size - special_token_list.size()); }

        // Segmentation used by train() and encode(). Merges learned under one segmentation do
        // not carry over to another, so changing it resets the tokenizer to its base vocabulary.
        void set_segmentation(bpe_segmentation mode)
        {
            if (mode == segmentation) return;
            segmentation = mode;
            reset_to_base_vocabulary();
            vocab_size = merges.size() + special_token_list.size();
            initialize_special_tokens();
            rebuild_merge_index();
        }
        bpe_segmentation get_segmentation() const { return segmentation; }

        // The merge table, as the (left, right) pair each merge joins, in the order they were learned.
        std::vector<std::pair<int, int>> get_merges() const
        {
            std::vector<std::pair<int, int>> result;
            result.reserve(merges.size() - BPE_BASE_VOCAB_SIZE);
            for (size_t i = BPE_BASE_VOCAB_SIZE; i < merges.size(); ++i)
                result.push_back({ merges[i].left, merges[i].right });
            return result;
        }

        // Rebuild the tokenizer from a merge table, as get_merges() returns it.
        void set_merges(const std::vector<std::pair<int, int>>& merge_list)
        {
            if (merge_list.size() > static_cast<size_t>(MAX_TOKEN_ID - BPE_BASE_VOCAB_SIZE))
                throw std::invalid_argument("bpe_tokenizer::set_merges: too many merges.");
            reset_to_base_vocabulary();
            for (size_t i = 0; i < merge_list.size(); ++i) {
                const int id = static_cast<int>(BPE_BASE_VOCAB_SIZE + i);
                const int left = merge_list[i].first, right = merge_list[i].second;
                if (left < 0 || right < 0 || left >= id || right >= id)
                    throw std::invalid_argument("bpe_tokenizer::set_merges: merge " + std::to_string(i)
                        + " refers to a token that does not precede it.");
                Merge m;
                m.token_id = id;
                m.left = left;
                m.right = right;
                m.pattern = merges[left].pattern;
                m.pattern.insert(m.pattern.end(), merges[right].pattern.begin(), merges[right].pattern.end());
                merges.push_back(m);
            }
            vocab_size = merges.size() + special_token_list.size();
            initialize_special_tokens();
            rebuild_merge_index();
        }

        // Serialization
        friend void serialize(const bpe_tokenizer& item, std::ostream& out)
        {
            // The historical segmentation keeps the historical format, byte for byte; any other
            // is written under its own tag so that a reader that predates it refuses the stream.
            if (item.segmentation == bpe_segmentation::whitespace_split) {
                serialize("bpe_tokenizer_", out);
            }
            else {
                serialize("bpe_tokenizer_segmented_", out);
                serialize(static_cast<int>(item.segmentation), out);
            }
            serialize(item.vocab_size, out);

            // Serialize only the merge entries beyond base vocabulary
            size_t num_merges = item.merges.size() - bpe_tokenizer::BPE_BASE_VOCAB_SIZE;
            serialize(num_merges, out);

            for (size_t i = bpe_tokenizer::BPE_BASE_VOCAB_SIZE; i < item.merges.size(); ++i) {
                const auto& m = item.merges[i];
                serialize(m.left, out);
                serialize(m.right, out);
                serialize(m.pattern, out);
            }
        }

        // Deserialization
        friend void deserialize(bpe_tokenizer& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "bpe_tokenizer_") {
                item.segmentation = bpe_segmentation::whitespace_split;
            }
            else if (version == "bpe_tokenizer_segmented_") {
                int mode = 0;
                deserialize(mode, in);
                if (mode != static_cast<int>(bpe_segmentation::whitespace_split) &&
                    mode != static_cast<int>(bpe_segmentation::space_prefixed) &&
                    mode != static_cast<int>(bpe_segmentation::character_classes))
                    throw serialization_error("Unknown segmentation found while deserializing dlib::bpe_tokenizer.");
                item.segmentation = static_cast<bpe_segmentation>(mode);
            }
            else {
                throw serialization_error("Unexpected version found while deserializing dlib::bpe_tokenizer.");
            }
            deserialize(item.vocab_size, in);

            // Initialize base vocabulary
            item.reset_to_base_vocabulary();

            // Deserialize merge entries
            size_t num_merges;
            deserialize(num_merges, in);
            for (size_t i = 0; i < num_merges; ++i) {
                Merge m;
                m.token_id = bpe_tokenizer::BPE_BASE_VOCAB_SIZE + i;
                deserialize(m.left, in);
                deserialize(m.right, in);
                deserialize(m.pattern, in);
                item.merges.push_back(m);
            }

            // Initialize special tokens
            item.initialize_special_tokens();
            item.rebuild_merge_index();
        }

    private:
        // Define special tokens
        const std::vector<std::string> special_token_list = {
            "<text>", "</text>", "<url>", "</url>",
            "<image>", "</image>", "<video>", "</video>",
            "<audio>", "</audio>", "<file>", "</file>",
            "<code>", "</code>", "<summary>", "</summary>",
            "<think>", "</think>", "<start>", "<end>",
            "<user>", "<bot>", "<system>", "<question>",
            "<answer>", "<search>", "<unk>", "<pad>"
        };
        static const int BPE_BASE_VOCAB_SIZE = 256;
        static const int MAX_TOKEN_ID = 65536;

        // Merge structure
        struct Merge {
            int token_id;
            int left;
            int right;
            std::vector<uint8_t> pattern;
        };

        // Data members
        size_t vocab_size;
        std::vector<Merge> merges;
        bpe_segmentation segmentation = bpe_segmentation::whitespace_split;

        // Special tokens handling
        std::map<std::string, int> special_tokens;
        std::unordered_map<int, std::string> special_token_map;

        // Rank of each merge, keyed by its pair, for the encoder. Rebuilt whenever the merges change.
        std::unordered_map<uint64_t, int> merge_rank;
        // False when a merge holds a segment delimiter: the per-segment encoder would then differ
        // from applying the merges over the whole text, and the reference encoder is used instead.
        bool segment_encoder_valid = true;

        static bool is_delimiter(uint8_t byte)
        {
            return byte == ' ' || byte == '\n' || byte == '\t' || byte == '\r';
        }

        // Class of a byte under character_classes: 0 delimiter, 1 letter, 2 digit, 3 other.
        // Bytes 0x80 and above count as letters, so that UTF-8 text stays within its words.
        static int byte_class(uint8_t byte)
        {
            if (is_delimiter(byte)) return 0;
            if (byte >= 0x80 || (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z')) return 1;
            if (byte >= '0' && byte <= '9') return 2;
            return 3;
        }

        /*
            Length of the segment that starts at text[i]: a lone delimiter byte, or a run of
            non-delimiter bytes, opened under space_prefixed and character_classes by the space
            directly before it. Under character_classes the run holds bytes of one class only.
        */
        static size_t segment_length(const char* text, size_t size, size_t i, bpe_segmentation mode)
        {
            size_t j = i;
            if (is_delimiter(static_cast<uint8_t>(text[i]))) {
                const bool opens_run = mode != bpe_segmentation::whitespace_split && text[i] == ' '
                    && i + 1 < size && !is_delimiter(static_cast<uint8_t>(text[i + 1]));
                if (!opens_run) return 1;
                ++j;
            }
            if (mode == bpe_segmentation::character_classes) {
                const int run_class = byte_class(static_cast<uint8_t>(text[j]));
                ++j;
                while (j < size && byte_class(static_cast<uint8_t>(text[j])) == run_class) ++j;
            }
            else {
                ++j;
                while (j < size && !is_delimiter(static_cast<uint8_t>(text[j]))) ++j;
            }
            return j - i;
        }

        // Whether a merge pattern fits in one segment, so that it can never join two of them.
        static bool fits_one_segment(const std::vector<uint8_t>& pattern, bpe_segmentation mode)
        {
            size_t k = 0;
            if (mode != bpe_segmentation::whitespace_split && pattern.size() > 1 && pattern[0] == ' ') k = 1;
            if (k == pattern.size()) return false;
            const int run_class = byte_class(pattern[k]);
            for (; k < pattern.size(); ++k) {
                const int c = byte_class(pattern[k]);
                if (c == 0) return false;
                if (mode == bpe_segmentation::character_classes && c != run_class) return false;
            }
            return true;
        }

        static uint64_t pair_key(uint32_t left, uint32_t right)
        {
            return (static_cast<uint64_t>(left) << 32) | right;
        }

        void reset_to_base_vocabulary()
        {
            merges.clear();
            merges.reserve(BPE_BASE_VOCAB_SIZE);
            for (int i = 0; i < BPE_BASE_VOCAB_SIZE; ++i) {
                Merge m;
                m.token_id = i;
                m.left = i;
                m.right = i;
                m.pattern.push_back(static_cast<uint8_t>(i));
                merges.push_back(m);
            }
        }

        void initialize_special_tokens()
        {
            special_tokens.clear();
            special_token_map.clear();

            // Initialize special tokens with sequential IDs
            int next_id = get_vocab_without_specials_size();

            for (const auto& token : special_token_list) {
                special_tokens[token] = next_id;
                special_token_map[next_id] = token;
                next_id++;
            }
        }

        void rebuild_merge_index()
        {
            merge_rank.clear();
            segment_encoder_valid = true;
            for (size_t i = BPE_BASE_VOCAB_SIZE; i < merges.size(); ++i) {
                const Merge& m = merges[i];
                const uint64_t key = pair_key(static_cast<uint32_t>(m.left), static_cast<uint32_t>(m.right));
                // The first merge of a pair is the one the reference encoder applies.
                merge_rank.insert(std::make_pair(key, static_cast<int>(i - BPE_BASE_VOCAB_SIZE)));
                if (!fits_one_segment(m.pattern, segmentation)) segment_encoder_valid = false;
            }
        }

        // Compute the number of worker threads for parallel encode/decode.
        // Progressive scaling: below 10k elements use 1 thread (sequential),
        // then scale proportionally up to (hardware_concurrency - 4).
        static long compute_num_threads(size_t input_size)
        {
            if (input_size < 10000) return 1;
            long hw = static_cast<long>(std::thread::hardware_concurrency());
            long max_threads = std::max(1L, hw - 4);
            long scaled = static_cast<long>(input_size / 10000);
            return std::max(1L, std::min(max_threads, scaled));
        }

        // Sequential encode, used directly for small inputs and as the per-chunk worker of the
        // parallel encode.
        std::vector<int> encode_sequential(const std::string& text) const
        {
            return encode_range(text.data(), text.size());
        }

        /*
            The merges never join across a segment boundary, so applying them over the whole text
            gives the same tokens as applying them inside each segment. Within a segment, the merge of
            lowest rank among the adjacent pairs is always the next one the whole-text algorithm
            would apply there, since a merge can only join tokens that existed before it. Each
            distinct segment is encoded once and its tokens reused from a cache.
        */
        std::vector<int> encode_range(const char* text, size_t size) const
        {
            if (size == 0) return {};
            if (!segment_encoder_valid) return encode_reference(std::string(text, size));

            std::vector<int> tokens;
            tokens.reserve(size / 2 + 16);
            std::unordered_map<std::string, std::vector<int>> cache;
            std::vector<int> work;
            const size_t max_cache_entries = 1 << 20;
            const size_t max_cached_length = 256;

            size_t i = 0;
            while (i < size) {
                const uint8_t byte = static_cast<uint8_t>(text[i]);
                const size_t len = segment_length(text, size, i, segmentation);
                const size_t j = i + len;

                if (len == 1) {
                    tokens.push_back(byte);
                }
                else if (len > max_cached_length) {
                    encode_segment(text + i, len, work);
                    tokens.insert(tokens.end(), work.begin(), work.end());
                }
                else {
                    std::string key(text + i, len);
                    auto it = cache.find(key);
                    if (it == cache.end()) {
                        encode_segment(text + i, len, work);
                        if (cache.size() >= max_cache_entries) cache.clear();
                        it = cache.emplace(std::move(key), work).first;
                    }
                    tokens.insert(tokens.end(), it->second.begin(), it->second.end());
                }
                i = j;
            }
            return tokens;
        }

        // Applies the merges to one segment, lowest rank first, every occurrence left to right.
        void encode_segment(const char* text, size_t len, std::vector<int>& work) const
        {
            work.resize(len);
            for (size_t k = 0; k < len; ++k) work[k] = static_cast<uint8_t>(text[k]);

            while (work.size() > 1) {
                int best = INT_MAX;
                for (size_t k = 0; k + 1 < work.size(); ++k) {
                    auto it = merge_rank.find(pair_key(static_cast<uint32_t>(work[k]), static_cast<uint32_t>(work[k + 1])));
                    if (it != merge_rank.end() && it->second < best) best = it->second;
                }
                if (best == INT_MAX) break;

                const Merge& m = merges[BPE_BASE_VOCAB_SIZE + best];
                size_t out = 0;
                for (size_t k = 0; k < work.size(); ) {
                    if (k + 1 < work.size() && work[k] == m.left && work[k + 1] == m.right) {
                        work[out++] = m.token_id;
                        k += 2;
                    }
                    else {
                        work[out++] = work[k++];
                    }
                }
                work.resize(out);
            }
        }

        // Applies every merge in order over the whole text. Used when a merge holds a delimiter.
        std::vector<int> encode_reference(const std::string& text) const
        {
            if (text.empty()) return {};
            std::vector<int> tokens;
            tokens.reserve(text.size());
            for (unsigned char byte : text)
                tokens.push_back(static_cast<int>(byte));

            for (size_t merge_idx = BPE_BASE_VOCAB_SIZE; merge_idx < merges.size(); merge_idx++) {
                const Merge& m = merges[merge_idx];
                std::vector<int> new_tokens;
                new_tokens.reserve(tokens.size());
                for (size_t i = 0; i < tokens.size(); ) {
                    if (i < tokens.size() - 1 && tokens[i] == m.left && tokens[i + 1] == m.right) {
                        new_tokens.push_back(m.token_id);
                        i += 2;
                    }
                    else {
                        new_tokens.push_back(tokens[i]);
                        i++;
                    }
                }
                tokens = std::move(new_tokens);
            }
            return tokens;
        }

        // Sequential decode: maps each token ID to its byte pattern.
        // Used directly for small inputs and as the per-chunk worker
        // for parallel decode.
        std::string decode_sequential(const std::vector<int>& tokens, bool display_special_tokens) const
        {
            std::vector<uint8_t> result;
            result.reserve(tokens.size() * 4);
            for (int token : tokens) {
                if (token >= 0 && token < static_cast<int>(merges.size())) {
                    const std::vector<uint8_t>& pattern = merges[token].pattern;
                    result.insert(result.end(), pattern.begin(), pattern.end());
                }
                else if (token >= static_cast<int>(merges.size()) &&
                    token < static_cast<int>(get_vocab_size())) {
                    if (display_special_tokens) {
                        auto it = special_token_map.find(token);
                        if (it != special_token_map.end()) {
                            const std::string& special_str = it->second;
                            result.insert(result.end(), special_str.begin(), special_str.end());
                        }
                    }
                }
                else {
                    const std::string tok_unk = "<unk>";
                    result.insert(result.end(), tok_unk.begin(), tok_unk.end());
                }
            }
            return std::string(result.begin(), result.end());
        }

        // Distinct training segments, their symbols stored end to end, and their frequencies.
        struct segment_set
        {
            std::vector<uint32_t> symbols;     // the symbols of every distinct segment
            std::vector<uint64_t> start;       // where each segment begins in symbols
            std::vector<uint32_t> length;      // its current length, which merges shorten
            std::vector<uint64_t> count;       // how many times it occurs
            uint64_t total = 0;                // segments in the text, delimiters included
        };

        static uint64_t mix_hash(uint64_t h)
        {
            h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return h;
        }

        /*
            Splits the text into segments as the encoder does and gathers the distinct segments
            of two or more bytes with their frequencies. Segments of one byte hold no pair and
            play no part in the merges.
        */
        static void count_segments(const char* text, size_t size, bpe_segmentation mode, segment_set& out)
        {
            struct slot { uint64_t hash; uint64_t offset; uint32_t length; uint32_t index; };
            std::vector<slot> table(1 << 16, slot{ 0, 0, 0, UINT32_MAX });
            size_t mask = table.size() - 1, used = 0;
            std::vector<uint64_t> first_offset;

            auto grow = [&]() {
                std::vector<slot> bigger(table.size() * 2, slot{ 0, 0, 0, UINT32_MAX });
                const size_t bigger_mask = bigger.size() - 1;
                for (const slot& s : table) {
                    if (s.index == UINT32_MAX) continue;
                    size_t p = s.hash & bigger_mask;
                    while (bigger[p].index != UINT32_MAX) p = (p + 1) & bigger_mask;
                    bigger[p] = s;
                }
                table.swap(bigger);
                mask = table.size() - 1;
            };

            out = segment_set();
            size_t i = 0;
            while (i < size) {
                const size_t len = segment_length(text, size, i, mode);
                const size_t j = i + len;
                ++out.total;
                if (len >= 2) {
                    uint64_t h = 0x9e3779b97f4a7c15ULL ^ len;
                    for (size_t k = i; k < j; ++k) h = (h ^ static_cast<uint8_t>(text[k])) * 0x100000001b3ULL;
                    h = mix_hash(h);
                    size_t p = h & mask;
                    for (;;) {
                        slot& s = table[p];
                        if (s.index == UINT32_MAX) {
                            if (first_offset.size() >= UINT32_MAX - 1)
                                throw std::runtime_error("bpe_tokenizer: too many distinct segments.");
                            s.hash = h; s.offset = i; s.length = static_cast<uint32_t>(len);
                            s.index = static_cast<uint32_t>(first_offset.size());
                            first_offset.push_back(i);
                            out.count.push_back(1);
                            if (++used * 2 > table.size()) grow();
                            break;
                        }
                        if (s.hash == h && s.length == len && std::equal(text + i, text + j, text + s.offset)) {
                            ++out.count[s.index];
                            break;
                        }
                        p = (p + 1) & mask;
                    }
                }
                i = j;
            }

            out.start.resize(first_offset.size());
            out.length.resize(first_offset.size());
            uint64_t total_symbols = 0;
            for (const slot& s : table) {
                if (s.index == UINT32_MAX) continue;
                out.length[s.index] = s.length;
            }
            for (size_t w = 0; w < first_offset.size(); ++w) {
                out.start[w] = total_symbols;
                total_symbols += out.length[w];
            }
            out.symbols.resize(total_symbols);
            for (size_t w = 0; w < first_offset.size(); ++w) {
                const uint8_t* src = reinterpret_cast<const uint8_t*>(text + first_offset[w]);
                for (uint32_t k = 0; k < out.length[w]; ++k) out.symbols[out.start[w] + k] = src[k];
            }
        }

        /*
            Learns up to num_merges merges over the distinct segments. Pair counts are updated
            from one merge to the next by difference over the segments the merged pair occurred
            in, which each pair lists. A max-heap ordered by (count descending, pair ascending)
            gives the next merge; an entry whose count is no longer current is discarded when it
            reaches the top, and every pair whose count changes gets a fresh entry.
        */
        static void learn_merges(
            segment_set& seg,
            size_t num_merges,
            uint32_t first_new_id,
            std::vector<std::pair<uint32_t, uint32_t>>& learned,
            bool verbose
        )
        {
            learned.clear();

            // Pair index: open addressing from the pair key to a dense pair number.
            std::vector<uint64_t> keys(1 << 16, UINT64_MAX);
            std::vector<uint32_t> slots(1 << 16, 0);
            size_t mask = keys.size() - 1;
            std::vector<int64_t> pair_count;
            std::vector<uint32_t> pair_left, pair_right, pair_dirty;
            std::vector<std::vector<uint32_t>> pair_where;

            auto grow = [&]() {
                std::vector<uint64_t> bigger_keys(keys.size() * 2, UINT64_MAX);
                std::vector<uint32_t> bigger_slots(keys.size() * 2, 0);
                const size_t bigger_mask = bigger_keys.size() - 1;
                for (size_t p = 0; p < keys.size(); ++p) {
                    if (keys[p] == UINT64_MAX) continue;
                    size_t q = mix_hash(keys[p]) & bigger_mask;
                    while (bigger_keys[q] != UINT64_MAX) q = (q + 1) & bigger_mask;
                    bigger_keys[q] = keys[p];
                    bigger_slots[q] = slots[p];
                }
                keys.swap(bigger_keys);
                slots.swap(bigger_slots);
                mask = keys.size() - 1;
            };

            auto pair_id = [&](uint32_t left, uint32_t right) -> uint32_t {
                const uint64_t key = pair_key(left, right);
                size_t p = mix_hash(key) & mask;
                for (;;) {
                    if (keys[p] == key) return slots[p];
                    if (keys[p] == UINT64_MAX) {
                        const uint32_t id = static_cast<uint32_t>(pair_count.size());
                        keys[p] = key;
                        slots[p] = id;
                        pair_count.push_back(0);
                        pair_left.push_back(left);
                        pair_right.push_back(right);
                        pair_dirty.push_back(UINT32_MAX);
                        pair_where.emplace_back();
                        if (pair_count.size() * 2 > keys.size()) grow();
                        return id;
                    }
                    p = (p + 1) & mask;
                }
            };

            struct entry { int64_t count; uint32_t left; uint32_t right; };
            struct entry_order {
                bool operator()(const entry& a, const entry& b) const {
                    if (a.count != b.count) return a.count < b.count;
                    if (a.left != b.left) return a.left > b.left;
                    return a.right > b.right;
                }
            };
            std::priority_queue<entry, std::vector<entry>, entry_order> heap;

            const size_t num_segments = seg.count.size();
            for (uint32_t w = 0; w < num_segments; ++w) {
                const uint32_t* s = &seg.symbols[seg.start[w]];
                const uint32_t len = seg.length[w];
                for (uint32_t k = 0; k + 1 < len; ++k) {
                    const uint32_t id = pair_id(s[k], s[k + 1]);
                    pair_count[id] += static_cast<int64_t>(seg.count[w]);
                    auto& where = pair_where[id];
                    if (where.empty() || where.back() != w) where.push_back(w);
                }
            }
            for (uint32_t id = 0; id < pair_count.size(); ++id)
                if (pair_count[id] > 0) heap.push(entry{ pair_count[id], pair_left[id], pair_right[id] });

            std::vector<uint32_t> visited(num_segments, UINT32_MAX);
            std::vector<std::pair<uint64_t, int>> local;
            std::vector<uint32_t> dirty;
            size_t live_pairs = heap.size();

            for (size_t merge_idx = 0; merge_idx < num_merges; ++merge_idx) {
                // Most frequent pair, ties going to the smallest pair.
                entry best{ 0, 0, 0 };
                while (!heap.empty()) {
                    const entry top = heap.top();
                    const uint32_t id = pair_id(top.left, top.right);
                    if (pair_count[id] != top.count) { heap.pop(); continue; }
                    best = top;
                    break;
                }
                if (best.count <= 0) {
                    if (verbose) std::cout << "\nNo more pairs to merge at iteration " << merge_idx << std::endl;
                    break;
                }
                heap.pop();

                const uint32_t left = best.left, right = best.right;
                const uint32_t new_id = static_cast<uint32_t>(first_new_id + merge_idx);
                if (new_id >= static_cast<uint32_t>(MAX_TOKEN_ID)) break;
                learned.push_back({ left, right });

                if (verbose && (merge_idx % 1000 == 0 || merge_idx < 10)) {
                    std::cout << "Merge " << merge_idx << ": (" << left << ", " << right << ") -> " << new_id
                        << " (occurrences: " << best.count << ")" << std::endl;
                }

                const uint32_t merged_id = pair_id(left, right);
                std::vector<uint32_t> where;
                where.swap(pair_where[merged_id]);
                dirty.clear();

                for (uint32_t w : where) {
                    if (visited[w] == merge_idx) continue;
                    visited[w] = static_cast<uint32_t>(merge_idx);

                    uint32_t* s = &seg.symbols[seg.start[w]];
                    const uint32_t len = seg.length[w];
                    bool present = false;
                    for (uint32_t k = 0; k + 1 < len; ++k)
                        if (s[k] == left && s[k + 1] == right) { present = true; break; }
                    if (!present) continue;

                    local.clear();
                    for (uint32_t k = 0; k + 1 < len; ++k) local.push_back({ pair_key(s[k], s[k + 1]), -1 });
                    uint32_t out = 0;
                    for (uint32_t k = 0; k < len; ) {
                        if (k + 1 < len && s[k] == left && s[k + 1] == right) { s[out++] = new_id; k += 2; }
                        else s[out++] = s[k++];
                    }
                    seg.length[w] = out;
                    for (uint32_t k = 0; k + 1 < out; ++k) local.push_back({ pair_key(s[k], s[k + 1]), +1 });

                    std::sort(local.begin(), local.end());
                    const int64_t freq = static_cast<int64_t>(seg.count[w]);
                    for (size_t a = 0; a < local.size(); ) {
                        size_t b = a;
                        int delta = 0;
                        while (b < local.size() && local[b].first == local[a].first) delta += local[b++].second;
                        if (delta != 0) {
                            const uint32_t id = pair_id(static_cast<uint32_t>(local[a].first >> 32),
                                static_cast<uint32_t>(local[a].first & 0xffffffffu));
                            pair_count[id] += delta * freq;
                            if (delta > 0) {
                                auto& list = pair_where[id];
                                if (list.empty() || list.back() != w) list.push_back(w);
                            }
                            if (pair_dirty[id] != merge_idx) {
                                pair_dirty[id] = static_cast<uint32_t>(merge_idx);
                                dirty.push_back(id);
                            }
                        }
                        a = b;
                    }
                }

                pair_count[merged_id] = 0;
                for (uint32_t id : dirty)
                    if (pair_count[id] > 0) heap.push(entry{ pair_count[id], pair_left[id], pair_right[id] });

                // Stale entries pile up; rebuild from the current counts when they dominate.
                if (heap.size() > 4 * live_pairs + (1u << 20)) {
                    std::vector<entry> fresh;
                    for (uint32_t id = 0; id < pair_count.size(); ++id)
                        if (pair_count[id] > 0) fresh.push_back(entry{ pair_count[id], pair_left[id], pair_right[id] });
                    live_pairs = fresh.size();
                    heap = std::priority_queue<entry, std::vector<entry>, entry_order>(entry_order(), std::move(fresh));
                }
            }
        }
    };

}

#endif // DLIB_BPE_TOKENIZER_H

