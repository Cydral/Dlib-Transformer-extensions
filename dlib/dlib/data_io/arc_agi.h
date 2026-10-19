// Copyright (C) 2026  Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_ARC_AGI_H_
#define DLIB_ARC_AGI_H_

#include "arc_agi_abstract.h"
#include <array>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <random>
#include "../matrix.h"
#include "../dir_nav.h"
#include "../serialize.h"

namespace dlib
{

    // ----------------------------------------------------------------------------------------
    // Type aliases and constants
    // ----------------------------------------------------------------------------------------

    /*!
        Type aliases for ARC-AGI data structures. Grids are represented as matrices
        of unsigned char values (0-9), and token sequences are column vectors of int
        (aligned with the int32 input type expected by all transformer networks).
    !*/
    using arc_grid_t = matrix<unsigned char>;
    using arc_token_sequence_t = matrix<int, 0, 1>;

    // ----------------------------------------------------------------------------------------
    // Token vocabulary
    // ----------------------------------------------------------------------------------------

    /*!
        Token vocabulary for the Hierarchical Reasoning Model. The vocabulary includes:
        - COLOR_0 to COLOR_9: Grid cell colors (10 values)
        - TOKEN_SEP_IO: Separator between input and output grids
        - TOKEN_SEP_PAIR: Separator between demonstration pairs
        - TOKEN_QUERY_START: Marks the beginning of a test query
        - TOKEN_GEN_START: Marks the beginning of generation phase
        - TOKEN_END_OF_OUTPUT: Marks the end of generated output
        - TOKEN_PADDING: Padding token for variable-length sequences
        - TOKEN_ROW_END: Marks the end of a grid row (for dimension encoding)
    !*/
    enum arc_token_id : int
    {
        COLOR_0 = 0, COLOR_1 = 1, COLOR_2 = 2, COLOR_3 = 3, COLOR_4 = 4,
        COLOR_5 = 5, COLOR_6 = 6, COLOR_7 = 7, COLOR_8 = 8, COLOR_9 = 9,
        TOKEN_SEP_IO = 10,
        TOKEN_SEP_PAIR = 11,
        TOKEN_QUERY_START = 12,
        TOKEN_GEN_START = 13,
        TOKEN_END_OF_OUTPUT = 14,
        TOKEN_PADDING = 15,
        TOKEN_ROW_END = 16
    };

    /*!
        Vocabulary size constants for the token set.
    !*/
    constexpr int ARC_VOCAB_SIZE_COLORS = 10;
    constexpr int ARC_VOCAB_SIZE_TOTAL = arc_token_id::TOKEN_ROW_END + 1;

    // ----------------------------------------------------------------------------------------
    // ARC-AGI task data structures
    // ----------------------------------------------------------------------------------------

    /*!
        Represents a single input-output pair in an ARC-AGI task. Each pair consists
        of an input grid and its corresponding output grid, along with their dimensions.
    !*/
    struct arc_task_pair
    {
        arc_grid_t input;
        arc_grid_t output;
        long input_rows;
        long input_cols;
        long output_rows;
        long output_cols;

        friend void serialize(const arc_task_pair& item, std::ostream& out)
        {
            dlib::serialize(item.input, out);
            dlib::serialize(item.output, out);
            dlib::serialize(item.input_rows, out);
            dlib::serialize(item.input_cols, out);
            dlib::serialize(item.output_rows, out);
            dlib::serialize(item.output_cols, out);
        }

        friend void deserialize(arc_task_pair& item, std::istream& in)
        {
            dlib::deserialize(item.input, in);
            dlib::deserialize(item.output, in);
            dlib::deserialize(item.input_rows, in);
            dlib::deserialize(item.input_cols, in);
            dlib::deserialize(item.output_rows, in);
            dlib::deserialize(item.output_cols, in);
        }
    };

    /*!
        Represents a complete ARC-AGI task. Each task contains:
        - A unique task identifier
        - A set of training demonstration pairs
        - A set of test pairs (where outputs are to be predicted)
    !*/
    struct arc_task
    {
        std::string task_id;
        std::vector<arc_task_pair> train_pairs;
        std::vector<arc_task_pair> test_pairs;

        friend void serialize(const arc_task& item, std::ostream& out)
        {
            dlib::serialize(item.task_id, out);
            dlib::serialize(item.train_pairs, out);
            dlib::serialize(item.test_pairs, out);
        }

        friend void deserialize(arc_task& item, std::istream& in)
        {
            dlib::deserialize(item.task_id, in);
            dlib::deserialize(item.train_pairs, in);
            dlib::deserialize(item.test_pairs, in);
        }
    };

    // ----------------------------------------------------------------------------------------
    // Internal JSON parsing utilities
    // ----------------------------------------------------------------------------------------

    namespace internal
    {
        using raw_arc_grid_t = std::vector<std::vector<int>>;

        // ------------------------------------------------------------------------------------

        inline std::string read_file_to_string(const std::string& path)
            /*!
                ensures
                    - Reads the entire contents of a file and returns it as a string
                    - Throws std::runtime_error if the file cannot be opened
            !*/
        {
            std::ifstream file(path);
            if (!file.is_open())
                throw std::runtime_error("Failed to open file: " + path);
            std::stringstream buffer;
            buffer << file.rdbuf();
            return buffer.str();
        }

        // ------------------------------------------------------------------------------------

        inline std::vector<int> parse_int_array(const std::string& str)
            /*!
                ensures
                    - Parses a comma-separated string of integers
                    - Returns a vector containing the parsed integers
                    - Whitespace around numbers is automatically stripped
            !*/
        {
            std::vector<int> result;
            std::stringstream ss(str);
            std::string segment;
            while (std::getline(ss, segment, ','))
            {
                segment.erase(0, segment.find_first_not_of(" \t\n\r"));
                segment.erase(segment.find_last_not_of(" \t\n\r") + 1);
                if (!segment.empty())
                    result.push_back(std::stoi(segment));
            }
            return result;
        }

        // ------------------------------------------------------------------------------------

        inline raw_arc_grid_t parse_arc_grid(std::string::const_iterator& it,
            const std::string::const_iterator& end)
            /*!
                ensures
                    - Parses a 2D grid from JSON array-of-arrays format
                    - Advances the iterator 'it' past the parsed content
                    - Returns a vector of vectors representing the grid rows
                    - Throws std::runtime_error on malformed input
            !*/
        {
            raw_arc_grid_t grid;

            it = std::find(it, end, '[');
            if (it == end) return grid;
            ++it;

            while (it != end && std::isspace(static_cast<unsigned char>(*it))) ++it;

            if (it == end || *it != '[') return grid;

            while (it != end)
            {
                while (it != end && std::isspace(static_cast<unsigned char>(*it))) ++it;

                if (it == end || *it == ']') break;

                if (*it != '[') { ++it; continue; }
                ++it;

                auto inner_end = std::find(it, end, ']');
                if (inner_end == end)
                    throw std::runtime_error("Missing inner array closing bracket");

                std::string row_str(it, inner_end);
                auto row = parse_int_array(row_str);

                if (!row.empty())
                    grid.push_back(row);

                it = inner_end;
                ++it;

                while (it != end && (*it == ' ' || *it == ',' || *it == '\n' ||
                    *it == '\r' || *it == '\t'))
                    ++it;
            }

            if (it != end && *it == ']') ++it;

            return grid;
        }

        // ------------------------------------------------------------------------------------

        inline std::string::const_iterator find_key_value_start(
            const std::string& /*content*/,
            const std::string& key,
            std::string::const_iterator start_it,
            std::string::const_iterator end_it)
            /*!
                ensures
                    - Searches for a JSON key-value pair in [start_it, end_it)
                    - Returns an iterator pointing to the first character of the value
                    - Returns end_it if the key is not found within the specified range
                    - Bounded search prevents false matches from sibling JSON objects
            !*/
        {
            std::string search_str = "\"" + key + "\":";
            auto pos = std::search(start_it, end_it,
                search_str.begin(), search_str.end());
            if (pos == end_it) return end_it;
            pos += static_cast<std::ptrdiff_t>(search_str.length());
            while (pos != end_it && std::isspace(static_cast<unsigned char>(*pos))) ++pos;
            return pos;
        }

        // ------------------------------------------------------------------------------------

        inline std::string extract_task_id_from_filename(const std::string& filename)
            /*!
                ensures
                    - Extracts the task ID from a filename by removing the file extension
                    - If no extension is found, returns the filename unchanged
            !*/
        {
            size_t dot_pos = filename.find_last_of('.');
            if (dot_pos == std::string::npos)
                return filename;
            return filename.substr(0, dot_pos);
        }

    } // namespace internal

    static std::string token_to_string(int token)
    {
        switch (token)
        {
        case TOKEN_SEP_IO:        return "[SEP_IO]";
        case TOKEN_SEP_PAIR:      return "[SEP_PAIR]";
        case TOKEN_QUERY_START:   return "[QUERY_START]";
        case TOKEN_GEN_START:     return "[GEN_START]";
        case TOKEN_END_OF_OUTPUT: return "[END_OF_OUTPUT]";
        case TOKEN_PADDING:       return "[PAD]";
        case TOKEN_ROW_END:       return "[ROW_END]";
        default:
            if (token >= COLOR_0 && token <= COLOR_9)
                return std::to_string(token);
            return "[UNK:" + std::to_string(token) + "]";
        }
    }

    static void print_sequence(const std::string& label,
        const std::vector<int>& S,
        long from = 0, long to = -1,
        int tokens_per_line = 20)
    {
        if (to < 0) to = static_cast<long>(S.size()) - 1;
        std::cout << "  [" << label << "] (" << (to - from + 1) << " tokens):\n";
        int col = 0;
        for (long i = from; i <= to && i < static_cast<long>(S.size()); ++i)
        {
            if (col == 0) std::cout << "    ";
            std::cout << token_to_string(S[i]);
            if (i < to) std::cout << " ";
            col++;
            if (col >= tokens_per_line) { std::cout << "\n"; col = 0; }
        }
        if (col > 0) std::cout << "\n";
    }

    static void print_window_and_target(
        const arc_token_sequence_t& window,
        long target,
        long sample_idx,
        long window_len)
    {
        std::cout << "    Sample #" << sample_idx << " => target: "
            << token_to_string(static_cast<int>(target)) << "\n";
        std::cout << "      Window: ";
        int col = 0;
        for (long i = 0; i < window_len; ++i)
        {
            std::cout << token_to_string(window(i));
            if (i < window_len - 1) std::cout << " ";
            col++;
            if (col >= 20 && i < window_len - 1) { std::cout << "\n             "; col = 0; }
        }
        std::cout << "\n";
    }

    // ----------------------------------------------------------------------------------------
    // arc_agi_manager class
    // ----------------------------------------------------------------------------------------

    /*!
        The arc_agi_manager class provides functionality to:
        - Load ARC-AGI tasks from JSON files
        - Manage training and evaluation datasets
        - Convert grids to token sequences for LLM training
        - Generate training batches with sliding window context
        - Serialize and deserialize task data

        THREAD SAFETY
            This class is not thread-safe. External synchronization is required
            if accessing the same instance from multiple threads.

        TOKENIZATION STRATEGY
            Grids are tokenized row-by-row with TOKEN_ROW_END markers to preserve
            dimensional information. This allows the model to learn the structure
            of non-square grids (ranging from 1x1 to 30x30) without explicit
            dimension encoding.
    !*/
    class arc_agi_manager
    {
    private:
        std::vector<arc_task> training_tasks;
        std::vector<arc_task> evaluation_tasks;
        std::map<std::string, size_t> training_task_id_map;
        std::map<std::string, size_t> evaluation_task_id_map;

        // ------------------------------------------------------------------------------------

        static void append_flat_grid(std::vector<int>& sequence, const arc_grid_t& grid)
            /*!
                requires
                    - grid contains valid color values (0-9)
                ensures
                    - Appends the grid to the sequence in row-major order
                    - Each row is terminated with TOKEN_ROW_END
                    - This encoding preserves grid dimensions for reconstruction
            !*/
        {
            for (long r = 0; r < grid.nr(); ++r)
            {
                for (long c = 0; c < grid.nc(); ++c) {
                    int v = grid(r, c);
                    DLIB_CASSERT(v >= 0 && v <= 9);
                    sequence.push_back(v);
                }
                sequence.push_back(TOKEN_ROW_END);
            }
        }

        // ------------------------------------------------------------------------------------

        static arc_grid_t to_dlib_matrix(const internal::raw_arc_grid_t& grid)
            /*!
                requires
                    - grid is a valid 2D array with consistent row lengths
                    - all values are in the range [0, 9]
                ensures
                    - Converts a raw vector-of-vectors grid to a dlib matrix
                    - Returns an empty matrix if the input grid is empty
                throws
                    - DLIB_CASSERT if row lengths are inconsistent
                    - DLIB_CASSERT if pixel values are outside [0, 9]
            !*/
        {
            if (grid.empty()) return arc_grid_t(0, 0);
            long rows = static_cast<long>(grid.size());
            long cols = static_cast<long>(grid[0].size());

            DLIB_CASSERT(rows >= 1 && rows <= 30);
            DLIB_CASSERT(cols >= 1 && cols <= 30);

            arc_grid_t mat(rows, cols);

            for (long r = 0; r < rows; ++r)
            {
                DLIB_CASSERT(static_cast<long>(grid[r].size()) == cols,
                    "Inconsistent column size in grid");
                for (long c = 0; c < cols; ++c)
                {
                    DLIB_CASSERT(grid[r][c] >= 0 && grid[r][c] <= 9,
                        "Invalid pixel value (must be 0-9)");
                    mat(r, c) = static_cast<unsigned char>(grid[r][c]);
                }
            }
            return mat;
        }

        // ------------------------------------------------------------------------------------

        arc_task parse_arc_task_from_content(const std::string& content,
            const std::string& filename)
            /*!
                ensures
                    - Parses a complete ARC task from JSON content
                    - Returns an arc_task structure with all training and test pairs
                    - Task ID is extracted from the filename
                throws
                    - std::runtime_error on malformed JSON or missing required fields
            !*/
        {
            arc_task task;
            task.task_id = internal::extract_task_id_from_filename(filename);

            auto parse_pairs = [&](const std::string& key,
                std::vector<arc_task_pair>& pairs)
                {
                    // Find the top-level array for this key (scoped to full content)
                    auto it = internal::find_key_value_start(
                        content, key, content.begin(), content.end());
                    if (it == content.end() || *it != '[')
                        throw std::runtime_error("'" + key + "' array not found");
                    ++it;

                    while (it != content.end())
                    {
                        while (it != content.end() && std::isspace(static_cast<unsigned char>(*it))) ++it;
                        if (it == content.end() || *it == ']') break;

                        if (*it != '{') { ++it; continue; }

                        auto object_start = it;
                        ++it;

                        // Find the matching closing brace for this object
                        int brace_depth = 1;
                        auto object_end = it;
                        while (object_end != content.end() && brace_depth > 0)
                        {
                            if (*object_end == '{') ++brace_depth;
                            else if (*object_end == '}') --brace_depth;
                            ++object_end;
                        }

                        if (object_end == content.end())
                            throw std::runtime_error("Missing object closing bracket");

                        arc_task_pair pair;

                        // Parse "input" field — search bounded to this object
                        auto input_it = internal::find_key_value_start(
                            content, "input", object_start, object_end);
                        if (input_it == object_end)
                            throw std::runtime_error("'input' not found in " + key + " object");

                        auto raw_input = internal::parse_arc_grid(input_it, object_end);
                        pair.input = to_dlib_matrix(raw_input);
                        pair.input_rows = pair.input.nr();
                        pair.input_cols = pair.input.nc();

                        // Parse "output" field — search bounded to this object,
                        // starting from where input_it was left after parse_arc_grid
                        auto output_it = internal::find_key_value_start(
                            content, "output", input_it, object_end);
                        if (output_it == object_end)
                            throw std::runtime_error("'output' not found in " + key + " object");

                        auto raw_output = internal::parse_arc_grid(output_it, object_end);
                        pair.output = to_dlib_matrix(raw_output);
                        pair.output_rows = pair.output.nr();
                        pair.output_cols = pair.output.nc();

                        pairs.push_back(pair);
                        it = object_end;
                    }
                };

            parse_pairs("train", task.train_pairs);
            parse_pairs("test", task.test_pairs);
            return task;
        }

        // ------------------------------------------------------------------------------------

        std::vector<arc_task> load_all_tasks(const std::string& directory_path,
            std::map<std::string, size_t>& id_map)
            /*!
                ensures
                    - Loads all .json files from the specified directory
                    - Each file is parsed as an ARC task
                    - Returns a vector of successfully loaded tasks
                    - Populates id_map with task_id to index mappings
                    - Outputs diagnostic information to stdout/stderr
            !*/
        {
            std::vector<arc_task> tasks;
            std::cout << "Loading tasks from: " << directory_path << std::endl;

            try {
                const dlib::directory dir(directory_path);
                std::vector<dlib::file> all_files = dir.get_files();

                std::cout << "Found " << all_files.size() << " files in directory" << std::endl;

                std::vector<dlib::file> json_files;
                for (const auto& file : all_files)
                {
                    const std::string& filename = file.name();
                    if (filename.size() >= 5 &&
                        filename.substr(filename.size() - 5) == ".json")
                    {
                        json_files.push_back(file);
                    }
                }

                std::cout << "Found " << json_files.size() << " .json files" << std::endl;

                if (json_files.empty()) {
                    std::cout << "WARNING: No .json files found in "
                        << directory_path << std::endl;
                    return tasks;
                }

                size_t success_count = 0;
                size_t error_count = 0;

                for (const auto& file : json_files)
                {
                    try {
                        std::string content = internal::read_file_to_string(file.full_name());
                        arc_task task = parse_arc_task_from_content(content, file.name());
                        id_map[task.task_id] = tasks.size();
                        tasks.push_back(task);
                        ++success_count;
                    }
                    catch (const std::exception& e) {
                        std::cerr << "ERROR parsing " << file.name()
                            << ": " << e.what() << std::endl;
                        ++error_count;
                    }
                }

                std::cout << "Successfully loaded " << success_count << " tasks" << std::endl;
                if (error_count > 0) {
                    std::cout << "Failed to load " << error_count << " tasks" << std::endl;
                }

            }
            catch (const dlib::directory::dir_not_found& e) {
                std::cerr << "ERROR: Directory not found: " << directory_path << std::endl;
                std::cerr << "Details: " << e.info << std::endl;
            }
            catch (const dlib::directory::listing_error& e) {
                std::cerr << "ERROR: Cannot list directory: " << directory_path << std::endl;
                std::cerr << "Details: " << e.info << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "ERROR during directory navigation: " << e.what() << std::endl;
            }
            return tasks;
        }

    public:
        arc_agi_manager() = default;

        // ------------------------------------------------------------------------------------

        void load_data(const std::string& training_path,
            const std::string& evaluation_path)
            /*!
                ensures
                    - Loads all ARC tasks from training and evaluation directories
                    - Clears any previously loaded data
                    - Outputs a summary of loaded tasks to stdout
            !*/
        {
            training_task_id_map.clear();
            evaluation_task_id_map.clear();

            training_tasks = load_all_tasks(training_path, training_task_id_map);
            evaluation_tasks = load_all_tasks(evaluation_path, evaluation_task_id_map);

            std::cout << "--- ARC Data Loading Summary ---" << std::endl;
            std::cout << "Loaded " << training_tasks.size() << " training tasks" << std::endl;
            std::cout << "Loaded " << evaluation_tasks.size() << " evaluation tasks" << std::endl;
            std::cout << "--------------------------------" << std::endl;
        }

        // ------------------------------------------------------------------------------------

        const arc_task& get_training_task(size_t index) const
        {
            DLIB_CASSERT(index < training_tasks.size(),
                "Training task index out of bounds"
                << "\n\tRequested index: " << index
                << "\n\tAvailable tasks: " << training_tasks.size());
            return training_tasks[index];
        }

        const arc_task& get_evaluation_task(size_t index) const
        {
            DLIB_CASSERT(index < evaluation_tasks.size(),
                "Evaluation task index out of bounds");
            return evaluation_tasks[index];
        }

        const arc_task& get_training_task_by_id(const std::string& task_id) const
        {
            auto it = training_task_id_map.find(task_id);
            if (it == training_task_id_map.end())
                throw std::runtime_error("Training task ID not found: " + task_id);
            return training_tasks[it->second];
        }

        const arc_task& get_evaluation_task_by_id(const std::string& task_id) const
        {
            auto it = evaluation_task_id_map.find(task_id);
            if (it == evaluation_task_id_map.end())
                throw std::runtime_error("Evaluation task ID not found: " + task_id);
            return evaluation_tasks[it->second];
        }

        size_t num_training_tasks()   const { return training_tasks.size(); }
        size_t num_evaluation_tasks() const { return evaluation_tasks.size(); }

        // ------------------------------------------------------------------------------------

        void serialize(std::ostream& out) const
        {
            dlib::serialize("arc_agi_v1", out);
            dlib::serialize(training_tasks, out);
            dlib::serialize(evaluation_tasks, out);
            dlib::serialize(training_task_id_map, out);
            dlib::serialize(evaluation_task_id_map, out);
        }

        void deserialize(std::istream& in)
        {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "arc_agi_v1")
                throw serialization_error("Unexpected version in arc_agi_manager");
            dlib::deserialize(training_tasks, in);
            dlib::deserialize(evaluation_tasks, in);
            dlib::deserialize(training_task_id_map, in);
            dlib::deserialize(evaluation_task_id_map, in);
        }

        // ----------------------------------------------------------------------------------------
        // Tokenization for LLM-style training
        // ----------------------------------------------------------------------------------------

        static arc_token_sequence_t tokenize_input_context(const arc_task& task,
            const arc_task_pair& test_pair)
            /*!
                ensures
                    - Creates a token sequence representing the input context for a test pair
                    - Format: [train_input SEP_IO train_output SEP_PAIR]* QUERY_START test_input GEN_START
                    - Each grid is tokenized with TOKEN_ROW_END markers preserving dimensions
                    - Returns a column vector of int tokens
            !*/
        {
            std::vector<int> sequence;

            for (const auto& pair : task.train_pairs)
            {
                append_flat_grid(sequence, pair.input);
                sequence.push_back(TOKEN_SEP_IO);
                append_flat_grid(sequence, pair.output);
                sequence.push_back(TOKEN_SEP_PAIR);
            }

            sequence.push_back(TOKEN_QUERY_START);
            append_flat_grid(sequence, test_pair.input);
            sequence.push_back(TOKEN_GEN_START);

            arc_token_sequence_t result(static_cast<long>(sequence.size()));
            for (long i = 0; i < static_cast<long>(sequence.size()); ++i)
                result(i) = sequence[static_cast<size_t>(i)];
            return result;
        }

        // ------------------------------------------------------------------------------------

        static arc_token_sequence_t tokenize_target_output(const arc_task_pair& test_pair)
            /*!
                ensures
                    - Creates a token sequence for the target output grid
                    - Format: output_grid END_OF_OUTPUT
                    - Returns a column vector of int tokens
            !*/
        {
            std::vector<int> sequence;
            append_flat_grid(sequence, test_pair.output);
            sequence.push_back(TOKEN_END_OF_OUTPUT);

            arc_token_sequence_t result(static_cast<long>(sequence.size()));
            for (long i = 0; i < static_cast<long>(sequence.size()); ++i)
                result(i) = sequence[static_cast<size_t>(i)];
            return result;
        }

        // ------------------------------------------------------------------------------------

        static void prepare_training_data_pair_only(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch,
            bool debug = false)
        {
            DLIB_CASSERT(window_len > 1, "Window length must be greater than 1");

            // Collect ALL pairs (train + test) as independent input/output sequences
            std::vector<const arc_task_pair*> all_pairs;
            for (const auto& p : task.train_pairs) all_pairs.push_back(&p);
            for (const auto& p : task.test_pairs)  all_pairs.push_back(&p);

            if (debug)
            {
                std::cout << "\n=== prepare_training_data_pair_only DEBUG ===\n";
                std::cout << "Task: " << task.task_id
                    << "  train_pairs=" << task.train_pairs.size()
                    << "  test_pairs=" << task.test_pairs.size()
                    << "  window_len=" << window_len << "\n";
            }

            for (size_t pair_idx = 0; pair_idx < all_pairs.size(); ++pair_idx)
            {
                const arc_task_pair& pair = *all_pairs[pair_idx];

                // Build sequence: input_grid GEN_START output_grid END_OF_OUTPUT
                std::vector<int> S;
                append_flat_grid(S, pair.input);
                S.push_back(TOKEN_GEN_START);
                append_flat_grid(S, pair.output);
                S.push_back(TOKEN_END_OF_OUTPUT);

                const long L = static_cast<long>(S.size());

                // Mark output zone: tokens after GEN_START up to END_OF_OUTPUT
                long gen_start_pos = -1;
                for (long i = 0; i < L; ++i)
                    if (S[i] == TOKEN_GEN_START) { gen_start_pos = i; break; }

                if (gen_start_pos < 0) continue;

                if (debug)
                {
                    std::cout << "  Pair #" << pair_idx
                        << "  input=" << pair.input_rows << "x" << pair.input_cols
                        << "  output=" << pair.output_rows << "x" << pair.output_cols
                        << "  L=" << L
                        << "  gen_start_pos=" << gen_start_pos << "\n";

                    // Print full sequence
                    std::cout << "    Sequence: ";
                    int col = 0;
                    for (long i = 0; i < L; ++i)
                    {
                        bool in_out = (i > gen_start_pos && S[i] != TOKEN_END_OF_OUTPUT);
                        std::string t = token_to_string(S[i]);
                        if (in_out) t = "*" + t + "*";
                        std::cout << t << " ";
                        if (++col >= 20) { std::cout << "\n              "; col = 0; }
                    }
                    std::cout << "\n";
                }

                // Sliding window over the full sequence.
                // Loss only on output zone (pos > gen_start_pos).
                // Always include the GEN_START anchor sample.
                long samples_added = 0;
                long samples_skipped = 0;

                for (long pos = 0; pos < L - 1; ++pos)
                {
                    bool is_anchor = (S[pos] == TOKEN_GEN_START);
                    bool is_in_output = (pos > gen_start_pos);

                    if (!is_anchor && !is_in_output)
                    {
                        samples_skipped++;
                        continue;
                    }

                    // Build window of size window_len ending at pos,
                    // left-padded if needed (max 25% padding)
                    const long max_padding = window_len / 4;
                    long padding_needed = std::max(0L, window_len - (pos + 1));

                    if (padding_needed > max_padding)
                    {
                        samples_skipped++;
                        continue;
                    }

                    arc_token_sequence_t X(window_len);
                    for (long i = 0; i < window_len; ++i)
                    {
                        long idx = pos - window_len + 1 + i;
                        X(i) = (idx < 0) ? TOKEN_PADDING : S[idx];
                    }

                    training_X_batch.push_back(std::move(X));
                    training_Y_batch.push_back(static_cast<long>(S[pos + 1]));
                    samples_added++;
                }

                if (debug)
                {
                    std::cout << "    samples_added=" << samples_added
                        << "  samples_skipped=" << samples_skipped << "\n";
                }
            }

            if (debug)
            {
                std::cout << "  Total X=" << training_X_batch.size()
                    << "  Y=" << training_Y_batch.size() << "\n";

                std::map<std::string, int> dist;
                for (long y : training_Y_batch)
                    dist[token_to_string(static_cast<int>(y))]++;
                std::cout << "  Target distribution:\n";
                for (const auto& kv : dist)
                    std::cout << "    " << kv.first << " : " << kv.second << "\n";

                if (!training_X_batch.empty())
                {
                    std::cout << "  First sample => target: "
                        << token_to_string(static_cast<int>(training_Y_batch[0])) << "\n";
                    std::cout << "    Window: ";
                    int col = 0;
                    for (long i = 0; i < window_len; ++i)
                    {
                        std::cout << token_to_string(training_X_batch[0](i)) << " ";
                        if (++col >= 20) { std::cout << "\n             "; col = 0; }
                    }
                    std::cout << "\n";
                }
                std::cout << "=============================================\n\n";
            }
        }

        static void prepare_training_data_sliding_window(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch,
            bool debug = false)
        {
            DLIB_CASSERT(window_len > 1, "Window length must be greater than 1");
            training_X_batch.clear();
            training_Y_batch.clear();

            // Build the complete token sequence for this task:
            // [train_in_1 ... SEP_IO train_out_1 ... SEP_PAIR
            //  train_in_2 ... SEP_IO train_out_2 ... SEP_PAIR
            //  ...
            //  QUERY_START test_in ... GEN_START test_out ... END_OF_OUTPUT]
            std::vector<int> S;
            for (const auto& pair : task.train_pairs)
            {
                append_flat_grid(S, pair.input);
                S.push_back(TOKEN_SEP_IO);
                append_flat_grid(S, pair.output);
                S.push_back(TOKEN_SEP_PAIR);
            }
            for (const auto& pair : task.test_pairs)
            {
                S.push_back(TOKEN_QUERY_START);
                append_flat_grid(S, pair.input);
                S.push_back(TOKEN_GEN_START);
                append_flat_grid(S, pair.output);
                S.push_back(TOKEN_END_OF_OUTPUT);
            }

            const long L = static_cast<long>(S.size());

            // Find positions where the loss should be computed:
            // only tokens that are part of an output zone (after GEN_START or after SEP_IO,
            // up to the next SEP_PAIR / END_OF_OUTPUT).
            // We mark each position as "predict" or "context-only".
            std::vector<bool> is_output_zone(L, false);
            bool in_output = false;
            for (long i = 0; i < L; ++i)
            {
                if (S[i] == TOKEN_SEP_IO || S[i] == TOKEN_GEN_START)
                {
                    in_output = true;
                    continue;
                }
                if (S[i] == TOKEN_SEP_PAIR || S[i] == TOKEN_END_OF_OUTPUT)
                {
                    in_output = false;
                    continue;
                }
                if (in_output)
                    is_output_zone[i] = true;
            }

            if (debug)
            {
                std::cout << "\n=== prepare_training_data_sliding_window DEBUG ===\n";
                std::cout << "Task: " << task.task_id
                    << "  |  train_pairs: " << task.train_pairs.size()
                    << "  |  test_pairs: " << task.test_pairs.size()
                    << "  |  L=" << L
                    << "  |  window_len=" << window_len << "\n";

                // Print full sequence with output zones highlighted
                std::cout << "  Full sequence (" << L << " tokens):\n    ";
                int col = 0;
                for (long i = 0; i < L; ++i)
                {
                    std::string t = token_to_string(S[i]);
                    if (is_output_zone[i]) t = "*" + t + "*"; // mark output zone
                    std::cout << t << " ";
                    if (++col >= 20) { std::cout << "\n    "; col = 0; }
                }
                std::cout << "\n";

                long output_positions = 0;
                for (bool b : is_output_zone) if (b) output_positions++;
                std::cout << "  Output zone positions (trainable): "
                    << output_positions << " / " << L << "\n";
            }

            // Sliding window: for each position pos in [0, L-2],
            // build a window of size window_len ending at pos,
            // padded on the left if needed.
            // The label is S[pos+1], but we only keep samples where
            // S[pos+1] is in an output zone (paradigm B: learn only to generate outputs).
            //
            // Additionally: always include the "anchor" sample where the window ends
            // exactly at GEN_START (so the model learns to start generation),
            // even if that token is not strictly in an output zone.

            long samples_added = 0;
            long samples_skipped = 0;

            // Max left-padding allowed: 25% of window
            const long max_padding = window_len / 4;

            for (long pos = 0; pos < L - 1; ++pos)
            {
                const int next_token = S[pos + 1];

                // Only train on positions where the next token is in an output zone,
                // OR where the current token is GEN_START (anchor sample)
                bool is_gen_start_anchor = (S[pos] == TOKEN_GEN_START);
                bool next_in_output = (pos + 1 < L && is_output_zone[pos + 1]);

                if (!is_gen_start_anchor && !next_in_output)
                {
                    samples_skipped++;
                    continue;
                }

                // Build window ending at pos
                arc_token_sequence_t X(window_len);
                long padding_count = 0;

                for (long i = 0; i < window_len; ++i)
                {
                    long idx = pos - window_len + 1 + i;
                    if (idx < 0)
                    {
                        X(i) = TOKEN_PADDING;
                        padding_count++;
                    }
                    else
                    {
                        X(i) = S[idx];
                    }
                }

                // Enforce max padding ratio (25%)
                if (padding_count > max_padding)
                {
                    samples_skipped++;
                    continue;
                }

                training_X_batch.push_back(std::move(X));
                training_Y_batch.push_back(static_cast<long>(next_token));
                samples_added++;
            }

            if (debug)
            {
                std::cout << "  Samples added:   " << samples_added << "\n";
                std::cout << "  Samples skipped: " << samples_skipped << "\n";

                // Target distribution
                std::map<std::string, int> dist;
                for (long y : training_Y_batch)
                    dist[token_to_string(static_cast<int>(y))]++;
                std::cout << "  Target distribution:\n";
                for (const auto& kv : dist)
                    std::cout << "    " << kv.first << " : " << kv.second << "\n";

                // Show first 2 samples
                for (size_t k = 0; k < std::min<size_t>(2, training_X_batch.size()); ++k)
                {
                    std::cout << "  Sample #" << k
                        << " => target: "
                        << token_to_string(static_cast<int>(training_Y_batch[k])) << "\n";
                    std::cout << "    Window: ";
                    int col = 0;
                    for (long i = 0; i < window_len; ++i)
                    {
                        std::cout << token_to_string(training_X_batch[k](i)) << " ";
                        if (++col >= 20) { std::cout << "\n            "; col = 0; }
                    }
                    std::cout << "\n";
                }
                std::cout << "=================================================\n\n";
            }
        }

        static void prepare_training_data_batch(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch,
            bool debug = false)
        {
            DLIB_CASSERT(window_len > 1, "Window length must be greater than 1");
            training_X_batch.clear();
            training_Y_batch.clear();

            static thread_local std::mt19937 rng(std::random_device{}());
            const long MAX_SAMPLES_PER_OUTPUT = 256;

            if (debug)
            {
                std::cout << "\n=== prepare_training_data_batch DEBUG ===\n";
                std::cout << "Task: " << task.task_id << "\n";
                std::cout << "train_pairs: " << task.train_pairs.size()
                    << "  |  test_pairs: " << task.test_pairs.size()
                    << "  |  window_len: " << window_len << "\n";
            }

            for (size_t held_out = 0; held_out < task.train_pairs.size(); ++held_out)
            {
                if (debug)
                    std::cout << "\n--- held_out = " << held_out << " ---\n";

                arc_task synthetic_task;
                synthetic_task.task_id = task.task_id;

                for (size_t i = 0; i < task.train_pairs.size(); ++i)
                    if (i != held_out)
                        synthetic_task.train_pairs.push_back(task.train_pairs[i]);
                const arc_task_pair& target_pair = task.train_pairs[held_out];

                if (debug)
                {
                    std::cout << "  synthetic_task.train_pairs.size() = "
                        << synthetic_task.train_pairs.size() << "\n";
                    for (size_t i = 0; i < synthetic_task.train_pairs.size(); ++i)
                    {
                        std::cout << "    demo[" << i << "]: input="
                            << synthetic_task.train_pairs[i].input_rows << "x"
                            << synthetic_task.train_pairs[i].input_cols
                            << "  output="
                            << synthetic_task.train_pairs[i].output_rows << "x"
                            << synthetic_task.train_pairs[i].output_cols << "\n";
                    }
                    std::cout << "  target_pair: input="
                        << target_pair.input_rows << "x" << target_pair.input_cols
                        << "  output="
                        << target_pair.output_rows << "x" << target_pair.output_cols << "\n";
                }                

                arc_token_sequence_t input_context =
                    tokenize_input_context(synthetic_task, target_pair);
                arc_token_sequence_t target_output =
                    tokenize_target_output(target_pair);

                const long L_in = input_context.size();
                const long L_out = target_output.size();
                const long L_full = L_in + L_out;

                std::vector<int> S;
                S.reserve(L_full);
                for (long i = 0; i < L_in; ++i) S.push_back(input_context(i));
                for (long i = 0; i < L_out; ++i) S.push_back(target_output(i));

                if (debug)
                {
                    std::cout << "  L_in=" << L_in
                        << "  L_out=" << L_out
                        << "  L_full=" << L_full << "\n";

                    long preview_end = std::min(30L, L_in - 1);
                    print_sequence("input_context (start)", S, 0, preview_end);

                    long gs_pos = -1;
                    for (long i = 0; i < L_full; ++i)
                        if (S[i] == TOKEN_GEN_START) { gs_pos = i; break; }

                    if (gs_pos >= 0)
                    {
                        long jct_from = std::max(0L, gs_pos - 5);
                        long jct_to = std::min(L_full - 1, gs_pos + 15);
                        print_sequence("around GEN_START", S, jct_from, jct_to);
                    }

                    long tail_from = std::max(0L, L_full - 20);
                    print_sequence("end of sequence", S, tail_from, L_full - 1);
                }

                long gen_start_pos = -1;
                for (long i = 0; i < L_full; ++i)
                {
                    if (S[i] == TOKEN_GEN_START) { gen_start_pos = i; break; }
                }

                if (debug)
                    std::cout << "  gen_start_pos=" << gen_start_pos << "\n";

                if (gen_start_pos < 0) { if (debug) std::cout << "  !! GEN_START not found, skipped\n"; continue; }

                const long first_predict_pos = gen_start_pos + 1;
                const long last_predict_pos = L_full - 2;

                if (debug)
                {
                    std::cout << "  first_predict_pos=" << first_predict_pos
                        << "  (token: " << token_to_string(S[first_predict_pos]) << ")\n";
                    std::cout << "  last_predict_pos=" << last_predict_pos
                        << "  (token: " << token_to_string(S[last_predict_pos]) << ")"
                        << "  => next: " << token_to_string(S[last_predict_pos + 1]) << "\n";
                }

                if (last_predict_pos < first_predict_pos)
                {
                    if (debug) std::cout << "  !! Empty prediction range, skipped\n";
                    continue;
                }

                std::vector<long> candidate_positions;
                for (long pos = first_predict_pos; pos <= last_predict_pos; ++pos)
                {
                    int next = S[pos + 1];
                    if ((next >= COLOR_0 && next <= COLOR_9) ||
                        next == TOKEN_ROW_END ||
                        next == TOKEN_END_OF_OUTPUT)
                    {
                        candidate_positions.push_back(pos);
                    }
                }

                if (debug)
                {
                    std::cout << "  Candidate positions: " << candidate_positions.size()
                        << " / " << (last_predict_pos - first_predict_pos + 1) << "\n";

                    std::map<std::string, int> target_dist;
                    for (long pos : candidate_positions)
                        target_dist[token_to_string(S[pos + 1])]++;

                    std::cout << "  Target distribution:\n";
                    for (const auto& kv : target_dist)
                        std::cout << "    " << kv.first << " : " << kv.second << "\n";
                }

                if (candidate_positions.empty()) continue;

                std::shuffle(candidate_positions.begin(), candidate_positions.end(), rng);

                long samples_to_take = std::min<long>(
                    MAX_SAMPLES_PER_OUTPUT,
                    static_cast<long>(candidate_positions.size()));

                long samples_added = 0;
                long samples_dropped = 0;

                for (long k = 0; k < samples_to_take; ++k)
                {
                    long pos = candidate_positions[k];

                    arc_token_sequence_t X_window(window_len);
                    long padding_count = 0;

                    for (long i = 0; i < window_len; ++i)
                    {
                        long ctx_idx = pos - window_len + 1 + i;
                        if (ctx_idx < 0)
                        {
                            X_window(i) = TOKEN_PADDING;
                            padding_count++;
                        }
                        else
                        {
                            X_window(i) = S[ctx_idx];
                            if (S[ctx_idx] == TOKEN_PADDING) padding_count++;
                        }
                    }

                    double padding_ratio = static_cast<double>(padding_count) / window_len;

                    if (padding_ratio > 0.80)
                    {
                        samples_dropped++;
                        continue;
                    }

                    if (debug && samples_added < 3)
                        print_window_and_target(X_window, S[pos + 1], samples_added, window_len);

                    training_X_batch.push_back(std::move(X_window));
                    training_Y_batch.push_back(static_cast<long>(S[pos + 1]));
                    samples_added++;
                }

                if (debug)
                {
                    std::cout << "  Samples kept: " << samples_added
                        << "  |  Samples dropped (padding): " << samples_dropped << "\n";
                }
            }

            if (debug)
            {
                std::cout << "\n--- Totals ---\n";
                std::cout << "  X.size()=" << training_X_batch.size()
                    << "  Y.size()=" << training_Y_batch.size() << "\n";

                if (!training_Y_batch.empty())
                {
                    std::map<std::string, int> global_dist;
                    for (long y : training_Y_batch)
                        global_dist[token_to_string(static_cast<int>(y))]++;
                    std::cout << "  Overall target distribution:\n";
                    for (const auto& kv : global_dist)
                        std::cout << "    " << kv.first << " : " << kv.second << "\n";
                }
                std::cout << "=========================================\n\n";
            }        
        }
        /*static void prepare_training_data_batch(
            const arc_task& task,
            long window_len,
            std::vector<arc_token_sequence_t>& training_X_batch,
            std::vector<long>& training_Y_batch)
        {
            DLIB_CASSERT(window_len > 1, "Window length must be greater than 1");

            training_X_batch.clear();
            training_Y_batch.clear();

            for (const arc_task_pair& test_pair : task.test_pairs)
            {
                arc_token_sequence_t input_context = tokenize_input_context(task, test_pair);
                arc_token_sequence_t target_output = tokenize_target_output(test_pair);

                long L_in = input_context.size();
                long L_out = target_output.size();
                long L_full = L_in + L_out;

                // Build the complete token sequence as int
                std::vector<int> S_vec;
                S_vec.reserve(static_cast<size_t>(L_full));

                for (long i = 0; i < L_in; ++i) S_vec.push_back(input_context(i));
                for (long i = 0; i < L_out; ++i) S_vec.push_back(target_output(i));

                // Generate sliding window samples
                for (long pos = 0; pos < L_full; ++pos)
                {
                    arc_token_sequence_t X_window(window_len);

                    for (long i = 0; i < window_len; ++i)
                    {
                        long context_idx = pos - window_len + 1 + i;
                        if (context_idx < 0 || context_idx >= L_full)
                            X_window(i) = TOKEN_PADDING;
                        else
                            X_window(i) = S_vec[static_cast<size_t>(context_idx)];
                    }

                    // Y label stays long for dlib unsigned long label compatibility
                    long y_token = (pos + 1 < L_full)
                        ? static_cast<long>(S_vec[static_cast<size_t>(pos + 1)])
                        : static_cast<long>(TOKEN_PADDING);

                    training_X_batch.push_back(std::move(X_window));
                    training_Y_batch.push_back(y_token);
                }
            }
        }*/

        // ----------------------------------------------------------------------------------------
        // Detokenization utilities
        // ----------------------------------------------------------------------------------------

        static arc_grid_t detokenize_to_grid(const arc_token_sequence_t& tokens,
            long start_idx = 0)
            /*!
                ensures
                    - Reconstructs a grid from a tokenized sequence
                    - Uses TOKEN_ROW_END markers to determine row boundaries
                    - Stops at TOKEN_END_OF_OUTPUT, TOKEN_SEP_IO, or TOKEN_SEP_PAIR
                    - Returns a matrix with the reconstructed grid
                    - Returns an empty matrix if no valid grid is found
                throws
                    - DLIB_CASSERT if row lengths are inconsistent
            !*/
        {
            std::vector<std::vector<unsigned char>> rows;
            std::vector<unsigned char> current_row;

            for (long i = start_idx; i < tokens.size(); ++i)
            {
                int token = tokens(i);

                if (token == TOKEN_ROW_END)
                {
                    if (!current_row.empty())
                    {
                        rows.push_back(current_row);
                        current_row.clear();
                    }
                }
                else if (token == TOKEN_END_OF_OUTPUT ||
                    token == TOKEN_SEP_IO ||
                    token == TOKEN_SEP_PAIR)
                {
                    break;
                }
                else if (token >= COLOR_0 && token <= COLOR_9)
                {
                    current_row.push_back(static_cast<unsigned char>(token));
                }
                // Ignore padding and other non-grid tokens
            }

            if (rows.empty())
                return arc_grid_t(0, 0);

            long n_rows = static_cast<long>(rows.size());
            long n_cols = static_cast<long>(rows[0].size());

            arc_grid_t grid(n_rows, n_cols);
            for (long r = 0; r < n_rows; ++r)
            {
                DLIB_CASSERT(static_cast<long>(rows[r].size()) == n_cols,
                    "Inconsistent row length during detokenization"
                    << "\n\tRow " << r << " has " << rows[r].size() << " columns"
                    << "\n\tExpected " << n_cols << " columns");
                for (long c = 0; c < n_cols; ++c)
                    grid(r, c) = rows[r][static_cast<size_t>(c)];
            }

            return grid;
        }
    };

// ----------------------------------------------------------------------------------------

    /*!
        THE PREPARED ARC DATASET

        Reads what build_arc_dataset.py writes: every grid already written into a fixed
        window, and the two index arrays that say which examples belong to which puzzle
        and which puzzles belong to which task.

        The conditioning deserves a word, because it is what the layout is built around.
        A model trained on this data is shown one input grid and no worked examples of
        the same task. What tells it which transformation to apply is the puzzle
        identifier, carried as the first token of the sequence. The identifiers are
        numbered above the colours, so that one index per position addresses both, and a
        puzzle_embeddings layer reads the colours from its token table and an identifier
        from its puzzle table. That table keeps no optimizer state and moves only the rows
        a batch names, which is what a table of several hundred thousand puzzles needs.

        The price is stated plainly. An identifier the model never trained on carries no
        meaning, so a puzzle absent from training cannot be solved by naming it. Passing
        arc_blank_puzzle_id asks the model to work without being told, which is the
        harder setting.

        The cells are held one byte each, the examples one after the other, so that a
        split of a thousand variants per task fits in memory.
    !*/

    struct arc_dataset
    {
        long seq_len = 0;               // cells per grid, 900 for a 30 by 30 window
        long colour_vocab = 0;          // padding, the end marker and the ten colours

        std::vector<unsigned char> inputs;   // seq_len cells per example
        std::vector<unsigned char> labels;   // seq_len cells per example

        std::vector<int> puzzle_identifiers; // the blank puzzle first, then one per puzzle
        std::vector<int> puzzle_indices;     // first example of each puzzle, plus the end
        std::vector<int> group_indices;      // first puzzle of each group, plus the end

        // The name of every puzzle, the blank one first, read from the .names file the
        // builder writes beside the split. A name records the variant a puzzle is.
        std::vector<std::string> puzzle_names;

        long num_puzzles () const { return (long)puzzle_identifiers.size(); }
        long num_groups  () const { return (long)group_indices.size() - 1; }
        long num_examples() const { return seq_len > 0 ? (long)(inputs.size() / (size_t)seq_len) : 0; }

        /*!
            The number of indices an input of this split can hold: the colours, then one
            per puzzle of the split.
        !*/
        long vocabulary () const { return colour_vocab + num_puzzles(); }

        /*!
            Length of the sequence the network sees: the identifier token and the window.
        !*/
        long sequence_length () const { return seq_len + 1; }

        /*!
            The cells of one example, seq_len of them.
        !*/
        const unsigned char* input_cells (long example) const
        {
            DLIB_CASSERT(example >= 0 && example < num_examples());
            return inputs.data() + (size_t)example * (size_t)seq_len;
        }
        const unsigned char* label_cells (long example) const
        {
            DLIB_CASSERT(example >= 0 && example < num_examples());
            return labels.data() + (size_t)example * (size_t)seq_len;
        }

        /*!
            Where in this split an example's puzzle sits, counted as the identifiers and
            the names are, the blank puzzle being 0. This is a position, not a name.
        !*/
        long puzzle_of (long example) const
        {
            const auto it = std::upper_bound(puzzle_indices.begin(), puzzle_indices.end(),
                                             (int)example);
            return (long)(it - puzzle_indices.begin());
        }

        /*!
            The examples of the puzzle at a position given by puzzle_of(), from the first
            to one past the last.
        !*/
        long first_example (long puzzle) const
        {
            DLIB_CASSERT(puzzle >= 1 && puzzle < (long)puzzle_indices.size());
            return puzzle_indices[(size_t)puzzle - 1];
        }
        long end_example (long puzzle) const
        {
            DLIB_CASSERT(puzzle >= 1 && puzzle < (long)puzzle_indices.size());
            return puzzle_indices[(size_t)puzzle];
        }

        /*!
            The identifier of the puzzle an example belongs to.

            This is the number the network looks up, and it is not the position above. A
            split built under the reference protocol stores identifiers assigned while the
            training split was built, so an evaluation puzzle sitting at position three may
            be named twenty, and under the held-out protocol every evaluation puzzle carries
            the blank identifier. Passing the position would name a different puzzle,
            silently.
        !*/
        int identifier_of (long example) const
        {
            const long p = puzzle_of(example);
            DLIB_CASSERT(p >= 0 && p < (long)puzzle_identifiers.size(),
                "arc dataset: example " << example << " falls outside the puzzle index");
            return puzzle_identifiers[(size_t)p];
        }

        /*!
            The largest identifier this split can name, which is what an embedding table
            has to reach for the split to be usable at all.
        !*/
        int largest_identifier () const
        {
            return puzzle_identifiers.empty() ? 0
                 : *std::max_element(puzzle_identifiers.begin(), puzzle_identifiers.end());
        }
    };

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        /* The names file the builder writes beside a split: the same path, with .names in
           place of .bin. */
        inline std::string arc_names_file (const std::string& filename)
        {
            const std::string ext = ".bin";
            if (filename.size() > ext.size() &&
                filename.compare(filename.size() - ext.size(), ext.size(), ext) == 0)
                return filename.substr(0, filename.size() - ext.size()) + ".names";
            return filename + ".names";
        }
    }

    inline arc_dataset load_arc_dataset (
        const std::string& filename
    )
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
            throw std::runtime_error("arc dataset: cannot open " + filename);

        char magic[4] = {0, 0, 0, 0};
        in.read(magic, 4);
        if (std::string(magic, 4) != "ARCD")
            throw std::runtime_error("arc dataset: " + filename +
                                     " is not a prepared ARC file");

        auto read_i32 = [&in]() -> int {
            unsigned char b[4];
            in.read((char*)b, 4);
            return (int)((std::uint32_t)b[0] | ((std::uint32_t)b[1] << 8) |
                         ((std::uint32_t)b[2] << 16) | ((std::uint32_t)b[3] << 24));
        };
        auto read_vec = [&](long n) {
            std::vector<int> v((size_t)n);
            for (long i = 0; i < n; ++i) v[(size_t)i] = read_i32();
            return v;
        };

        const int version = read_i32();
        if (version != 1)
            throw std::runtime_error("arc dataset: " + filename + " has version " +
                                     std::to_string(version) + ", expected 1");

        arc_dataset d;
        d.seq_len      = read_i32();
        d.colour_vocab = read_i32();
        const long n_examples = read_i32();
        const long n_ids      = read_i32();
        const long n_pidx     = read_i32();
        const long n_gidx     = read_i32();

        DLIB_CASSERT(d.seq_len > 0 && d.colour_vocab > 0 && d.colour_vocab <= 256 &&
                     n_examples >= 0 && n_ids >= 0 && n_pidx >= 0 && n_gidx >= 0,
            "arc dataset: header of " << filename << " is not sensible");

        /* The file stores every cell as an int32. They are read in blocks and kept one byte
           each, which takes a quarter of the memory. */
        auto read_cells = [&](std::vector<unsigned char>& out) {
            const size_t count = (size_t)n_examples * (size_t)d.seq_len;
            out.resize(count);
            std::vector<unsigned char> block(4 * std::min<size_t>(std::max<size_t>(count, 1), 1 << 18));
            for (size_t done = 0; done < count; )
            {
                const size_t n = std::min(count - done, block.size() / 4);
                in.read((char*)block.data(), (std::streamsize)(4 * n));
                if (!in)
                    throw std::runtime_error("arc dataset: " + filename + " ended early");
                for (size_t i = 0; i < n; ++i)
                {
                    const unsigned char* b = block.data() + 4 * i;
                    const std::uint32_t v = (std::uint32_t)b[0] | ((std::uint32_t)b[1] << 8) |
                                            ((std::uint32_t)b[2] << 16) | ((std::uint32_t)b[3] << 24);
                    if (v >= (std::uint32_t)d.colour_vocab)
                        throw std::runtime_error("arc dataset: " + filename + " holds a cell of value " +
                            std::to_string((long long)(std::int32_t)v) + " outside its vocabulary of " +
                            std::to_string(d.colour_vocab));
                    out[done + i] = (unsigned char)v;
                }
                done += n;
            }
        };
        read_cells(d.inputs);
        read_cells(d.labels);

        d.puzzle_identifiers = read_vec(n_ids);
        d.puzzle_indices     = read_vec(n_pidx);
        d.group_indices      = read_vec(n_gidx);

        if (!in)
            throw std::runtime_error("arc dataset: " + filename + " ended early");

        DLIB_CASSERT(!d.puzzle_indices.empty() &&
                     d.puzzle_indices.back() == (int)n_examples,
            "arc dataset: the puzzle index of " << filename << " does not end on the "
            "example count, so the file is inconsistent");
        DLIB_CASSERT(!d.group_indices.empty() &&
                     d.group_indices.back() == (int)(n_pidx - 1),
            "arc dataset: the group index of " << filename << " does not end on the "
            "puzzle count, so the file is inconsistent");

        const std::string names_file = impl::arc_names_file(filename);
        std::ifstream names(names_file);
        if (names)
        {
            std::string line;
            while (std::getline(names, line))
            {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                d.puzzle_names.push_back(line);
            }
            if ((long)d.puzzle_names.size() != d.num_puzzles())
                throw std::runtime_error("arc dataset: " + names_file + " names " +
                    std::to_string(d.puzzle_names.size()) + " puzzles where " + filename +
                    " holds " + std::to_string(d.num_puzzles()) + ", the blank one included");
        }

        return d;
    }

// ----------------------------------------------------------------------------------------

    /*!
        Passed to arc_make_input() in place of an identifier, this withholds it, which is
        how a model is asked to solve a puzzle it was never told the name of.
    !*/
    constexpr int arc_blank_puzzle_id = 0;

    /*!
        The tokens a prepared grid is written in: padding outside the grid, an end marker
        on the row below it and the column to its right, and the ten colours after those.
    !*/
    constexpr unsigned long arc_pad_token     = 0;
    constexpr unsigned long arc_eos_token     = 1;
    constexpr unsigned long arc_colour_offset = 2;
    constexpr unsigned long arc_num_colours   = 10;

    /*!
        The label a position carries when it is not to be scored.

        Padding is most of the window: a 10 by 10 answer leaves eight hundred of its nine
        hundred cells empty, and scoring them would train a model to predict emptiness and
        count it as accuracy. The padding and the identifier are therefore labelled with
        this value and ignored by the loss.

        It sits beyond every token an input can hold, identifiers included, on purpose.
        The loss also masks a position whose own input token equals its padding index,
        and unless told otherwise that index falls back to this one. A value no input
        ever takes keeps that mask inert, so that a cell is scored by what it should be
        and never skipped because the question was empty there, as it is wherever the
        answer is larger than the question.
    !*/
    constexpr unsigned long arc_ignore_label = 1ul << 30;

    /*!
        Builds the sequence the network reads: the identifier token, then the window.
    !*/
    inline matrix<int, 0, 1> arc_make_input (
        const arc_dataset& d,
        long example,
        int puzzle_id
    )
    {
        DLIB_CASSERT(example >= 0 && example < d.num_examples());
        DLIB_CASSERT(puzzle_id >= 0,
            "arc dataset: identifier " << puzzle_id << " is not a name");

        const unsigned char* cells = d.input_cells(example);
        matrix<int, 0, 1> seq(d.sequence_length());

        // The identifier is numbered above the colours, so that one index addresses both.
        seq(0) = d.colour_vocab + puzzle_id;
        for (long i = 0; i < d.seq_len; ++i)
            seq(i + 1) = cells[i];
        return seq;
    }

    /*!
        The labels for that sequence. The first position carries the identifier, which is
        given rather than predicted, and the padding carries no answer, so both are
        labelled with arc_ignore_label and the loss ignores them.
    !*/
    inline matrix<unsigned long, 0, 1> arc_make_label (
        const arc_dataset& d,
        long example
    )
    {
        DLIB_CASSERT(example >= 0 && example < d.num_examples());

        const unsigned char* cells = d.label_cells(example);
        matrix<unsigned long, 0, 1> seq(d.sequence_length());
        seq(0) = arc_ignore_label;                 // the identifier is given, not predicted
        for (long i = 0; i < d.seq_len; ++i)
        {
            const unsigned long tok = cells[i];
            seq(i + 1) = tok == arc_pad_token ? arc_ignore_label : tok;
        }
        return seq;
    }

// ----------------------------------------------------------------------------------------

    /*!
        Reads a whole grid out of the network in one pass, by taking the most likely
        colour at each position of the output.

        The alternative would be to generate the window token by token, which for nine
        hundred positions would be nine hundred forward passes to answer one puzzle. The
        loss is per position, so every position already has its logits after a single
        pass and there is nothing to gain by asking again.
    !*/
    inline std::vector<unsigned long> arc_predict_sequence (
        const float* logits,
        long row_stride,
        long sequence_length,
        long vocabulary
    )
    {
        DLIB_CASSERT(logits != nullptr && row_stride >= vocabulary && vocabulary > 0,
            "arc dataset: a row of " << row_stride << " cannot hold " << vocabulary
            << " colours");

        std::vector<unsigned long> out((size_t)sequence_length, 0);
        for (long i = 0; i < sequence_length; ++i)
        {
            const float* row = logits + i * row_stride;
            long best = 0;
            for (long c = 1; c < vocabulary; ++c)
                if (row[c] > row[best]) best = c;
            out[(size_t)i] = (unsigned long)best;
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        /* The grid a window holds from its top-left corner, cropped as the reference crops
           it: the largest rectangle anchored at the corner whose cells are all colours. The
           width a row allows is how far it runs in colours, never more than the rows above
           it allowed, and the rectangle of largest area over all heights is kept. */
        template <typename T>
        arc_grid_t arc_crop_window (
            const T* cells,
            long grid_side
        )
        {
            auto is_colour = [](T tok) {
                return (unsigned long)tok >= arc_colour_offset &&
                       (unsigned long)tok <  arc_colour_offset + arc_num_colours; };

            long width = grid_side, best_rows = 0, best_cols = 0, best_area = 0;
            for (long r = 0; r < grid_side && width > 0; ++r)
            {
                for (long c = 0; c < width; ++c)
                {
                    if (!is_colour(cells[r * grid_side + c]))
                    {
                        width = c;
                        break;
                    }
                }
                if ((r + 1) * width > best_area)
                {
                    best_area = (r + 1) * width;
                    best_rows = r + 1;
                    best_cols = width;
                }
            }

            arc_grid_t g(best_rows, best_cols);
            for (long r = 0; r < best_rows; ++r)
                for (long c = 0; c < best_cols; ++c)
                    g(r, c) = (unsigned char)((unsigned long)cells[r * grid_side + c] - arc_colour_offset);
            return g;
        }

        inline long arc_grid_side (long seq_len)
        {
            long side = 0;
            while ((side + 1) * (side + 1) <= seq_len) ++side;
            DLIB_CASSERT(side * side == seq_len,
                "arc dataset: a window of " << seq_len << " cells is not a square");
            return side;
        }
    }

    /*!
        Turns a predicted sequence back into a grid, read from the top-left corner of the
        window and cropped as the reference crops it, to the largest rectangle of colours
        anchored there. An answer whose first cell holds no colour is an empty grid, which
        matches no expected answer.
    !*/
    inline arc_grid_t arc_decode_grid (
        const std::vector<unsigned long>& predicted,
        long seq_len,
        long grid_side
    )
    {
        DLIB_CASSERT(grid_side > 0 && seq_len == grid_side * grid_side);
        const long offset = (long)predicted.size() - seq_len;   // skip the identifier
        DLIB_CASSERT(offset >= 0, "arc dataset: the prediction is shorter than a window");
        return impl::arc_crop_window(predicted.data() + offset, grid_side);
    }

    /*!
        The input and the expected answer of an example as grids, read from the top-left
        corner of its window. This is where they sit in a split the builder never moves,
        the evaluation split.
    !*/
    inline arc_grid_t arc_input_grid (
        const arc_dataset& d,
        long example
    )
    {
        return impl::arc_crop_window(d.input_cells(example), impl::arc_grid_side(d.seq_len));
    }

    inline arc_grid_t arc_label_grid (
        const arc_dataset& d,
        long example
    )
    {
        return impl::arc_crop_window(d.label_cells(example), impl::arc_grid_side(d.seq_len));
    }

// ----------------------------------------------------------------------------------------

    /*!
        A variant of a task, as the builder draws it: a symmetry of the square, and a
        mapping of the colours, colour c becoming colours[c]. The task itself is the
        identity of both.

        The symmetries are numbered as the reference numbers them, after numpy: 0 is the
        identity, 1 to 3 are one to three quarter turns counter-clockwise, 4 mirrors left
        to right, 5 top to bottom, 6 transposes and 7 reflects across the other diagonal.
    !*/
    struct arc_variant
    {
        int symmetry = 0;
        std::array<unsigned char, arc_num_colours> colours;

        arc_variant ()
        {
            for (unsigned long c = 0; c < arc_num_colours; ++c)
                colours[c] = (unsigned char)c;
        }

        bool is_identity () const
        {
            for (unsigned long c = 0; c < arc_num_colours; ++c)
                if (colours[c] != c)
                    return false;
            return symmetry == 0;
        }
    };

    /*!
        Applies one of the eight symmetries, numbered as above.
    !*/
    inline arc_grid_t arc_apply_symmetry (
        const arc_grid_t& g,
        int symmetry
    )
    {
        DLIB_CASSERT(0 <= symmetry && symmetry < 8,
            "arc dataset: there is no symmetry numbered " << symmetry);
        const long rows = g.nr(), cols = g.nc();
        const bool turned = symmetry == 1 || symmetry == 3 || symmetry == 6 || symmetry == 7;
        arc_grid_t out(turned ? cols : rows, turned ? rows : cols);
        for (long i = 0; i < out.nr(); ++i)
        {
            for (long j = 0; j < out.nc(); ++j)
            {
                switch (symmetry)
                {
                    case 0:  out(i, j) = g(i, j); break;
                    case 1:  out(i, j) = g(j, cols - 1 - i); break;
                    case 2:  out(i, j) = g(rows - 1 - i, cols - 1 - j); break;
                    case 3:  out(i, j) = g(rows - 1 - j, i); break;
                    case 4:  out(i, j) = g(i, cols - 1 - j); break;
                    case 5:  out(i, j) = g(rows - 1 - i, j); break;
                    case 6:  out(i, j) = g(j, i); break;
                    default: out(i, j) = g(rows - 1 - j, cols - 1 - i); break;
                }
            }
        }
        return out;
    }

    /*!
        The symmetry that undoes another: the quarter turns undo one another and every
        reflection undoes itself.
    !*/
    inline int arc_inverse_symmetry (
        int symmetry
    )
    {
        DLIB_CASSERT(0 <= symmetry && symmetry < 8,
            "arc dataset: there is no symmetry numbered " << symmetry);
        const int inverse[8] = {0, 3, 2, 1, 4, 5, 6, 7};
        return inverse[symmetry];
    }

    /*!
        Makes a variant of a grid, the colours first and then the symmetry, as the builder
        does, and undoes it. A cell holding no colour is left as it is.
    !*/
    inline arc_grid_t arc_apply_variant (
        const arc_grid_t& g,
        const arc_variant& v
    )
    {
        arc_grid_t mapped = g;
        for (long r = 0; r < mapped.nr(); ++r)
            for (long c = 0; c < mapped.nc(); ++c)
                if (mapped(r, c) < arc_num_colours)
                    mapped(r, c) = v.colours[mapped(r, c)];
        return arc_apply_symmetry(mapped, v.symmetry);
    }

    inline arc_grid_t arc_undo_variant (
        const arc_grid_t& g,
        const arc_variant& v
    )
    {
        std::array<unsigned char, arc_num_colours> back;
        for (unsigned long c = 0; c < arc_num_colours; ++c)
            back[c] = (unsigned char)c;
        for (unsigned long c = 0; c < arc_num_colours; ++c)
            back[v.colours[c]] = (unsigned char)c;
        arc_grid_t out = arc_apply_symmetry(g, arc_inverse_symmetry(v.symmetry));
        for (long r = 0; r < out.nr(); ++r)
            for (long c = 0; c < out.nc(); ++c)
                if (out(r, c) < arc_num_colours)
                    out(r, c) = back[out(r, c)];
        return out;
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        /* A variant's name is its task's name followed by _t, the symmetry, _ and the ten
           digits of the colour mapping, as the reference writes it. */
        inline bool arc_split_puzzle_name (
            const std::string& name,
            std::string& task,
            arc_variant& variant
        )
        {
            const size_t suffix = 4 + arc_num_colours;
            if (name.size() <= suffix)
                return false;
            const size_t s = name.size() - suffix;
            if (name[s] != '_' || name[s + 1] != 't' || name[s + 2] < '0' || name[s + 2] > '7' ||
                name[s + 3] != '_')
                return false;

            arc_variant v;
            bool seen[arc_num_colours] = {};
            for (unsigned long c = 0; c < arc_num_colours; ++c)
            {
                const char ch = name[s + 4 + c];
                if (ch < '0' || ch > '9' || seen[ch - '0'])
                    return false;
                seen[ch - '0'] = true;
                v.colours[c] = (unsigned char)(ch - '0');
            }
            v.symmetry = name[s + 2] - '0';
            task = name.substr(0, s);
            variant = v;
            return true;
        }
    }

    /*!
        The task a puzzle name belongs to, and the variant it records. A name with no
        variant is a task itself, under the identity.
    !*/
    inline std::string arc_task_of (
        const std::string& puzzle_name
    )
    {
        std::string task;
        arc_variant v;
        return impl::arc_split_puzzle_name(puzzle_name, task, v) ? task : puzzle_name;
    }

    inline arc_variant arc_variant_of (
        const std::string& puzzle_name
    )
    {
        std::string task;
        arc_variant v;
        impl::arc_split_puzzle_name(puzzle_name, task, v);
        return v;
    }

// ----------------------------------------------------------------------------------------

    /*!
        The vote between the variants of a task, as the reference casts it.

        Every variant of a task answers each of its test inputs. Brought back to the task
        by undoing the variant, each answer is a vote for a grid. The grids a test input
        receives are ranked by their votes, then by the mean confidence of those votes,
        then by the order in which they first came, and the first of them are the attempts.
        A task is solved with k attempts when every one of its test inputs has its expected
        answer among its first k grids.
    !*/
    class arc_vote
    {
    public:
        /*!
            Records one variant's answer to one test input of a task, both grids already
            brought back to the task, and the confidence the model gave it. The expected
            answer is the same whichever variant it comes from. Two variants bringing back
            different ones means the variants were not undone as they were made, and that
            throws.
        !*/
        void add (
            const std::string& task,
            long question,
            const arc_grid_t& expected,
            const arc_grid_t& answer,
            double confidence
        )
        {
            question_tally& q = tasks_[task][question];
            const std::string expected_key = key_of(expected);
            if (q.candidates.empty())
                q.expected = expected_key;
            else if (q.expected != expected_key)
                throw std::runtime_error("arc_vote: two variants of " + task + " expect "
                    "different answers to test input " + std::to_string(question) +
                    ", so the variants were not undone as they were made");

            const std::string key = key_of(answer);
            auto found = q.index.find(key);
            if (found == q.index.end())
            {
                found = q.index.emplace(key, q.candidates.size()).first;
                q.candidates.push_back(candidate{key, 0, 0.0});
            }
            candidate& c = q.candidates[found->second];
            ++c.votes;
            c.confidence += confidence;
            ++answers_;
        }

        long num_tasks () const { return (long)tasks_.size(); }

        long num_questions () const
        {
            long n = 0;
            for (const auto& t : tasks_)
                n += (long)t.second.size();
            return n;
        }

        long num_answers () const { return answers_; }

        /*!
            The number of tasks solved with the given number of attempts.
        !*/
        long solved (
            long attempts
        ) const
        {
            long count = 0;
            for (const auto& t : tasks_)
            {
                bool all = true;
                for (const auto& q : t.second)
                {
                    const std::vector<size_t> order = ranking(q.second);
                    bool found = false;
                    for (size_t i = 0; i < order.size() && (long)i < attempts; ++i)
                        found = found || q.second.candidates[order[i]].key == q.second.expected;
                    all = all && found;
                }
                count += all ? 1 : 0;
            }
            return count;
        }

        /*!
            The first grids a test input of a task received, at most the given number, in
            the order of the ranking.
        !*/
        std::vector<arc_grid_t> attempts (
            const std::string& task,
            long question,
            long count
        ) const
        {
            std::vector<arc_grid_t> out;
            const auto t = tasks_.find(task);
            if (t == tasks_.end())
                return out;
            const auto q = t->second.find(question);
            if (q == t->second.end())
                return out;
            const std::vector<size_t> order = ranking(q->second);
            for (size_t i = 0; i < order.size() && (long)i < count; ++i)
                out.push_back(grid_of(q->second.candidates[order[i]].key));
            return out;
        }

        void clear ()
        {
            tasks_.clear();
            answers_ = 0;
        }

    private:
        struct candidate
        {
            std::string key;
            long votes;
            double confidence;
        };

        struct question_tally
        {
            std::string expected;
            std::vector<candidate> candidates;
            std::map<std::string, size_t> index;
        };

        /* A grid as a string of bytes, its height and width on two bytes each and then its
           cells, which is what two answers are compared on. */
        static std::string key_of (const arc_grid_t& g)
        {
            std::string key(4 + (size_t)(g.nr() * g.nc()), '\0');
            key[0] = (char)(g.nr() & 0xff);
            key[1] = (char)((g.nr() >> 8) & 0xff);
            key[2] = (char)(g.nc() & 0xff);
            key[3] = (char)((g.nc() >> 8) & 0xff);
            size_t i = 4;
            for (long r = 0; r < g.nr(); ++r)
                for (long c = 0; c < g.nc(); ++c)
                    key[i++] = (char)g(r, c);
            return key;
        }

        static arc_grid_t grid_of (const std::string& key)
        {
            auto byte = [&key](size_t i) { return (long)(unsigned char)key[i]; };
            const long rows = byte(0) | (byte(1) << 8), cols = byte(2) | (byte(3) << 8);
            arc_grid_t g(rows, cols);
            size_t i = 4;
            for (long r = 0; r < rows; ++r)
                for (long c = 0; c < cols; ++c)
                    g(r, c) = (unsigned char)key[i++];
            return g;
        }

        static std::vector<size_t> ranking (const question_tally& q)
        {
            std::vector<size_t> order(q.candidates.size());
            for (size_t i = 0; i < order.size(); ++i)
                order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&q](size_t a, size_t b) {
                const candidate& x = q.candidates[a];
                const candidate& y = q.candidates[b];
                if (x.votes != y.votes)
                    return x.votes > y.votes;
                return x.confidence / x.votes > y.confidence / y.votes;
            });
            return order;
        }

        std::map<std::string, std::map<long, question_tally>> tasks_;
        long answers_ = 0;
    };

// ----------------------------------------------------------------------------------------

} // namespace dlib

#endif // DLIB_ARC_AGI_H_