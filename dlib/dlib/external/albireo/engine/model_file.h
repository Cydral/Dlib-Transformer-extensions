// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_MODEL_FILE_H_
#define ALBIREO_ENGINE_MODEL_FILE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlib/data_io/gguf_reader.h>

#include "qkernels.h"

namespace albireo
{

    class mapped_file
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A file mapped read-only into the address space of the process.  The
                weights of a model are read in place from the mapping: the operating
                system loads the pages on first use, shares them between processes and
                drops them under memory pressure without writing anything back, which a
                copy in the heap would not allow.

                The path is UTF-8 on every platform.
        !*/
    public:
        explicit mapped_file(const std::string& utf8_path);
        ~mapped_file();
        mapped_file(const mapped_file&) = delete;
        mapped_file& operator=(const mapped_file&) = delete;

        const uint8_t* data() const { return data_; }
        uint64_t size() const { return size_; }

        // Asks the system to read a range ahead of its use.
        void prefetch(uint64_t offset, uint64_t length) const;

    private:
        const uint8_t* data_ = nullptr;
        uint64_t size_ = 0;
#if defined(_WIN32)
        void* file_ = nullptr;
        void* mapping_ = nullptr;
#endif
    };

    struct tensor_view
    {
        std::string name;
        dlib::ggml_type type = dlib::ggml_type::F32;
        std::vector<uint64_t> dims;   // ggml order, dims[0] contiguous
        const uint8_t* data = nullptr;
        uint64_t nbytes = 0;

        uint64_t elements() const
        {
            uint64_t n = 1;
            for (uint64_t d : dims) n *= d;
            return n;
        }
    };

    class model_file
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A GGUF model opened for inference: its metadata, parsed by
                dlib::gguf_reader from the mapping, and a view of every tensor pointing
                into the mapping.
        !*/
    public:
        explicit model_file(const std::string& utf8_path);

        const dlib::gguf_reader& reader() const { return *reader_; }
        const std::string& path() const { return path_; }
        uint64_t size() const { return file_->size(); }

        const tensor_view* find(const std::string& name) const;
        bool has(const std::string& name) const { return find(name) != nullptr; }

        // The tensor of the given name as a matrix of rows x cols, rows being its outer
        // dimension.  Throws when the tensor is missing or has other dimensions.
        weight_matrix matrix(const std::string& name, long rows, long cols) const;

        // Same for an optional tensor, which yields an invalid matrix when absent.
        weight_matrix optional_matrix(const std::string& name, long rows, long cols) const;

        // A small tensor expanded to floats (norms, biases, frequency factors).  Throws
        // when it is missing or does not hold exactly count values.
        std::vector<float> floats(const std::string& name, size_t count) const;
        bool optional_floats(const std::string& name, size_t count, std::vector<float>& out) const;

        // Expert slice e of a 3D tensor holding n_experts matrices of rows x cols.
        weight_matrix expert_matrix(const std::string& name, long rows, long cols, long expert, long n_experts) const;

        // Reads the whole data section once so that the first token does not wait for the
        // disk, on a background thread when asynchronous is true.
        void warm_up(bool asynchronous) const;

    private:
        std::string path_;
        std::unique_ptr<mapped_file> file_;
        std::unique_ptr<dlib::gguf_reader> reader_;
        std::vector<tensor_view> tensors_;
        std::unordered_map<std::string, size_t> index_;
    };

}

#endif // ALBIREO_ENGINE_MODEL_FILE_H_
