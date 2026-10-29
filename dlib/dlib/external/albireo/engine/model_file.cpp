// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "model_file.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <dlib/data_io/gguf_dequantize.h>

namespace albireo
{

#if defined(_WIN32)
    namespace
    {
        std::wstring widen(const std::string& s)
        {
            if (s.empty()) return std::wstring();
            const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(),
                static_cast<int>(s.size()), nullptr, 0);
            if (n <= 0) throw std::runtime_error("invalid UTF-8 in path " + s);
            std::wstring w(static_cast<size_t>(n), L'\0');
            MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), &w[0], n);
            return w;
        }
    }

    mapped_file::mapped_file(const std::string& utf8_path)
    {
        const std::wstring w = widen(utf8_path);
        HANDLE f = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (f == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open " + utf8_path + " (error " + std::to_string(GetLastError()) + ")");
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0)
        {
            CloseHandle(f);
            throw std::runtime_error("cannot read the size of " + utf8_path);
        }
        HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m)
        {
            const DWORD e = GetLastError();
            CloseHandle(f);
            throw std::runtime_error("cannot map " + utf8_path + " (error " + std::to_string(e) + ")");
        }
        void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!p)
        {
            const DWORD e = GetLastError();
            CloseHandle(m);
            CloseHandle(f);
            throw std::runtime_error("cannot map a view of " + utf8_path + " (error " + std::to_string(e) + ")");
        }
        file_ = f;
        mapping_ = m;
        data_ = static_cast<const uint8_t*>(p);
        size_ = static_cast<uint64_t>(sz.QuadPart);
    }

    mapped_file::~mapped_file()
    {
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(static_cast<HANDLE>(mapping_));
        if (file_) CloseHandle(static_cast<HANDLE>(file_));
    }

    void mapped_file::prefetch(uint64_t offset, uint64_t length) const
    {
        if (!data_ || offset >= size_) return;
        if (offset + length > size_) length = size_ - offset;
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = const_cast<uint8_t*>(data_ + offset);
        range.NumberOfBytes = static_cast<SIZE_T>(length);
        // Available from Windows 8; a failure only means the pages come in on first use.
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    }
#else
    mapped_file::mapped_file(const std::string& utf8_path)
    {
        const int fd = ::open(utf8_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot open " + utf8_path + ": " + std::strerror(errno));
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size <= 0)
        {
            ::close(fd);
            throw std::runtime_error("cannot read the size of " + utf8_path);
        }
        void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map " + utf8_path + ": " + std::strerror(errno));
        data_ = static_cast<const uint8_t*>(p);
        size_ = static_cast<uint64_t>(st.st_size);
    }

    mapped_file::~mapped_file()
    {
        if (data_) munmap(const_cast<uint8_t*>(data_), static_cast<size_t>(size_));
    }

    void mapped_file::prefetch(uint64_t offset, uint64_t length) const
    {
        if (!data_ || offset >= size_) return;
        if (offset + length > size_) length = size_ - offset;
        const uint64_t page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
        const uint64_t start = offset / page * page;
        madvise(const_cast<uint8_t*>(data_) + start, static_cast<size_t>(offset + length - start), MADV_WILLNEED);
    }
#endif

    model_file::model_file(const std::string& utf8_path) : path_(utf8_path)
    {
        file_.reset(new mapped_file(utf8_path));
        reader_.reset(new dlib::gguf_reader(file_->data(), file_->size()));
        const uint64_t base = reader_->data_offset();
        tensors_.reserve(reader_->tensors().size());
        for (const auto& t : reader_->tensors())
        {
            tensor_view v;
            v.name = t.name;
            v.type = t.type;
            v.dims = t.dims;
            v.data = file_->data() + base + t.offset;
            v.nbytes = t.nbytes;
            index_[v.name] = tensors_.size();
            tensors_.push_back(std::move(v));
        }
    }

    const tensor_view* model_file::find(const std::string& name) const
    {
        auto it = index_.find(name);
        return it == index_.end() ? nullptr : &tensors_[it->second];
    }

    weight_matrix model_file::optional_matrix(const std::string& name, long rows, long cols) const
    {
        const tensor_view* t = find(name);
        if (!t) return weight_matrix();
        if (t->dims.empty() || static_cast<long>(t->dims[0]) != cols
            || static_cast<long>(t->elements() / t->dims[0]) != rows)
        {
            std::string shape;
            for (uint64_t d : t->dims) shape += (shape.empty() ? "" : "x") + std::to_string(d);
            throw std::runtime_error("tensor " + name + " has shape " + shape + ", expected "
                + std::to_string(cols) + "x" + std::to_string(rows));
        }
        if (!qk_supported(t->type))
            throw std::runtime_error("tensor " + name + " uses a quantization type this engine does not support ("
                + std::to_string(static_cast<unsigned>(t->type)) + ")");
        weight_matrix m;
        m.type = t->type;
        m.data = t->data;
        m.rows = rows;
        m.cols = cols;
        m.row_bytes = qk_row_bytes(t->type, cols);
        if (m.row_bytes * static_cast<size_t>(rows) != t->nbytes)
            throw std::runtime_error("tensor " + name + " does not hold whole rows");
        return m;
    }

    weight_matrix model_file::matrix(const std::string& name, long rows, long cols) const
    {
        weight_matrix m = optional_matrix(name, rows, cols);
        if (!m.valid()) throw std::runtime_error("missing tensor " + name);
        return m;
    }

    weight_matrix model_file::expert_matrix(const std::string& name, long rows, long cols, long expert,
        long n_experts) const
    {
        const tensor_view* t = find(name);
        if (!t) throw std::runtime_error("missing tensor " + name);
        if (t->dims.size() != 3 || static_cast<long>(t->dims[0]) != cols || static_cast<long>(t->dims[1]) != rows
            || static_cast<long>(t->dims[2]) != n_experts)
            throw std::runtime_error("tensor " + name + " does not hold " + std::to_string(n_experts) + " experts of "
                + std::to_string(rows) + "x" + std::to_string(cols));
        if (!qk_supported(t->type))
            throw std::runtime_error("tensor " + name + " uses an unsupported quantization type");
        weight_matrix m;
        m.type = t->type;
        m.rows = rows;
        m.cols = cols;
        m.row_bytes = qk_row_bytes(t->type, cols);
        m.data = t->data + static_cast<size_t>(expert) * m.row_bytes * static_cast<size_t>(rows);
        return m;
    }

    bool model_file::optional_floats(const std::string& name, size_t count, std::vector<float>& out) const
    {
        const tensor_view* t = find(name);
        if (!t) return false;
        if (t->elements() != count)
            throw std::runtime_error("tensor " + name + " holds " + std::to_string(t->elements())
                + " values, expected " + std::to_string(count));
        out.resize(count);
        const long block = static_cast<long>(dlib::get_ggml_type_traits(t->type).block_size);
        if (count % static_cast<size_t>(block) != 0)
            throw std::runtime_error("tensor " + name + " is not block aligned");
        qk_dequantize_row(t->type, t->data, out.data(), static_cast<long>(count));
        return true;
    }

    std::vector<float> model_file::floats(const std::string& name, size_t count) const
    {
        std::vector<float> v;
        if (!optional_floats(name, count, v)) throw std::runtime_error("missing tensor " + name);
        return v;
    }

    void model_file::warm_up(bool) const
    {
        file_->prefetch(reader_->data_offset(), file_->size() - reader_->data_offset());
    }

}
