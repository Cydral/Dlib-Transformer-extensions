// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_AVX2_DLIB_CPP_
#define DLIB_DNN_AVX2_DLIB_CPP_

/*
    This file contains the functions of avx2_dlib.h.  It is compiled with the options of the
    rest of the library, checks the processor and the arguments, and hands the pointers of the
    tensors to the kernels of avx2_kernels.h, the only code compiled for AVX2.
*/

#include "avx2_dlib.h"
#include "avx2_kernels.h"
#include "gguf_blocks.h"
#include "../simd/cpu_features.h"
#include "../threads.h"

#include <cstdint>
#include <limits>

namespace dlib
{
    namespace avx2
    {

    // -----------------------------------------------------------------------------------

        bool is_available (
        )
        {
            // The processor is asked first: no function of the kernels runs on another one.
            static const bool available = get_cpu_features().avx2 && get_cpu_features().fma
                && get_cpu_features().f16c && kernels::compiled();
            return available;
        }

    // -----------------------------------------------------------------------------------

        bool tanh (
            tensor& dest,
            const tensor& src
        )
        {
            if (!is_available()) return false;
            const auto d = dest.host();
            const auto s = src.host();
            kernels::tanh(d, s, src.size());
            return true;
        }

        bool sigmoid (
            tensor& dest,
            const tensor& src
        )
        {
            if (!is_available()) return false;
            const auto d = dest.host();
            const auto s = src.host();
            kernels::sigmoid(d, s, src.size());
            return true;
        }

        bool silu (
            tensor& dest,
            const tensor& src
        )
        {
            if (!is_available()) return false;
            const auto d = dest.host();
            const auto s = src.host();
            kernels::silu(d, s, src.size());
            return true;
        }

        bool gelu (
            tensor& dest,
            const tensor& src
        )
        {
            if (!is_available()) return false;
            const auto d = dest.host();
            const auto s = src.host();
            kernels::gelu(d, s, src.size());
            return true;
        }

    // -----------------------------------------------------------------------------------

        bool softmax (
            tensor& dest,
            const tensor& src,
            operation_mode mode
        )
        {
            if (!is_available()) return false;
            DLIB_CASSERT(have_same_dimensions(dest, src));
            DLIB_CASSERT(mode == operation_mode::CHANNEL_WISE || mode == operation_mode::PLANE_WISE,
                "Invalid softmax mode");
            if (src.size() == 0) return true;
            size_t rows, n;
            const bool planes = mode == operation_mode::PLANE_WISE;
            if (planes)
            {
                // One softmax per row of each plane.
                rows = static_cast<size_t>(src.num_samples()) * src.k() * src.nr();
                n = static_cast<size_t>(src.nc());
            }
            else if (src.nr() * src.nc() == 1)
            {
                rows = static_cast<size_t>(src.num_samples());
                n = static_cast<size_t>(src.k());
            }
            else
                return false;
            const auto d = dest.host();
            const auto s = src.host();
            kernels::softmax_rows(d, s, rows, n, planes);
            return true;
        }

        bool softmax_all (
            tensor& dest,
            const tensor& src
        )
        {
            if (!is_available()) return false;
            DLIB_CASSERT(have_same_dimensions(dest, src));
            if (src.size() == 0) return true;
            const auto d = dest.host();
            const auto s = src.host();
            kernels::softmax_rows(d, s, static_cast<size_t>(src.num_samples()), src.size() / src.num_samples(), false);
            return true;
        }

    // -----------------------------------------------------------------------------------

        bool gguf_matvec (
            tensor& dest,
            const tensor& src,
            const tensor& weights,
            int type,
            long rows,
            long cols,
            size_t row_bytes
        )
        {
            if (!is_available()) return false;
            DLIB_CASSERT(gguf_blocks::known_format(type) && rows > 0 && cols > 0);
            DLIB_CASSERT(static_cast<long long>(rows) <= std::numeric_limits<int>::max()
                && static_cast<long long>(cols) <= std::numeric_limits<int>::max());
            DLIB_CASSERT(row_bytes != 0 && row_bytes == gguf_blocks::row_bytes(type, cols));
            DLIB_CASSERT(!is_same_object(dest, src) && !is_same_object(dest, weights));
            DLIB_CASSERT(src.size() % static_cast<size_t>(cols) == 0);
            const long n = static_cast<long>(src.size() / static_cast<size_t>(cols));
            DLIB_CASSERT(dest.size() == static_cast<size_t>(n) * static_cast<size_t>(rows));
            DLIB_CASSERT(weights.size() * sizeof(float) >= static_cast<size_t>(rows) * row_bytes);
            if (n == 0) return true;
            const uint8_t* w = reinterpret_cast<const uint8_t*>(weights.host());
            const float* x = src.host();
            float* y = dest.host_write_only();
            /*
                The rows are independent of one another.  A product of about a million
                multiplications takes a thread less time than the others take to start: it
                runs on the calling thread.
            */
            const long long multiplications = static_cast<long long>(rows) * cols * n;
            if (multiplications < 1048576 || rows < 2)
                kernels::gguf_matvec_rows(y, x, w, type, rows, cols, row_bytes, n, 0, rows);
            else
                parallel_for_blocked(0, rows, [&](long first, long last)
                {
                    kernels::gguf_matvec_rows(y, x, w, type, rows, cols, row_bytes, n, first, last);
                });
            return true;
        }

    // -----------------------------------------------------------------------------------

    }
}

#ifdef NO_MAKEFILE
#include "avx2_kernels.cpp"
#endif

#endif // DLIB_DNN_AVX2_DLIB_CPP_
