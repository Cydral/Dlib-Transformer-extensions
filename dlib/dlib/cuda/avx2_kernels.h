// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AVX2_KERNELS_H_
#define DLIB_AVX2_KERNELS_H_

#include <cstddef>
#include <cstdint>

namespace dlib
{
    namespace avx2
    {
        namespace kernels
        {
            /*!
                The functions of this namespace are compiled for AVX2, FMA and F16C whatever the
                instructions the rest of the library is compiled for (avx2_kernels.cpp has its
                own compiler options).  A program calls them only on a processor that has these
                instructions, which avx2::is_available() tells; avx2_dlib.h holds the functions
                that take tensors and call them.

                They take pointers and sizes, and avx2_kernels.cpp includes no other header of
                dlib than this one and gguf_blocks.h, read under another namespace: no inline
                function of the library is then compiled a second time with other instructions
                and kept by the linker for the whole program.
            !*/

            bool compiled (
            );
            /*!
                ensures
                    - returns true when the build compiled the functions below: an x86 target
                      whose compiler was given the instructions.  The other functions do
                      nothing when it returns false.
            !*/

            void tanh (float* d, const float* s, size_t n);
            void sigmoid (float* d, const float* s, size_t n);
            void silu (float* d, const float* s, size_t n);
            void gelu (float* d, const float* s, size_t n);
            void mish (float* d, const float* s, size_t n);
            void elu (float* d, const float* s, size_t n, float alpha);
            /*!
                requires
                    - d and s point to n values; they are the same pointer or do not overlap.
                ensures
                    - d[i] receives the function of s[i], as cpu::tanh(), cpu::sigmoid(),
                      cpu::silu(), cpu::gelu(), cpu::mish() and cpu::elu() compute it, eight
                      values at a time.  The exponentials are polynomials whose results lie
                      within a few units of the last place of those of the math library;
                      gelu() uses an approximation of the error function whose absolute error
                      stays under 1.5e-7.  A value that is not a number stays one.
            !*/

            void softmax_rows (float* d, const float* s, size_t rows, size_t n, bool zero_empty_rows);
            /*!
                requires
                    - d and s point to rows*n values; they are the same pointer or do not
                      overlap.
                ensures
                    - Each row of n consecutive values of s becomes, in d,
                      exp(s - max)/sum(exp(s - max)) over the row, as cpu::softmax() computes
                      it: a row that holds a value that is not a number, or whose largest
                      value is infinite, becomes values that are not numbers.
                    - With zero_empty_rows, a row whose values are all minus infinity, or not
                      numbers, becomes zeros, as the rows of cpu::softmax() over the columns
                      do: such a row is a row of an attention that every position masks.
            !*/

            void gguf_matvec_rows (
                float* y,
                const float* x,
                const uint8_t* w,
                int type,
                long rows,
                long cols,
                size_t row_bytes,
                long n,
                long first,
                long last
            );
            /*!
                requires
                    - The arguments of cpu::gguf_matvec(): w holds rows rows of row_bytes
                      bytes in the GGUF format type, x holds n vectors of cols values and y
                      n vectors of rows values.
                    - 0 <= first <= last <= rows
                ensures
                    - y[t*rows + r] receives the product of row r of the matrix with the
                      vector t, for every r in [first, last) and every t in [0, n).  Calls
                      over disjoint ranges of rows write disjoint values and can run on
                      different threads.
            !*/
        }
    }
}

#endif // DLIB_AVX2_KERNELS_H_
