// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// Device side fingerprint for the extended memory subsystem.
//
// The subsystem has to decide, at every eviction, whether the block on the device still
// matches what the store holds. It cannot know: a layer receives its parameters through the
// non-const device(), which hands out a raw pointer, and nothing afterwards distinguishes a
// kernel that read that pointer from one that wrote through it. Assuming a write is the only
// safe reading, and it costs a transfer of the whole block down the bus on every eviction of
// every weight, for content that a forward pass never touches.
//
// Reading the block where it already is settles the question for a fiftieth of the price.
// Device memory runs at several hundred gigabytes a second against ten or so over PCIe, so
// hashing a block on the card and comparing sixteen bytes replaces moving it. Nothing above
// gpu_data has to be told anything, which is the point: a mechanism that required a network
// to declare its own weights would not be a memory manager, it would be a contract.
//
// The fingerprint is a pair of 64 bit accumulators over the block read as 32 bit words. Each
// word is mixed with its own index through a splitmix64 finaliser, so position matters and a
// permutation is not invisible, and the mixed values are then combined both by exclusive or
// and by addition. Both combiners are associative and commutative, which is what allows the
// reduction to be done in any order across threads and blocks, and using two of them costs
// nothing and makes the pair behave as 128 bits rather than 64.
//
// Two blocks that differ therefore collide with a probability around 2^-128, which is many
// orders of magnitude below the chance of an undetected memory error on the same data. What
// the fingerprint cannot do is tell a changed block from an unchanged one across a
// non-deterministic recomputation: a kernel that rewrites a block with bitwise identical
// values is reported as unchanged, which is correct, and one that rewrites it with values
// differing in the last mantissa bit is reported as changed, which is also correct.
//
// The cross block reduction is finished on the host. A block level reduction can be closed
// with __syncthreads(), which orders threads within one block and nothing beyond it; the
// partials then come back in a single small transfer rather than through a second kernel or
// a set of atomics.

#ifdef DLIB_USE_CUDA

#include "cuda_utils.h"

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace dlib
{
    namespace cuda
    {

    // ------------------------------------------------------------------------------------

        namespace
        {
            const unsigned fingerprint_threads = 256;
            const unsigned fingerprint_blocks  = 512;

            __global__ void fingerprint_kernel (
                const std::uint32_t* data,
                std::size_t words,
                unsigned long long* partial_xor,
                unsigned long long* partial_add
            )
            {
                __shared__ unsigned long long sx[fingerprint_threads];
                __shared__ unsigned long long sa[fingerprint_threads];

                unsigned long long x = 0, a = 0;
                const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;

                for (std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
                     i < words; i += stride)
                {
                    // splitmix64 finaliser over the word combined with its position
                    unsigned long long v = (unsigned long long)data[i] ^
                                           ((unsigned long long)i * 0x9E3779B97F4A7C15ULL);
                    v ^= v >> 30; v *= 0xBF58476D1CE4E5B9ULL;
                    v ^= v >> 27; v *= 0x94D049BB133111EBULL;
                    v ^= v >> 31;
                    x ^= v;
                    a += v;
                }

                sx[threadIdx.x] = x;
                sa[threadIdx.x] = a;
                __syncthreads();

                for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
                {
                    if (threadIdx.x < s)
                    {
                        sx[threadIdx.x] ^= sx[threadIdx.x + s];
                        sa[threadIdx.x] += sa[threadIdx.x + s];
                    }
                    __syncthreads();
                }

                if (threadIdx.x == 0)
                {
                    partial_xor[blockIdx.x] = sx[0];
                    partial_add[blockIdx.x] = sa[0];
                }
            }

            /*
                Scratch for the partials, allocated once. Sixteen kilobytes on the device and
                as much pinned on the host, whatever the size of the blocks being hashed.
            */
            class fingerprint_scratch
            {
            public:

                fingerprint_scratch ()
                {
                    const std::size_t n = fingerprint_blocks * sizeof(unsigned long long);
                    CHECK_CUDA(cudaMalloc(&dev_xor, n));
                    CHECK_CUDA(cudaMalloc(&dev_add, n));
                    CHECK_CUDA(cudaMallocHost(&host_xor, n));
                    CHECK_CUDA(cudaMallocHost(&host_add, n));
                }

                ~fingerprint_scratch ()
                {
                    if (dev_xor)  cudaFree(dev_xor);
                    if (dev_add)  cudaFree(dev_add);
                    if (host_xor) cudaFreeHost(host_xor);
                    if (host_add) cudaFreeHost(host_add);
                }

                fingerprint_scratch(const fingerprint_scratch&) = delete;
                fingerprint_scratch& operator=(const fingerprint_scratch&) = delete;

                unsigned long long* dev_xor  = nullptr;
                unsigned long long* dev_add  = nullptr;
                unsigned long long* host_xor = nullptr;
                unsigned long long* host_add = nullptr;
            };
        }

    // ------------------------------------------------------------------------------------

        void extended_memory_fingerprint (
            const void* device_ptr,
            std::size_t bytes,
            unsigned long long& out_xor,
            unsigned long long& out_add,
            cudaStream_t stream
        )
        {
            out_xor = 0;
            out_add = 0;
            if (!device_ptr || bytes < sizeof(std::uint32_t))
                return;

            static fingerprint_scratch scratch;

            const std::size_t words = bytes / sizeof(std::uint32_t);
            const unsigned blocks = (unsigned)std::min<std::size_t>(
                fingerprint_blocks, (words + fingerprint_threads - 1) / fingerprint_threads);

            fingerprint_kernel<<<blocks, fingerprint_threads, 0, stream>>>(
                (const std::uint32_t*)device_ptr, words, scratch.dev_xor, scratch.dev_add);
            CHECK_CUDA(cudaGetLastError());

            const std::size_t n = blocks * sizeof(unsigned long long);
            CHECK_CUDA(cudaMemcpyAsync(scratch.host_xor, scratch.dev_xor, n,
                                       cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaMemcpyAsync(scratch.host_add, scratch.dev_add, n,
                                       cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));

            for (unsigned i = 0; i < blocks; ++i)
            {
                out_xor ^= scratch.host_xor[i];
                out_add += scratch.host_add[i];
            }

            /* A block whose size is not a whole number of words is not something gpu_data
               produces, but folding the remainder in costs nothing and keeps the fingerprint
               a function of every byte. */
            const std::size_t tail = bytes - words * sizeof(std::uint32_t);
            if (tail != 0)
            {
                unsigned char buf[sizeof(std::uint32_t)] = {0};
                CHECK_CUDA(cudaMemcpy(buf, (const char*)device_ptr + words*sizeof(std::uint32_t),
                                      tail, cudaMemcpyDeviceToHost));
                unsigned long long v = 0;
                for (std::size_t i = 0; i < tail; ++i)
                    v = (v << 8) | buf[i];
                v ^= (unsigned long long)words * 0x9E3779B97F4A7C15ULL;
                v ^= v >> 30; v *= 0xBF58476D1CE4E5B9ULL;
                v ^= v >> 27; v *= 0x94D049BB133111EBULL;
                v ^= v >> 31;
                out_xor ^= v;
                out_add += v;
            }
        }

    // ------------------------------------------------------------------------------------

    }
}

#endif // DLIB_USE_CUDA
