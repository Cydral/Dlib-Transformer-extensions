// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_GPU_DaTA_CPP_
#define DLIB_GPU_DaTA_CPP_

// Only things that require CUDA are declared in this cpp file.  Everything else is in the
// gpu_data.h header so that it can operate as "header-only" code when using just the CPU.
#ifdef DLIB_USE_CUDA

#include "gpu_data.h"
#include <iostream>
#include "cuda_utils.h"
#include <cstring>
#include <cuda.h>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    void memcpy (
        gpu_data& dest, 
        const gpu_data& src
    )
    {
        DLIB_CASSERT(dest.size() == src.size());
        if (src.size() == 0 || &dest == &src)
            return;

        memcpy(dest,0, src, 0, src.size());
    }

    void memcpy (
        gpu_data& dest, 
        size_t dest_offset,
        const gpu_data& src,
        size_t src_offset,
        size_t num
    )
    {
        DLIB_CASSERT(dest_offset + num <= dest.size());
        DLIB_CASSERT(src_offset + num <= src.size());
        if (num == 0)
            return;

        // if there is aliasing
        if (&dest == &src && std::max(dest_offset, src_offset) < std::min(dest_offset,src_offset)+num)
        {
            // if they perfectly alias each other then there is nothing to do
            if (dest_offset == src_offset)
                return;
            else
                std::memmove(dest.host()+dest_offset, src.host()+src_offset, sizeof(float)*num);
        }
        else
        {
            /* Both operands are pinned for the duration of the copy.  Without this, a
               large destination can push its own source out of the device between the two
               accessor calls below, since the second call is what triggers the eviction
               and the first has already handed out its pointer. */
            device_scope keep_both{&dest, &src};

            // if we write to the entire thing then we can use device_write_only()
            if (dest_offset == 0 && num == dest.size())
            {
                // copy the memory efficiently based on which copy is current in each object.
                if (src.device_ready())
                    CHECK_CUDA(cudaMemcpy(dest.device_write_only(), src.device()+src_offset,  num*sizeof(float), cudaMemcpyDeviceToDevice));
                else 
                    CHECK_CUDA(cudaMemcpy(dest.device_write_only(), src.host()+src_offset,    num*sizeof(float), cudaMemcpyHostToDevice));
            }
            else
            {
                // copy the memory efficiently based on which copy is current in each object.
                if (dest.device_ready() && src.device_ready())
                    CHECK_CUDA(cudaMemcpy(dest.device()+dest_offset, src.device()+src_offset, num*sizeof(float), cudaMemcpyDeviceToDevice));
                else if (!dest.device_ready() && src.device_ready())
                    CHECK_CUDA(cudaMemcpy(dest.host()+dest_offset, src.device()+src_offset,   num*sizeof(float), cudaMemcpyDeviceToHost));
                else if (dest.device_ready() && !src.device_ready())
                    CHECK_CUDA(cudaMemcpy(dest.device()+dest_offset, src.host()+src_offset,   num*sizeof(float), cudaMemcpyHostToDevice));
                else 
                    CHECK_CUDA(cudaMemcpy(dest.host()+dest_offset, src.host()+src_offset,     num*sizeof(float), cudaMemcpyHostToHost));
            }
        }
    }
// ----------------------------------------------------------------------------------------

    void synchronize_stream(cudaStream_t stream)
    {
#if !defined CUDA_VERSION
#error CUDA_VERSION not defined
#elif CUDA_VERSION >= 9020 && CUDA_VERSION < 11000
        // We will stop using this alternative version with cuda V11, hopefully the bug in
        // cudaStreamSynchronize is fixed by then.
        //
        // This should be pretty much the same as cudaStreamSynchronize, which for some
        // reason makes training freeze in some cases.
        // (see https://github.com/davisking/dlib/issues/1513)
        while (true)
        {
            cudaError_t err = cudaStreamQuery(stream);
            switch (err)
            {
            case cudaSuccess: return;      // now we are synchronized
            case cudaErrorNotReady: break; // continue waiting
            default: CHECK_CUDA(err);      // unexpected error: throw
            }
        }
#else // CUDA_VERSION
        CHECK_CUDA(cudaStreamSynchronize(stream));
#endif // CUDA_VERSION
    }

    void gpu_data::
    wait_for_transfer_to_finish() const
    {
        if (have_active_transfer)
        {
            synchronize_stream((cudaStream_t)cuda_stream.get());
            have_active_transfer = false;
            // Check for errors.  These calls to cudaGetLastError() are what help us find
            // out if our kernel launches have been failing.
            CHECK_CUDA(cudaGetLastError());
        }
    }

    void gpu_data::
    copy_to_device() const
    {
        // We want transfers to the device to always be concurrent with any device
        // computation.  So we use our non-default stream to do the transfer.
        async_copy_to_device();
        wait_for_transfer_to_finish();
    }

    void gpu_data::
    copy_to_host() const
    {
        if (!host_current)
        {
            /* The host mirror can be absent when the extended memory subsystem has pushed
               this block down to the file tier.  Callers reach this function through the
               accessors, which reinstate the mirror first, so a null here means the block
               has no content anywhere and there is nothing to bring back. */
            if (!data_host || !data_device)
            {
                host_current = true;
                return;
            }
            wait_for_transfer_to_finish();
            CHECK_CUDA(cudaMemcpy(data_host.get(), data_device.get(), data_size*sizeof(float), cudaMemcpyDeviceToHost));
            host_current = true;
            // At this point we know our RAM block isn't in use because cudaMemcpy()
            // implicitly syncs with the device. 
            device_in_use = false;
            // Check for errors.  These calls to cudaGetLastError() are what help us find
            // out if our kernel launches have been failing.
            CHECK_CUDA(cudaGetLastError());
        }
    }

    void gpu_data::
    async_copy_to_device() const
    {
        if (!device_current)
        {
            if (!data_device)
            {
                /* The trainer calls this to overlap the next batch with the current step,
                   so the block is wanted even though nothing has read it yet.  Bring it
                   back rather than silently skipping the prefetch that was asked for. */
                if (xrec)
                    xmem::restore_device(xrec, true);

                if (!data_device)
                    return;
            }

            // The restore may have satisfied the request on its own, which happens when
            // the content came straight from the file tier into the device buffer.
            if (!device_current)
            {
                if (device_in_use)
                {
                    // Wait for any possible CUDA kernels that might be using our memory block to
                    // complete before we overwrite the memory.
                    synchronize_stream(0);
                    device_in_use = false;
                }
                CHECK_CUDA(cudaMemcpyAsync(data_device.get(), data_host.get(), data_size*sizeof(float), cudaMemcpyHostToDevice, (cudaStream_t)cuda_stream.get()));
                have_active_transfer = true;
                device_current = true;
            }
        }
    }

    void gpu_data::
    set_size(
        size_t new_size
    )
    {
        if (new_size == 0)
        {
            if (device_in_use)
            {
                // Wait for any possible CUDA kernels that might be using our memory block to
                // complete before we free the memory.
                synchronize_stream(0);
                device_in_use = false;
            }
            wait_for_transfer_to_finish();
            if (xrec)
            {
                xmem::unregister_block(xrec);
                xrec = nullptr;
            }
            data_size = 0;
            host_current = true;
            device_current = true;
            device_in_use = false;
            data_host.reset();
            data_device.reset();
        }
        else if (new_size != data_size)
        {
            if (device_in_use)
            {
                // Wait for any possible CUDA kernels that might be using our memory block to
                // complete before we free the memory.
                synchronize_stream(0);
                device_in_use = false;
            }
            wait_for_transfer_to_finish();
            if (xrec)
            {
                xmem::unregister_block(xrec);
                xrec = nullptr;
            }
            data_size = new_size;
            host_current = true;
            device_current = true;
            device_in_use = false;
            xmem::note_block_created();

            try
            {
                CHECK_CUDA(cudaGetDevice(&the_device_id));

                // free memory blocks before we allocate new ones.
                data_host.reset();
                data_device.reset();

                void* data;
                const bool managed = xmem::active();

                /* Under the manager no host buffer is allocated here at all. A block that
                   is only ever read by a kernel never needs one, and a block that does gets
                   it on its first host access or when it is evicted: a window onto the
                   store where there is one, a pinned mirror otherwise, and in that case
                   bounded by the ceiling the manager keeps. Allocating it eagerly put the
                   whole graph into pinned host memory before a single tensor was used, which
                   is not something any ceiling can bound after the fact. */
                if (!managed)
                {
                    CHECK_CUDA(cudaMallocHost(&data, new_size*sizeof(float)));
                    // Note that we don't throw exceptions since the free calls are invariably
                    // called in destructors.  They also shouldn't fail anyway unless someone
                    // is resetting the GPU card in the middle of their program.
                    data_host.reset((float*)data, [](float* ptr){
                        auto err = cudaFreeHost(ptr);
                        if(err!=cudaSuccess)
                            std::cerr << "cudaFreeHost() failed. Reason: " << cudaGetErrorString(err) << std::endl;
                    });
                }

                if (managed)
                {
                    /* No device buffer is allocated here.  The block starts off the device
                       and the first device access brings it in under the budget, which is
                       the whole point: a graph can now describe more memory than the card
                       holds, because describing it no longer occupies it. */
                    xrec = xmem::register_block(this, new_size*sizeof(float), the_device_id);
                    if (xrec)
                    {
                        device_current = false;
                        if (!data_host)
                            host_current = false;
                    }
                }

                if (!xrec)
                {
                    if (!data_host)
                    {
                        CHECK_CUDA(cudaMallocHost(&data, new_size*sizeof(float)));
                        data_host.reset((float*)data, [](float* ptr){
                            auto err = cudaFreeHost(ptr);
                            if(err!=cudaSuccess)
                                std::cerr << "cudaFreeHost() failed. Reason: " << cudaGetErrorString(err) << std::endl;
                        });
                    }
                    CHECK_CUDA(cudaMalloc(&data, new_size*sizeof(float)));
                    data_device.reset((float*)data, [](float* ptr){
                        auto err = cudaFree(ptr);
                        if(err!=cudaSuccess)
                            std::cerr << "cudaFree() failed. Reason: " << cudaGetErrorString(err) << std::endl;
                    });
                }

                if (!cuda_stream)
                {
                    cudaStream_t cstream;
                    CHECK_CUDA(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
                    cuda_stream.reset(cstream, [](void* ptr){
                        auto err = cudaStreamDestroy((cudaStream_t)ptr);
                        if(err!=cudaSuccess)
                            std::cerr << "cudaStreamDestroy() failed. Reason: " << cudaGetErrorString(err) << std::endl;
                    });
                }

            }
            catch(...)
            {
                set_size(0);
                throw;
            }
        }
    }

// ----------------------------------------------------------------------------------------
}

#endif // DLIB_USE_CUDA

#endif // DLIB_GPU_DaTA_CPP_

