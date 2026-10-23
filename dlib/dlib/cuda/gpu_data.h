// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_GPU_DaTA_H_
#define DLIB_GPU_DaTA_H_

#include "gpu_data_abstract.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <cstring>
#include "cuda_errors.h"
#include "extended_memory.h"
#include "../serialize.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class gpu_data 
    {
        /*!
            CONVENTION
                - if (size() != 0) then
                    - data_host == a pointer to size() floats in CPU memory, unless the
                      extended memory subsystem has released the host mirror, in which case
                      data_host == nullptr and host_current == false.
                - if (data_device) then 
                    - data_device == a pointer to size() floats in device memory.

                - if (there might be an active async transfer from host to device) then
                    - have_active_transfer == true

                - We use the host_current and device_current bools to keep track of which
                  copy of the data (or both) are most current.  e.g. if the CPU has
                  modified the data and it hasn't been copied to the device yet then
                  host_current==true and device_current==false.

                  Similarly, we use device_in_use==true to indicate that device() has been
                  called and no operation to wait for all CUDA kernel completion has been
                  executed.  So if device_in_use==true then there might be a CUDA kernel
                  executing that is using the device memory block contained in this object.

                - if (xrec != nullptr) then
                    - this block is managed by the extended memory subsystem and xrec
                      points at its record there.  Both buffers may then be released and
                      reinstated behind the accessors, which is why every accessor consults
                      the subsystem before touching a pointer.  data_host may in that case
                      be a window onto the subsystem's mapped store rather than a pinned
                      buffer of its own.  When xrec == nullptr the object behaves exactly
                      as it does in stock dlib.
        !*/
    public:

        gpu_data(
        ) : data_size(0), host_current(true), device_current(true),have_active_transfer(false),device_in_use(false), the_device_id(0), xrec(nullptr)
        {
        }

        ~gpu_data()
        {
            if (xrec)
                xmem::unregister_block(xrec);
        }

        // Not copyable
        gpu_data(const gpu_data&) = delete;
        gpu_data& operator=(const gpu_data&) = delete;

        // but is movable
        gpu_data(gpu_data&& item) : gpu_data() { swap(item); }
        gpu_data& operator=(gpu_data&& item) { swap(item); return *this; }

        int device_id() const { return the_device_id; }

#ifdef DLIB_USE_CUDA
        void async_copy_to_device() const; 
        void set_size(size_t new_size);
#else
        // Note that calls to host() or device() will block until any async transfers are complete.
        void async_copy_to_device() const{}

        void set_size(size_t new_size)
        {
            if (new_size == 0)
            {
                forget_copies();
                data_size = 0;
                host_current = true;
                device_current = true;
                device_in_use.store(false, std::memory_order_relaxed);
                data_host.reset();
                data_device.reset();
            }
            else if (new_size != data_size)
            {
                forget_copies();
                data_size = new_size;
                host_current = true;
                device_current = true;
                device_in_use.store(false, std::memory_order_relaxed);
                data_host.reset(new float[new_size], std::default_delete<float[]>());
                data_device.reset();
            }
        }
#endif

        const float* host() const 
        { 
            if (!enter_host(true, false))
                copy_to_host();
            return data_host.get(); 
        }

        float* host()
        {
            forget_copies();
            if (!enter_host(true, true))
            {
                copy_to_host();
                device_current = false;
            }
            return data_host.get();
        }

        float* host_write_only()
        {
            forget_copies();
            if (!enter_host(false, true))
            {
                host_current = true;
                device_current = false;
            }
            return data_host.get(); 
        }

        const float* device() const 
        { 
#ifndef DLIB_USE_CUDA
            DLIB_CASSERT(false, "CUDA NOT ENABLED");
#endif
            enter_device(true, false);
            copy_to_device();
            device_in_use.store(true, std::memory_order_relaxed);
            return data_device.get(); 
        }

        float* device()
        {
#ifndef DLIB_USE_CUDA
            DLIB_CASSERT(false, "CUDA NOT ENABLED");
#endif
            forget_copies();
            enter_device(true, true);
            copy_to_device();
            host_current = false;
            device_in_use.store(true, std::memory_order_relaxed);
            return data_device.get(); 
        }

        float* device_write_only()
        {
#ifndef DLIB_USE_CUDA
            DLIB_CASSERT(false, "CUDA NOT ENABLED");
#endif
            forget_copies();
            enter_device(false, true);
            wait_for_transfer_to_finish();
            host_current = false;
            device_current = true;
            device_in_use.store(true, std::memory_order_relaxed);
            return data_device.get(); 
        }

        bool host_ready (
        ) const { return host_current; }

        bool device_ready (
        ) const { return device_current && !have_active_transfer; }

        size_t size() const { return data_size; }

        uint64_t copy_mark (
        ) const { return mark.load(std::memory_order_relaxed); }

        uint64_t set_copy_mark (
            uint64_t fresh
        ) const
        {
            uint64_t current = 0;
            if (mark.compare_exchange_strong(current, fresh, std::memory_order_relaxed))
                return fresh;
            return current;
        }

        void swap (gpu_data& item)
        {
            /* The subsystem tracks a block through a pointer back to the object holding it,
               and a swap moves the buffers and the records in two separate steps. Held
               across both, this keeps the observation thread from reading a record whose
               owner has come apart from its memory. It costs nothing when neither block is
               managed. */
            xmem::registry_guard guard(xrec != nullptr || item.xrec != nullptr);

            std::swap(data_size, item.data_size);
            std::swap(host_current, item.host_current);
            std::swap(device_current, item.device_current);
            std::swap(have_active_transfer, item.have_active_transfer);
            /* The flag describes the device buffer, so it moves with it. Left behind, a
               buffer a kernel is still using would be released without the synchronization
               the flag exists to trigger. */
            const bool in_use = device_in_use.load(std::memory_order_relaxed);
            device_in_use.store(item.device_in_use.load(std::memory_order_relaxed), std::memory_order_relaxed);
            item.device_in_use.store(in_use, std::memory_order_relaxed);
            // The mark describes the values, so it moves with the buffers as well.
            const uint64_t m = mark.load(std::memory_order_relaxed);
            mark.store(item.mark.load(std::memory_order_relaxed), std::memory_order_relaxed);
            item.mark.store(m, std::memory_order_relaxed);
            std::swap(data_host, item.data_host);
            std::swap(data_device, item.data_device);
            std::swap(cuda_stream, item.cuda_stream);
            std::swap(the_device_id, item.the_device_id);
            std::swap(xrec, item.xrec);
            // Both records have to learn where their block went before the guard lets go.
            if (xrec)       xrec->owner = this;
            if (item.xrec)  item.xrec->owner = &item;
        }

    private:

        // Called by every access that may change the values: copies taken under the mark
        // stop being current.
        void forget_copies (
        )
        {
            if (mark.load(std::memory_order_relaxed) != 0)
                mark.store(0, std::memory_order_relaxed);
        }

#ifdef DLIB_USE_CUDA
        void copy_to_device() const;
        void copy_to_host() const;
        void wait_for_transfer_to_finish() const;

        void enter_device(bool need_content, bool writes) const
        {
            if (xrec)
                xmem::before_device(xrec, need_content, writes);
        }

        /*
            True when the subsystem has carried out the whole host access itself: the host
            buffer, the copy back from the device and the flags, all under its lock. The
            accessor then writes no flag of its own, because the thread filling a tensor on
            the host is not necessarily the one computing, and the computing thread may move
            the same block at any moment.
        */
        bool enter_host(bool need_content, bool writes) const
        {
            if (!xrec)
                return false;
            xmem::before_host(xrec, need_content, writes);
            return true;
        }
#else
        void copy_to_device() const{}
        void copy_to_host() const{}
        void wait_for_transfer_to_finish() const{}
        void enter_device(bool,bool) const{}
        bool enter_host(bool,bool) const{ return false; }
#endif

        friend class xmem::manager;

        size_t data_size;
        mutable bool host_current;
        mutable bool device_current;
        mutable bool have_active_transfer;
        /* Atomic because the manager clears it on many blocks at once, from whichever
           thread synchronised the device, while the owner's thread may be reading it. */
        mutable std::atomic<bool> device_in_use;

        mutable std::shared_ptr<float> data_host;
        mutable std::shared_ptr<float> data_device;
        std::shared_ptr<void> cuda_stream;
        int the_device_id;

        // Null unless the extended memory subsystem manages this block.
        mutable xmem::block_record* xrec;

        /* Set by set_copy_mark() and cleared by forget_copies(). Atomic because a reader
           marks blocks it only reads, possibly from another thread than their writer. */
        mutable std::atomic<uint64_t> mark{0};
    };

    inline void serialize(const gpu_data& item, std::ostream& out)
    {
        int version = 1;
        serialize(version, out);
        serialize(item.size(), out);
        auto data = item.host();
        for (size_t i = 0; i < item.size(); ++i)
            serialize(data[i], out);
    }

    inline void deserialize(gpu_data& item, std::istream& in)
    {
        int version;
        deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::gpu_data.");
        size_t s;
        deserialize(s, in);
        item.set_size(s);
        auto data = item.host();
        for (size_t i = 0; i < item.size(); ++i)
            deserialize(data[i], in);
    }

#ifdef DLIB_USE_CUDA
    void memcpy (gpu_data& dest, const gpu_data& src);

    void memcpy (
        gpu_data& dest, 
        size_t dest_offset,
        const gpu_data& src,
        size_t src_offset,
        size_t num
    );

#else

    inline void memcpy (gpu_data& dest, const gpu_data& src)
    {
        DLIB_CASSERT(dest.size() == src.size());
        if (src.size() == 0 || &dest == &src)
            return;
        std::memcpy(dest.host_write_only(), src.host(), sizeof(float)*src.size());
    }

    inline void memcpy (
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
            // if we write to the entire thing then we can use host_write_only()
            if (dest_offset == 0 && num == dest.size())
                std::memcpy(dest.host_write_only(), src.host()+src_offset, sizeof(float)*num);
            else
                std::memcpy(dest.host()+dest_offset, src.host()+src_offset, sizeof(float)*num);
        }
    }
#endif

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GPU_DaTA_H_

