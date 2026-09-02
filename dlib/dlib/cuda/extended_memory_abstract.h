// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_EXTENDED_MEMORY_ABSTRACT_H_
#ifdef DLIB_EXTENDED_MEMORY_ABSTRACT_H_

#include <cstddef>
#include <iosfwd>
#include <string>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct extended_memory_options
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object holds the configuration of the extended memory subsystem, which
                lets a network occupy more memory than the device physically has by keeping
                only part of its tensors resident in VRAM and holding the rest in pinned
                host memory or in a mapped store on disk.

                The subsystem is off by default. When it is off, nothing in this header has
                any effect and gpu_data behaves exactly as it does in stock dlib.
        !*/

        size_t vram_budget;
        /*!
            The number of bytes of device memory the subsystem is allowed to hold in managed
            blocks. When set to 0 the budget is derived from the free memory reported by the
            device at the moment extended memory is enabled, scaled by vram_budget_fraction,
            which is usually the better choice because it adapts to whatever else is already
            on the card.

            The budget covers managed blocks and the workspaces allocated through
            cuda_data_ptr, which are routed through the same pool so that one can force the
            eviction of the other. What it cannot cover is the CUDA context and whatever
            cuDNN, cuBLAS and cuSOLVER allocate inside their own handles, which is why the
            default fraction leaves headroom. Compare device_peak_bytes plus scratch_bytes
            against what nvidia-smi reports to see how much of the card that remainder
            accounts for on a given machine.

            It is a target for the resident set rather than a hard ceiling on one allocation.
            A block larger than the budget still succeeds if it fits in device memory once
            everything evictable is gone; a block larger than the device itself is refused
            when it is sized, because residency management spreads the total across tiers and
            cannot split a single tensor.
        !*/

        double vram_budget_fraction;
        /*!
            Fraction of the free device memory used as the budget when vram_budget == 0.
            Values above 0.95 are rejected because the unmanaged allocations mentioned above
            still have to fit.
        !*/

        std::string store_path;
        /*!
            Directory in which the store file is created, or an empty string to run without a
            store. Without one, evicted blocks stay in pinned host memory and the run is
            bounded by host RAM; with one, they move into a mapping whose residency the
            kernel manages, and the run is bounded by disk instead.

            Use default_extended_memory_store_path() rather than a literal, so that where the
            store lives stays a deployment decision rather than a build-time one.

            The file is created with a unique name, unlinked immediately on POSIX and opened
            with FILE_FLAG_DELETE_ON_CLOSE on Windows, so it does not survive the process
            even after a crash.
        !*/

        size_t store_bytes;
        /*!
            Size of the store mapping. The file is sparse, so this reserves address space and
            a file length rather than disk blocks: only the pages actually written take room.
            Pick it generously. When set to 0 it defaults to eight times the device budget.

            The request is capped at ninety percent of the free space on the volume, with a
            line on stderr saying so. Without that cap the mapping would be created happily
            and then fail on a write, as a bus error rather than an exception, once the
            volume filled.

            A block that asks for a slot after the store is full falls back on a pinned
            mirror, which still works but reintroduces the host memory limit for that block.
        !*/

        size_t staging_bytes;
        /*!
            Size of the pinned buffer used to move data between the store and the device. It
            is split in two so that the host copy of one chunk overlaps the transfer of the
            one before it. This is the only pinned allocation in the subsystem whose size
            does not grow with the model.
        !*/

        size_t min_block_bytes;
        /*!
            Blocks smaller than this are registered for tracing but are never evicted and
            keep an ordinary pinned mirror. The transfer cost of a small tensor dominates its
            footprint, and the biases, scales and normalization parameters of a network are
            read on every layer.
        !*/

        unsigned hot_window;
        /*!
            Number of most recently handed out device pointers that are treated as live and
            are therefore never evicted. See the note on pointer liveness in
            extended_memory.h. This value must exceed the number of distinct tensors any
            single tensor operation holds pointers to at once, and should be raised when more
            than one thread computes on the same device.
        !*/

        unsigned lookahead;
        /*!
            Number of blocks the prefetcher tries to have brought in ahead of the cursor once
            an access cycle has been identified. This counts blocks that actually needed
            moving, not positions in the cycle: most of a network's accesses land on small
            tensors that never leave the device, and a horizon counted in positions would
            cover far fewer real transfers than its value suggests.

            Larger values hide more transfer latency and consume more of the budget. The
            observation thread paces itself off the measured access rate to stay roughly a
            quarter of a horizon ahead, so this value also sets how often it wakes.
        !*/

        unsigned advise_horizon;
        /*!
            Number of future accesses over which the store's pages are hinted into memory.
            This is the deeper of the two horizons: bringing a page in costs only a hint, so
            it runs well ahead of the transfers and gives the kernel time to fetch from disk
            before the block's turn arrives. Must be at least as large as lookahead.
        !*/

        unsigned idle_release_ms;
        /*!
            Milliseconds of inactivity after which the subsystem falls back to
            idle_release_keep of its budget. Zero, the default, never releases on inactivity.

            This is worth setting on any host that keeps more than one model, whether they
            live in one process or several, and it earns its keep in two different ways. Across
            processes it is the only mechanism that exists: nothing else returns memory a
            neighbour can use. Within one process the victim policy already favours the idle
            model, since a model absent from the observed cycle has no next use and therefore
            ranks ahead of everything still in play, but that work only happens when another
            model asks for room, which is to say on the critical path of an incoming request.
            Doing it during the pause moves it off that path.

            Blocks in the hot window are kept, which bounds what a release can reclaim and is
            what makes it safe: during inactivity the window has not moved, so it still covers
            anything a stalled thread might hold a pointer to.
        !*/

        double idle_release_keep;
        /*!
            Fraction of the budget to fall back to at the first step. The default of 0.5 keeps
            the model that was just used warm while dropping what was not, which is the right
            trade when the next request is as likely to be for the same model as for another
            one. Lower it towards zero on a host where requests alternate between models.
        !*/

        unsigned idle_purge_ms;
        /*!
            Milliseconds of inactivity after which everything releasable goes and the pool is
            trimmed, which is the call that actually returns memory to the driver: a buffer
            sitting in the pool is still charged to the process as far as anything outside it
            can tell. When set to 0 this is taken as eight times idle_release_ms.
        !*/

        bool predictive;
        /*!
            When true, an observation thread records the access sequence, identifies its
            period and uses the resulting schedule to prefetch, to warm the store and to pick
            eviction victims by furthest next use. When false the subsystem falls back to
            least recently used eviction and does neither.
        !*/

        bool verbose;
        /*!
            When true, the subsystem reports on stderr when it starts, when it identifies or
            loses an access cycle, and when the store fills up.
        !*/

        bool paranoid;
        /*!
            When true, every device block released back to the pool is first overwritten with
            a signalling NaN. Code that kept a device pointer across an eviction then produces
            visible NaNs instead of silently wrong results. This is a debugging aid: it costs
            one memset per eviction and should be left off in production.
        !*/

        extended_memory_options();
        /*!
            ensures
                - #vram_budget == 0
                - #vram_budget_fraction == 0.9
                - #store_path == ""
                - #store_bytes == 0
                - #staging_bytes == 64 MiB
                - #min_block_bytes == 1 MiB
                - #hot_window == 64
                - #lookahead == 24
                - #advise_horizon == 96
                - #predictive == true
                - #verbose == false
                - #paranoid == false
        !*/
    };

// ----------------------------------------------------------------------------------------

    struct extended_memory_stats
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A snapshot of the subsystem's counters, useful to check that the access cycle
                was found and that the transfer volume is what you expect.
        !*/

        bool     enabled;              // whether the subsystem is running
        size_t   vram_budget;          // the budget in bytes
        size_t   device_bytes;         // managed bytes currently resident on the device
        size_t   device_peak_bytes;    // high water mark of device_bytes
        size_t   scratch_bytes;        // workspace bytes held outside gpu_data
        size_t   pinned_bytes;         // bytes held in pinned host mirrors
        size_t   store_bytes;          // slots in use in the store
        size_t   store_capacity;       // size of the store mapping
        size_t   managed_blocks;       // number of registered blocks
        size_t   evictions;            // device blocks released to a lower tier
        size_t   restores;             // device blocks brought back on demand
        size_t   prefetch_hits;        // restores that had been anticipated
        size_t   prefetch_issued;      // restores started by the observation thread
        size_t   store_writes;         // block writes into the store
        size_t   store_reads;          // block reads out of the store
        size_t   pages_advised;        // slots hinted into memory ahead of use
        unsigned long long h2d_bytes;  // host to device traffic caused by the subsystem
        unsigned long long d2h_bytes;  // device to host traffic caused by the subsystem
        bool     cycle_locked;         // whether an access cycle is currently identified
        size_t   cycle_period;         // length of that cycle in accesses, 0 when not locked
        unsigned prefetch_depth;       // blocks the last prefetch pass managed to queue
        double   access_rate;          // block accesses per second, as measured
        size_t   idle_releases;        // idle sweeps performed, of either step
        size_t   idle_purges;          // of those, sweeps that also trimmed the pool
        size_t   idle_released_bytes;  // bytes freed by those sweeps
    };

// ----------------------------------------------------------------------------------------

    std::string default_extended_memory_store_path (
    );
    /*!
        ensures
            - returns the value of the DLIB_XMEM_STORE environment variable when it is set
              and not empty, so that where the store lives can be decided per machine without
              rebuilding.
            - otherwise returns the platform's temporary directory: TMPDIR or /tmp on POSIX,
              TEMP or TMP on Windows.
            - This function is available whether or not this build uses CUDA.
    !*/

// ----------------------------------------------------------------------------------------

    bool enable_extended_memory (
        const extended_memory_options& opts
    );
    /*!
        requires
            - opts.vram_budget_fraction is in the range (0, 0.95]
            - opts.hot_window > 0
            - opts.advise_horizon >= opts.lookahead
            - extended_memory_enabled() == false
            - no gpu_data block has been sized yet in this process
        ensures
            - If dlib was built with CUDA and a device is available, starts the extended
              memory subsystem on the currently selected device and returns true.
            - If dlib was built without CUDA, does nothing and returns false.
            - #extended_memory_enabled() == true
        throws
            - dlib::error if the subsystem is already running, if any tensor has already been
              sized, or if the directory named by store_path has no usable room. Both restrictions exist because a block settles where its host copy
              lives when it is sized, so a network built before this call would keep pinned
              mirrors for the rest of the run. The call belongs in the first statements of
              main().
            - cuda_error if the device cannot be queried or the store cannot be mapped.

            There is deliberately no way to switch the subsystem off again. Doing so would
            mean bringing every block back to VRAM at once, which is the thing the subsystem
            exists because you cannot do, and converting every mapped mirror into a pinned
            one, which doubles the host footprint at the worst moment.

            This call also registers an atexit handler that joins the observation thread.
            Nothing else needs a teardown step: the driver reclaims the device buffers and
            the pinned staging halves with the CUDA context, and the kernel unmaps the store
            and frees its blocks when the process ends, however it ends.
    !*/

    bool extended_memory_enabled (
    );
    /*!
        ensures
            - returns true if and only if the subsystem is currently running.
    !*/

    extended_memory_stats get_extended_memory_stats (
    );
    /*!
        ensures
            - returns a snapshot of the subsystem's counters. When the subsystem is not
              running, every field is zero except enabled, which is false.
    !*/

    void print_extended_memory_stats (
        std::ostream& out
    );
    /*!
        ensures
            - writes a short human readable summary of get_extended_memory_stats() to out.
    !*/

// ----------------------------------------------------------------------------------------

    class device_scope
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object pins a set of tensors on the device for as long as it is alive. A
                pinned block is never chosen as an eviction victim, so the raw pointers
                obtained from it stay valid.

                The subsystem already protects the most recently used pointers through
                extended_memory_options::hot_window, and that protection is enough for every
                operation in dlib itself. This object exists for code that holds a device
                pointer across an unusually long stretch, for instance a routine that gathers
                pointers to many tensors before launching a single kernel.

                Constructing one when the subsystem is not running costs nothing. At most 16
                blocks are pinned by one scope; anything beyond that is ignored, so a scope
                is not a substitute for a large enough hot window.

            THREAD SAFETY
                Instances of this object are not thread-safe, but the pin counts they
                manipulate are. Two threads may hold scopes on the same tensor.
        !*/

    public:

        explicit device_scope (
            std::initializer_list<const tensor*> tensors
        );
        /*!
            ensures
                - Every block backing one of the given tensors is pinned until this object is
                  destroyed. Null pointers in the list are ignored.
        !*/

        explicit device_scope (
            std::initializer_list<const gpu_data*> blocks
        );
        /*!
            ensures
                - As above, for code holding gpu_data objects rather than tensors.
        !*/

        ~device_scope (
        );
        /*!
            ensures
                - releases the pins taken by the constructor.
        !*/

        device_scope(const device_scope&) = delete;
        device_scope& operator=(const device_scope&) = delete;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_EXTENDED_MEMORY_ABSTRACT_H_
