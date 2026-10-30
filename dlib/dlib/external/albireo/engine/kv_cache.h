// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_KV_CACHE_H_
#define ALBIREO_ENGINE_KV_CACHE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <vector>

namespace albireo
{

    class kv_pool
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The memory that holds the keys and values of every sequence of one model,
                as blocks of a fixed number of positions.  A sequence owns the blocks its
                positions fall in and nothing more, so a short exchange takes a few blocks
                and a long one takes many, and the memory the engine reserves follows the
                text actually held rather than the longest context it could hold.

                Keys and values are stored in half precision.  Inside a block, the rows of
                one layer and one key/value head are contiguous, which is the order in
                which attention reads them.

                Blocks are allocated in groups as sequences grow, up to a budget in bytes;
                a block given back is reused before any new memory is taken, and trim()
                returns the memory of the blocks no sequence holds to the system.

                Several sequences may hold the same block: a conversation that starts with
                the text of a cached one holds the full blocks of that text along with it.
                A block counts the sequences that hold it and returns to the pool when the
                last one gives it back; a sequence copies a shared block before writing in
                it (kv_sequence::reserve).
        !*/
    public:
        kv_pool(long n_layers, long n_kv_heads, long head_dim, long block_positions, size_t budget_bytes)
            : layers_(n_layers), heads_(n_kv_heads), head_dim_(head_dim), positions_(block_positions)
        {
            if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || block_positions <= 0)
                throw std::invalid_argument("kv_pool: invalid geometry");
            // Keys then values, each layers x heads x positions x head_dim halves.
            half_per_kind_ = static_cast<size_t>(layers_) * heads_ * positions_ * head_dim_;
            block_bytes_ = 2 * half_per_kind_ * sizeof(uint16_t);
            max_blocks_ = std::max<size_t>(1, budget_bytes / block_bytes_);
            // The table of blocks never moves, so that the threads reading a block during
            // a forward pass never see it relocated by an allocation, and the lists never
            // grow, so that giving a block back cannot fail.
            blocks_.reserve(max_blocks_);
            refs_.reserve(max_blocks_);
            free_.reserve(max_blocks_);
            vacant_.reserve(max_blocks_);
        }

        long block_positions() const { return positions_; }
        size_t block_bytes() const { return block_bytes_; }
        size_t max_blocks() const { return max_blocks_; }

        size_t allocated_blocks() const
        {
            std::lock_guard<std::mutex> l(m_);
            return blocks_.size() - vacant_.size();
        }
        size_t used_blocks() const
        {
            std::lock_guard<std::mutex> l(m_);
            return blocks_.size() - free_.size() - vacant_.size();
        }
        size_t free_capacity_blocks() const
        {
            std::lock_guard<std::mutex> l(m_);
            return free_.size() + vacant_.size() + (max_blocks_ - blocks_.size());
        }

        // Returns a block index, held once, or -1 when the budget is exhausted; throws
        // std::bad_alloc when the system has no memory left.
        int allocate()
        {
            std::lock_guard<std::mutex> l(m_);
            if (!free_.empty())
            {
                const int b = free_.back();
                free_.pop_back();
                refs_[static_cast<size_t>(b)] = 1;
                return b;
            }
            if (!vacant_.empty())
            {
                const int b = vacant_.back();
                blocks_[static_cast<size_t>(b)].reset(new uint16_t[2 * half_per_kind_]);
                vacant_.pop_back();
                refs_[static_cast<size_t>(b)] = 1;
                return b;
            }
            if (blocks_.size() >= max_blocks_) return -1;
            std::unique_ptr<uint16_t[]> memory(new uint16_t[2 * half_per_kind_]);
            blocks_.push_back(std::move(memory));
            refs_.push_back(1);
            return static_cast<int>(blocks_.size() - 1);
        }

        // Gives a hold on a block back; the block returns to the pool with the last one.
        void release(int b)
        {
            if (b < 0) return;
            std::lock_guard<std::mutex> l(m_);
            int& holders = refs_[static_cast<size_t>(b)];
            if (holders > 0 && --holders == 0) free_.push_back(b);
        }

        // One more sequence holds block b.
        void share(int b)
        {
            std::lock_guard<std::mutex> l(m_);
            ++refs_[static_cast<size_t>(b)];
        }

        // Whether more than one sequence holds block b.
        bool shared(int b) const
        {
            std::lock_guard<std::mutex> l(m_);
            return refs_[static_cast<size_t>(b)] > 1;
        }

        // Frees the memory of the blocks no sequence holds; their slots in the table stay,
        // so that the indices of the other blocks never change.
        void trim()
        {
            std::lock_guard<std::mutex> l(m_);
            for (int b : free_)
            {
                blocks_[static_cast<size_t>(b)].reset();
                vacant_.push_back(b);
            }
            free_.clear();
        }

        // Copies the keys and values of block src into block dst.
        void copy_block(int src, int dst)
        {
            if (src == dst) return;
            const uint16_t* from = blocks_[static_cast<size_t>(src)].get();
            std::copy(from, from + 2 * half_per_kind_, blocks_[static_cast<size_t>(dst)].get());
        }

        // Rows of positions x head_dim halves for one layer and one key/value head.
        uint16_t* keys(int block, long layer, long head)
        {
            return blocks_[static_cast<size_t>(block)].get() + offset(layer, head);
        }
        uint16_t* values(int block, long layer, long head)
        {
            return blocks_[static_cast<size_t>(block)].get() + half_per_kind_ + offset(layer, head);
        }

    private:
        size_t offset(long layer, long head) const
        {
            return (static_cast<size_t>(layer) * heads_ + head) * positions_ * head_dim_;
        }

        long layers_, heads_, head_dim_, positions_;
        size_t half_per_kind_ = 0;
        size_t block_bytes_ = 0;
        size_t max_blocks_ = 0;
        mutable std::mutex m_;
        std::vector<std::unique_ptr<uint16_t[]>> blocks_;
        std::vector<int> refs_;     // the number of sequences holding each block
        std::vector<int> free_;     // blocks with memory, held by no sequence
        std::vector<int> vacant_;   // slots whose memory trim() freed
    };

    struct kv_sequence
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The positions of one sequence held in a kv_pool: the blocks it owns, in
                position order, and how many positions they hold.
        !*/
        std::vector<int> blocks;
        long length = 0;

        /*
            Makes room for positions [0, n), taking blocks from the pool, and copies the
            block where the next position goes when other sequences hold it too, so that
            the positions this sequence writes belong to it alone.  Returns false, leaving
            the sequence unchanged, when the pool or the system cannot provide the blocks.
        */
        bool reserve(kv_pool& pool, long n)
        {
            const long per = pool.block_positions();
            const size_t needed = static_cast<size_t>((n + per - 1) / per);
            const size_t next = static_cast<size_t>(length / per);
            const bool copy_next = next < blocks.size() && next < needed && pool.shared(blocks[next]);
            const size_t extra = (needed > blocks.size() ? needed - blocks.size() : 0) + (copy_next ? 1 : 0);
            if (extra == 0) return true;
            std::vector<int> taken;
            try
            {
                blocks.reserve(blocks.size() + extra);
                taken.reserve(extra);
                while (taken.size() < extra)
                {
                    const int b = pool.allocate();
                    if (b < 0) break;
                    taken.push_back(b);
                }
            }
            catch (const std::bad_alloc&)
            {
                // Out of memory: as a full pool, the caller frees cached sequences or waits.
            }
            if (taken.size() < extra)
            {
                for (int t : taken) pool.release(t);
                return false;
            }
            if (copy_next)
            {
                const int own = taken.back();
                taken.pop_back();
                pool.copy_block(blocks[next], own);
                pool.release(blocks[next]);
                blocks[next] = own;
            }
            blocks.insert(blocks.end(), taken.begin(), taken.end());
            return true;
        }

        /*
            Holds positions [0, n) of another sequence: the full blocks of these positions
            are shared with it, and the last block, partly filled, is copied.  Returns
            false, leaving this sequence empty, when the pool or the system cannot provide
            the copy.
        */
        bool share_prefix(kv_pool& pool, const kv_sequence& from, long n)
        {
            clear(pool);
            n = std::max<long>(0, std::min(n, from.length));
            const long per = pool.block_positions();
            const size_t full = static_cast<size_t>(n / per);
            int copy = -1;
            try
            {
                blocks.reserve(full + 1);
                if (n % per != 0)
                {
                    copy = pool.allocate();
                    if (copy < 0) return false;
                }
            }
            catch (const std::bad_alloc&)
            {
                return false;
            }
            if (copy >= 0) pool.copy_block(from.blocks[full], copy);
            for (size_t b = 0; b < full; ++b)
            {
                pool.share(from.blocks[b]);
                blocks.push_back(from.blocks[b]);
            }
            if (copy >= 0) blocks.push_back(copy);
            length = n;
            return true;
        }

        // Keeps positions [0, n) and gives the blocks beyond them back.
        void truncate(kv_pool& pool, long n)
        {
            if (n < 0) n = 0;
            if (n < length) length = n;
            const long per = pool.block_positions();
            const size_t keep = static_cast<size_t>((n + per - 1) / per);
            while (blocks.size() > keep)
            {
                pool.release(blocks.back());
                blocks.pop_back();
            }
        }

        void clear(kv_pool& pool) { truncate(pool, 0); }
    };

}

#endif // ALBIREO_ENGINE_KV_CACHE_H_
