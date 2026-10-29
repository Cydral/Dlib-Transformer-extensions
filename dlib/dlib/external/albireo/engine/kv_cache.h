// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_ENGINE_KV_CACHE_H_
#define ALBIREO_ENGINE_KV_CACHE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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
                a block given back is reused before any new memory is taken.
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
            // a forward pass never see it relocated by an allocation.
            blocks_.reserve(max_blocks_);
        }

        long block_positions() const { return positions_; }
        size_t block_bytes() const { return block_bytes_; }
        size_t max_blocks() const { return max_blocks_; }

        size_t allocated_blocks() const { std::lock_guard<std::mutex> l(m_); return blocks_.size(); }
        size_t used_blocks() const { std::lock_guard<std::mutex> l(m_); return blocks_.size() - free_.size(); }
        size_t free_capacity_blocks() const
        {
            std::lock_guard<std::mutex> l(m_);
            return free_.size() + (max_blocks_ - blocks_.size());
        }

        // Returns a block index, or -1 when the budget is exhausted.
        int allocate()
        {
            std::lock_guard<std::mutex> l(m_);
            if (!free_.empty())
            {
                const int b = free_.back();
                free_.pop_back();
                return b;
            }
            if (blocks_.size() >= max_blocks_) return -1;
            blocks_.emplace_back(new uint16_t[2 * half_per_kind_]);
            return static_cast<int>(blocks_.size() - 1);
        }

        void release(int b)
        {
            if (b < 0) return;
            std::lock_guard<std::mutex> l(m_);
            free_.push_back(b);
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
        std::vector<int> free_;
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

        // Makes room for positions [0, n), taking blocks from the pool.  Returns false,
        // leaving the sequence unchanged, when the pool cannot provide them.
        bool reserve(kv_pool& pool, long n)
        {
            const long per = pool.block_positions();
            const size_t needed = static_cast<size_t>((n + per - 1) / per);
            std::vector<int> taken;
            while (blocks.size() + taken.size() < needed)
            {
                const int b = pool.allocate();
                if (b < 0)
                {
                    for (int t : taken) pool.release(t);
                    return false;
                }
                taken.push_back(b);
            }
            blocks.insert(blocks.end(), taken.begin(), taken.end());
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
