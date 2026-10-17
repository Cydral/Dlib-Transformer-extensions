// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DEFLATE_H_
#define DLIB_DEFLATE_H_

#include "deflate_abstract.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "../error.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class deflate_error : public error
    {
    public:
        deflate_error(const std::string& message) : error(message) {}
    };

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline const std::uint16_t* deflate_length_base()
        {
            static const std::uint16_t v[29] = {
                3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
            return v;
        }
        inline const std::uint8_t* deflate_length_extra()
        {
            static const std::uint8_t v[29] = {
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
            return v;
        }
        inline const std::uint16_t* deflate_dist_base()
        {
            static const std::uint16_t v[30] = {
                1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
                257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
            return v;
        }
        inline const std::uint8_t* deflate_dist_extra()
        {
            static const std::uint8_t v[30] = {
                0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
            return v;
        }
        inline const std::uint8_t* deflate_code_length_order()
        {
            static const std::uint8_t v[19] = {
                16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
            return v;
        }

        inline std::uint32_t reverse_bits(std::uint32_t code, int length)
        {
            std::uint32_t r = 0;
            for (int i = 0; i < length; ++i)
            {
                r = (r << 1) | (code & 1);
                code >>= 1;
            }
            return r;
        }

    // ------------------------------------------------------------------------------------

        /*
            Canonical Huffman decoding table. Codes of up to fast_bits bits are resolved
            with a single lookup on the next input bits (stored LSB first, hence the bit
            reversal when filling the table); longer codes fall back to a comparison
            against the canonical code limits of each length.
        */
        class inflate_table
        {
        public:
            static const int fast_bits = 9;

            bool build(const std::uint8_t* lengths, int count)
            {
                int length_count[17] = {};
                for (int i = 0; i < count; ++i)
                {
                    if (lengths[i] > 15) return false;
                    ++length_count[lengths[i]];
                }
                length_count[0] = 0;
                std::memset(fast_, 0, sizeof(fast_));
                std::memset(sizes_, 0, sizeof(sizes_));
                std::memset(values_, 0, sizeof(values_));
                count_ = count;

                int next_code[16] = {};
                int code = 0, k = 0;
                for (int i = 1; i < 16; ++i)
                {
                    next_code[i] = code;
                    first_code_[i] = static_cast<std::uint16_t>(code);
                    first_symbol_[i] = static_cast<std::uint16_t>(k);
                    code += length_count[i];
                    if (length_count[i] != 0 && code - 1 >= (1 << i))
                        return false;
                    max_code_[i] = static_cast<std::uint32_t>(code) << (16 - i);
                    code <<= 1;
                    k += length_count[i];
                }
                max_code_[16] = 0x10000;

                for (int i = 0; i < count; ++i)
                {
                    const int s = lengths[i];
                    if (s == 0) continue;
                    const int c = next_code[s] - first_code_[s] + first_symbol_[s];
                    sizes_[c] = static_cast<std::uint8_t>(s);
                    values_[c] = static_cast<std::uint16_t>(i);
                    if (s <= fast_bits)
                    {
                        std::uint32_t j = reverse_bits(static_cast<std::uint32_t>(next_code[s]), s);
                        while (j < (1u << fast_bits))
                        {
                            fast_[j] = static_cast<std::uint16_t>((s << 9) | i);
                            j += 1u << s;
                        }
                    }
                    ++next_code[s];
                }
                return true;
            }

            // Returns the symbol, or -1 when no code matches. Consumes the code bits.
            template <typename reader_type>
            int decode(reader_type& in) const
            {
                in.fill();
                const std::uint32_t peek = static_cast<std::uint32_t>(in.peek(16));
                const std::uint16_t f = fast_[peek & ((1u << fast_bits) - 1)];
                if (f != 0)
                {
                    in.drop(f >> 9);
                    return f & 0x1FF;
                }
                const std::uint32_t k = reverse_bits(peek, 16);
                int s = fast_bits + 1;
                while (s < 16 && k >= max_code_[s])
                    ++s;
                if (s >= 16)
                    return -1;
                const int idx = static_cast<int>(k >> (16 - s)) - first_code_[s] + first_symbol_[s];
                if (idx < 0 || idx >= count_ || sizes_[idx] != s)
                    return -1;
                in.drop(s);
                return values_[idx];
            }

        private:
            std::uint16_t fast_[1 << fast_bits];
            std::uint16_t first_code_[17];
            std::uint16_t first_symbol_[17];
            std::uint32_t max_code_[17];
            std::uint8_t sizes_[288];
            std::uint16_t values_[288];
            int count_ = 0;
        };

        class inflate_bit_reader
        {
        public:
            inflate_bit_reader(const unsigned char* data, std::size_t size)
                : data_(data), size_(size), pos_(0), bits_(0), count_(0) {}

            void fill()
            {
                while (count_ <= 56)
                {
                    const std::uint64_t byte = pos_ < size_ ? data_[pos_] : 0;
                    ++pos_;
                    bits_ |= byte << count_;
                    count_ += 8;
                }
            }

            std::uint64_t peek(int n) const { return bits_ & ((std::uint64_t(1) << n) - 1); }

            void drop(int n)
            {
                bits_ >>= n;
                count_ -= n;
            }

            std::uint32_t get(int n)
            {
                if (n == 0) return 0;
                fill();
                const std::uint32_t v = static_cast<std::uint32_t>(peek(n));
                drop(n);
                return v;
            }

            void align_to_byte() { drop(count_ % 8); }

            // Number of input bytes consumed so far, partially used byte included.
            std::size_t consumed() const
            {
                const std::size_t bits_used = pos_ * 8 - static_cast<std::size_t>(count_);
                return (bits_used + 7) / 8;
            }

            bool overrun() const { return pos_ * 8 - static_cast<std::size_t>(count_) > size_ * 8; }

            // Copies raw bytes after the bit buffer has been aligned, as stored blocks require.
            bool copy_bytes(std::size_t n, std::string& out)
            {
                const std::size_t start = consumed();
                if (start + n > size_)
                    return false;
                out.append(reinterpret_cast<const char*>(data_ + start), n);
                pos_ = start + n;
                bits_ = 0;
                count_ = 0;
                return true;
            }

        private:
            const unsigned char* data_;
            std::size_t size_;
            std::size_t pos_;
            std::uint64_t bits_;
            int count_;
        };

    // ------------------------------------------------------------------------------------

        class deflate_bit_writer
        {
        public:
            explicit deflate_bit_writer(std::string& out) : out_(out), bits_(0), count_(0) {}

            void put(std::uint32_t value, int n)
            {
                bits_ |= static_cast<std::uint64_t>(value) << count_;
                count_ += n;
                while (count_ >= 8)
                {
                    out_.push_back(static_cast<char>(bits_ & 0xFF));
                    bits_ >>= 8;
                    count_ -= 8;
                }
            }

            void align()
            {
                if (count_ > 0)
                    put(0, 8 - count_);
            }

            void flush() { align(); }

            std::string& out() { return out_; }

        private:
            std::string& out_;
            std::uint64_t bits_;
            int count_;
        };

        /*
            Huffman code lengths limited to max_bits. The unrestricted lengths come from
            the usual merge of the two lightest nodes; when the tree is deeper than
            allowed, the overflowing leaves are moved to max_bits and the Kraft sum is
            restored by lengthening the deepest admissible leaves, the most frequent
            symbols keeping the shortest codes.
        */
        inline void build_code_lengths(
            std::vector<std::uint32_t> freq,
            int max_bits,
            std::vector<std::uint8_t>& lengths
        )
        {
            const int n = static_cast<int>(freq.size());
            lengths.assign(n, 0);

            int used = 0;
            for (int i = 0; i < n; ++i)
                if (freq[i] != 0) ++used;
            for (int i = 0; used < 2 && i < n; ++i)
            {
                if (freq[i] == 0)
                {
                    freq[i] = 1;
                    ++used;
                }
            }

            std::vector<int> symbols;
            for (int i = 0; i < n; ++i)
                if (freq[i] != 0) symbols.push_back(i);
            std::stable_sort(symbols.begin(), symbols.end(),
                [&](int a, int b) { return freq[a] < freq[b]; });

            const int m = static_cast<int>(symbols.size());
            std::vector<std::uint64_t> weight(2 * m);
            std::vector<int> parent(2 * m, -1);
            for (int i = 0; i < m; ++i)
                weight[i] = freq[symbols[i]];

            /* Two-queue construction: leaves are already sorted and internal nodes are
               created in nondecreasing weight order. */
            int leaf = 0, node = m, next_internal = m;
            auto pick = [&]() {
                if (leaf < m && (node >= next_internal || weight[leaf] <= weight[node]))
                    return leaf++;
                return node++;
            };
            for (int k = 0; k < m - 1; ++k)
            {
                const int a = pick();
                const int b = pick();
                weight[next_internal] = weight[a] + weight[b];
                parent[a] = next_internal;
                parent[b] = next_internal;
                ++next_internal;
            }

            std::vector<int> depth(2 * m, 0);
            for (int i = next_internal - 2; i >= 0; --i)
                depth[i] = depth[parent[i]] + 1;

            std::vector<int> count(std::max(max_bits, 64) + 1, 0);
            for (int i = 0; i < m; ++i)
                ++count[std::min(depth[i], 64)];
            for (int i = max_bits + 1; i < static_cast<int>(count.size()); ++i)
            {
                count[max_bits] += count[i];
                count[i] = 0;
            }
            std::uint64_t total = 0;
            for (int i = max_bits; i > 0; --i)
                total += static_cast<std::uint64_t>(count[i]) << (max_bits - i);
            while (total != (std::uint64_t(1) << max_bits))
            {
                --count[max_bits];
                for (int i = max_bits - 1; i > 0; --i)
                {
                    if (count[i] != 0)
                    {
                        --count[i];
                        count[i + 1] += 2;
                        break;
                    }
                }
                --total;
            }

            // symbols are sorted by increasing frequency: the first ones get the longest codes.
            int idx = 0;
            for (int len = max_bits; len > 0; --len)
                for (int c = 0; c < count[len]; ++c)
                    lengths[symbols[idx++]] = static_cast<std::uint8_t>(len);
        }

        inline void canonical_codes(
            const std::vector<std::uint8_t>& lengths,
            std::vector<std::uint16_t>& codes
        )
        {
            int bl_count[16] = {};
            for (std::uint8_t l : lengths)
                ++bl_count[l];
            bl_count[0] = 0;
            int next_code[16] = {};
            int code = 0;
            for (int bits = 1; bits < 16; ++bits)
            {
                code = (code + bl_count[bits - 1]) << 1;
                next_code[bits] = code;
            }
            codes.assign(lengths.size(), 0);
            for (std::size_t i = 0; i < lengths.size(); ++i)
            {
                const int l = lengths[i];
                if (l != 0)
                    codes[i] = static_cast<std::uint16_t>(reverse_bits(static_cast<std::uint32_t>(next_code[l]++), l));
            }
        }

        struct lz_token
        {
            std::uint16_t length;   // literal byte when distance == 0
            std::uint16_t distance;
        };

        inline int length_symbol(int length)
        {
            const std::uint16_t* base = deflate_length_base();
            int i = 28;
            while (base[i] > length) --i;
            return i;
        }

        inline int distance_symbol(int distance)
        {
            const std::uint16_t* base = deflate_dist_base();
            int i = 29;
            while (base[i] > distance) --i;
            return i;
        }

        class deflate_encoder
        {
        public:
            deflate_encoder(const std::string& data, int level) : data_(data), level_(level)
            {
                static const int chains[10] = { 0, 8, 16, 32, 32, 64, 128, 256, 1024, 4096 };
                max_chain_ = chains[level];
                lazy_ = level >= 4;
                good_length_ = level >= 8 ? 258 : 32;
            }

            std::string run()
            {
                std::string out;
                deflate_bit_writer bw(out);
                const std::size_t n = data_.size();
                if (level_ == 0 || n == 0)
                {
                    write_stored(bw, 0, n, true);
                    bw.flush();
                    return out;
                }

                head_.assign(hash_size, -1);
                prev_.assign(window_size, -1);
                std::vector<lz_token> tokens;
                tokens.reserve(block_tokens + 2);
                std::size_t block_start = 0;
                std::size_t p = 0;
                bool cached = false;
                int cached_len = 0, cached_dist = 0;

                while (p < n)
                {
                    int len = 0, dist = 0;
                    if (cached)
                    {
                        len = cached_len;
                        dist = cached_dist;
                        cached = false;
                    }
                    else
                    {
                        find_match(p, len, dist);
                    }

                    if (len >= 3)
                    {
                        if (lazy_ && len < good_length_ && p + 1 < n)
                        {
                            insert(p);
                            int len2 = 0, dist2 = 0;
                            find_match(p + 1, len2, dist2);
                            if (len2 > len)
                            {
                                tokens.push_back(
                                    { static_cast<std::uint16_t>(static_cast<unsigned char>(data_[p])), 0 });
                                ++p;
                                cached = true;
                                cached_len = len2;
                                cached_dist = dist2;
                                maybe_flush(bw, tokens, block_start, p);
                                continue;
                            }
                        }
                        else
                        {
                            insert(p);
                        }
                        tokens.push_back({ static_cast<std::uint16_t>(len), static_cast<std::uint16_t>(dist) });
                        for (std::size_t q = p + 1; q < p + static_cast<std::size_t>(len); ++q)
                            insert(q);
                        p += static_cast<std::size_t>(len);
                    }
                    else
                    {
                        insert(p);
                        tokens.push_back({ static_cast<std::uint16_t>(static_cast<unsigned char>(data_[p])), 0 });
                        ++p;
                    }
                    if (!cached)
                        maybe_flush(bw, tokens, block_start, p);
                }
                write_block(bw, tokens, block_start, n, true);
                bw.flush();
                return out;
            }

        private:
            static const int window_size = 32768;
            static const int hash_bits = 15;
            static const int hash_size = 1 << hash_bits;
            static const std::size_t block_tokens = 32768;

            int hash_at(std::size_t p) const
            {
                const unsigned char* d = reinterpret_cast<const unsigned char*>(data_.data()) + p;
                const std::uint32_t h = (static_cast<std::uint32_t>(d[0]) << 16) ^
                                        (static_cast<std::uint32_t>(d[1]) << 8) ^ d[2];
                return static_cast<int>((h * 2654435761u) >> (32 - hash_bits));
            }

            void insert(std::size_t p)
            {
                if (p + 2 >= data_.size()) return;
                const int h = hash_at(p);
                prev_[p & (window_size - 1)] = head_[h];
                head_[h] = static_cast<std::int64_t>(p);
            }

            void find_match(std::size_t p, int& best_len, int& best_dist) const
            {
                best_len = 0;
                best_dist = 0;
                const std::size_t n = data_.size();
                if (p + 2 >= n) return;
                const int max_len = static_cast<int>(std::min<std::size_t>(258, n - p));
                const unsigned char* d = reinterpret_cast<const unsigned char*>(data_.data());
                std::int64_t cand = head_[hash_at(p)];
                int chain = max_chain_;
                const std::int64_t limit = static_cast<std::int64_t>(p) - window_size;
                while (cand >= 0 && cand > limit && chain-- > 0)
                {
                    const std::size_t c = static_cast<std::size_t>(cand);
                    if (d[c + best_len] == d[p + best_len] && d[c] == d[p])
                    {
                        int l = 0;
                        while (l < max_len && d[c + l] == d[p + l])
                            ++l;
                        if (l > best_len)
                        {
                            best_len = l;
                            best_dist = static_cast<int>(p - c);
                            if (l >= max_len) break;
                        }
                    }
                    const std::int64_t next = prev_[c & (window_size - 1)];
                    if (next >= cand) break;
                    cand = next;
                }
                if (best_len < 3)
                {
                    best_len = 0;
                    best_dist = 0;
                }
            }

            void maybe_flush(deflate_bit_writer& bw, std::vector<lz_token>& tokens, std::size_t& block_start,
                             std::size_t p)
            {
                if (tokens.size() >= block_tokens)
                {
                    write_block(bw, tokens, block_start, p, false);
                    tokens.clear();
                    block_start = p;
                }
            }

            void write_stored(deflate_bit_writer& bw, std::size_t begin, std::size_t end, bool final_block)
            {
                std::size_t pos = begin;
                do
                {
                    const std::size_t len = std::min<std::size_t>(65535, end - pos);
                    const bool last = final_block && pos + len == end;
                    bw.put(last ? 1 : 0, 1);
                    bw.put(0, 2);
                    bw.align();
                    bw.put(static_cast<std::uint32_t>(len), 16);
                    bw.put(static_cast<std::uint32_t>(~len & 0xFFFF), 16);
                    bw.out().append(data_, pos, len);
                    pos += len;
                } while (pos < end);
            }

            void write_block(
                deflate_bit_writer& bw,
                const std::vector<lz_token>& tokens,
                std::size_t begin,
                std::size_t end,
                bool final_block
            )
            {
                const std::uint8_t* lext = deflate_length_extra();
                const std::uint8_t* dext = deflate_dist_extra();

                std::vector<std::uint32_t> lit_freq(286, 0), dist_freq(30, 0);
                for (const lz_token& t : tokens)
                {
                    if (t.distance == 0)
                    {
                        ++lit_freq[t.length];
                    }
                    else
                    {
                        ++lit_freq[257 + length_symbol(t.length)];
                        ++dist_freq[distance_symbol(t.distance)];
                    }
                }
                lit_freq[256] = 1;

                std::vector<std::uint8_t> lit_len, dist_len;
                build_code_lengths(lit_freq, 15, lit_len);
                build_code_lengths(dist_freq, 15, dist_len);

                int num_lit = 286;
                while (num_lit > 257 && lit_len[num_lit - 1] == 0) --num_lit;
                int num_dist = 30;
                while (num_dist > 1 && dist_len[num_dist - 1] == 0) --num_dist;

                // Run-length encode the concatenated code lengths with symbols 16, 17 and 18.
                std::vector<std::uint8_t> all(lit_len.begin(), lit_len.begin() + num_lit);
                all.insert(all.end(), dist_len.begin(), dist_len.begin() + num_dist);
                std::vector<std::uint8_t> rle_sym, rle_extra;
                for (std::size_t i = 0; i < all.size();)
                {
                    const std::uint8_t cur = all[i];
                    std::size_t run = 1;
                    while (i + run < all.size() && all[i + run] == cur) ++run;
                    if (cur == 0)
                    {
                        std::size_t left = run;
                        while (left >= 11)
                        {
                            const std::size_t r = std::min<std::size_t>(left, 138);
                            rle_sym.push_back(18); rle_extra.push_back(static_cast<std::uint8_t>(r - 11));
                            left -= r;
                        }
                        if (left >= 3)
                        {
                            rle_sym.push_back(17); rle_extra.push_back(static_cast<std::uint8_t>(left - 3));
                            left = 0;
                        }
                        while (left-- > 0)
                        {
                            rle_sym.push_back(0); rle_extra.push_back(0);
                        }
                    }
                    else
                    {
                        rle_sym.push_back(cur); rle_extra.push_back(0);
                        std::size_t left = run - 1;
                        while (left >= 3)
                        {
                            const std::size_t r = std::min<std::size_t>(left, 6);
                            rle_sym.push_back(16); rle_extra.push_back(static_cast<std::uint8_t>(r - 3));
                            left -= r;
                        }
                        while (left-- > 0)
                        {
                            rle_sym.push_back(cur); rle_extra.push_back(0);
                        }
                    }
                    i += run;
                }

                std::vector<std::uint32_t> cl_freq(19, 0);
                for (std::uint8_t s : rle_sym) ++cl_freq[s];
                std::vector<std::uint8_t> cl_len;
                build_code_lengths(cl_freq, 7, cl_len);
                const std::uint8_t* order = deflate_code_length_order();
                int num_cl = 19;
                while (num_cl > 4 && cl_len[order[num_cl - 1]] == 0) --num_cl;

                // Cost of the three encodings, in bits, excluding the 3 bit block header.
                std::uint64_t dyn_bits = 5 + 5 + 4 + 3 * static_cast<std::uint64_t>(num_cl);
                for (std::size_t i = 0; i < rle_sym.size(); ++i)
                {
                    dyn_bits += cl_len[rle_sym[i]];
                    if (rle_sym[i] == 16) dyn_bits += 2;
                    else if (rle_sym[i] == 17) dyn_bits += 3;
                    else if (rle_sym[i] == 18) dyn_bits += 7;
                }
                std::uint64_t fixed_bits = 0;
                for (int s = 0; s < 286; ++s)
                {
                    if (lit_freq[s] == 0) continue;
                    const int fl = s < 144 ? 8 : (s < 256 ? 9 : (s < 280 ? 7 : 8));
                    std::uint64_t extra = 0;
                    if (s >= 257) extra = lext[s - 257];
                    dyn_bits += static_cast<std::uint64_t>(lit_freq[s]) * (lit_len[s] + extra);
                    fixed_bits += static_cast<std::uint64_t>(lit_freq[s]) * (fl + extra);
                }
                for (int s = 0; s < 30; ++s)
                {
                    if (dist_freq[s] == 0) continue;
                    dyn_bits += static_cast<std::uint64_t>(dist_freq[s]) * (dist_len[s] + dext[s]);
                    fixed_bits += static_cast<std::uint64_t>(dist_freq[s]) * (5 + dext[s]);
                }
                const std::size_t raw = end - begin;
                const std::uint64_t stored_bits = (static_cast<std::uint64_t>(raw) + 5 * (raw / 65535 + 1)) * 8 + 7;

                if (stored_bits <= dyn_bits && stored_bits <= fixed_bits)
                {
                    write_stored(bw, begin, end, final_block);
                    return;
                }

                std::vector<std::uint8_t> use_lit, use_dist;
                if (fixed_bits <= dyn_bits)
                {
                    use_lit.assign(288, 8);
                    for (int s = 144; s < 256; ++s) use_lit[s] = 9;
                    for (int s = 256; s < 280; ++s) use_lit[s] = 7;
                    use_dist.assign(30, 5);
                    bw.put(final_block ? 1 : 0, 1);
                    bw.put(1, 2);
                }
                else
                {
                    use_lit = lit_len;
                    use_dist = dist_len;
                    bw.put(final_block ? 1 : 0, 1);
                    bw.put(2, 2);
                    bw.put(static_cast<std::uint32_t>(num_lit - 257), 5);
                    bw.put(static_cast<std::uint32_t>(num_dist - 1), 5);
                    bw.put(static_cast<std::uint32_t>(num_cl - 4), 4);
                    for (int i = 0; i < num_cl; ++i)
                        bw.put(cl_len[order[i]], 3);
                    std::vector<std::uint16_t> cl_codes;
                    canonical_codes(cl_len, cl_codes);
                    for (std::size_t i = 0; i < rle_sym.size(); ++i)
                    {
                        const std::uint8_t s = rle_sym[i];
                        bw.put(cl_codes[s], cl_len[s]);
                        if (s == 16) bw.put(rle_extra[i], 2);
                        else if (s == 17) bw.put(rle_extra[i], 3);
                        else if (s == 18) bw.put(rle_extra[i], 7);
                    }
                }

                std::vector<std::uint16_t> lit_codes, dist_codes;
                canonical_codes(use_lit, lit_codes);
                canonical_codes(use_dist, dist_codes);
                const std::uint16_t* lbase = deflate_length_base();
                const std::uint16_t* dbase = deflate_dist_base();
                for (const lz_token& t : tokens)
                {
                    if (t.distance == 0)
                    {
                        bw.put(lit_codes[t.length], use_lit[t.length]);
                    }
                    else
                    {
                        const int ls = length_symbol(t.length);
                        bw.put(lit_codes[257 + ls], use_lit[257 + ls]);
                        if (lext[ls]) bw.put(static_cast<std::uint32_t>(t.length - lbase[ls]), lext[ls]);
                        const int ds = distance_symbol(t.distance);
                        bw.put(dist_codes[ds], use_dist[ds]);
                        if (dext[ds]) bw.put(static_cast<std::uint32_t>(t.distance - dbase[ds]), dext[ds]);
                    }
                }
                bw.put(lit_codes[256], use_lit[256]);
            }

            const std::string& data_;
            int level_;
            int max_chain_;
            bool lazy_;
            int good_length_;
            std::vector<std::int64_t> head_;
            std::vector<std::int64_t> prev_;
        };
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        /* Decodes data into out. When deflate_error is thrown, out holds what was decoded
           before the error. */
        inline void inflate_into(
            const std::string& data,
            std::size_t max_output_size,
            std::size_t* consumed,
            std::string& out
        )
        {
            inflate_bit_reader in(reinterpret_cast<const unsigned char*>(data.data()), data.size());
            out.clear();
            out.reserve(std::min<std::size_t>(max_output_size, data.size() * 4 + 64));
            inflate_table lit, dist;
            const std::uint16_t* lbase = deflate_length_base();
            const std::uint8_t* lext = deflate_length_extra();
            const std::uint16_t* dbase = deflate_dist_base();
            const std::uint8_t* dext = deflate_dist_extra();

            bool final_block = false;
            while (!final_block)
            {
                final_block = in.get(1) != 0;
                const std::uint32_t type = in.get(2);
                if (type == 0)
                {
                    in.align_to_byte();
                    const std::uint32_t len = in.get(16);
                    const std::uint32_t nlen = in.get(16);
                    if ((len ^ 0xFFFF) != nlen)
                        throw deflate_error("inflate_raw(): corrupt stored block length");
                    if (out.size() + len > max_output_size)
                        throw deflate_error("inflate_raw(): output exceeds the size limit");
                    if (!in.copy_bytes(len, out))
                        throw deflate_error("inflate_raw(): truncated stored block");
                    continue;
                }
                if (type == 1)
                {
                    std::uint8_t l[288], d[30];
                    for (int i = 0; i < 144; ++i) l[i] = 8;
                    for (int i = 144; i < 256; ++i) l[i] = 9;
                    for (int i = 256; i < 280; ++i) l[i] = 7;
                    for (int i = 280; i < 288; ++i) l[i] = 8;
                    for (int i = 0; i < 30; ++i) d[i] = 5;
                    lit.build(l, 288);
                    dist.build(d, 30);
                }
                else if (type == 2)
                {
                    const int hlit = static_cast<int>(in.get(5)) + 257;
                    const int hdist = static_cast<int>(in.get(5)) + 1;
                    const int hclen = static_cast<int>(in.get(4)) + 4;
                    if (hlit > 286 || hdist > 30)
                        throw deflate_error("inflate_raw(): invalid dynamic block header");
                    std::uint8_t cl[19] = {};
                    const std::uint8_t* order = deflate_code_length_order();
                    for (int i = 0; i < hclen; ++i)
                        cl[order[i]] = static_cast<std::uint8_t>(in.get(3));
                    inflate_table cl_table;
                    if (!cl_table.build(cl, 19))
                        throw deflate_error("inflate_raw(): invalid code length code");
                    std::uint8_t lengths[286 + 30] = {};
                    int i = 0;
                    while (i < hlit + hdist)
                    {
                        const int sym = cl_table.decode(in);
                        if (sym < 0)
                            throw deflate_error("inflate_raw(): invalid code length symbol");
                        if (sym < 16)
                        {
                            lengths[i++] = static_cast<std::uint8_t>(sym);
                        }
                        else
                        {
                            int rep = 0;
                            std::uint8_t val = 0;
                            if (sym == 16)
                            {
                                if (i == 0) throw deflate_error("inflate_raw(): repeat without a previous length");
                                val = lengths[i - 1];
                                rep = 3 + static_cast<int>(in.get(2));
                            }
                            else if (sym == 17)
                            {
                                rep = 3 + static_cast<int>(in.get(3));
                            }
                            else
                            {
                                rep = 11 + static_cast<int>(in.get(7));
                            }
                            if (i + rep > hlit + hdist)
                                throw deflate_error("inflate_raw(): code lengths overflow");
                            while (rep-- > 0)
                                lengths[i++] = val;
                        }
                    }
                    if (lengths[256] == 0)
                        throw deflate_error("inflate_raw(): missing end-of-block code");
                    if (!lit.build(lengths, hlit) || !dist.build(lengths + hlit, hdist))
                        throw deflate_error("inflate_raw(): over-subscribed Huffman code");
                }
                else
                {
                    throw deflate_error("inflate_raw(): invalid block type");
                }

                for (;;)
                {
                    const int sym = lit.decode(in);
                    if (sym < 0)
                        throw deflate_error("inflate_raw(): invalid literal/length code");
                    // Past the end the reader supplies zero bits, which a code may decode.
                    if (in.overrun())
                        throw deflate_error("inflate_raw(): truncated stream");
                    if (sym < 256)
                    {
                        if (out.size() >= max_output_size)
                            throw deflate_error("inflate_raw(): output exceeds the size limit");
                        out.push_back(static_cast<char>(sym));
                        continue;
                    }
                    if (sym == 256)
                        break;
                    const int ls = sym - 257;
                    if (ls >= 29)
                        throw deflate_error("inflate_raw(): invalid length symbol");
                    const std::size_t length = lbase[ls] + in.get(lext[ls]);
                    const int ds = dist.decode(in);
                    if (ds < 0 || ds >= 30)
                        throw deflate_error("inflate_raw(): invalid distance code");
                    const std::size_t distance = dbase[ds] + in.get(dext[ds]);
                    if (in.overrun())
                        throw deflate_error("inflate_raw(): truncated stream");
                    if (distance > out.size())
                        throw deflate_error("inflate_raw(): distance beyond the start of the output");
                    if (out.size() + length > max_output_size)
                        throw deflate_error("inflate_raw(): output exceeds the size limit");
                    std::size_t from = out.size() - distance;
                    for (std::size_t k = 0; k < length; ++k)
                        out.push_back(out[from + k]);
                }
                if (in.overrun())
                    throw deflate_error("inflate_raw(): truncated stream");
            }
            if (in.overrun())
                throw deflate_error("inflate_raw(): truncated stream");
            if (consumed)
                *consumed = in.consumed();
        }
    }

    inline std::string inflate_raw(
        const std::string& data,
        std::size_t max_output_size = std::size_t(1) << 30,
        std::size_t* consumed = nullptr
    )
    {
        std::string out;
        impl::inflate_into(data, max_output_size, consumed, out);
        return out;
    }

    inline std::string inflate_raw_prefix(
        const std::string& data,
        std::size_t max_output_size,
        bool& complete
    )
    {
        std::string out;
        complete = false;
        try
        {
            impl::inflate_into(data, max_output_size, nullptr, out);
            complete = true;
        }
        catch (deflate_error&)
        {
        }
        return out;
    }

    inline std::string deflate_raw(
        const std::string& data,
        int level = 6
    )
    {
        if (level < 0) level = 0;
        if (level > 9) level = 9;
        impl::deflate_encoder enc(data, level);
        return enc.run();
    }

// ----------------------------------------------------------------------------------------

    inline std::uint32_t adler32(
        const void* data,
        std::size_t size,
        std::uint32_t initial = 1
    )
    {
        const unsigned char* p = static_cast<const unsigned char*>(data);
        std::uint32_t a = initial & 0xFFFF, b = initial >> 16;
        while (size > 0)
        {
            std::size_t chunk = size < 5552 ? size : 5552;
            size -= chunk;
            while (chunk--)
            {
                a += *p++;
                b += a;
            }
            a %= 65521;
            b %= 65521;
        }
        return (b << 16) | a;
    }

    inline std::string zlib_compress(
        const std::string& data,
        int level = 6
    )
    {
        std::string out;
        out.push_back(static_cast<char>(0x78));
        out.push_back(static_cast<char>(0x9C));
        out += deflate_raw(data, level);
        const std::uint32_t a = adler32(data.data(), data.size());
        for (int i = 3; i >= 0; --i)
            out.push_back(static_cast<char>((a >> (8 * i)) & 0xFF));
        return out;
    }

    inline std::string zlib_decompress(
        const std::string& data,
        std::size_t max_output_size = std::size_t(1) << 30
    )
    {
        if (data.size() < 2)
            throw deflate_error("zlib_decompress(): stream too short");
        const unsigned char cmf = static_cast<unsigned char>(data[0]);
        const unsigned char flg = static_cast<unsigned char>(data[1]);
        if ((cmf & 0x0F) != 8 || (cmf >> 4) > 7 || ((cmf << 8) | flg) % 31 != 0)
            throw deflate_error("zlib_decompress(): invalid zlib header");
        if (flg & 0x20)
            throw deflate_error("zlib_decompress(): preset dictionaries are not supported");
        std::size_t consumed = 0;
        std::string out = inflate_raw(data.substr(2), max_output_size, &consumed);
        const std::size_t trailer = 2 + consumed;
        if (trailer + 4 > data.size())
            throw deflate_error("zlib_decompress(): missing Adler-32 checksum");
        std::uint32_t expected = 0;
        for (std::size_t i = 0; i < 4; ++i)
            expected = (expected << 8) | static_cast<unsigned char>(data[trailer + i]);
        if (expected != adler32(out.data(), out.size()))
            throw deflate_error("zlib_decompress(): Adler-32 mismatch");
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DEFLATE_H_
