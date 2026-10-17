// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FF1_H_
#define DLIB_FF1_H_

#include "ff1_abstract.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "aes.h"
#include "crypto_utils.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        /*
            Minimal unsigned integer used by FF1: little-endian 32 bit limbs, with only the
            operations the standard needs (multiply-add and division by a small value).
            Reducing modulo radix^m is done by extracting m base-radix digits, so no
            general long division is ever required.
        */
        class ff1_bignum
        {
        public:
            bool is_zero() const
            {
                for (std::uint32_t l : limbs_)
                    if (l != 0) return false;
                return true;
            }

            void mul_add_small(std::uint32_t multiplier, std::uint32_t addend)
            {
                std::uint64_t carry = addend;
                for (std::uint32_t& l : limbs_)
                {
                    const std::uint64_t t = static_cast<std::uint64_t>(l) * multiplier + carry;
                    l = static_cast<std::uint32_t>(t);
                    carry = t >> 32;
                }
                if (carry != 0)
                    limbs_.push_back(static_cast<std::uint32_t>(carry));
            }

            std::uint32_t divmod_small(std::uint32_t divisor)
            {
                std::uint64_t rem = 0;
                for (std::size_t i = limbs_.size(); i-- > 0;)
                {
                    const std::uint64_t cur = (rem << 32) | limbs_[i];
                    limbs_[i] = static_cast<std::uint32_t>(cur / divisor);
                    rem = cur % divisor;
                }
                trim();
                return static_cast<std::uint32_t>(rem);
            }

            std::size_t bit_length() const
            {
                for (std::size_t i = limbs_.size(); i-- > 0;)
                {
                    if (limbs_[i] != 0)
                    {
                        std::size_t bits = 0;
                        std::uint32_t top = limbs_[i];
                        while (top != 0) { ++bits; top >>= 1; }
                        return 32 * i + bits;
                    }
                }
                return 0;
            }

            void decrement()
            {
                for (std::uint32_t& l : limbs_)
                {
                    if (l != 0) { --l; break; }
                    l = 0xFFFFFFFFu;
                }
                trim();
            }

            static ff1_bignum from_bytes(const unsigned char* data, std::size_t size)
            {
                ff1_bignum r;
                for (std::size_t i = 0; i < size; ++i)
                    r.mul_add_small(256, data[i]);
                return r;
            }

            static ff1_bignum from_numerals(const std::uint32_t* digits, std::size_t count, std::uint32_t radix)
            {
                ff1_bignum r;
                for (std::size_t i = 0; i < count; ++i)
                    r.mul_add_small(radix, digits[i]);
                return r;
            }

            bool to_bytes(std::size_t length, unsigned char* out) const
            {
                std::memset(out, 0, length);
                for (std::size_t i = 0; i < limbs_.size(); ++i)
                {
                    for (int k = 0; k < 4; ++k)
                    {
                        const unsigned char byte = static_cast<unsigned char>(limbs_[i] >> (8 * k));
                        const std::size_t pos = 4 * i + k;
                        if (pos < length)
                            out[length - 1 - pos] = byte;
                        else if (byte != 0)
                            return false;
                    }
                }
                return true;
            }

            ~ff1_bignum()
            {
                if (!limbs_.empty())
                    secure_wipe(limbs_.data(), limbs_.size() * sizeof(std::uint32_t));
            }

        private:
            void trim()
            {
                while (!limbs_.empty() && limbs_.back() == 0)
                    limbs_.pop_back();
            }

            std::vector<std::uint32_t> limbs_;
        };

        inline bool ff1_domain_is_large_enough(std::uint32_t radix, std::size_t n)
        {
            if (n < 2)
                return false;
            std::uint64_t p = 1;
            for (std::size_t i = 0; i < n; ++i)
            {
                p *= radix;
                if (p >= 1000000u)
                    return true;
            }
            return false;
        }
    }

// ----------------------------------------------------------------------------------------

    class ff1_cipher
    {
    public:
        ff1_cipher(
            const void* key,
            std::size_t key_size,
            std::uint32_t radix
        ) : aes_(key, key_size), radix_(radix)
        {
            if (radix < 2 || radix > 65536)
                throw crypto_error("ff1_cipher: the radix must lie in [2, 65536]");
        }

        std::uint32_t get_radix() const { return radix_; }

        std::size_t min_length() const
        {
            std::size_t n = 2;
            while (!impl::ff1_domain_is_large_enough(radix_, n))
                ++n;
            return n;
        }

        std::vector<std::uint32_t> encrypt(
            const std::vector<std::uint32_t>& numerals,
            const std::string& tweak
        ) const
        {
            return process(numerals, tweak, true);
        }

        std::vector<std::uint32_t> decrypt(
            const std::vector<std::uint32_t>& numerals,
            const std::string& tweak
        ) const
        {
            return process(numerals, tweak, false);
        }

        std::string encrypt(
            const std::string& text,
            const std::string& alphabet,
            const std::string& tweak
        ) const
        {
            return map_back(encrypt(map_in(text, alphabet), tweak), alphabet);
        }

        std::string decrypt(
            const std::string& text,
            const std::string& alphabet,
            const std::string& tweak
        ) const
        {
            return map_back(decrypt(map_in(text, alphabet), tweak), alphabet);
        }

    private:
        std::vector<std::uint32_t> map_in(const std::string& text, const std::string& alphabet) const
        {
            if (alphabet.size() != radix_)
                throw crypto_error("ff1_cipher: the alphabet size must equal the radix");
            std::vector<std::uint32_t> x;
            x.reserve(text.size());
            for (char ch : text)
            {
                const std::size_t pos = alphabet.find(ch);
                if (pos == std::string::npos)
                    throw crypto_error("ff1_cipher: character outside the alphabet");
                x.push_back(static_cast<std::uint32_t>(pos));
            }
            return x;
        }

        static std::string map_back(const std::vector<std::uint32_t>& x, const std::string& alphabet)
        {
            std::string out;
            out.reserve(x.size());
            for (std::uint32_t d : x)
                out.push_back(alphabet[d]);
            return out;
        }

        void cbc_mac_continue(unsigned char* state, const unsigned char* data, std::size_t size) const
        {
            for (std::size_t off = 0; off < size; off += 16)
            {
                for (int i = 0; i < 16; ++i)
                    state[i] = static_cast<unsigned char>(state[i] ^ data[off + i]);
                aes_.encrypt_block(state, state);
            }
        }

        std::vector<std::uint32_t> process(
            const std::vector<std::uint32_t>& x,
            const std::string& tweak,
            bool forward
        ) const
        {
            const std::size_t n = x.size();
            if (!impl::ff1_domain_is_large_enough(radix_, n))
                throw crypto_error("ff1_cipher: radix^n must be at least 1,000,000 and n at least 2");
            if (n >= 0xFFFFFFFFu || tweak.size() >= 0xFFFFFFFFu)
                throw crypto_error("ff1_cipher: input or tweak too long");
            for (std::uint32_t d : x)
                if (d >= radix_)
                    throw crypto_error("ff1_cipher: numeral out of range for the radix");

            const std::size_t u = n / 2;
            const std::size_t v = n - u;
            const std::size_t t = tweak.size();

            // b = ceil(ceil(v * log2(radix)) / 8), computed as the bit length of radix^v - 1.
            impl::ff1_bignum rv;
            rv.mul_add_small(1, 1);
            for (std::size_t i = 0; i < v; ++i)
                rv.mul_add_small(radix_, 0);
            rv.decrement();
            const std::size_t b = (rv.bit_length() + 7) / 8;
            const std::size_t d = 4 * ((b + 3) / 4) + 4;

            unsigned char p_block[16] = {
                1, 2, 1,
                static_cast<unsigned char>(radix_ >> 16), static_cast<unsigned char>(radix_ >> 8),
                static_cast<unsigned char>(radix_),
                10,
                static_cast<unsigned char>(u & 0xFF),
                static_cast<unsigned char>(n >> 24), static_cast<unsigned char>(n >> 16),
                static_cast<unsigned char>(n >> 8), static_cast<unsigned char>(n),
                static_cast<unsigned char>(t >> 24), static_cast<unsigned char>(t >> 16),
                static_cast<unsigned char>(t >> 8), static_cast<unsigned char>(t)
            };
            unsigned char after_p[16];
            aes_.encrypt_block(p_block, after_p);

            const std::size_t pad = (16 - ((t + b + 1) % 16)) % 16;
            std::vector<unsigned char> q(t + pad + 1 + b);
            std::memcpy(q.data(), tweak.data(), t);
            std::memset(q.data() + t, 0, pad);

            std::vector<std::uint32_t> a(x.begin(), x.begin() + u);
            std::vector<std::uint32_t> bb(x.begin() + u, x.end());
            std::vector<unsigned char> s(((d + 15) / 16) * 16);
            std::vector<std::uint32_t> y_digits;

            for (int step = 0; step < 10; ++step)
            {
                const int i = forward ? step : 9 - step;
                // The PRF input carries the half that is not being modified in this round.
                const std::vector<std::uint32_t>& keep = forward ? bb : a;
                std::vector<std::uint32_t>& change = forward ? a : bb;

                q[t + pad] = static_cast<unsigned char>(i);
                const impl::ff1_bignum num_keep = impl::ff1_bignum::from_numerals(keep.data(), keep.size(), radix_);
                if (!num_keep.to_bytes(b, q.data() + t + pad + 1))
                    throw crypto_error("ff1_cipher: internal size error");

                unsigned char r[16];
                std::memcpy(r, after_p, 16);
                cbc_mac_continue(r, q.data(), q.size());

                std::memcpy(s.data(), r, 16);
                for (std::size_t j = 1; 16 * j < d; ++j)
                {
                    unsigned char blk[16];
                    std::memcpy(blk, r, 16);
                    for (int k = 0; k < 8; ++k)
                        blk[15 - k] = static_cast<unsigned char>(
                            blk[15 - k] ^ static_cast<unsigned char>(static_cast<std::uint64_t>(j) >> (8 * k)));
                    aes_.encrypt_block(blk, s.data() + 16 * j);
                }

                impl::ff1_bignum y = impl::ff1_bignum::from_bytes(s.data(), d);
                const std::size_t m = (i % 2 == 0) ? u : v;
                y_digits.assign(m, 0);
                for (std::size_t k = 0; k < m; ++k)
                    y_digits[k] = y.divmod_small(radix_);

                // change has m numerals, most significant first; y_digits are least significant first.
                std::vector<std::uint32_t> c(m);
                if (forward)
                {
                    std::uint32_t carry = 0;
                    for (std::size_t k = 0; k < m; ++k)
                    {
                        const std::uint64_t sum = static_cast<std::uint64_t>(change[m - 1 - k]) + y_digits[k] + carry;
                        c[m - 1 - k] = static_cast<std::uint32_t>(sum % radix_);
                        carry = static_cast<std::uint32_t>(sum / radix_);
                    }
                    a = bb;
                    bb = c;
                }
                else
                {
                    std::int64_t borrow = 0;
                    for (std::size_t k = 0; k < m; ++k)
                    {
                        std::int64_t diff = static_cast<std::int64_t>(change[m - 1 - k]) - y_digits[k] - borrow;
                        borrow = 0;
                        if (diff < 0)
                        {
                            diff += radix_;
                            borrow = 1;
                        }
                        c[m - 1 - k] = static_cast<std::uint32_t>(diff);
                    }
                    bb = a;
                    a = c;
                }
                secure_wipe(r, sizeof(r));
            }

            secure_wipe(s.data(), s.size());
            std::vector<std::uint32_t> out;
            out.reserve(n);
            out.insert(out.end(), a.begin(), a.end());
            out.insert(out.end(), bb.begin(), bb.end());
            return out;
        }

        aes_block_cipher aes_;
        std::uint32_t radix_;
    };

// ----------------------------------------------------------------------------------------

    inline std::vector<std::uint32_t> bytes_to_numerals(
        const std::string& bytes,
        std::uint32_t radix,
        std::size_t length
    )
    {
        if (radix < 2 || radix > 65536)
            throw crypto_error("bytes_to_numerals(): the radix must lie in [2, 65536]");
        impl::ff1_bignum v = impl::ff1_bignum::from_bytes(
            reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
        std::vector<std::uint32_t> digits(length, 0);
        for (std::size_t k = 0; k < length; ++k)
            digits[length - 1 - k] = v.divmod_small(radix);
        if (!v.is_zero())
            throw crypto_error("bytes_to_numerals(): the value does not fit in the requested length");
        return digits;
    }

    inline bool numerals_to_bytes(
        const std::vector<std::uint32_t>& numerals,
        std::uint32_t radix,
        std::size_t length,
        std::string& bytes
    )
    {
        const impl::ff1_bignum v = impl::ff1_bignum::from_numerals(numerals.data(), numerals.size(), radix);
        std::vector<unsigned char> buf(length);
        if (length > 0 && !v.to_bytes(length, buf.data()))
            return false;
        if (length == 0 && !v.is_zero())
            return false;
        bytes.assign(reinterpret_cast<const char*>(buf.data()), buf.size());
        return true;
    }

    inline std::size_t numerals_needed(
        std::size_t byte_count,
        std::uint32_t radix
    )
    {
        if (radix < 2 || radix > 65536)
            throw crypto_error("numerals_needed(): the radix must lie in [2, 65536]");
        impl::ff1_bignum p;
        p.mul_add_small(1, 1);
        std::size_t m = 0;
        while (p.bit_length() < 8 * byte_count + 1)
        {
            p.mul_add_small(radix, 0);
            ++m;
        }
        return m;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FF1_H_
