// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHA256_H_
#define DLIB_SHA256_H_

#include "sha256_abstract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "crypto_utils.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline const std::uint32_t* sha256_round_constants()
        {
            static const std::uint32_t k[64] = {
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
            };
            return k;
        }

        inline std::uint32_t sha256_rotr(std::uint32_t x, unsigned int n)
        {
            return (x >> n) | (x << (32 - n));
        }
    }

// ----------------------------------------------------------------------------------------

    class sha256
    {
    public:
        sha256() { clear(); }

        void clear()
        {
            static const std::uint32_t init[8] = {
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
            };
            std::memcpy(h_, init, sizeof(h_));
            total_ = 0;
            buffered_ = 0;
        }

        void update(const void* data, std::size_t size)
        {
            const unsigned char* p = static_cast<const unsigned char*>(data);
            total_ += size;
            if (buffered_ > 0)
            {
                const std::size_t take = (size < 64 - buffered_) ? size : 64 - buffered_;
                std::memcpy(buffer_ + buffered_, p, take);
                buffered_ += take;
                p += take;
                size -= take;
                if (buffered_ == 64)
                {
                    compress(buffer_);
                    buffered_ = 0;
                }
            }
            while (size >= 64)
            {
                compress(p);
                p += 64;
                size -= 64;
            }
            if (size > 0)
            {
                std::memcpy(buffer_, p, size);
                buffered_ = size;
            }
        }

        void update(const std::string& data) { update(data.data(), data.size()); }

        std::array<unsigned char,32> finalize()
        {
            const std::uint64_t bit_length = total_ * 8;
            unsigned char pad[72];
            std::size_t pad_len = (buffered_ < 56) ? (56 - buffered_) : (120 - buffered_);
            std::memset(pad, 0, sizeof(pad));
            pad[0] = 0x80;
            for (int i = 0; i < 8; ++i)
                pad[pad_len + i] = static_cast<unsigned char>(bit_length >> (56 - 8 * i));
            update(pad, pad_len + 8);

            std::array<unsigned char,32> digest;
            for (int i = 0; i < 8; ++i)
            {
                digest[4 * i + 0] = static_cast<unsigned char>(h_[i] >> 24);
                digest[4 * i + 1] = static_cast<unsigned char>(h_[i] >> 16);
                digest[4 * i + 2] = static_cast<unsigned char>(h_[i] >> 8);
                digest[4 * i + 3] = static_cast<unsigned char>(h_[i]);
            }
            clear();
            return digest;
        }

    private:
        void compress(const unsigned char* block)
        {
            using impl::sha256_rotr;
            const std::uint32_t* k = impl::sha256_round_constants();
            std::uint32_t w[64];
            for (int i = 0; i < 16; ++i)
            {
                w[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) |
                       (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
                       (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
                       static_cast<std::uint32_t>(block[4 * i + 3]);
            }
            for (int i = 16; i < 64; ++i)
            {
                const std::uint32_t s0 = sha256_rotr(w[i - 15], 7) ^ sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const std::uint32_t s1 = sha256_rotr(w[i - 2], 17) ^ sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16] + s0 + w[i - 7] + s1;
            }

            std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
            std::uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
            for (int i = 0; i < 64; ++i)
            {
                const std::uint32_t S1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
                const std::uint32_t ch = (e & f) ^ (~e & g);
                const std::uint32_t t1 = h + S1 + ch + k[i] + w[i];
                const std::uint32_t S0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
                const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
                const std::uint32_t t2 = S0 + maj;
                h = g; g = f; f = e; e = d + t1;
                d = c; c = b; b = a; a = t1 + t2;
            }
            h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
            h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
        }

        std::uint32_t h_[8];
        std::uint64_t total_;
        unsigned char buffer_[64];
        std::size_t buffered_;
    };

    inline std::array<unsigned char,32> sha256_digest(const void* data, std::size_t size)
    {
        sha256 h;
        h.update(data, size);
        return h.finalize();
    }

    inline std::array<unsigned char,32> sha256_digest(const std::string& data)
    {
        return sha256_digest(data.data(), data.size());
    }

// ----------------------------------------------------------------------------------------

    class hmac_sha256
    {
    public:
        hmac_sha256(const void* key, std::size_t key_size) { init(key, key_size); }
        explicit hmac_sha256(const std::string& key) { init(key.data(), key.size()); }

        hmac_sha256(const hmac_sha256&) = default;
        hmac_sha256& operator=(const hmac_sha256&) = default;

        ~hmac_sha256()
        {
            secure_wipe(&inner_init_, sizeof(inner_init_));
            secure_wipe(&outer_init_, sizeof(outer_init_));
            secure_wipe(&inner_, sizeof(inner_));
        }

        void update(const void* data, std::size_t size) { inner_.update(data, size); }
        void update(const std::string& data) { inner_.update(data.data(), data.size()); }

        std::array<unsigned char,32> finalize()
        {
            std::array<unsigned char,32> inner_digest = inner_.finalize();
            sha256 outer = outer_init_;
            outer.update(inner_digest.data(), inner_digest.size());
            inner_ = inner_init_;
            const std::array<unsigned char,32> mac = outer.finalize();
            secure_wipe(inner_digest.data(), inner_digest.size());
            secure_wipe(&outer, sizeof(outer));
            return mac;
        }

    private:
        void init(const void* key, std::size_t key_size)
        {
            unsigned char block[64];
            std::memset(block, 0, sizeof(block));
            if (key_size > 64)
            {
                const std::array<unsigned char,32> d = sha256_digest(key, key_size);
                std::memcpy(block, d.data(), d.size());
            }
            else if (key_size > 0)
            {
                std::memcpy(block, key, key_size);
            }

            unsigned char pad[64];
            for (int i = 0; i < 64; ++i) pad[i] = static_cast<unsigned char>(block[i] ^ 0x36);
            inner_init_.update(pad, 64);
            for (int i = 0; i < 64; ++i) pad[i] = static_cast<unsigned char>(block[i] ^ 0x5c);
            outer_init_.update(pad, 64);
            inner_ = inner_init_;
            secure_wipe(block, sizeof(block));
            secure_wipe(pad, sizeof(pad));
        }

        sha256 inner_init_;
        sha256 outer_init_;
        sha256 inner_;
    };

    inline std::array<unsigned char,32> hmac_sha256_digest(
        const std::string& key,
        const std::string& message
    )
    {
        hmac_sha256 mac(key);
        mac.update(message);
        return mac.finalize();
    }

// ----------------------------------------------------------------------------------------

    inline std::string hkdf_sha256(
        const std::string& input_key_material,
        const std::string& salt,
        const std::string& info,
        std::size_t length
    )
    {
        if (length > 255 * 32)
            throw crypto_error("hkdf_sha256(): requested length exceeds 255 hash blocks");

        const std::string effective_salt = salt.empty() ? std::string(32, '\0') : salt;
        std::array<unsigned char,32> prk = hmac_sha256_digest(effective_salt, input_key_material);

        hmac_sha256 mac(prk.data(), prk.size());
        secure_wipe(prk.data(), prk.size());
        std::string okm;
        okm.reserve(length);
        std::array<unsigned char,32> t{};
        std::size_t t_len = 0;
        unsigned char counter = 1;
        while (okm.size() < length)
        {
            mac.update(t.data(), t_len);
            mac.update(info);
            mac.update(&counter, 1);
            t = mac.finalize();
            t_len = t.size();
            const std::size_t take = (length - okm.size() < t.size()) ? length - okm.size() : t.size();
            okm.append(reinterpret_cast<const char*>(t.data()), take);
            ++counter;
        }
        secure_wipe(t.data(), t.size());
        return okm;
    }

    inline std::string pbkdf2_hmac_sha256(
        const std::string& password,
        const std::string& salt,
        unsigned long iterations,
        std::size_t length
    )
    {
        if (iterations == 0)
            throw crypto_error("pbkdf2_hmac_sha256(): the iteration count must be positive");
        if (length == 0)
            throw crypto_error("pbkdf2_hmac_sha256(): the output length must be positive");

        hmac_sha256 prf(password);
        std::string out;
        out.reserve(length);
        for (std::uint32_t block_index = 1; out.size() < length; ++block_index)
        {
            const unsigned char be_index[4] = {
                static_cast<unsigned char>(block_index >> 24), static_cast<unsigned char>(block_index >> 16),
                static_cast<unsigned char>(block_index >> 8), static_cast<unsigned char>(block_index)
            };
            prf.update(salt);
            prf.update(be_index, 4);
            std::array<unsigned char,32> u = prf.finalize();
            std::array<unsigned char,32> t = u;
            for (unsigned long j = 1; j < iterations; ++j)
            {
                prf.update(u.data(), u.size());
                u = prf.finalize();
                for (std::size_t i = 0; i < t.size(); ++i)
                    t[i] ^= u[i];
            }
            const std::size_t take = (length - out.size() < t.size()) ? length - out.size() : t.size();
            out.append(reinterpret_cast<const char*>(t.data()), take);
            secure_wipe(u.data(), u.size());
            secure_wipe(t.data(), t.size());
        }
        return out;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHA256_H_
