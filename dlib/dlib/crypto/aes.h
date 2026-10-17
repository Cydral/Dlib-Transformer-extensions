// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AES_H_
#define DLIB_AES_H_

#include "aes_abstract.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "../assert.h"
#include "crypto_utils.h"

#if !defined(DLIB_NO_AESNI) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#  if defined(_MSC_VER) && !defined(__clang__)
#    define DLIB_AES_NI_AVAILABLE 1
#    include <intrin.h>
#    include <wmmintrin.h>
#    define DLIB_AES_NI_TARGET
#  elif defined(__GNUC__) || defined(__clang__)
#    define DLIB_AES_NI_AVAILABLE 1
#    include <cpuid.h>
#    include <wmmintrin.h>
#    define DLIB_AES_NI_TARGET __attribute__((target("aes,sse2")))
#  endif
#endif

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        inline std::uint8_t aes_xtime(std::uint8_t a)
        {
            const std::uint8_t mask = static_cast<std::uint8_t>(0u - static_cast<unsigned int>(a >> 7));
            return static_cast<std::uint8_t>((a << 1) ^ (0x1b & mask));
        }

        // Multiplication in GF(2^8) without data-dependent branches or table reads.
        inline std::uint8_t aes_gmul(std::uint8_t a, std::uint8_t b)
        {
            std::uint8_t r = 0;
            for (int i = 0; i < 8; ++i)
            {
                const std::uint8_t mask = static_cast<std::uint8_t>(0u - static_cast<unsigned int>(b & 1));
                r = static_cast<std::uint8_t>(r ^ (a & mask));
                a = aes_xtime(a);
                b = static_cast<std::uint8_t>(b >> 1);
            }
            return r;
        }

        inline std::uint8_t aes_rotl8(std::uint8_t x, unsigned int n)
        {
            return static_cast<std::uint8_t>((x << n) | (x >> (8 - n)));
        }

        /*
            S-box computed as the affine map applied to x^254, which is the multiplicative
            inverse of x in GF(2^8) and maps 0 to 0. The addition chain uses 11 products:
            x^2, x^3, x^6, x^12, x^15, x^30, x^60, x^120, x^240, x^252, x^254.
        */
        inline std::uint8_t aes_sbox(std::uint8_t x)
        {
            const std::uint8_t x2 = aes_gmul(x, x);
            const std::uint8_t x3 = aes_gmul(x2, x);
            const std::uint8_t x6 = aes_gmul(x3, x3);
            const std::uint8_t x12 = aes_gmul(x6, x6);
            const std::uint8_t x15 = aes_gmul(x12, x3);
            const std::uint8_t x30 = aes_gmul(x15, x15);
            const std::uint8_t x60 = aes_gmul(x30, x30);
            const std::uint8_t x120 = aes_gmul(x60, x60);
            const std::uint8_t x240 = aes_gmul(x120, x120);
            const std::uint8_t x252 = aes_gmul(x240, x12);
            const std::uint8_t inv = aes_gmul(x252, x2);
            return static_cast<std::uint8_t>(inv ^ aes_rotl8(inv, 1) ^ aes_rotl8(inv, 2) ^
                                             aes_rotl8(inv, 3) ^ aes_rotl8(inv, 4) ^ 0x63);
        }

#ifdef DLIB_AES_NI_AVAILABLE
        inline bool cpu_has_aes_ni()
        {
#  if defined(_MSC_VER) && !defined(__clang__)
            int regs[4];
            __cpuid(regs, 1);
            return (regs[2] & (1 << 25)) != 0;
#  else
            unsigned int a = 0, b = 0, c = 0, d = 0;
            if (!__get_cpuid(1, &a, &b, &c, &d))
                return false;
            return (c & (1u << 25)) != 0;
#  endif
        }

        DLIB_AES_NI_TARGET
        inline void aes_ni_encrypt(
            const unsigned char* round_keys,
            int rounds,
            const unsigned char* in,
            unsigned char* out
        )
        {
            __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));
            s = _mm_xor_si128(s, _mm_loadu_si128(reinterpret_cast<const __m128i*>(round_keys)));
            for (int r = 1; r < rounds; ++r)
                s = _mm_aesenc_si128(s, _mm_loadu_si128(reinterpret_cast<const __m128i*>(round_keys + 16 * r)));
            s = _mm_aesenclast_si128(s, _mm_loadu_si128(reinterpret_cast<const __m128i*>(round_keys + 16 * rounds)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out), s);
        }
#endif
    }

// ----------------------------------------------------------------------------------------

    class aes_block_cipher
    {
    public:
        aes_block_cipher() : rounds_(0), key_size_(0), hardware_(false)
        {
            std::memset(round_keys_, 0, sizeof(round_keys_));
        }

        aes_block_cipher(const void* key, std::size_t key_size) : aes_block_cipher()
        {
            set_key(key, key_size);
        }

        aes_block_cipher(const aes_block_cipher&) = default;
        aes_block_cipher& operator=(const aes_block_cipher&) = default;

        ~aes_block_cipher()
        {
            secure_wipe(round_keys_, sizeof(round_keys_));
        }

        void set_key(const void* key, std::size_t key_size)
        {
            if (key_size != 16 && key_size != 24 && key_size != 32)
                throw crypto_error("aes_block_cipher: the key must be 16, 24 or 32 bytes long");

            secure_wipe(round_keys_, sizeof(round_keys_));
            const int nk = static_cast<int>(key_size / 4);
            rounds_ = nk + 6;
            key_size_ = key_size;
            const int total_words = 4 * (rounds_ + 1);

            std::memcpy(round_keys_, key, key_size);
            std::uint8_t rcon = 0x01;
            for (int i = nk; i < total_words; ++i)
            {
                std::uint8_t t[4];
                std::memcpy(t, round_keys_ + 4 * (i - 1), 4);
                if (i % nk == 0)
                {
                    const std::uint8_t first = t[0];
                    t[0] = static_cast<std::uint8_t>(impl::aes_sbox(t[1]) ^ rcon);
                    t[1] = impl::aes_sbox(t[2]);
                    t[2] = impl::aes_sbox(t[3]);
                    t[3] = impl::aes_sbox(first);
                    rcon = impl::aes_xtime(rcon);
                }
                else if (nk > 6 && i % nk == 4)
                {
                    for (int j = 0; j < 4; ++j)
                        t[j] = impl::aes_sbox(t[j]);
                }
                for (int j = 0; j < 4; ++j)
                    round_keys_[4 * i + j] = static_cast<unsigned char>(round_keys_[4 * (i - nk) + j] ^ t[j]);
            }

#ifdef DLIB_AES_NI_AVAILABLE
            hardware_ = impl::cpu_has_aes_ni();
#else
            hardware_ = false;
#endif
        }

        bool has_key() const { return key_size_ != 0; }
        std::size_t key_size() const { return key_size_; }
        bool uses_hardware_acceleration() const { return hardware_; }

        void encrypt_block(const unsigned char* in, unsigned char* out) const
        {
            DLIB_ASSERT(has_key(), "\t aes_cipher::encrypt_block(): the cipher has no key");
#ifdef DLIB_AES_NI_AVAILABLE
            if (hardware_)
            {
                impl::aes_ni_encrypt(round_keys_, rounds_, in, out);
                return;
            }
#endif
            encrypt_block_portable(in, out);
        }

    private:
        void encrypt_block_portable(const unsigned char* in, unsigned char* out) const
        {
            std::uint8_t s[16];
            for (int i = 0; i < 16; ++i)
                s[i] = static_cast<std::uint8_t>(in[i] ^ round_keys_[i]);

            for (int round = 1; round <= rounds_; ++round)
            {
                std::uint8_t t[16];
                // SubBytes and ShiftRows: row r of column c takes the byte of column c + r.
                for (int c = 0; c < 4; ++c)
                    for (int r = 0; r < 4; ++r)
                        t[r + 4 * c] = impl::aes_sbox(s[r + 4 * ((c + r) & 3)]);

                if (round != rounds_)
                {
                    for (int c = 0; c < 4; ++c)
                    {
                        const std::uint8_t a0 = t[4 * c], a1 = t[4 * c + 1], a2 = t[4 * c + 2], a3 = t[4 * c + 3];
                        const std::uint8_t all = static_cast<std::uint8_t>(a0 ^ a1 ^ a2 ^ a3);
                        s[4 * c + 0] =
                            static_cast<std::uint8_t>(a0 ^ all ^ impl::aes_xtime(static_cast<std::uint8_t>(a0 ^ a1)));
                        s[4 * c + 1] =
                            static_cast<std::uint8_t>(a1 ^ all ^ impl::aes_xtime(static_cast<std::uint8_t>(a1 ^ a2)));
                        s[4 * c + 2] =
                            static_cast<std::uint8_t>(a2 ^ all ^ impl::aes_xtime(static_cast<std::uint8_t>(a2 ^ a3)));
                        s[4 * c + 3] =
                            static_cast<std::uint8_t>(a3 ^ all ^ impl::aes_xtime(static_cast<std::uint8_t>(a3 ^ a0)));
                    }
                }
                else
                {
                    std::memcpy(s, t, 16);
                }

                const unsigned char* rk = round_keys_ + 16 * round;
                for (int i = 0; i < 16; ++i)
                    s[i] = static_cast<std::uint8_t>(s[i] ^ rk[i]);
            }
            std::memcpy(out, s, 16);
            secure_wipe(s, sizeof(s));
        }

        unsigned char round_keys_[240];
        int rounds_;
        std::size_t key_size_;
        bool hardware_;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_AES_H_
