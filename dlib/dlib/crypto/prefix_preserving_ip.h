// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PREFIX_PRESERVING_IP_H_
#define DLIB_PREFIX_PRESERVING_IP_H_

#include "prefix_preserving_ip_abstract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "aes.h"
#include "crypto_utils.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class prefix_preserving_ip_cipher
    {
    public:
        prefix_preserving_ip_cipher(const void* key, std::size_t key_size)
        {
            if (key_size != 32)
                throw crypto_error("prefix_preserving_ip_cipher: the key must be 32 bytes long");
            const unsigned char* k = static_cast<const unsigned char*>(key);
            aes_.set_key(k, 16);
            aes_.encrypt_block(k + 16, pad_);
        }

        ~prefix_preserving_ip_cipher()
        {
            secure_wipe(pad_, sizeof(pad_));
        }

        std::uint32_t encrypt_ipv4(std::uint32_t address) const
        {
            unsigned char in[16] = {}, out[16];
            for (int i = 0; i < 4; ++i)
                in[i] = static_cast<unsigned char>(address >> (24 - 8 * i));
            transform(in, out, 32, true);
            return (static_cast<std::uint32_t>(out[0]) << 24) | (static_cast<std::uint32_t>(out[1]) << 16) |
                   (static_cast<std::uint32_t>(out[2]) << 8) | static_cast<std::uint32_t>(out[3]);
        }

        std::uint32_t decrypt_ipv4(std::uint32_t address) const
        {
            unsigned char in[16] = {}, out[16];
            for (int i = 0; i < 4; ++i)
                in[i] = static_cast<unsigned char>(address >> (24 - 8 * i));
            transform(in, out, 32, false);
            return (static_cast<std::uint32_t>(out[0]) << 24) | (static_cast<std::uint32_t>(out[1]) << 16) |
                   (static_cast<std::uint32_t>(out[2]) << 8) | static_cast<std::uint32_t>(out[3]);
        }

        std::array<unsigned char,16> encrypt_ipv6(const std::array<unsigned char,16>& address) const
        {
            std::array<unsigned char,16> out;
            transform(address.data(), out.data(), 128, true);
            return out;
        }

        std::array<unsigned char,16> decrypt_ipv6(const std::array<unsigned char,16>& address) const
        {
            std::array<unsigned char,16> out;
            transform(address.data(), out.data(), 128, false);
            return out;
        }

    private:
        static int get_bit(const unsigned char* p, int i)
        {
            return (p[i >> 3] >> (7 - (i & 7))) & 1;
        }

        /*
            The mask bit for position pos is drawn from the original address prefix of
            length pos. When encrypting, that prefix is the input; when decrypting it is
            the part of the output already recovered, which is why both directions walk
            the bits in the same order.
        */
        void transform(const unsigned char* in, unsigned char* out, int bits, bool forward) const
        {
            unsigned char original[16];
            std::memset(original, 0, sizeof(original));
            if (forward)
                std::memcpy(original, in, static_cast<std::size_t>(bits / 8));
            std::memset(out, 0, 16);

            for (int pos = 0; pos < bits; ++pos)
            {
                unsigned char block[16];
                std::memcpy(block, pad_, 16);
                const int full = pos / 8;
                const int rest = pos % 8;
                std::memcpy(block, original, static_cast<std::size_t>(full));
                if (rest != 0)
                {
                    const unsigned char keep_mask = static_cast<unsigned char>(0xFF << (8 - rest));
                    block[full] = static_cast<unsigned char>((original[full] & keep_mask) | (pad_[full] & ~keep_mask));
                }
                unsigned char enc[16];
                aes_.encrypt_block(block, enc);
                const int flip = enc[0] >> 7;
                const int out_bit = get_bit(in, pos) ^ flip;
                if (out_bit)
                    out[pos >> 3] = static_cast<unsigned char>(out[pos >> 3] | (0x80 >> (pos & 7)));
                if (!forward && out_bit)
                    original[pos >> 3] = static_cast<unsigned char>(original[pos >> 3] | (0x80 >> (pos & 7)));
            }
            secure_wipe(original, sizeof(original));
        }

        aes_block_cipher aes_;
        unsigned char pad_[16];
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PREFIX_PRESERVING_IP_H_
