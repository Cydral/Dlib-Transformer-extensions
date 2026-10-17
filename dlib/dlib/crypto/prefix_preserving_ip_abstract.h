// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PREFIX_PRESERVING_IP_ABSTRACT_H_
#ifdef DLIB_PREFIX_PRESERVING_IP_ABSTRACT_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class prefix_preserving_ip_cipher
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The Crypto-PAn construction of Xu, Fan, Ammar and Moon (ICNP 2002): a keyed
                bijection on IPv4 and IPv6 addresses that preserves prefixes. Two
                addresses sharing their first k bits are mapped to two addresses sharing
                their first k bits, and to no longer common prefix. Subnet membership,
                routing aggregation and "same network" comparisons therefore remain
                computable on the protected addresses.

                Bit i of the output is bit i of the input XORed with the first bit of
                AES_K(first i bits of the input || pad bits), where pad is AES_K applied
                to the second half of the key. Decryption recovers the input bit by bit,
                since bit i of the mask depends only on input bits already recovered.

                The 32 byte key is split as in the reference implementation: the first 16
                bytes are the AES-128 key, the last 16 bytes are encrypted to form pad.
                IPv4 outputs are bit-identical to the reference implementation, which
                handles IPv4 only; IPv6 addresses follow the same construction over 128
                bits.
        !*/
    public:
        prefix_preserving_ip_cipher(const void* key, std::size_t key_size);
        /*!
            ensures
                - #*this encrypts addresses under the given key.
            throws
                - crypto_error if key_size != 32.
        !*/

        std::uint32_t encrypt_ipv4(std::uint32_t address) const;
        std::uint32_t decrypt_ipv4(std::uint32_t address) const;
        /*!
            ensures
                - The address is a host-order integer, 192.0.2.1 being 0xC0000201.
                - decrypt_ipv4(encrypt_ipv4(a)) == a
        !*/

        std::array<unsigned char,16> encrypt_ipv6(const std::array<unsigned char,16>& address) const;
        std::array<unsigned char,16> decrypt_ipv6(const std::array<unsigned char,16>& address) const;
        /*!
            ensures
                - The address is in network byte order.
                - decrypt_ipv6(encrypt_ipv6(a)) == a
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PREFIX_PRESERVING_IP_ABSTRACT_H_
