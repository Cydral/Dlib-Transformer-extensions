// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_AES_ABSTRACT_H_
#ifdef DLIB_AES_ABSTRACT_H_

#include <cstddef>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class aes_block_cipher
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The AES forward block cipher (FIPS 197) with 128, 192 or 256 bit keys.
                Only the encryption direction is provided. The modes built on it in this
                library use no other: the FF1 Feistel network, CBC-MAC, counter-based
                keystreams and the Crypto-PAn construction.

            SIDE CHANNELS
                The portable path computes the S-box arithmetically, as an inversion in
                GF(2^8) followed by the affine map, so that its memory access pattern is
                the same whatever the key and the data. Where the AES-NI instruction set
                extension is available, the rounds use it; its timing does not depend on
                the key or the data either. The choice is made at run time and does not
                change any output.

            THREAD SAFETY
                encrypt_block() is const and may be called concurrently once the key is
                set.
        !*/
    public:
        aes_block_cipher();
        /*!
            ensures
                - #has_key() == false
        !*/

        aes_block_cipher(const void* key, std::size_t key_size);
        /*!
            ensures
                - #*this encrypts under the given key.
            throws
                - crypto_error if key_size is not 16, 24 or 32.
        !*/

        void set_key(const void* key, std::size_t key_size);
        /*!
            ensures
                - Replaces the key, erasing the previous key schedule.
            throws
                - crypto_error if key_size is not 16, 24 or 32.
        !*/

        bool has_key() const;
        /*!
            ensures
                - Returns true if a key is set.
        !*/

        std::size_t key_size() const;
        /*!
            ensures
                - Returns the key length in bytes, or 0 when no key is set.
        !*/

        void encrypt_block(const unsigned char* in, unsigned char* out) const;
        /*!
            requires
                - has_key() == true
                - in and out point to 16 bytes each; they may be the same buffer.
            ensures
                - Writes AES_K(in) to out.
        !*/

        bool uses_hardware_acceleration() const;
        /*!
            ensures
                - Returns true when the AES-NI path is in use.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_AES_ABSTRACT_H_
