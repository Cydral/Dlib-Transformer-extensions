// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SHA256_ABSTRACT_H_
#ifdef DLIB_SHA256_ABSTRACT_H_

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class sha256
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                An incremental SHA-256 hash (FIPS 180-4). Data is absorbed with update()
                in as many pieces as convenient and the 32 byte digest is obtained with
                finalize().

            TYPICAL USAGE
                sha256 h;
                h.update(header);
                h.update(body);
                const std::array<unsigned char,32> digest = h.finalize();
        !*/
    public:
        sha256();
        /*!
            ensures
                - #*this is ready to absorb a new message.
        !*/

        void clear();
        /*!
            ensures
                - Discards anything absorbed so far.
        !*/

        void update(const void* data, std::size_t size);
        void update(const std::string& data);
        /*!
            ensures
                - Appends the given bytes to the message being hashed.
        !*/

        std::array<unsigned char,32> finalize();
        /*!
            ensures
                - Returns the digest of everything absorbed since construction or the last
                  clear()/finalize().
                - #*this is ready to absorb a new message.
        !*/
    };

    std::array<unsigned char,32> sha256_digest(const void* data, std::size_t size);
    std::array<unsigned char,32> sha256_digest(const std::string& data);
    /*!
        ensures
            - Returns the SHA-256 digest of the given bytes.
    !*/

// ----------------------------------------------------------------------------------------

    class hmac_sha256
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                HMAC-SHA-256 (RFC 2104, FIPS 198-1) bound to one key. The inner and outer
                padded key blocks are absorbed once at construction, so computing many
                MACs under the same key costs two compressions per short message.

                The object erases its key-dependent state when destroyed, and finalize()
                erases its intermediate digests.
        !*/
    public:
        hmac_sha256(const void* key, std::size_t key_size);
        explicit hmac_sha256(const std::string& key);
        /*!
            ensures
                - #*this computes MACs under the given key, which may have any length.
        !*/

        void update(const void* data, std::size_t size);
        void update(const std::string& data);
        /*!
            ensures
                - Appends the given bytes to the message being authenticated.
        !*/

        std::array<unsigned char,32> finalize();
        /*!
            ensures
                - Returns the tag of the message absorbed since construction or the last
                  finalize().
                - #*this is ready to authenticate a new message under the same key.
        !*/
    };

    std::array<unsigned char,32> hmac_sha256_digest(
        const std::string& key,
        const std::string& message
    );
    /*!
        ensures
            - Returns HMAC-SHA-256(key, message).
    !*/

// ----------------------------------------------------------------------------------------

    std::string hkdf_sha256(
        const std::string& input_key_material,
        const std::string& salt,
        const std::string& info,
        std::size_t length
    );
    /*!
        ensures
            - Returns length bytes of HKDF-SHA-256 output (RFC 5869): extract with salt,
              then expand with info. An empty salt is replaced by 32 zero bytes, as the
              RFC specifies.
            - Distinct info strings yield independent keys from the same input key
              material.
            - The pseudorandom key of the extract step is erased before returning.
        throws
            - crypto_error if length > 255*32.
    !*/

    std::string pbkdf2_hmac_sha256(
        const std::string& password,
        const std::string& salt,
        unsigned long iterations,
        std::size_t length
    );
    /*!
        ensures
            - Returns length bytes of PBKDF2 (RFC 8018, NIST SP 800-132) with HMAC-SHA-256
              as the pseudorandom function.
        throws
            - crypto_error if iterations == 0 or length == 0.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHA256_ABSTRACT_H_
