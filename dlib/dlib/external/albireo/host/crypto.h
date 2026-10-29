// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_CRYPTO_H_
#define ALBIREO_HOST_CRYPTO_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace albireo
{

    class sha256_stream
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A SHA-256 computed over data given in pieces.
        !*/
    public:
        sha256_stream();
        ~sha256_stream();
        void update(const void* data, size_t n);
        std::string hex();   // ends the computation

    private:
        struct impl;
        std::unique_ptr<impl> p_;
    };

    std::string sha256_hex(const std::string& data);

    /*!
        ensures
            - Returns the SHA-256 of a file in lower case hexadecimal, or an empty string
              when it cannot be read or when cancel becomes true.
            - progress, when given, receives the bytes read so far.
    !*/
    std::string sha256_file(const std::string& path, const std::function<void(uint64_t)>& progress = nullptr,
        const std::atomic<bool>* cancel = nullptr);

    /*!
        ensures
            - Checks an ECDSA P-256 signature with SHA-256 (DER encoded, given in base64)
              of data against a public key in PEM form.
    !*/
    bool verify_signature(const std::string& data, const std::string& signature_base64,
        const std::string& public_key_pem);

    // Signs data with a private key in PEM form; returns the DER signature in base64.
    std::string sign_data(const std::string& data, const std::string& private_key_pem, std::string& error);

    std::string base64_encode(const std::string& data);
    std::string base64_decode(const std::string& text);

    // The public key that signs the update feed and the model catalog.
    extern const char* const feed_public_key_pem;

}

#endif // ALBIREO_HOST_CRYPTO_H_
