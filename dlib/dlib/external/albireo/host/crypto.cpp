// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "crypto.h"

#include <fstream>
#include <vector>

#include <mbedtls/base64.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/entropy.h>
#include <mbedtls/md.h>
#include <mbedtls/pk.h>
#include <mbedtls/sha256.h>
#include <psa/crypto.h>

#include "../common/platform.h"

namespace albireo
{

    const char* const feed_public_key_pem =
        "-----BEGIN PUBLIC KEY-----\n"
        "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEJ5dUhne+WUdHXTPxWFhgRfy8aANq\n"
        "nkkeFLVzToFfiWTl9sGwuRkECPcyhYVP3dC1/yhsV/KCYxxhlwGcmeAevw==\n"
        "-----END PUBLIC KEY-----\n";

    struct sha256_stream::impl
    {
        mbedtls_sha256_context ctx;
    };

    sha256_stream::sha256_stream() : p_(new impl())
    {
        mbedtls_sha256_init(&p_->ctx);
        mbedtls_sha256_starts(&p_->ctx, 0);
    }

    sha256_stream::~sha256_stream()
    {
        mbedtls_sha256_free(&p_->ctx);
    }

    void sha256_stream::update(const void* data, size_t n)
    {
        mbedtls_sha256_update(&p_->ctx, static_cast<const unsigned char*>(data), n);
    }

    std::string sha256_stream::hex()
    {
        unsigned char h[32];
        mbedtls_sha256_finish(&p_->ctx, h);
        static const char* digits = "0123456789abcdef";
        std::string out;
        for (unsigned char b : h)
        {
            out += digits[b >> 4];
            out += digits[b & 15];
        }
        return out;
    }

    std::string sha256_hex(const std::string& data)
    {
        sha256_stream s;
        s.update(data.data(), data.size());
        return s.hex();
    }

    std::string sha256_file(const std::string& path, const std::function<void(uint64_t)>& progress,
        const std::atomic<bool>* cancel)
    {
#if defined(_WIN32)
        std::ifstream in(widen(path).c_str(), std::ios::binary);
#else
        std::ifstream in(path, std::ios::binary);
#endif
        if (!in) return std::string();
        sha256_stream s;
        std::vector<char> buf(1 << 20);
        uint64_t total = 0;
        while (in)
        {
            if (cancel && cancel->load()) return std::string();
            in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
            const std::streamsize n = in.gcount();
            if (n <= 0) break;
            s.update(buf.data(), static_cast<size_t>(n));
            total += static_cast<uint64_t>(n);
            if (progress) progress(total);
        }
        return s.hex();
    }

    std::string base64_encode(const std::string& data)
    {
        size_t n = 0;
        mbedtls_base64_encode(nullptr, 0, &n, reinterpret_cast<const unsigned char*>(data.data()), data.size());
        std::vector<unsigned char> buf(n + 1);
        if (mbedtls_base64_encode(buf.data(), buf.size(), &n, reinterpret_cast<const unsigned char*>(data.data()),
                data.size()) != 0)
            return std::string();
        return std::string(reinterpret_cast<const char*>(buf.data()), n);
    }

    std::string base64_decode(const std::string& text)
    {
        std::string clean;
        for (char c : text)
            if (c != '\n' && c != '\r' && c != ' ' && c != '\t') clean += c;
        size_t n = 0;
        mbedtls_base64_decode(nullptr, 0, &n, reinterpret_cast<const unsigned char*>(clean.data()), clean.size());
        std::vector<unsigned char> buf(n + 1);
        if (mbedtls_base64_decode(buf.data(), buf.size(), &n, reinterpret_cast<const unsigned char*>(clean.data()),
                clean.size()) != 0)
            return std::string();
        return std::string(reinterpret_cast<const char*>(buf.data()), n);
    }

    bool verify_signature(const std::string& data, const std::string& signature_base64,
        const std::string& public_key_pem)
    {
        psa_crypto_init();
        const std::string sig = base64_decode(signature_base64);
        if (sig.empty()) return false;
        unsigned char hash[32];
        mbedtls_sha256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash, 0);
        mbedtls_pk_context pk;
        mbedtls_pk_init(&pk);
        bool ok = mbedtls_pk_parse_public_key(&pk, reinterpret_cast<const unsigned char*>(public_key_pem.c_str()),
            public_key_pem.size() + 1) == 0;
        ok = ok && mbedtls_pk_verify(&pk, MBEDTLS_MD_SHA256, hash, sizeof(hash),
            reinterpret_cast<const unsigned char*>(sig.data()), sig.size()) == 0;
        mbedtls_pk_free(&pk);
        return ok;
    }

    std::string sign_data(const std::string& data, const std::string& private_key_pem, std::string& error)
    {
        psa_crypto_init();
        mbedtls_entropy_context entropy;
        mbedtls_ctr_drbg_context drbg;
        mbedtls_entropy_init(&entropy);
        mbedtls_ctr_drbg_init(&drbg);
        mbedtls_pk_context pk;
        mbedtls_pk_init(&pk);
        std::string out;
        do
        {
            if (mbedtls_ctr_drbg_seed(&drbg, mbedtls_entropy_func, &entropy, nullptr, 0) != 0)
            {
                error = "no entropy source";
                break;
            }
            if (mbedtls_pk_parse_key(&pk, reinterpret_cast<const unsigned char*>(private_key_pem.c_str()),
                    private_key_pem.size() + 1, nullptr, 0, mbedtls_ctr_drbg_random, &drbg) != 0)
            {
                error = "the private key cannot be read";
                break;
            }
            unsigned char hash[32];
            mbedtls_sha256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash, 0);
            unsigned char sig[MBEDTLS_PK_SIGNATURE_MAX_SIZE];
            size_t n = 0;
            if (mbedtls_pk_sign(&pk, MBEDTLS_MD_SHA256, hash, sizeof(hash), sig, sizeof(sig), &n,
                    mbedtls_ctr_drbg_random, &drbg) != 0)
            {
                error = "signature failed";
                break;
            }
            out = base64_encode(std::string(reinterpret_cast<const char*>(sig), n));
        } while (false);
        mbedtls_pk_free(&pk);
        mbedtls_ctr_drbg_free(&drbg);
        mbedtls_entropy_free(&entropy);
        return out;
    }

}
