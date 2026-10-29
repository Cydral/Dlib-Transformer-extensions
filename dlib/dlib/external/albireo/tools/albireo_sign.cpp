// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    albireo-sign: the signatures of the documents the service downloads.

        albireo-sign keygen <private key> <public key>
            Creates a key pair (ECDSA P-256) in PEM files.  The public key goes into
            host/crypto.cpp (feed_public_key_pem); the private key stays with the publisher.

        albireo-sign sign <private key> <file> [<file>...]
            Writes <file>.sig, the signature of each file in base64, as the service expects
            it next to the update feed and the model catalog.

        albireo-sign verify <file> [<public key>]
            Checks <file>.sig against the given public key, or against the key the
            programs were built with.

        albireo-sign hash <file> [<file>...]
            Prints the size and the SHA-256 of each file, the values of the catalog and of
            the feed.
*/

#include <iostream>
#include <string>
#include <vector>

#include <mbedtls/ctr_drbg.h>
#include <mbedtls/ecp.h>
#include <mbedtls/entropy.h>
#include <mbedtls/pk.h>
#include <psa/crypto.h>

#include "../common/platform.h"
#include "../host/crypto.h"

namespace
{
    bool read_text(const std::string& path, std::string& out)
    {
        return albireo::read_file(path, out);
    }

    int usage()
    {
        std::cerr << "usage: albireo-sign keygen <private key> <public key>\n"
                     "       albireo-sign sign <private key> <file> [<file>...]\n"
                     "       albireo-sign verify <file> [<public key>]\n"
                     "       albireo-sign hash <file> [<file>...]\n";
        return 2;
    }

    int keygen(const std::string& private_path, const std::string& public_path)
    {
        psa_crypto_init();
        mbedtls_entropy_context entropy;
        mbedtls_ctr_drbg_context drbg;
        mbedtls_pk_context key;
        mbedtls_entropy_init(&entropy);
        mbedtls_ctr_drbg_init(&drbg);
        mbedtls_pk_init(&key);
        int code = 1;
        unsigned char private_pem[1024] = { 0 };
        unsigned char public_pem[512] = { 0 };
        if (mbedtls_ctr_drbg_seed(&drbg, mbedtls_entropy_func, &entropy, nullptr, 0) == 0
            && mbedtls_pk_setup(&key, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY)) == 0
            && mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(key), mbedtls_ctr_drbg_random, &drbg) == 0
            && mbedtls_pk_write_key_pem(&key, private_pem, sizeof(private_pem)) == 0
            && mbedtls_pk_write_pubkey_pem(&key, public_pem, sizeof(public_pem)) == 0)
        {
            // The private key is readable by its owner only.
            if (!albireo::write_file_restricted(private_path, reinterpret_cast<const char*>(private_pem), false))
                std::cerr << "cannot write " << private_path << "\n";
            else if (!albireo::write_file_atomic(public_path, reinterpret_cast<const char*>(public_pem)))
                std::cerr << "cannot write " << public_path << "\n";
            else
            {
                std::cout << reinterpret_cast<const char*>(public_pem);
                code = 0;
            }
        }
        else
        {
            std::cerr << "the key pair could not be created\n";
        }
        mbedtls_pk_free(&key);
        mbedtls_ctr_drbg_free(&drbg);
        mbedtls_entropy_free(&entropy);
        return code;
    }
}

int main(int argc, char** argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3) return usage();
    const std::string& command = args[1];

    if (command == "keygen" && args.size() == 4) return keygen(args[2], args[3]);

    if (command == "sign" && args.size() >= 4)
    {
        std::string key;
        if (!read_text(args[2], key))
        {
            std::cerr << "cannot read " << args[2] << "\n";
            return 1;
        }
        for (size_t i = 3; i < args.size(); ++i)
        {
            std::string data, error;
            if (!read_text(args[i], data))
            {
                std::cerr << "cannot read " << args[i] << "\n";
                return 1;
            }
            const std::string sig = albireo::sign_data(data, key, error);
            if (sig.empty() || !albireo::write_file_atomic(args[i] + ".sig", sig + "\n"))
            {
                std::cerr << args[i] << ": " << (error.empty() ? "cannot write the signature" : error) << "\n";
                return 1;
            }
            std::cout << args[i] << ".sig\n";
        }
        return 0;
    }

    if (command == "verify" && (args.size() == 3 || args.size() == 4))
    {
        std::string data, sig, key = albireo::feed_public_key_pem;
        if (!read_text(args[2], data) || !read_text(args[2] + ".sig", sig))
        {
            std::cerr << "cannot read " << args[2] << " or its signature\n";
            return 1;
        }
        if (args.size() == 4 && !read_text(args[3], key))
        {
            std::cerr << "cannot read " << args[3] << "\n";
            return 1;
        }
        const bool ok = albireo::verify_signature(data, sig, key);
        std::cout << args[2] << (ok ? ": valid signature\n" : ": INVALID signature\n");
        return ok ? 0 : 1;
    }

    if (command == "hash")
    {
        for (size_t i = 2; i < args.size(); ++i)
        {
            const std::string sha = albireo::sha256_file(args[i]);
            if (sha.empty())
            {
                std::cerr << "cannot read " << args[i] << "\n";
                return 1;
            }
            std::cout << albireo::file_size(args[i]) << " " << sha << " " << args[i] << "\n";
        }
        return 0;
    }
    return usage();
}
