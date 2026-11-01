// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
/*
    albireo-sign: the signatures of the documents the service downloads.

        albireo-sign keygen <private key> <public key>
            Creates a key pair (ECDSA P-256) in two new PEM files: a file in place is never
            replaced.  Only the account that runs the program may read the private key.  The
            public key goes into host/crypto.cpp (feed_public_key_pem); the private key stays
            with the publisher.

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

#if defined(_WIN32)
#include <windows.h>
#include <sddl.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif

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

#if defined(_WIN32)
    // The security identifier of the account that runs this program, as text (S-1-5-21-...).
    std::wstring current_user_sid()
    {
        HANDLE token = nullptr;
        if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return std::wstring();
        DWORD size = 0;
        GetTokenInformation(token, TokenUser, nullptr, 0, &size);
        std::vector<unsigned char> buffer(size);
        std::wstring sid;
        LPWSTR text = nullptr;
        if (size > 0 && GetTokenInformation(token, TokenUser, buffer.data(), size, &size)
            && ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER*>(buffer.data())->User.Sid, &text))
        {
            sid = text;
            LocalFree(text);
        }
        CloseHandle(token);
        return sid;
    }
#endif

    /*
        Creates the file path, which must not exist, with data.  With owner_only, the account
        that runs this program alone reads and writes it: the access list names that account
        only on Windows, and the mode is 0600 elsewhere.  A file the write leaves incomplete
        is removed.
    */
    bool create_new_file(const std::string& path, const std::string& data, bool owner_only, std::string& error)
    {
        const std::string exists = path + " exists already: a key in place is never replaced";
#if defined(_WIN32)
        SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, FALSE };
        PSECURITY_DESCRIPTOR sd = nullptr;
        if (owner_only)
        {
            const std::wstring sid = current_user_sid();
            const std::wstring sddl = L"D:P(A;;FA;;;" + sid + L")";
            if (sid.empty()
                || !ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &sd, nullptr))
            {
                error = "cannot make the access list of " + path + " for the account of this program";
                return false;
            }
            sa.lpSecurityDescriptor = sd;
        }
        const std::wstring w = albireo::widen(path);
        HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, 0, &sa, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        const DWORD create_error = GetLastError();
        if (sd) LocalFree(sd);
        if (h == INVALID_HANDLE_VALUE)
        {
            error = create_error == ERROR_FILE_EXISTS || create_error == ERROR_ALREADY_EXISTS ? exists
                : "cannot create " + path;
            return false;
        }
        DWORD written = 0;
        const bool ok = WriteFile(h, data.data(), static_cast<DWORD>(data.size()), &written, nullptr)
            && written == data.size() && FlushFileBuffers(h);
        CloseHandle(h);
        if (!ok) DeleteFileW(w.c_str());
#else
        // O_EXCL refuses any entry of that name, a link included.
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, owner_only ? 0600 : 0644);
        if (fd < 0)
        {
            error = errno == EEXIST ? exists : "cannot create " + path;
            return false;
        }
        size_t done = 0;
        while (done < data.size())
        {
            const ssize_t w = ::write(fd, data.data() + done, data.size() - done);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) break;
            done += static_cast<size_t>(w);
        }
        const bool ok = done == data.size() && fsync(fd) == 0;
        ::close(fd);
        if (!ok) ::unlink(path.c_str());
#endif
        if (!ok) error = "cannot write " + path;
        return ok;
    }

    int keygen(const std::string& private_path, const std::string& public_path)
    {
        // The services hold the public half of the key of the feed: a key in place is never replaced.
        for (const std::string& p : { private_path, public_path })
        {
            if (albireo::file_exists(p) || albireo::dir_exists(p) || albireo::is_link(p))
            {
                std::cerr << p << " exists already: a key in place is never replaced\n";
                return 1;
            }
        }
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
            std::string error;
            if (!create_new_file(private_path, reinterpret_cast<const char*>(private_pem), true, error))
            {
                std::cerr << error << "\n";
            }
            else if (!create_new_file(public_path, reinterpret_cast<const char*>(public_pem), false, error))
            {
                std::cerr << error << "\n";
                // The private key goes with the pair that could not be written whole.
                albireo::remove_file(private_path);
            }
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
