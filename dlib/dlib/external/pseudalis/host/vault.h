// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_VAULT_H_
#define PSEUDALIS_HOST_VAULT_H_

#include <string>

namespace pseudalis
{
    class vault
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The encrypted files of the service: the originals of the files it
                protected, their pictures, and the content of the files of the drive.  Each
                blob lies in <dir>/<two digits>/<identifier>.bin, its identifier 32 random
                hexadecimal digits, encrypted with AES-256-CTR under a random counter block
                and authenticated with HMAC-SHA-256 over its identifier, its counter block
                and its ciphertext: a blob altered, or moved under another identifier, does
                not open.  The folder is one of SYSTEM and the administrators.

            THREAD SAFETY
                The methods may be called from several threads at once.
        !*/
    public:
        vault(const std::string& dir, const std::string& encryption_key, const std::string& authentication_key);
        ~vault();

        // Stores data; returns its identifier, or an empty string when it cannot be written.
        std::string put(const std::string& data);
        // Reads a blob; false when it is missing, altered or unreadable.
        bool get(const std::string& id, std::string& data) const;
        bool remove(const std::string& id);
        bool exists(const std::string& id) const;
        static bool valid_id(const std::string& id);

    private:
        std::string path_of(const std::string& id) const;
        std::string dir_;
        std::string enc_, mac_;
    };
}

#endif // PSEUDALIS_HOST_VAULT_H_
