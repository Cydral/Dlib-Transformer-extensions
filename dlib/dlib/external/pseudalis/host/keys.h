// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_KEYS_H_
#define PSEUDALIS_HOST_KEYS_H_

#include <string>

/*
    The keys of Pseudalis.  One master key of 32 random bytes, made at the first start of the
    service, gives with HKDF-SHA-256 the key of the pseudonymization (the tags of the values)
    and the keys of the vault (the originals and the pictures, AES-256-CTR and HMAC-SHA-256).

    On Windows the master key is kept in keys\master.key of the data folder, encrypted with
    DPAPI for the account of the service (LocalSystem), in a folder that only SYSTEM and the
    administrators open: a copy of the file serves no other account and no other computer.
    Elsewhere it lies in clear in a folder that only its owner opens.  Its export, protected
    by a passphrase (PBKDF2-SHA-256, 600,000 iterations, AES-256-CTR and HMAC-SHA-256), is the
    way to keep a copy or to move the installation to another computer.
*/

namespace pseudalis
{
    struct key_set
    {
        std::string pseudonymization;       // 32 bytes
        std::string vault_encryption;       // 32 bytes
        std::string vault_authentication;   // 32 bytes
    };

    key_set derive_keys(const std::string& master);

    /*!
        ensures
            - Reads the master key of keys_dir, or creates it when the folder holds none
              (created is then true).  Returns false with error when the key cannot be read
              (another account, another computer, a damaged file) or written.
    !*/
    bool load_master_key(const std::string& keys_dir, std::string& master, bool& created, std::string& error);
    bool save_master_key(const std::string& keys_dir, const std::string& master, std::string& error);

    // The export of the master key: a JSON document protected by passphrase (12 characters at least).
    std::string export_master_key(const std::string& master, const std::string& passphrase);
    bool import_master_key(const std::string& exported, const std::string& passphrase, std::string& master,
        std::string& error);

    /*!
        ensures
            - Makes path a folder that only SYSTEM and the administrators open, on Windows,
              and only its owner elsewhere: created so when missing, its permissions
              replaced otherwise.
    !*/
    bool private_folder(const std::string& path, std::string& error);

    // Erases a secret, then empties it.
    void wipe(std::string& s);

    // AES-256-CTR with a counter block iv of 16 bytes, encryption and decryption alike.
    std::string aes256_ctr(const std::string& key, const std::string& iv, const std::string& data);
    std::string hmac_sha256(const std::string& key, const std::string& data);
}

#endif // PSEUDALIS_HOST_KEYS_H_
