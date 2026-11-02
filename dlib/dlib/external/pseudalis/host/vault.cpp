// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "vault.h"

#include <dlib/crypto/crypto_utils.h>
#include <dlib/crypto/secure_random.h>

#include "../common/platform.h"
#include "keys.h"

namespace pseudalis
{
    namespace
    {
        const char magic[] = "PSDV1";
    }

    vault::vault(const std::string& dir, const std::string& encryption_key, const std::string& authentication_key)
        : dir_(dir), enc_(encryption_key), mac_(authentication_key)
    {
    }

    vault::~vault()
    {
        wipe(enc_);
        wipe(mac_);
    }

    bool vault::valid_id(const std::string& id)
    {
        if (id.size() != 32) return false;
        for (char c : id)
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        return true;
    }

    std::string vault::path_of(const std::string& id) const
    {
        return join_path(join_path(dir_, id.substr(0, 2)), id + ".bin");
    }

    std::string vault::put(const std::string& data)
    {
        const std::string id = dlib::to_hex(dlib::generate_secure_random_bytes(16));
        const std::string iv = dlib::generate_secure_random_bytes(16);
        const std::string sealed = aes256_ctr(enc_, iv, data);
        const std::string head = std::string(magic, 5) + iv;
        const std::string tag = hmac_sha256(mac_, id + head + sealed);
        const std::string folder = join_path(dir_, id.substr(0, 2));
        if (!dir_exists(folder) && !make_dirs(folder)) return std::string();
        if (!write_file_atomic(path_of(id), head + sealed + tag)) return std::string();
        return id;
    }

    bool vault::get(const std::string& id, std::string& data) const
    {
        data.clear();
        if (!valid_id(id)) return false;
        std::string file;
        if (!read_file(path_of(id), file, 4ull << 30) || file.size() < 5 + 16 + 32 || file.compare(0, 5, magic, 5) != 0)
            return false;
        const std::string head = file.substr(0, 21);
        const std::string sealed = file.substr(21, file.size() - 21 - 32);
        const std::string tag = file.substr(file.size() - 32);
        const std::string expected = hmac_sha256(mac_, id + head + sealed);
        if (!dlib::constant_time_equal(expected.data(), tag.data(), 32)) return false;
        data = aes256_ctr(enc_, head.substr(5), sealed);
        return true;
    }

    bool vault::remove(const std::string& id)
    {
        return valid_id(id) && remove_file(path_of(id));
    }

    bool vault::exists(const std::string& id) const
    {
        return valid_id(id) && file_exists(path_of(id));
    }
}
