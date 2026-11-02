// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "vault.h"

#include <dlib/crypto/crypto_utils.h>
#include <dlib/crypto/secure_random.h>
#include <dlib/crypto/sha256.h>

#include <array>
#include <cstring>

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

    /*
        A blob is "PSDV1", the IV (16 bytes), the data encrypted and the HMAC (32 bytes) of the
        identifier and of all that precedes it.  It is built and read in one buffer, so that
        the data never stands in memory more than twice.
    */
    std::string vault::put(const std::string& data)
    {
        const std::string id = dlib::to_hex(dlib::generate_secure_random_bytes(16));
        const std::string iv = dlib::generate_secure_random_bytes(16);
        std::string file;
        file.reserve(21 + data.size() + 32);
        file.append(magic, 5);
        file.append(iv);
        file.resize(21 + data.size());
        if (!data.empty()) aes256_ctr_into(enc_, iv.data(), data.data(), &file[21], data.size());
        dlib::hmac_sha256 mac(mac_);
        mac.update(id);
        mac.update(file.data(), file.size());
        const std::array<unsigned char, 32> tag = mac.finalize();
        file.append(reinterpret_cast<const char*>(tag.data()), tag.size());
        const std::string folder = join_path(dir_, id.substr(0, 2));
        if (!dir_exists(folder) && !make_dirs(folder)) return std::string();
        if (!write_file_atomic(path_of(id), file)) return std::string();
        return id;
    }

    bool vault::get(const std::string& id, std::string& data) const
    {
        data.clear();
        if (!valid_id(id)) return false;
        std::string file;
        if (!read_file(path_of(id), file, 4ull << 30) || file.size() < 5 + 16 + 32 || file.compare(0, 5, magic, 5) != 0)
            return false;
        const size_t sealed = file.size() - 21 - 32;
        dlib::hmac_sha256 mac(mac_);
        mac.update(id);
        mac.update(file.data(), 21 + sealed);
        const std::array<unsigned char, 32> expected = mac.finalize();
        if (!dlib::constant_time_equal(expected.data(), file.data() + 21 + sealed, 32)) return false;
        // Decrypted in place, then moved to the start of the buffer, which becomes the data.
        char iv[16];
        std::memcpy(iv, file.data() + 5, 16);
        if (sealed > 0) aes256_ctr_into(enc_, iv, file.data() + 21, &file[21], sealed);
        file.erase(0, 21);
        file.resize(sealed);
        data.swap(file);
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

    std::vector<std::string> vault::list_and_clean()
    {
        std::vector<std::string> ids;
        for (const std::string& sub : list_dir(dir_))
        {
            const std::string folder = join_path(dir_, sub);
            if (sub.size() != 2 || !dir_exists(folder)) continue;
            for (const std::string& name : list_dir(folder))
            {
                const std::string id = name.substr(0, 32);
                if (name.size() == 36 && name.compare(32, 4, ".bin") == 0 && valid_id(id) && id.compare(0, 2, sub) == 0)
                    ids.push_back(id);
                // A blob that write_file_atomic() did not finish: "<id>.bin.tmp-<random>".
                else if (name.size() > 41 && name.compare(32, 9, ".bin.tmp-") == 0 && valid_id(id))
                    remove_file(join_path(folder, name));
            }
        }
        return ids;
    }
}
