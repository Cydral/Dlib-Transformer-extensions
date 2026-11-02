// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "updater.h"

#include <cstdlib>
#include <exception>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "../common/log.h"
#include "../common/platform.h"
#include "../common/version.h"
#include "crypto.h"
#include "http_fetch.h"
#include "public_link.h"

namespace albireo
{

    namespace
    {
        // A version "1.2.3", which names files of the updates folder.
        bool plain_version(const std::string& v)
        {
            if (v.empty() || v.size() > 32 || v[0] == '.') return false;
            for (char c : v)
                if (!((c >= '0' && c <= '9') || c == '.')) return false;
            return true;
        }

        // A name of a file within a folder: no separator, stream, wildcard or name "." and "..".
        bool plain_name(const std::string& name)
        {
            return !name.empty() && name.size() <= 255 && name != "." && name != ".."
                && name.find_first_of("/\\:*?\"<>|") == std::string::npos;
        }

        // An error in the state or in an answer: its English text, its code and its values.
        void put_error(json& out, const user_error& e)
        {
            out["error"] = e.text;
            out["code"] = e.code;
            out["args"] = e.args.is_object() ? e.args : json::object();
        }

        void clear_error(json& out)
        {
            out.erase("error");
            out.erase("code");
            out.erase("args");
        }

        user_error stopping_error()
        {
            return user_error().set("the service stops", "stopping");
        }
    }

    updater::updater(const std::string& updates_dir)
        : dir_(updates_dir)
    {
        status_ = json::object();
        status_["current"] = ALBIREO_VERSION;
        status_["state"] = "idle";
        // The folder by its own path, at which the files written there check where they lie (new_file).
        make_dirs(dir_);
        const std::string resolved = final_path(dir_);
        if (!resolved.empty()) dir_ = resolved;
        // An update downloaded by a previous run, checked again against its signed feed.
        std::string text;
        if (read_file(join_path(dir_, "staged.json"), text, 1u << 20))
        {
            const json staged = json::parse_or_null(text);
            const std::string version = staged["version"].as_string();
            user_error error;
            if (plain_version(version) && compare_versions(version, ALBIREO_VERSION) > 0)
            {
                if (check_staged(staged, error))
                {
                    staged_ = staged;
                    status_["state"] = "ready";
                    status_["latest"] = staged["version"];
                    status_["notes"] = staged["notes"];
                    status_["file"] = staged["file"];
                }
                else
                {
                    log_warning("updates: " + error.text + ", the update waits for the next check");
                }
            }
        }
    }

    updater::~updater()
    {
        cancel();
        if (worker_.joinable()) worker_.join();
    }

    void updater::check_async()
    {
        std::lock_guard<std::mutex> start(worker_m_);
        if (cancelled_ || busy_.exchange(true)) return;
        if (worker_.joinable()) worker_.join();
        auto failed = [this](const std::string& why) {
            std::lock_guard<std::mutex> lock(m_);
            status_["state"] = "error";
            put_error(status_, user_error().set(why, "update_check").arg("e", why));
        };
        try
        {
            // No exception leaves the thread of a check.
            worker_ = std::thread([this, failed]() {
                try
                {
                    check();
                }
                catch (const std::exception& e)
                {
                    failed(e.what());
                }
                catch (...)
                {
                    failed("the check of the updates fails");
                }
                busy_ = false;
            });
        }
        catch (const std::exception& e)
        {
            // A thread the system refuses: the check fails, and the next one may start.
            failed(std::string("the check of the updates cannot start: ") + e.what());
            busy_ = false;
        }
    }

    void updater::cancel()
    {
        cancelled_ = true;
    }

    std::string updater::installer_path(const std::string& version) const
    {
        return join_path(dir_, "albireo-setup-" + version +
#if defined(_WIN32)
            ".exe");
#else
            ".bin");
#endif
    }

    std::string updater::feed_path(const std::string& version) const
    {
        return join_path(dir_, "feed-" + version + ".json");
    }

    updater::stage_result updater::stage(const std::string& version, const std::string& sha, const json& notes,
        const std::string& feed, const std::string& signature, user_error& error)
    {
        const std::string file = installer_path(version);
        json staged = json::object();
        staged["version"] = version;
        staged["file"] = file;
        staged["sha256"] = sha;
        staged["notes"] = notes;
        std::lock_guard<std::mutex> lock(m_);
        // Of a check and an import that end at once, the newer version stays staged.
        const json& current = staged_;
        if (current.is_object() && compare_versions(version, current["version"].as_string()) < 0)
            return stage_result::older;
        // The signed feed first, then staged.json, which names it.
        make_dirs(dir_);
        const std::string copy = feed_path(version);
        if (!write_file_atomic(copy + ".sig", signature) || !write_file_atomic(copy, feed)
            || !write_file_atomic(join_path(dir_, "staged.json"), staged.dump()))
        {
            error.set("the update of version " + version + " cannot be recorded in " + dir_, "update_record")
                .arg("version", version);
            return stage_result::failed;
        }
        staged_ = staged;
        status_["latest"] = version;
        status_["notes"] = notes;
        status_["file"] = file;
        status_["state"] = "ready";
        clear_error(status_);
        return stage_result::staged;
    }

    bool updater::staged_feed(const std::string& version, uint64_t& size, std::string& sha, user_error& error) const
    {
        const std::string path = feed_path(version);
        std::string body, sig;
        if (!read_file(path, body, 1u << 20) || !read_file(path + ".sig", sig, 4096))
        {
            error.set("no signed feed beside the installer of version " + version, "update_feed_missing")
                .arg("version", version);
            return false;
        }
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            error.set("the signature of " + path + " is not valid", "update_signature").arg("file", path);
            return false;
        }
        const json feed = json::parse_or_null(body);
        const json& stable = feed["channels"]["stable"];
        sha = stable["installer"]["sha256"].as_string();
        size = static_cast<uint64_t>(stable["installer"]["size"].as_int(0));
        if (stable["version"].as_string() != version || sha.size() != 64 || size == 0)
        {
            error.set(path + " does not describe the installer of version " + version, "update_feed_mismatch")
                .arg("file", path).arg("version", version);
            return false;
        }
        return true;
    }

    bool updater::check_staged(const json& staged, user_error& error) const
    {
        const std::string version = staged["version"].as_string();
        const std::string file = staged["file"].as_string();
        // The installer of the version, in the updates folder by its own path.
        const std::string expected = installer_path(version);
        if (file_name(file) != file_name(expected) || final_path(parent_path(file)) != dir_)
        {
            error.set("the staged installer " + file + " lies outside " + dir_, "update_outside").arg("file", file);
            return false;
        }
        uint64_t size = 0;
        std::string sha;
        if (!staged_feed(version, size, sha, error)) return false;
        if (sha != staged["sha256"].as_string() || file_size(expected) != size || sha256_file(expected) != sha)
        {
            error.set("the installer of version " + version + " does not match its signed feed",
                "update_installer_mismatch").arg("version", version);
            return false;
        }
        return true;
    }

    void updater::drop_staged(const user_error& error)
    {
        log_error("updates: " + error.text + ", the update waits for the next check");
        remove_file(join_path(dir_, "staged.json"));
        std::lock_guard<std::mutex> lock(m_);
        staged_ = json();
        status_["state"] = "error";
        put_error(status_, error);
        status_.erase("file");
    }

    json updater::status() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return status_;
    }

    void updater::check()
    {
        auto set = [&](const std::string& key, const json& v) {
            std::lock_guard<std::mutex> lock(m_);
            status_[key] = v;
        };
        // A check ended by the service, which stops or goes offline, leaves no error, and
        // an update staged earlier, from a medium say, stays ready whatever the check finds.
        auto ended = [&]() { return cancelled_ || http_cancelled(); };
        auto settle = [&](const std::string& state, const user_error& error) {
            std::lock_guard<std::mutex> lock(m_);
            const json& staged = staged_;
            if (staged.is_object())
            {
                status_["state"] = "ready";
                status_["latest"] = staged["version"];
                status_["notes"] = staged["notes"];
                status_["file"] = staged["file"];
            }
            else
            {
                status_["state"] = state;
            }
            if (error.text.empty()) clear_error(status_);
            else put_error(status_, error);
        };
        auto fail = [&](const user_error& error) {
            if (ended()) settle("idle", user_error());
            else settle("error", error);
        };
        if (ended()) return;
        set("state", "checking");
        set("checked_at", static_cast<long long>(now_unix()));
        std::string body, sig, error;
        // Where the installer comes from, for the log: empty when it was downloaded before.
        std::string origin;
        // The feed and its signature come from the folder dist of the publication space alone.
        auto read = [&](const std::string& path, std::string& into, size_t max_bytes) {
            link_status status = link_status::ok;
            if (public_link_get(path, into, error, max_bytes, &status)) return true;
            // A file the space does not hold yet has its own code, which names it.
            if (status == link_status::missing) fail(user_error().set(error, "update_not_published").arg("file", path));
            else fail(user_error().set(error, "update_download").arg("e", error));
            if (!ended()) log_warning("updates: " + error);
            return false;
        };
        if (!read("dist/feed.json", body, 1u << 20) || !read("dist/feed.json.sig", sig, 4096)) return;
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            fail(user_error().set("the signature of the update feed is not valid", "update_signature")
                .arg("file", "dist/feed.json"));
            log_error("updates: the signature of the feed of the publication space is not valid, feed ignored");
            return;
        }
        log_info("updates: feed read from the publication space of Albireo");
        const json feed = json::parse_or_null(body);
        const json& stable = feed["channels"]["stable"];
        const std::string latest = stable["version"].as_string();
        const json notes = stable["notes"];
        // A signed feed that names no version announces nothing: an error, never an installation up to date.
        if (!plain_version(latest))
        {
            fail(user_error().set("the update feed names no version", "update_feed_invalid"));
            log_error("updates: the update feed names no version");
            return;
        }
        std::string staged_version;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& staged = staged_;
            staged_version = staged["version"].as_string();
        }
        if (compare_versions(latest, ALBIREO_VERSION) <= 0
            || (!staged_version.empty() && compare_versions(latest, staged_version) <= 0))
        {
            {
                std::lock_guard<std::mutex> lock(m_);
                status_["latest"] = latest;
                status_["notes"] = notes;
            }
            settle("up_to_date", user_error());
            return;
        }
        set("latest", latest);
        set("notes", notes);
        const json& inst = stable["installer"];
        // The installer lies in the folder dist of the publication space, under the name the feed gives it ("file").
        const std::string in_link = inst["file"].as_string();
        const std::string sha = inst["sha256"].as_string();
        // A size the feed gives as negative, or of another type, makes it incomplete.
        const long long given = inst["size"].as_int(0);
        const uint64_t size = given > 0 ? static_cast<uint64_t>(given) : 0;
        if (in_link.empty() || !plain_name(in_link) || sha.size() != 64 || size == 0)
        {
            fail(user_error().set(in_link.empty() ? "the update feed names no installer in the publication space of "
                "Albireo" : "the update feed is incomplete", "update_feed_invalid"));
            return;
        }
        make_dirs(dir_);
        const std::string file = installer_path(latest);
        if (!(file_exists(file) && sha256_file(file, nullptr, &cancelled_) == sha))
        {
            set("state", "downloading");
            set("total", static_cast<unsigned long long>(size));
            set("received", 0ull);
            const user_error unrecorded = user_error().set("the update of version " + latest
                + " cannot be recorded in " + dir_, "update_record").arg("version", latest);
            const std::string tmp = file + ".partial";
            /*
                One download of the installer from address into the partial file, checked against
                the feed; label names the file in the messages, which never name its address.
            */
            auto download = [&](const std::string& address, const std::string& label, user_error& why) {
                // The partial file of an earlier attempt goes: the new one must not exist yet (new_file).
                remove_file(tmp);
                set("received", 0ull);
                new_file out;
                const bool created = out.create(tmp);
                uint64_t got = 0;
                bool too_large = false, unwritten = false;
                std::string e;
                const bool ok = created && http_get_range(address, 0, -1, [&](const char* d, size_t n) {
                    if (got + n > size)
                    {
                        too_large = true;
                        return false;
                    }
                    if (!out.write(d, n))
                    {
                        unwritten = true;
                        return false;
                    }
                    got += n;
                    set("received", static_cast<unsigned long long>(got));
                    return !cancelled_.load();
                }, e);
                const bool closed = out.close();
                e = label + " cannot be downloaded: " + e;
                why = user_error();
                if (!created || unwritten || (ok && !closed)) why = unrecorded;
                else if (!ok && !too_large) why.set(e, "update_download").arg("e", e);
                else if (too_large || got != size || sha256_file(tmp, nullptr, &cancelled_) != sha)
                    why.set("the downloaded installer does not match the feed", "update_installer_mismatch")
                        .arg("version", latest);
                else if (!rename_file(tmp, file)) why = unrecorded;
                if (!why.code.empty() && created) remove_file(tmp);
                return why.code.empty();
            };
            user_error why;
            bool done = false;
            /*
                The copy of the installer in the publication space: an address of the cache may
                have expired, and a file the listing of the cache lacks may have been published
                since, so that the second attempt asks the listing and the address again.
            */
            const std::string path = "dist/" + in_link;
            const std::string label = "the copy of " + path + " in the publication space of Albireo";
            for (int attempt = 0; attempt < 2 && !done && !ended(); ++attempt)
            {
                std::string address, e;
                uint64_t listed = 0;
                link_status status = link_status::ok;
                if (!public_link_file(path, address, listed, e, attempt > 0, &status))
                {
                    if (status == link_status::missing) why.set(e, "update_not_published").arg("file", path);
                    else why.set(e, "update_download").arg("e", e);
                    if (status == link_status::missing && attempt == 0) continue;
                    break;
                }
                if (listed != size)
                {
                    e = label + " does not have the size the feed gives";
                    why.set(e, "update_download").arg("e", e);
                    continue;
                }
                done = download(address, label, why);
                // A file that cannot be written ends the attempts.
                if (!done && why.code == "update_record") break;
            }
            if (!done)
            {
                if (why.code.empty()) why.set("the update feed is incomplete", "update_feed_invalid");
                fail(why);
                if (!ended()) log_error("updates: download of " + latest + " failed: " + why.text);
                return;
            }
            origin = " from the publication space of Albireo";
        }
        user_error staging;
        switch (stage(latest, sha, notes, body, sig, staging))
        {
        case stage_result::staged:
            log_info("updates: version " + latest + " downloaded" + origin + ", applied at the next restart");
            break;
        case stage_result::older:
            // A newer version, imported from a medium meanwhile, stays the one installed.
            settle("up_to_date", user_error());
            break;
        case stage_result::failed:
            fail(staging);
            log_error("updates: " + staging.text);
            break;
        }
    }

    json updater::import_offline(const std::string& folder, const std::string& base)
    {
        json r = json::object();
        std::string body, sig;
        if (!read_file_link_free(join_path(folder, "feed.json"), base, body, 1u << 20)
            || !read_file_link_free(join_path(folder, "feed.json.sig"), base, sig, 4096))
        {
            put_error(r, user_error().set("no signed update feed (feed.json and feed.json.sig) in " + folder,
                "update_feed_missing_folder").arg("folder", folder));
            return r;
        }
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            const std::string file = join_path(folder, "feed.json");
            put_error(r, user_error().set("the signature of " + file + " is not valid", "update_signature")
                .arg("file", file));
            log_error("updates: " + r["error"].as_string());
            return r;
        }
        const json stable = json::parse_or_null(body)["channels"]["stable"];
        const std::string latest = stable["version"].as_string();
        r["version"] = latest;
        if (!plain_version(latest))
        {
            put_error(r, user_error().set("the update feed names no version", "update_feed_invalid"));
            return r;
        }
        std::string staged_version;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& staged = staged_;
            staged_version = staged["version"].as_string();
        }
        if (compare_versions(latest, ALBIREO_VERSION) <= 0)
        {
            r["state"] = "up_to_date";
            return r;
        }
        // An older medium never replaces the update staged from a newer one, which stays
        // the one installed at the next start.
        if (!staged_version.empty() && compare_versions(latest, staged_version) <= 0)
        {
            r["state"] = "ready";
            r["version"] = staged_version;
            return r;
        }
        const json& inst = stable["installer"];
        const std::string sha = inst["sha256"].as_string();
        const uint64_t size = static_cast<uint64_t>(inst["size"].as_int(0));
        // The name of the installer in the folder dist of the publication space, else the one of its address.
        const std::string name = inst["file"].is_string() ? inst["file"].as_string()
            : file_name(inst["url"].as_string());
        const std::string source = join_path(folder, name);
        r["installer"] = name;
        if (sha.size() != 64 || size == 0 || !plain_name(name))
        {
            put_error(r, user_error().set("the update feed is incomplete", "update_feed_invalid"));
            return r;
        }
        // Each pass over the installer ends at once when the service stops.
        const std::string file = installer_path(latest);
        if (!(file_exists(file) && sha256_file(file, nullptr, &cancelled_) == sha))
        {
            // The installer, read through no link, with its size, then its hash, then copied as it was read.
            link_free_file installer;
            std::string why;
            const bool found = installer.open(source, base, why) && installer.size() == size && installer.links() == 1;
            if (!found || sha256_open_file(installer, &cancelled_) != sha)
            {
                if (cancelled_) put_error(r, stopping_error());
                else
                    put_error(r, user_error().set("the installer " + source + " is missing or does not match the feed",
                        "update_installer_mismatch").arg("version", latest));
                return r;
            }
            make_dirs(dir_);
            const std::string tmp = file + ".import";
            remove_file(tmp);
            if (copy_open_file(installer, tmp, sha, &cancelled_).empty() || !rename_file(tmp, file))
            {
                remove_file(tmp);
                if (cancelled_) put_error(r, stopping_error());
                else
                    put_error(r, user_error().set("the installer cannot be copied into " + dir_, "update_record")
                        .arg("version", latest));
                return r;
            }
        }
        user_error staging;
        const stage_result staged = stage(latest, sha, stable["notes"], body, sig, staging);
        if (staged == stage_result::failed)
        {
            log_error("updates: " + staging.text);
            put_error(r, staging);
            return r;
        }
        r["state"] = "ready";
        if (staged == stage_result::older)
        {
            // A newer version, downloaded meanwhile, stays the one installed.
            std::lock_guard<std::mutex> lock(m_);
            const json& current = staged_;
            r["version"] = current["version"];
            return r;
        }
        log_info("updates: version " + latest + " imported from " + folder + ", applied at the next restart");
        return r;
    }

    std::string updater::staged_installer() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return status_["state"].as_string() == "ready" ? status_["file"].as_string() : std::string();
    }

    bool updater::can_apply(user_error& error) const
    {
        if (staged_installer().empty())
        {
            error.set("no update is ready", "update_not_ready");
            return false;
        }
#if defined(_WIN32)
        return true;
#else
        error.set("updates are applied by the package manager on this system", "update_package_manager");
        return false;
#endif
    }

    bool updater::launch_staged(user_error& error)
    {
        std::string version;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& staged = staged_;
            if (status_["state"].as_string() == "ready") version = staged["version"].as_string();
        }
        if (version.empty())
        {
            error.set("no update is ready", "update_not_ready");
            return false;
        }
        // A failure stays in the state, the update ready, so that the tray tells why it waits.
        auto keep = [&]() {
            std::lock_guard<std::mutex> lock(m_);
            put_error(status_, error);
        };
#if defined(_WIN32)
        // The installer is checked against the signed feed beside it once more, held open and
        // shared for reading only from the check to the start of its process.
        const std::string file = installer_path(version);
        uint64_t size = 0;
        std::string sha;
        if (!plain_version(version) || !staged_feed(version, size, sha, error))
        {
            if (error.text.empty())
                error.set("the staged version " + version + " is not valid", "update_invalid").arg("version", version);
            drop_staged(error);
            return false;
        }
        const HANDLE h = CreateFileW(widen(file).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL, nullptr);
        LARGE_INTEGER length = {};
        bool same = h != INVALID_HANDLE_VALUE && GetFileSizeEx(h, &length) != 0
            && static_cast<uint64_t>(length.QuadPart) == size;
        if (same)
        {
            sha256_stream hash;
            std::vector<char> buf(1u << 20);
            for (;;)
            {
                DWORD n = 0;
                if (cancelled_ || !ReadFile(h, buf.data(), static_cast<DWORD>(buf.size()), &n, nullptr))
                {
                    same = false;
                    break;
                }
                if (n == 0) break;
                hash.update(buf.data(), n);
            }
            same = same && hash.hex() == sha;
        }
        if (!same)
        {
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
            if (cancelled_)
            {
                error = stopping_error();
                return false;
            }
            error.set("the installer " + file + " does not match the signed feed of version " + version,
                "update_installer_mismatch").arg("version", version);
            drop_staged(error);
            return false;
        }
        // Detached from the service, which the installer stops.
        std::wstring cmd = L"\"" + widen(file) + L"\" /update /silent";
        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {};
        const bool started = CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE,
                DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB, nullptr, nullptr, &si, &pi)
            || CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                nullptr, nullptr, &si, &pi);
        const DWORD failure = started ? 0 : GetLastError();
        CloseHandle(h);
        if (!started)
        {
            error.set("cannot start the installer (error " + std::to_string(failure) + ")", "update_start")
                .arg("code", std::to_string(failure));
            keep();
            return false;
        }
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        // A failing installer must not be started at every restart.
        remove_file(join_path(dir_, "staged.json"));
        log_info("updates: installer started: " + file);
        return true;
#else
        error.set("updates are applied by the package manager on this system", "update_package_manager");
        keep();
        return false;
#endif
    }

}
