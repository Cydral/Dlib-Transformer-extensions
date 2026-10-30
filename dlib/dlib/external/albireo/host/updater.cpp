// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "updater.h"

#include <cstdlib>
#include <exception>
#include <fstream>
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
    }

    updater::updater(const std::string& feed_url, const std::string& updates_dir)
        : feed_url_(feed_url), dir_(updates_dir)
    {
        status_ = json::object();
        status_["current"] = ALBIREO_VERSION;
        status_["state"] = "idle";
        // An update downloaded by a previous run, checked again against its signed feed.
        std::string text;
        if (read_file(join_path(dir_, "staged.json"), text, 1u << 20))
        {
            const json staged = json::parse_or_null(text);
            const std::string version = staged["version"].as_string();
            std::string error;
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
                    log_warning("updates: " + error + ", the update waits for the next check");
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
        worker_ = std::thread([this]() {
            try
            {
                check();
            }
            catch (const std::exception& e)
            {
                std::lock_guard<std::mutex> lock(m_);
                status_["state"] = "error";
                status_["error"] = e.what();
            }
            busy_ = false;
        });
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
        const std::string& feed, const std::string& signature, std::string& error)
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
            error = "the update of version " + version + " cannot be recorded in " + dir_;
            return stage_result::failed;
        }
        staged_ = staged;
        status_["latest"] = version;
        status_["notes"] = notes;
        status_["file"] = file;
        status_["state"] = "ready";
        status_.erase("error");
        return stage_result::staged;
    }

    bool updater::staged_feed(const std::string& version, uint64_t& size, std::string& sha, std::string& error) const
    {
        const std::string path = feed_path(version);
        std::string body, sig;
        if (!read_file(path, body, 1u << 20) || !read_file(path + ".sig", sig, 4096))
        {
            error = "no signed feed beside the installer of version " + version;
            return false;
        }
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            error = "the signature of " + path + " is not valid";
            return false;
        }
        const json feed = json::parse_or_null(body);
        const json& stable = feed["channels"]["stable"];
        sha = stable["installer"]["sha256"].as_string();
        size = static_cast<uint64_t>(stable["installer"]["size"].as_int(0));
        if (stable["version"].as_string() != version || sha.size() != 64 || size == 0)
        {
            error = path + " does not describe the installer of version " + version;
            return false;
        }
        return true;
    }

    bool updater::check_staged(const json& staged, std::string& error) const
    {
        const std::string version = staged["version"].as_string();
        const std::string file = staged["file"].as_string();
        if (file != installer_path(version))
        {
            error = "the staged installer " + file + " lies outside " + dir_;
            return false;
        }
        uint64_t size = 0;
        std::string sha;
        if (!staged_feed(version, size, sha, error)) return false;
        if (sha != staged["sha256"].as_string() || file_size(file) != size || sha256_file(file) != sha)
        {
            error = "the installer of version " + version + " does not match its signed feed";
            return false;
        }
        return true;
    }

    void updater::drop_staged(const std::string& error)
    {
        log_error("updates: " + error + ", the update waits for the next check");
        remove_file(join_path(dir_, "staged.json"));
        std::lock_guard<std::mutex> lock(m_);
        staged_ = json();
        status_["state"] = "error";
        status_["error"] = error;
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
        auto settle = [&](const std::string& state, const std::string& error) {
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
            if (error.empty()) status_.erase("error");
            else status_["error"] = error;
        };
        auto fail = [&](const std::string& error) {
            if (ended()) settle("idle", std::string());
            else settle("error", error);
        };
        if (ended()) return;
        set("state", "checking");
        set("checked_at", static_cast<long long>(now_unix()));
        std::string body, sig, error;
        if (!http_get(feed_url_, body, error, 1u << 20) || !http_get(feed_url_ + ".sig", sig, error, 4096))
        {
            fail(error);
            if (!ended()) log_warning("updates: " + error);
            return;
        }
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            fail("the signature of the update feed is not valid");
            log_error("updates: the signature of the feed is not valid, feed ignored");
            return;
        }
        const json feed = json::parse_or_null(body);
        const json& stable = feed["channels"]["stable"];
        const std::string latest = stable["version"].as_string();
        const json notes = stable["notes"];
        std::string staged_version;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& staged = staged_;
            staged_version = staged["version"].as_string();
        }
        if (latest.empty() || compare_versions(latest, ALBIREO_VERSION) <= 0
            || (!staged_version.empty() && compare_versions(latest, staged_version) <= 0))
        {
            {
                std::lock_guard<std::mutex> lock(m_);
                status_["latest"] = latest;
                status_["notes"] = notes;
            }
            settle("up_to_date", std::string());
            return;
        }
        set("latest", latest);
        set("notes", notes);
        const json& inst = stable["installer"];
        const std::string url = inst["url"].as_string();
        const std::string sha = inst["sha256"].as_string();
        const uint64_t size = static_cast<uint64_t>(inst["size"].as_int(0));
        if (url.empty() || sha.size() != 64 || size == 0 || !plain_version(latest))
        {
            fail("the update feed is incomplete");
            return;
        }
        make_dirs(dir_);
        const std::string file = installer_path(latest);
        if (!(file_exists(file) && sha256_file(file, nullptr, &cancelled_) == sha))
        {
            set("state", "downloading");
            set("total", static_cast<unsigned long long>(size));
            const std::string tmp = file + ".partial";
            std::ofstream out;
#if defined(_WIN32)
            out.open(widen(tmp).c_str(), std::ios::binary | std::ios::trunc);
#else
            out.open(tmp, std::ios::binary | std::ios::trunc);
#endif
            uint64_t got = 0;
            const bool ok = out && http_get_range(url, 0, -1, [&](const char* d, size_t n) {
                out.write(d, static_cast<std::streamsize>(n));
                got += n;
                set("received", static_cast<unsigned long long>(got));
                return static_cast<bool>(out) && got <= size && !cancelled_;
            }, error);
            out.close();
            if (!ok || got != size || sha256_file(tmp, nullptr, &cancelled_) != sha || !rename_file(tmp, file))
            {
                remove_file(tmp);
                fail(ok ? std::string("the downloaded installer does not match the feed") : error);
                if (!ended()) log_error("updates: download of " + latest + " failed");
                return;
            }
        }
        std::string staging;
        switch (stage(latest, sha, notes, body, sig, staging))
        {
        case stage_result::staged:
            log_info("updates: version " + latest + " downloaded, applied at the next restart");
            break;
        case stage_result::older:
            // A newer version, imported from a medium meanwhile, stays the one installed.
            settle("up_to_date", std::string());
            break;
        case stage_result::failed:
            fail(staging);
            log_error("updates: " + staging);
            break;
        }
    }

    json updater::import_offline(const std::string& folder)
    {
        json r = json::object();
        std::string body, sig;
        if (!read_file(join_path(folder, "feed.json"), body, 1u << 20)
            || !read_file(join_path(folder, "feed.json.sig"), sig, 4096))
        {
            r["error"] = "no signed update feed (feed.json and feed.json.sig) in " + folder;
            return r;
        }
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            r["error"] = "the signature of " + join_path(folder, "feed.json") + " is not valid";
            log_error("updates: " + r["error"].as_string());
            return r;
        }
        const json stable = json::parse_or_null(body)["channels"]["stable"];
        const std::string latest = stable["version"].as_string();
        r["version"] = latest;
        if (latest.empty())
        {
            r["error"] = "the update feed names no version";
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
        const std::string name = file_name(inst["url"].as_string());
        const std::string source = join_path(folder, name);
        r["installer"] = name;
        if (sha.size() != 64 || size == 0 || name.empty() || !plain_version(latest))
        {
            r["error"] = "the update feed is incomplete";
            return r;
        }
        // Each pass over the installer ends at once when the service stops.
        const std::string file = installer_path(latest);
        if (!(file_exists(file) && sha256_file(file, nullptr, &cancelled_) == sha))
        {
            if (!file_exists(source) || file_size(source) != size || sha256_file(source, nullptr, &cancelled_) != sha)
            {
                r["error"] = cancelled_ ? std::string("the service stops")
                    : "the installer " + source + " is missing or does not match the feed";
                return r;
            }
            make_dirs(dir_);
            const std::string tmp = file + ".import";
            if (!copy_file(source, tmp, &cancelled_) || sha256_file(tmp, nullptr, &cancelled_) != sha
                || !rename_file(tmp, file))
            {
                remove_file(tmp);
                r["error"] = cancelled_ ? std::string("the service stops")
                    : "the installer cannot be copied into " + dir_;
                return r;
            }
        }
        std::string staging;
        const stage_result staged = stage(latest, sha, stable["notes"], body, sig, staging);
        if (staged == stage_result::failed)
        {
            log_error("updates: " + staging);
            r["error"] = staging;
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

    bool updater::launch_staged(std::string& error)
    {
        std::string version;
        {
            std::lock_guard<std::mutex> lock(m_);
            const json& staged = staged_;
            if (status_["state"].as_string() == "ready") version = staged["version"].as_string();
        }
        if (version.empty())
        {
            error = "no update is ready";
            return false;
        }
#if defined(_WIN32)
        // The installer is checked against the signed feed beside it once more, held open and
        // shared for reading only from the check to the start of its process.
        const std::string file = installer_path(version);
        uint64_t size = 0;
        std::string sha;
        if (!plain_version(version) || !staged_feed(version, size, sha, error))
        {
            if (error.empty()) error = "the staged version " + version + " is not valid";
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
                error = "the service stops";
                return false;
            }
            error = "the installer " + file + " does not match the signed feed of version " + version;
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
            error = "cannot start the installer (error " + std::to_string(failure) + ")";
            return false;
        }
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        // A failing installer must not be started at every restart.
        remove_file(join_path(dir_, "staged.json"));
        log_info("updates: installer started: " + file);
        return true;
#else
        error = "updates are applied by the package manager on this system";
        return false;
#endif
    }

}
