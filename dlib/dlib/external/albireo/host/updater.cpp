// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "updater.h"

#include <cstdlib>
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

    int compare_versions(const std::string& a, const std::string& b)
    {
        std::istringstream sa(a), sb(b);
        for (int i = 0; i < 4; ++i)
        {
            long x = 0, y = 0;
            std::string pa, pb;
            if (std::getline(sa, pa, '.')) x = std::atol(pa.c_str());
            if (std::getline(sb, pb, '.')) y = std::atol(pb.c_str());
            if (x != y) return x < y ? -1 : 1;
        }
        return 0;
    }

    updater::updater(const std::string& feed_url, const std::string& updates_dir)
        : feed_url_(feed_url), dir_(updates_dir)
    {
        status_ = json::object();
        status_["current"] = ALBIREO_VERSION;
        status_["state"] = "idle";
        // An update downloaded by a previous run.
        std::string text;
        if (read_file(join_path(dir_, "staged.json"), text))
        {
            const json staged = json::parse_or_null(text);
            const std::string file = staged["file"].as_string();
            if (compare_versions(staged["version"].as_string(), ALBIREO_VERSION) > 0 && file_exists(file)
                && sha256_file(file) == staged["sha256"].as_string())
            {
                status_["state"] = "ready";
                status_["latest"] = staged["version"];
                status_["notes"] = staged["notes"];
                status_["file"] = file;
            }
        }
    }

    updater::~updater()
    {
        if (worker_.joinable()) worker_.join();
    }

    void updater::check_async()
    {
        if (busy_.exchange(true)) return;
        if (worker_.joinable()) worker_.join();
        worker_ = std::thread([this]() {
            check();
            busy_ = false;
        });
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
        set("state", "checking");
        set("checked_at", static_cast<long long>(now_unix()));
        std::string body, sig, error;
        if (!http_get(feed_url_, body, error, 1u << 20) || !http_get(feed_url_ + ".sig", sig, error, 4096))
        {
            set("state", "error");
            set("error", error);
            log_warning("updates: " + error);
            return;
        }
        if (!verify_signature(body, sig, feed_public_key_pem))
        {
            set("state", "error");
            set("error", "the signature of the update feed is not valid");
            log_error("updates: the signature of the feed is not valid, feed ignored");
            return;
        }
        const json feed = json::parse_or_null(body);
        const json& stable = feed["channels"]["stable"];
        const std::string latest = stable["version"].as_string();
        set("latest", latest);
        const json notes = stable["notes"];
        set("notes", notes);
        if (latest.empty() || compare_versions(latest, ALBIREO_VERSION) <= 0)
        {
            set("state", "up_to_date");
            return;
        }
        const json& inst = stable["installer"];
        const std::string url = inst["url"].as_string();
        const std::string sha = inst["sha256"].as_string();
        const uint64_t size = static_cast<uint64_t>(inst["size"].as_int(0));
        if (url.empty() || sha.size() != 64 || size == 0)
        {
            set("state", "error");
            set("error", "the update feed is incomplete");
            return;
        }
        make_dirs(dir_);
        const std::string file = join_path(dir_, "albireo-setup-" + latest +
#if defined(_WIN32)
            ".exe");
#else
            ".bin");
#endif
        if (!(file_exists(file) && sha256_file(file) == sha))
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
                return static_cast<bool>(out) && got <= size;
            }, error);
            out.close();
            if (!ok || got != size || sha256_file(tmp) != sha || !rename_file(tmp, file))
            {
                remove_file(tmp);
                set("state", "error");
                set("error", ok ? std::string("the downloaded installer does not match the feed") : error);
                log_error("updates: download of " + latest + " failed");
                return;
            }
        }
        json staged = json::object();
        staged["version"] = latest;
        staged["file"] = file;
        staged["sha256"] = sha;
        staged["notes"] = notes;
        write_file_atomic(join_path(dir_, "staged.json"), staged.dump());
        set("file", file);
        set("state", "ready");
        log_info("updates: version " + latest + " downloaded, applied at the next restart");
    }

    std::string updater::staged_installer() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return status_["state"].as_string() == "ready" ? status_["file"].as_string() : std::string();
    }

    bool updater::launch_staged(std::string& error)
    {
        const std::string file = staged_installer();
        if (file.empty())
        {
            error = "no update is ready";
            return false;
        }
#if defined(_WIN32)
        // Detached from the service, which the installer stops.
        std::wstring cmd = L"\"" + widen(file) + L"\" /update /silent";
        STARTUPINFOW si = {};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi = {};
        if (!CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE,
                DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB, nullptr, nullptr, &si, &pi)
            && !CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                nullptr, nullptr, &si, &pi))
        {
            error = "cannot start the installer (error " + std::to_string(GetLastError()) + ")";
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
