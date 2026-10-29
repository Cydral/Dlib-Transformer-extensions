// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "log.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>

#include "platform.h"

namespace albireo
{

    log_file& log_file::instance()
    {
        static log_file f;
        return f;
    }

    void log_file::open(const std::string& path, uint64_t max_bytes)
    {
        std::lock_guard<std::mutex> lock(m_);
        path_ = path;
        max_bytes_ = max_bytes;
        make_dirs(parent_path(path));
    }

    void log_file::write(log_level level, const std::string& message)
    {
        if (static_cast<int>(level) < static_cast<int>(level_)) return;
        static const char* names[] = { "debug", "info", "warning", "error" };
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        const long ms = static_cast<long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000);
        std::tm tm;
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char stamp[64];
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);
        char ms_part[8];
        std::snprintf(ms_part, sizeof(ms_part), ".%03ld", ms);
        const std::string line = std::string(stamp) + ms_part + " [" + names[static_cast<int>(level)] + "] " + message;

        std::lock_guard<std::mutex> lock(m_);
        recent_.push_back(line);
        while (recent_.size() > 500) recent_.pop_front();
        if (console_)
        {
            std::fputs(line.c_str(), stderr);
            std::fputc('\n', stderr);
        }
        if (path_.empty()) return;
        if (file_size(path_) > max_bytes_) rename_file(path_, path_ + ".1");
#if defined(_WIN32)
        std::ofstream out(widen(path_).c_str(), std::ios::app | std::ios::binary);
#else
        std::ofstream out(path_, std::ios::app | std::ios::binary);
#endif
        if (out)
        {
            out << line;
#if defined(_WIN32)
            out << "\r\n";
#else
            out << "\n";
#endif
        }
    }

    std::deque<std::string> log_file::recent() const
    {
        std::lock_guard<std::mutex> lock(m_);
        return recent_;
    }

}
