// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_LOG_H_
#define ALBIREO_COMMON_LOG_H_

#include <cstdint>
#include <deque>
#include <mutex>
#include <sstream>
#include <string>

namespace albireo
{

    enum class log_level { debug = 0, info = 1, warning = 2, error = 3 };

    class log_file
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The log of a program: lines with their time and severity, written to a
                file that is renamed with a ".1" suffix when it reaches its size limit, so
                that two files at most are kept.  The last lines also stay in memory, for
                the windows that show them.
        !*/
    public:
        static log_file& instance();

        void open(const std::string& path, uint64_t max_bytes = 4u << 20);
        void set_console(bool on) { console_ = on; }
        void set_level(log_level l) { level_ = l; }

        void write(log_level level, const std::string& message);
        std::deque<std::string> recent() const;

    private:
        log_file() {}
        mutable std::mutex m_;
        std::string path_;
        uint64_t max_bytes_ = 4u << 20;
        bool console_ = false;
        log_level level_ = log_level::info;
        std::deque<std::string> recent_;
    };

    inline void log_debug(const std::string& m) { log_file::instance().write(log_level::debug, m); }
    inline void log_info(const std::string& m) { log_file::instance().write(log_level::info, m); }
    inline void log_warning(const std::string& m) { log_file::instance().write(log_level::warning, m); }
    inline void log_error(const std::string& m) { log_file::instance().write(log_level::error, m); }

}

#endif // ALBIREO_COMMON_LOG_H_
