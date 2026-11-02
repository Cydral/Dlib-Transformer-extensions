// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// The drive outside Windows: none, the service runs without it.

#include "drive.h"

namespace pseudalis
{
    struct drive_host::impl
    {
    };

    drive_host::drive_host(service&) : p_(new impl) {}
    drive_host::~drive_host() {}

    bool drive_host::start(const std::string&, std::string& error)
    {
        error = "the drive needs Windows and WinFsp";
        return false;
    }

    void drive_host::stop() {}

    json drive_host::status() const
    {
        json j = json::object();
        j["letter"] = "";
        j["mounted"] = false;
        j["error"] = "the drive needs Windows and WinFsp";
        return j;
    }
}
