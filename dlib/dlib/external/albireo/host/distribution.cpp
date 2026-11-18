// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#include "distribution.h"

#include <mutex>

#include "../common/version.h"

namespace albireo
{

    namespace
    {
        // The folder distribution_for_tests() gives in place of the one of the website, empty otherwise.
        std::mutex folder_m;
        std::string folder_of_tests;
    }

    std::string distribution_address(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(folder_m);
        return (folder_of_tests.empty() ? std::string(ALBIREO_DISTRIBUTION_URL) : folder_of_tests) + "/" + name;
    }

    void distribution_for_tests(const std::string& folder)
    {
        std::lock_guard<std::mutex> lock(folder_m);
        folder_of_tests = folder;
    }

}
