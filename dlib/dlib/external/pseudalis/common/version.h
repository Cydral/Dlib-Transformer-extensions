// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_VERSION_H_
#define PSEUDALIS_COMMON_VERSION_H_

// Version of the Pseudalis programs and engines, compared by the update feed and by the
// installer over an installation.
#define PSEUDALIS_VERSION_MAJOR 1
#define PSEUDALIS_VERSION_MINOR 0
#define PSEUDALIS_VERSION_PATCH 0
#define PSEUDALIS_VERSION "1.0.0"

#define PSEUDALIS_PRODUCT_NAME "Pseudalis"
/*
    The edition: Community, free and open source under the Boost Software License, which
    the FLOSS exception of WinFsp covers (README.md, Editions).  The installed programs, the
    About window, the web application and the files of the programs name it.
*/
#ifndef PSEUDALIS_EDITION
#define PSEUDALIS_EDITION "Community"
#endif
#define PSEUDALIS_FULL_NAME PSEUDALIS_PRODUCT_NAME " " PSEUDALIS_EDITION
#define PSEUDALIS_PUBLISHER "Cydral Technology"
#define PSEUDALIS_AUTHOR_NAME "Aldric Pierrain"
#define PSEUDALIS_AUTHOR_EMAIL "cydraltechnology@gmail.com"
#define PSEUDALIS_COPYRIGHT "Copyright (C) 2026 Cydral Technology"
#define PSEUDALIS_PROJECT_URL "https://github.com/Cydral/Dlib-Transformer-extensions"
#define PSEUDALIS_WINFSP_NOTICE "WinFsp - Windows File System Proxy, Copyright (C) Bill Zissimopoulos"
#define PSEUDALIS_WINFSP_URL "https://github.com/winfsp/winfsp"

#endif // PSEUDALIS_COMMON_VERSION_H_
