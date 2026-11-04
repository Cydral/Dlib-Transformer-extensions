// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_VERSION_H_
#define PSEUDALIS_COMMON_VERSION_H_

// Version of the Pseudalis programs and engines, compared by the update feed and by the
// installer over an installation.
#define PSEUDALIS_VERSION_MAJOR 1
#define PSEUDALIS_VERSION_MINOR 0
#define PSEUDALIS_VERSION_PATCH 1
#define PSEUDALIS_VERSION "1.0.1"

#define PSEUDALIS_PRODUCT_NAME "Pseudalis"
/*
    The edition: Community, free and open source under the Boost Software License (README.md,
    Editions).  The installed programs, the About window, the web application and the files of
    the programs name it.
*/
#ifndef PSEUDALIS_EDITION
#define PSEUDALIS_EDITION "Community"
#endif
// The name and its edition in one literal, which an edition defines with PSEUDALIS_EDITION: the
// resource compiler of Windows joins no adjacent literals.
#ifndef PSEUDALIS_FULL_NAME
#define PSEUDALIS_FULL_NAME "Pseudalis Community"
#endif
#define PSEUDALIS_PUBLISHER "Cydral Technology"
#define PSEUDALIS_AUTHOR_NAME "Aldric Pierrain"
#define PSEUDALIS_AUTHOR_EMAIL "cydraltechnology@gmail.com"
#define PSEUDALIS_COPYRIGHT "Copyright (C) 2026 Cydral Technology"
#define PSEUDALIS_PROJECT_URL "https://github.com/Cydral/Dlib-Transformer-extensions"

#endif // PSEUDALIS_COMMON_VERSION_H_
