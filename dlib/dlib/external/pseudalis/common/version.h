// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_COMMON_VERSION_H_
#define PSEUDALIS_COMMON_VERSION_H_

/*
    Version of the Pseudalis programs and engines, compared by the update feed and by the
    installer over an installation.  The three numbers go into the version resources of the
    programs, which the installer reads, and the text into their messages: a new version changes
    the four lines together.  A compilation stops when the text does not spell the numbers (at the
    end of this file), and the configuration of the build, which reads the version here, stops
    for the same reason (CMakeLists.txt).
*/
#define PSEUDALIS_VERSION_MAJOR 1
#define PSEUDALIS_VERSION_MINOR 0
#define PSEUDALIS_VERSION_PATCH 16
#define PSEUDALIS_VERSION "1.0.16"

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
#define PSEUDALIS_COPYRIGHT "Copyright (C) 2026 Cydral Technology, France"
// The website, where the versions are published, and the address to write to.
#define PSEUDALIS_WEBSITE_URL "https://cydral-technology.eu"
#define PSEUDALIS_CONTACT_EMAIL "contact@cydral-technology.eu"
// The sources.
#define PSEUDALIS_PROJECT_URL "https://github.com/Cydral/Dlib-Transformer-extensions"

#if defined(__cplusplus) && !defined(RC_INVOKED)
namespace pseudalis
{
    namespace version_impl
    {
        constexpr bool same_text(const char* a, const char* b)
        {
            return *a == *b && (*a == '\0' || same_text(a + 1, b + 1));
        }
    }
}
#define PSEUDALIS_VERSION_TEXT_(x) #x
#define PSEUDALIS_VERSION_TEXT(x) PSEUDALIS_VERSION_TEXT_(x)
static_assert(pseudalis::version_impl::same_text(PSEUDALIS_VERSION,
                  PSEUDALIS_VERSION_TEXT(PSEUDALIS_VERSION_MAJOR) "." PSEUDALIS_VERSION_TEXT(PSEUDALIS_VERSION_MINOR)
                  "." PSEUDALIS_VERSION_TEXT(PSEUDALIS_VERSION_PATCH)),
              "common/version.h: PSEUDALIS_VERSION does not spell PSEUDALIS_VERSION_MAJOR, MINOR and PATCH");
#endif

#endif // PSEUDALIS_COMMON_VERSION_H_
