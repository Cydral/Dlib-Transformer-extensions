// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_VERSION_H_
#define ALBIREO_COMMON_VERSION_H_

// Version of the Albireo programs and engines, compared by the update feed and by the
// installer over an installation.
#define ALBIREO_VERSION_MAJOR 1
#define ALBIREO_VERSION_MINOR 0
#define ALBIREO_VERSION_PATCH 29
#define ALBIREO_VERSION "1.0.29"

#define ALBIREO_PRODUCT_NAME "Albireo"
// The edition of this version, shown beside its name by every program.
#define ALBIREO_EDITION "Community Edition"
#define ALBIREO_FULL_NAME "Albireo Community Edition"
#define ALBIREO_COPYRIGHT "Copyright (C) 2026 Cydral Technology, France"
// The website of Cydral Technology and the address to write to.
#define ALBIREO_WEBSITE_URL "https://www.cydral-technology.eu"
#define ALBIREO_CONTACT_EMAIL "contact@cydral-technology.eu"
/*
    The folder of the website where the versions are published: the update feed, the catalog of
    the models, their signatures and the installers (host/distribution.h).  The website serves
    this address itself, without a redirection, which tools/publish_release.py checks at each
    publication.
*/
#define ALBIREO_DISTRIBUTION_URL "https://cydral-technology.eu/files/albireo/dist"
// The sources.
#define ALBIREO_PROJECT_URL "https://github.com/Cydral/Dlib-Transformer-extensions"

#endif // ALBIREO_COMMON_VERSION_H_
