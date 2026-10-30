// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_BUILTIN_MODELS_H_
#define ALBIREO_COMMON_BUILTIN_MODELS_H_

#include <cstdint>
#include <cstring>

namespace albireo
{

    /*!
        The models that come with Albireo: a language model that serves the three functions
        of the web application and the embedding model of the document search, prepared by
        the build (tools/albireo_builtin.cpp, from resources/builtin_models.json) and named
        with the suffix "-albireo".

        In an installation they lie in the "models" folder of the programs, with their
        manifest builtin.json, which the service reads at each start: builtin_manifest_name
        in builtin_models_folder.  The manifest is {"format": 1, "version", "models": [...]},
        each model with the fields of a catalog entry and "file", "size" and "sha256".

        The complete installer carries them after the sections of its program, as an overlay
        that Windows does not map:

            the files of the models, one after the other
            the manifest, each model with "offset", the position of its file from the start
                of the overlay
            zeros, so that the overlay ends on a multiple of 8 bytes
            builtin_trailer

        The overlay ends at the end of the file, or at the certificate table of a signed
        installer, which follows the overlay and covers it; the reader skips up to 7 bytes
        of zeros a signing tool may add before the table.
    !*/

    const char* const builtin_models_folder = "models";
    const char* const builtin_manifest_name = "builtin.json";

    struct builtin_trailer
    {
        char magic[16];               // builtin_magic, without its terminating zero
        uint64_t manifest_offset = 0; // from the start of the overlay
        uint64_t manifest_size = 0;
        uint64_t overlay_size = 0;    // the whole overlay, trailer included
        uint64_t reserved = 0;

        builtin_trailer() { std::memcpy(magic, "ALBIREO-MODELS-1", 16); }
        bool valid() const { return std::memcmp(magic, "ALBIREO-MODELS-1", 16) == 0; }
    };

    static_assert(sizeof(builtin_trailer) == 48, "the trailer of the overlay is 48 bytes");

}

#endif // ALBIREO_COMMON_BUILTIN_MODELS_H_
