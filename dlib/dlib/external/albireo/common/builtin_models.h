// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_COMMON_BUILTIN_MODELS_H_
#define ALBIREO_COMMON_BUILTIN_MODELS_H_

#include <cstdint>
#include <cstring>
#include <string>

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
        each model with the fields of a catalog entry and "file", "size" and "sha256"; a model
        that reads images has a "projector" with its own "file", "size" and "sha256", the file
        of its image encoder.

        The complete installer carries them after the sections of its program, as an overlay
        that Windows does not map:

            the files of the models, one after the other, each followed by its projector
            the manifest, each model and each projector with "offset", the position of its
                file from the start of the overlay
            zeros, so that the overlay ends on a multiple of 8 bytes
            builtin_trailer

        The overlay ends at the end of the file, or at the certificate table of a signed
        installer, which follows the overlay and covers it; the reader skips up to 7 bytes
        of zeros a signing tool may add before the table.

        Windows runs no program of 4 GiB or more (max_installer_size): when the program and
        its models would reach that size, the overlay lies in a file of its own beside the
        installer, named after it with the extension builtin_overlay_extension
        (albireo-setup.exe, albireo-setup.models), the overlay alone from its first byte, and
        the installer is the program without models.  Its manifest names the version of the
        installer it goes with; the installer reads it when it carries no overlay, and both
        files travel together.
    !*/

    const char* const builtin_models_folder = "models";
    const char* const builtin_manifest_name = "builtin.json";
    const char* const builtin_overlay_extension = ".models";
    // The largest program Windows runs: 4 GiB less one byte.
    const uint64_t max_installer_size = 0xFFFFFFFFull;

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

    /*!
        requires
            - read(position, data, size) reads size bytes of the file at position, true when it can.
        ensures
            - Finds the overlay that ends at end, or up to 7 bytes of zeros before it, and starts at
              lower or after: start receives its position, trailer its trailer and manifest the
              text of its manifest.
            - Returns false when no valid trailer ends there, or when its sizes do not fit.
    !*/
    template <typename reader>
    bool find_builtin_overlay(const reader& read, uint64_t lower, uint64_t end, uint64_t& start,
        builtin_trailer& trailer, std::string& manifest)
    {
        for (uint64_t pad = 0; pad < 8; ++pad)
        {
            if (end < lower + sizeof(builtin_trailer) + pad) return false;
            const uint64_t tail = end - pad;
            builtin_trailer t;
            if (!read(tail - sizeof(t), &t, sizeof(t)) || !t.valid()) continue;
            if (t.overlay_size > tail - lower || t.overlay_size < sizeof(t) || t.manifest_size > (16u << 20)
                || t.manifest_offset + t.manifest_size > t.overlay_size - sizeof(t))
                return false;
            start = tail - t.overlay_size;
            trailer = t;
            manifest.assign(static_cast<size_t>(t.manifest_size), '\0');
            return manifest.empty() || read(start + t.manifest_offset, &manifest[0], manifest.size());
        }
        return false;
    }

}

#endif // ALBIREO_COMMON_BUILTIN_MODELS_H_
