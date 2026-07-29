// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
// The converted-model archive, written once and read from several places.
//
// A model that has been imported into the Dlib stack lives in one file: a format tag, the
// model's name, the parameter-bearing subnet, the tokenizer, and, when the network carries
// a vision tower, the pixel normalization that tower was trained with. That last part is
// geometry rather than weights, but a tower fed pictures centred differently from the ones
// it learned on sees other images, so the archive is not self-contained without it.
//
// This header exists because the format had come to be described in two places, the
// converter and the fine-tuner, and the second had already drifted: it rewrote archives
// without the vision block, producing files the first could no longer read. A format
// spelled out once cannot drift.
//
// The subnet is serialized rather than the whole loss network, which carries no
// parameters. That lets a reader deserialize straight into its own network, so only one
// copy of the parameters is ever allocated; going through the full network would need a
// temporary and would transiently double the pinned host memory.

#ifndef DLIB_MODEL_ARCHIVE_H_
#define DLIB_MODEL_ARCHIVE_H_

#include "model_archive_abstract.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include "gguf_vision_spec.h"
#include "../dnn.h"
#include "../tokenizer/hf_tokenizer.h"

namespace dlib
{
    struct model_archive_info
    {
        std::string model_name;
        bool has_vision = false;
        vision_spec vision;      // meaningful only when has_vision
        bool tail_missing = false;  // archive written before the trailing block existed
    };

    inline const std::string& model_archive_tag()
    {
        static const std::string tag = "gguf_import_model";
        return tag;
    }

// ----------------------------------------------------------------------------------------

    template <typename subnet_type>
    void save_model_archive(const std::string& path, const model_archive_info& info,
        const subnet_type& subnet, const hf_tokenizer& tok)
    {
        auto out = serialize(path);
        out << model_archive_tag() << info.model_name << subnet << tok << info.has_vision;
        if (info.has_vision)
            out << info.vision.image_size << info.vision.image_mean << info.vision.image_std;
    }

// ----------------------------------------------------------------------------------------

    /* Reads an archive into a network that must already have the shape the file describes.

       expects_vision is what the compiled network carries, and a mismatch is reported here
       rather than left to a deserialization that would fail deeper down with a message
       naming a layer class and nothing actionable. */
    template <typename subnet_type>
    void load_model_archive(const std::string& path, subnet_type& subnet, hf_tokenizer& tok,
        model_archive_info& info, bool expects_vision)
    {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) throw std::runtime_error("cannot open " + path);

        std::string tag;
        deserialize(tag, fin);
        if (tag != model_archive_tag())
            throw std::runtime_error("'" + path + "' is not a model archive produced by "
                "--convert; regenerate it");

        deserialize(info.model_name, fin);
        try
        {
            deserialize(subnet, fin);
            deserialize(tok, fin);
        }
        catch (const serialization_error& e)
        {
            /* The library carries no serialization versioning by design, so a layout change
               makes existing archives unreadable rather than silently wrong. The raw
               message names a class and says nothing actionable, hence this one. */
            throw std::runtime_error("cannot read " + path + ": " + e.what()
                + "\nThis archive was written by a build whose network layout differs from "
                "this one. Regenerate it with --convert.");
        }

        /* An archive that ends right after the tokenizer predates the trailing block. It
           is accepted rather than refused, because the weights it carries may represent
           hours of training and are perfectly usable for everything that does not touch
           images. The caller is told, through tail_missing, and the pixel normalization is
           left empty so that an image path fails on the missing value rather than on a
           plausible default. */
        if (fin.peek() == std::char_traits<char>::eof())
        {
            info.tail_missing = true;
            info.has_vision = expects_vision;
            return;
        }

        deserialize(info.has_vision, fin);
        if (info.has_vision != expects_vision)
            throw std::runtime_error("'" + path + "' "
                + (info.has_vision ? "carries" : "does not carry") + " a vision tower and "
                "this build " + (expects_vision ? "expects" : "does not expect")
                + " one; regenerate the header and the archive together");

        if (info.has_vision)
        {
            deserialize(info.vision.image_size, fin);
            deserialize(info.vision.image_mean, fin);
            deserialize(info.vision.image_std, fin);
        }
    }
}

#endif // DLIB_MODEL_ARCHIVE_H_
