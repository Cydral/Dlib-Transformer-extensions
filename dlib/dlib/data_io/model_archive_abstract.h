// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_MODEL_ARCHIVE_ABSTRACT_H_
#ifdef DLIB_MODEL_ARCHIVE_ABSTRACT_H_

#include <string>
#include "gguf_vision_spec_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct model_archive_info
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Everything an archive holds beside the network and the tokenizer.

                - model_name: the model's display name.
                - has_vision: whether the network carries a vision tower.
                - vision:     the pixel normalization that tower was trained with,
                              meaningful only when has_vision. It is geometry rather than
                              weights, but a tower fed pictures centred differently from
                              the ones it learned on sees other images, so an archive is
                              not self-contained without it.
        !*/

        std::string model_name;
        bool has_vision = false;
        vision_spec vision;
    };

    const std::string& model_archive_tag();
    /*!
        ensures
            - Returns the format tag every archive begins with.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename subnet_type>
    void save_model_archive(
        const std::string& path,
        const model_archive_info& info,
        const subnet_type& subnet,
        const hf_tokenizer& tok
    );
    /*!
        ensures
            - Writes the tag, the model name, subnet, tok and, when info.has_vision, the
              pixel normalization.
            - subnet is the parameter-bearing subnetwork rather than the whole loss
              network, which carries no parameters. A reader can then deserialize straight
              into its own network, so only one copy of the parameters is ever allocated.
        throws
            - serialization_error if the file cannot be written.
    !*/

    template <typename subnet_type>
    void load_model_archive(
        const std::string& path,
        subnet_type& subnet,
        hf_tokenizer& tok,
        model_archive_info& info,
        bool expects_vision
    );
    /*!
        requires
            - subnet already has the shape the archive describes.
            - expects_vision is what the compiled network carries.
        ensures
            - Fills subnet, tok and info from the archive at path.
        throws
            - std::runtime_error if the file cannot be opened, is not an archive, was
              written by a build whose network layout differs, or disagrees with
              expects_vision. Each case is reported in terms of what to do about it rather
              than in terms of the class where the read failed.

        WHY THIS LIVES HERE

            The format had come to be described in two places, the converter and the
            fine-tuner, and the second had drifted: it rewrote archives without the vision
            block, producing files the first could no longer read. A format spelled out
            once cannot drift.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MODEL_ARCHIVE_ABSTRACT_H_
