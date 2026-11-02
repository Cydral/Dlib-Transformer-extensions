// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_ENGINE_ENGINE_H_
#define PSEUDALIS_ENGINE_ENGINE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../common/json.h"

namespace pseudalis
{
    /*
        The kinds of files the engine processes: the documents of dlib/privacy (plain text,
        delimited values, markup, Word, Excel, PowerPoint and PDF), the pictures (PNG, JPEG,
        GIF, BMP, TIFF, WebP), whose place takes a text that describes them, and the archives
        (ZIP, TAR, TAR.GZ) that hold them.
    */
    enum class file_kind { unsupported, document, picture, archive };

    struct picture_found
    {
        std::string id;           // the first 16 hexadecimal digits of the SHA-256 of the bytes
        std::string sha256;
        std::string media_type;   // "image/png"...
        std::string location;     // the file, the entry of an archive, the part or the page
        const std::string* bytes = nullptr;
    };

    // Receives each picture found and returns its description in clear, or an empty string.
    typedef std::function<std::string(const picture_found&)> picture_handler;

    struct engine_options
    {
        std::string date_order = "dmy";     // or "mdy": the order of the numeric dates
        std::string context = "default";    // the context of the pseudonymization
    };

    struct engine_result
    {
        std::string bytes;                  // the file written
        std::string name;                   // its name: "photo.png.txt" for a picture
        std::string kind;                   // "txt", "csv", "xml", "docx", ..., "image", "zip", "tar", "tar.gz"
        json report = json::object();
        std::vector<std::string> pictures;  // the identifiers of the pictures found, once each
    };

    class engine
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The anonymization and the restoration of files with the detector of
                dlib/privacy, its rules and its two learned recognizers (the sequence
                tagger and the span model compiled into the engine), and one key.
                Each value found is replaced by its tag [[TYPE:body#CCC]], encrypted under
                the key, so that restore() gives it back; the dates move by a secret shift.

                A picture becomes "[Image@<id> : <description>]", the description
                protected like any text, or "[Image@<id>]" without description: in its
                place in a Word document or on a slide, over its gray box in a PDF, as the
                text file <name>.txt for a picture alone or in an archive.  The archives
                keep their documents, protected, and their pictures as text files; the
                entries of another kind are left out and listed in report["dropped"].

            THREAD SAFETY
                The methods may be called from several threads at once.
        !*/
    public:
        // key: 32 bytes.  Throws std::runtime_error when the models cannot be read.
        engine(const std::string& key, const engine_options& options);

        static file_kind kind_of_name(const std::string& name);
        static file_kind kind_of(const std::string& bytes, const std::string& name);
        static std::string picture_marker(const std::string& id, const std::string& description);

        // Throws std::runtime_error for a file of no supported kind, or that cannot be read.
        engine_result anonymize(const std::string& bytes, const std::string& name,
            const picture_handler& pictures) const;
        // Puts back the values of the tags; a picture marker stays (the service keeps the pictures).
        engine_result restore(const std::string& bytes, const std::string& name) const;

        std::string anonymize_text(const std::string& text, json& report) const;
        std::string restore_text(const std::string& text, json& report) const;
        // [{"begin", "end", "type", "confidence", "text"}] of the values of a text.
        json detect(const std::string& text) const;

        std::string key_fingerprint() const;
        // {"tagger", "span_model", "size"}
        static json models();

    private:
        struct impl;
        std::shared_ptr<impl> p_;
    };
}

#endif // PSEUDALIS_ENGINE_ENGINE_H_
