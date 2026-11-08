// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_ENGINE_ENGINE_H_
#define PSEUDALIS_ENGINE_ENGINE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../common/imposed_terms.h"
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
        std::string id;           // 16 hexadecimal digits of a digest of the bytes keyed by the key
        std::string sha256;
        std::string media_type;   // "image/png"...
        std::string location;     // the file, the entry of an archive, the part or the page
        const std::string* bytes = nullptr;
    };

    // Receives each picture found and returns its description in clear, or an empty string.
    typedef std::function<std::string(const picture_found&)> picture_handler;

    /*
        Returns the tags recorded for a substitute equal to candidate, the most likely first: the
        names and the properties of the documents hold substitutes, values written without
        their tags, which the caller records when they are made (engine_result::substitutions).
    */
    typedef std::function<std::vector<std::string>(const std::string& candidate)> substitute_lookup;

    struct substitution
    {
        std::string substitute;             // as written in a name, a property or a text without tags
        std::string tag;                    // the tag it stands for
        bool text = false;                  // written in the text of a document protected without tags
    };

    struct engine_options
    {
        std::string date_order = "dmy";     // or "mdy": the order of the numeric dates
        std::string context = "default";    // the context of the pseudonymization
    };

    /*
        An entry that anonymize() wrote in an archive: its path in the archive read, its path in the
        archive written, the SHA-256 of the bytes written and its kind.  The entries of an archive
        inside the archive follow it, their paths after its own and '!' ("dossier/notes.tar.gz!a.txt").
    */
    struct entry_written
    {
        std::string original;
        std::string name;
        std::string sha256;
        std::string kind;
    };

    struct engine_result
    {
        std::string bytes;                  // the file written
        std::string name;                   // its name, protected: "photo.png.txt" for a picture
        std::string kind;                   // "txt", "csv", "xml", "docx", ..., "image", "zip", "tar", "tar.gz"
        json report = json::object();
        std::vector<std::string> pictures;  // the identifiers of the pictures found, once each
        std::vector<substitution> substitutions;   // anonymize(): the substitutes written
        std::vector<entry_written> entries;        // anonymize() of an archive: its entries written
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
                entries of another kind are left out and listed in report["dropped"], and a
                restoration keeps them as they are.  A file is of the kind its name gives
                when its content is of that kind, and a file of any other name is a text when
                its content is text through and through (kind_of()); a restoration also reads
                a text under any name, since it may hold tags.

                The names of the files, of the entries of the archives and of their
                folders, and the properties of the documents (title, authors, subject,
                keywords, company... of Word, Excel and PowerPoint, the document
                information of a PDF) take substitutes rather than tags: each value is
                written in its protected form without the brackets, the type and the
                check of its tag, which a name cannot hold and a window of properties
                would show as they are.  The name of a document is read with what its
                content taught the detector.  A substitute holds no character that
                Windows refuses in a name.  engine_result::substitutions lists each one
                with its tag, which the caller records: restore() asks the lookup for
                them.

                Without tags (anonymize() and anonymize_text() with tags false), each value of
                the text takes its substitute too: the file holds no tag, and its values come
                back only from the substitutes that the caller keeps (engine_result::
                substitutions, text true) and gives back to restore().

            THREAD SAFETY
                The methods may be called from several threads at once.
        !*/
    public:
        // key: 32 bytes.  Throws std::runtime_error when the models cannot be read.
        engine(const std::string& key, const engine_options& options);

        static file_kind kind_of_name(const std::string& name);
        /*
            The kind of a file: the kind its extension gives (common/file_kinds.h) when the content is
            of that kind, an Office document, a PDF or a text for a document, any archive for an
            archive.  A name whose extension gives no kind, or without extension, is a text when its
            content is text through and through: dlib finds in it no binary signature and the measures
            of a text (dlib/data_io), every character decodes with no control character but the
            tabulation, the ends of lines and the form feed, nor U+FFFE or U+FFFF, and it holds no
            encoded part.  An encoded part is a block of base64, base64url or hexadecimal digits (1024
            characters in a row, or 16 runs in a row of 60, the lines of a block, in a string of JSON or
            of a program too, which a line of dashes or a path does not make), data or JSON in base64,
            bytes in hexadecimal, a dump or an array (64 in a row, or 4 lines of 8, with letters or
            marks among them), or a format that writes its values encoded: a message with parts in
            base64, quoted-printable or uuencode, or with encoded words, an RTF document, a vCard or an
            iCalendar, LDIF in base64, a block armored by PEM or OpenPGP, its header and its body, in a
            string too, a block of uuencode or of ASCII85, and, by its first line, a PostScript program
            or an export of the registry.  Without extension, a picture, an Office document, an archive
            or a PDF is also taken by its signature.  The other files are unsupported: an OpenDocument,
            or an e-mail with attachments, would come out half protected.
        */
        static file_kind kind_of(const std::string& bytes, const std::string& name);
        static std::string picture_marker(const std::string& id, const std::string& description);

        /*
            Throws std::runtime_error for a file of no supported kind, or that cannot be read.  The
            substitutes that lookup knows, in a name or a property protected before, are not
            replaced again.  Without tags, each value of the text takes its substitute, as the names
            and the properties do, the descriptions of the pictures included, and
            engine_result::substitutions lists it with text true, once for each place it is
            written in: its value comes back only from the substitutes the caller keeps.  The
            substitutes of known, those of a file protected without tags before, stay as they are.
        */
        engine_result anonymize(const std::string& bytes, const std::string& name,
            const picture_handler& pictures, const substitute_lookup& lookup = substitute_lookup(),
            bool tags = true, const std::vector<substitution>& known = std::vector<substitution>()) const;
        /*
            Puts back the values of the tags, those of the substitutes that lookup knows in the
            names and the properties, and those of the substitutes of text in the text, a file
            protected without tags.  A picture cannot come back into a document: its marker leaves
            with the tags of its description (from a run of Word, with the gray text box of a
            slide, from over the gray box of a PDF, which stays), and so does the text file of a
            picture from an archive, report["pictures_removed"] counting them; the service gives
            back the picture of a text file alone from its vault.
        */
        engine_result restore(const std::string& bytes, const std::string& name,
            const substitute_lookup& lookup = substitute_lookup(),
            const std::vector<substitution>& text = std::vector<substitution>()) const;
        /*
            The bytes of the entry at path in the archive bytes named name, read as anonymize()
            reads it: path is entry_written::original, the archives inside joined by '!'.  Throws
            std::runtime_error when the archive holds no such entry.
        */
        std::string extract(const std::string& bytes, const std::string& name, const std::string& path) const;

        // A name alone, a folder for instance: its values become substitutes, its extension stays.
        std::string anonymize_name(const std::string& name, const substitute_lookup& lookup,
            std::vector<substitution>& made, json& report) const;
        std::string restore_name(const std::string& name, const substitute_lookup& lookup, json& report) const;

        // Without tags, the values take their substitutes, which made receives.
        std::string anonymize_text(const std::string& text, json& report, bool tags = true,
            std::vector<substitution>* made = nullptr) const;
        // A text keeps the picture markers it holds, which its reader sees; substitutes come back too.
        std::string restore_text(const std::string& text, json& report,
            const std::vector<substitution>& substitutes = std::vector<substitution>()) const;
        /*
            [{"begin", "end", "type"}] of the tags of a protected text that this key restores and of
            the substitutes of the list, and of the values that restore_text() wrote in restored in
            their place in tagged: byte offsets of the text, for the colouring of the web
            application. The values are given when the tags and the substitutes alone account for
            the restoration (a tag written in a link, which the restoration also reads, gives none).
        */
        json tag_spans(const std::string& text,
            const std::vector<substitution>& substitutes = std::vector<substitution>()) const;
        json restored_spans(const std::string& tagged, const std::string& restored,
            const std::vector<substitution>& substitutes = std::vector<substitution>()) const;
        // [{"begin", "end", "type", "confidence", "text"}] of the values of a text.
        json detect(const std::string& text) const;

        /*
            Replaces the imposed terms (common/imposed_terms.h), which every file, text and name
            processed afterwards reads before anything else; a call in progress keeps the terms
            it started with for the detectors it already made.
        */
        void set_terms(const std::vector<imposed_term>& terms);
        std::size_t term_count() const;

        std::string key_fingerprint() const;
        // {"tagger", "span_model", "size"}
        static json models();

    private:
        struct impl;
        std::shared_ptr<impl> p_;
    };
}

#endif // PSEUDALIS_ENGINE_ENGINE_H_
