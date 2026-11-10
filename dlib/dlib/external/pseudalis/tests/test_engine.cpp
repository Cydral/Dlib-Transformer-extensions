// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// Tests of the engine through its C interface: documents, pictures and archives.

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <dlib/crypto/crypto_utils.h>
#include <dlib/crypto/sha256.h>
#include <dlib/documents/document_image.h>
#include <dlib/zip/zip_archive.h>

#include "../common/engine_api.h"
#include "../common/json.h"
#include "../common/version.h"
#include "../engine/archives.h"

using pseudalis::json;

namespace
{
    int failures = 0;

#define CHECK(cond) check((cond), #cond, __LINE__)
    void check(bool ok, const char* what, int line)
    {
        if (ok) return;
        ++failures;
        std::cerr << "test_engine.cpp:" << line << ": failed: " << what << std::endl;
    }

    std::vector<std::string> seen_pictures;
    // The identifier and the SHA-256 of the last picture described.
    std::string last_id, last_sha256;
    // The description that the callback gives, which names what a careless model could name.
    const std::string careless = "Portrait of Jean Dupont [smiling], jean.dupont@example.fr, 06 12 34 56 78";
    std::string description_given = careless;

    // The text of element i of the array key of j.
    std::string item(const json& j, const char* key, size_t i)
    {
        const json& a = j[key];
        return i < a.size() ? a.elements()[i].str() : std::string();
    }

    int PSD_CALL describe(void*, const psd_picture* p, psd_buffer* d)
    {
        seen_pictures.push_back(std::string(p->media_type) + " " + p->location + " " + p->id);
        last_id = p->id;
        last_sha256 = p->sha256;
        const std::string& text = description_given;
        d->data = psd_alloc(text.size());
        std::memcpy(d->data, text.data(), text.size());
        d->size = text.size();
        return 1;
    }

    // The substitutes written so far, with their tags, as the service records them.
    std::map<std::string, std::vector<std::string>> recorded;

    int PSD_CALL lookup(void*, const char* candidate, size_t size, psd_buffer* tags)
    {
        const auto it = recorded.find(std::string(candidate, size));
        if (it == recorded.end()) return 0;
        std::string all;
        for (const std::string& t : it->second) all += (all.empty() ? "" : "\n") + t;
        tags->data = psd_alloc(all.size());
        std::memcpy(tags->data, all.data(), all.size());
        tags->size = all.size();
        return 1;
    }

    // The error of the last call of process(), which it prints unless the call was meant to fail.
    std::string last_error;

    std::string sha256_of(const std::string& data)
    {
        const std::array<unsigned char, 32> d = dlib::sha256_digest(data);
        return dlib::to_hex(d.data(), d.size());
    }

    // The entry path of the archive data named name, as the service asks the engine for it.
    bool extract(psd_engine* e, const std::string& name, const std::string& data, const std::string& path,
                 std::string& output)
    {
        json req = json::object();
        req["operation"] = "extract";
        req["name"] = name;
        req["entry"] = path;
        psd_buffer r = { nullptr, 0 }, o = { nullptr, 0 }, err = { nullptr, 0 };
        const int ok = psd_process(e, req.dump().c_str(), data.data(), data.size(), nullptr, nullptr, nullptr, &r,
                                   &o, &err);
        output = ok ? std::string(o.data, o.size) : std::string();
        psd_free(&r);
        psd_free(&o);
        psd_free(&err);
        return ok == 1;
    }

    // The entry of the result res of an archive whose original path is original, or null.
    const json* entry_of(const json& res, const std::string& original)
    {
        for (const json& w : res["entries"].elements())
            if (w["original"].str() == original) return &w;
        return nullptr;
    }

    bool process(psd_engine* e, const std::string& op, const std::string& name, const std::string& data, json& result,
                 std::string& output, bool pictures = true, bool failure_expected = false, json req = json::object())
    {
        req["operation"] = op;
        req["name"] = name;
        psd_buffer r = { nullptr, 0 }, o = { nullptr, 0 }, err = { nullptr, 0 };
        const int ok = psd_process(e, req.dump().c_str(), data.data(), data.size(), pictures ? describe : nullptr,
                                   lookup, nullptr, &r, &o, &err);
        last_error = !ok && err.data ? std::string(err.data, err.size) : std::string();
        if (!ok && !failure_expected) std::cerr << "psd_process: " << last_error << std::endl;
        result = ok ? json::parse_or_null(std::string(r.data, r.size)) : json();
        output = ok ? std::string(o.data, o.size) : std::string();
        psd_free(&r);
        psd_free(&o);
        psd_free(&err);
        // The substitutes of the names and the properties, as the service records them for any file.
        for (const json& s : result["substitutions"].elements())
        {
            if (s["text"].as_bool(false)) continue;
            std::vector<std::string>& tags = recorded[s["substitute"].str()];
            if (std::find(tags.begin(), tags.end(), s["tag"].str()) == tags.end()) tags.push_back(s["tag"].str());
        }
        return ok == 1;
    }

    // The substitutes of the text of a result, without tags, as the record of a file keeps them.
    json text_substitutes(const json& result)
    {
        json list = json::array();
        for (const json& s : result["substitutions"].elements())
            if (s["text"].as_bool(false))
            {
                json one = json::object();
                one["substitute"] = s["substitute"];
                one["tag"] = s["tag"];
                list.push_back(one);
            }
        return list;
    }
}

int main()
{
    CHECK(psd_api_version() == PSEUDALIS_ENGINE_API_VERSION);
    // The version and the build, which the service requires equal to its own.
    const std::string build = psd_build();
    CHECK(build.compare(0, std::strlen(PSEUDALIS_VERSION "+"), PSEUDALIS_VERSION "+") == 0 &&
          build.size() > std::strlen(PSEUDALIS_VERSION "+"));
    psd_buffer err = { nullptr, 0 };
    CHECK(psd_create("short", 5, "{}", &err) == nullptr && err.data && std::strstr(err.data, "32 bytes"));
    psd_free(&err);
    const std::string key(32, '\x5a');
    psd_engine* e = psd_create(key.data(), key.size(), "{\"date_order\": \"dmy\"}", &err);
    CHECK(e != nullptr);
    if (!e) return 1;

    json res;
    std::string out;
    // A text file.
    const std::string letter =
        "Bonjour,\nJe suis Jean Dupont, joignable au 06 12 34 56 78 ou jean.dupont@example.fr.\n";
    CHECK(process(e, "anonymize", "lettre.txt", letter, res, out));
    CHECK(res["kind"].str() == "txt" && out.find("Dupont") == std::string::npos
        && out.find("[[PERSON:") != std::string::npos);
    CHECK(res["report"]["total"].as_int() >= 3);
    std::string back;
    CHECK(process(e, "restore", "lettre.txt", out, res, back));
    CHECK(back == letter);

    // A picture alone becomes a text file with its protected description.
    const std::string png = dlib::encode_png(std::string("\x10\x20\x30\x40\x50\x60", 6), 2, 1, 3);
    seen_pictures.clear();
    CHECK(process(e, "anonymize", "photo.png", png, res, out));
    CHECK(res["name"].str() == "photo.png.txt" && res["kind"].str() == "image" && res["pictures"].size() == 1);
    CHECK(seen_pictures.size() == 1 && seen_pictures[0].find("image/png photo.png ") == 0);
    const std::string head = "[Image@" + item(res, "pictures", 0) + " : ";
    CHECK(out.compare(0, head.size(), head) == 0);
    CHECK(out.find("Dupont") == std::string::npos && out.find("(smiling)") != std::string::npos);
    // The values of the description become tags, counted in the report.
    CHECK(out.find("jean.dupont@") == std::string::npos && out.find("06 12 34 56 78") == std::string::npos);
    CHECK(out.find("[[EMAIL:") != std::string::npos && res["report"]["total"].as_int() >= 3);
    /* Its identifier: a digest keyed by the key of the engine, which the SHA-256 alone does not
       give, the same for the same picture and the same key, another under another key. */
    const std::string id = item(res, "pictures", 0);
    CHECK(id == last_id && id.size() == 16 && id.find_first_not_of("0123456789abcdef") == std::string::npos);
    CHECK(last_sha256.size() == 64 && id != last_sha256.substr(0, 16));
    CHECK(process(e, "anonymize", "photo.png", png, res, out) && item(res, "pictures", 0) == id);
    {
        const std::string other_key(32, '\x33');
        psd_engine* other = psd_create(other_key.data(), other_key.size(), "{}", &err);
        CHECK(other != nullptr);
        if (other)
        {
            CHECK(process(other, "anonymize", "photo.png", png, res, out) && item(res, "pictures", 0).size() == 16 &&
                  item(res, "pictures", 0) != id);
            psd_destroy(other);
        }
        psd_free(&err);
    }
    // A picture inside a Word document: its protected description in its place.
    {
        const std::string rels_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/";
        dlib::zip_archive_writer w;
        w.add("[Content_Types].xml",
              "<?xml version=\"1.0\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
              "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
              "<Default Extension=\"png\" ContentType=\"image/png\"/><Override PartName=\"/word/document.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
              "</Types>");
        w.add("_rels/.rels", "<?xml version=\"1.0\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/"
              "package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"" + rels_type + "officeDocument\" "
              "Target=\"word/document.xml\"/></Relationships>");
        w.add("word/_rels/document.xml.rels", "<?xml version=\"1.0\"?><Relationships xmlns=\"http://schemas."
              "openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId5\" Type=\"" + rels_type +
              "image\" Target=\"media/image1.png\"/></Relationships>");
        w.add("word/document.xml",
              "<?xml version=\"1.0\"?><w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/"
              "main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" xmlns:a=\"http://"
              "schemas.openxmlformats.org/drawingml/2006/main\" xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/"
              "2006/wordprocessingDrawing\" xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">"
              "<w:body><w:p><w:r><w:t>Lettre de Marie Curie</w:t></w:r></w:p><w:p><w:r><w:t>Madame Jeanne "
              "Marchand habite à Quimper.</w:t></w:r></w:p><w:p><w:r><w:drawing><wp:inline>"
              "<a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed=\"rId5\"/></pic:blipFill></pic:pic>"
              "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p></w:body></w:document>");
        w.add("word/media/image1.png", png);
        const std::string docx = w.finish();
        seen_pictures.clear();
        CHECK(process(e, "anonymize", "lettre.docx", docx, res, out));
        const dlib::zip_archive_reader z(out);
        const std::string xml = z.contains("word/document.xml") ? z.read("word/document.xml") : std::string();
        CHECK(seen_pictures.size() == 1 && xml.find("[Image@") != std::string::npos);
        CHECK(xml.find("Dupont") == std::string::npos && xml.find("jean.dupont@") == std::string::npos &&
              xml.find("Curie") == std::string::npos);
        CHECK(res["report"]["total"].as_int() >= 4);
        /* A description that names bare a person of the text, which its reading alone leaves: Word
           asks for it before its text is read, and the document is protected again with what its
           text taught the detector, Albireo asked once. */
        description_given = "Marchand in her garden, beside the bench";
        seen_pictures.clear();
        CHECK(process(e, "anonymize", "lettre.docx", docx, res, out));
        description_given = careless;
        const dlib::zip_archive_reader again(out);
        const std::string xml2 = again.contains("word/document.xml") ? again.read("word/document.xml") : "";
        CHECK(seen_pictures.size() == 1 && xml2.find("in her garden") != std::string::npos &&
              xml2.find("Marchand") == std::string::npos && xml2.find("[Image@" + id + " : ") != std::string::npos);
        // Restored from its tags, its original forgotten: the marker leaves the document, the values come back.
        std::string restored_docx;
        CHECK(process(e, "restore", "lettre.docx", out, res, restored_docx));
        const dlib::zip_archive_reader rz(restored_docx);
        const std::string rxml = rz.contains("word/document.xml") ? rz.read("word/document.xml") : std::string();
        CHECK(rxml.find("[Image@") == std::string::npos && rxml.find("in her garden") == std::string::npos &&
              rxml.find("Marie Curie") != std::string::npos && rxml.find("Jeanne Marchand") != std::string::npos);
        CHECK(res["report"]["pictures_removed"].as_int() == 1);
    }
    // A text written elsewhere from protected values: its tags come back, the marker of a picture leaves.
    {
        std::string tagged, answered;
        CHECK(process(e, "anonymize", "note.txt", "Photo de Jean Dupont.\n", res, tagged));
        tagged.erase(tagged.find_last_not_of("\r\n") + 1);
        const std::string answer = "Voir [Image@0123456789abcdef : " + tagged + "] puis " + tagged + "\n";
        CHECK(process(e, "restore", "reponse.txt", answer, res, answered));
        CHECK(answered == "Voir puis Photo de Jean Dupont.\n" && res["report"]["pictures_removed"].as_int() == 1 &&
              res["report"]["tags_restored"].as_int() >= 1);
        // At the end and at the start of a line, the space beside the marker leaves with it.
        const std::string edges = "Voir " + tagged + " [Image@0123456789abcdef]\n[Image@0123456789abcdef] Fin\n";
        CHECK(process(e, "restore", "reponse.txt", edges, res, answered));
        CHECK(answered == "Voir Photo de Jean Dupont.\nFin\n" && res["report"]["pictures_removed"].as_int() == 2);
    }
    // Without description.
    CHECK(process(e, "anonymize", "photo.png", png, res, out, false));
    CHECK(out == "[Image@" + item(res, "pictures", 0) + "]\r\n");

    // A ZIP archive: documents protected, pictures as text, other entries left out, a TAR.GZ inside.
    std::vector<pseudalis::archive_entry> inner(1);
    inner[0].name = std::string(120, 'd') + "/note.txt";   // a name longer than a TAR header holds
    inner[0].data = "Note de Marie Curie, 12 rue des Lilas, 75005 Paris.\n";
    const std::string tgz = pseudalis::write_archive(pseudalis::archive_kind::tar_gz, inner);
    const std::vector<pseudalis::archive_entry> again = pseudalis::read_archive(pseudalis::archive_kind::tar_gz, tgz);
    CHECK(again.size() == 1 && again[0].name == inner[0].name && again[0].data == inner[0].data);
    dlib::zip_archive_writer w;
    w.add("dossier/", "", false);
    w.add("dossier/lettre.txt", letter);
    w.add("dossier/photo.png", png);
    w.add("dossier/photo.png.txt", "already there");
    w.add("outil.exe", std::string("MZ\x90\0\x03\0\0\0", 8) + std::string(64, '\0'));
    w.add("archive.tar.gz", tgz);
    const std::string zip = w.finish();
    seen_pictures.clear();
    CHECK(process(e, "anonymize", "envoi.zip", zip, res, out));
    CHECK(res["kind"].str() == "zip");
    CHECK(res["report"]["dropped"].size() == 1 && item(res["report"], "dropped", 0) == "outil.exe");
    CHECK(seen_pictures.size() == 1 && seen_pictures[0].find("image/png photo.png ") == 0);
    {
        const dlib::zip_archive_reader z(out);
        CHECK(!z.contains("outil.exe") && !z.contains("dossier/photo.png") && z.contains("dossier/"));
        CHECK(z.contains("dossier/photo (2).png.txt") || z.contains("dossier/photo.png (2).txt"));
        CHECK(z.read("dossier/photo.png.txt") == "already there");
        CHECK(z.read("dossier/lettre.txt").find("Dupont") == std::string::npos);
        const std::vector<pseudalis::archive_entry> t =
            pseudalis::read_archive(pseudalis::archive_kind::tar_gz, z.read("archive.tar.gz"));
        CHECK(t.size() == 1 && t[0].name == inner[0].name && t[0].data.find("Curie") == std::string::npos);
        /* Each entry written, by which a copy extracted from the archive is known: its paths in
           both archives, the SHA-256 of what was written and its kind, the entries of the TAR.GZ
           after it; the entries left out have none. */
        CHECK(res["entries"].size() == 5 && entry_of(res, "outil.exe") == nullptr);
        const json* l = entry_of(res, "dossier/lettre.txt");
        CHECK(l && (*l)["name"].str() == "dossier/lettre.txt" && (*l)["kind"].str() == "txt" &&
              (*l)["sha256"].str() == sha256_of(z.read("dossier/lettre.txt")));
        const json* ph = entry_of(res, "dossier/photo.png");
        CHECK(ph && (*ph)["kind"].str() == "image" && z.contains((*ph)["name"].str()) &&
              (*ph)["sha256"].str() == sha256_of(z.read((*ph)["name"].str())));
        const json* a = entry_of(res, "archive.tar.gz");
        CHECK(a && (*a)["kind"].str() == "tar.gz" && (*a)["sha256"].str() == sha256_of(z.read("archive.tar.gz")));
        const json* n = entry_of(res, "archive.tar.gz!" + inner[0].name);
        CHECK(n && (*n)["name"].str() == "archive.tar.gz!" + inner[0].name &&
              (*n)["sha256"].str() == sha256_of(t[0].data));
    }
    // The original of an entry, in the archive and in an archive inside it; none for an entry left out.
    {
        std::string original;
        CHECK(extract(e, "envoi.zip", zip, "dossier/lettre.txt", original) && original == letter);
        CHECK(extract(e, "envoi.zip", zip, "dossier/photo.png", original) && original == png);
        CHECK(extract(e, "envoi.zip", zip, "archive.tar.gz", original) && original == tgz);
        CHECK(extract(e, "envoi.zip", zip, "archive.tar.gz!" + inner[0].name, original) && original == inner[0].data);
        CHECK(!extract(e, "envoi.zip", zip, "dossier/absent.txt", original));
        CHECK(!extract(e, "envoi.zip", zip, "dossier/lettre.txt!x", original));
    }
    CHECK(process(e, "restore", "envoi.zip", out, res, back));
    {
        const dlib::zip_archive_reader z(back);
        CHECK(z.read("dossier/lettre.txt") == letter);
        // The text file of the picture, which no restoration gives back, leaves the archive.
        CHECK(res["report"]["pictures_removed"].as_int() == 1 && z.read("dossier/photo.png.txt") == "already there");
        for (const pseudalis::archive_entry& x : pseudalis::read_archive(pseudalis::archive_kind::zip, back))
            CHECK(x.data.find("[Image@") == std::string::npos);
        const std::vector<pseudalis::archive_entry> t =
            pseudalis::read_archive(pseudalis::archive_kind::tar_gz, z.read("archive.tar.gz"));
        CHECK(t.size() == 1 && t[0].data == inner[0].data);
    }

    // A restoration keeps the entries of another kind as they are, and reads a text under any name.
    {
        std::vector<pseudalis::archive_entry> entries = pseudalis::read_archive(pseudalis::archive_kind::zip, out);
        std::string tagged;
        for (const pseudalis::archive_entry& x : entries)
            if (x.name == "dossier/lettre.txt") tagged = x.data;
        pseudalis::archive_entry tool, script;
        tool.name = "outil.exe";
        tool.data = std::string("MZ\x90\0\x03\0\0\0", 8) + std::string(64, '\0');
        script.name = "dossier/script.py";
        script.data = "note = \"\"\"" + tagged + "\"\"\"\n";
        entries.push_back(tool);
        entries.push_back(script);
        std::string zback;
        CHECK(process(e, "restore", "envoi.zip", pseudalis::write_archive(pseudalis::archive_kind::zip, entries), res,
                      zback));
        const dlib::zip_archive_reader z(zback);
        CHECK(z.contains("outil.exe") && z.read("outil.exe") == tool.data);
        CHECK(z.read("dossier/script.py") == "note = \"\"\"" + letter + "\"\"\"\n");
        CHECK(z.read("dossier/lettre.txt") == letter && res["report"]["dropped"].size() == 0);
        CHECK(process(e, "restore", "script.py", script.data, res, back));
        CHECK(back == "note = \"\"\"" + letter + "\"\"\"\n");
    }

    // The kind of a file: its content must be the kind its name gives, and decides without extension.
    {
        dlib::zip_archive_writer od;
        od.add("mimetype", "application/vnd.oasis.opendocument.text", false);
        od.add("content.xml", "<office:text>Jean Dupont</office:text>");
        const std::string odt = od.finish();
        CHECK(!process(e, "anonymize", "lettre.odt", odt, res, out, true, true));
        CHECK(last_error.find("not supported") != std::string::npos);
        CHECK(!process(e, "anonymize", "lettre.docx", letter, res, out, true, true));
        CHECK(last_error.find("does not match") != std::string::npos);
        CHECK(!process(e, "anonymize", "photo.png", letter, res, out, true, true));
        CHECK(process(e, "anonymize", "clients.sql", "INSERT INTO c VALUES ('jean.dupont@example.fr');\n", res, out));
        CHECK(res["kind"].str() == "txt" && out.find("dupont") == std::string::npos);
        CHECK(process(e, "anonymize", "LISEZMOI", letter, res, out));
        CHECK(out.find("Dupont") == std::string::npos);
        CHECK(process(e, "anonymize", "Lettre à M. Dupont", letter, res, out));
        CHECK(res["kind"].str() == "txt" && out.find("Dupont") == std::string::npos);
        // An Office document renamed .zip is read as an archive.
        dlib::zip_archive_writer ox;
        ox.add("[Content_Types].xml", "<Types><Default ContentType=\"wordprocessingml\"/></Types>");
        ox.add("word/document.xml", "<w:document><w:t>Jean Dupont</w:t></w:document>");
        CHECK(process(e, "anonymize", "rapport.zip", ox.finish(), res, out));
        CHECK(res["kind"].str() == "zip");
    }

    // A name of no kind: processed when its content is text through and through, as dlib reads it.
    {
        CHECK(process(e, "anonymize", "journal.log2", letter, res, out));
        CHECK(res["kind"].str() == "txt" && out.find("Dupont") == std::string::npos);
        CHECK(process(e, "restore", res["name"].str(), out, res, back) && back == letter);
        CHECK(process(e, "anonymize", "BMW Jean Dupont.notes", "BMW de Jean Dupont, 06 12 34 56 78.\n", res, out));
        CHECK(out.find("Dupont") == std::string::npos);
        // UTF-16 with its byte order mark, written back in UTF-16.
        std::string utf16("\xFF\xFE", 2);
        for (char c : std::string("Jean Dupont habite 12 rue de la Paix, 75002 Paris.\r\n"))
            utf16 += std::string(1, c) + std::string(1, '\0');
        CHECK(process(e, "anonymize", "export.reg", utf16, res, out));
        CHECK(out.compare(0, 2, "\xFF\xFE") == 0 && out.find(std::string("D\0u\0p\0o\0n\0t", 11)) == std::string::npos);
        // A message with a part in base64, a document of RTF, binary bytes: of another kind.
        std::string message = "From: Jean Dupont <jean.dupont@example.fr>\r\nSubject: Contrat\r\n"
                              "Content-Type: application/pdf\r\nContent-Transfer-Encoding: base64\r\n\r\n";
        for (int i = 0; i < 20; ++i) message += std::string(76, 'Q') + "\r\n";
        CHECK(!process(e, "anonymize", "message.eml", message, res, out, true, true));
        CHECK(last_error.find("encoded parts") != std::string::npos);
        CHECK(process(e, "anonymize", "message.eml", "From: Jean Dupont <jean.dupont@example.fr>\r\n\r\nBonjour.\r\n",
                      res, out));
        CHECK(out.find("dupont") == std::string::npos);
        CHECK(!process(e, "anonymize", "encoded.eml", "Subject: =?UTF-8?B?Sm9obiBEb2U=?=\r\n\r\nBonjour.\r\n", res,
                       out, true, true));
        CHECK(!process(e, "anonymize", "lettre.rtf", "{\\rtf1\\ansi Fran\\'e7ois Dupont}", res, out, true, true));
        CHECK(!process(e, "anonymize", "cle.key", "-----BEGIN KEY-----\n" + std::string(1100, 'A') + "\n", res, out,
                       true, true));
        CHECK(!process(e, "anonymize", "donnees.bin", std::string("Jean Dupont\0\x01\x02", 14), res, out, true,
                       true));
        CHECK(!process(e, "anonymize", "photo.jpeg2", png, res, out, true, true));
        CHECK(!process(e, "anonymize", "journal.log2", "Jean Dupont\x1b[31m rouge\n", res, out, true, true));
        // In an archive, the text of no kind is protected, the binary file left out.
        dlib::zip_archive_writer zw;
        zw.add("notes/journal.log2", letter);
        zw.add("notes/donnees.bin", std::string("\0\x01\x02\x03", 4));
        CHECK(process(e, "anonymize", "envoi.zip", zw.finish(), res, out));
        CHECK(res["report"]["dropped"].size() == 1 && item(res["report"], "dropped", 0) == "notes/donnees.bin");
        const dlib::zip_archive_reader z(out);
        CHECK(z.contains("notes/journal.log2") && z.read("notes/journal.log2").find("Dupont") == std::string::npos);
    }

    // A text of a name of no kind that writes its values encoded is refused, with its reason.
    {
        auto encoded = [&](const char* name, const std::string& data) {
            const bool refused = !process(e, "anonymize", name, data, res, out, true, true);
            if (!refused || last_error.find("encoded parts") == std::string::npos)
                std::cerr << name << ": " << (refused ? last_error : std::string("accepted")) << std::endl;
            return refused && last_error.find("encoded parts") != std::string::npos;
        };
        auto accepted = [&](const char* name, const std::string& data) {
            const bool done = process(e, "anonymize", name, data, res, out, true, true);
            if (!done) std::cerr << name << ": " << last_error << std::endl;
            return done && out.find("Dupont") == std::string::npos;
        };
        std::string photo = "BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Jean Dupont\r\nPHOTO;ENCODING=b;TYPE=JPEG:\r\n";
        for (int i = 0; i < 30; ++i) photo += " " + std::string(74, 'Q') + "\r\n";
        CHECK(encoded("contact.vcf", photo + "END:VCARD\r\n"));
        CHECK(encoded("contacts.vcf2",
                      "VERSION:2.1\r\nFN;CHARSET=UTF-8;ENCODING=QUOTED-PRINTABLE:H=C3=A9l=C3=A8ne\r\n"));
        CHECK(encoded("carnet.ldif", "dn: cn=x,dc=example\ncn:: SMOpbMOobmUgRHVwcsOp\nmail: x@example.fr\n"));
        CHECK(encoded("annuaire.ldif", "dn: cn=x,dc=example\ncn::SMOpbMOobmUgRHVwcsOp\n"));
        CHECK(encoded("agenda.ics", "BEGIN:VCALENDAR\r\nSUMMARY:Rendez-vous avec Jean Dupont\r\nEND:VCALENDAR\r\n"));
        CHECK(encoded("export.reg", "Windows Registry Editor Version 5.00\r\n\r\n[HKEY_CURRENT_USER\\Software]\r\n"));
        CHECK(encoded("cert.pem", "Certificat de Jean Dupont\n-----BEGIN CERTIFICATE-----\nMIIBszCCAVmgAwIBAgIU\n"));
        CHECK(encoded("cle.pem", "Cle de Jean Dupont\r-----BEGIN PRIVATE KEY-----\rMIIEvQIBADANBgkqhkiG9w0B\r"));
        CHECK(encoded("piece.uue", "Voici le fichier.\nbegin 644 contrat.pdf\nM" + std::string(60, '5') +
                                   "\n`\nend\n"));
        CHECK(encoded("doc.ps2", "%!PS-Adobe-3.0\n(Jean Dupont) show\n"));
        CHECK(encoded("data.a85", "Jean Dupont <~" + std::string(70, 'G') + "~>\n"));
        CHECK(encoded("courrier.rtf", "  {\\rtf1\\ansi Fran\\'e7ois Dupont}"));
        CHECK(encoded("message.eml", "From: Jean Dupont\r\nContent-Transfer-Encoding:\r\n base64\r\n\r\nSm9obg==\r\n"));
        CHECK(encoded("reponse.eml", "> Content-Transfer-Encoding: quoted-printable\n> Fran=C3=A7ois\n"));
        CHECK(encoded("sujet.log2", "Subject: =?UTF-8*fr?Q?Fran=C3=A7ois_Dupont?=\n"));
        CHECK(encoded("app.log.1", "GET /cb?id_token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc HTTP/1.1\n"));
        CHECK(encoded("page.htm2", "<p>Jean Dupont</p><img src=\"data:image/png;base64," + std::string(80, 'A') +
                                   "\">\n"));
        std::string dump;
        for (int i = 0; i < 4; ++i)
            dump += "00000010  4a 65 61 6e 20 44 75 70  6f 6e 74 2c 20 30 36 20  |Jean Dupont, 06 |\n";
        CHECK(encoded("dump.hex", dump));
        // The bytes of a text in hexadecimal digits alone, 30 by line.
        const std::string said = "Jean Dupont habite 12 rue de la Paix, 75002 Paris. ";
        std::string digits;
        for (int i = 0; i < 16 * 30; ++i)
        {
            const unsigned char c = static_cast<unsigned char>(said[i % said.size()]);
            digits += "0123456789abcdef"[c >> 4];
            digits += "0123456789abcdef"[c & 15];
            if (i % 30 == 29) digits += "\n";
        }
        CHECK(encoded("octets.hexa", digits));
        // xxd with its groups of 4 digits, hexdump -C of UTF-16 rich in digits, the bytes of a string of Python.
        std::string wide;
        for (char ch : std::string("Jean Dupont, tel 06 12 34 56 78 et 07 98 76 54 32, Rennes"))
            wide += std::string(1, ch) + std::string(1, '\0');
        auto hex2 = [](unsigned char ch) {
            return std::string(1, "0123456789abcdef"[ch >> 4]) + "0123456789abcdef"[ch & 15];
        };
        std::string xxd, canonical, escaped = "b\"";
        for (size_t row = 0; row * 16 < wide.size(); ++row)
        {
            xxd += "00000000: ";
            canonical += "00000000  ";
            for (size_t k = row * 16; k < std::min(wide.size(), row * 16 + 16); ++k)
            {
                xxd += hex2(static_cast<unsigned char>(wide[k])) + (k % 2 ? " " : "");
                canonical += hex2(static_cast<unsigned char>(wide[k])) + " ";
            }
            xxd += " ................\n";
            canonical += " |................|\n";
        }
        for (char ch : wide) escaped += "\\x" + hex2(static_cast<unsigned char>(ch));
        CHECK(encoded("dump.xxd", xxd));
        CHECK(encoded("dump2.hex", canonical));
        CHECK(encoded("octets.py", "donnees = " + escaped + "\"\n"));
        // A byte order mark before the first line, markers inside strings, a folded value of LDIF.
        CHECK(encoded("piece2.uue", "\xEF\xBB\xBF" "begin 644 x\nM" + std::string(60, '5') + "\n`\nend\n"));
        CHECK(encoded("entree.ldif", "\xEF\xBB\xBF" "dn: uid=hdupre,dc=example\ncn:: SMOpbMOobmUgRHVwcsOp\n"));
        CHECK(encoded("annuaire2.ldif", "dn: uid=jdupont,dc=example\ncn: Jean Dupont\n"
                                        "description:: TsOpIGxlIDEyLzAzLzE5ODAsIDEyIHJ1ZSBkZSBsYSBQYWl4LCA3NTAwMiBQYX\n"
                                        " JpcywgMDYgMTIgMzQgNTYgNzg=\n"));
        CHECK(encoded("config.json5", "{\"nom\": \"Jean Dupont\", \"cert\": \"-----BEGIN CERTIFICATE-----\\n"
                                      "Q2VydGlmaWNhdCBkZSB0ZXN0IGRlIEplYW4gRHVwb250LCAxMiBydWUgZGUgbGEg"
                                      "\\n-----END CERTIFICATE-----\\n\"}\n"));
        CHECK(encoded("cle.c", "const char* cle = \"-----BEGIN PRIVATE KEY-----\\n\"\n"
                               "    \"MIIEvQIBADANBgkqhkiG9w0B\\n\";\n"));
        CHECK(encoded("notes.bib", "@misc{x, note={\\rtf1\\ansi Fran\\'e7ois M\\'fcller}}\n"));
        CHECK(encoded("partie.asc", "Jean Dupont\n-----BEGIN PGP MESSAGE, PART 1/2-----\n\nhQEMA3bS1zDzNQwJAQf/\n"));
        CHECK(encoded("ancien.pem", "-----BEGIN PRIVACY-ENHANCED MESSAGE-----\nProc-Type: 4,ENCRYPTED\n"));
        CHECK(encoded("carte.vcf2", "\fBEGIN:VCARD\r\nFN:Jean Dupont\r\nEND:VCARD\r\n"));
        std::string iban;
        for (int i = 0; i < 4; ++i)
            iban += "0000000" + std::to_string(i) + ": 4652 3736 2033 3030 3030 2030 3030 3030  FR76 3000 0000\n";
        CHECK(encoded("iban.xxd", iban));
        std::string json = "{\"photo\": \"";
        for (int i = 0; i < 20; ++i) json += "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJz\\/ejAx\\n";
        CHECK(encoded("fiche.json5", json + "\", \"nom\": \"Jean Dupont\"}\n"));
        // A vCard or an iCalendar inside a message, the bare encoding of vCard 2.1.
        CHECK(encoded("invitation.eml", "From: x@example.fr\r\nContent-Type: text/calendar\r\n\r\nBEGIN:VCALENDAR\r\n"
                                        "ATTENDEE:mailto:jean.dupo\r\n nt@example.fr\r\nEND:VCALENDAR\r\n"));
        CHECK(encoded("nom.txt2", "Nom de famille\r\nN;CHARSET=UTF-8;QUOTED-PRINTABLE:=44=75=70=6F=6E=74\r\n"));
        // Thousands of openings of encoded words without their end: read in one pass, then refused.
        std::string openings;
        while (openings.size() < (300u << 10)) openings += "=?a?b? ";
        CHECK(encoded("ouvertures.log2", openings + "\ntoken eyJhbGciOiJIUzI1NiJ9xx\n"));
        // A byte order mark of UTF-8 before bytes of Windows-1252, or UTF-16 with an unpaired surrogate, is not
        // text through and through; U+FFFD written in UTF-8 is a character of a text.
        CHECK(!process(e, "anonymize", "journal.log2", "\xEF\xBB\xBF" "Fran\xE7ois Dupont, 06 12 34 56 78\n", res,
                       out, true, true));
        const std::string broken("\xFF\xFEJ\0e\0a\0n\0\x00\xD8 \0D\0u\0p\0o\0n\0t\0", 26);
        CHECK(!process(e, "anonymize", "export.reg", broken, res, out, true, true));
        CHECK(accepted("remplacement.py", "print('\xEF\xBF\xBD')  # Jean Dupont\n"));
        // Code, tables of numbers and what only looks encoded stay texts.
        CHECK(accepted("main.cpp", "int main()\n{\n    std::move(x);\n    std::cout << \"Jean Dupont\";\n}\n"));
        std::string numbers;
        for (int i = 0; i < 40; ++i) numbers += std::to_string(10 + i) + ", ";
        CHECK(accepted("mesures.dat", "Jean Dupont\n" + numbers + "\n"));
        CHECK(accepted("notes.org", "title:: Réunion\ncollapsed:: true\n- Appeler Jean Dupont\n"));
        CHECK(accepted("chapitre.ltx", "%!TEX root = main.tex\n\\section{Jean Dupont}\n"));
        CHECK(accepted("lot.job", "begin 100 records\nJean Dupont\nend\n"));
        CHECK(accepted("astuces.wiki", "Windows Registry Editor : les astuces de Jean Dupont\n"));
        CHECK(accepted("cle.fpr", "Cle de Jean Dupont\nEmpreinte : 9e:10:7d:9d:37:2b:b6:82:6b:d8:1d:35:42:a4:19:d6\n"));
        CHECK(accepted("rapport.out", "Rapport de Jean Dupont\n" + std::string(1100, '=') + "\n" +
                                      std::string(80, '-') + "\n"));
        std::string paths = "Dossiers de Jean Dupont\n";
        for (int i = 0; i < 20; ++i) paths += "/home/jean/projets/rapport-annuel-de-la-societe/archives/annee-" +
                                              std::string(1, static_cast<char>('a' + i)) + "\n";
        CHECK(accepted("dossiers.lst", paths));
        std::string years = "Releves de Jean Dupont\n", hosts = "Poste de Jean Dupont\n";
        for (int i = 0; i < 6; ++i)
        {
            years += "2019 2020 2021 2022 2023 2024 2025 2026\n";
            hosts += "2001:0db8:85a3:0000:0000:8a2e:0370:733" + std::to_string(i) + "\n";
        }
        CHECK(accepted("annees.dat", years));
        std::string points = "Sequences de Jean Dupont\n";
        for (int i = 0; i < 6; ++i)
            points += "SYLLABE " + std::to_string(i) + "; 0B95 0BCD 0BB7 0BBE 0B95 0BCD 0BB7 0BBE\n";
        CHECK(accepted("sequences.uni", points));
        CHECK(accepted("reseau.conf2", hosts));
        CHECK(accepted("export.csv2", "nom;encodage;taille\nJean Dupont;base64;12\n"));
        // A program that only names the header of a block, as a constant, holds no block.
        CHECK(accepted("pem.c", "/* Jean Dupont */\n#define PEM_BEGIN_CRT \"-----BEGIN CERTIFICATE-----\\n\"\n"
                                "#define PEM_END_CRT \"-----END CERTIFICATE-----\\n\"\n"));
        // A text of an extension of the table keeps its rules: read as it lies.
        CHECK(accepted("cert.txt", "Certificat de Jean Dupont\n-----BEGIN CERTIFICATE-----\nMIIBszCCAVmgAwIBAgIU\n"
                                   "-----END CERTIFICATE-----\n"));
        CHECK(accepted("notes.md", "# Jean Dupont\n\ncollapsed:: true\nContent-Transfer-Encoding: base64\n"));
    }

    // Names: their values become substitutes, which the lookup gives back.
    CHECK(process(e, "anonymize", "Lettre_jean.dupont@example.fr_2024-03-12.txt", letter, res, out));
    const std::string renamed = res["name"].str();
    CHECK(renamed.find("dupont") == std::string::npos && renamed.find("2024-03-12") == std::string::npos);
    CHECK(renamed.compare(0, 7, "Lettre_") == 0 && renamed.size() > 4 && renamed.substr(renamed.size() - 4) == ".txt");
    CHECK(renamed.find_first_of("<>:\"/\\|?*[]") == std::string::npos);
    CHECK(res["substitutions"].size() >= 2 && res["report"]["substitutes"].as_int() >= 2);
    CHECK(process(e, "restore", renamed, out, res, back));
    CHECK(back == letter && res["name"].str() == "Lettre_jean.dupont@example.fr_2024-03-12.txt");
    CHECK(res["report"]["substitutes_restored"].as_int() >= 2);
    CHECK(process(e, "anonymize_name", "", "Dossier jean.dupont@example.fr", res, out));
    CHECK(out.find("dupont") == std::string::npos && out.compare(0, 8, "Dossier ") == 0 && res["name"].str() == out);
    std::string folder_back;
    CHECK(process(e, "restore_name", "", out, res, folder_back));
    CHECK(folder_back == "Dossier jean.dupont@example.fr");
    // The common word that opens a name stays: it names the document, not the person.
    CHECK(process(e, "anonymize_name", "", "Contrat Jean Dupont", res, out));
    CHECK(out.find("Jean") == std::string::npos && out.compare(0, 8, "Contrat ") == 0);
    CHECK(process(e, "restore_name", "", out, res, folder_back));
    CHECK(folder_back == "Contrat Jean Dupont");
    // A value before an extension of no kind is found too, with or without the extension.
    CHECK(process(e, "anonymize", "journal jean.dupont@example.fr.log2", letter, res, out));
    const std::string journal = res["name"].str();
    CHECK(journal.find("dupont") == std::string::npos && journal.find(".log2") != std::string::npos);
    CHECK(process(e, "restore", res["name"].str(), out, res, back));
    CHECK(back == letter && res["name"].str() == "journal jean.dupont@example.fr.log2");
    CHECK(process(e, "anonymize_name", "", "Appeler 06.12.34.56.78", res, out));
    CHECK(out.find("34.56") == std::string::npos);
    // A name of no kind that its substitutes lengthen past 255 characters keeps its extension.
    std::string long_name;
    for (int i = 0; i < 18; ++i) long_name += "Jean Dupont ";
    CHECK(process(e, "anonymize", long_name + "jean.dupont@example.fr.log2", letter, res, out));
    const std::string cut = res["name"].str();
    CHECK(cut.size() <= 255 && cut.size() > 5 && cut.compare(cut.size() - 5, 5, ".log2") == 0);
    CHECK(cut.find("dupont") == std::string::npos && cut.find("Dupont") == std::string::npos);
    // The folders of an archive, the same in all their entries.
    {
        dlib::zip_archive_writer zw;
        zw.add("jean.dupont@example.fr/", "", false);
        zw.add("jean.dupont@example.fr/lettre.txt", letter);
        zw.add("jean.dupont@example.fr/copie.txt", letter);
        std::string zout, zback;
        CHECK(process(e, "anonymize", "envoi.zip", zw.finish(), res, zout));
        const dlib::zip_archive_reader z(zout);
        std::string first;
        bool same = true;
        for (const auto& entry : z.entries())
        {
            CHECK(entry.name.find("dupont") == std::string::npos);
            const std::string folder = entry.name.substr(0, entry.name.find('/'));
            if (first.empty()) first = folder;
            same = same && folder == first;
        }
        CHECK(same && z.entries().size() == 3);
        CHECK(process(e, "restore", "envoi.zip", zout, res, zback));
        const dlib::zip_archive_reader r(zback);
        CHECK(r.contains("jean.dupont@example.fr/lettre.txt") && r.read("jean.dupont@example.fr/copie.txt") == letter);
    }

    // Unsupported content, refused with its reason, and text.
    CHECK(!process(e, "anonymize", "outil.exe", std::string("MZ\x90\0\x03\0\0\0", 8) + std::string(64, '\0'),
                   res, out, true, true));
    CHECK(last_error.find("not supported") != std::string::npos);
    CHECK(process(e, "anonymize_text", "prompt", "Écrire à Jean Dupont", res, out));
    CHECK(out.find("Dupont") == std::string::npos);
    CHECK(process(e, "detect", "prompt", "Écrire à Jean Dupont", res, out));
    CHECK(res["spans"].size() >= 1 && res["spans"].elements()[0]["type"].str() == "PERSON");

    // The places of the tags written and of the values restored, in bytes of the UTF-8 text.
    {
        const std::string text = "Écrire à Jean Dupont, jean.dupont@example.fr";
        CHECK(process(e, "anonymize_text", "prompt", text, res, out) && res["tags"].size() >= 2);
        bool tags = true;
        for (const json& s : res["tags"].elements())
        {
            const std::size_t b = static_cast<std::size_t>(s["begin"].as_int());
            const std::size_t n = static_cast<std::size_t>(s["end"].as_int()) - b;
            tags = tags && b + n <= out.size() && n > 4 && out.compare(b, 2, "[[") == 0 &&
                   out.compare(b + n - 2, 2, "]]") == 0;
        }
        CHECK(tags);
        std::string back;
        CHECK(process(e, "restore_text", "prompt", out, res, back) && back == text);
        std::vector<std::string> values;
        for (const json& s : res["restored"].elements())
            values.push_back(back.substr(static_cast<std::size_t>(s["begin"].as_int()),
                                         static_cast<std::size_t>(s["end"].as_int() - s["begin"].as_int())));
        CHECK(std::find(values.begin(), values.end(), "Jean Dupont") != values.end() &&
              std::find(values.begin(), values.end(), "jean.dupont@example.fr") != values.end());
        // A text in Windows-1252 gets no places, which the service could not keep in its answer.
        CHECK(process(e, "anonymize_text", "prompt", "\xC9t\xE9 de Jean Dupont", res, out) &&
              out.find("Dupont") == std::string::npos && res["tags"].size() == 0);
    }

    // The imposed terms: replaced before anything else, in the texts, the files and the names.
    {
        const std::string terms = "[{\"term\": \"Projet Alpha\", \"type\": \"ORG\"}, "
                                  "{\"term\": \"KX-77\", \"type\": \"ID\", \"case_sensitive\": true}]";
        CHECK(psd_set_terms(e, terms.data(), terms.size(), &err) == 1);
        psd_free(&err);
        const std::string text = "Le PROJET ALPHA de Jean Dupont suit la référence KX-77, pas kx-77.";
        CHECK(process(e, "detect", "prompt", text, res, out));
        bool project = false, reference = false, lower = false;
        for (const json& s : res["spans"].elements())
        {
            project = project || (s["text"].str() == "PROJET ALPHA" && s["type"].str() == "ORG");
            reference = reference || (s["text"].str() == "KX-77" && s["type"].str() == "ID");
            lower = lower || s["text"].str() == "kx-77";
        }
        CHECK(project && reference && !lower);
        CHECK(process(e, "anonymize", "note.txt", text, res, out));
        CHECK(out.find("ALPHA") == std::string::npos && out.find("KX-77") == std::string::npos &&
              out.find("kx-77") != std::string::npos && out.find("[[ORG:") != std::string::npos);
        std::string restored;
        CHECK(process(e, "restore", "note.txt", out, res, restored) && restored == text);
        CHECK(process(e, "anonymize_name", "", "Rapport Projet Alpha.docx", res, out));
        CHECK(out.find("Alpha") == std::string::npos);
        // A list the engine does not take changes nothing; an empty one removes the terms.
        const std::string wrong = "[{\"term\": \"Alpha\", \"type\": \"DATE\"}]";
        CHECK(psd_set_terms(e, wrong.data(), wrong.size(), &err) == 0 && err.data);
        psd_free(&err);
        const std::string plain = "la note KX-77 est close";
        CHECK(process(e, "anonymize_text", "prompt", plain, res, out) && out.find("KX-77") == std::string::npos);
        CHECK(psd_set_terms(e, "[]", 2, &err) == 1);
        psd_free(&err);
        CHECK(process(e, "anonymize_text", "prompt", plain, res, out) && out == plain);
    }

    // Without tags: substitutes alone, which the substitutes of the record alone give back.
    {
        json untagged = json::object();
        untagged["tags"] = false;
        std::string bare, restored;
        CHECK(process(e, "anonymize", "lettre.txt", letter, res, bare, true, false, untagged));
        CHECK(bare.find("[[") == std::string::npos && bare.find("Dupont") == std::string::npos &&
              bare.find("06 12 34 56 78") == std::string::npos);
        const json kept = text_substitutes(res);
        CHECK(kept.size() >= 3);
        // Without its substitutes, nothing comes back; with them, the letter.
        CHECK(process(e, "restore", "lettre.txt", bare, res, restored) && restored == bare &&
              res["report"]["tags_restored"].as_int() == 0 && res["report"]["substitutes_restored"].as_int() == 0);
        json known = json::object();
        known["substitutions"] = kept;
        CHECK(process(e, "restore", "lettre.txt", bare, res, restored, false, false, known) && restored == letter);
        CHECK(res["report"]["substitutes_restored"].as_int() == static_cast<long long>(kept.size()));
        // Protected again with its substitutes known, it stays as it is.
        json again = known;
        again["tags"] = false;
        std::string twice;
        CHECK(process(e, "anonymize", "lettre.txt", bare, res, twice, true, false, again) && twice == bare);
        // Protected again with tags, by a newer engine for instance: its old substitutes stay, and come back.
        json tagged_again = known;
        std::string mixed, mixed_back;
        CHECK(process(e, "anonymize", "lettre.txt", "Visa : Hélène Martin.\n" + bare, res, mixed, true, false,
                      tagged_again));
        CHECK(mixed.find("Martin") == std::string::npos && mixed.find("[[PERSON:") != std::string::npos);
        CHECK(process(e, "restore", "lettre.txt", mixed, res, mixed_back, false, false, known));
        CHECK(mixed_back == "Visa : Hélène Martin.\n" + letter);
        // Protected with tags, then again without them: its tags give way to substitutes, which come back.
        std::string tagged_letter, converted, converted_back;
        CHECK(process(e, "anonymize", "lettre.txt", letter, res, tagged_letter));
        CHECK(process(e, "anonymize", "lettre.txt", tagged_letter, res, converted, true, false, untagged));
        CHECK(converted.find("[[") == std::string::npos && converted.find("Dupont") == std::string::npos);
        json converted_known = json::object();
        converted_known["substitutions"] = text_substitutes(res);
        CHECK(process(e, "restore", "lettre.txt", converted, res, converted_back, false, false, converted_known) &&
              converted_back == letter);

        // A picture: its description takes substitutes too.
        CHECK(process(e, "anonymize", "photo.png", png, res, bare, true, false, untagged));
        CHECK(bare.find("[[") == std::string::npos && bare.find("Dupont") == std::string::npos &&
              bare.find("[Image@") == 0 && text_substitutes(res).size() >= 3);

        // A Word document: the runs of its text, its name with substitutes as with tags.
        dlib::zip_archive_writer w;
        w.add("[Content_Types].xml",
              "<?xml version=\"1.0\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
              "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
              "<Override PartName=\"/word/document.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
              "</Types>");
        w.add("_rels/.rels", "<?xml version=\"1.0\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/"
              "package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/"
              "officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>");
        w.add("word/document.xml",
              "<?xml version=\"1.0\"?><w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/"
              "main\"><w:body><w:p><w:r><w:t>Madame Jeanne Marchand habite à Quimper.</w:t></w:r></w:p></w:body>"
              "</w:document>");
        const std::string docx = w.finish();
        CHECK(process(e, "anonymize", "Jeanne Marchand.docx", docx, res, bare, true, false, untagged));
        const dlib::zip_archive_reader z(bare);
        const std::string xml = z.contains("word/document.xml") ? z.read("word/document.xml") : std::string();
        CHECK(xml.find("[[") == std::string::npos && xml.find("Marchand") == std::string::npos &&
              res["name"].str().find("Marchand") == std::string::npos);
        json docx_known = json::object();
        docx_known["substitutions"] = text_substitutes(res);
        CHECK(process(e, "restore", res["name"].str(), bare, res, restored, false, false, docx_known));
        const dlib::zip_archive_reader rz(restored);
        CHECK(rz.contains("word/document.xml") &&
              rz.read("word/document.xml").find("Madame Jeanne Marchand habite à Quimper.") != std::string::npos &&
              res["name"].str() == "Jeanne Marchand.docx");

        // A text: its substitutes listed, which the caller keeps to give it back.
        const std::string prompt = "Résumer le dossier de Jean Dupont, né le 12/03/1980 à Rennes.";
        CHECK(process(e, "anonymize_text", "text", prompt, res, bare, false, false, untagged));
        CHECK(bare.find("[[") == std::string::npos && bare.find("Dupont") == std::string::npos &&
              res["substitutions"].size() >= 2 && res["tags"].size() == res["substitutions"].size());
        json text_known = json::object();
        text_known["substitutions"] = res["substitutions"];
        const std::string answer = "Le dossier de " + bare.substr(bare.find("de ") + 3);
        CHECK(process(e, "restore_text", "text", bare, res, restored, false, false, text_known) && restored == prompt);
        CHECK(res["restored"].size() >= 2);
        CHECK(process(e, "restore_text", "text", answer, res, restored, false, false, text_known) &&
              restored.find("Jean Dupont") != std::string::npos);
    }

    // The imposed columns: the values of a CSV file below their names take substitutes, with tags elsewhere.
    {
        const std::string columns = "[{\"column\": \"Nom\", \"type\": \"PERSON\"}, "
                                    "{\"column\": \"courriel\", \"type\": \"EMAIL\"}, "
                                    "{\"column\": \"N° client\", \"type\": \"ID\"}]";
        CHECK(psd_set_columns(e, columns.data(), columns.size(), &err) == 1);
        psd_free(&err);
        const std::string csv = "Export du 12/03/2024\n"
                                "NOM;Courriel;N° client;Commentaire\n"
                                "Dupont;jean.dupont@exemple.fr;CL-2024-0042;Rappeler M. Dupont\n"
                                "Martin;h.martin@exemple.fr;CL-2024-0043;RAS\n";
        std::string bare, restored;
        CHECK(process(e, "anonymize", "clients.csv", csv, res, bare));
        CHECK(bare.find("NOM;Courriel;N° client;Commentaire\n") != std::string::npos);
        CHECK(bare.find("Dupont") == std::string::npos && bare.find("Martin") == std::string::npos &&
              bare.find("CL-2024-0042") == std::string::npos && bare.find("[[PERSON:") != std::string::npos);
        const json kept = text_substitutes(res);
        CHECK(kept.size() == 6 && res["report"]["values"]["ID"].as_int() == 2);
        // The cells hold the substitutes alone, which the record gives back.
        for (const json& s : kept.elements())
            CHECK(bare.find(";" + s["substitute"].str() + ";") != std::string::npos ||
                  bare.find("\n" + s["substitute"].str() + ";") != std::string::npos);
        CHECK(process(e, "restore", "clients.csv", bare, res, restored) && restored != csv &&
              restored.find("Rappeler M. Dupont") != std::string::npos);
        json known = json::object();
        known["substitutions"] = kept;
        CHECK(process(e, "restore", "clients.csv", bare, res, restored, false, false, known) && restored == csv);
        CHECK(res["report"]["substitutes_restored"].as_int() == 6);
        // A Windows-1252 file protected in plain ASCII comes back in that encoding when the caller tells it.
        const std::string accents = "Nom;Ville\nH\xE9l\xE8ne;Rennes\n";
        std::string ascii, back;
        CHECK(process(e, "anonymize", "accents.csv", accents, res, ascii) &&
              res["report"]["encoding"].str() == "windows-1252" &&
              std::all_of(ascii.begin(), ascii.end(), [](char c) { return (c & 0x80) == 0; }));
        json told = json::object();
        told["substitutions"] = text_substitutes(res);
        CHECK(process(e, "restore", "accents.csv", ascii, res, back, false, false, told) &&
              back == "Nom;Ville\nH\xC3\xA9l\xC3\xA8ne;Rennes\n" && res["report"]["encoding"].str() == "utf-8");
        told["ascii_encoding"] = "windows-1252";
        CHECK(process(e, "restore", "accents.csv", ascii, res, back, false, false, told) && back == accents &&
              res["report"]["encoding"].str() == "windows-1252");
        // A list the engine does not take changes nothing; an empty one removes the columns.
        const std::string wrong = "[{\"column\": \"Nom\", \"type\": \"NONE\"}]";
        CHECK(psd_set_columns(e, wrong.data(), wrong.size(), &err) == 0 && err.data);
        psd_free(&err);
        CHECK(process(e, "anonymize", "clients.csv", csv, res, bare) && text_substitutes(res).size() == 6);
        CHECK(psd_set_columns(e, "[]", 2, &err) == 1);
        psd_free(&err);
        CHECK(process(e, "anonymize", "clients.csv", csv, res, bare) && text_substitutes(res).size() == 0);
        // An IP column takes both forms, each address keeping its form, its prefix length and its port.
        const std::string ip = "[{\"column\": \"ip_address\", \"type\": \"IP\"}]";
        CHECK(psd_set_columns(e, ip.data(), ip.size(), &err) == 1);
        psd_free(&err);
        const std::string hosts = "host;ip_address\nweb;192.168.1.5\ndb;2001:db8::1\nnet;10.0.0.0/8\n"
                                  "api;[2001:db8::1]:443\nnone;N/A\n";
        std::string hidden, shown;
        CHECK(process(e, "anonymize", "hosts.csv", hosts, res, hidden));
        const json ip_known = text_substitutes(res);
        // A value of no IP form takes a substitute of the type of the column, as any value of the column.
        CHECK(ip_known.size() == 4 && res["report"]["values"]["IPV4"].as_int() == 3 &&
              res["report"]["values"]["IPV6"].as_int() == 2);
        CHECK(hidden.find("192.168.1.5") == std::string::npos && hidden.find("2001:db8::1") == std::string::npos &&
              hidden.find("/8\n") != std::string::npos && hidden.find("]:443\n") != std::string::npos);
        json ip_told = json::object();
        ip_told["substitutions"] = ip_known;
        CHECK(process(e, "restore", "hosts.csv", hidden, res, shown, false, false, ip_told) && shown == hosts);
        CHECK(psd_set_columns(e, "[]", 2, &err) == 1);
        psd_free(&err);
    }

    psd_buffer fp = { nullptr, 0 };
    CHECK(psd_key_fingerprint(e, &fp) == 1 && fp.size > 0);
    psd_free(&fp);
    psd_destroy(e);
    if (failures) std::cerr << failures << " failure(s)" << std::endl;
    else std::cout << "test_engine: all tests passed (" << psd_variant() << ")" << std::endl;
    return failures ? 1 : 0;
}
