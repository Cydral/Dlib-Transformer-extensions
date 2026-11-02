// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// Tests of the engine through its C interface: documents, pictures and archives.

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <dlib/documents/document_image.h>
#include <dlib/zip/zip_archive.h>

#include "../common/engine_api.h"
#include "../common/json.h"
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

    // The text of element i of the array key of j.
    std::string item(const json& j, const char* key, size_t i)
    {
        const json& a = j[key];
        return i < a.size() ? a.elements()[i].str() : std::string();
    }

    int PSD_CALL describe(void*, const psd_picture* p, psd_buffer* d)
    {
        seen_pictures.push_back(std::string(p->media_type) + " " + p->location + " " + p->id);
        const std::string text = "Portrait of Jean Dupont [smiling]";
        d->data = psd_alloc(text.size());
        std::memcpy(d->data, text.data(), text.size());
        d->size = text.size();
        return 1;
    }

    bool process(psd_engine* e, const std::string& op, const std::string& name, const std::string& data, json& result,
                 std::string& output, bool pictures = true)
    {
        json req = json::object();
        req["operation"] = op;
        req["name"] = name;
        psd_buffer r = { nullptr, 0 }, o = { nullptr, 0 }, err = { nullptr, 0 };
        const int ok = psd_process(e, req.dump().c_str(), data.data(), data.size(), pictures ? describe : nullptr,
                                   nullptr, &r, &o, &err);
        if (!ok) std::cerr << "psd_process: " << (err.data ? err.data : "") << std::endl;
        result = ok ? json::parse_or_null(std::string(r.data, r.size)) : json();
        output = ok ? std::string(o.data, o.size) : std::string();
        psd_free(&r);
        psd_free(&o);
        psd_free(&err);
        return ok == 1;
    }
}

int main()
{
    CHECK(psd_api_version() == PSEUDALIS_ENGINE_API_VERSION);
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
    }
    CHECK(process(e, "restore", "envoi.zip", out, res, back));
    {
        const dlib::zip_archive_reader z(back);
        CHECK(z.read("dossier/lettre.txt") == letter);
        const std::vector<pseudalis::archive_entry> t =
            pseudalis::read_archive(pseudalis::archive_kind::tar_gz, z.read("archive.tar.gz"));
        CHECK(t.size() == 1 && t[0].data == inner[0].data);
    }

    // Unsupported content, and text.
    CHECK(!process(e, "anonymize", "outil.exe", std::string("MZ\x90\0\x03\0\0\0", 8) + std::string(64, '\0'),
                   res, out));
    CHECK(process(e, "anonymize_text", "prompt", "Écrire à Jean Dupont", res, out));
    CHECK(out.find("Dupont") == std::string::npos);
    CHECK(process(e, "detect", "prompt", "Écrire à Jean Dupont", res, out));
    CHECK(res["spans"].size() >= 1 && res["spans"].elements()[0]["type"].str() == "PERSON");

    psd_buffer fp = { nullptr, 0 };
    CHECK(psd_key_fingerprint(e, &fp) == 1 && fp.size > 0);
    psd_free(&fp);
    psd_destroy(e);
    if (failures) std::cerr << failures << " failure(s)" << std::endl;
    else std::cout << "test_engine: all tests passed (" << psd_variant() << ")" << std::endl;
    return failures ? 1 : 0;
}
