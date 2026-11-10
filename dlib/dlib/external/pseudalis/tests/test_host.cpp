// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

// Tests of the keys, the vault, the database and the settings of the service, in a temporary folder.

#include <iostream>
#include <string>
#include <vector>

#include <dlib/privacy/pii_types.h>
#include <sqlite3.h>

#include "../common/imposed_columns.h"
#include "../common/imposed_terms.h"
#include "../common/platform.h"
#include "../host/keys.h"
#include "../host/settings.h"
#include "../host/store.h"
#include "../host/vault.h"

using namespace pseudalis;

namespace
{
    int failures = 0;

#define CHECK(cond) check((cond), #cond, __LINE__)
    void check(bool ok, const char* what, int line)
    {
        if (ok) return;
        ++failures;
        std::cerr << "test_host.cpp:" << line << ": failed: " << what << std::endl;
    }
}

int main()
{
    const std::string dir = join_path(join_path(executable_dir(), "test-host-data"), random_hex(8));
    make_dirs(dir);

    // The master key: made once, read again, and its export.
    std::string master, again, error;
    bool created = false;
    CHECK(load_master_key(join_path(dir, "keys"), master, created, error) && created && master.size() == 32);
    CHECK(load_master_key(join_path(dir, "keys"), again, created, error) && !created && again == master);
    const std::string exported = export_master_key(master, "a passphrase of the tests");
    std::string imported;
    CHECK(!import_master_key(exported, "another passphrase", imported, error));
    CHECK(import_master_key(exported, "a passphrase of the tests", imported, error) && imported == master);
    const key_set keys = derive_keys(master);
    CHECK(keys.pseudonymization.size() == 32 && keys.vault_encryption != keys.vault_authentication);
    CHECK(aes256_ctr(keys.vault_encryption, std::string(16, '\1'),
                     aes256_ctr(keys.vault_encryption, std::string(16, '\1'), "round trip")) == "round trip");

    // The vault: blobs read back, altered ones refused.
    vault v(join_path(dir, "vault"), keys.vault_encryption, keys.vault_authentication);
    const std::string data(100000, 'x');
    const std::string id = v.put(data);
    std::string back;
    CHECK(vault::valid_id(id) && v.get(id, back) && back == data);
    std::string file;
    const std::string path = join_path(join_path(join_path(dir, "vault"), id.substr(0, 2)), id + ".bin");
    CHECK(read_file(path, file) && file.find(std::string(100, 'x')) == std::string::npos);
    file[40] ^= 1;
    CHECK(write_file_atomic(path, file) && !v.get(id, back));
    vault other(join_path(dir, "vault"), derive_keys(std::string(32, '\7')).vault_encryption,
                derive_keys(std::string(32, '\7')).vault_authentication);
    const std::string id2 = v.put("second");
    CHECK(!other.get(id2, back) && v.get(id2, back) && back == "second");
    CHECK(v.remove(id2) && !v.exists(id2));
    const std::string empty_id = v.put(std::string());
    CHECK(v.get(empty_id, back) && back.empty());
    // The listing of the vault, which removes the writes cut short.
    const std::string leftover =
        join_path(join_path(join_path(dir, "vault"), id.substr(0, 2)), id + ".bin.tmp-0a1b2c3d");
    CHECK(write_file_atomic(leftover, "cut short"));
    {
        const std::vector<std::string> ids = v.list_and_clean();
        bool has_id = false, has_empty = false;
        for (const std::string& x : ids)
        {
            has_id = has_id || x == id;
            has_empty = has_empty || x == empty_id;
        }
        CHECK(has_id && has_empty && ids.size() == 2 && !file_exists(leftover));
    }
    CHECK(v.remove(empty_id));

    /*
        The settings: a value out of its range loses no original (days without limit, the largest
        space), the space holds the largest file, and a file of a version before 1.0.2, without
        space, keeps every original, its 30 days of default becoming 45.
    */
    {
        CHECK(host_settings().removed_days == 45 && host_settings().history_mb == 10240);
        json j = json::object();
        j["removed_days"] = -1;
        j["history_mb"] = 0;
        j["max_file_mb"] = 64;
        j["drive_letter"] = "c";
        host_settings s = host_settings::from_json(j);
        CHECK(s.removed_days == 0 && s.history_mb == host_settings::max_history_mb && s.max_file_mb == 64 &&
              s.drive_letter == "A");
        j["removed_days"] = 0;
        j["history_mb"] = 100;
        s = host_settings::from_json(j);
        CHECK(s.removed_days == 0 && s.history_mb == 100);
        CHECK(host_settings::from_json(s.to_json()).removed_days == 0 &&
              host_settings::from_json(s.to_json()).history_mb == 100);
        j["history_mb"] = 2;
        CHECK(host_settings::from_json(j).history_mb == 64);
        json earlier = json::object();
        earlier["removed_days"] = 30;
        earlier["max_file_mb"] = 150;
        CHECK(host_settings::from_json(earlier).removed_days == 45 &&
              host_settings::from_json(earlier).history_mb == host_settings::max_history_mb);
        earlier["removed_days"] = 60;
        CHECK(host_settings::from_json(earlier).removed_days == 60);
        j["history_mb"] = 100;
        j["removed_days"] = 90;
        j["drive_letter"] = "p";
        s = host_settings::from_json(j);
        CHECK(s.removed_days == 90 && s.drive_letter == "P");
        CHECK(host_settings::from_json(s.to_json()).removed_days == 90);
        // The place of the publication of the build stays out of the file and of the answers.
        CHECK(!s.to_json().contains("feed_url") && host_settings::from_json(s.to_json()).feed_url == default_feed_url);
        s.feed_url = "https://example.org/feed.json";
        CHECK(host_settings::from_json(s.to_json()).feed_url == "https://example.org/feed.json");
        // The tags by default: written unless the setting says otherwise, a file of a version before keeping them.
        CHECK(host_settings().tags && host_settings::from_json(earlier).tags);
        s.tags = false;
        CHECK(!host_settings::from_json(s.to_json()).tags);
    }
    // A damaged settings.json keeps the originals without limit of time, in the largest space.
    {
        const std::string path = join_path(dir, "settings.json");
        CHECK(write_file_atomic(path, "{ not json"));
        settings_store damaged(path);
        CHECK(damaged.get().removed_days == 0 && damaged.get().history_mb == host_settings::max_history_mb &&
              file_exists(path + ".invalid"));
        host_settings changed;
        CHECK(damaged.update([](host_settings& s) { s.removed_days = 45; }, changed) && changed.removed_days == 45);
        settings_store again(path);
        CHECK(again.get().removed_days == 45 && again.get().tags);
        // The setting of the tags, saved, is read again when the service starts again.
        CHECK(again.update([](host_settings& s) { s.tags = false; }, changed) && !changed.tags);
        settings_store restarted(path);
        CHECK(!restarted.get().tags && restarted.get().removed_days == 45);
    }

    // The imposed terms: a list the service takes, or the reason and the place of its refusal.
    {
        std::vector<imposed_term> terms;
        std::string message, code;
        long long index = -1;
        json list = json::parse_or_null("[{\"term\": \"  Projet Alpha \", \"type\": \"ORG\"}, "
                                        "{\"term\": \"ACME\", \"type\": \"ORG\", \"case_sensitive\": true}, "
                                        "{\"term\": \"acme\", \"type\": \"PERSON\"}]");
        CHECK(parse_imposed_terms(list, terms, message, code, index) && terms.size() == 3 &&
              terms[0].term == "Projet Alpha" && !terms[0].case_sensitive && terms[1].case_sensitive &&
              terms[2].type == "PERSON" && index == -1);
        const json canonical = imposed_terms_json(terms);
        std::vector<imposed_term> back;
        CHECK(parse_imposed_terms(canonical, back, message, code, index) && back.size() == 3 &&
              back[1].term == "ACME");
        CHECK(parse_imposed_terms(json::parse_or_null("[]"), terms, message, code, index) && terms.empty());
        auto refused = [&](const char* text, const char* expected, long long at) {
            std::vector<imposed_term> t;
            std::string m, c;
            long long i = -2;
            return !parse_imposed_terms(json::parse_or_null(text), t, m, c, i) && c == expected && i == at &&
                   !m.empty();
        };
        CHECK(refused("{}", "terms_invalid", -1));
        CHECK(refused("[{\"term\": \"   \"}]", "terms_invalid", 0));
        CHECK(refused("[{\"term\": \"Alpha\"}, {\"term\": \"---\"}]", "terms_invalid", 1));
        CHECK(refused("[{\"term\": \"Alpha\\nBeta\"}]", "terms_invalid", 0));
        CHECK(refused("[{\"term\": \"Alpha\", \"type\": \"DATE\"}]", "terms_type", 0));
        CHECK(refused("[{\"term\": \"Société\"}, {\"term\": \"SOCIETE\"}]", "terms_duplicate", 1));
        CHECK(parse_imposed_terms(json::parse_or_null("[{\"term\": \"Société\"}, {\"term\": \"SOCIETE\", "
                                                      "\"case_sensitive\": true}]"), terms, message, code, index));
        std::string many = "[";
        for (int k = 0; k <= static_cast<int>(max_imposed_terms); ++k)
            many += std::string(k ? "," : "") + "{\"term\": \"term " + std::to_string(k) + "\"}";
        CHECK(refused((many + "]").c_str(), "terms_too_many", -1));
        CHECK(refused(("[{\"term\": \"" + std::string(max_imposed_term_length + 1, 'a') + "\"}]").c_str(),
                      "terms_invalid", 0));
        CHECK(parse_imposed_terms(json::parse_or_null("[{\"term\": \"" + std::string(max_imposed_term_length, 'a') +
                                                      "\"}]"), terms, message, code, index));
    }

    // The imposed columns: a list the service takes, or the reason and the place of its refusal.
    {
        std::vector<imposed_column> columns;
        std::string message, code;
        long long index = -1;
        json list = json::parse_or_null("[{\"column\": \"  Nom \", \"type\": \"PERSON\"}, "
                                        "{\"column\": \"E-mail\", \"type\": \"EMAIL\"}, "
                                        "{\"column\": \"Naissance\", \"type\": \"DATE\"}, "
                                        "{\"column\": \"N° client\"}]");
        CHECK(parse_imposed_columns(list, columns, message, code, index) && columns.size() == 4 &&
              columns[0].name == "Nom" && columns[2].type == "DATE" && columns[3].type == "PERSON" && index == -1);
        const json canonical = imposed_columns_json(columns);
        std::vector<imposed_column> back;
        CHECK(parse_imposed_columns(canonical, back, message, code, index) && back.size() == 4 &&
              back[3].name == "N° client" && canonical[0]["column"].str() == "Nom");
        CHECK(parse_imposed_columns(json::parse_or_null("[]"), columns, message, code, index) && columns.empty());
        auto refused = [&](const char* text, const char* expected, long long at) {
            std::vector<imposed_column> c;
            std::string m, k;
            long long i = -2;
            return !parse_imposed_columns(json::parse_or_null(text), c, m, k, i) && k == expected && i == at &&
                   !m.empty();
        };
        CHECK(refused("{}", "columns_invalid", -1));
        CHECK(refused("[{\"column\": \"   \"}]", "columns_invalid", 0));
        CHECK(refused("[{\"column\": \"Nom\"}, {\"column\": \"---\"}]", "columns_invalid", 1));
        CHECK(refused("[{\"column\": \"Nom\\tPrenom\"}]", "columns_invalid", 0));
        CHECK(refused("[{\"term\": \"Nom\"}]", "columns_invalid", 0));
        CHECK(refused("[{\"column\": \"Nom\", \"type\": \"NONE\"}]", "columns_type", 0));
        CHECK(refused("[{\"column\": \"Prénom\"}, {\"column\": \"PRENOM\"}]", "columns_duplicate", 1));
        std::string many = "[";
        for (int k = 0; k <= static_cast<int>(max_imposed_columns); ++k)
            many += std::string(k ? "," : "") + "{\"column\": \"column " + std::to_string(k) + "\"}";
        CHECK(refused((many + "]").c_str(), "columns_too_many", -1));
        CHECK(refused(("[{\"column\": \"" + std::string(max_imposed_column_length + 1, 'a') + "\"}]").c_str(),
                      "columns_invalid", 0));
        CHECK(parse_imposed_columns(json::parse_or_null("[{\"column\": \"" +
                                                        std::string(max_imposed_column_length, 'a') + "\"}]"),
                                    columns, message, code, index));
        for (const std::string& t : imposed_column_types())
            CHECK(t == "IP" || dlib::pii_type_from_keyword(t) != dlib::pii_type::none);
        // The two forms of IP address make one type, IP; the keywords of the forms are no types of column.
        CHECK(parse_imposed_columns(json::parse_or_null("[{\"column\": \"ip_address\", \"type\": \"IP\"}]"),
                                    columns, message, code, index) &&
              columns.size() == 1 && columns[0].type == "IP");
        CHECK(refused("[{\"column\": \"ip_address\", \"type\": \"IPV4\"}]", "columns_type", 0));
        CHECK(refused("[{\"column\": \"ip\", \"type\": \"IP\"}, {\"column\": \"ip6\", \"type\": \"IPV6\"}]",
                      "columns_type", 1));
    }

    // The database: records, pictures and the tree of the drive.
    {
        store db(join_path(dir, "pseudalis.db"));
        file_record r;
        r.source = "shell";
        r.original_name = "CV Jean Dupont.docx";
        r.output_name = "CV Jean Dupont.docx";
        r.original_sha256 = std::string(64, 'a');
        r.output_sha256 = std::string(64, 'b');
        r.original_blob = v.put("original");
        r.kind = "docx";
        r.original_size = 1000;
        const long long rec = db.add_file(r);
        CHECK(rec > 0 && db.count_files() == 1 && db.count_files("shell") == 1 && db.count_files("web") == 0);
        CHECK(db.originals_bytes() == 1000 && db.oldest_files(10).size() == 1 && db.oldest_files(10)[0].first == rec);
        file_record found;
        CHECK(db.find_by_output(std::string(64, 'b'), found) && found.id == rec
            && found.original_name == r.original_name);
        db.set_file_location(rec, std::string(32, 'c'), "C:\\docs\\cv.docx");
        CHECK(db.find_by_object(std::string(32, 'c'), found) && found.path == "C:\\docs\\cv.docx");
        // A picture, under a keyed identifier that its SHA-256 does not give, kept first without description.
        picture_record p;
        p.id = "0123456789abcdef";
        p.sha256 = std::string(64, 'e');
        p.media_type = "image/png";
        p.blob = v.put("png");
        db.add_picture(p);
        db.link_picture(rec, p.id);
        picture_record q;
        std::string text, before;
        CHECK(db.get_picture(p.id, q) && q.sha256 == p.sha256 && q.blob == p.blob && q.description.empty());
        // The description received later, encrypted in the vault; the blob named before would leave it.
        p.description = v.put("a chart");
        CHECK(db.set_picture_description(p.id, p.description, before) && before.empty());
        CHECK(db.get_picture(p.id, q) && q.description == p.description && v.get(q.description, text) &&
              text == "a chart");
        CHECK(!db.set_picture_description("fedcba9876543210", p.description, before));
        // A picture that no record names, the protection that kept it having failed, leaves at the start.
        picture_record lone = p;
        lone.id = "fedcba9876543210";
        lone.blob = v.put("gif");
        lone.description.clear();
        db.add_picture(lone);
        CHECK(db.remove_unlinked_pictures() == 1 && !db.get_picture(lone.id, q) && db.get_picture(p.id, q));
        CHECK(db.referenced_blobs().count(lone.blob) == 0 && db.remove_unlinked_pictures() == 0);
        // The entries of a protected archive: found by what was written, those inside an archive taken from inside it.
        file_record archive = r;
        archive.original_name = "Dossier Jean Dupont.zip";
        archive.output_sha256 = std::string(64, '9');
        archive.original_blob = v.put("archive");
        archive.original_size = 2500;
        const long long zip = db.add_file(archive);
        // The space of the originals, and the oldest first.
        CHECK(db.originals_bytes() == 3500 && db.oldest_files(1).size() == 1 && db.oldest_files(1)[0].first == rec &&
              db.oldest_files(5).size() == 2 && db.oldest_files(5)[1].first == zip &&
              db.oldest_files(5)[1].second == 2500);
        std::vector<entry_record> entries(3);
        entries[0].name = "Dossier X/lettre.txt";
        entries[0].original = "Dossier Jean Dupont/lettre.txt";
        entries[0].sha256 = std::string(64, '1');
        entries[0].kind = "txt";
        entries[1].name = "notes.tar.gz";
        entries[1].original = "notes.tar.gz";
        entries[1].sha256 = std::string(64, '2');
        entries[1].kind = "tar.gz";
        entries[2].name = "notes.tar.gz!Y/note.txt";
        entries[2].original = "notes.tar.gz!Marie Curie/note.txt";
        entries[2].sha256 = std::string(64, '3');
        entries[2].kind = "txt";
        db.add_entries(zip, entries);
        entry_record entry_found;
        CHECK(db.find_entry(std::string(64, '1'), entry_found) && entry_found.record == zip &&
              entry_found.original == "Dossier Jean Dupont/lettre.txt" && entry_found.kind == "txt");
        CHECK(!db.find_entry(std::string(64, '4'), entry_found));
        CHECK(db.find_entry(std::string(64, '2'), entry_found));
        const std::vector<entry_record> inside = db.entries_inside(entry_found);
        CHECK(inside.size() == 1 && inside[0].name == "Y/note.txt" && inside[0].original == "Marie Curie/note.txt" &&
              inside[0].sha256 == std::string(64, '3'));
        // Forgotten with its archive: its entries go, and are remembered as forgotten.
        std::vector<std::string> zip_blobs;
        CHECK(db.remove_file(zip, zip_blobs, "forgotten") && zip_blobs.size() == 1);
        int64_t entry_time = 0;
        std::string entry_reason;
        CHECK(!db.find_entry(std::string(64, '1'), entry_found) && !db.find_entry(std::string(64, '3'), entry_found));
        CHECK(db.find_forgotten(std::string(64, '3'), std::string(), entry_time, entry_reason) &&
              entry_reason == "forgotten");

        // A record whose file left the drive keeps its original until its time is over.
        CHECK(found.removed == 0 && db.files_removed_before(now_unix()).empty());
        db.set_file_removed(rec, now_unix() - 31 * 86400);
        CHECK(db.get_file(rec, found) && found.removed > 0);
        CHECK(db.files_removed_before(now_unix() - 30 * 86400).size() == 1);
        CHECK(db.files_removed_before(now_unix() - 32 * 86400).empty());
        CHECK(db.referenced_blobs().count(r.original_blob) == 1 && db.referenced_blobs().count(p.blob) == 1 &&
              db.referenced_blobs().count(p.description) == 1);
        std::vector<std::string> blobs;
        int64_t when = 0;
        std::string reason;
        CHECK(!db.find_forgotten(std::string(64, 'b'), std::string(32, 'c'), when, reason));
        // An original forgotten: its protected file is remembered, by its hash and its object identifier.
        CHECK(db.remove_file(rec, blobs, "forgotten") && blobs.size() == 3 && db.count_files() == 0);
        CHECK(db.find_forgotten(std::string(64, 'b'), std::string(), when, reason) && reason == "forgotten");
        CHECK(db.find_forgotten(std::string(64, 'f'), std::string(32, 'c'), when, reason) && when > 0);
        CHECK(!db.find_forgotten(std::string(64, 'f'), std::string(), when, reason));
        CHECK(!db.get_picture(p.id, q));
        // The identifier of a record deleted is never given again.
        const long long next = db.add_file(r);
        // A record that leaves without reason (the drive replaces its file) leaves no trace.
        CHECK(next > rec && db.remove_file(next, blobs) && db.count_files() == 0);

        // A database of the version 1 of the schema: the descriptions that it kept in clear are erased.
        const std::string old_path = join_path(dir, "version1.db");
        {
            store fresh(old_path);
        }
        const std::string kept = v.put("a map");
        sqlite3* raw = nullptr;
        CHECK(sqlite3_open(old_path.c_str(), &raw) == SQLITE_OK);
        const std::string old_rows =
            "INSERT INTO pictures VALUES('aaaaaaaaaaaaaaaa', '" + std::string(64, 'a') + "', 'image/png', '"
            + v.put("old") + "', 'Portrait of Jean Dupont', 1);"
            "INSERT INTO pictures VALUES('bbbbbbbbbbbbbbbb', '" + std::string(64, 'b') + "', 'image/png', '"
            + v.put("new") + "', '" + kept + "', 1);"
            "PRAGMA user_version = 1;";
        CHECK(sqlite3_exec(raw, old_rows.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
        {
            store migrated(old_path);
            CHECK(migrated.get_picture("aaaaaaaaaaaaaaaa", q) && q.description.empty() && !q.blob.empty());
            CHECK(migrated.get_picture("bbbbbbbbbbbbbbbb", q) && q.description == kept);
        }
        // A database of the version 2, as Pseudalis 1.0.4 left it: its files hold their tags, and the
        // substitutes of the files protected without tags are kept from then on.
        const std::string v2_path = join_path(dir, "version2.db");
        {
            store fresh(v2_path);
            CHECK(fresh.add_file(r) > 0);
        }
        CHECK(sqlite3_open(v2_path.c_str(), &raw) == SQLITE_OK);
        CHECK(sqlite3_exec(raw, "ALTER TABLE files DROP COLUMN tags; ALTER TABLE forgotten DROP COLUMN tags;"
                                "DROP TABLE text_substitutes; PRAGMA user_version = 2;",
                           nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
        {
            store migrated(v2_path);
            file_record tagged;
            CHECK(migrated.find_by_output(r.output_sha256, tagged) && tagged.tags);
            file_record bare = r;
            bare.output_sha256 = std::string(64, 'd');
            bare.tags = false;
            const long long id = migrated.add_file(bare);
            migrated.add_text_substitutes(id, { { "Xoydoeh", "[[PERSON:Xoydoeh#EKR]]" } });
            file_record back;
            CHECK(migrated.find_by_output(bare.output_sha256, back) && !back.tags &&
                  migrated.text_substitutes(id).size() == 1);
        }
        // A database of the version 3, as Pseudalis 1.0.8 left it: the encoding of its records is unknown, and
        // kept from then on.
        const std::string v3_path = join_path(dir, "version3.db");
        {
            store fresh(v3_path);
            CHECK(fresh.add_file(r) > 0);
        }
        CHECK(sqlite3_open(v3_path.c_str(), &raw) == SQLITE_OK);
        CHECK(sqlite3_exec(raw, "ALTER TABLE files DROP COLUMN encoding; PRAGMA user_version = 3;",
                           nullptr, nullptr, nullptr) == SQLITE_OK);
        sqlite3_close(raw);
        {
            store migrated(v3_path);
            file_record before;
            CHECK(migrated.find_by_output(r.output_sha256, before) && before.encoding.empty());
            file_record text = r;
            text.output_sha256 = std::string(64, 'e');
            text.encoding = "windows-1252";
            CHECK(migrated.add_file(text) > 0);
        }
        {
            store reopened(v3_path);
            file_record back;
            CHECK(reopened.find_by_output(std::string(64, 'e'), back) && back.encoding == "windows-1252");
        }

        // The files of the folder of the drive as the service left them, and what they weigh.
        folder_file note;
        note.folder = "drive";
        note.path = "Dossier/Note.txt";
        note.size = 12;
        note.written = 7;
        note.sha256 = std::string(64, 'e');
        note.state = "protected";
        db.set_folder_file(note);
        folder_file folder = note;
        folder.path = "Dossier";
        folder.size = 0;
        folder.state = "folder";
        db.set_folder_file(folder);
        CHECK(db.folder_files("drive").size() == 2 && db.folder_files("other").empty());
        long long files = 0, bytes = 0;
        db.drive_usage(files, bytes);
        CHECK(files == 1 && bytes == 12);
        // The paths compare without case, as Windows names its files.
        note.path = "DOSSIER/note.TXT";
        note.size = 13;
        db.set_folder_file(note);
        db.drive_usage(files, bytes);
        CHECK(db.folder_files("drive").size() == 2 && files == 1 && bytes == 13);
        db.remove_folder_file("drive", "dossier/NOTE.txt");
        db.drive_usage(files, bytes);
        CHECK(files == 0 && bytes == 0 && db.folder_files("drive").size() == 1);
        // A substitute comes back through the tags recorded for it, never through its value.
        db.add_substitutes({ { "Xoydoeh Nuggwdfsorn", "[[PERSON:Xoydoeh Nuggwdfsorn#EKR]]" } });
        CHECK(db.tags_for("Xoydoeh Nuggwdfsorn").size() == 1 && db.tags_for("Jean Dupont").empty());
        db.add_event("protected", "\\Dossier\\Note.txt", "3 values");
        CHECK(db.events(0).size() == 1);
        db.set_value("k", "v");
        CHECK(db.value("k") == "v");
    }
    remove_tree(dir);
    if (failures) std::cerr << failures << " failure(s)" << std::endl;
    else std::cout << "test_host: all tests passed" << std::endl;
    return failures ? 1 : 0;
}
