// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/privacy.h>
#include <dlib/privacy/pii_sequence_tagger.h>
#include <dlib/privacy/pii_span_model.h>
#include <dlib/rand.h>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.privacy");

    string hex(const std::array<unsigned char, 32>& a) { return to_hex(a.data(), a.size()); }

// ----------------------------------------------------------------------------------------

    void test_crypto()
    {
        DLIB_TEST(hex(sha256_digest(string("abc"))) ==
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        DLIB_TEST(hex(sha256_digest(string(""))) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        DLIB_TEST(hex(sha256_digest(string("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))) ==
                  "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        {
            sha256 h;
            const string a(100000, 'a');
            for (size_t i = 0; i < a.size(); i += 777) h.update(a.data() + i, std::min<size_t>(777, a.size() - i));
            DLIB_TEST(hex(h.finalize()) == hex(sha256_digest(a)));
        }

        // RFC 4231 and RFC 5869.
        DLIB_TEST(hex(hmac_sha256_digest(string(20, '\x0b'), "Hi There")) ==
                  "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
        DLIB_TEST(hex(hmac_sha256_digest("Jefe", "what do ya want for nothing?")) ==
                  "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
        DLIB_TEST(
            hex(hmac_sha256_digest(string(131, '\xaa'), "Test Using Larger Than Block-Size Key - Hash Key First")) ==
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
        DLIB_TEST(to_hex(hkdf_sha256(string(22, '\x0b'), from_hex("000102030405060708090a0b0c"),
                                     from_hex("f0f1f2f3f4f5f6f7f8f9"), 42)) ==
                  "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865");
        DLIB_TEST(to_hex(hkdf_sha256(string(22, '\x0b'), "", "", 42)) ==
                  "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8");
        DLIB_TEST(to_hex(pbkdf2_hmac_sha256("password", "salt", 1, 32)) ==
                  "120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b");
        DLIB_TEST(to_hex(pbkdf2_hmac_sha256("password", "salt", 2, 32)) ==
                  "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43");
        DLIB_TEST(to_hex(pbkdf2_hmac_sha256("passwd", "salt", 1, 64)) ==
                  "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc"
                  "49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783");

        // FIPS-197.
        const string pt = from_hex("00112233445566778899aabbccddeeff");
        unsigned char out[16];
        aes_block_cipher a128(from_hex("000102030405060708090a0b0c0d0e0f").data(), 16);
        a128.encrypt_block(reinterpret_cast<const unsigned char*>(pt.data()), out);
        DLIB_TEST(to_hex(out, 16) == "69c4e0d86a7b0430d8cdb78070b4c55a");
        aes_block_cipher a192(from_hex("000102030405060708090a0b0c0d0e0f1011121314151617").data(), 24);
        a192.encrypt_block(reinterpret_cast<const unsigned char*>(pt.data()), out);
        DLIB_TEST(to_hex(out, 16) == "dda97ca4864cdfe06eaf70a0ec0d7191");
        aes_block_cipher a256(from_hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f").data(), 32);
        a256.encrypt_block(reinterpret_cast<const unsigned char*>(pt.data()), out);
        DLIB_TEST(to_hex(out, 16) == "8ea2b7ca516745bfeafc49904b496089");

        // NIST SP 800-38G FF1 samples 1 to 9.
        const string k128 = from_hex("2B7E151628AED2A6ABF7158809CF4F3C");
        const string k192 = from_hex("2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F");
        const string k256 = from_hex("2B7E151628AED2A6ABF7158809CF4F3CEF4359D8D580AA4F7F036D6F04FC6A94");
        const string d10 = "0123456789", d36 = "0123456789abcdefghijklmnopqrstuvwxyz";
        const string tw = from_hex("39383736353433323130"), tw36 = from_hex("3737373770717273373737");
        struct sample { const string* key; uint32_t radix; string pt, tweak, ct; };
        const sample samples[] = {
            { &k128, 10, "0123456789", "", "2433477484" },
            { &k128, 10, "0123456789", tw, "6124200773" },
            { &k128, 36, "0123456789abcdefghi", tw36, "a9tv40mll9kdu509eum" },
            { &k192, 10, "0123456789", "", "2830668132" },
            { &k192, 10, "0123456789", tw, "2496655549" },
            { &k192, 36, "0123456789abcdefghi", tw36, "xbj3kv35jrawxv32ysr" },
            { &k256, 10, "0123456789", "", "6657667009" },
            { &k256, 10, "0123456789", tw, "1001623463" },
            { &k256, 36, "0123456789abcdefghi", tw36, "xs8a0azh2avyalyzuwd" } };
        for (const sample& s : samples)
        {
            const ff1_cipher c(s.key->data(), s.key->size(), s.radix);
            const string& alphabet = s.radix == 10 ? d10 : d36;
            const string ct = c.encrypt(s.pt, alphabet, s.tweak);
            DLIB_TEST_MSG(ct == s.ct, ct);
            DLIB_TEST(c.decrypt(ct, alphabet, s.tweak) == s.pt);
        }
        dlib::rand rnd;
        const ff1_cipher c26(k256.data(), 32, 26), c65536(k256.data(), 32, 65536);
        for (int it = 0; it < 100; ++it)
        {
            for (const ff1_cipher* c : { &c26, &c65536 })
            {
                std::vector<uint32_t> x(c->min_length() + rnd.get_random_32bit_number() % 30);
                for (auto& d : x) d = rnd.get_random_32bit_number() % c->get_radix();
                DLIB_TEST(c->decrypt(c->encrypt(x, "tweak"), "tweak") == x);
            }
        }
        bool small_domain_rejected = false;
        try { c26.encrypt(std::vector<uint32_t>(4, 1), ""); }
        catch (crypto_error&) { small_domain_rejected = true; }
        DLIB_TEST(small_domain_rejected);

        // Crypto-PAn keeps prefixes and inverts.
        string key;
        for (int i = 0; i < 32; ++i) key.push_back(static_cast<char>(i));
        const prefix_preserving_ip_cipher cp(key.data(), 32);
        const uint32_t a = cp.encrypt_ipv4(0xC0A80119), b = cp.encrypt_ipv4(0xC0A8011A);
        // 192.168.1.25 and .26 share exactly 30 leading bits, and so do their codes.
        DLIB_TEST((a >> 2) == (b >> 2) && (a >> 1) != (b >> 1));
        DLIB_TEST(cp.decrypt_ipv4(a) == 0xC0A80119);
        const std::array<unsigned char, 16> v6 = { 0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
        DLIB_TEST(cp.decrypt_ipv6(cp.encrypt_ipv6(v6)) == v6);

        DLIB_TEST(generate_secure_random_bytes(32) != generate_secure_random_bytes(32));
    }

// ----------------------------------------------------------------------------------------

    void test_deflate_and_zip()
    {
        const string sample = "Le contrat de Jean Dupont. ";
        string repeated;
        for (int i = 0; i < 20; ++i) repeated += sample;
        // zlib.compress(repeated, 9) from Python.
        DLIB_TEST(zlib_decompress(from_hex(
                      "78daf3495548cecf2b294a2c51484955f04a4dcc5370292d008ae829f88c4a8d6c29004f97b89d")) == repeated);

        dlib::rand rnd;
        string noise(20000, '\0');
        for (char& c : noise) c = static_cast<char>(rnd.get_random_32bit_number() & 0xFF);
        string text;
        while (text.size() < 50000) text += "<w:t>Jean Dupont " + to_string(text.size() % 97) + "</w:t>";
        for (const string* data : { &repeated, &noise, &text })
        {
            for (int level = 0; level <= 9; level += 3)
                DLIB_TEST(inflate_raw(deflate_raw(*data, level)) == *data);
            DLIB_TEST(zlib_decompress(zlib_compress(*data)) == *data);
        }
        DLIB_TEST(deflate_raw(text, 6).size() < text.size() / 5);

        zip_archive_writer w;
        w.add("[Content_Types].xml", "<Types/>");
        w.add("word/document.xml", text);
        w.add("media/noise.bin", noise);
        w.add("empty.txt", "");
        bool duplicate_rejected = false;
        try { w.add("empty.txt", "x"); }
        catch (zip_error&) { duplicate_rejected = true; }
        DLIB_TEST(duplicate_rejected);
        const string archive = w.finish();
        const zip_archive_reader z(archive);
        DLIB_TEST(z.entries().size() == 4);
        DLIB_TEST(z.read("word/document.xml") == text);
        DLIB_TEST(z.read("media/noise.bin") == noise);
        DLIB_TEST(z.read("empty.txt").empty());

        // Corrupted archives are rejected.
        int rejected = 0;
        for (int it = 0; it < 200; ++it)
        {
            string bad = archive;
            for (int f = 0; f < 4; ++f)
                bad[rnd.get_random_32bit_number() % bad.size()] ^=
                    static_cast<char>(1 << (rnd.get_random_32bit_number() % 8));
            try
            {
                const zip_archive_reader zb(bad);
                for (size_t i = 0; i < zb.entries().size(); ++i) zb.read(i);
            }
            catch (zip_error&) { ++rejected; }
        }
        DLIB_TEST(rejected > 0);

        // A decompression bomb is refused by the size limits.
        zip_archive_writer bw;
        bw.add("bomb.bin", string(16u << 20, '\0'));
        const string bomb = bw.finish();
        zip_limits limits;
        limits.max_entry_size = 4u << 20;
        bool bomb_rejected = false;
        try { const zip_archive_reader zb(bomb, limits); }
        catch (zip_error&) { bomb_rejected = true; }
        DLIB_TEST(bomb_rejected);
    }

// ----------------------------------------------------------------------------------------

    bool round_trip(const pseudonymizer& p, pii_type t, const string& value)
    {
        pii_type back_type;
        string back;
        return p.decode(p.tag(t, value), back_type, back) && back_type == t && back == value;
    }

    string tag_body(const string& tag)
    {
        const size_t colon = tag.find(':'), hash = tag.rfind('#');
        return tag.substr(colon + 1, hash - colon - 1);
    }

    bool luhn(const string& digits)
    {
        int sum = 0;
        bool twice = false;
        for (size_t i = digits.size(); i-- > 0;)
        {
            int d = digits[i] - '0';
            if (twice) { d *= 2; if (d > 9) d -= 9; }
            sum += d;
            twice = !twice;
        }
        return sum % 10 == 0;
    }

    string only_digits(const string& s)
    {
        string d;
        for (char c : s) if (c >= '0' && c <= '9') d.push_back(c);
        return d;
    }

    void test_pseudonymizer()
    {
        const pseudonymization_key key = pseudonymization_key::from_bytes(string(32, '\x42'));
        pseudonymizer_options options;
        options.context = "test";
        const pseudonymizer p(key, options);

        const pair<pii_type, string> values[] = { { pii_type::person, "Jean Dupont" },
                                                  { pii_type::person, "DUPONT" },
                                                  { pii_type::person, "Hélène Le Goff" },
                                                  { pii_type::person, "Valéry Giscard d'Estaing" },
                                                  { pii_type::person, "McDonald" },
                                                  { pii_type::person, "Li" },
                                                  { pii_type::person, "Сергей Иванов" },
                                                  { pii_type::person, "王小明" },
                                                  { pii_type::organization, "Cabinet Dupont SAS" },
                                                  { pii_type::location, "35000 Rennes" },
                                                  { pii_type::address, "12 bis rue des Lilas" },
                                                  { pii_type::email, "J.Dupont+news@Mail.Example.COM" },
                                                  { pii_type::url, "https://www.linkedin.com/in/jean-dupont-12345678" },
                                                  { pii_type::phone, "06 12 34 56 78" },
                                                  { pii_type::phone, "+33 (0)6 12 34 56 78" },
                                                  { pii_type::phone,
                                                    "06\xC2\xA0"
                                                    "12\xC2\xA0"
                                                    "34\xC2\xA0"
                                                    "56\xC2\xA0"
                                                    "78" },
                                                  { pii_type::iban, "FR76 3000 6000 0112 3456 7890 189" },
                                                  { pii_type::iban, "GB29 NWBK 6016 1331 9268 19" },
                                                  { pii_type::payment_card, "4970 1012 3456 7899" },
                                                  { pii_type::nir, "1 85 03 75 123 456 13" },
                                                  { pii_type::nir, "2 69 05 2A 123 456 09" },
                                                  { pii_type::siret, "732 829 320 00074" },
                                                  { pii_type::mac_address, "00:1A:2B:3C:4D:5E" },
                                                  { pii_type::license_plate, "AB-123-CD" },
                                                  { pii_type::identifier, "JDupont2024" },
                                                  { pii_type::identifier, "AB-12" },
                                                  { pii_type::ipv4, "192.168.1.25" },
                                                  { pii_type::ipv6, "2001:db8::1" },
                                                  { pii_type::date, "12/03/1985" },
                                                  { pii_type::date, "1er mars 1985" },
                                                  { pii_type::date, "March 1st, 1985" },
                                                  { pii_type::date, "29 février 2024" },
                                                  { pii_type::date, "mars 2022" },
                                                  { pii_type::date, "March, 2022" },
                                                  { pii_type::date, "SEPT. 2019" },
                                                  { pii_type::date, "03/2022" },
                                                  { pii_type::date, "10.2025" },
                                                  { pii_type::date, "2019" } };
        for (const auto& v : values)
            DLIB_TEST_MSG(round_trip(p, v.first, v.second), v.second);

        // Same value, same tag; the code of a word is shared across types.
        DLIB_TEST(p.tag(pii_type::person, "Jean Dupont") == p.tag(pii_type::person, "Jean Dupont"));
        const string surname = tag_body(p.tag(pii_type::person, "Dupont"));
        DLIB_TEST(p.tag(pii_type::person, "M. Dupont").find(surname) != string::npos);
        string lower_surname = surname;
        for (char& c : lower_surname) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
        DLIB_TEST(p.tag(pii_type::email, "jean.dupont@acme.fr").find(lower_surname) != string::npos);

        // Checksums still verify on protected numbers.
        DLIB_TEST(luhn(only_digits(tag_body(p.tag(pii_type::payment_card, "4970 1012 3456 7899")))));
        const string siret = only_digits(tag_body(p.tag(pii_type::siret, "732 829 320 00074")));
        DLIB_TEST(siret.size() == 14 && luhn(siret) && luhn(siret.substr(0, 9)));
        DLIB_TEST(siret.substr(0, 9) == only_digits(tag_body(p.tag(pii_type::siret, "732 829 320"))));
        const string iban = tag_body(p.tag(pii_type::iban, "FR76 3000 6000 0112 3456 7890 189"));
        string moved;
        for (char c : iban.substr(4) + iban.substr(0, 4))
        {
            if (c == ' ') continue;
            if (c >= 'A' && c <= 'Z') moved += to_string(c - 'A' + 10);
            else moved.push_back(c);
        }
        int r = 0;
        for (char c : moved) r = (r * 10 + (c - '0')) % 97;
        DLIB_TEST(r == 1);

        // Dates move together: intervals and weekdays are kept.
        civil_date a, b;
        date_format fa, fb;
        DLIB_TEST(parse_date(tag_body(p.tag(pii_type::date, "2026-10-14")), numeric_date_order::day_month_year, a, fa));
        DLIB_TEST(parse_date(tag_body(p.tag(pii_type::date, "2026-10-21")), numeric_date_order::day_month_year, b, fb));
        DLIB_TEST(days_from_civil(b.year, b.month, b.day) - days_from_civil(a.year, a.month, a.day) == 7);
        DLIB_TEST(p.date_shift_days() < 0 && p.date_shift_days() % 7 == 0 && p.date_shift_days() >= -7 * 260);

        // A month and its year move by whole months, a year alone by whole years, never by zero.
        DLIB_TEST(p.date_shift_months() < 0 && p.date_shift_years() < 0);
        DLIB_TEST(std::abs(p.date_shift_months() * 30.436875 - p.date_shift_days()) <= 16);
        const string month_body = tag_body(p.tag(pii_type::date, "mars 2022"));
        const long months = 12L * 2022 + 2 + p.date_shift_months();
        DLIB_TEST(month_body.size() > 8 && month_body[7] == '~' &&
                  month_body.substr(0, 7) == to_string(months / 12) + "-" + (months % 12 < 9 ? "0" : "") +
                  to_string(months % 12 + 1));
        DLIB_TEST(tag_body(p.tag(pii_type::date, "2019")).substr(0, 5) == to_string(2019 + p.date_shift_years()) + "~");

        // The particles of a person name are encrypted with its other words.
        DLIB_TEST(p.tag(pii_type::person, "Hélène Le Goff").find(" Le ") == string::npos);
        DLIB_TEST(p.tag(pii_type::person, "Ludwig van Beethoven").find(" van ") == string::npos);

        // Restoration in derived text, percent-encoded, tampered, foreign context.
        const string text = "Contact: Jean Dupont <jean.dupont@acme.fr>, tel 06 12 34 56 78.";
        const std::vector<pii_entity> entities = { { 9, 20, pii_type::person, 1 }, { 22, 41, pii_type::email, 1 },
                                              { 48, 62, pii_type::phone, 1 } };
        const string protected_text = p.apply(text, entities);
        DLIB_TEST(protected_text.find("Dupont") == string::npos);
        pseudonymizer::restore_report report;
        DLIB_TEST(p.restore("Summary: " + protected_text, &report) == "Summary: " + text);
        DLIB_TEST(report.restored == 3 && report.rejected == 0);
        DLIB_TEST(p.restore("mailto:" + percent_encode_tag(p.tag(pii_type::email, "jean.dupont@acme.fr"))) ==
                  "mailto:jean.dupont@acme.fr");
        string tampered = protected_text;
        const size_t at = tampered.find("[[PERSON:") + 10;
        tampered[at] = tampered[at] == 'a' ? 'b' : 'a';
        pseudonymizer::restore_report tampered_report;
        p.restore(tampered, &tampered_report);
        DLIB_TEST(tampered_report.rejected == 1 && tampered_report.restored == 2);
        pseudonymizer_options other = options;
        other.context = "other";
        pseudonymizer::restore_report other_report;
        pseudonymizer(key, other).restore(protected_text, &other_report);
        DLIB_TEST(other_report.restored == 0);
    }

// ----------------------------------------------------------------------------------------

    bool found(const std::vector<pii_entity>& entities, const string& text, pii_type type, const string& value)
    {
        for (const pii_entity& e : entities)
            if (e.type == type && text.substr(e.begin, e.end - e.begin) == value) return true;
        return false;
    }

    bool covered(const std::vector<pii_entity>& entities, const string& text, const string& value)
    {
        const size_t at = text.find(value);
        for (const pii_entity& e : entities)
            if (e.begin <= at && at < e.end) return true;
        return false;
    }

    const char* letter =
        "Objet : Dossier de Madame Hélène LE GOFF\n"
        "De : Jean Dupont <jean.dupont@acme-conseil.fr>\n"
        "À : Service RH\n"
        "\n"
        "Je vous transmets le dossier de Mme Hélène Le Goff, née le 3 mars 1985 à Saint-Malo, domiciliée au "
        "12 bis rue des Lilas, 35000 Rennes. Elle est joignable au 06 12 34 56 78. Son numéro de sécurité sociale "
        "est le 2 85 03 35 238 012 05 et son IBAN FR76 3000 6000 0112 3456 7890 189.\n"
        "\n"
        "Nom :\tLE GOFF\n"
        "Immatriculation : AB-123-CD\n"
        "Connexion depuis 192.168.12.45 avec la carte 4970 1012 3456 7893.\n"
        "\n"
        "Le Goff a signé le contrat. La Pierre Philosophale reste un roman. Dupont validera.\n";

    void test_detector()
    {
        pii_detector detector;
        const string text = letter;
        detector.begin_document();
        detector.learn(text);
        const std::vector<pii_entity> e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::person, "Hélène LE GOFF"));
        DLIB_TEST(found(e, text, pii_type::person, "Jean Dupont"));
        DLIB_TEST(found(e, text, pii_type::email, "jean.dupont@acme-conseil.fr"));
        DLIB_TEST(found(e, text, pii_type::date, "3 mars 1985"));
        DLIB_TEST(found(e, text, pii_type::location, "Saint-Malo"));
        DLIB_TEST(covered(e, text, "12 bis rue des Lilas"));
        DLIB_TEST(found(e, text, pii_type::phone, "06 12 34 56 78"));
        DLIB_TEST(found(e, text, pii_type::nir, "2 85 03 35 238 012 05"));
        DLIB_TEST(found(e, text, pii_type::iban, "FR76 3000 6000 0112 3456 7890 189"));
        DLIB_TEST(found(e, text, pii_type::person, "LE GOFF"));
        DLIB_TEST(found(e, text, pii_type::license_plate, "AB-123-CD"));
        DLIB_TEST(found(e, text, pii_type::ipv4, "192.168.12.45"));
        DLIB_TEST(found(e, text, pii_type::payment_card, "4970 1012 3456 7893"));
        // Bare later mentions, learned from the first pass.
        DLIB_TEST(found(e, text, pii_type::person, "Le Goff"));
        DLIB_TEST(found(e, text, pii_type::person, "Dupont"));
        // Common nouns stay readable.
        DLIB_TEST(!covered(e, text, "Service RH"));
        DLIB_TEST(!covered(e, text, "Pierre Philosophale"));

        // Typographic spaces and invisible marks do not hide a value.
        const string nbsp = "Appeler le 06\xC2\xA0" "12\xC2\xA0" "34\xC2\xA0" "56\xC2\xA0" "78 demain.";
        DLIB_TEST(covered(detector.detect(nbsp), nbsp, "06"));
        DLIB_TEST(detector.label_type("Date de naissance :") == pii_type::date);
        DLIB_TEST(detector.label_type("E-mail") == pii_type::email);

        pii_detector_options only_emails;
        only_emails.types = { pii_type::email };
        const pii_detector restricted(only_emails);
        for (const pii_entity& x : restricted.detect(text))
            DLIB_TEST(x.type == pii_type::email);
    }

// ----------------------------------------------------------------------------------------

    size_t element_count(const string& xml)
    {
        size_t n = 0;
        for (const xml_token& t : xml_tokenize(xml))
            if (t.kind == xml_token::start_tag || t.kind == xml_token::end_tag || t.kind == xml_token::empty_tag) ++n;
        return n;
    }

    string utf16le(const string& utf8)
    {
        string out = "\xFF\xFE";
        size_t i = 0;
        while (i < utf8.size())
        {
            const char32_t c = privacy_impl::utf8_next(utf8, i);
            out.push_back(static_cast<char>(c & 0xFF));
            out.push_back(static_cast<char>((c >> 8) & 0xFF));
        }
        return out;
    }

    void test_text_documents()
    {
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x17')));
        pii_detector detector;
        document_anonymizer a(p, detector);

        const string utf8 = letter;
        string cp1252;
        for (size_t i = 0; i < utf8.size();)
            cp1252.push_back(static_cast<char>(privacy_impl::utf8_next(utf8, i)));
        for (const string& source : { utf8, "\xEF\xBB\xBF" + utf8, utf16le(utf8), cp1252 })
        {
            DLIB_TEST(detect_document_format(source) == document_format::text);
            document_report report;
            const string out = a.anonymize(source, document_format::text, report);
            DLIB_TEST(report.total_entities() >= 15);
            DLIB_TEST(out.find("Dupont") == string::npos && out.find("LE GOFF") == string::npos);
            document_report back;
            DLIB_TEST(a.restore(out, document_format::text, back) == source);
            DLIB_TEST(back.tags_restored == report.total_entities());
        }

        const string csv =
            "Nom;Prénom;Email;Téléphone;Commentaire\r\n"
            "DUPONT;Jean;jean.dupont@acme-conseil.fr;06 12 34 56 78;\"Client depuis 2019; RAS\"\r\n"
            "MARTIN;Paul;p.martin@laposte.net;01 45 67 89 10;\"Dit \"\"Polo\"\", voir M. Dupont\"\r\n";
        DLIB_TEST(detect_document_format(csv, "contacts.csv") == document_format::delimited);
        document_report csv_report;
        const string csv_out = a.anonymize(csv, document_format::delimited, csv_report);
        DLIB_TEST(csv_out.find("DUPONT") == string::npos && csv_out.find("MARTIN") == string::npos);
        // The quoted cell keeps its quotes and its separator; the year it dates is protected.
        DLIB_TEST(csv_out.find("\"Client depuis [[DATE:") != string::npos && csv_out.find("; RAS\"") != string::npos);
        DLIB_TEST(csv_out.find("Nom;Prénom;Email") == 0);
        document_report csv_back;
        DLIB_TEST(a.restore(csv_out, document_format::delimited, csv_back) == csv);

        const string xml =
            "<?xml version=\"1.0\"?><employes><employe email=\"jean.dupont@acme-conseil.fr\">"
            "<nom>Jean <b>Dupont</b></nom><note><![CDATA[Vu par Hélène Le Goff <urgent>]]></note></employe></employes>";
        DLIB_TEST(detect_document_format(xml) == document_format::markup);
        document_report xml_report;
        const string xml_out = a.anonymize(xml, document_format::markup, xml_report);
        DLIB_TEST(xml_out.find("Dupont") == string::npos && xml_out.find("Le Goff") == string::npos);
        DLIB_TEST(element_count(xml_out) == element_count(xml));
        document_report xml_back;
        const string xml_restored = a.restore(xml_out, document_format::markup, xml_back);
        DLIB_TEST(xml_restored.find("Jean Dupont") != string::npos &&
                  xml_restored.find("Hélène Le Goff") != string::npos);
        DLIB_TEST(xml_restored.find("email=\"jean.dupont@acme-conseil.fr\"") != string::npos);

        /* A tag followed by a '>' neither closes a CDATA section early nor writes "]]>" in
           character data, and the restoration gives back the original bytes. */
        const string mail =
            "<?xml version=\"1.0\"?><mail><from>Jean Dupont &lt;jean.dupont@acme-conseil.fr></from>"
            "<body><![CDATA[Voir <jean.dupont@acme-conseil.fr> et <urgent>]]></body></mail>";
        document_report mail_report;
        const string mail_out = a.anonymize(mail, document_format::markup, mail_report);
        DLIB_TEST(mail_out.find("jean.dupont") == string::npos);
        auto count_of = [](const string& s, const string& what) {
            size_t n = 0;
            for (size_t at = s.find(what); at != string::npos; at = s.find(what, at + 1)) ++n;
            return n;
        };
        DLIB_TEST(count_of(mail_out, "]]>") == count_of(mail_out, "<![CDATA["));
        DLIB_TEST(count_of(mail_out, "<![CDATA[") == 2);
        document_report mail_back;
        DLIB_TEST(a.restore(mail_out, document_format::markup, mail_back) == mail);
        DLIB_TEST(mail_back.tags_rejected == 0);
    }

// ----------------------------------------------------------------------------------------

    string document_text(const string& xml)
    {
        string text;
        for (const xml_token& t : xml_tokenize(xml))
            if (t.kind == xml_token::text)
                text += xml_unescape(xml.substr(t.content_begin, t.content_end - t.content_begin));
        return text;
    }

    void test_office_document()
    {
        zip_archive_writer w;
        w.add("[Content_Types].xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
              "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
              "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
              "<Default Extension=\"png\" ContentType=\"image/png\"/>"
              "<Override PartName=\"/word/document.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
              "<Override PartName=\"/docProps/core.xml\" "
              "ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
              "</Types>");
        w.add("_rels/.rels",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
              "<Relationship Id=\"rId1\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
              "Target=\"word/document.xml\"/>"
              "<Relationship Id=\"rId2\" "
              "Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" "
              "Target=\"docProps/core.xml\"/></Relationships>");
        w.add("word/_rels/document.xml.rels",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
              "<Relationship Id=\"rId5\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" "
              "Target=\"media/image1.png\"/></Relationships>");
        w.add("word/document.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
            "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
            "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\"><w:body>"
            "<w:p><w:r><w:t xml:space=\"preserve\">Je soussigné, M. </w:t></w:r><w:r><w:rPr><w:b/></w:rPr>"
            "<w:t xml:space=\"preserve\">Jean </w:t></w:r><w:r><w:t>Dupont</w:t></w:r>"
            "<w:r><w:t xml:space=\"preserve\">, joignable au 06 12 34 56 78.</w:t></w:r></w:p>"
            "<w:p><w:r><w:drawing><wp:inline><a:graphic><a:graphicData><a:blip r:embed=\"rId5\"/>"
            "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
            "<w:p><w:r><w:t>Dupont a signé.</w:t></w:r></w:p><w:sectPr/></w:body></w:document>");
        w.add("word/media/image1.png", string("\x89PNG\r\n\x1a\n", 8) + string(64, 'x'));
        w.add("docProps/core.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
            "xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:creator>Jean Dupont</dc:creator></cp:coreProperties>");
        const string docx = w.finish();
        DLIB_TEST(detect_document_format(docx) == document_format::docx);

        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x29')));
        pii_detector detector;
        document_anonymizer a(p, detector);
        document_report report;
        const string out = a.anonymize(docx, document_format::docx, report);
        const zip_archive_reader z(out);
        DLIB_TEST(!z.contains("word/media/image1.png"));
        for (size_t i = 0; i < z.entries().size(); ++i)
            DLIB_TEST_MSG(z.read(i).find("Dupont") == string::npos, z.entries()[i].name);
        DLIB_TEST(z.read("word/document.xml").find("[[PERSON:") != string::npos);
        DLIB_TEST(z.read("word/document.xml").find("blip") == string::npos);

        document_report back;
        const zip_archive_reader r(a.restore(out, document_format::docx, back));
        const string restored = document_text(r.read("word/document.xml"));
        DLIB_TEST_MSG(restored.find("Je soussigné, M. Jean Dupont, joignable au 06 12 34 56 78.") != string::npos,
                      restored);
        DLIB_TEST(restored.find("Dupont a signé.") != string::npos);
        DLIB_TEST(r.read("docProps/core.xml").find("<dc:creator>Jean Dupont</dc:creator>") != string::npos);
    }

// ----------------------------------------------------------------------------------------

    void test_pdf_document()
    {
        pdf_text_document source;
        source.pages.resize(1);
        source.pages[0].x1 = 595;
        source.pages[0].y1 = 842;
        std::vector<std::vector<pdf_text_segment>> lines(1);
        const char* text[] = {
            "Je soussigné, M. Jean Dupont, certifie que Mme Hélène LE GOFF, née le 3 mars 1985 à",
            "Saint-Malo, est joignable au 06 12 34 56 78. Son numéro de sécurité sociale est le 2 85 03",
            "35 238 012 05. Wałęsa signe pour Dupont." };
        for (int k = 0; k < 3; ++k)
        {
            pdf_text_segment s;
            s.text = text[k];
            s.x = 72;
            s.y = 700 - 14 * k;
            s.font_size = 11;
            s.width = 440;
            s.line = static_cast<size_t>(k);
            lines[0].push_back(s);
        }
        const string pdf = write_pdf_text(source, lines);
        DLIB_TEST(detect_document_format(pdf) == document_format::pdf);
        const pdf_text_document reread = read_pdf_text(pdf);
        DLIB_TEST(reread.produced_by_dlib && reread.pages.size() == 1);
        const auto layout = layout_pdf_text(reread);
        DLIB_TEST(layout[0].size() == 3 && layout[0][2].text == text[2]);

        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x31')));
        pii_detector detector;
        document_anonymizer a(p, detector);
        document_report report;
        const string out = a.anonymize(pdf, document_format::pdf, report);
        string all;
        for (const auto& page : layout_pdf_text(read_pdf_text(out)))
            for (const pdf_text_segment& s : page) all += s.text + "\n";
        DLIB_TEST_MSG(all.find("Dupont") == string::npos && all.find("LE GOFF") == string::npos, all);
        DLIB_TEST_MSG(all.find("06 12 34 56 78") == string::npos && all.find("238 012 05") == string::npos, all);

        document_report back;
        string restored;
        for (const auto& page : layout_pdf_text(read_pdf_text(a.restore(out, document_format::pdf, back))))
            for (const pdf_text_segment& s : page) restored += s.text + " ";
        DLIB_TEST_MSG(restored.find("M. Jean Dupont, certifie que Mme Hélène LE GOFF") != string::npos, restored);
        DLIB_TEST(restored.find("Wałęsa") != string::npos);
        DLIB_TEST(back.tags_restored == report.total_entities());

        bool encrypted_refused = false;
        try
        {
            read_pdf_text(
                "%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R /Encrypt 2 0 R >>\n%%EOF\n");
        }
        catch (pdf_error&) { encrypted_refused = true; }
        DLIB_TEST(encrypted_refused);
    }

// ----------------------------------------------------------------------------------------

    void test_more_identifiers()
    {
        pii_detector detector;
        auto check = [&](const string& text, pii_type type, const string& value) {
            const std::vector<pii_entity> e = detector.detect(text);
            const bool ok = found(e, text, type, value);
            if (!ok) dlog << LINFO << "missed " << pii_type_keyword(type) << " [" << value << "] in " << text;
            DLIB_TEST(ok);
        };
        check("Position GPS : 48.8566, 2.3522 relevée.", pii_type::geolocation, "48.8566, 2.3522");
        check("Coordonnées 48°51'24\"N 2°21'03\"E du domicile.", pii_type::geolocation, "48°51'24\"N 2°21'03\"E");
        check("IMEI : 35-209900-176148-1", pii_type::identifier, "35-209900-176148-1");
        check("IMEI 490154203237518", pii_type::identifier, "490154203237518");
        check("ICCID 8933150319011234560", pii_type::identifier, "8933150319011234560");
        check("device 3f2504e0-4f89-11d3-9a0c-0305e82c3301 seen", pii_type::identifier,
              "3f2504e0-4f89-11d3-9a0c-0305e82c3301");
        check("VIN : VF1RFB00X56123456", pii_type::identifier, "VF1RFB00X56123456");
        check("autre véhicule 1M8GDM9AXKP042788", pii_type::identifier, "1M8GDM9AXKP042788");
        check("DNI 12345678Z", pii_type::identifier, "12345678Z");
        check("NIE X1234567L", pii_type::identifier, "X1234567L");
        check("codice fiscale RSSMRA85T10A562S", pii_type::identifier, "RSSMRA85T10A562S");
        check("NINO AB 12 34 56 C", pii_type::identifier, "AB 12 34 56 C");
        check("SSN 123-45-6789", pii_type::identifier, "123-45-6789");
        check("AVS 756.9217.0769.85", pii_type::identifier, "756.9217.0769.85");
        check("registre national 85.07.30-033.28", pii_type::identifier, "85.07.30-033.28");
        check("BSN 111222333", pii_type::identifier, "111222333");
        check("Steuer-ID 86095742719", pii_type::identifier, "86095742719");
        check("PESEL 44051401359", pii_type::identifier, "44051401359");
        check("NIF 501964843", pii_type::identifier, "501964843");
        check("Numéro fiscal 3023217600053", pii_type::identifier, "3023217600053");
        check("RIB : 30004 00550 00012345678 40", pii_type::identifier, "30004 00550 00012345678 40");
        check("Passeport n° 12AB34567", pii_type::identifier, "12AB34567");
        check("Tél. : 020 7946 0958", pii_type::phone, "020 7946 0958");
        check("call (555) 123-4567 now", pii_type::phone, "(555) 123-4567");
        check("fax +44 20 7946 0959", pii_type::phone, "+44 20 7946 0959");
        check("Suivez @jdupont35 sur X", pii_type::identifier, "@jdupont35");
        check("Votre dossier n° 2024-00123 est ouvert", pii_type::identifier, "2024-00123");
        check("la commande CMD-2024-000123 arrive", pii_type::identifier, "CMD-2024-000123");
        check("Véhicule immatriculé 1234 AB 35 depuis", pii_type::license_plate, "1234 AB 35");

        // What these scanners must leave alone.
        auto clean = [&](const string& text) {
            const bool none = detector.detect(text).empty();
            if (!none) dlog << LINFO << "false positive in " << text;
            DLIB_TEST(none);
        };
        clean("Le prix unitaire est 12.5000, 3.2000 par lot.");
        clean("L'hôtel 12345678 n'a rien à voir.");
        clean("La méthode @Override ne compte pas.");
        clean("Le contrat 2024 prévoit une hausse de 2024 euros.");   // a year dates a fact after "en", "depuis"...
        clean("Carte 4970 1012 3456 7894 refusée");     // fails Luhn

        // The new values are protected and restored exactly.
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x37')));
        for (const auto& v : { std::make_pair(pii_type::geolocation, string("48.8566, 2.3522")),
                               std::make_pair(pii_type::geolocation, string("48°51'24\"N 2°21'03\"E")),
                               std::make_pair(pii_type::identifier, string("3f2504e0-4f89-11d3-9a0c-0305e82c3301")),
                               std::make_pair(pii_type::identifier, string("RSSMRA85T10A562S")) })
        {
            const string tag = p.tag(v.first, v.second);
            DLIB_TEST(tag.compare(0, 2 + string(pii_type_keyword(v.first)).size(),
                                  "[[" + string(pii_type_keyword(v.first))) == 0);
            DLIB_TEST(p.restore(tag) == v.second);
        }
        const string uuid_tag = p.tag(pii_type::identifier, "3f2504e0-4f89-11d3-9a0c-0305e82c3301");
        DLIB_TEST(privacy_impl::is_uuid_shape(uuid_tag.substr(5, 36)));
        DLIB_TEST(pii_type_from_keyword("GEO") == pii_type::geolocation);
        DLIB_TEST(pii_type_from_keyword("gps") == pii_type::geolocation);
    }

// ----------------------------------------------------------------------------------------

    void test_rule_refinements()
    {
        pii_detector detector;
        auto entities = [&](const string& text) {
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };
        // "Dr" before a name is a doctor; "Dr" after one is a drive.
        string text = "Vu par Dr Octave ROCHE le matin.";
        std::vector<pii_entity> e = entities(text);
        DLIB_TEST(found(e, text, pii_type::person, "Octave ROCHE"));
        DLIB_TEST(!found(e, text, pii_type::address, "Dr Octave ROCHE"));
        text = "Send it to 221B Baker Street, London, or to 56 Oak Dr.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::address, "221B Baker Street"));
        DLIB_TEST(found(e, text, pii_type::address, "56 Oak Dr"));
        // A labelled address is split into the street and the place.
        text = "Adresse : 12 rue des Lilas, 35000 Rennes";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::address, "12 rue des Lilas"));
        DLIB_TEST(found(e, text, pii_type::location, "35000 Rennes"));
        // Without a number, "le mail de" someone is an email and "la place de" someone a replacement.
        text = "Tu as vu le mail de Federica Marino ? Marie signe à la place de Jean Martin. "
               "Il habite 12 mail François Mitterrand.";
        e = entities(text);
        DLIB_TEST(!covered(e, text, "mail de Federica") && !covered(e, text, "place de Jean"));
        DLIB_TEST(found(e, text, pii_type::person, "Federica Marino"));
        DLIB_TEST(found(e, text, pii_type::address, "12 mail François Mitterrand"));
        // References announced by the noun they number, and a file name that is none.
        text = "Following up on file c4837928, your request 2022-73789 and reference \"A4435122\". Ref: 2476486. "
               "Helys Systems, company number 2192507. Asset tag: 1109797. The file export_2010.csv is attached.";
        e = entities(text);
        for (const char* value : { "c4837928", "2022-73789", "A4435122", "2476486", "2192507", "1109797" })
            DLIB_TEST(found(e, text, pii_type::identifier, value));
        DLIB_TEST(!covered(e, text, "export_2010"));
        text = "Username: <username>\nPassword: <password>";
        e = entities(text);
        DLIB_TEST(!covered(e, text, "username>"));
        // Organisation labels.
        text = "Raison sociale : TAVELIS SAS\nEmployeur : Novexa (Rennes)";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "TAVELIS SAS"));
        DLIB_TEST(found(e, text, pii_type::organization, "Novexa"));

        /* A recognizer whose span wins over a name the rules found with a title keeps the
           strength of that name: its bare mention later in the document is still found. */
        pii_detector with_model;
        with_model.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            const size_t p = t.find("Hélène LE GOFF");
            if (p != string::npos)
            {
                pii_span g, f;
                g.begin = p; g.end = p + string("Hélène").size(); g.label = "GIVEN_NAME"; g.confidence = 0.8;
                f.begin = g.end + 1; f.end = p + string("Hélène LE GOFF").size();
                f.label = "FAMILY_NAME"; f.confidence = 0.8;
                spans.push_back(g);
                spans.push_back(f);
            }
            return spans;
        });
        const string first = "Dossier de Madame Hélène LE GOFF, reçu hier.";
        const string later = "LE GOFF a signé le contrat.";
        with_model.begin_document();
        with_model.learn(first);
        with_model.learn(later);
        DLIB_TEST(found(with_model.detect(first), first, pii_type::person, "Hélène LE GOFF"));
        DLIB_TEST(covered(with_model.detect(later), later, "LE GOFF"));
    }

// ----------------------------------------------------------------------------------------

    void test_patterns_and_recognizers()
    {
        // Regular expressions: a capture group, a check, and case folding.
        pii_detector detector;
        pii_pattern badge;
        badge.type = pii_type::identifier;
        badge.regex = "badge\\s*:?\\s*(E-[0-9]{5})";
        badge.group = 1;
        badge.ignore_case = true;
        detector.add_pattern(badge);
        pii_pattern card;
        card.type = pii_type::payment_card;
        card.regex = "[0-9]{4}( [0-9]{4}){3}";
        card.check = "luhn";
        detector.add_pattern(card);
        DLIB_TEST(detector.pattern_count() == 2);
        const string text = "BADGE : E-48213 ; cartes 4970 1012 3456 7893 et 4970 1012 3456 7894.";
        const std::vector<pii_entity> e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "E-48213"));
        DLIB_TEST(found(e, text, pii_type::payment_card, "4970 1012 3456 7893"));
        DLIB_TEST(!covered(e, text, "4970 1012 3456 7894"));
        bool refused = false;
        try
        {
            pii_pattern bad;
            bad.regex = "([a-z";
            detector.add_pattern(bad);
        }
        catch (error&) { refused = true; }
        DLIB_TEST(refused);

        // A recognizer: adjacent given and family names make one person, low confidence is
        // dropped, and a confident full name teaches the bare surname to the second pass.
        pii_detector with_model;
        with_model.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& value, const string& label, double confidence) {
                const size_t p = t.find(value);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p;
                s.end = p + value.size();
                s.label = label;
                s.confidence = confidence;
                spans.push_back(s);
            };
            add("Maëlys", "GIVEN_NAME", 0.97);
            add("Quintrec", "FAMILY_NAME", 0.95);
            add("Kerlouan", "CITY", 0.3);
            return spans;
        });
        DLIB_TEST(with_model.recognizer_count() == 1);
        const string doc = "Signature de Maëlys Quintrec, de passage à Kerlouan. Quintrec reviendra.";
        with_model.begin_document();
        with_model.learn(doc);
        const std::vector<pii_entity> m = with_model.detect(doc);
        DLIB_TEST(found(m, doc, pii_type::person, "Maëlys Quintrec"));
        DLIB_TEST(!covered(m, doc, "Kerlouan"));
        DLIB_TEST(found(m, doc, pii_type::person, "Quintrec"));
        DLIB_TEST(pii_type_from_label("CITY") == pii_type::location);
        DLIB_TEST(pii_type_from_label("USERNAME") == pii_type::identifier);

        // A greeting read as a name or a place by a recognizer stays readable.
        pii_detector greeted;
        greeted.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& value, const string& label) {
                const size_t p = t.find(value);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p;
                s.end = p + value.size();
                s.label = label;
                s.confidence = 0.95;
                spans.push_back(s);
            };
            add("Bonsoir", "FAMILY_NAME");
            add("Hello", "CITY");
            add("Maëlys", "GIVEN_NAME");
            return spans;
        });
        const string note = "Bonsoir, Maëlys a rappelé. Hello, la suite demain.";
        greeted.begin_document();
        greeted.learn(note);
        const std::vector<pii_entity> g = greeted.detect(note);
        DLIB_TEST(!covered(g, note, "Bonsoir"));
        DLIB_TEST(!covered(g, note, "Hello"));
        DLIB_TEST(found(g, note, pii_type::person, "Maëlys"));

        /* Neither function words, nor a common word, nor pieces of words, nor a capitalized
           word joined to lower case words make a name; a file name holds no login. */
        pii_detector guarded;
        guarded.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& value, const string& label) {
                const size_t p = t.find(value);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p;
                s.end = p + value.size();
                s.label = label;
                s.confidence = 0.95;
                spans.push_back(s);
            };
            add("Le projet", "FAMILY_NAME");
            add("Votre", "GIVEN_NAME");
            add("Rem", "FAMILY_NAME");
            add("commande", "FAMILY_NAME");
            add("final.docx", "USERNAME");
            add("Le logiciel Excel", "ORGANIZATION");
            add("Jean", "GIVEN_NAME");
            add("d'Arras", "FAMILY_NAME");
            add("Le Goff", "FAMILY_NAME");
            add("Saint-Aubin-sur-Oust", "CITY");
            return spans;
        });
        const string memo = "Le projet avance. Votre commande part. Remplacez rapport_final.docx dans "
                            "Le logiciel Excel. Jean d'Arras et Hélène Le Goff sont à Saint-Aubin-sur-Oust.";
        guarded.begin_document();
        guarded.learn(memo);
        const std::vector<pii_entity> gm = guarded.detect(memo);
        DLIB_TEST(!covered(gm, memo, "Le projet") && !covered(gm, memo, "Votre"));
        DLIB_TEST(!covered(gm, memo, "Rem") && !covered(gm, memo, "commande"));
        DLIB_TEST(!covered(gm, memo, "final.docx") && !covered(gm, memo, "logiciel"));
        DLIB_TEST(found(gm, memo, pii_type::person, "Jean d'Arras"));
        DLIB_TEST(covered(gm, memo, "Le Goff"));
        DLIB_TEST(found(gm, memo, pii_type::location, "Saint-Aubin-sur-Oust"));

        /* The pieces a recognizer cuts inside a word or a name are judged once merged: the
           pieces of "Esch-sur-Alzette" and of "christina", the particle of "Inès de Brunet",
           and "City" after "Quebec" belong to the name. */
        pii_detector pieced;
        pieced.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& word, size_t offset, size_t length, const string& label) {
                const size_t p = t.find(word);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p + offset;
                s.end = p + offset + (length == string::npos ? word.size() - offset : length);
                s.label = label;
                s.confidence = 0.95;
                spans.push_back(s);
            };
            add("4002 ", 0, 4, "POSTAL_CODE");
            add("Esch-sur-Alzette", 0, 1, "CITY");
            add("Esch-sur-Alzette", 1, string::npos, "CITY");
            add("Inès", 0, string::npos, "GIVEN_NAME");
            add(" de Brunet", 1, 2, "FAMILY_NAME");
            add("Brunet", 0, string::npos, "FAMILY_NAME");
            add("Quebec", 0, string::npos, "CITY");
            add("City", 0, string::npos, "CITY");
            add("christina", 0, 5, "GIVEN_NAME");
            add("christina", 5, string::npos, "GIVEN_NAME");
            return spans;
        });
        const string pieces = "Le bailleur demeure au 4002 Esch-sur-Alzette. Inès de Brunet part pour Quebec City. "
                              "Hey christina, did Isla reply?";
        pieced.begin_document();
        pieced.learn(pieces);
        const std::vector<pii_entity> gp = pieced.detect(pieces);
        DLIB_TEST(found(gp, pieces, pii_type::location, "4002 Esch-sur-Alzette"));
        DLIB_TEST(found(gp, pieces, pii_type::person, "Inès de Brunet"));
        DLIB_TEST(found(gp, pieces, pii_type::location, "Quebec City"));
        DLIB_TEST(found(gp, pieces, pii_type::person, "christina"));

        /* The pieces of a login meet and form one login, the at sign of a handle belongs to it,
           and a common word is no login. */
        pii_detector logins;
        logins.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& word, size_t offset, size_t length) {
                const size_t p = t.find(word);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p + offset;
                s.end = p + offset + (length == string::npos ? word.size() - offset : length);
                s.label = "USERNAME";
                s.confidence = 0.95;
                spans.push_back(s);
            };
            add("vwojcikski", 0, 1);
            add("vwojcikski", 1, string::npos);
            add("hpetit38", 0, string::npos);
            add("month", 0, string::npos);
            return spans;
        });
        const string chat = "<vwojcikski> anyone got access to the server?\n@hpetit38 merci. Rent is due every month.";
        logins.begin_document();
        logins.learn(chat);
        const std::vector<pii_entity> gl = logins.detect(chat);
        DLIB_TEST(found(gl, chat, pii_type::identifier, "vwojcikski"));
        DLIB_TEST(found(gl, chat, pii_type::identifier, "@hpetit38"));
        DLIB_TEST(!covered(gl, chat, "month"));
    }

// ----------------------------------------------------------------------------------------

    void test_attributes_and_public_bodies()
    {
        pii_detector detector;
        auto entities = [&](const string& text) {
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };
        // A civility before a name tells the sex of the person and is replaced too.
        string text = "M. et Mme Martin habitent Rennes. Monsieur le Maire a reçu Monsieur Jean Petit.";
        std::vector<pii_entity> e = entities(text);
        DLIB_TEST(found(e, text, pii_type::title, "M."));
        DLIB_TEST(found(e, text, pii_type::title, "Mme"));
        DLIB_TEST(found(e, text, pii_type::person, "Jean Petit"));
        DLIB_TEST(!covered(e, text, "Monsieur le Maire"));
        text = "Madame, Monsieur,\nNous accusons réception de votre courrier.";
        DLIB_TEST(entities(text).empty());

        // Labelled values of closed vocabularies, and values announced by a cue.
        text = "Civilité : Mme\nSexe : F\nNationalités : française et marocaine\nTitre : Rapport annuel 2025\n"
               "Genre : roman policier";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::title, "Mme"));
        DLIB_TEST(found(e, text, pii_type::gender, "F"));
        DLIB_TEST(found(e, text, pii_type::nationality, "française"));
        DLIB_TEST(found(e, text, pii_type::nationality, "marocaine"));
        DLIB_TEST(!covered(e, text, "Rapport annuel") && !covered(e, text, "roman policier"));
        text = "Le patient, de sexe masculin, est ressortissant algérien. Mrs Smith is a British citizen.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::gender, "masculin"));
        DLIB_TEST(found(e, text, pii_type::nationality, "algérien"));
        DLIB_TEST(found(e, text, pii_type::nationality, "British"));

        // Public bodies are organisations; a head noun with generic words only is not.
        text = "La CAF de Rennes, l'URSSAF et le Tribunal judiciaire de Rennes ; la direction générale.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "CAF de Rennes"));
        DLIB_TEST(found(e, text, pii_type::organization, "URSSAF"));
        DLIB_TEST(found(e, text, pii_type::organization, "Tribunal judiciaire de Rennes"));
        DLIB_TEST(!covered(e, text, "direction générale"));
        text = "A letter from HMRC and the Department for Work and Pensions, copied to the Home Office.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "Department for Work and Pensions"));
        DLIB_TEST(found(e, text, pii_type::organization, "Home Office"));
        pii_detector_options no_bodies;
        no_bodies.public_bodies = false;
        const pii_detector without(no_bodies);
        DLIB_TEST(!covered(without.detect(text), text, "HMRC"));

        // The new values are protected and restored.
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x42')));
        for (const auto& v : { std::make_pair(pii_type::title, string("Madame")),
                               std::make_pair(pii_type::gender, string("F")),
                               std::make_pair(pii_type::nationality, string("française")) })
            DLIB_TEST(p.restore(p.tag(v.first, v.second)) == v.second);
        DLIB_TEST(p.tag(pii_type::title, "Madame") != p.tag(pii_type::title, "Madame"));
    }

// ----------------------------------------------------------------------------------------

    void test_known_names()
    {
        pii_detector detector;
        auto entities = [&](const string& text) {
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };
        // Organisations known by their own name, from any country.
        string text = "Il travaille chez Airbus depuis 2010, après Johnson & Johnson et AG2R La Mondiale.";
        std::vector<pii_entity> e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "Airbus"));
        DLIB_TEST(found(e, text, pii_type::organization, "Johnson & Johnson"));
        DLIB_TEST(found(e, text, pii_type::organization, "AG2R La Mondiale"));
        text = "Son compte est à La Banque postale ; AXA France assure le logement de Chiamaka Okafor.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "La Banque postale"));
        DLIB_TEST(found(e, text, pii_type::organization, "AXA France"));
        DLIB_TEST(found(e, text, pii_type::person, "Chiamaka Okafor"));
        // Given names of any origin, a family name of two letters, a greeting outside the name.
        text = "Je soussigné Moussa BA, demeurant à Dakar. Bonjour Murat, merci.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::person, "Moussa BA"));
        DLIB_TEST(found(e, text, pii_type::person, "Murat") && !covered(e, text, "Bonjour"));
        // A common phrase, a product, a person and a domain stay what they are.
        text = "L'équipe a gagné. The economist said so. Microsoft Teams ne démarre pas dans l'Airbus A320.";
        e = entities(text);
        DLIB_TEST(!covered(e, text, "équipe") && !covered(e, text, "economist"));
        DLIB_TEST(!covered(e, text, "Microsoft") && !covered(e, text, "Airbus"));
        text = "John Harvard a écrit à contact@airbus.com.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::person, "John Harvard"));
        DLIB_TEST(found(e, text, pii_type::email, "contact@airbus.com"));
        // Known organisations are no public bodies; names of public bodies may hold commas.
        pii_detector_options no_bodies;
        no_bodies.public_bodies = false;
        const pii_detector without(no_bodies);
        text = "Il est employé par L'Oréal.";
        DLIB_TEST(found(without.detect(text), text, pii_type::organization, "L'Oréal"));
        text = "A letter from the Foreign, Commonwealth & Development Office.";
        DLIB_TEST(found(entities(text), text, pii_type::organization, "Foreign, Commonwealth & Development Office"));
        /* The pieces a recognizer reads inside a known organisation leave it whole, while a
           span that goes beyond it keeps its own boundaries. */
        pii_detector pieces;
        pieces.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& value, const char* label) {
                const size_t p = t.find(value);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p; s.end = p + value.size(); s.label = label; s.confidence = 0.95;
                spans.push_back(s);
            };
            add("Banco", "ORGANIZATION");
            add("Santander", "CITY");
            add("Bahn", "ORGANIZATION");
            add("Projet Atlas Airbus", "ORGANIZATION");
            return spans;
        });
        text = "Virement chez Banco Santander pour Deutsche Bahn, au titre du Projet Atlas Airbus.";
        e = pieces.detect(text);
        DLIB_TEST(found(e, text, pii_type::organization, "Banco Santander"));
        DLIB_TEST(found(e, text, pii_type::organization, "Deutsche Bahn"));
        DLIB_TEST(found(e, text, pii_type::organization, "Projet Atlas Airbus"));
        /* Learned spans that name nothing: a country read as an organisation, the civility
           "M." read as a given name, a participle before "à"; a postal code of another country
           read in part keeps the boundaries of its layout. */
        pii_detector guarded;
        guarded.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& value, const char* label) {
                const size_t p = t.find(value);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p; s.end = p + value.size(); s.label = label; s.confidence = 0.95;
                spans.push_back(s);
            };
            add("Pologne", "ORGANIZATION");
            add("M", "GIVEN_NAME");
            add("Takeshi", "GIVEN_NAME");
            add("Yamamoto", "FAMILY_NAME");
            add("Fait", "ORGANIZATION");
            add("Tokyo", "CITY");
            add("153", "POSTAL_CODE");
            return spans;
        });
        text = "Pologne";
        DLIB_TEST(!covered(guarded.detect(text), text, "Pologne"));
        text = "Suite à notre échange avec M. Takeshi Yamamoto, 2-11-3 Meguro, Tokyo 153-0063.";
        e = guarded.detect(text);
        DLIB_TEST(found(e, text, pii_type::title, "M."));
        DLIB_TEST(found(e, text, pii_type::person, "Takeshi Yamamoto"));
        DLIB_TEST(found(e, text, pii_type::location, "Tokyo 153-0063"));
        text = "Fait à Lyon, le 12 mars 2026.";
        e = guarded.detect(text);
        DLIB_TEST(!covered(e, text, "Fait") && covered(e, text, "Lyon"));

        // Nicknames, stage names and pen names after their cue; reported speech stays readable.
        text = "Jean Dupont, dit « Jojo », a vu Marie Martin, surnommée Mimi. John Smith a.k.a. \"Smithy\" fled.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::person, "Jojo"));
        DLIB_TEST(found(e, text, pii_type::person, "Mimi"));
        DLIB_TEST(found(e, text, pii_type::person, "Smithy"));
        text = "Il a publié sous le pseudonyme de Vernon Sullivan.";
        DLIB_TEST(found(entities(text), text, pii_type::person, "Vernon Sullivan"));
        text = "Il a dit « bonjour » en entrant. Jean Dupont dit « Je viens » et sort.";
        e = entities(text);
        DLIB_TEST(!covered(e, text, "bonjour") && !covered(e, text, "Je viens"));
    }

// ----------------------------------------------------------------------------------------

    void test_learned_organizations()
    {
        /* The organisations a recognizer reads take the boundaries of the annotation: its
           pieces are joined, legal forms and head nouns taken, the words of the sentence left
           out, and a place after a legal form kept apart. */
        auto spans = std::make_shared<std::vector<std::pair<string, string>>>();
        pii_detector detector;
        detector.add_recognizer([spans](const string& t) {
            std::vector<pii_span> out;
            for (const auto& v : *spans)
            {
                const size_t p = t.find(v.first);
                if (p == string::npos) continue;
                pii_span s;
                s.begin = p; s.end = p + v.first.size(); s.label = v.second; s.confidence = 0.95;
                out.push_back(s);
            }
            std::sort(out.begin(), out.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
            return out;
        });
        auto read = [&](const string& text, const std::vector<std::pair<string, string>>& values) {
            *spans = values;
            detector.begin_document();
            return detector.detect(text);
        };
        // Legal forms after the name, written as forms.
        string text = "Signé par Armium a.s. et par Foxglove A/S, puis par Kerexa SARL AU.";
        std::vector<pii_entity> e = read(text, {{"Armium", "ORGANIZATION"}, {"Foxglove", "ORGANIZATION"},
                                                 {"Kerexa SARL", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Armium a.s."));
        DLIB_TEST(found(e, text, pii_type::organization, "Foxglove A/S"));
        DLIB_TEST(found(e, text, pii_type::organization, "Kerexa SARL AU"));
        text = "HERMINE PVT. LTD.\nReferences: Xanium Insaat Ltd. Sti. (Istanbul)";
        e = read(text, {{"HERMINE PVT", "ORGANIZATION"}, {"Xanium Insaat Ltd.", "ORGANIZATION"},
                        {"Sti", "FAMILY_NAME"}});
        DLIB_TEST(found(e, text, pii_type::organization, "HERMINE PVT. LTD."));
        DLIB_TEST(found(e, text, pii_type::organization, "Xanium Insaat Ltd. Sti."));
        // Pieces of one name, the trade words between them, partners after a conjunction.
        text = "Customer Service, Merrow Oil and Gas Plc\nCLUB: BARAN I WSPÓLNICY S.A.";
        e = read(text, {{"Merrow Oil", "ORGANIZATION"}, {"Gas Plc", "ORGANIZATION"}, {"BARAN", "GIVEN_NAME"},
                        {"SPÓLNICY S.A", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Merrow Oil and Gas Plc"));
        DLIB_TEST(found(e, text, pii_type::organization, "BARAN I WSPÓLNICY S.A."));
        // Head nouns and legal forms alone take the name after them.
        text = "Committee minutes, Club Deportivo Madrid: Dr Caitlin Hall, TOV Briora, was elected chair.";
        e = read(text, {{"Club Deportivo", "ORGANIZATION"}, {"Caitlin", "GIVEN_NAME"}, {"Hall", "FAMILY_NAME"},
                        {"TOV", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Club Deportivo Madrid"));
        DLIB_TEST(found(e, text, pii_type::organization, "TOV Briora"));
        DLIB_TEST(found(e, text, pii_type::person, "Caitlin Hall"));
        // A place after a legal form is a place; a legal form ends the name.
        text = "Branch: Larkspur ltée, São Paulo\nCustomer Service, Solix FZE Save the file as export.xlsx.";
        e = read(text, {{"Larkspur ltée", "ORGANIZATION"}, {"São Paulo", "ORGANIZATION"},
                        {"Solix FZE Save", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Larkspur ltée"));
        DLIB_TEST(found(e, text, pii_type::location, "São Paulo"));
        DLIB_TEST(found(e, text, pii_type::organization, "Solix FZE") && !covered(e, text, "Save"));
        // The words of the sentence around the name: capitals, an article, a person before an elision.
        text = "DE MAILLARD CÉLESTIN DIRIGE DEPUIS DIX ANS ELEXA LTÉE, INSTALLÉE À PATRA.";
        e = read(text, {{"CÉLESTIN DIRIGE DEPUIS DIX ANS ELEXA LTÉE", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "ELEXA LTÉE"));
        text = "Courrier de La Police nationale. Typhaine Marnier d'Abeona S.r.l. souhaite te rencontrer.";
        e = read(text, {{"La Police nationale", "ORGANIZATION"},
                        {"Typhaine Marnier d'Abeona S.r.l.", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Police nationale") && !covered(e, text, "La Police"));
        DLIB_TEST(found(e, text, pii_type::organization, "Abeona S.r.l."));
        // A particle before a family name opens the name, at the start of a line too.
        text = "Le Moigne & Sons plc\nDirection des ressources humaines";
        e = read(text, {{"Le", "ORGANIZATION"}, {"Moigne & Sons plc", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Le Moigne & Sons plc"));
        // A common word alone or a common word in lower case names no organisation.
        text = "Convocation\nThe embassy of Spain in Warsaw informs Mr. Larry Begum that his passport is ready.";
        e = read(text, {{"Convocation", "ORGANIZATION"}, {"embassy of Spain", "ORGANIZATION"},
                        {"Warsaw", "CITY"}});
        DLIB_TEST(!covered(e, text, "Convocation") && !covered(e, text, "embassy"));
        DLIB_TEST(found(e, text, pii_type::location, "Warsaw"));
        // A known organisation keeps its type against a piece read as a place.
        text = "LOPEZ Tony, Vueling";
        e = read(text, {{"LOPEZ", "FAMILY_NAME"}, {"Tony", "GIVEN_NAME"}, {"Vueling", "CITY"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Vueling"));
        DLIB_TEST(found(e, text, pii_type::person, "LOPEZ Tony"));
    }

// ----------------------------------------------------------------------------------------

    void test_learned_places()
    {
        /* A country or a state stays readable when a recognizer reads it as a place, and a
           country after an address value stays out of it. */
        auto spans = std::make_shared<std::vector<std::pair<string, string>>>();
        pii_detector detector;
        detector.add_recognizer([spans](const string& t) {
            std::vector<pii_span> out;
            for (const auto& v : *spans)
            {
                const size_t p = t.find(v.first);
                if (p == string::npos) continue;
                pii_span s;
                s.begin = p; s.end = p + v.first.size(); s.label = v.second; s.confidence = 0.95;
                out.push_back(s);
            }
            std::sort(out.begin(), out.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
            return out;
        });
        auto read = [&](const string& text, const std::vector<std::pair<string, string>>& values) {
            *spans = values;
            detector.begin_document();
            return detector.detect(text);
        };
        string text = "34 Hay Street, Sydney NSW 2010, Australia";
        std::vector<pii_entity> e = read(text, {{"Sydney", "CITY"}, {"NSW", "POSTAL_CODE"}, {"2010", "POSTAL_CODE"},
                                                 {"Australia", "CITY"}});
        DLIB_TEST(found(e, text, pii_type::location, "Sydney"));
        DLIB_TEST(found(e, text, pii_type::location, "2010"));
        DLIB_TEST(!covered(e, text, "NSW") && !covered(e, text, "Australia"));
        text = "Chevalier Holdings LLC\n47, rue Joss\nDouala\nCameroun\n\nPietermaritzburg 3201, Afrique du Sud";
        e = read(text, {{"Douala", "CITY"}, {"Cameroun", "CITY"}, {"Pietermaritzburg", "CITY"}, {"3201", "POSTAL_CODE"},
                        {"Sud", "CITY"}});
        DLIB_TEST(found(e, text, pii_type::location, "Douala"));
        DLIB_TEST(found(e, text, pii_type::location, "Pietermaritzburg 3201"));
        DLIB_TEST(!covered(e, text, "Cameroun") && !covered(e, text, "Sud"));
        // The pieces of an address cut inside a word are one address.
        text = "ul. Długa 5\n00-238 Warszawa";
        e = read(text, {{"u", "STREET_ADDRESS"}, {"l. Długa 5", "STREET_ADDRESS"}, {"00-238", "POSTAL_CODE"},
                        {"Warszawa", "CITY"}});
        DLIB_TEST(found(e, text, pii_type::address, "ul. Długa 5"));
        // A country that is also the name of its capital stays a place.
        text = "Siège : 3 Rue de la Gare, L-2449 Luxembourg";
        e = read(text, {{"L-2449", "POSTAL_CODE"}, {"Luxembourg", "CITY"}});
        DLIB_TEST(covered(e, text, "Luxembourg"));
        // The streets of other countries, the number after the name, by the rules alone.
        text = "ul. Długa 5\nCalle Sol, 86\nRua das Flores, 143\nKossuth Lajos utca 16\nAtatürk Caddesi No:12\n"
               "Hauptstraße 12\nKeizersgracht 123\nKarl Johans gate 1\nBerliner Str. 5\nGarantie 2 ans";
        e = read(text, {});
        for (const char* street : { "ul. Długa 5", "Calle Sol, 86", "Rua das Flores, 143", "Kossuth Lajos utca 16",
                                    "Atatürk Caddesi No:12", "Hauptstraße 12", "Keizersgracht 123", "Karl Johans gate 1",
                                    "Berliner Str. 5" })
            DLIB_TEST_MSG(found(e, text, pii_type::address, street), street);
        DLIB_TEST(!covered(e, text, "Garantie"));
        // The rules alone, after the label of an address.
        text = "Adresse : 40 rue Tarik Ibn Ziad, 60000 Oujda, Maroc\nAddress: 15 Harambee Avenue, Nairobi 00200, Kenya";
        e = read(text, {});
        DLIB_TEST(covered(e, text, "Oujda") && !covered(e, text, "Maroc"));
        DLIB_TEST(covered(e, text, "Nairobi") && !covered(e, text, "Kenya"));
        text = "Adresse : 21 Oak Avenue, Miami, FL 33130, États-Unis. Est-elle toujours valable ?";
        e = read(text, {});
        DLIB_TEST(covered(e, text, "Miami") && !covered(e, text, "États-Unis") && !covered(e, text, "Est-elle"));
    }

    void test_learned_terms()
    {
        /* The terms of a medical record and the cells of its tables name nobody, whatever a
           recognizer reads in them; the names around them stay found. */
        struct value { string text; string label; double confidence; };
        auto spans = std::make_shared<std::vector<value>>();
        pii_detector detector;
        detector.add_recognizer([spans](const string& t) {
            std::vector<pii_span> out;
            size_t from = 0;
            for (const auto& v : *spans)
            {
                const size_t p = t.find(v.text, from);
                if (p == string::npos) continue;
                pii_span s;
                s.begin = p; s.end = p + v.text.size(); s.label = v.label; s.confidence = v.confidence;
                out.push_back(s);
                from = s.end;
            }
            return out;
        });
        auto read = [&](const string& text, const std::vector<value>& values) {
            *spans = values;
            detector.begin_document();
            return detector.detect(text);
        };
        string text = "TSH 2,1 mUI/L\nLDL-cholestérol 1,52 g/L\nCréatinine (DFG CKD-EPI) 8,0 mg/L\n"
                      "Hypertension artérielle\nFréquence cardiaque 68 bpm 60 – 100 Normale\nPage 1";
        std::vector<pii_entity> e = read(text, {{"UI", "CITY", 0.9}, {"L", "CITY", 0.9},
                                                {"DL-cholestérol", "ORGANIZATION", 0.9},
                                                {"Créatinine", "GIVEN_NAME", 0.9}, {"Hypertension", "ORGANIZATION", 0.9},
                                                {"60", "STREET_ADDRESS", 0.9}, {"Page", "CITY", 0.9}});
        for (const pii_entity& x : e)
            DLIB_TEST_MSG(false, text.substr(x.begin, x.end - x.begin));
        text = "Bruit (quais de chargement) : Mme Laure Quéméner, 12 rue des Lilas, Quimper (29)";
        e = read(text, {{"quais", "ORGANIZATION", 0.9}, {"Laure", "GIVEN_NAME", 0.95},
                        {"Quéméner", "FAMILY_NAME", 0.95}, {"12", "STREET_ADDRESS", 0.9},
                        {"rue des Lilas", "STREET_ADDRESS", 0.9}, {"Quimper", "CITY", 0.95}});
        DLIB_TEST(!covered(e, text, "quais"));
        DLIB_TEST(found(e, text, pii_type::person, "Laure Quéméner"));
        DLIB_TEST(covered(e, text, "12 rue") && covered(e, text, "rue des Lilas"));
        DLIB_TEST(found(e, text, pii_type::location, "Quimper (29)"));
        // A line in lower case keeps its organisations; an elided article stays outside one.
        text = "elle travaille chez kerlys conseil depuis peu";
        e = read(text, {{"kerlys conseil", "ORGANIZATION", 0.9}});
        DLIB_TEST(found(e, text, pii_type::organization, "kerlys conseil"));
        text = "Véhicule de service, formation de l'Auto-école du Steir";
        e = read(text, {{"l'Auto-école du", "ORGANIZATION", 0.6}});
        DLIB_TEST(found(e, text, pii_type::organization, "Auto-école du Steir"));
    }

    void test_name_edges()
    {
        /* A name keeps the end of its word after an apostrophe and the capitalized particle
           written before it, and a first name joins the family name that a recognizer reads as
           an organisation or an address. */
        struct value { string text; string label; double confidence; };
        auto spans = std::make_shared<std::vector<value>>();
        pii_detector detector;
        detector.add_recognizer([spans](const string& t) {
            std::vector<pii_span> out;
            size_t from = 0;
            for (const auto& v : *spans)
            {
                const size_t p = t.find(v.text, from);
                if (p == string::npos) continue;
                pii_span s;
                s.begin = p; s.end = p + v.text.size(); s.label = v.label; s.confidence = v.confidence;
                out.push_back(s);
                from = s.end;
            }
            return out;
        });
        auto read = [&](const string& text, const std::vector<value>& values) {
            *spans = values;
            detector.begin_document();
            return detector.detect(text);
        };
        // The Breton "c'h" and "Prud'homme" after a civility, with the rules alone.
        string text = "Visite de Mme Prud'homme et de M. Le Floc'h, puis de Mme Le Bec'h.";
        std::vector<pii_entity> e = read(text, {});
        DLIB_TEST(found(e, text, pii_type::person, "Prud'homme"));
        DLIB_TEST(found(e, text, pii_type::person, "Le Floc'h"));
        DLIB_TEST(found(e, text, pii_type::person, "Le Bec'h"));
        // A place and an address that a recognizer stops at the apostrophe.
        text = "Né à Penmarc'h, domicilié 12 rue de Penmarc'h.";
        e = read(text, {{"Penmarc", "CITY", 0.95}, {"12 rue de Penmarc", "STREET_ADDRESS", 0.95}});
        DLIB_TEST(found(e, text, pii_type::location, "Penmarc'h"));
        DLIB_TEST(found(e, text, pii_type::address, "12 rue de Penmarc'h"));
        // An English possessive and the elision of "que" stay outside.
        text = "Mr Martin's report was late.";
        e = read(text, {{"Martin", "FAMILY_NAME", 0.95}});
        DLIB_TEST(found(e, text, pii_type::person, "Martin"));
        DLIB_TEST(!covered(e, text, "'s report"));
        // A particle in the case of the name opens it; "de" before a name without particle does not.
        text = "LE FLOC'H Yann, né à Brest\nDE LA FONTAINE Jean\nRAPPORT DE DUPONT Marie\nAvec Le Bihan";
        e = read(text, {{"LE", "ORGANIZATION", 0.55}, {"FLOC'H", "FAMILY_NAME", 0.62}, {"Yann", "GIVEN_NAME", 0.85},
                        {"LA FONTAINE", "FAMILY_NAME", 0.95}, {"Jean", "GIVEN_NAME", 0.99},
                        {"DUPONT", "FAMILY_NAME", 0.95}, {"Marie", "GIVEN_NAME", 0.99},
                        {"Bihan", "FAMILY_NAME", 0.9}});
        DLIB_TEST(found(e, text, pii_type::person, "LE FLOC'H Yann"));
        DLIB_TEST(found(e, text, pii_type::person, "DE LA FONTAINE Jean"));
        DLIB_TEST(found(e, text, pii_type::person, "DUPONT Marie"));
        DLIB_TEST(found(e, text, pii_type::person, "Le Bihan"));
        // A first name and the family name read as an address or an organisation make one person.
        text = "Contact : Yann Le Floc'h, 29760 Penmarc'h.";
        e = read(text, {{"Yann", "GIVEN_NAME", 0.99}, {"Le Floc'h", "STREET_ADDRESS", 0.77},
                        {"29760", "POSTAL_CODE", 0.99}, {"Penmarc'h", "CITY", 0.99}});
        DLIB_TEST(found(e, text, pii_type::person, "Yann Le Floc'h"));
        DLIB_TEST(found(e, text, pii_type::location, "29760 Penmarc'h"));
        text = "Yann Le Floc'h et Anne Le Bec'h sont arrivés.";
        e = read(text, {{"Yann", "GIVEN_NAME", 0.92}, {"Le Floc'", "ORGANIZATION", 0.71},
                        {"Anne", "GIVEN_NAME", 0.99}, {"Le", "FAMILY_NAME", 0.6}});
        DLIB_TEST(found(e, text, pii_type::person, "Yann Le Floc'h"));
        DLIB_TEST(found(e, text, pii_type::person, "Anne Le Bec'h"));
        // A civility or a name label keeps a person that a recognizer reads as an organisation.
        text = "Nom : LE FLOC'H\nSignature de Mme Le Floc'h\nNom : Garage Le Floc'h";
        e = read(text, {{"LE FLOC'H", "ORGANIZATION", 0.83}, {"Le Floc'h", "ORGANIZATION", 0.92},
                        {"Garage Le Floc'h", "ORGANIZATION", 0.9}});
        DLIB_TEST(found(e, text, pii_type::person, "LE FLOC'H"));
        DLIB_TEST(found(e, text, pii_type::person, "Le Floc'h"));
        DLIB_TEST(found(e, text, pii_type::organization, "Garage Le Floc'h"));
        // Another label may announce an organisation: the recognizer keeps its type.
        text = "Expéditeur : FENYS, 91 rue de Brest, 29000 Quimper";
        e = read(text, {{"FENYS", "ORGANIZATION", 0.9}});
        DLIB_TEST(found(e, text, pii_type::organization, "FENYS"));
        // An organisation known by its own name stays one.
        text = "Le colis de Paul La Poste l'a livré.";
        e = read(text, {{"Paul", "GIVEN_NAME", 0.95}, {"La Poste", "ORGANIZATION", 0.9}});
        DLIB_TEST(found(e, text, pii_type::person, "Paul"));
        DLIB_TEST(found(e, text, pii_type::organization, "La Poste"));
    }

    void test_partial_dates()
    {
        // A month and its year, a year that dates something: dates, which move with the others.
        pii_detector detector;
        string text = "Hypertension découverte en mars 2022, lombalgie en 2023, tabac arrêté depuis 2021. "
                      "En baisse depuis octobre 2025 (10/2025). Axes : 03/2022 09/2023. Née en 1984, "
                      "suivie de 2015 à 2019, mi-2021, the class of 2015, born in March, 1990.";
        std::vector<pii_entity> e = detector.detect(text);
        for (const char* date : { "mars 2022", "2023", "2021", "octobre 2025", "10/2025", "03/2022", "09/2023",
                                  "1984", "2015", "2019", "March, 1990" })
            DLIB_TEST_MSG(found(e, text, pii_type::date, date), date);
        DLIB_TEST(found(e, text, pii_type::date, "2021") && covered(e, text, "2015 à") &&
                  covered(e, text, "2021, the"));

        // A CV line, alone or as a range before a colon.
        text = "Expérience\n2015 – 2019 : technicien\n2012 : BTS\nLicence 2019-2020";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::date, "2015") && found(e, text, pii_type::date, "2019") &&
                  found(e, text, pii_type::date, "2012"));
        for (const pii_entity& x : e)
            DLIB_TEST(!(x.type == pii_type::date && x.begin >= text.find("2019-2020")));

        // Capital ordinals, a range in any order, a diploma session, a date after a phone cue.
        text = "EXTRACTION DU 1ER FÉVRIER 2025\n2022–2005 : conducteur\nobtenu à la session de 2011. "
               "Suite à notre entretien téléphonique du 16-02-2027, voici l'offre.";
        e = detector.detect(text);
        for (const char* date : { "1ER FÉVRIER 2025", "2022", "2005", "2011", "16-02-2027" })
            DLIB_TEST_MSG(found(e, text, pii_type::date, date), date);

        // A reference that carries a year is protected without its cue.
        text = "Référence\nINC-2026-0147\nNorme ISO 27001:2022, version 3.2-2024.";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "INC-2026-0147"));
        DLIB_TEST(!covered(e, text, "27001") && !covered(e, text, "3.2-2024"));

        // Numbers, references and quantities are no dates; the date of a street names it.
        text = "Tiré en 2000 exemplaires, dossier n° 03/2022, version 10.2025, 2001:db8::1, 1500 € en 2000 euros, "
               "1-29-17 Gion, Tokyo.";
        e = detector.detect(text);
        for (const pii_entity& x : e)
            DLIB_TEST_MSG(x.type != pii_type::date, text.substr(x.begin, x.end - x.begin));
        text = "Il habite 182 rue du 18 Juin 1940 à Rodez. Elle est née à Quimper (29) en 1984, à Brest (29200).";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::address, "182 rue du 18 Juin 1940"));
        DLIB_TEST(found(e, text, pii_type::location, "Quimper (29)"));
        DLIB_TEST(found(e, text, pii_type::location, "Brest (29200)"));

        // Round trip through a pseudonymizer: the protected text keeps no date in clear.
        const pseudonymization_key key = pseudonymization_key::from_bytes(string(32, '\x17'));
        const pseudonymizer p(key);
        text = "Découverte en mars 2022, suivie depuis 03/2022 et en 2023.";
        e = detector.detect(text);
        const string protected_text = p.apply(text, e);
        DLIB_TEST(protected_text.find("mars 2022") == string::npos && protected_text.find("03/2022") == string::npos &&
                  protected_text.find("en 2023") == string::npos);
        DLIB_TEST(p.restore(protected_text) == text);
    }

    void test_attribute_coverage()
    {
        pii_detector detector;
        auto entities = [&](const string& text) {
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };
        // Civilities before a labelled name, glued to it, abbreviated, and initials after a title.
        string text = "Titulaire : Mme VAILLANT Danielle\nNom : M. et Mme MARTIN\nTitle: Mrs.\n"
                      "Destinataire : MonsieurVERDIER Pierre-Henri\nMédecin : Dr GERVAIS François";
        std::vector<pii_entity> e = entities(text);
        DLIB_TEST(found(e, text, pii_type::title, "Mme") && found(e, text, pii_type::person, "VAILLANT Danielle"));
        DLIB_TEST(found(e, text, pii_type::title, "M.") && found(e, text, pii_type::person, "MARTIN"));
        DLIB_TEST(found(e, text, pii_type::title, "Mrs."));
        DLIB_TEST(found(e, text, pii_type::title, "Monsieur") &&
                  found(e, text, pii_type::person, "VERDIER Pierre-Henri"));
        DLIB_TEST(found(e, text, pii_type::person, "GERVAIS François") && !covered(e, text, "Dr "));
        text = "Ms M. Cook signed. Je soussignée MmeLamia Chaoui et Mme J.-M. BROCHARD, avec M. de La Guen, "
               "Madame DE LA BRETON et Monsieur d'Aubin. Monsieur le Procureur, Madame la Présidente.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::title, "Ms") && found(e, text, pii_type::person, "M. Cook"));
        DLIB_TEST(found(e, text, pii_type::person, "Lamia Chaoui"));
        DLIB_TEST(found(e, text, pii_type::person, "J.-M. BROCHARD"));
        DLIB_TEST(found(e, text, pii_type::person, "de La Guen"));
        DLIB_TEST(found(e, text, pii_type::person, "DE LA BRETON"));
        DLIB_TEST(found(e, text, pii_type::person, "d'Aubin"));
        DLIB_TEST(!covered(e, text, "Procureur") && !covered(e, text, "Présidente"));

        // Tables written as text: header typing, and the cells of a row that holds a name.
        text = "| Nom | Prénom | Sexe | Nationalité |\n|---|---|---|---|\n| MASOL | Vivien | M | française |";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::person, "MASOL") && found(e, text, pii_type::person, "Vivien"));
        DLIB_TEST(found(e, text, pii_type::gender, "M") && found(e, text, pii_type::nationality, "française"));
        text = "Civilité;Nom;Prénom;Sexe;Nationalité\nMme;CARRÉ;Stéphanie;F;portugaise";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::title, "Mme") && found(e, text, pii_type::gender, "F"));
        DLIB_TEST(found(e, text, pii_type::nationality, "portugaise"));
        text = "first_name,last_name,email,city\nMorag,Reid,mreid@proton.me,Auckland\nid;nom;ville\n"
               "4120780;ROUPAS;N/A";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::location, "Auckland"));
        DLIB_TEST(found(e, text, pii_type::identifier, "4120780") && !covered(e, text, "N/A"));
        text = "Liste du personnel\nPEREIRA Awa – F – belge – agent d'accueil\nJENKINS Claire, F, British, nurse";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::gender, "F") && found(e, text, pii_type::nationality, "belge"));
        DLIB_TEST(found(e, text, pii_type::nationality, "British"));

        // Nationalities and sexes announced by their context.
        text = "Jennifer Nelson, an 82 year old American woman. Mme Évelyne Oliveida, française de 73 ans. "
               "Il est titulaire d'un passeport algérien et de votre Bulgarian identity card. Mr Louis Patel, "
               "Irish, currently renting. Good morning, sex m.\nVous êtes : masculin";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::nationality, "American"));
        DLIB_TEST(found(e, text, pii_type::nationality, "française"));
        DLIB_TEST(found(e, text, pii_type::nationality, "algérien"));
        DLIB_TEST(found(e, text, pii_type::nationality, "Bulgarian"));
        DLIB_TEST(found(e, text, pii_type::nationality, "Irish"));
        DLIB_TEST(found(e, text, pii_type::gender, "m") && found(e, text, pii_type::gender, "masculin"));
        text = "Née le : 26-06-1994\nà : Carnac\nContact : Séverin au 06 94 19 67 86";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::location, "Carnac"));
        DLIB_TEST(found(e, text, pii_type::person, "Séverin"));

        // Companies.
        text = "Contrat avec SAS Iroise et SELARL Martin, puis Zorvane Finance, Williams & Sons, Martin et Fils "
               "SARL et Étude Le Calvez notaires. LA SOCIÉTÉ CALIXE COURTAGE, SIRET 533 244 521 96629. Bail entre "
               "Groupe Iroise et Mme Solène Moreno. Envoyer à Le Saux SCI.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "SAS Iroise"));
        DLIB_TEST(found(e, text, pii_type::organization, "SELARL Martin"));
        DLIB_TEST(found(e, text, pii_type::organization, "Zorvane Finance"));
        DLIB_TEST(found(e, text, pii_type::organization, "Williams & Sons"));
        DLIB_TEST(found(e, text, pii_type::organization, "Martin et Fils SARL"));
        DLIB_TEST(found(e, text, pii_type::organization, "Étude Le Calvez notaires"));
        DLIB_TEST(found(e, text, pii_type::organization, "CALIXE COURTAGE"));
        DLIB_TEST(found(e, text, pii_type::organization, "Groupe Iroise"));
        DLIB_TEST(found(e, text, pii_type::person, "Solène Moreno"));
        DLIB_TEST(found(e, text, pii_type::organization, "Le Saux SCI"));
        text = "La Direction financière et le Conseil Scientifique ont rendu leur avis. The Steering Group met "
               "over the Bank Holiday. La région Centre-Val de Loire. THE OFFICE WILL BE CLOSED. Le Groupe de "
               "Travail se réunit.";
        e = entities(text);
        DLIB_TEST(!covered(e, text, "Direction financière") && !covered(e, text, "Conseil Scientifique"));
        DLIB_TEST(!covered(e, text, "Steering Group") && !covered(e, text, "Bank Holiday"));
        DLIB_TEST(!covered(e, text, "Centre-Val") && !covered(e, text, "OFFICE") && !covered(e, text, "Travail"));

        // Public bodies, their branches and their places.
        text = "Délivré par la Sous-préfecture de Villejuif, la CPAM de L'Isle, la Brigade de gendarmerie de "
               "Montauban, l'Agence France Travail d'Arras, la Ville de Vannes, la DGFiP et Rennes Métropole. "
               "Né dans la ville de Rennes. L'AGENCE DE VILLEBŒUF.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "Sous-préfecture de Villejuif"));
        DLIB_TEST(found(e, text, pii_type::organization, "CPAM de L'Isle"));
        DLIB_TEST(found(e, text, pii_type::organization, "Brigade de gendarmerie de Montauban"));
        DLIB_TEST(found(e, text, pii_type::organization, "Agence France Travail d'Arras"));
        DLIB_TEST(found(e, text, pii_type::organization, "Ville de Vannes"));
        DLIB_TEST(found(e, text, pii_type::organization, "DGFiP"));
        DLIB_TEST(found(e, text, pii_type::organization, "Rennes Métropole"));
        bool town_place = true;
        for (const pii_entity& x : e)
            if (x.type == pii_type::organization && text.substr(x.begin, x.end - x.begin).find("ville de") == 0)
                town_place = false;
        DLIB_TEST(town_place);
        DLIB_TEST(!covered(e, text, "AGENCE"));
        text = "Approved by the East Stanley Jobcentre Plus, the East Ellham Magistrates' Court, St Mary's "
               "Hospital and the Town of Vancouver, not by the Employment Tribunal. She rents 19 Clover Court.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::organization, "East Stanley Jobcentre Plus"));
        DLIB_TEST(found(e, text, pii_type::organization, "East Ellham Magistrates' Court"));
        DLIB_TEST(found(e, text, pii_type::organization, "St Mary's Hospital"));
        DLIB_TEST(found(e, text, pii_type::organization, "Town of Vancouver"));
        DLIB_TEST(!covered(e, text, "Employment Tribunal"));
        DLIB_TEST(!found(e, text, pii_type::organization, "Clover Court"));
    }

// ----------------------------------------------------------------------------------------

    void test_foreign_postcodes()
    {
        pii_detector detector;
        auto place = [&](const string& text, const string& value, bool present) {
            const bool hit = found(detector.detect(text), text, pii_type::location, value);
            if (hit != present) dlog << LINFO << (present ? "missed " : "false positive ") << value << " in " << text;
            DLIB_TEST(hit == present);
        };
        // The postal code before the town, after the street line or before a country.
        place("Rue de Lausanne 1, 1400 Yverdon-les-Bains, Switzerland", "1400 Yverdon-les-Bains", true);
        place("ul. Długa 5\n00-238 Warszawa\nPologne", "00-238 Warszawa", true);
        place("00-238 Warszawa", "00-238 Warszawa", true);
        place("1000 Bruxelles", "1000 Bruxelles", true);
        place("Damrak 34, 1015 CJ Amsterdam", "1015 CJ Amsterdam", true);
        place("Tyršova 123, 779 00 Olomouc", "779 00 Olomouc", true);
        place("Rua Augusta 40, 1100-053 Lisboa", "1100-053 Lisboa", true);
        place("24 Grand-Rue, L-9201 Diekirch", "L-9201 Diekirch", true);
        place("Freie Strasse 98, 9000 St. Gallen", "9000 St. Gallen", true);
        place("Nádražní 80, 360 01 Karlovy Vary, République tchèque", "360 01 Karlovy Vary", true);
        // The postal code after the town.
        place("38 Herzl Street, Tel Aviv 6688116, Israël", "Tel Aviv 6688116", true);
        place("10 Oxford Road\nEast London 5201\nSouth Africa", "East London 5201", true);
        place("2-11-3 Meguro, Tokyo 153-0063", "Tokyo 153-0063", true);
        place("130 Dame Street, Dublin D02 X285", "Dublin D02 X285", true);
        place("164 Barrington Street, Gatineau (Québec) J8X 3X7", "J8X 3X7", true);
        place("164 Barrington Street, Gatineau (Québec) J8X 3X7", "Gatineau", true);
        place("Rua Bahia, 151, Rio de Janeiro - RJ, 22021-001", "Rio de Janeiro", true);
        place("Rua Bahia, 151, Rio de Janeiro - RJ, 22021-001", "22021-001", true);
        place("9 Bourke Street, Newcastle NSW 2300", "2300", true);
        // Without a street before or a country after, or with a year, a number stays a number.
        place("Toronto M4W 1A8", "M4W 1A8", false);
        place("12 rue des Lilas, BP 5405, 35000 Rennes", "BP 5405", false);
        place("Véhicule : 2 portes, 2018 Toyota Corolla", "2018 Toyota Corolla", false);
        place("Inscrits le 12 mars, 1500 Participants au total", "1500 Participants", false);
    }

    void test_foreign_identifiers()
    {
        pii_detector detector;
        auto expect = [&](const string& text, const string& value, bool present) {
            const bool hit = found(detector.detect(text), text, pii_type::identifier, value);
            if (hit != present) dlog << LINFO << (present ? "missed " : "false positive ") << value << " in " << text;
            DLIB_TEST(hit == present);
        };
        expect("SVNR: 8574911862", "8574911862", true);
        expect("Rodné číslo : 790424/0025", "790424/0025", true);
        expect("Rodné číslo : 790424/0021", "790424/0021", false);
        expect("CPR-nr.: 160996-8197", "160996-8197", true);
        expect("HETU 200884+602E", "200884+602E", true);
        expect("personnummer 520711-8679", "520711-8679", true);
        expect("fødselsnummer 06095129283", "06095129283", true);
        expect("PPSN 2572286DB", "2572286DB", true);
        expect("CNP 1580806427203", "1580806427203", true);
        expect("EGN 4702217404", "4702217404", true);
        expect("OIB: 72515816758", "72515816758", true);
        expect("isikukood 52004025897", "52004025897", true);
        expect("AMKA 24094834215", "24094834215", true);
        expect("NHS number 887 166 1850", "887 166 1850", true);
        expect("SIN 405 628 785", "405 628 785", true);
        expect("CPF 218.374.823-06", "218.374.823-06", true);
        expect("Aadhaar 5972 5156 8660", "5972 5156 8660", true);
        expect("CURP PUCD781104HDFRMT02", "PUCD781104HDFRMT02", true);
        expect("RUT 10.514.579-9", "10.514.579-9", true);
        expect("CUIT 23-71020739-9", "23-71020739-9", true);
        expect("T.C. Kimlik No: 89761191158", "89761191158", true);
        expect("South African ID number 8308212992191", "8308212992191", true);
        expect("RRN 010218-4438030", "010218-4438030", true);
        expect("kennitala 010293-2940", "010293-2940", true);
        expect("N° CIN : AB123456", "AB123456", true);
        expect("Numéro RAMQ : TREM 8001 0112", "TREM 8001 0112", true);
        // A cue is a whole word: "itinéraire" is no ITIN, "single" no SIN.
        expect("L'itinéraire 948727063 passe par Rennes.", "948727063", false);
        expect("Le single 405628785 est sorti.", "405628785", false);
        expect("Lot 520711-8679 livré.", "520711-8679", false);
    }

// ----------------------------------------------------------------------------------------

    void test_detector_boundaries()
    {
        pii_detector detector;
        auto entities = [&](const string& text) {
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };
        // Public links are protected; profiles, credentials, IP hosts and e-mails are not.
        string text = "Voir https://www.legifrance.gouv.fr/jorf/2024/05/12/article et https://x.com/jdupont "
                      "ou https://box.com/file/0612345678 et http://192.168.1.10/admin";
        std::vector<pii_entity> e = entities(text);
        DLIB_TEST(found(e, text, pii_type::url, "https://x.com/jdupont"));
        DLIB_TEST(found(e, text, pii_type::url, "http://192.168.1.10/admin"));
        DLIB_TEST(!covered(e, text, "2024/05/12") && !covered(e, text, "0612345678"));
        text = "Lien https://example.com/unsubscribe?email=jean.dupont@gmail.com envoyé.";
        DLIB_TEST(covered(entities(text), text, "jean.dupont@gmail.com"));

        // Addresses of several forms.
        text = "Adresse ::ffff:192.168.1.1 et contact élodie.dupont@société.fr";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::ipv6, "::ffff:192.168.1.1"));
        DLIB_TEST(found(e, text, pii_type::email, "élodie.dupont@société.fr"));

        // Phone numbers stop at a group boundary: a postal code after them stays a place.
        text = "Tel : 02 99 12 34 56 35000 Rennes, ou le +49 6221 12345.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::phone, "02 99 12 34 56"));
        DLIB_TEST(found(e, text, pii_type::location, "35000 Rennes"));
        DLIB_TEST(found(e, text, pii_type::phone, "+49 6221 12345"));
        text = "Tel: 612 345 678 35000 Madrid";
        DLIB_TEST(found(entities(text), text, pii_type::phone, "612 345 678"));
        text = "RIB : 30004 00550 00012345678 40";
        DLIB_TEST(found(entities(text), text, pii_type::identifier, "30004 00550 00012345678 40"));
        // A card number followed by its expiry date; a longer run is no card.
        text = "Carte 4970 1012 3456 7893 12/27";
        DLIB_TEST(found(entities(text), text, pii_type::payment_card, "4970 1012 3456 7893"));
        text = "Compte 4970 1012 3456 7893 1234 5678";
        DLIB_TEST(!found(entities(text), text, pii_type::payment_card, "4970 1012 3456 7893"));

        // The account name of a path ends with its segment, spaces allowed inside.
        text = "Fichier C:\\Users\\Jean Dupont\\Documents\\cv.docx et /home/jdupont is mine.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "Jean Dupont"));
        DLIB_TEST(found(e, text, pii_type::identifier, "jdupont"));
        DLIB_TEST(!covered(e, text, "is mine"));

        // Labels whose number word is abbreviated.
        text = "N° allocataire : 1234567\nPhone No. : 06 12 34 56 78";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "1234567"));
        DLIB_TEST(found(e, text, pii_type::phone, "06 12 34 56 78"));

        // Names with a particle after a civility, and names with an apostrophe.
        text = "M. Le Goff et Mme De Smet ont signé, puis M. O'Connor et Mme N'Diaye.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::person, "Le Goff"));
        DLIB_TEST(found(e, text, pii_type::person, "De Smet"));
        DLIB_TEST(found(e, text, pii_type::person, "O'Connor"));
        DLIB_TEST(found(e, text, pii_type::person, "N'Diaye"));

        // Places and amounts are not persons, articles and possessives not companies.
        text = "Jean Dupont habite à Saint-Jean-de-Luz. Le prix est de 35000 Euros.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::person, "Jean Dupont"));
        DLIB_TEST(!found(e, text, pii_type::person, "Jean-de-Luz") && !covered(e, text, "Euros"));
        text = "La SARL a été dissoute. IL A PERDU SA VOITURE. Dupont Conseil SAS a repris l'activité.";
        e = entities(text);
        DLIB_TEST(!covered(e, text, "La SARL") && !covered(e, text, "PERDU"));
        DLIB_TEST(found(e, text, pii_type::organization, "Dupont Conseil SAS"));
        text = "Inscrite à l'Université de Rennes depuis 2020.";
        DLIB_TEST(found(entities(text), text, pii_type::organization, "Université de Rennes"));

        // Terms with digits; a number is a whole token.
        detector.add_term(pii_type::identifier, "EMP-4521");
        detector.add_term(pii_type::identifier, "12345");
        text = "Badge EMP-4521, code 12345, pas A12345 ni EMP-4522.";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "EMP-4521"));
        DLIB_TEST(found(e, text, pii_type::identifier, "12345"));
        DLIB_TEST(!covered(e, text, "A12345") && !covered(e, text, "EMP-4522"));

        // Brackets that only look like a tag are read like any text.
        text = "Voir [[NOTE: appeler Jean Dupont]] demain.";
        DLIB_TEST(found(entities(text), text, pii_type::person, "Jean Dupont"));

        // Patterns: precedence over the scanners, and matches across window boundaries.
        pii_detector with_patterns;
        pii_pattern account;
        account.type = pii_type::identifier;
        account.regex = "ACC[0-9]{6}";
        with_patterns.add_pattern(account);
        pii_pattern code;
        code.type = pii_type::identifier;
        code.regex = "REF 0[0-9]( [0-9]{2}){4}";
        with_patterns.add_pattern(code);
        text = string(1530, 'x') + " ACC123456 " + string(5000, 'y') + " ACC654321 et REF 06 12 34 56 78";
        e = with_patterns.detect(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "ACC123456"));
        DLIB_TEST(found(e, text, pii_type::identifier, "ACC654321"));
        DLIB_TEST(found(e, text, pii_type::identifier, "REF 06 12 34 56 78"));
        pii_pattern luhn_card;
        luhn_card.type = pii_type::payment_card;
        luhn_card.regex = "CARD-[0-9A-Z-]+";
        luhn_card.check = "luhn";
        with_patterns.add_pattern(luhn_card);
        text = "Ref CARD-AB4970101234567893 et CARD-AB4970101234567894";
        e = with_patterns.detect(text);
        DLIB_TEST(found(e, text, pii_type::payment_card, "CARD-AB4970101234567893"));
        DLIB_TEST(!found(e, text, pii_type::payment_card, "CARD-AB4970101234567894"));

        // A recognizer span starting inside a character takes the whole character.
        pii_detector with_model;
        with_model.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            const size_t p = t.find("Élodie");
            if (p != string::npos)
            {
                pii_span s;
                s.begin = p + 1;
                s.end = p + string("Élodie").size();
                s.label = "GIVEN_NAME";
                s.confidence = 0.9;
                spans.push_back(s);
            }
            return spans;
        });
        text = "Signé : Élodie";
        with_model.begin_document();
        DLIB_TEST(found(with_model.detect(text), text, pii_type::person, "Élodie"));
    }

// ----------------------------------------------------------------------------------------

    void test_dataset()
    {
        const pii_example ex = parse_pii_example(
            "Jean Dupont, 35000 Rennes\\tTel 06\t0:4:GIVEN_NAME,5:11:FAMILY_NAME,13:18:POSTAL_CODE,19:25:CITY");
        DLIB_TEST(ex.text == "Jean Dupont, 35000 Rennes\tTel 06");
        DLIB_TEST(ex.spans.size() == 4);
        DLIB_TEST(ex.spans[1].type == pii_type::person);
        DLIB_TEST(parse_pii_example(format_pii_example(ex)).text == ex.text);
        for (const string& bad : { string("abc\t0:9:CITY"), string("abc\t0:2:CITY,1:3:CITY"), string("\xC3\xA9t\t1:3:CITY"),
                                  string("abc\t0:2:"), string("a\\qb\t") })
        {
            bool refused = false;
            try { parse_pii_example(bad); }
            catch (error&) { refused = true; }
            DLIB_TEST(refused);
        }

        std::vector<std::pair<size_t, size_t>> units = { {0, 4}, {5, 11}, {11, 12}, {13, 18}, {19, 25} };
        const std::vector<unsigned long> bio = pii_spans_to_bio(units, ex.spans);
        DLIB_TEST(bio[0] == pii_bio_class("GIVEN_NAME", false));
        DLIB_TEST(bio[2] == 0);
        const std::vector<pii_span> back = pii_bio_to_spans(units, bio);
        DLIB_TEST(back.size() == 4 && back[3].begin == 19 && back[3].end == 25 && back[3].label == "CITY");
        DLIB_TEST(!pii_bio_transition_allowed(0, pii_bio_class("CITY", true)));
        DLIB_TEST(pii_bio_transition_allowed(pii_bio_class("CITY", false), pii_bio_class("CITY", true)));
        DLIB_TEST(pii_bio_class_count() == 15);

        const std::vector<pii_span> coarse = coarsen_pii_spans(ex.text, ex.spans);
        DLIB_TEST(coarse.size() == 2 && coarse[0].label == "PERSON" && coarse[1].label == "LOCATION");
        DLIB_TEST(coarse[0].end == 11 && coarse[1].begin == 13);

        std::vector<pii_span> predicted = ex.spans;
        predicted[3].end = 24;
        predicted.pop_back();
        const pii_evaluation ev = evaluate_pii_spans({ ex.spans }, { predicted });
        DLIB_TEST(ev.overall.true_positives == 3 && ev.overall.false_negatives == 1 && ev.overall.false_positives == 0);
        // In exact mode a boundary error is both a false positive and a false negative.
        std::vector<pii_span> cut = ex.spans;
        cut[3].end = 24;
        const pii_evaluation ex_ev = evaluate_pii_spans({ ex.spans }, { cut });
        DLIB_TEST(ex_ev.overall.true_positives == 3 && ex_ev.overall.false_positives == 1 &&
                  ex_ev.overall.false_negatives == 1);
        const pii_evaluation nothing = evaluate_pii_spans({ std::vector<pii_span>() }, { std::vector<pii_span>() });
        DLIB_TEST(nothing.overall.precision() == 1 && nothing.overall.recall() == 1);

        // A text starting with '#' survives a file, and malformed lines say why.
        pii_example hash;
        hash.text = "#tag Rennes";
        pii_span city;
        city.begin = 5;
        city.end = 11;
        city.label = "CITY";
        hash.spans.push_back(city);
        DLIB_TEST(format_pii_example(hash)[0] == '\\');
        DLIB_TEST(parse_pii_example(format_pii_example(hash)).text == hash.text);
        for (const string& bad : { string("a\tb\t0:1:CITY"), string("abc\t0:2:NOT_A_LABEL"), string("a\\nb\t0:3:CITY") })
        {
            string message;
            try { parse_pii_example(bad); }
            catch (error& err) { message = err.what(); }
            DLIB_TEST_MSG(message.find("dataset line") != string::npos, bad);
        }
        const string file = "privacy_test_dataset.tsv";
        {
            std::ofstream out(file, std::ios::binary);
            out << "\xEF\xBB\xBF" << format_pii_example(hash) << "\n# comment\n" << format_pii_example(ex) << "\n";
        }
        const std::vector<pii_example> loaded = load_pii_dataset(file);
        std::remove(file.c_str());
        DLIB_TEST(loaded.size() == 2 && loaded[0].text == hash.text && loaded[1].spans.size() == 4);

        // Units labelled from unsorted spans; parts of a name joined by a no-break space.
        std::vector<pii_span> reversed(ex.spans.rbegin(), ex.spans.rend());
        DLIB_TEST(pii_spans_to_bio(units, reversed) == bio);
        pii_example nbsp;
        nbsp.text = "Jean\xC2\xA0" "Dupont";
        pii_span g, f;
        g.begin = 0; g.end = 4; g.label = "GIVEN_NAME";
        f.begin = 6; f.end = 12; f.label = "FAMILY_NAME";
        DLIB_TEST(coarsen_pii_spans(nbsp.text, { g, f }).size() == 1);
        std::vector<pii_span> shifted = ex.spans;
        shifted[3].end = 24;
        const pii_evaluation ov = evaluate_pii_spans({ ex.spans }, { shifted }, span_match::overlap);
        DLIB_TEST(ov.overall.true_positives == 4);
    }

// ----------------------------------------------------------------------------------------

    // A small corpus made here, so that the learned recognizers are tested without a dataset file.
    std::vector<pii_example> make_corpus(dlib::rand& rnd, size_t count, bool held_out)
    {
        const char* given_seen[] = { "Jean", "Marie", "Luc", "Anne", "Paul", "Claire", "Hugo", "Léa", "Yves", "Rose" };
        const char* family_seen[] = { "Martin", "Durand", "Le Goff", "Kerjean", "Moreau", "Petit", "Garnier", "Roux" };
        const char* city_seen[] = { "Rennes", "Brest", "Vannes", "Quimper", "Saint-Malo", "Lorient" };
        const char* given_new[] = { "Gwenaël", "Soizic", "Brieuc", "Nolwenn" };
        const char* family_new[] = { "Quéméner", "Abgrall", "Le Bihan", "Cloarec" };
        const char* city_new[] = { "Lannion", "Dinan", "Fougères", "Vitré" };
        std::vector<pii_example> out;
        for (size_t i = 0; i < count; ++i)
        {
            pii_example ex;
            auto lit = [&](const string& s) { ex.text += s; };
            auto ent = [&](const string& s, const string& label) {
                pii_span sp;
                sp.begin = ex.text.size();
                ex.text += s;
                sp.end = ex.text.size();
                sp.label = label;
                sp.type = pii_type_from_label(label);
                ex.spans.push_back(sp);
            };
            const string g = held_out ? given_new[rnd.get_random_32bit_number() % 4]
                                      : given_seen[rnd.get_random_32bit_number() % 10];
            const string f = held_out ? family_new[rnd.get_random_32bit_number() % 4]
                                      : family_seen[rnd.get_random_32bit_number() % 8];
            const string c = held_out ? city_new[rnd.get_random_32bit_number() % 4]
                                      : city_seen[rnd.get_random_32bit_number() % 6];
            switch (rnd.get_random_32bit_number() % 4)
            {
                case 0: lit("Je m'appelle "); ent(g, "GIVEN_NAME"); lit(" "); ent(f, "FAMILY_NAME");
                        lit(" et j'habite à "); ent(c, "CITY"); lit("."); break;
                case 1: lit("Nom : "); ent(f, "FAMILY_NAME"); lit("\nPrénom : "); ent(g, "GIVEN_NAME");
                        lit("\nVille : "); ent(c, "CITY"); break;
                case 2: lit("Le dossier de "); ent(g, "GIVEN_NAME"); lit(" "); ent(f, "FAMILY_NAME");
                        lit(" a été transmis au service de "); ent(c, "CITY"); lit(" hier."); break;
                default: lit("Rendez-vous avec M. "); ent(f, "FAMILY_NAME"); lit(" mardi à "); ent(c, "CITY");
                         lit(", merci."); break;
            }
            out.push_back(ex);
        }
        return out;
    }

    double fine_f1(const std::vector<pii_example>& data, const std::function<std::vector<pii_span>(const string&)>& f)
    {
        std::vector<std::vector<pii_span>> truth, predicted;
        for (const pii_example& ex : data)
        {
            truth.push_back(ex.spans);
            predicted.push_back(f(ex.text));
        }
        return evaluate_pii_spans(truth, predicted).overall.f1();
    }

    void test_sequence_tagger()
    {
        dlib::rand rnd(7);
        const std::vector<pii_example> train = make_corpus(rnd, 80, false);
        const std::vector<pii_example> unseen = make_corpus(rnd, 30, true);
        const std::vector<pii_token> tokens = tokenize_for_tagging("Jean-Pierre d'Arc, 35000");
        DLIB_TEST(tokens.size() == 8);
        // Glued words are split where the case or the kind of character changes.
        auto pieces = [](const string& text) {
            std::vector<string> out;
            for (const pii_token& t : tokenize_for_tagging(text)) out.push_back(text.substr(t.begin, t.end - t.begin));
            return out;
        };
        DLIB_TEST(pieces("DupontMarie") == std::vector<string>({ "Dupont", "Marie" }));
        DLIB_TEST(pieces("DUPONTMarie") == std::vector<string>({ "DUPONT", "Marie" }));
        DLIB_TEST(pieces("75008Paris") == std::vector<string>({ "75008", "Paris" }));
        DLIB_TEST(pieces("J.Dupont") == std::vector<string>({ "J", ".", "Dupont" }));
        DLIB_TEST(pieces("olga.horvath @jdupont35") == std::vector<string>({ "olga.horvath", "@jdupont35" }));

        pii_sequence_tagger tagger;
        pii_tagger_training_options options;
        options.hash_bits = 16;
        options.max_iterations = 100;
        tagger.train(train, options);
        DLIB_TEST(tagger.is_trained());
        const double seen_f1 = fine_f1(train, [&](const string& t) { return tagger.find(t); });
        const double unseen_f1 = fine_f1(unseen, [&](const string& t) { return tagger.find(t); });
        dlog << LINFO << "tagger f1 on training " << seen_f1 << ", on unseen names " << unseen_f1;
        DLIB_TEST(seen_f1 > 0.95);
        DLIB_TEST(unseen_f1 > 0.6);
        DLIB_TEST(tagger.find("").empty());

        std::ostringstream sout;
        serialize(tagger, sout);
        std::istringstream sin(sout.str());
        pii_sequence_tagger again;
        deserialize(again, sin);
        // The weights are saved in single precision: the same spans, confidences within 1e-5.
        const string probe = unseen[0].text;
        const std::vector<pii_span> a = tagger.find(probe), b = again.find(probe);
        DLIB_TEST(a.size() == b.size());
        for (size_t i = 0; i < a.size() && i < b.size(); ++i)
            DLIB_TEST(a[i].begin == b[i].begin && a[i].end == b[i].end && a[i].label == b[i].label &&
                      std::abs(a[i].confidence - b[i].confidence) < 1e-5);
        std::ostringstream sout2;
        serialize(again, sout2);
        DLIB_TEST(sout2.str() == sout.str());
        for (const pii_span& s : a) DLIB_TEST(0 <= s.confidence && s.confidence <= 1);
        DLIB_TEST(again.get_feature_extractor().hash_bits() == 16);
        // A span never crosses a line break.
        for (const pii_span& s : tagger.find("Nom : Martin\nDurand"))
            DLIB_TEST(string("Nom : Martin\nDurand").substr(s.begin, s.end - s.begin).find('\n') == string::npos);
    }

    void test_span_model()
    {
        // Span limits move to word limits, except at a word glued to another and never across
        // a neighbouring span.
        auto snapped = [](const string& text, const std::vector<std::pair<size_t, size_t>>& bounds) {
            std::vector<pii_span> spans;
            for (const auto& l : bounds)
            {
                pii_span s;
                s.begin = l.first;
                s.end = l.second;
                s.label = "CITY";
                spans.push_back(s);
            }
            privacy_impl::snap_to_words(text, spans);
            std::vector<std::pair<size_t, size_t>> out;
            for (const pii_span& s : spans) out.push_back(std::make_pair(s.begin, s.end));
            return out;
        };
        typedef std::vector<std::pair<size_t, size_t>> limits;
        DLIB_TEST(snapped("Clinique Saint-Vincent", { { 0, 21 } }) == limits({ { 0, 22 } }));
        DLIB_TEST(snapped("Mme Kerbrat", { { 6, 11 } }) == limits({ { 4, 11 } }));
        DLIB_TEST(snapped("Hélène Martin", { { 0, 3 } }) == limits({ { 0, 8 } }));
        DLIB_TEST(snapped("DUPONTMarie", { { 0, 6 }, { 6, 11 } }) == limits({ { 0, 6 }, { 6, 11 } }));
        DLIB_TEST(snapped("DupontMarie", { { 0, 6 } }) == limits({ { 0, 6 } }));
        DLIB_TEST(snapped("abcdef", { { 0, 2 }, { 2, 6 } }) == limits({ { 0, 2 }, { 2, 6 } }));
        DLIB_TEST(snapped("au Mans, 72000", { { 3, 7 }, { 9, 14 } }) == limits({ { 3, 7 }, { 9, 14 } }));

        // Spaces and no-break spaces are trimmed in any order.
        {
            const string text = "a \xC2\xA0 Rennes \xC2\xA0";
            pii_span s;
            s.begin = 1;
            s.end = text.size();
            s.label = "CITY";
            std::vector<pii_span> spans = { s };
            privacy_impl::tidy_spans(text, spans);
            DLIB_TEST(spans.size() == 1 && text.substr(spans[0].begin, spans[0].end - spans[0].begin) == "Rennes");
        }
        // Decoding keeps the BIO grammar and line breaks outside the spans.
        {
            const long c = pii_span_model_tiny::num_classes;
            const unsigned long city_b = pii_bio_class("CITY", false), city_i = pii_bio_class("CITY", true);
            std::vector<float> logp(3 * c, -10.0f);
            logp[0 * c + city_i] = 0;       // an I class cannot open a span
            logp[1 * c + city_i] = 0;       // a line break stays outside
            logp[2 * c + city_b] = 0;
            const std::vector<unsigned long> classes = privacy_impl::viterbi_bio(logp, 3, c, { false, true, false });
            DLIB_TEST(classes[0] != city_i && classes[1] == 0 && classes[2] == city_b);
        }

        dlib::rand rnd(11);
        const std::vector<pii_example> train = make_corpus(rnd, 24, false);
        pii_span_model_tiny model;
        pii_span_model_training_options options;
        options.steps = 400;
        options.batch_size = 16;
        options.window_lengths = { 32, 64 };
        options.window_weights = { 1, 1 };
        options.warmup_steps = 40;
        options.learning_rate = 3e-3;
        model.train(train, std::vector<pii_example>(), options);
        DLIB_TEST(model.is_trained());
        DLIB_TEST(model.parameter_count() > 50000);
        DLIB_TEST(model.window() == 64);
        const double f1 = fine_f1(train, [&](const string& t) { return model.find(t); });
        dlog << LINFO << "span model f1 on training " << f1;
        DLIB_TEST(f1 > 0.8);
        DLIB_TEST(model.find("").empty());

        // Offsets fall on character boundaries, also through the windows of a long text, and
        // the spans of the last examples are found as well as those of the first.
        model.set_window(32);
        string longer;
        for (int k = 0; k < 6; ++k) longer += train[k].text + "\n";
        const std::vector<pii_span> long_spans = model.find(longer);
        for (const pii_span& s : long_spans)
        {
            DLIB_TEST(s.begin < s.end && s.end <= longer.size());
            DLIB_TEST((static_cast<unsigned char>(longer[s.begin]) & 0xC0) != 0x80);
            DLIB_TEST(s.end == longer.size() || (static_cast<unsigned char>(longer[s.end]) & 0xC0) != 0x80);
            DLIB_TEST(0 <= s.confidence && s.confidence <= 1);
        }
        DLIB_TEST(!long_spans.empty() && long_spans.back().begin > longer.size() / 2);
        model.set_window(64);

        // A soft hyphen inside a name changes nothing, and offsets point into the original text.
        {
            const string plain = train[0].text;
            const pii_span first = model.find(plain).front();
            const size_t middle = first.begin + 2;
            const string marked = plain.substr(0, middle) + "\xC2\xAD" + plain.substr(middle);
            const std::vector<pii_span> spans = model.find(marked);
            DLIB_TEST(!spans.empty() && spans[0].begin == first.begin && spans[0].end == first.end + 2);
        }

        std::ostringstream sout;
        serialize(model, sout);
        std::istringstream sin(sout.str());
        pii_span_model_tiny again;
        deserialize(again, sin);
        const std::vector<pii_span> a = model.find(train[0].text), b = again.find(train[0].text);
        DLIB_TEST(a.size() == b.size());
        for (size_t i = 0; i < a.size() && i < b.size(); ++i)
            DLIB_TEST(a[i].begin == b[i].begin && a[i].end == b[i].end && a[i].label == b[i].label);

        // A complementary training starts from the current weights: a few steps at a low rate
        // keep what the model knows, which as few steps from new weights cannot reach.
        pii_span_model_training_options more = options;
        more.steps = 40;
        more.warmup_steps = 4;
        more.learning_rate = 3e-4;
        more.optimizer = pii_span_optimizer::adam_atan2;
        more.continue_training = true;
        again.train(train, std::vector<pii_example>(), more);
        DLIB_TEST(again.is_trained());
        const double continued = fine_f1(train, [&](const string& t) { return again.find(t); });
        pii_span_model_tiny fresh;
        fresh.train(train, std::vector<pii_example>(), more);
        const double restarted = fine_f1(train, [&](const string& t) { return fresh.find(t); });
        dlog << LINFO << "span model f1 after a complementary training " << continued << ", from new weights "
             << restarted;
        DLIB_TEST(continued > 0.8 && continued > restarted + 0.2);

        // A model of another size is refused with a clear message.
        std::istringstream sin2(sout.str());
        pii_span_model_small small;
        string message;
        try { deserialize(small, sin2); }
        catch (serialization_error& err) { message = err.what(); }
        DLIB_TEST(message.find("embedding dimension 64") != string::npos);
    }

// ----------------------------------------------------------------------------------------

    class privacy_tester : public tester
    {
    public:
        privacy_tester (
        ) :
            tester ("test_privacy",
                    "Runs tests on the cryptographic, archive, document, pseudonymization and recognition components.")
        {}

        void perform_test (
        )
        {
            print_spinner();
            test_crypto();
            print_spinner();
            test_deflate_and_zip();
            print_spinner();
            test_pseudonymizer();
            print_spinner();
            test_detector();
            print_spinner();
            test_text_documents();
            print_spinner();
            test_office_document();
            print_spinner();
            test_pdf_document();
            print_spinner();
            test_more_identifiers();
            print_spinner();
            test_rule_refinements();
            print_spinner();
            test_patterns_and_recognizers();
            print_spinner();
            test_attributes_and_public_bodies();
            print_spinner();
            test_known_names();
            test_learned_organizations();
            test_learned_places();
            print_spinner();
            test_partial_dates();
            print_spinner();
            test_learned_terms();
            test_name_edges();
            print_spinner();
            test_attribute_coverage();
            print_spinner();
            test_foreign_postcodes();
            print_spinner();
            test_foreign_identifiers();
            print_spinner();
            test_detector_boundaries();
            print_spinner();
            test_dataset();
            print_spinner();
            test_sequence_tagger();
            print_spinner();
            test_span_model();
        }
    } a;

}
