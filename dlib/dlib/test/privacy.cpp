// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/privacy.h>
#include <dlib/privacy/pii_sequence_tagger.h>
#include <dlib/privacy/pii_span_model.h>
#include <dlib/rand.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
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

    // Whether the two last digits of a French social security number are its key, 97 minus its
    // first 13 digits modulo 97, Corsica's 2A and 2B counting as 19 and 18.
    bool nir_key_valid(const string& nir)
    {
        string a;
        for (char c : nir) if ((c >= '0' && c <= '9') || c == 'A' || c == 'B') a.push_back(c);
        if (a.size() != 15) return false;
        if (a[6] == 'A' || a[6] == 'B')
        {
            a[6] = a[6] == 'A' ? '9' : '8';
            a[5] = '1';
        }
        long long r = 0;
        for (size_t k = 0; k < 13; ++k) r = (r * 10 + (a[k] - '0')) % 97;
        return 97 - r == (a[13] - '0') * 10 + (a[14] - '0');
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
                                                  { pii_type::payment_card, "4970 1012 3456 7893" },
                                                  { pii_type::nir, "1 85 03 75 123 456 41" },
                                                  { pii_type::nir, "2 69 05 2A 123 456 88" },
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

        // Checksums still verify on protected numbers, which keep their layout: the card its first
        // six digits, the social security number the department 2A of Corsica.
        const string card = tag_body(p.tag(pii_type::payment_card, "4970 1012 3456 7893"));
        DLIB_TEST(card.size() == 19 && card.back() != '!' && card[4] == ' ' && card[9] == ' ' && card[14] == ' ');
        DLIB_TEST(only_digits(card).substr(0, 6) == "497010" && luhn(only_digits(card)));
        DLIB_TEST(card != "4970 1012 3456 7893");
        const string nir = tag_body(p.tag(pii_type::nir, "1 85 03 75 123 456 41"));
        DLIB_TEST(nir.size() == 21 && nir.back() != '!' && nir_key_valid(nir) && nir != "1 85 03 75 123 456 41");
        const string corsica = tag_body(p.tag(pii_type::nir, "2 69 05 2A 123 456 88"));
        DLIB_TEST(corsica.size() == 21 && corsica.substr(8, 2) == "2A" && nir_key_valid(corsica));
        // A number whose check digits are wrong is no card and no social security number: it
        // takes an opaque code, which still restores it.
        DLIB_TEST(tag_body(p.tag(pii_type::payment_card, "4970 1012 3456 7899")).back() == '!');
        DLIB_TEST(tag_body(p.tag(pii_type::nir, "1 85 03 75 123 456 13")).back() == '!');
        DLIB_TEST(round_trip(p, pii_type::payment_card, "4970 1012 3456 7899"));
        DLIB_TEST(round_trip(p, pii_type::nir, "1 85 03 75 123 456 13"));
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
              "<Override PartName=\"/docProps/app.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>"
              "<Override PartName=\"/word/charts/chart1.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.drawingml.chart+xml\"/>"
              "</Types>");
        w.add("_rels/.rels",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
              "<Relationship Id=\"rId1\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
              "Target=\"word/document.xml\"/>"
              "<Relationship Id=\"rId2\" "
              "Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" "
              "Target=\"docProps/core.xml\"/>"
              "<Relationship Id=\"rId3\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" "
              "Target=\"docProps/app.xml\"/></Relationships>");
        w.add("word/_rels/document.xml.rels",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
              "<Relationship Id=\"rId5\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" "
              "Target=\"media/image1.png\"/>"
              "<Relationship Id=\"rId6\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart\" "
              "Target=\"charts/chart1.xml\"/></Relationships>");
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
            "<w:p><w:r><w:drawing><wp:inline><a:graphic><a:graphicData "
            "uri=\"http://schemas.openxmlformats.org/drawingml/2006/chart\"><c:chart "
            "xmlns:c=\"http://schemas.openxmlformats.org/drawingml/2006/chart\" r:id=\"rId6\"/>"
            "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
            "<w:p><w:r><w:t>Dupont a signé.</w:t></w:r></w:p><w:sectPr/></w:body></w:document>");
        w.add("word/media/image1.png", string("\x89PNG\r\n\x1a\n", 8) + string(64, 'x'));
        w.add("word/charts/chart1.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<c:chartSpace xmlns:c=\"http://schemas.openxmlformats.org/drawingml/2006/chart\" "
            "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\"><c:chart><c:title><c:tx><c:rich>"
            "<a:p><a:r><a:t>Ventes de Marie Curie le 12/03/2024</a:t></a:r></a:p></c:rich></c:tx></c:title>"
            "</c:chart></c:chartSpace>");
        w.add("docProps/app.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" "
            "xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
            "<HeadingPairs><vt:vector size=\"2\" baseType=\"variant\"><vt:variant><vt:lpstr>Titre</vt:lpstr>"
            "</vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs>"
            "<TitlesOfParts><vt:vector size=\"1\" baseType=\"lpstr\"><vt:lpstr>Attestation de Jean Dupont</vt:lpstr>"
            "</vt:vector></TitlesOfParts><Company>Dupont Conseil SARL</Company></Properties>");
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
        // The chart goes with its drawing, the lists of parts of the properties too.
        DLIB_TEST(!z.contains("word/charts/chart1.xml"));
        DLIB_TEST(z.read("docProps/app.xml").find("TitlesOfParts") == string::npos);
        DLIB_TEST(z.read("docProps/app.xml").find("<Company>[[ORG:") != string::npos);
        // The report counts the values of the document as written, without those of removed parts.
        size_t tags = 0;
        for (size_t i = 0; i < z.entries().size(); ++i)
        {
            const string part = z.read(i);
            for (size_t k = part.find("[[");  k != string::npos; k = part.find("[[", k + 2)) ++tags;
        }
        DLIB_TEST_MSG(tags == report.total_entities(), tags << " " << report.total_entities());

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


    // The substitutes a report lists, as a resolver gives them back.
    pseudonymizer::substitute_resolver resolver_of(const std::vector<document_substitution>& made)
    {
        std::map<string, std::vector<string>> table;
        for (const document_substitution& s : made) table[s.substitute].push_back(s.tag);
        return [table](const string& candidate) {
            const auto it = table.find(candidate);
            return it == table.end() ? std::vector<string>() : it->second;
        };
    }

    void test_untagged_values()
    {
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x37')));

        // A substitute is the body of the tag, a shifted date keeps its form.
        string tag;
        const string person = p.substitute(pii_type::person, "Jean Dupont", &tag);
        DLIB_TEST(tag == p.tag(pii_type::person, "Jean Dupont"));
        DLIB_TEST_MSG(tag.find(":" + person + "#") != string::npos, tag << " " << person);
        DLIB_TEST(person.find("Jean") == string::npos && person.find(' ') != string::npos);
        const string date = p.substitute(pii_type::date, "12/03/2024", &tag);
        DLIB_TEST_MSG(date.size() == 10 && date[2] == '/' && date[5] == '/' && date != "12/03/2024", date);
        pii_type type;
        string original;
        DLIB_TEST(p.decode(tag, type, original) && original == "12/03/2024");

        // The substitutes of a text come back through the tags recorded for them.
        std::vector<pii_entity> entities(2);
        const string text = "Contrat de Jean Dupont, le 12/03/2024";
        entities[0].begin = 11;
        entities[0].end = 22;
        entities[0].type = pii_type::person;
        entities[1].begin = 27;
        entities[1].end = 37;
        entities[1].type = pii_type::date;
        std::vector<pseudonymizer::substitution> made;
        const string protected_text = p.apply_substitutes(text, entities, &made);
        DLIB_TEST(made.size() == 2 && protected_text.find("[[") == string::npos);
        DLIB_TEST(protected_text == "Contrat de " + person + ", le " + date);
        std::vector<document_substitution> listed;
        for (const auto& s : made) listed.push_back(document_substitution{ s.substitute, s.tag });
        pseudonymizer::restore_report rr;
        DLIB_TEST(p.restore_substitutes(protected_text, resolver_of(listed), &rr) == text);
        DLIB_TEST(rr.restored == 2);
        // Without the recorded tags, nothing is given back.
        DLIB_TEST(p.restore_substitutes(protected_text, pseudonymizer::substitute_resolver()) == protected_text);

        // A file name: the underscores read as spaces and stay.
        pii_detector detector;
        document_anonymizer a(p, detector);
        document_report report;
        const string name = a.anonymize_name("contrat_jean.dupont@acme.fr_2024-03-12", report);
        DLIB_TEST_MSG(name.find("dupont") == string::npos && name.find("2024-03-12") == string::npos, name);
        DLIB_TEST_MSG(name.compare(0, 8, "contrat_") == 0 && name.find("[[") == string::npos, name);
        DLIB_TEST(report.substitutions.size() == 2);
        a.set_substitute_resolver(resolver_of(report.substitutions));
        document_report back;
        const string restored_name = a.restore_name(name, back);
        DLIB_TEST_MSG(restored_name == "contrat_jean.dupont@acme.fr_2024-03-12", restored_name);
        DLIB_TEST(back.substitutes_restored == 2);
        // A name found with the common word that opens it leaves the word: it names the document.
        {
            pii_detector wide;
            wide.add_term(pii_type::person, "Contrat Jean Dupont");
            document_anonymizer b(p, wide);
            document_report r;
            const string named = b.anonymize_name("Contrat Jean Dupont", r);
            DLIB_TEST_MSG(named == "Contrat " + person, named);
        }

        // The properties of a Word document take substitutes, and come back.
        zip_archive_writer w;
        w.add("[Content_Types].xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
              "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
              "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
              "<Override PartName=\"/word/document.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
              "<Override PartName=\"/docProps/core.xml\" "
              "ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
              "<Override PartName=\"/docProps/app.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/></Types>");
        w.add("_rels/.rels",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
              "<Relationship Id=\"rId1\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
              "Target=\"word/document.xml\"/>"
              "<Relationship Id=\"rId2\" "
              "Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" "
              "Target=\"docProps/core.xml\"/>"
              "<Relationship Id=\"rId3\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" "
              "Target=\"docProps/app.xml\"/></Relationships>");
        w.add("word/document.xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
              "<w:p><w:r><w:t>Je soussigné, M. Jean Dupont, atteste.</w:t></w:r></w:p><w:sectPr/></w:body>"
              "</w:document>");
        w.add("docProps/app.xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\">"
              "<Company>Dupont Conseil SARL</Company></Properties>");
        w.add("docProps/core.xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
              "xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>Attestation de M. Jean Dupont</dc:title>"
              "<dc:creator>Jean Dupont</dc:creator></cp:coreProperties>");
        const string docx = w.finish();
        document_anonymizer office(p, detector);
        office.set_untagged_properties(true);
        document_report oreport;
        const string oout = office.anonymize(docx, document_format::docx, oreport);
        const zip_archive_reader z(oout);
        const string core = z.read("docProps/core.xml");
        DLIB_TEST_MSG(core.find("Dupont") == string::npos && core.find("[[") == string::npos, core);
        DLIB_TEST_MSG(core.find("<dc:creator>" + person + "</dc:creator>") != string::npos, core);
        DLIB_TEST(z.read("docProps/app.xml").find("[[") == string::npos);
        DLIB_TEST(z.read("docProps/app.xml").find("Dupont") == string::npos);
        DLIB_TEST(z.read("word/document.xml").find("[[PERSON:") != string::npos);
        DLIB_TEST(oreport.substitutions.size() >= 3);
        office.set_substitute_resolver(resolver_of(oreport.substitutions));
        document_report oback;
        const zip_archive_reader r(office.restore(oout, document_format::docx, oback));
        DLIB_TEST_MSG(r.read("docProps/core.xml").find("<dc:creator>Jean Dupont</dc:creator>") != string::npos,
                      r.read("docProps/core.xml"));
        DLIB_TEST(r.read("docProps/core.xml").find("<dc:title>Attestation de M. Jean Dupont</dc:title>") !=
                  string::npos);
        DLIB_TEST(r.read("docProps/app.xml").find("<Company>Dupont Conseil SARL</Company>") != string::npos);
        DLIB_TEST(oback.substitutes_restored >= 3 && oback.tags_restored >= 1);

        // The document information of a PDF too.
        pdf_text_document source;
        source.pages.resize(1);
        source.pages[0].x1 = 595;
        source.pages[0].y1 = 842;
        std::vector<std::vector<pdf_text_segment>> lines(1);
        pdf_text_segment s;
        s.text = "Je soussigné, M. Jean Dupont, atteste.";
        s.x = 72;
        s.y = 700;
        s.font_size = 11;
        s.width = 300;
        lines[0].push_back(s);
        std::vector<std::pair<string, string>> info;
        info.push_back(std::make_pair("Title", "Attestation de M. Jean Dupont"));
        info.push_back(std::make_pair("Author", "Jean Dupont"));
        info.push_back(std::make_pair("Producer", "Writer 7"));
        const string pdf = write_pdf_text(source, lines, nullptr, info);
        DLIB_TEST(read_pdf_text(pdf).info.size() == 3);
        document_anonymizer printer(p, detector);
        printer.set_untagged_properties(true);
        document_report preport;
        const string pout = printer.anonymize(pdf, document_format::pdf, preport);
        const pdf_text_document pdoc = read_pdf_text(pout);
        DLIB_TEST(pdoc.info.size() == 2);
        for (const auto& kv : pdoc.info)
            DLIB_TEST_MSG(kv.second.find("Dupont") == string::npos && kv.second.find("[[") == string::npos, kv.second);
        printer.set_substitute_resolver(resolver_of(preport.substitutions));
        document_report pback;
        const pdf_text_document restored = read_pdf_text(printer.restore(pout, document_format::pdf, pback));
        DLIB_TEST(restored.info.size() == 2);
        for (const auto& kv : restored.info)
            DLIB_TEST_MSG(kv.second == (kv.first == "Title" ? "Attestation de M. Jean Dupont" : "Jean Dupont"),
                          kv.first << ": " << kv.second);
        // Without untagged properties, the document information stays out.
        document_report plain;
        DLIB_TEST(read_pdf_text(a.anonymize(pdf, document_format::pdf, plain)).info.empty());
    }

    void test_untagged_text()
    {
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x38')));
        pii_detector detector;

        // A text without tags: substitutes alone, listed with their tags, which alone give the values back.
        const string text = "Contrat de M. Jean Dupont, signé à Rennes le 12/03/2024.\n\n"
                            "Contact : jean.dupont@exemple.fr, 06 12 34 56 78.\n";
        document_anonymizer a(p, detector);
        a.set_untagged_text(true);
        document_report report;
        const string out = a.anonymize(text, document_format::text, report);
        DLIB_TEST_MSG(out.find("[[") == string::npos && out.find("Dupont") == string::npos, out);
        DLIB_TEST_MSG(out.find("dupont@") == string::npos && out.find("06 12 34 56 78") == string::npos, out);
        DLIB_TEST(!report.substitutions.empty() && report.substitutions.size() >= report.total_entities());
        for (const document_substitution& s : report.substitutions)
            DLIB_TEST_MSG(s.text && out.find(s.substitute) != string::npos && s.tag.compare(0, 2, "[[") == 0,
                          s.substitute);
        document_anonymizer b(p, detector);
        document_report none;
        DLIB_TEST(b.restore(out, document_format::text, none) == out);
        DLIB_TEST(none.tags_restored == 0 && none.substitutes_restored == 0);
        b.set_text_substitute_resolver(resolver_of(report.substitutions));
        document_report back;
        const string values = b.restore(out, document_format::text, back);
        DLIB_TEST_MSG(values == text, values);
        DLIB_TEST(back.substitutes_restored == report.substitutions.size());

        // Protected again with the substitutes known, the text keeps them as they are.
        document_anonymizer again(p, detector);
        again.set_untagged_text(true);
        again.set_text_substitute_resolver(resolver_of(report.substitutions));
        document_report second;
        const string twice = again.anonymize(out, document_format::text, second);
        DLIB_TEST_MSG(twice == out, twice);
        DLIB_TEST(second.substitutions.empty());

        // A text holding tags and substitutes, the old values of a file protected again: both come back.
        const string tagged = p.tag(pii_type::person, "Hélène Martin");
        const string mixed = "Visa de " + tagged + ". " + out;
        document_report both;
        const string mixed_back = b.restore(mixed, document_format::text, both);
        DLIB_TEST_MSG(mixed_back == "Visa de Hélène Martin. " + text, mixed_back);
        DLIB_TEST(both.tags_restored == 1 && both.substitutes_restored == report.substitutions.size());

        // A text protected before with tags, by an older engine for instance, and protected again
        // without them: its tags take the substitutes of their values, given back with the others.
        DLIB_TEST(p.substitute(pii_type::person, tagged) == p.substitute(pii_type::person, "Hélène Martin"));
        DLIB_TEST(p.substitute(pii_type::organization, tagged) != p.substitute(pii_type::person, "Hélène Martin"));
        const string old_tags = "Visa de " + tagged + ".\n\n" + text;
        document_report converted;
        const string bare = a.anonymize(old_tags, document_format::text, converted);
        DLIB_TEST_MSG(bare.find("[[") == string::npos && bare.find("Martin") == string::npos &&
                      bare.find("Dupont") == string::npos, bare);
        DLIB_TEST(bare.find(p.substitute(pii_type::person, "Hélène Martin")) != string::npos);
        document_anonymizer bare_back(p, detector);
        bare_back.set_text_substitute_resolver(resolver_of(converted.substitutions));
        document_report bare_report;
        const string bare_values = bare_back.restore(bare, document_format::text, bare_report);
        DLIB_TEST_MSG(bare_values == "Visa de Hélène Martin.\n\n" + text, bare_values);
        // A tag that an entity overlaps stays out; the others join the entities in text order.
        const string two = "ab" + tagged + " cd " + tagged;
        std::vector<pii_entity> one(1);
        one[0].begin = 2;
        one[0].end = 10;
        one[0].type = pii_type::location;
        const std::vector<pii_entity> with = p.with_tag_entities(two, one);
        DLIB_TEST(with.size() == 2 && with[0].type == pii_type::location && with[1].type == pii_type::person);
        DLIB_TEST(with[1].begin == two.size() - tagged.size() && with[1].end == two.size());
        DLIB_TEST(p.with_tag_entities("Jean [[PERSON:abc#XYZ]]", one).size() == 1);

        // The cells of a CSV file.
        const string csv = "nom;prenom;email\nDupont;Jean;jean.dupont@exemple.fr\nMartin;Hélène;h.martin@exemple.fr\n";
        document_report creport;
        const string cout_ = a.anonymize(csv, document_format::delimited, creport);
        DLIB_TEST_MSG(cout_.find("[[") == string::npos && cout_.find("Dupont") == string::npos, cout_);
        document_anonymizer c(p, detector);
        c.set_text_substitute_resolver(resolver_of(creport.substitutions));
        document_report cback;
        const string cvalues = c.restore(cout_, document_format::delimited, cback);
        DLIB_TEST_MSG(cvalues == csv, cvalues);

        // Markup: the text and a link, whose substitute stays a valid address.
        const string html = "<html><body><p>Écrire à <a href=\"mailto:jean.dupont@exemple.fr\">M. Jean Dupont</a>."
                            "</p></body></html>";
        document_report hreport;
        const string hout = a.anonymize(html, document_format::markup, hreport);
        DLIB_TEST_MSG(hout.find("[[") == string::npos && hout.find("Dupont") == string::npos, hout);
        DLIB_TEST_MSG(hout.find("href=\"mailto:") != string::npos && hout.find("%5B") == string::npos, hout);
        document_anonymizer h(p, detector);
        h.set_text_substitute_resolver(resolver_of(hreport.substitutions));
        document_report hback;
        const string hvalues = h.restore(hout, document_format::markup, hback);
        DLIB_TEST_MSG(hvalues == html, hvalues);

        // A Word document: the runs of its text.
        zip_archive_writer w;
        w.add("[Content_Types].xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
              "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
              "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
              "<Override PartName=\"/word/document.xml\" "
              "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
              "</Types>");
        w.add("_rels/.rels",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
              "<Relationship Id=\"rId1\" "
              "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
              "Target=\"word/document.xml\"/></Relationships>");
        w.add("word/document.xml",
              "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
              "<w:p><w:r><w:t>Je soussigné, M. Jean </w:t></w:r><w:r><w:t>Dupont, demeurant à Rennes, atteste.</w:t>"
              "</w:r></w:p><w:sectPr/></w:body></w:document>");
        const string docx = w.finish();
        document_report dreport;
        const string dout = a.anonymize(docx, document_format::docx, dreport);
        const string body = zip_archive_reader(dout).read("word/document.xml");
        DLIB_TEST_MSG(body.find("[[") == string::npos && body.find("Dupont") == string::npos, body);
        document_anonymizer d(p, detector);
        d.set_text_substitute_resolver(resolver_of(dreport.substitutions));
        document_report dback;
        const string dtext =
            zip_archive_reader(d.restore(dout, document_format::docx, dback)).read("word/document.xml");
        // A value written across runs comes back in the first one, as from its tag.
        DLIB_TEST_MSG(dtext.find("<w:t>Je soussigné, M. Jean Dupont</w:t>") != string::npos &&
                      dtext.find("<w:t>, demeurant à Rennes, atteste.</w:t>") != string::npos, dtext);
        DLIB_TEST(dback.substitutes_restored == dreport.substitutions.size() && dback.tags_restored == 0);
        // Its tags, written by a protection before, take substitutes too.
        zip_archive_writer w2;
        w2.add("[Content_Types].xml", zip_archive_reader(docx).read("[Content_Types].xml"));
        w2.add("_rels/.rels", zip_archive_reader(docx).read("_rels/.rels"));
        w2.add("word/document.xml",
               "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
               "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
               "<w:p><w:r><w:t>Visa de " + tagged + ", demeurant à Rennes.</w:t></w:r></w:p><w:sectPr/>"
               "</w:body></w:document>");
        document_report treport;
        const string tout = a.anonymize(w2.finish(), document_format::docx, treport);
        const string tbody = zip_archive_reader(tout).read("word/document.xml");
        DLIB_TEST_MSG(tbody.find("[[") == string::npos && tbody.find("Martin") == string::npos, tbody);
        document_anonymizer t(p, detector);
        t.set_text_substitute_resolver(resolver_of(treport.substitutions));
        document_report tback;
        const string ttext =
            zip_archive_reader(t.restore(tout, document_format::docx, tback)).read("word/document.xml");
        DLIB_TEST_MSG(ttext.find("<w:t>Visa de Hélène Martin, demeurant à Rennes.</w:t>") != string::npos, ttext);

        // A PDF: the text of its pages.
        pdf_text_document source;
        source.pages.resize(1);
        source.pages[0].x1 = 595;
        source.pages[0].y1 = 842;
        std::vector<std::vector<pdf_text_segment>> lines(1);
        pdf_text_segment seg;
        seg.text = "Je soussigné, M. Jean Dupont, demeurant à Rennes, atteste.";
        seg.x = 72;
        seg.y = 700;
        seg.font_size = 11;
        seg.width = 330;
        lines[0].push_back(seg);
        const string pdf = write_pdf_text(source, lines);
        document_report preport;
        const string pout = a.anonymize(pdf, document_format::pdf, preport);
        string pall;
        for (const auto& page : layout_pdf_text(read_pdf_text(pout)))
            for (const pdf_text_segment& ps : page) pall += ps.text + " ";
        DLIB_TEST_MSG(pall.find("[[") == string::npos && pall.find("Dupont") == string::npos, pall);
        document_anonymizer q(p, detector);
        q.set_text_substitute_resolver(resolver_of(preport.substitutions));
        document_report pback;
        string prestored;
        for (const auto& page : layout_pdf_text(read_pdf_text(q.restore(pout, document_format::pdf, pback))))
            for (const pdf_text_segment& ps : page) prestored += ps.text + " ";
        DLIB_TEST_MSG(prestored.find("M. Jean Dupont") != string::npos && prestored.find("Rennes") != string::npos,
                      prestored);
        // Its tags, each in one piece of its text, take substitutes too.
        lines[0][0].text = "Visa de " + tagged + ", demeurant à Rennes.";
        document_report p2report;
        const string p2out = a.anonymize(write_pdf_text(source, lines), document_format::pdf, p2report);
        string p2all;
        for (const auto& page : layout_pdf_text(read_pdf_text(p2out)))
            for (const pdf_text_segment& ps : page) p2all += ps.text + " ";
        DLIB_TEST_MSG(p2all.find("[[") == string::npos && p2all.find("Martin") == string::npos, p2all);
        document_anonymizer q2(p, detector);
        q2.set_text_substitute_resolver(resolver_of(p2report.substitutions));
        document_report p2back;
        string p2restored;
        for (const auto& page : layout_pdf_text(read_pdf_text(q2.restore(p2out, document_format::pdf, p2back))))
            for (const pdf_text_segment& ps : page) p2restored += ps.text + " ";
        DLIB_TEST_MSG(p2restored.find("Visa de Hélène Martin") != string::npos, p2restored);
    }

    // A PDF written by hand, its objects numbered from 1, with a cross-reference table.
    string pdf_file(const std::vector<string>& objects)
    {
        string out = "%PDF-1.7\n";
        std::vector<size_t> offsets;
        for (size_t i = 0; i < objects.size(); ++i)
        {
            offsets.push_back(out.size());
            out += std::to_string(i + 1) + " 0 obj\n" + objects[i] + "\nendobj\n";
        }
        const size_t xref = out.size();
        out += "xref\n0 " + std::to_string(objects.size() + 1) + "\n0000000000 65535 f \n";
        for (size_t off : offsets)
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%010lu 00000 n \n", static_cast<unsigned long>(off));
            out += buf;
        }
        return out + "trailer\n<< /Size " + std::to_string(objects.size() + 1) + " /Root 1 0 R /Info 9 0 R >>\n"
               "startxref\n" + std::to_string(xref) + "\n%%EOF\n";
    }

    string pdf_stream(const string& dict, const string& data)
    {
        return "<< " + dict + " /Length " + std::to_string(data.size()) + " >>\nstream\n" + data + "\nendstream";
    }

    // Every byte of a PDF, its Flate streams inflated.
    string pdf_bytes_inflated(const string& pdf)
    {
        string all = pdf;
        for (size_t p = pdf.find("stream\n"); p != string::npos; p = pdf.find("stream\n", p + 7))
        {
            const size_t e = pdf.find("\nendstream", p);
            if (e == string::npos) break;
            try { all += zlib_decompress(pdf.substr(p + 7, e - p - 7)); }
            catch (std::exception&) {}
        }
        return all;
    }

    void test_pdf_layout()
    {
        /* A page with graphics, a name cut by the kerning of a TJ operator, text drawn twice
           (fake bold), a letterhead form drawn twice, a picture, an inline picture, an
           alternate text, a comment of the content stream, a filled form field and a text
           annotation. */
        const string content =
            "% Jean Dupont\n"
            "0.9 0.1 0.1 rg 50 750 500 40 re f\n"
            "q 1 0 0 1 0 0 cm /Head Do Q q 1 0 0 1 0 -100 cm /Head Do Q\n"
            "/Span << /ActualText (Jean Dupont) >> BDC\n"
            "BT /F1 11 Tf 72 600 Td [(Rapport de M. Jean Du) -15 (pont, joignable au 06 12 34 56 78.)] TJ ET\n"
            "EMC\n"
            "BT /F2 11 Tf 72 580 Td (Sign\\351 : Mme H\\351l\\350ne LE GOFF) Tj ET\n"
            "BT /F2 11 Tf 72.3 580 Td (Sign\\351 : Mme H\\351l\\350ne LE GOFF) Tj ET\n"
            "q 100 0 0 50 72 400 cm /Im1 Do Q\n"
            "q 10 0 0 10 300 400 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x80 EI Q\n";
        const string head = "BT /F1 9 Tf 72 760 Td (Clinique des Glycines, Dr Paul Martin, Quimper) Tj ET";
        const string widget = "/Tx BMC BT /F1 10 Tf 2 4 Td (Anne Morel) Tj ET EMC";
        const string pdf = pdf_file({
            "<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [11 0 R] >> /StructTreeRoot << >> >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Annots [11 0 R 12 0 R] "
            "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> /XObject << /Head 7 0 R /Im1 8 0 R >> >> >>",
            pdf_stream("", content),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold /Encoding /WinAnsiEncoding >>",
            pdf_stream("/Type /XObject /Subtype /Form /BBox [0 0 595 842] /Resources << /Font << /F1 5 0 R >> >>",
                       head),
            pdf_stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",
                       string("\xFF\x00\x00\x00\xFF\x00", 6)),
            "<< /Author (Jean Dupont) /Title (Rapport) >>",
            pdf_stream("/Type /XObject /Subtype /Form /BBox [0 0 100 20] /Resources << /Font << /F1 5 0 R >> >>",
                       widget),
            "<< /Type /Annot /Subtype /Widget /FT /Tx /T (nom) /V (Anne Morel) /Rect [72 300 172 320] "
            "/AP << /N 10 0 R >> >>",
            "<< /Type /Annot /Subtype /Text /Rect [400 700 420 720] /Contents (Note de Jean Dupont) >>" });
        DLIB_TEST(detect_document_format(pdf) == document_format::pdf);
        string text;
        for (const auto& page : layout_pdf_text(read_pdf_text(pdf)))
            for (const pdf_text_segment& s : page) text += s.text + "\n";
        DLIB_TEST_MSG(text.find("Jean Dupont") != string::npos && text.find("Anne Morel") != string::npos, text);

        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x37')));
        pii_detector detector;
        document_anonymizer a(p, detector);
        DLIB_TEST(a.pdf_layout());
        document_report report;
        const string out = a.anonymize(pdf, document_format::pdf, report);
        const pdf_text_document reread = read_pdf_text(out);
        DLIB_TEST(reread.rewritten_by_dlib && !reread.produced_by_dlib && reread.pages.size() == 1);
        // No value left anywhere in the file, streams included; the graphics and the pictures' places stay.
        const string bytes = pdf_bytes_inflated(out);
        for (const char* v : { "Dupont", "LE GOFF", "06 12 34 56 78", "Morel", "Martin", "Note de", "ActualText" })
            DLIB_TEST_MSG(bytes.find(v) == string::npos, v);
        DLIB_TEST(bytes.find("500 40 re") != string::npos);
        DLIB_TEST(bytes.find("/Width 1 /Height 1") != string::npos && bytes.find("BI /W 1 /H 1") != string::npos);
        DLIB_TEST(bytes.find("/Annots") == string::npos && bytes.find("/StructTreeRoot") == string::npos &&
                  bytes.find("/Author") == string::npos);
        DLIB_TEST(bytes.find("/BaseFont /Times-Bold") != string::npos);
        string anon;
        for (const auto& page : layout_pdf_text(reread))
            for (const pdf_text_segment& s : page) anon += s.text + "\n";
        DLIB_TEST_MSG(anon.find("[[PERSON:") != string::npos && anon.find("[[PHONE:") != string::npos, anon);
        // The letterhead drawn twice and the field drawn into the page, both protected.
        size_t heads = 0;
        for (size_t k = anon.find("Clinique des Glycines"); k != string::npos; k = anon.find("Clinique des Glycines", k + 1))
            ++heads;
        DLIB_TEST_MSG(heads == 0 || anon.find("Dr [[PERSON:") != string::npos, anon);

        document_report back;
        const string restored_pdf = a.restore(out, document_format::pdf, back);
        DLIB_TEST_MSG(back.tags_restored == report.total_entities(),
                      back.tags_restored << " " << report.total_entities() << "\n" << anon);
        string restored;
        for (const auto& page : layout_pdf_text(read_pdf_text(restored_pdf)))
            for (const pdf_text_segment& s : page) restored += s.text + "\n";
        for (const char* v : { "Rapport de M. Jean Dupont, joignable au 06 12 34 56 78.", "Mme Hélène LE GOFF",
                               "Dr Paul Martin", "Anne Morel" })
            DLIB_TEST_MSG(restored.find(v) != string::npos, v << "\n" << restored);
        // A PDF without tag that was not protected here comes back unchanged.
        document_report none;
        DLIB_TEST(a.restore(pdf, document_format::pdf, none) == pdf);
        // The text-only reconstruction stays available.
        a.set_pdf_layout(false);
        document_report flat;
        DLIB_TEST(read_pdf_text(a.anonymize(pdf, document_format::pdf, flat)).produced_by_dlib);
    }

// ----------------------------------------------------------------------------------------

    void test_pdf_wrapped_cell()
    {
        // An e-mail address cut by its hyphen in a narrow cell of a table is found whole.
        const string content =
            "BT /F1 8.5 Tf 72 197.6 Td (Assistante de direction) Tj ET\n"
            "BT /F1 8.5 Tf 300 197.6 Td (fatou.ndiaye@arvelis-) Tj ET\n"
            "BT /F1 8.5 Tf 300 187.2 Td (services.fr) Tj ET\n"
            "BT /F1 8.5 Tf 450 197.6 Td (Lien malveillant recu) Tj ET\n";
        const string pdf = pdf_file({
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>",
            pdf_stream("", content),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>" });
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x43')));
        pii_detector detector;
        document_anonymizer a(p, detector);
        document_report report;
        const string out = a.anonymize(pdf, document_format::pdf, report);
        string text;
        for (const auto& page : layout_pdf_text(read_pdf_text(out)))
            for (const pdf_text_segment& s : page) text += s.text + "\n";
        DLIB_TEST_MSG(text.find("arvelis") == string::npos && text.find("[[EMAIL:") != string::npos, text);
        document_report back;
        string restored;
        for (const auto& page : layout_pdf_text(read_pdf_text(a.restore(out, document_format::pdf, back))))
            for (const pdf_text_segment& s : page) restored += s.text;
        DLIB_TEST_MSG(restored.find("fatou.ndiaye@arvelis-services.fr") != string::npos ||
                      restored.find("fatou.ndiaye@arvelis-") != string::npos, restored);

        /* The end of the address drawn with the text of the cell on its left, in one segment,
           alone on its line or with another cell. */
        for (int other = 0; other < 2; ++other)
        {
            const string merged = string(
                "BT /F1 8.5 Tf 72 115.7 Td (Client, Transports) Tj ET\n"
                "BT /F1 8.5 Tf 185 115.7 Td (b.lefeuvre@ouessant-) Tj ET\n"
                "BT /F1 8.5 Tf 330 115.7 Td (02 97 64 33 81) Tj ET\n"
                "BT /F1 8.5 Tf 72 105.3 Td (Ouessant Logistique SARL) Tj ET\n"
                "BT /F1 8.5 Tf 185 105.3 Td (logistique.fr) Tj ET\n") +
                (other ? "BT /F1 8.5 Tf 400 105.3 Td (recue) Tj ET\n" : "");
            const string mpdf = pdf_file({
                "<< /Type /Catalog /Pages 2 0 R >>",
                "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
                "/Resources << /Font << /F1 5 0 R >> >> >>",
                pdf_stream("", merged),
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>" });
            string before;
            for (const auto& page : layout_pdf_text(read_pdf_text(mpdf)))
                for (const pdf_text_segment& s : page) before += s.text + "\n";
            DLIB_TEST_MSG(before.find("Ouessant Logistique SARL logistique.fr") != string::npos, before);
            document_report mreport;
            const string mout = a.anonymize(mpdf, document_format::pdf, mreport);
            string mtext;
            for (const auto& page : layout_pdf_text(read_pdf_text(mout)))
                for (const pdf_text_segment& s : page) mtext += s.text + "\n";
            DLIB_TEST_MSG(mtext.find("lefeuvre") == string::npos && mtext.find("logistique.fr") == string::npos &&
                          mtext.find("[[EMAIL:") != string::npos, mtext);
            DLIB_TEST_MSG(mtext.find("97 64") == string::npos && (!other || mtext.find("recue") != string::npos), mtext);
            document_report mback;
            string mrestored;
            for (const auto& page : layout_pdf_text(read_pdf_text(a.restore(mout, document_format::pdf, mback))))
                for (const pdf_text_segment& s : page) mrestored += s.text + "\n";
            DLIB_TEST_MSG(mrestored.find("b.lefeuvre@ouessant-") != string::npos &&
                          mrestored.find("logistique.fr") != string::npos &&
                          mrestored.find("Ouessant Logistique SARL") != string::npos, mrestored);
        }
    }

    void test_image_describer()
    {
        // The pictures given to a describer, whose text takes their place in each format.
        std::vector<document_image> seen;
        const document_image_describer describer = [&seen](const document_image& img) {
            seen.push_back(img);
            return img.media_type == "image/png" ? string("[Image@42 : a red & green square]") : string();
        };
        const string png = encode_png(string("\xFF\x00\x00\x00\xFF\x00", 6), 2, 1, 3);
        DLIB_TEST(image_media_type(png) == "image/png" && png.size() > 40);
        DLIB_TEST(image_media_type("\xFF\xD8\xFF\xE0") == "image/jpeg" && image_media_type("abc").empty());
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x41')));
        // The texts of the describer, which a restoration without the pictures removes.
        const document_picture_text_finder pictures = [](const string& text) {
            std::vector<std::pair<size_t, size_t>> out;
            size_t b = text.find("[Image@");
            while (b != string::npos)
            {
                size_t e = b + 7;
                while (e < text.size() && text[e] != ']')
                    e = text.compare(e, 2, "[[") == 0 ? text.find("]]", e) + 2 : e + 1;
                if (e >= text.size()) break;
                out.push_back(std::make_pair(b, e + 1));
                b = text.find("[Image@", e);
            }
            return out;
        };
        // A text: the picture text leaves with the tag of its description, the tag outside comes back.
        {
            pii_detector detector;
            document_anonymizer a(p, detector);
            a.set_picture_text_finder(pictures);
            document_report report;
            string tagged = a.anonymize("Photo de Jean Dupont\n", document_format::text, report);
            tagged.pop_back();
            document_report back;
            const string restored = a.restore("Voir [Image@42 : " + tagged + "] et " + tagged + ".\n",
                                              document_format::text, back);
            DLIB_TEST_MSG(restored == "Voir  et Photo de Jean Dupont.\n" && back.pictures_removed == 1 &&
                          back.tags_restored == 1, restored);
        }

        // Word: the drawing of a run becomes a run text.
        {
            zip_archive_writer w;
            w.add("[Content_Types].xml",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types "
                  "xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                  "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                  "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                  "<Default Extension=\"png\" ContentType=\"image/png\"/>"
                  "<Override PartName=\"/word/document.xml\" "
                  "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
                  "</Types>");
            w.add("_rels/.rels",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships "
                  "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" "
                  "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
                  "Target=\"word/document.xml\"/></Relationships>");
            w.add("word/_rels/document.xml.rels",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships "
                  "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId5\" "
                  "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" "
                  "Target=\"media/image1.png\"/></Relationships>");
            w.add("word/document.xml",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                  "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
                  "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
                  "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
                  "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\"><w:body>"
                  "<w:p><w:r><w:t>Photo de Jean Dupont :</w:t></w:r></w:p>"
                  "<w:p><w:r><w:drawing><wp:inline><a:graphic><a:graphicData><a:blip r:embed=\"rId5\"/>"
                  "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"
                  "<w:r><w:drawing><wp:inline><a:graphic><a:graphicData><a:blip r:embed=\"rId5\"/>"
                  "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p><w:sectPr/></w:body></w:document>");
            w.add("word/media/image1.png", png);
            pii_detector detector;
            document_anonymizer a(p, detector);
            a.set_image_describer(describer);
            document_report report;
            const string protected_docx = a.anonymize(w.finish(), document_format::docx, report);
            const zip_archive_reader z(protected_docx);
            const string doc = z.read("word/document.xml");
            DLIB_TEST(seen.size() == 1 && seen[0].location == "word/media/image1.png" && seen[0].bytes == png);
            DLIB_TEST_MSG(doc.find("<w:t xml:space=\"preserve\">[Image@42 : a red &amp; green square]</w:t>") !=
                          string::npos, doc);
            DLIB_TEST(doc.find("blip") == string::npos && doc.find("Dupont") == string::npos);
            DLIB_TEST(!z.contains("word/media/image1.png"));
            // Restored without its pictures: their texts leave the runs, the values come back.
            document_report back;
            a.set_picture_text_finder(pictures);
            const zip_archive_reader r(a.restore(protected_docx, document_format::docx, back));
            const string restored = r.read("word/document.xml");
            DLIB_TEST_MSG(restored.find("Image@") == string::npos && restored.find("Jean Dupont") != string::npos &&
                          back.pictures_removed == 2, restored);
        }

        // PowerPoint: the picture becomes a text box at its position.
        seen.clear();
        {
            zip_archive_writer w;
            w.add("[Content_Types].xml",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types "
                  "xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                  "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                  "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                  "<Default Extension=\"png\" ContentType=\"image/png\"/>"
                  "<Override PartName=\"/ppt/presentation.xml\" "
                  "ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/>"
                  "<Override PartName=\"/ppt/slides/slide1.xml\" "
                  "ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>"
                  "</Types>");
            w.add("_rels/.rels",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships "
                  "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" "
                  "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
                  "Target=\"ppt/presentation.xml\"/></Relationships>");
            w.add("ppt/presentation.xml",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><p:presentation "
                  "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\" "
                  "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><p:sldIdLst>"
                  "<p:sldId id=\"256\" r:id=\"rId2\"/></p:sldIdLst></p:presentation>");
            w.add("ppt/_rels/presentation.xml.rels",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships "
                  "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId2\" "
                  "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide\" "
                  "Target=\"slides/slide1.xml\"/></Relationships>");
            w.add("ppt/slides/_rels/slide1.xml.rels",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships "
                  "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId3\" "
                  "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" "
                  "Target=\"../media/image1.png\"/></Relationships>");
            w.add("ppt/slides/slide1.xml",
                  "<?xml version=\"1.0\" encoding=\"UTF-8\"?><p:sld "
                  "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\" "
                  "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
                  "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><p:cSld><p:spTree>"
                  "<p:sp><p:nvSpPr><p:cNvPr id=\"2\" name=\"Titre\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr/>"
                  "<p:txBody><a:bodyPr/><a:p><a:r><a:t>Equipe de Jean Dupont</a:t></a:r></a:p></p:txBody></p:sp>"
                  "<p:pic><p:nvPicPr><p:cNvPr id=\"4\" name=\"Picture 3\"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>"
                  "<p:blipFill><a:blip r:embed=\"rId3\"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>"
                  "<p:spPr><a:xfrm><a:off x=\"914400\" y=\"914400\"/><a:ext cx=\"3048000\" cy=\"1524000\"/></a:xfrm>"
                  "<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr></p:pic>"
                  "</p:spTree></p:cSld></p:sld>");
            w.add("ppt/media/image1.png", png);
            pii_detector detector;
            document_anonymizer a(p, detector);
            a.set_image_describer(describer);
            document_report report;
            const string protected_pptx = a.anonymize(w.finish(), document_format::pptx, report);
            const zip_archive_reader z(protected_pptx);
            const string slide = z.read("ppt/slides/slide1.xml");
            DLIB_TEST(seen.size() == 1 && seen[0].location == "ppt/media/image1.png");
            DLIB_TEST_MSG(slide.find("<p:sp><p:nvSpPr><p:cNvPr id=\"4\" name=\"Picture 3\"/><p:cNvSpPr txBox=\"1\"/>") !=
                          string::npos, slide);
            DLIB_TEST(slide.find("<a:xfrm><a:off x=\"914400\" y=\"914400\"/><a:ext cx=\"3048000\" cy=\"1524000\"/>"
                                 "</a:xfrm>") != string::npos);
            DLIB_TEST(slide.find("<a:t>[Image@42 : a red &amp; green square]</a:t>") != string::npos);
            DLIB_TEST(slide.find("blip") == string::npos && slide.find("Dupont") == string::npos);
            DLIB_TEST(!z.contains("ppt/media/image1.png"));
            // Restored without its picture: the gray text box leaves the slide, the title comes back.
            document_report back;
            a.set_picture_text_finder(pictures);
            const zip_archive_reader r(a.restore(protected_pptx, document_format::pptx, back));
            const string restored = r.read("ppt/slides/slide1.xml");
            DLIB_TEST_MSG(restored.find("Image@") == string::npos && restored.find("txBox") == string::npos &&
                          restored.find("E6E6E6") == string::npos && restored.find("Jean Dupont") != string::npos &&
                          back.pictures_removed == 1, restored);
        }

        // PDF: the text is written over the gray box of the picture, drawn twice, described once.
        seen.clear();
        {
            const string content =
                "BT /F1 11 Tf 72 700 Td (Photo de Jean Dupont :) Tj ET\n"
                "q 200 0 0 100 72 500 cm /Im1 Do Q\n"
                "q 200 0 0 -100 72 400 cm /Im1 Do Q\n";
            const string pdf = pdf_file({
                "<< /Type /Catalog /Pages 2 0 R >>",
                "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
                "/Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> >>",
                pdf_stream("", content),
                "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
                pdf_stream("/Type /XObject /Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB /BitsPerComponent 8",
                           string("\xFF\x00\x00\x00\xFF\x00", 6)) });
            pii_detector detector;
            document_anonymizer a(p, detector);
            a.set_image_describer(describer);
            document_report report;
            const string out = a.anonymize(pdf, document_format::pdf, report);
            DLIB_TEST(seen.size() == 1 && seen[0].bytes == png && seen[0].location == "page 1");
            string text;
            for (const auto& page : layout_pdf_text(read_pdf_text(out)))
                for (const pdf_text_segment& s : page) text += s.text + "\n";
            size_t captions = 0;
            for (size_t k = text.find("red & green"); k != string::npos; k = text.find("red & green", k + 1)) ++captions;
            DLIB_TEST_MSG(captions == 2 && text.find("Dupont") == string::npos, text);
            // Upright in both boxes: the caption of the flipped picture reads inside its box.
            const pdf_text_document reread = read_pdf_text(out);
            for (const pdf_text_span& s : reread.pages[0].spans)
                if (s.text.find("Image@42") != string::npos)
                    DLIB_TEST_MSG(std::abs(s.angle) < 1 && s.x >= 72 && s.x < 272 &&
                                  ((s.y > 500 && s.y < 600) || (s.y > 300 && s.y < 400)), s.x << " " << s.y);
            // Restored without its pictures: the texts over the gray boxes leave, the values come back.
            document_report back;
            a.set_picture_text_finder(pictures);
            string restored;
            for (const auto& page : layout_pdf_text(read_pdf_text(a.restore(out, document_format::pdf, back))))
                for (const pdf_text_segment& s : page) restored += s.text + "\n";
            DLIB_TEST_MSG(restored.find("Image@") == string::npos && restored.find("Jean Dupont") != string::npos &&
                          back.pictures_removed == 2, restored << back.pictures_removed);
            // Without a describer the boxes stay empty.
            document_anonymizer plain(p, detector);
            document_report r2;
            string none;
            for (const auto& page : layout_pdf_text(read_pdf_text(plain.anonymize(pdf, document_format::pdf, r2))))
                for (const pdf_text_segment& s : page) none += s.text + "\n";
            DLIB_TEST(none.find("Image@") == string::npos);
        }
    }

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
        // A family name in capitals before a first name, as forms and lists write them; a firm stays one.
        text = "LE FLOC'H Anne, 12 rue de Penmarc'h\nGARAGE KERLANN Anne";
        e = read(text, {{"LE FLOC'H", "ORGANIZATION", 0.77}, {"Anne", "GIVEN_NAME", 0.59},
                        {"GARAGE KERLANN", "ORGANIZATION", 0.9}, {"Anne", "GIVEN_NAME", 0.9}});
        DLIB_TEST(found(e, text, pii_type::person, "LE FLOC'H Anne"));
        DLIB_TEST(found(e, text, pii_type::organization, "GARAGE KERLANN"));
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

    void test_short_texts()
    {
        /* Cells, titles of slides and comments: a common word capitalized by its place names
           nothing, a number before a unit or a common noun is no address, the label of a column
           keeps its person, and a first name read as a place or an organisation joins the family
           name after it. */
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
        auto read = [&](const string& text, const std::vector<value>& values, pii_type label) {
            *spans = values;
            detector.begin_document();
            detection_context ctx;
            ctx.label = label;
            return detector.detect(text, ctx);
        };
        auto count = [](const std::vector<pii_entity>& e, const string& text, const string& value) {
            size_t n = 0;
            for (const pii_entity& x : e)
                if (text.substr(x.begin, x.end - x.begin) == value) ++n;
            return n;
        };
        string text = "Recette\nBascule confirmée le 30 avril 2026\n• Doublon possible avec ADH-2021-11502\n"
                      "Prélèvement rejeté en janvier 2026.\nMédecin-conseil\nCotisation mensuelle";
        std::vector<pii_entity> e = read(text, {{"Recette", "FAMILY_NAME", 0.65}, {"Bascule", "CITY", 0.61},
                                                {"Doublon", "CITY", 0.69}, {"Prélèvement", "GIVEN_NAME", 0.54},
                                                {"Médecin-conseil", "ORGANIZATION", 0.9},
                                                {"Cotisation mensuelle", "ORGANIZATION", 0.78}}, pii_type::none);
        for (const char* w : { "Recette", "Bascule", "Doublon", "Prélèvement", "Médecin-conseil", "Cotisation" })
            DLIB_TEST_MSG(!covered(e, text, w), w);
        DLIB_TEST(found(e, text, pii_type::date, "30 avril 2026"));
        // A name after a civility or inside a sentence, a name that is no common word, a town.
        text = "Signé par M. Page. Le rapport de Page est prêt.\nLe Floc'h arrive.\nParis, le 3 mars 2026";
        e = read(text, {{"Page", "FAMILY_NAME", 0.9}, {"Page", "FAMILY_NAME", 0.9}, {"Le Floc'h", "FAMILY_NAME", 0.9},
                        {"Paris", "CITY", 0.9}}, pii_type::none);
        DLIB_TEST(count(e, text, "Page") == 2);
        DLIB_TEST(found(e, text, pii_type::person, "Le Floc'h"));
        DLIB_TEST(found(e, text, pii_type::location, "Paris"));
        // A number before a unit or before words in lower case, without a street type.
        text = "Comité à 14 h, dont 212 doublons, au 12 rue des Goélands.";
        e = read(text, {{"14", "STREET_ADDRESS", 0.63}, {"212 doublons", "STREET_ADDRESS", 0.94},
                        {"12 rue des Goélands", "STREET_ADDRESS", 0.99}}, pii_type::none);
        DLIB_TEST(!covered(e, text, "14"));
        DLIB_TEST(!covered(e, text, "212"));
        DLIB_TEST(found(e, text, pii_type::address, "12 rue des Goélands"));
        // A floor, a street without its type in a text in lower case, the value of a label, a firm.
        text = "Logement : 12 rue de Brest, 9e étage.\nre: 220 broadway, lowell\nApproved by: North\n"
               "estate agent, Tan Law, Bath.";
        e = read(text, {{"9e étage", "STREET_ADDRESS", 0.9}, {"220 broadway", "STREET_ADDRESS", 0.9},
                        {"North", "FAMILY_NAME", 0.8}, {"Tan", "ORGANIZATION", 0.98}, {"Law", "FAMILY_NAME", 0.71}},
                 pii_type::none);
        DLIB_TEST(found(e, text, pii_type::address, "9e étage"));
        DLIB_TEST(found(e, text, pii_type::address, "220 broadway"));
        DLIB_TEST(found(e, text, pii_type::person, "North"));
        DLIB_TEST(!found(e, text, pii_type::person, "Tan Law"));
        // The label of a column of names keeps the person a recognizer reads as an organisation.
        text = "LE FLOC'H";
        e = read(text, {{"FLOC'H", "ORGANIZATION", 0.85}}, pii_type::person);
        DLIB_TEST(found(e, text, pii_type::person, "LE FLOC'H"));
        // A first name read as an organisation or a place, before the family name.
        text = "Signalé par sa sœur, Nolwenn Le Floc'h, et par Malo Cozic.";
        e = read(text, {{"Nolwenn", "ORGANIZATION", 0.91}, {"Le Floc'h", "FAMILY_NAME", 0.83},
                        {"Malo", "CITY", 0.92}, {"Cozic", "FAMILY_NAME", 0.87}}, pii_type::none);
        DLIB_TEST(found(e, text, pii_type::person, "Nolwenn Le Floc'h"));
        DLIB_TEST(found(e, text, pii_type::person, "Malo Cozic"));
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

    void test_five_languages()
    {
        /* Italian, Spanish and German: civilities, dates, telephone numbers and tax numbers
           found by the rules; regulations in capitals and the words for a company left
           readable; the division of a group kept in its name. */
        pii_detector detector;
        string text = "Sehr geehrte Frau Schneider,\nvielen Dank für Ihre Nachricht vom 12. März 2024. Der Vertrag "
                      "wurde von Herrn Thomas Müller unterschrieben. Bitte rufen Sie mich unter 0221 123456 an.";
        std::vector<pii_entity> e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::title, "Frau") && found(e, text, pii_type::person, "Schneider"));
        DLIB_TEST(found(e, text, pii_type::title, "Herrn") && found(e, text, pii_type::person, "Thomas Müller"));
        DLIB_TEST(found(e, text, pii_type::date, "12. März 2024"));
        DLIB_TEST(found(e, text, pii_type::phone, "0221 123456"));
        DLIB_TEST(!covered(e, text, "Vertrag"));
        text = "Gentile Sig.ra Rossi, il contratto è stato firmato dal Dott. Marco Esposito il 3 aprile 2024. "
               "Per informazioni chiamare il 02 1234 5678. P.IVA 04808411203.";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::title, "Sig.ra") && found(e, text, pii_type::person, "Rossi"));
        DLIB_TEST(found(e, text, pii_type::person, "Marco Esposito") && !covered(e, text, "Dott."));
        DLIB_TEST(found(e, text, pii_type::date, "3 aprile 2024"));
        DLIB_TEST(found(e, text, pii_type::phone, "02 1234 5678"));
        DLIB_TEST(found(e, text, pii_type::identifier, "04808411203"));
        text = "Estimada Sra. García:\nel contrato fue firmado por D. Javier Martínez el 5 de mayo de 2024. "
               "Conforme al RGPD y a la LOPDGDD, puede llamarnos al 954 123 456. CIF: A07119548.";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::title, "Sra.") && found(e, text, pii_type::person, "García"));
        DLIB_TEST(found(e, text, pii_type::title, "D.") && found(e, text, pii_type::person, "Javier Martínez"));
        DLIB_TEST(found(e, text, pii_type::date, "5 de mayo de 2024"));
        DLIB_TEST(found(e, text, pii_type::phone, "954 123 456"));
        DLIB_TEST(found(e, text, pii_type::identifier, "A07119548"));
        DLIB_TEST(!covered(e, text, "RGPD") && !covered(e, text, "LOPDGDD"));
        /* Stacked civilities, a German mobile number in a table, a reference after its noun, the
           plates of the United Kingdom and of Spain before 2000 after a make of car or the word
           of a vehicle, and a profile link written without its scheme. */
        text = "Sr. D. Francisco Javier Ortega Ruiz y D.ª Esperanza Cid Lobo.\n"
               "49  01.12.2025  Tobias Brehm    0171 6043187\nMieternummer 0815-3307-02\n"
               "Vehicle: BMW 118i, YV19 HDX. Vehículo B: Renault Clio S-4512-BK.\n"
               "Perfil: linkedin.com/in/alvarodelasheras";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::title, "Sr.") && found(e, text, pii_type::title, "D."));
        DLIB_TEST(found(e, text, pii_type::title, "D.ª"));
        DLIB_TEST(found(e, text, pii_type::phone, "0171 6043187"));
        DLIB_TEST(found(e, text, pii_type::identifier, "0815-3307-02"));
        DLIB_TEST(found(e, text, pii_type::license_plate, "YV19 HDX"));
        DLIB_TEST(found(e, text, pii_type::license_plate, "S-4512-BK"));
        DLIB_TEST(found(e, text, pii_type::url, "linkedin.com/in/alvarodelasheras"));
        // A role after a civility names nobody; a name after the role is a person.
        text = "Illustrissimo Signor Sindaco,\nil Signor Rossi ha firmato. Caro Collega, Herr Bürgermeister.";
        e = detector.detect(text);
        DLIB_TEST(!covered(e, text, "Signor") && !covered(e, text, "Sindaco") && !covered(e, text, "Collega"));
        DLIB_TEST(!covered(e, text, "Herr") && !covered(e, text, "Bürgermeister"));
        DLIB_TEST(found(e, text, pii_type::person, "Rossi"));
        // The door after the floor of a Spanish street, and "Gran Vía".
        text = "Gran Vía de Don Diego López de Haro, 41, 3.º dcha., 48011 Bilbao. "
               "En Calle Cervantes, 14, 2.º izq. hoy.";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::address, "Gran Vía de Don Diego López de Haro, 41, 3.º dcha."));
        DLIB_TEST(found(e, text, pii_type::address, "Calle Cervantes, 14, 2.º izq."));
        // A known organisation with its division, before a civility.
        text = "Gemäß DSGVO hat die Deutsche Telekom IT Herrn Thomas Müller informiert.";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::organization, "Deutsche Telekom IT"));
        DLIB_TEST(found(e, text, pii_type::person, "Thomas Müller") && !covered(e, text, "DSGVO"));
        // Spanish companies and streets, a German public body with its place, a reference after its cue.
        text = "Construcciones Duero, S.A., Calle de los Almendros, 6, 1.º A, y el Finanzamt Köln (Aktenzeichen "
               "2511-004712, Polizza n. 0471/882331).";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::organization, "Construcciones Duero, S.A."));
        DLIB_TEST(found(e, text, pii_type::address, "Calle de los Almendros, 6, 1.º A"));
        DLIB_TEST(found(e, text, pii_type::organization, "Finanzamt Köln"));
        DLIB_TEST(found(e, text, pii_type::identifier, "2511-004712"));
        DLIB_TEST(found(e, text, pii_type::identifier, "0471/882331"));
        // A German address block: the civility on a line of its own; a VAT number.
        text = "Herrn\nMarius Kopp\nKanalstraße 9\n52349 Düren\nUSt-IdNr.: DE763208364";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::title, "Herrn") && found(e, text, pii_type::person, "Marius Kopp"));
        DLIB_TEST(found(e, text, pii_type::identifier, "DE763208364"));

        // The dates of the new languages move and come back.
        const pseudonymization_key key = pseudonymization_key::from_bytes(string(32, '\x29'));
        const pseudonymizer p(key);
        text = "Termin am 12. März 2024, firmato il 1° marzo 2023 e il 3 aprile 2024, el 5 de mayo de 2024, "
               "desde marzo de 2022.";
        e = detector.detect(text);
        for (const char* date : { "12. März 2024", "1° marzo 2023", "3 aprile 2024", "5 de mayo de 2024",
                                  "marzo de 2022" })
            DLIB_TEST_MSG(found(e, text, pii_type::date, date), date);
        const string protected_text = p.apply(text, e);
        for (const char* date : { "12. März 2024", "1° marzo 2023", "3 aprile 2024", "5 de mayo de 2024",
                                  "marzo de 2022" })
            DLIB_TEST_MSG(protected_text.find(date) == string::npos, date);
        DLIB_TEST(p.restore(protected_text) == text);

        /* What a recognizer reads: a regulation in capitals is no organisation, the word for a
           company before its name stays readable, a division after a group joins its name, and a
           civility read as a name is a title. */
        auto spans = std::make_shared<std::vector<std::pair<string, string>>>();
        auto confidence = std::make_shared<double>(0.9);
        pii_detector learned;
        learned.add_recognizer([spans, confidence](const string& t) {
            std::vector<pii_span> out;
            for (const auto& v : *spans)
            {
                const size_t at = t.find(v.first);
                if (at == string::npos) continue;
                pii_span s;
                s.begin = at; s.end = at + v.first.size(); s.label = v.second; s.confidence = *confidence;
                out.push_back(s);
            }
            std::sort(out.begin(), out.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
            return out;
        });
        auto read_with = [&](const string& t, const std::vector<std::pair<string, string>>& values, double c) {
            *spans = values;
            *confidence = c;
            learned.begin_document();
            return learned.detect(t);
        };
        auto read = [&](const string& t, const std::vector<std::pair<string, string>>& values) {
            return read_with(t, values, 0.9);
        };
        text = "Est-ce que cette information est une donnée personnelle au sens RGPD ? Mon nom est Albert. "
               "Ce document est issu d'informations techniques de Veolia Water.";
        e = read(text, {{"RGPD", "ORGANIZATION"}, {"Albert", "GIVEN_NAME"}, {"Veolia", "ORGANIZATION"},
                        {"Water", "FAMILY_NAME"}});
        DLIB_TEST(!covered(e, text, "RGPD"));
        DLIB_TEST(found(e, text, pii_type::person, "Albert"));
        DLIB_TEST(found(e, text, pii_type::organization, "Veolia Water"));
        text = "Der Vertrag mit der Firma Bauer Logistik GmbH gilt gemäß DSGVO. Dora Martin und die DORA-Prüfung.";
        e = read(text, {{"Firma Bauer Logistik GmbH", "ORGANIZATION"}, {"DSGVO", "ORGANIZATION"},
                        {"Dora", "GIVEN_NAME"}, {"Martin", "FAMILY_NAME"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Bauer Logistik GmbH") && !covered(e, text, "Firma"));
        DLIB_TEST(!covered(e, text, "DSGVO"));
        DLIB_TEST(found(e, text, pii_type::person, "Dora Martin"));
        text = "Caro Sig. Bianchi, il team di Deutsche Telekom IT e di Enel Green Power la contatterà.";
        e = read(text, {{"Sig", "FAMILY_NAME"}, {"Bianchi", "FAMILY_NAME"}, {"Deutsche Telekom", "ORGANIZATION"},
                        {"Enel", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::title, "Sig.") && found(e, text, pii_type::person, "Bianchi"));
        DLIB_TEST(found(e, text, pii_type::organization, "Deutsche Telekom IT"));
        DLIB_TEST(found(e, text, pii_type::organization, "Enel Green Power"));
        /* A warning read as a name stays readable, an article opens no company, and the floor of
           a street read apart joins its line. */
        text = "RISERVATO. Die Geschäftsführung der Müller & Partner GmbH, Calle Larios, 9, 4.º, 29005 Málaga.";
        e = read(text, {{"RISERVATO", "CITY"}, {"Müller & Partner GmbH", "ORGANIZATION"},
                        {"Calle Larios, 9", "STREET_ADDRESS"}, {"4.º", "STREET_ADDRESS"}});
        DLIB_TEST(!covered(e, text, "RISERVATO"));
        DLIB_TEST(found(e, text, pii_type::organization, "Müller & Partner GmbH") && !covered(e, text, "der M"));
        DLIB_TEST(found(e, text, pii_type::address, "Calle Larios, 9, 4.º"));
        /* Regions alone or joined, a country read as a person, the number of a standard and
           common words in lower case name nothing; a region read as a name after a civility
           stays one. */
        text = "Abbiamo uffici in Puglia e Sicilia e un cliente in Bretagne. Albania ha firmato l'accordo.\n"
               "Conforme alle norme ISO 27001 ed EN 50600. ok ci vediamo stasera, anche io.\nFirmato: Sig. Aragón";
        e = read(text, {{"Puglia e Sicilia", "ORGANIZATION"}, {"Bretagne", "CITY"}, {"Albania", "FAMILY_NAME"},
                        {"27001", "POSTAL_CODE"}, {"50600", "POSTAL_CODE"}, {"stasera", "GIVEN_NAME"},
                        {"anche", "FAMILY_NAME"}, {"Aragón", "FAMILY_NAME"}});
        DLIB_TEST(!covered(e, text, "Puglia") && !covered(e, text, "Sicilia") && !covered(e, text, "Bretagne"));
        DLIB_TEST(!covered(e, text, "Albania") && !covered(e, text, "27001") && !covered(e, text, "50600"));
        DLIB_TEST(!covered(e, text, "stasera") && !covered(e, text, "anche"));
        DLIB_TEST(found(e, text, pii_type::person, "Aragón"));
        // A street of the rules joins the street read on the same line, without the dash after it.
        text = "Entregar en: Calle Asunción, 51, 1.º - 41011 Sevilla. Bar en Plaza Mayor, 12, Valladolid.";
        e = read(text, {{"Calle Asunción, 51, 1.º -", "STREET_ADDRESS"}, {"Plaza Mayor", "STREET_ADDRESS"}});
        DLIB_TEST(found(e, text, pii_type::address, "Calle Asunción, 51, 1.º"));
        DLIB_TEST(found(e, text, pii_type::address, "Plaza Mayor, 12"));
        // After a comma, "de" says where the person comes from and stays out of the organisation.
        text = "Mon nom est Albert, de Veolia Water, au sens du RGPD.";
        e = read(text, {{"Albert", "GIVEN_NAME"}, {"Veolia Water", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Veolia Water") && !covered(e, text, "de Veolia"));
        // A file name with a numbered extension, a letter repeated and a common word read as a place.
        text = "journal.log2 dddddddddddd copies";
        e = read(text, {{"journal.log2", "USERNAME"}, {"dddddddddddd", "USERNAME"}, {"copies", "CITY"}});
        DLIB_TEST(!covered(e, text, "journal") && !covered(e, text, "dddd") && !covered(e, text, "copies"));
        // Two organisations that the rules read apart stay two values when a recognizer joins them.
        text = "Siemens Energy and Veolia Water signed the contract.";
        e = read(text, {{"Siemens Energy and Veolia", "ORGANIZATION"}, {"Water", "FAMILY_NAME"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Siemens Energy"));
        DLIB_TEST(found(e, text, pii_type::organization, "Veolia Water"));
        text = "Il contratto tra Enel Green Power e Veolia Water è firmato.";
        e = read(text, {{"Enel Green Power e Veolia", "ORGANIZATION"}, {"Water", "FAMILY_NAME"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Enel Green Power"));
        DLIB_TEST(found(e, text, pii_type::organization, "Veolia Water") && !covered(e, text, " e "));
        // A name whose rules read it whole keeps its conjunction.
        text = "Le contrat de Martin et Fils est signé.";
        e = read(text, {{"Martin et Fils", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Martin et Fils"));
        // A known organisation written with a conjunction for its ampersand.
        text = "Our supplier Marks and Spencer delivered, and Ernst und Young audited.";
        e = read(text, {{"Marks", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Marks and Spencer"));
        DLIB_TEST(found(e, text, pii_type::organization, "Ernst und Young"));
        // A firm name closed by partners after the conjunction stays whole.
        text = "La Gestoría Navarro y Asociados S.L. presentó la declaración.";
        e = read(text, {{"Gestoría Navarro y Asociados S.L.", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Gestoría Navarro y Asociados S.L."));
        // A capitalized word joined by a conjunction to a person found is another person.
        text = "Our lawyers Martin and Dupont signed the letter.";
        e = read(text, {{"Martin", "GIVEN_NAME"}});
        DLIB_TEST(found(e, text, pii_type::person, "Martin") && found(e, text, pii_type::person, "Dupont"));
        text = "Dupont et Martin ont signé. Les cabinets Clifford & Chance et Lefèvre ont répondu.";
        e = read(text, {{"Martin", "FAMILY_NAME"}, {"Clifford", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::person, "Dupont"));
        // Beside an organisation, it completes its name, unless a legal form ends the name.
        DLIB_TEST(found(e, text, pii_type::organization, "Clifford & Chance"));
        text = "La reunión con la Oficina de Asilo y Refugio queda confirmada.";
        e = read(text, {{"Oficina de Asilo", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Oficina de Asilo y Refugio"));
        text = "Portabilità richiesta verso Porcu e Lo Cascio S.a.s.: codice fornito.";
        e = read(text, {{"Lo Cascio S.a.s.", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Porcu e Lo Cascio S.a.s."));
        text = "The contract between Acme Ltd and Zorvane was signed.";
        e = read(text, {{"Acme Ltd", "ORGANIZATION"}});
        DLIB_TEST(found(e, text, pii_type::organization, "Acme Ltd") &&
                  found(e, text, pii_type::organization, "Zorvane"));
        // A name read with less confidence than 0.8 joins nothing.
        text = "Our lawyers Martin and Dupont signed the letter.";
        e = read_with(text, {{"Martin", "GIVEN_NAME"}}, 0.7);
        DLIB_TEST(found(e, text, pii_type::person, "Martin") && !covered(e, text, "Dupont"));
        // Partners, months, holidays, roles, regions, lists, capitals beside a name that is not, and
        // German nouns after "und" stay readable.
        text = "Pierre Martin et Partners, Paul Morel et Lundi, Anne Roux et Directeur, Jean Blanc et Bretagne. "
               "Thomas Müller und Kollegen.";
        e = read(text, {{"Pierre Martin", "GIVEN_NAME"}, {"Paul Morel", "GIVEN_NAME"}, {"Anne Roux", "GIVEN_NAME"},
                        {"Jean Blanc", "GIVEN_NAME"}, {"Thomas Müller", "GIVEN_NAME"}});
        DLIB_TEST(!covered(e, text, "Partners") && !covered(e, text, "Lundi") && !covered(e, text, "Directeur"));
        DLIB_TEST(!covered(e, text, "Bretagne") && !covered(e, text, "Kollegen"));
        text = "Los días festivos son Inmaculada y Navidad. Natale, Santo Stefano e Capodanno: sportelli chiusi.";
        e = read(text, {{"Inmaculada", "GIVEN_NAME"}, {"Stefano", "GIVEN_NAME"}});
        DLIB_TEST(!covered(e, text, "Navidad") && !covered(e, text, "Capodanno"));
        text = "El tribunal estima la demanda de la Sra. Luisa María Oliveira Roca y CONDENA a la empresa.";
        e = read(text, {{"Luisa María Oliveira Roca", "GIVEN_NAME"}});
        DLIB_TEST(found(e, text, pii_type::person, "Luisa María Oliveira Roca") && !covered(e, text, "CONDENA"));
        text = "Les clients, dont Société Quaevum SAS et Conseil de prud'hommes de La Flèche, et les équipes STD et "
               "DTCO sont informés.";
        e = read(text, {{"Société Quaevum SAS", "ORGANIZATION"}, {"DTCO", "ORGANIZATION"}});
        DLIB_TEST(!found(e, text, pii_type::organization, "Conseil") && !covered(e, text, "STD"));
        DLIB_TEST(found(e, text, pii_type::organization, "DTCO"));
        text = "ESTIMA la demanda interpuesta por el Sr. Dario LUQUE Y FONT y CONDENA a la empresa.";
        e = read(text, {{"Dario LUQUE Y FONT", "GIVEN_NAME"}, {"CONDENA", "FAMILY_NAME"}});
        DLIB_TEST(found(e, text, pii_type::person, "Dario LUQUE Y FONT") && !covered(e, text, "CONDENA"));
    }

// ----------------------------------------------------------------------------------------

    void test_business_documents()
    {
        /* Proposals, tender answers and audit reports: what a recognizer reads there in generic
           words stays readable, the acronym an organisation defines is found again, and the
           identifiers of public catalogues of vulnerabilities name nothing. */
        auto spans = std::make_shared<std::vector<pii_span>>();
        auto recognizer = [spans](const string& t) {
            std::vector<pii_span> out;
            for (const pii_span& v : *spans)
            {
                // The value is given by its text in begin's place: its offsets are found in t.
                const size_t at = t.find(v.label.substr(v.label.find('|') + 1));
                if (at == string::npos) continue;
                pii_span s = v;
                s.label = v.label.substr(0, v.label.find('|'));
                s.begin = at;
                s.end = at + v.label.size() - v.label.find('|') - 1;
                out.push_back(s);
            }
            std::sort(out.begin(), out.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
            return out;
        };
        pii_detector detector;
        detector.add_recognizer(recognizer, 0);
        auto value = [](const string& text, const string& label, double confidence) {
            pii_span s;
            s.label = label + "|" + text;
            s.confidence = confidence;
            return s;
        };
        auto read = [&](const string& text, const std::vector<pii_span>& values) {
            *spans = values;
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };

        // Generic words, whatever reads them; a given name, a civility, a head noun followed by
        // its complement or a known name keeps the value.
        string text = "Le Consortium mobilise un Architecte junior et la Direction Juridique et Technique. "
                      "Glossaire RACI : C I. La Politique de sauvegarde du ministère de la Pêche. M. Petit, Danone.";
        std::vector<pii_entity> e = read(text, {
            value("Le Consortium", "ORGANIZATION", 0.9), value("Architecte junior", "FAMILY_NAME", 0.9),
            value("Direction Juridique et Technique", "ORGANIZATION", 0.9), value("Glossaire", "GIVEN_NAME", 0.9),
            value("RACI", "FAMILY_NAME", 0.9), value("C I", "FAMILY_NAME", 0.9),
            value("Politique de sauvegarde", "ORGANIZATION", 0.9),
            value("ministère de la Pêche", "ORGANIZATION", 0.9), value("Petit", "FAMILY_NAME", 0.9) });
        DLIB_TEST(!covered(e, text, "Consortium") && !covered(e, text, "Architecte"));
        DLIB_TEST(!covered(e, text, "Direction Juridique") && !covered(e, text, "Glossaire"));
        DLIB_TEST(!covered(e, text, "RACI") && !covered(e, text, "C I") && !covered(e, text, "Politique"));
        DLIB_TEST(found(e, text, pii_type::organization, "ministère de la Pêche"));
        DLIB_TEST(found(e, text, pii_type::person, "Petit") && found(e, text, pii_type::organization, "Danone"));
        // A product and its maker, a code joined by underscores, an elision, the vocabulary of slides.
        text = "Google Chrome et Mozilla Firefox ; exigence REQ_SEC_B2 ; Jusqu'à 230 comptes VPN, allée du SIEM. "
               "Il travaille avec Google.";
        e = read(text, { value("Google Chrome", "ORGANIZATION", 0.9), value("Mozilla Firefox", "ORGANIZATION", 0.9),
                         value("REQ", "FAMILY_NAME", 0.9), value("Jusqu", "GIVEN_NAME", 0.9),
                         value("230 comptes VPN", "STREET_ADDRESS", 0.9),
                         value("allée du SIEM", "STREET_ADDRESS", 0.9) });
        DLIB_TEST(!covered(e, text, "Chrome") && !covered(e, text, "Firefox") && !covered(e, text, "REQ_"));
        DLIB_TEST(!covered(e, text, "Jusqu") && !covered(e, text, "230") && !covered(e, text, "allée du"));
        DLIB_TEST(found(e, text, pii_type::organization, "Google"));
        // A number before a common word in lower case, a word of slides after its number, the heading of a
        // numbered section and a verb that opens a bullet name nothing.
        text = "Le séminaire réunit 15 participants.\nPlanning 3\n2.1 Démarche\nLe chef de projet suit le planning.\n"
               "Pour M. Blanc : Vérifier les accès.";
        e = read(text, { value("15", "STREET_ADDRESS", 0.95), value("Planning 3", "STREET_ADDRESS", 0.8),
                         value("Démarche", "CITY", 0.7), value("Vérifier", "GIVEN_NAME", 0.9) });
        DLIB_TEST(!covered(e, text, "15") && !covered(e, text, "Planning") && !covered(e, text, "Démarche"));
        DLIB_TEST(!covered(e, text, "Vérifier") && found(e, text, pii_type::person, "Blanc"));
        // A street of common words keeps its street type on either side.
        text = "Adresse : 18 Mill Street, et 12 rue des Lilas.";
        e = read(text, { value("18 Mill Street", "STREET_ADDRESS", 0.9),
                         value("12 rue des Lilas", "STREET_ADDRESS", 0.9) });
        DLIB_TEST(found(e, text, pii_type::address, "18 Mill Street"));
        DLIB_TEST(found(e, text, pii_type::address, "12 rue des Lilas"));
        // Values labelled in a column of roles: initials name nobody, a short family name does.
        detection_context roles;
        roles.label = pii_type::person;
        *spans = {};
        detector.begin_document();
        text = "A, R";
        e = detector.detect(text, roles);
        DLIB_TEST(e.empty());
        text = "LE FUR";
        e = detector.detect(text, roles);
        DLIB_TEST(found(e, text, pii_type::person, "LE FUR"));

        // The acronym that an organisation defines is found again in capitals, the name taking its
        // words in lower case; a generic acronym is not learned.
        text = "Le ministère de la Pêche et des Eaux intérieures (MPEI) pilote le marché. Le "
               "Bénéficiaire suit la Procédure de Contrôle (PC).\nLe MPEI valide la PC.";
        e = read(text, { value("ministère de la Pêche et des Eaux", "ORGANIZATION", 0.9),
                         value("Procédure de Contrôle", "ORGANIZATION", 0.9) });
        DLIB_TEST(found(e, text, pii_type::organization, "ministère de la Pêche et des Eaux intérieures"));
        DLIB_TEST(found(e, text, pii_type::organization, "MPEI"));
        const size_t second = text.find("Le MPEI") + 3;
        bool again = false;
        for (const pii_entity& x : e)
            if (x.begin == second && x.end == second + 4 && x.type == pii_type::organization) again = true;
        DLIB_TEST(again);
        DLIB_TEST(!covered(e, text, "PC)") && !covered(e, text, "Procédure"));
        text = "La Société Nationale de Maintenance Industrielle (SNMI) répond.\nLa SNMI et la snmi signent.";
        e = read(text, { value("Société Nationale de Maintenance Industrielle", "ORGANIZATION", 0.9) });
        const size_t snmi = text.find("La SNMI") + 3;
        bool learned_acronym = false;
        for (const pii_entity& x : e)
            if (x.begin == snmi && x.end == snmi + 4 && x.type == pii_type::organization) learned_acronym = true;
        DLIB_TEST(learned_acronym && !covered(e, text, "snmi"));
        // An acronym the user allows is never reported, where it is defined included.
        {
            pii_detector allowing;
            allowing.add_recognizer(recognizer, 0);
            allowing.add_allowed_term("MPEI");
            text = "Le ministère de la Pêche et des Eaux intérieures (MPEI) pilote le marché.\nLe MPEI valide.";
            *spans = { value("ministère de la Pêche et des Eaux", "ORGANIZATION", 0.9) };
            allowing.begin_document();
            allowing.learn(text);
            e = allowing.detect(text);
            DLIB_TEST(found(e, text, pii_type::organization, "ministère de la Pêche et des Eaux intérieures"));
            DLIB_TEST(!covered(e, text, "MPEI"));
        }
        // The letters of an acronym and the initials of the name are compared whole, in any script.
        text = "Министерство внутренних дел (АБВ) сообщает. Позже АБВ ответило.";
        e = read(text, { value("Министерство внутренних дел", "ORGANIZATION", 0.9) });
        DLIB_TEST(!covered(e, text, "АБВ"));
        // The place that a name opened by a capital serves keeps it an organisation.
        text = "Le contrat avec Électricité de France et l'Opéra de Paris est signé.";
        e = read(text, { value("Électricité de France", "ORGANIZATION", 0.9),
                         value("Opéra de Paris", "ORGANIZATION", 0.9) });
        DLIB_TEST(found(e, text, pii_type::organization, "Électricité de France"));
        DLIB_TEST(found(e, text, pii_type::organization, "Opéra de Paris"));

        // Weak reads of the span model: a capitalized word joined to a person found is another person.
        text = "Our lawyers Martin and Dupont signed the letter.";
        e = read(text, { value("Martin", "GIVEN_NAME", 0.6), value("Dupont", "WEAK", 0.35) });
        DLIB_TEST(found(e, text, pii_type::person, "Martin") && found(e, text, pii_type::person, "Dupont"));
        e = read(text, { value("Martin", "GIVEN_NAME", 0.6) });
        DLIB_TEST(found(e, text, pii_type::person, "Martin") && !covered(e, text, "Dupont"));

        // Vulnerabilities: their public identifiers stay readable, a reference with a year does not.
        text = "Faille CVE-2014-0160 corrigée, ticket INC-2026-0147 clos.";
        e = read(text, {});
        DLIB_TEST(!covered(e, text, "CVE-2014") && found(e, text, pii_type::identifier, "INC-2026-0147"));
    }

    void test_technical_documents()
    {
        /* Security bulletins, architecture files, inventories and reports of scans: the versions of
           programs, their makers, the ports, the codes and the labels of their tables name nothing,
           while the host names of the domain of an e-mail address and the organisation named after
           its descriptor are found everywhere in the document. */
        auto spans = std::make_shared<std::vector<pii_span>>();
        auto recognizer = [spans](const string& t) {
            std::vector<pii_span> out;
            for (const pii_span& v : *spans)
            {
                const string wanted = v.label.substr(v.label.find('|') + 1);
                for (size_t at = t.find(wanted); at != string::npos; at = t.find(wanted, at + 1))
                {
                    pii_span s = v;
                    s.label = v.label.substr(0, v.label.find('|'));
                    s.begin = at;
                    s.end = at + wanted.size();
                    out.push_back(s);
                }
            }
            std::sort(out.begin(), out.end(), [](const pii_span& a, const pii_span& b) { return a.begin < b.begin; });
            return out;
        };
        pii_detector detector;
        detector.add_recognizer(recognizer, 0);
        auto value = [](const string& text, const string& label, double confidence) {
            pii_span s;
            s.label = label + "|" + text;
            s.confidence = confidence;
            return s;
        };
        auto read = [&](const string& text, const std::vector<pii_span>& values) {
            *spans = values;
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };
        auto dated = [](const std::vector<pii_entity>& e) {
            for (const pii_entity& x : e)
                if (x.type == pii_type::date) return true;
            return false;
        };

        // The versions of programs, whatever reads them as dates; a date after its word stays one.
        string text = "Le serveur exécute Tomcat 6.1.24. Mise à jour : Tomcat 6.2.19, 6.3.11 et 6.4.17 ; niveaux "
                      "< 5.4.18, tomcat@6.1.24 et v6.1.24 ; versions publiées 6.1.25, antérieure à 6.1.25, "
                      "IIS 7.5.19, Tomcat détecté 6.1.24.";
        DLIB_TEST_MSG(!dated(read(text, {})), text);
        text = "Cette version du 07.04.21 remplace la note du 7.4.21.";
        std::vector<pii_entity> e = read(text, {});
        DLIB_TEST(found(e, text, pii_type::date, "07.04.21") && found(e, text, pii_type::date, "7.4.21"));
        // Versions in table cells, after a revision of a package and after a German word of versions.
        for (const char* d : { "| Tomcat | 7.1.24 |", "| OpenSearch (moteur) | 1.1.19 |",
                               "Correctif : tomcat 4.0.17-3, 4.1.18.", "DIE KORRIGIERTEN VERSIONEN 6.4.28." })
            DLIB_TEST_MSG(!dated(read(d, {})), d);
        DLIB_TEST(dated(read("| Dupont | 3.4.85 |", {})));
        // The dates of a document, of its issue and of a birth, whatever word or capitals before them.
        for (const char* d : { "Fassung vom 01.02.24 des Vertrags.", "Stand 01.02.24 der Liste.",
                               "Data di rilascio 12.03.24 del documento.", "DDN 3.4.85 du patient.",
                               "NÉ LE 3.4.85 À BREST." })
            DLIB_TEST_MSG(dated(read(d, {})), d);
        // A version alone in its cell, written after its product elsewhere in the document.
        detector.begin_document();
        detector.learn("Tomcat 6.1.24 est déployé.");
        detector.learn("6.1.24");
        DLIB_TEST(!dated(detector.detect("6.1.24")));

        // The maker of the product that follows its name, and the department before "pour".
        text = "Le séminaire réunit 15 participants autour de Google Chrome et Mozilla Firefox.";
        e = read(text, { value("Google", "GIVEN_NAME", 0.6), value("Firefox", "FAMILY_NAME", 0.77) });
        DLIB_TEST(!covered(e, text, "Google") && !covered(e, text, "Firefox"));
        text = "La Direction Juridique et Technique pour Électricité de France valide le contrat.";
        e = read(text, { value("Électricité de France", "ORGANIZATION", 0.9) });
        DLIB_TEST(found(e, text, pii_type::organization, "Électricité de France"));
        DLIB_TEST(!covered(e, text, "Direction") && !covered(e, text, "Technique pour"));
        text = "La Fondation pour la Recherche Médicale finance le projet.";
        e = read(text, { value("Fondation pour la Recherche Médicale", "ORGANIZATION", 0.9) });
        DLIB_TEST(found(e, text, pii_type::organization, "Fondation pour la Recherche Médicale"));
        text = "The Centre for London published the report.";
        e = read(text, { value("Centre for London", "ORGANIZATION", 0.9) });
        DLIB_TEST(found(e, text, pii_type::organization, "Centre for London"));
        text = "L'Agenzia per l'Italia Digitale pubblica le linee guida.";
        e = read(text, { value("Agenzia per l'Italia Digitale", "ORGANIZATION", 0.9) });
        DLIB_TEST(found(e, text, pii_type::organization, "Agenzia per l'Italia Digitale"));
        text = "Das Bayerische Landesamt für Statistik veröffentlicht die Zahlen.";
        e = read(text, { value("Bayerische Landesamt für Statistik", "ORGANIZATION", 0.9) });
        DLIB_TEST(found(e, text, pii_type::organization, "Bayerische Landesamt für Statistik"));

        // The numbers of ports, codes, the labels of a table, a series of levels and the scheme of a package.
        text = "Flux W8 vers l'annuaire en LDAPS 636 et la messagerie en SMTP 25 ; lot H12.\n"
               "Échelon 3 Échelon 2 Échelon 1 Total\nRéférence NRV-DOC-ARC-2025-031 ; "
               "paquet pkg:maven/org.acme/tool@2.1, hôte kel-billing-api.\nBranch Tomcat Affected Fixed\n"
               "Utilitaires | Bastions";
        e = read(text, { value("W8", "CITY", 0.6), value("636", "CITY", 0.6), value("25", "CITY", 0.6),
                         value("H12", "CITY", 0.6), value("Échelon 3 Échelon 2 Échelon 1", "STREET_ADDRESS", 0.9),
                         value("DOC-ARC", "FAMILY_NAME", 0.7), value("pkg", "USERNAME", 0.7),
                         value("kel", "CITY", 0.6), value("Branch", "GIVEN_NAME", 0.57),
                         value("Tomcat Affected Fixed", "ORGANIZATION", 0.7), value("Utilitaires", "ORGANIZATION", 0.6),
                         value("Bastions", "ORGANIZATION", 0.6) });
        for (const char* v : { "W8", "636", "SMTP 25", "H12", "Échelon", "pkg:", "kel-", "Branch", "Affected",
                               "Utilitaires", "Bastions" })
            DLIB_TEST_MSG(!covered(e, text, v), v);
        // The reference stays one, which its label announces: no name inside it.
        DLIB_TEST(!found(e, text, pii_type::person, "DOC-ARC"));
        // A Spanish address with its number, floor and door stays one.
        text = "Domicilio: Calle de Sevilla, n.º 174, 3.º 1.ª, 41001 Sevilla.";
        e = read(text, { value("Calle de Sevilla, n.º 174, 3.º 1.ª", "STREET_ADDRESS", 0.9) });
        DLIB_TEST(found(e, text, pii_type::address, "Calle de Sevilla, n.º 174, 3.º 1.ª"));
        // A name damaged by a scanner and a postal code of Argentina keep their type.
        text = "Le courrier de Mi1let est arrivé ; envoi à C1425 Buenos Aires.";
        e = read(text, { value("Mi1let", "FAMILY_NAME", 0.8), value("C1425", "POSTAL_CODE", 0.8) });
        DLIB_TEST(found(e, text, pii_type::person, "Mi1let") && covered(e, text, "C1425"));

        // The host names of the domain of an e-mail address of the document, not those of a webmail.
        text = "Contact : jean.martin@acme-conseil.fr ; portail intranet.acme-conseil.fr/connexion et domaine "
               "acme-conseil.fr. Courrier de paul.roux@gmail.com, aide sur gmail.com et www.exemple.org.";
        e = read(text, {});
        DLIB_TEST(found(e, text, pii_type::url, "intranet.acme-conseil.fr/connexion"));
        DLIB_TEST(found(e, text, pii_type::url, "acme-conseil.fr"));
        DLIB_TEST(!covered(e, text, "aide sur gmail.com") && !covered(e, text, "www.exemple.org"));

        // The organisation named after its descriptor, found alone in the rest of the document.
        detector.begin_document();
        *spans = { value("société Novexa", "ORGANIZATION", 0.9) };
        detector.learn("La société Novexa signe le contrat.");
        *spans = {};
        detector.learn("Novexa livre le 3 mars.");
        text = "Novexa livre le 3 mars.";
        DLIB_TEST(found(detector.detect(text), text, pii_type::organization, "Novexa"));
        // A reference that starts with that name keeps no person inside it.
        text = "NOVEXA-DOC-ARC-2025-031 · version 2.4";
        e = detector.detect(text);
        DLIB_TEST(!found(e, text, pii_type::person, "DOC-ARC") && !covered(e, text, "DOC-ARC-2025"));

        // The glyphs of ligatures of a PDF font read as their letters, so that a name stays whole.
        const string pdf = pdf_file({
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>",
            pdf_stream("", "BT /F1 11 Tf 72 700 Td (Ra\\311aella Fiorini, \\310ole) Tj ET\n"),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding << /BaseEncoding /WinAnsiEncoding "
            "/Differences [200 /fi /ff] >> >>" });
        string read_back;
        for (const auto& page : layout_pdf_text(read_pdf_text(pdf)))
            for (const pdf_text_segment& s : page) read_back += s.text;
        DLIB_TEST_MSG(read_back.find("Raffaella Fiorini, fiole") != string::npos, read_back);
    }

    void test_imposed_terms()
    {
        pii_detector_options options;
        options.imposed_terms.push_back(pii_imposed_term{ "Projet Alpha", pii_type::organization, false });
        options.imposed_terms.push_back(pii_imposed_term{ "ACME", pii_type::organization, true });
        options.imposed_terms.push_back(pii_imposed_term{ "Contrat CT-2024-17", pii_type::identifier, false });
        options.imposed_terms.push_back(pii_imposed_term{ "dupont", pii_type::person, false });
        options.imposed_terms.push_back(pii_imposed_term{ "Société Générale", pii_type::organization, false });
        options.imposed_terms.push_back(pii_imposed_term{ "---", pii_type::organization, false });
        options.imposed_terms.push_back(pii_imposed_term{ "Kept", pii_type::none, false });
        pii_detector detector(options);
        DLIB_TEST(detector.imposed_term_count() == 5);

        // Case and accents ignored, separators of one or two characters, whole words only.
        string text = "Le projet alpha et le PROJET-ALPHA avancent ; Alphabet et Projet Alphas non.";
        std::vector<pii_entity> e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::organization, "projet alpha"));
        DLIB_TEST(found(e, text, pii_type::organization, "PROJET-ALPHA"));
        DLIB_TEST(!covered(e, text, "Alphabet") && !covered(e, text, "Projet Alphas"));
        text = "SOCIETE GENERALE et Societe Generale";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::organization, "SOCIETE GENERALE"));
        DLIB_TEST(found(e, text, pii_type::organization, "Societe Generale"));
        // Case sensitive: the case must match.
        text = "Acme n'est pas ACME, acme non plus.";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::organization, "ACME") && !covered(e, text, "Acme") &&
                  !covered(e, text, "acme"));
        // Not across a line, and not inside a tag of the pseudonymizer.
        text = "Projet\nAlpha [[ORG:Acme Projet Alpha#K7Q]]";
        e = detector.detect(text);
        DLIB_TEST(e.empty());
        // A value that overlaps a term keeps its parts outside it; a term inside an e-mail address stays in it.
        text = "Signé dans le cadre du contrat ct-2024-17 par Jean Dupont, jean.dupont@exemple.fr.";
        e = detector.detect(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "contrat ct-2024-17"));
        DLIB_TEST(found(e, text, pii_type::person, "Dupont"));
        DLIB_TEST(found(e, text, pii_type::email, "jean.dupont@exemple.fr"));
        DLIB_TEST(!found(e, text, pii_type::person, "Jean Dupont"));
        // Above the rules and the recognizers of every type.
        pii_detector learned(options);
        learned.add_recognizer([](const string& t) {
            std::vector<pii_span> out;
            const size_t at = t.find("Projet Alpha Industries");
            if (at != string::npos)
            {
                pii_span s;
                s.begin = at;
                s.end = at + string("Projet Alpha Industries").size();
                s.label = "ORGANIZATION";
                s.confidence = 0.95;
                out.push_back(s);
            }
            return out;
        });
        text = "La société Projet Alpha Industries, au 12 rue de la Paix, 75002 Paris.";
        learned.begin_document();
        e = learned.detect(text);
        DLIB_TEST(found(e, text, pii_type::organization, "Projet Alpha"));
        DLIB_TEST(found(e, text, pii_type::organization, "Industries"));
        // The terms of a type left out are ignored.
        pii_detector_options no_org = options;
        no_org.types.erase(std::remove(no_org.types.begin(), no_org.types.end(), pii_type::organization),
                           no_org.types.end());
        pii_detector partial(no_org);
        DLIB_TEST(partial.imposed_term_count() == 2);
        // A long text, read in pieces: each occurrence, across the pieces too.
        string longer;
        while (longer.size() < 200000) longer += "Le projet alpha avance. Rien d'autre ici. ";
        e = detector.detect(longer);
        size_t terms = 0;
        for (const pii_entity& x : e)
            if (x.type == pii_type::organization && longer.substr(x.begin, x.end - x.begin) == "projet alpha") ++terms;
        DLIB_TEST(terms == longer.size() / string("Le projet alpha avance. Rien d'autre ici. ").size());
        // Openings of tags without their end hide nothing: the term after them is found.
        string openings;
        while (openings.size() < 100000) openings += "[[ORG:a ";
        openings += "projet alpha";
        e = detector.detect(openings);
        DLIB_TEST(!e.empty() && e.back().type == pii_type::organization &&
                  openings.substr(e.back().begin, e.back().end - e.back().begin) == "projet alpha");
    }

    void test_ascii_windows_1252()
    {
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x41')));
        pii_detector detector;
        const std::vector<pii_imposed_column> columns = { { "Nom", pii_type::person } };
        // A Windows-1252 file whose accented letters all stand in an imposed column: its protected form is ASCII.
        const string csv = "Nom;Ville\nH\xE9l\xE8ne;Rennes\nJos\xE9;Lyon\n";
        document_anonymizer a(p, detector);
        a.set_imposed_columns(columns);
        document_report r;
        const string out = a.anonymize(csv, document_format::delimited, r, "a.csv");
        DLIB_TEST(r.encoding == "windows-1252" && !a.ascii_windows_1252());
        DLIB_TEST(std::all_of(out.begin(), out.end(), [](char c) { return (c & 0x80) == 0; }));
        // Nothing tells its encoding: UTF-8 by default, Windows-1252 when the caller knows it.
        document_anonymizer b(p, detector);
        b.set_imposed_columns(columns);
        b.set_text_substitute_resolver(resolver_of(r.substitutions));
        document_report r1;
        DLIB_TEST(b.restore(out, document_format::delimited, r1, "a.csv") ==
                  "Nom;Ville\nH\xC3\xA9l\xC3\xA8ne;Rennes\nJos\xC3\xA9;Lyon\n" && r1.encoding == "utf-8");
        b.set_ascii_windows_1252(true);
        document_report r2;
        DLIB_TEST(b.restore(out, document_format::delimited, r2, "a.csv") == csv && r2.encoding == "windows-1252");
        DLIB_TEST(b.restore(out, document_format::text, r2, "a.csv") == csv);
        // A letter that Windows-1252 cannot write keeps the text in UTF-8, as its original was.
        const string polish = "Nom;Ville\n\xC5\x81ukasz;Rennes\n";
        document_report rp;
        const string pout = a.anonymize(polish, document_format::delimited, rp, "a.csv");
        DLIB_TEST(rp.encoding == "utf-8" &&
                  std::all_of(pout.begin(), pout.end(), [](char c) { return (c & 0x80) == 0; }));
        document_anonymizer c(p, detector);
        c.set_imposed_columns(columns);
        c.set_text_substitute_resolver(resolver_of(rp.substitutions));
        c.set_ascii_windows_1252(true);
        document_report rc;
        DLIB_TEST(c.restore(pout, document_format::delimited, rc, "a.csv") == polish && rc.encoding == "utf-8");
        // A text that holds other bytes keeps its own encoding, and the other formats report none.
        const string utf8_text = "Caf\xC3\xA9 de M. Dupont\n";
        document_report ua, ur;
        DLIB_TEST(c.restore(a.anonymize(utf8_text, document_format::text, ua), document_format::text, ur) ==
                  utf8_text && ua.encoding == "utf-8" && ur.encoding == "utf-8");
    }

    void test_health_and_exports()
    {
        pii_detector detector;
        auto entities = [&](const string& text) {
            detector.begin_document();
            detector.learn(text);
            return detector.detect(text);
        };
        // Nationalities after the cues of Italian, Spanish and German, a German adjective with its ending.
        string text = "Su socio, Andrés Ocaña Prieto, de nacionalidad mexicana, abre una oficina. "
                      "Il socio Marco Rossi, di nazionalità messicana, segue la sede. "
                      "Ihr Disponent Mehmet Yılmaz, türkischer Staatsangehöriger, plant die Touren; "
                      "Frau Nowak lebt mit polnischer Staatsangehörigkeit in Köln.";
        std::vector<pii_entity> e = entities(text);
        for (const char* v : { "mexicana", "messicana", "türkischer", "polnischer" })
            DLIB_TEST_MSG(found(e, text, pii_type::nationality, v), v);

        // The numbers of doctors after their labels and after the board that holds them.
        text = "Dr. med. Anna Vogt, Fachärztin für Allgemeinmedizin\nLANR 384712903 · BSNR 563910482\n"
               "Dr Eleanor Fitzgerald, General Practitioner, GMC 7419286\n"
               "Iscrizione Ordine dei Medici di Bologna n. 15873\n"
               "Colegio Oficial de Médicos de Valladolid, n.º de colegiado 474712345\n"
               "Inscrite à l'Ordre des médecins de la Gironde sous le n° 33/12345\n"
               "Ordine del giorno n. 3: bilancio.";
        e = entities(text);
        for (const char* v : { "384712903", "563910482", "7419286", "15873", "474712345", "33/12345" })
            DLIB_TEST_MSG(found(e, text, pii_type::identifier, v), v);
        DLIB_TEST(!covered(e, text, "3: bilancio"));

        // The Spanish social security number, its groups separated or after its label, its control checked.
        text = "N.º Seguridad Social\n47/01838264/00\nNUSS: 281234567840\nOtro número: 47/01838264/41";
        e = entities(text);
        DLIB_TEST(found(e, text, pii_type::identifier, "47/01838264/00"));
        DLIB_TEST(found(e, text, pii_type::identifier, "281234567840"));
        DLIB_TEST(!found(e, text, pii_type::identifier, "47/01838264/41"));
        // An NHS number after "patient number", its check verified.
        text = "Her NHS letter lay on the bedside table, patient number 745 275 9125.";
        DLIB_TEST(found(entities(text), text, pii_type::identifier, "745 275 9125"));

        // The end of validity of a card is a month with its year, after its words only.
        text = "Card number 4012 8888 8888 1881, expiry date, 07/29. Valid thru 11/27. Ablaufdatum: 03/28.";
        e = entities(text);
        for (const char* v : { "07/29", "11/27", "03/28" })
            DLIB_TEST_MSG(found(e, text, pii_type::date, v), v);
        text = "Room 07/29 of building 11/27.";
        e = entities(text);
        DLIB_TEST(!covered(e, text, "07/29") && !covered(e, text, "11/27"));
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x43')));
        const string moved = p.substitute(pii_type::date, "07/29");
        DLIB_TEST(moved.size() == 5 && moved[2] == '/' && moved != "07/29");
        DLIB_TEST(p.restore(p.tag(pii_type::date, "07/29")) == "07/29");

        // What a recognizer reads in a unit, a literal of code or a code of capitals names nothing.
        pii_detector read;
        read.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            auto add = [&](const string& value, const string& label, size_t from) {
                const size_t p = t.find(value, from);
                if (p == string::npos) return;
                pii_span s;
                s.begin = p;
                s.end = p + value.size();
                s.label = label;
                s.confidence = 0.9;
                spans.push_back(s);
            };
            add("Prozent", "CITY", 0);
            add("False", "CITY", 0);
            add("REQ_SEC", "USERNAME", 0);
            add("Leipzig", "CITY", 0);
            return spans;
        });
        text = "Das Wachstum liegt bei 1,3 Prozent. joueurs = {'O': ('Ordinateur', False)}\n"
               "| Anforderung | REQ_SEC_B2 |\nPraxis: Kurt-Eisner-Straße 4, 04275 Leipzig";
        read.begin_document();
        read.learn(text);
        e = read.detect(text);
        DLIB_TEST(!covered(e, text, "Prozent") && !covered(e, text, "False") && !covered(e, text, "REQ_SEC"));
        DLIB_TEST(found(e, text, pii_type::location, "Leipzig") || found(e, text, pii_type::location, "04275 Leipzig"));

        // A file without header: its column of sexes is typed by its values, protected and restored.
        const string csv = "giulia.romano@example.org;Giulia;Romano;F;1987-03-12;Bergamo\n"
                           "marco.ferrara@example.net;Marco;Ferrara;M;1971-06-05;Perugia\n"
                           "sara.galli@example.net;Sara;Galli;F;1986-09-11;Napoli\n"
                           "luca.conti@example.com;Luca;Conti;M;1990-01-20;Roma\n";
        document_anonymizer a(p, detector);
        document_report r;
        const string out = a.anonymize(csv, document_format::delimited, r, "export.csv");
        DLIB_TEST(r.entities["GENDER"] == 4);
        document_report rr;
        DLIB_TEST(a.restore(out, document_format::delimited, rr, "export.csv") == csv);
        // A column of letters of no single usage, or of one letter, stays as it is.
        const string sizes = "Giulia;Romano;M;S\nMarco;Ferrara;M;XL\nSara;Galli;M;L\nLuca;Conti;M;M\n";
        document_report rs;
        a.anonymize(sizes, document_format::delimited, rs, "sizes.csv");
        DLIB_TEST(rs.entities.count("GENDER") == 0);

        /* Without a header, the cells of a column of names or of towns that the rows did not teach
           take the type of the others; a column of codes stays as it is. */
        pii_detector rows;
        rows.add_recognizer([](const string& t) {
            std::vector<pii_span> spans;
            for (const char* v : { "Romano", "Ferrara", "Conti", "Bergamo", "Perugia", "Roma" })
            {
                const string value = v;
                for (size_t p = t.find(value); p != string::npos; p = t.find(value, p + 1))
                {
                    if ((p > 0 && std::isalpha(static_cast<unsigned char>(t[p - 1]))) ||
                        (p + value.size() < t.size() && std::isalpha(static_cast<unsigned char>(t[p + value.size()]))))
                        continue;
                    pii_span s;
                    s.begin = p;
                    s.end = p + value.size();
                    s.label = value == "Bergamo" || value == "Perugia" || value == "Roma" ? "CITY" : "FAMILY_NAME";
                    s.confidence = 0.95;
                    spans.push_back(s);
                }
            }
            return spans;
        });
        const string export_rows = "giulia@example.org;Romano;Bergamo;B2C\nmarco@example.org;Ferrara;Perugia;B2B\n"
                                   "sara@example.org;Galli;Napoli;B2C\nluca@example.org;Conti;Roma;B2B\n";
        document_anonymizer b(p, rows);
        document_report rb;
        const string bout = b.anonymize(export_rows, document_format::delimited, rb, "export.csv");
        DLIB_TEST(bout.find("Galli") == string::npos && bout.find("Napoli") == string::npos);
        DLIB_TEST(bout.find(";B2C\n") != string::npos && bout.find(";B2B\n") != string::npos);
        document_report rbr;
        DLIB_TEST(b.restore(bout, document_format::delimited, rbr, "export.csv") == export_rows);
        // A given name that is also a word of code stays a name in its column: "Elif".
        const string named = "Anrede;Vorname;Nachname\nFrau;Elif;Weber\nHerr;Jonas;Krause\n";
        document_report rn;
        DLIB_TEST(a.anonymize(named, document_format::delimited, rn, "kunden.csv").find("Elif") == string::npos);
    }

    std::vector<string> lines_of(const string& text)
    {
        std::vector<string> lines;
        std::istringstream in(text);
        string line;
        while (std::getline(in, line)) lines.push_back(line);
        return lines;
    }

    void test_imposed_columns()
    {
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x41')));
        pii_detector detector;
        const std::vector<pii_imposed_column> columns = {
            { "Nom", pii_type::person }, { "prenom", pii_type::person }, { " E-mail ", pii_type::email },
            { "Matricule", pii_type::identifier }, { "Sexe", pii_type::gender }, { "  ", pii_type::person },
            { "NOM", pii_type::location }, { "Ville", pii_type::none } };
        document_anonymizer a(p, detector);
        a.set_imposed_columns(columns);
        DLIB_TEST(a.imposed_columns().size() == 5);

        // The header stays; each value below an imposed name takes its substitute, without detection nor tag.
        const string header = "PRENOM;Nom;E-MAIL;Ville;Commentaire;Matricule;Sexe";
        const string csv = header + "\n"
                           "Jean;Dupont;jean.dupont@exemple.fr;Rennes;Rappeler M. Dupont lundi;582134;M\n"
                           "Hélène;Martin;h.martin@exemple.fr;Lyon;RAS;4521;F\n"
                           "Ali; Ben Salah ;;Paris;Client de Jean Dupont depuis 2019;77;M\n";
        document_report report;
        const string out = a.anonymize(csv, document_format::delimited, report, "clients.csv");
        const std::vector<string> rows = lines_of(out);
        DLIB_TEST_MSG(rows.size() == 4 && rows[0] == header, out);
        DLIB_TEST_MSG(rows[1].find(p.substitute(pii_type::person, "Jean") + ";" +
                                   p.substitute(pii_type::person, "Dupont") + ";" +
                                   p.substitute(pii_type::email, "jean.dupont@exemple.fr") + ";") == 0, out);
        DLIB_TEST_MSG(rows[1].find(";" + p.substitute(pii_type::identifier, "582134") + ";") != string::npos, out);
        // The other cells are read as before, the names of the columns found there too.
        DLIB_TEST_MSG(rows[1].find(p.tag(pii_type::person, "Dupont")) != string::npos, out);
        DLIB_TEST_MSG(rows[3].find("; " + p.substitute(pii_type::person, "Ben Salah") + " ;;") != string::npos, out);
        DLIB_TEST_MSG(out.find("Dupont") == string::npos && out.find("Martin") == string::npos &&
                      out.find("dupont@") == string::npos && out.find("4521") == string::npos, out);
        DLIB_TEST(report.substitutions.size() == 14 && report.entities["GENDER"] == 3 && report.entities["ID"] == 3);
        for (const document_substitution& s : report.substitutions)
            DLIB_TEST_MSG(s.text && out.find(s.substitute) != string::npos && s.tag.compare(0, 2, "[[") == 0,
                          s.substitute);
        // A sex takes another substitute at each place.
        DLIB_TEST(rows[1].substr(rows[1].rfind(';')) != rows[3].substr(rows[3].rfind(';')));

        // The substitutes come back from their list, the tags as always.
        document_anonymizer b(p, detector);
        b.set_imposed_columns(columns);
        document_report none;
        const string tags_only = b.restore(out, document_format::delimited, none, "clients.csv");
        DLIB_TEST(none.substitutes_restored == 0 && none.tags_restored > 0);
        DLIB_TEST(tags_only.find(p.substitute(pii_type::person, "Martin")) != string::npos);
        b.set_text_substitute_resolver(resolver_of(report.substitutions));
        document_report back;
        DLIB_TEST(b.restore(out, document_format::delimited, back, "clients.csv") == csv);
        DLIB_TEST(back.substitutes_restored == 14);

        // Protected again with its substitutes known, the file keeps them as they are.
        document_anonymizer again(p, detector);
        again.set_imposed_columns(columns);
        again.set_text_substitute_resolver(resolver_of(report.substitutions));
        document_report second;
        DLIB_TEST(again.anonymize(out, document_format::delimited, second, "clients.csv") == out);
        DLIB_TEST(second.substitutions.empty());

        // Without tags, the other values take substitutes too, all given back.
        document_anonymizer untagged(p, detector);
        untagged.set_imposed_columns(columns);
        untagged.set_untagged_text(true);
        document_report ureport;
        const string uout = untagged.anonymize(csv, document_format::delimited, ureport, "clients.csv");
        DLIB_TEST_MSG(uout.find("[[") == string::npos && uout.find("Rennes") == string::npos, uout);
        document_anonymizer uback(p, detector);
        uback.set_imposed_columns(columns);
        uback.set_text_substitute_resolver(resolver_of(ureport.substitutions));
        document_report ur;
        DLIB_TEST(uback.restore(uout, document_format::delimited, ur, "clients.csv") == csv);

        // A title above the header, the "sep=" line of Excel, a text of another name.
        const string titled = "Export du 12/03/2024\n" + csv;
        for (const string& name : { string("clients.csv"), string("clients.txt"), string("") })
        {
            document_report r;
            const string t = a.anonymize(titled, document_format::text, r, name);
            DLIB_TEST_MSG(r.format == "csv" && lines_of(t)[1] == header && t.find("Martin") == string::npos, name);
            DLIB_TEST_MSG(t.find(p.substitute(pii_type::person, "Martin") + ";") != string::npos, name);
            document_anonymizer c(p, detector);
            c.set_imposed_columns(columns);
            c.set_text_substitute_resolver(resolver_of(r.substitutions));
            document_report rb;
            DLIB_TEST_MSG(c.restore(t, document_format::text, rb, name) == titled, name);
        }
        document_report excel;
        const string sep = a.anonymize("sep=;\r\n" + csv, document_format::delimited, excel, "export.csv");
        DLIB_TEST_MSG(sep.compare(0, 7 + header.size(), "sep=;\r\n" + header) == 0 && excel.substitutions.size() == 14,
                      sep);
        // A text that only names a column in a sentence stays a text.
        const string letter = "Nom: Dupont\nUne lettre, Nom et adresse.\nRien de plus.\n";
        document_report lr;
        a.anonymize(letter, document_format::text, lr, "lettre.txt");
        DLIB_TEST(lr.format == "txt" && lr.substitutions.empty());
        DLIB_TEST(privacy_impl::imposed_delimiter(letter, { "nom" }, "lettre.txt") == 0);
        DLIB_TEST(privacy_impl::imposed_delimiter("Nom;Ville\nDupont;Rennes\n", { "nom" }, "a.txt") == 0);
        DLIB_TEST(privacy_impl::imposed_delimiter("Nom;Ville\nDupont;Rennes\nMartin;Lyon\n", { "nom" }, "a.txt") ==
                  ';');
        DLIB_TEST(privacy_impl::imposed_delimiter("\"Nom\",\"Ville, pays\"\n", { "ville, pays" }, "a.csv") == ',');
        DLIB_TEST(privacy_impl::imposed_delimiter("Id\tNom\n", { "nom" }, "a.tsv") == '\t');
        DLIB_TEST(privacy_impl::imposed_delimiter("Id|Nom|Ville\n", { "prenom" }, "a.csv") == 0);

        // A column alone, the values holding commas and semicolons.
        const string single = "Nom\nDupont\nMartin, Jean\n\"Durand; Paul\"\n";
        document_report sr;
        const string sout = a.anonymize(single, document_format::delimited, sr, "noms.csv");
        DLIB_TEST_MSG(sout == "Nom\n" + p.substitute(pii_type::person, "Dupont") + "\n" +
                      p.substitute(pii_type::person, "Martin, Jean") + "\n\"" +
                      p.substitute(pii_type::person, "Durand; Paul") + "\"\n", sout);

        // Under names that are no labels, the values of the columns are found in the other cells too.
        const string partners = "Interlocuteur;Partenaire;Note\n"
                                "Kowalczyk;Norvexa;Point avec Kowalczyk sur le dossier Norvexa demain\n";
        pii_detector fresh;
        document_anonymizer pa(p, fresh);
        pa.set_imposed_columns({ { "Interlocuteur", pii_type::person }, { "Partenaire", pii_type::organization } });
        document_report par;
        const string partners_out = pa.anonymize(partners, document_format::delimited, par, "a.csv");
        DLIB_TEST_MSG(partners_out.find(p.substitute(pii_type::person, "Kowalczyk") + ";" +
                                        p.substitute(pii_type::organization, "Norvexa") + ";Point avec " +
                                        p.tag(pii_type::person, "Kowalczyk")) != string::npos, partners_out);
        DLIB_TEST_MSG(partners_out.find(p.tag(pii_type::organization, "Norvexa") + " demain") != string::npos,
                      partners_out);
        DLIB_TEST(fresh.learned_name_count() == 1);

        // A cell written without quotes keeps the quotes it holds as they are, with or without imposed columns.
        const string literal = "Nom;Remarque\nDupont;Ecran 5\" pour M. Dupont\n";
        for (int with = 0; with < 2; ++with)
        {
            document_anonymizer q(p, detector);
            if (with) q.set_imposed_columns(columns);
            document_report qr;
            const string qout = q.anonymize(literal, document_format::delimited, qr, "a.csv");
            DLIB_TEST_MSG(qout.find(";Ecran 5\" pour ") != string::npos && qout.find("Dupont") == string::npos, qout);
            document_anonymizer qb(p, detector);
            if (with) qb.set_imposed_columns(columns);
            qb.set_text_substitute_resolver(resolver_of(qr.substitutions));
            document_report qback;
            DLIB_TEST_MSG(qb.restore(qout, document_format::delimited, qback, "a.csv") == literal, qout);
        }

        // A tag in a column stands for its value; a header below the first 16 rows is no header.
        const string tagged = "Nom;Ville\n" + p.tag(pii_type::organization, "Dupont") + ";Rennes\n";
        document_report tr;
        DLIB_TEST(lines_of(a.anonymize(tagged, document_format::delimited, tr, "a.csv"))[1].find(
                  p.substitute(pii_type::person, "Dupont") + ";") == 0);
        string deep;
        for (int i = 0; i < 16; ++i) deep += "Ligne " + std::to_string(i) + "\n";
        document_report dr;
        const string dout = a.anonymize(deep + csv, document_format::delimited, dr, "a.csv");
        DLIB_TEST_MSG(dr.substitutions.empty() && dout.find(p.tag(pii_type::person, "Martin")) != string::npos, dout);

        // A date kept in clear by the policy stays in clear in its column.
        pseudonymizer_options keep;
        keep.dates = date_policy::keep;
        const pseudonymizer pk(pseudonymization_key::from_bytes(string(32, '\x41')), keep);
        document_anonymizer k(pk, detector);
        k.set_imposed_columns({ { "Naissance", pii_type::date }, { "Nom", pii_type::person } });
        document_report kr;
        const string kout = k.anonymize("Nom;Naissance\nDupont;12/03/1985\n", document_format::delimited, kr, "a.csv");
        DLIB_TEST_MSG(kout.find(";12/03/1985\n") != string::npos && kr.substitutions.size() == 1, kout);
        // Without imposed columns, nothing changes.
        document_anonymizer plain(p, detector);
        document_report pr;
        const string pout = plain.anonymize(csv, document_format::delimited, pr, "clients.csv");
        DLIB_TEST(pr.substitutions.empty() && pout.find(p.tag(pii_type::person, "Martin")) != string::npos);
        DLIB_TEST(lines_of(pout)[0] == header);
    }

    void test_ip_columns()
    {
        const pseudonymizer p(pseudonymization_key::from_bytes(string(32, '\x41')));
        pii_detector detector;
        // An IP column takes the IPv4 and the IPv6 addresses alike, whichever of the two types names it.
        const string csv = "host;ip_address\n"
                           "web;192.168.1.5\n"
                           "db;2001:db8::1\n"
                           "net;10.0.0.0/8\n"
                           "api;[2001:db8::1]:443\n"
                           "link;fe80::1%eth0\n"
                           "srv;192.0.2.1:8080\n"
                           "v6net;2001:db8::/32\n"
                           "long;2001:0db8:0000:0000:0000:0000:0000:0001\n"
                           "upper;[2001:0DB8::1]:443\n"
                           "zeros;fe80::0001%eth0\n"
                           "wide;2001:0db8::/32\n"
                           "none;N/A\n";
        const std::size_t values = 12;
        std::vector<string> first;
        for (pii_type column_type : { pii_type::ipv4, pii_type::ipv6 })
        {
            document_anonymizer a(p, detector);
            a.set_imposed_columns({ { "ip_address", column_type } });
            document_report r;
            const string out = a.anonymize(csv, document_format::delimited, r, "hosts.csv");
            const std::vector<string> rows = lines_of(out);
            DLIB_TEST_MSG(rows.size() == values + 1 && rows[0] == "host;ip_address", out);
            if (rows.size() != values + 1) continue;
            // Each address takes the substitute of its form; its prefix length, brackets, port and zone stay.
            const string v4 = p.substitute(pii_type::ipv4, "192.168.1.5");
            const string v6 = p.substitute(pii_type::ipv6, "2001:db8::1");
            DLIB_TEST_MSG(rows[1] == "web;" + v4 && rows[2] == "db;" + v6, out);
            DLIB_TEST_MSG(rows[3] == "net;" + p.substitute(pii_type::ipv4, "10.0.0.0") + "/8", out);
            DLIB_TEST_MSG(rows[4] == "api;[" + v6 + "]:443", out);
            DLIB_TEST_MSG(rows[5] == "link;" + p.substitute(pii_type::ipv6, "fe80::1") + "%eth0", out);
            DLIB_TEST_MSG(rows[6] == "srv;" + p.substitute(pii_type::ipv4, "192.0.2.1") + ":8080", out);
            DLIB_TEST_MSG(rows[7] == "v6net;" + p.substitute(pii_type::ipv6, "2001:db8::") + "/32", out);
            // An address written otherwise than in its canonical form takes an opaque code, whole with what
            // surrounds it, so that its restoration finds it.
            DLIB_TEST_MSG(rows[8] == "long;" + p.substitute(pii_type::ipv6, "2001:0db8:0000:0000:0000:0000:0000:0001"),
                          out);
            DLIB_TEST_MSG(rows[9] == "upper;" + p.substitute(pii_type::ipv6, "[2001:0DB8::1]:443"), out);
            DLIB_TEST_MSG(rows[10] == "zeros;" + p.substitute(pii_type::ipv6, "fe80::0001%eth0"), out);
            DLIB_TEST_MSG(rows[11] == "wide;" + p.substitute(pii_type::ipv6, "2001:0db8::/32"), out);
            DLIB_TEST(r.entities["IPV4"] + r.entities["IPV6"] == values);
            DLIB_TEST(out.find("192.168") == string::npos && out.find("2001:db8:") == string::npos &&
                      out.find("2001:0db8") == string::npos && out.find("0DB8") == string::npos &&
                      out.find("N/A") == string::npos);
            // Both types of column give the same addresses the same substitutes.
            if (first.empty()) first = rows;
            else for (std::size_t i = 0; i < values; ++i) DLIB_TEST_MSG(rows[i] == first[i], rows[i]);

            // The record gives every value back as it was written, and a protected file protected again stays.
            document_anonymizer b(p, detector);
            b.set_imposed_columns({ { "ip_address", column_type } });
            b.set_text_substitute_resolver(resolver_of(r.substitutions));
            document_report back;
            DLIB_TEST_MSG(b.restore(out, document_format::delimited, back, "hosts.csv") == csv, out);
            DLIB_TEST(back.substitutes_restored == values);
            document_report again;
            DLIB_TEST(b.anonymize(out, document_format::delimited, again, "hosts.csv") == out);
            DLIB_TEST(again.substitutions.empty());
        }
    }

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
        pii_span_model_small other_size;
        string message;
        try { deserialize(other_size, sin2); }
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
            test_untagged_values();
            test_untagged_text();
            print_spinner();
            test_pdf_document();
            test_pdf_layout();
            test_image_describer();
            test_pdf_wrapped_cell();
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
            test_short_texts();
            print_spinner();
            test_attribute_coverage();
            print_spinner();
            test_foreign_postcodes();
            print_spinner();
            test_foreign_identifiers();
            print_spinner();
            test_detector_boundaries();
            print_spinner();
            test_five_languages();
            print_spinner();
            test_business_documents();
            print_spinner();
            test_technical_documents();
            print_spinner();
            test_imposed_terms();
            test_imposed_columns();
            test_ip_columns();
            test_ascii_windows_1252();
            test_health_and_exports();
            print_spinner();
            test_dataset();
            print_spinner();
            test_sequence_tagger();
            print_spinner();
            test_span_model();
        }
    } a;

}
