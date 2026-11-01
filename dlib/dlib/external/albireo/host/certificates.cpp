// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "certificates.h"

#include <cstring>
#include <ctime>
#include <vector>

#include <mbedtls/asn1write.h>
#include <mbedtls/ctr_drbg.h>
#include <mbedtls/ecp.h>
#include <mbedtls/entropy.h>
#include <mbedtls/oid.h>
#include <mbedtls/pk.h>
#include <mbedtls/platform_util.h>
#include <mbedtls/sha256.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincrypt.h>
#include <dpapi.h>
#include <cwchar>
#endif

#include "../common/log.h"
#include "../common/platform.h"

namespace albireo
{

    namespace
    {
        // The validity of both certificates, and the time before their end when they are renewed.
        const long validity_days = 397;
        const long long renewal_seconds = 86400LL * 30;
        const char* const authority_name = "CN=Albireo Local Authority,O=Cydral Technology,C=FR";

        struct random_source
        {
            mbedtls_entropy_context entropy;
            mbedtls_ctr_drbg_context drbg;
            bool ok = false;

            random_source()
            {
                mbedtls_entropy_init(&entropy);
                mbedtls_ctr_drbg_init(&drbg);
                static const char personal[] = "albireo-certificates";
                ok = mbedtls_ctr_drbg_seed(&drbg, mbedtls_entropy_func, &entropy,
                    reinterpret_cast<const unsigned char*>(personal), sizeof(personal) - 1) == 0;
            }
            ~random_source()
            {
                mbedtls_ctr_drbg_free(&drbg);
                mbedtls_entropy_free(&entropy);
            }
        };

        // A key of mbedtls, freed with its secret erased (mbedtls_pk_free) on every path.
        struct pk_key
        {
            mbedtls_pk_context ctx;
            pk_key() { mbedtls_pk_init(&ctx); }
            ~pk_key() { mbedtls_pk_free(&ctx); }
            pk_key(const pk_key&) = delete;
            pk_key& operator=(const pk_key&) = delete;
        };

        std::string time_string(std::time_t t)
        {
            std::tm tm;
#if defined(_WIN32)
            gmtime_s(&tm, &t);
#else
            gmtime_r(&t, &tm);
#endif
            char buf[16];
            std::strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tm);
            return buf;
        }

        bool generate_key(mbedtls_pk_context& key, random_source& rng)
        {
            if (mbedtls_pk_setup(&key, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY)) != 0) return false;
            return mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(key), mbedtls_ctr_drbg_random,
                &rng.drbg) == 0;
        }

        bool key_pem(mbedtls_pk_context& key, std::string& pem)
        {
            std::vector<unsigned char> buf(4096);
            const bool ok = mbedtls_pk_write_key_pem(&key, buf.data(), buf.size()) == 0;
            if (ok) pem = reinterpret_cast<const char*>(buf.data());
            mbedtls_platform_zeroize(buf.data(), buf.size());
            return ok;
        }

        // Seconds until a certificate expires, negative when it cannot be read.
        long long seconds_left(const std::string& path)
        {
            mbedtls_x509_crt crt;
            mbedtls_x509_crt_init(&crt);
            std::string pem;
            long long left = -1;
            if (read_file(path, pem) &&
                mbedtls_x509_crt_parse(&crt, reinterpret_cast<const unsigned char*>(pem.c_str()), pem.size() + 1) == 0)
            {
                std::tm tm = {};
                tm.tm_year = crt.valid_to.year - 1900;
                tm.tm_mon = crt.valid_to.mon - 1;
                tm.tm_mday = crt.valid_to.day;
                tm.tm_hour = crt.valid_to.hour;
                tm.tm_min = crt.valid_to.min;
                tm.tm_sec = crt.valid_to.sec;
#if defined(_WIN32)
                const std::time_t expiry = _mkgmtime(&tm);
#else
                const std::time_t expiry = timegm(&tm);
#endif
                left = static_cast<long long>(expiry - std::time(nullptr));
            }
            mbedtls_x509_crt_free(&crt);
            return left;
        }

        // Whether a private key in PEM form is the one of the certificate of a PEM file.
        bool key_matches(const std::string& key, const std::string& cert_path, random_source& rng)
        {
            std::string pem;
            if (!read_file(cert_path, pem)) return false;
            mbedtls_x509_crt crt;
            mbedtls_x509_crt_init(&crt);
            pk_key k;
            const bool ok =
                mbedtls_x509_crt_parse(&crt, reinterpret_cast<const unsigned char*>(pem.c_str()), pem.size() + 1) == 0
                && mbedtls_pk_parse_key(&k.ctx, reinterpret_cast<const unsigned char*>(key.c_str()), key.size() + 1,
                    nullptr, 0, mbedtls_ctr_drbg_random, &rng.drbg) == 0
                && mbedtls_pk_check_pair(&crt.pk, &k.ctx, mbedtls_ctr_drbg_random, &rng.drbg) == 0;
            mbedtls_x509_crt_free(&crt);
            return ok;
        }

        void der_append(std::vector<unsigned char>& out, unsigned char tag, const std::vector<unsigned char>& content)
        {
            out.push_back(tag);
            const size_t n = content.size();
            if (n < 128) out.push_back(static_cast<unsigned char>(n));
            else if (n < 256) { out.push_back(0x81); out.push_back(static_cast<unsigned char>(n)); }
            else
            {
                out.push_back(0x82);
                out.push_back(static_cast<unsigned char>(n >> 8));
                out.push_back(static_cast<unsigned char>(n));
            }
            out.insert(out.end(), content.begin(), content.end());
        }

        /*  The name constraints of the local authority (RFC 5280 4.2.1.10): it may only
            certify localhost, the loopback addresses of IPv4 and ::1.  The authority is
            trusted by the whole machine, and the constraint is what keeps a certificate it
            signed from ever vouching for another site.  */
        std::vector<unsigned char> loopback_name_constraints()
        {
            std::vector<unsigned char> subtrees, subtree, name;
            const std::string dns = "localhost";
            name.assign(dns.begin(), dns.end());
            der_append(subtree, 0x82, name);                   // dNSName
            der_append(subtrees, 0x30, subtree);
            subtree.clear();
            name = { 127, 0, 0, 0, 255, 0, 0, 0 };              // 127.0.0.0/8
            der_append(subtree, 0x87, name);                   // iPAddress
            der_append(subtrees, 0x30, subtree);
            subtree.clear();
            name.assign(32, 0);
            name[15] = 1;                                      // ::1/128
            for (size_t i = 16; i < 32; ++i) name[i] = 0xFF;
            der_append(subtree, 0x87, name);
            der_append(subtrees, 0x30, subtree);
            std::vector<unsigned char> permitted, out;
            der_append(permitted, 0xA0, subtrees);             // permittedSubtrees [0]
            der_append(out, 0x30, permitted);
            return out;
        }

        bool write_certificate(mbedtls_pk_context& subject_key, mbedtls_pk_context& issuer_key,
            const std::string& subject, const std::string& issuer, bool is_ca, long days,
            random_source& rng, std::string& pem)
        {
            mbedtls_x509write_cert crt;
            mbedtls_x509write_crt_init(&crt);
            bool ok = false;
            do
            {
                mbedtls_x509write_crt_set_version(&crt, MBEDTLS_X509_CRT_VERSION_3);
                mbedtls_x509write_crt_set_md_alg(&crt, MBEDTLS_MD_SHA256);
                mbedtls_x509write_crt_set_subject_key(&crt, &subject_key);
                mbedtls_x509write_crt_set_issuer_key(&crt, &issuer_key);
                if (mbedtls_x509write_crt_set_subject_name(&crt, subject.c_str()) != 0) break;
                if (mbedtls_x509write_crt_set_issuer_name(&crt, issuer.c_str()) != 0) break;
                unsigned char serial[16];
                mbedtls_ctr_drbg_random(&rng.drbg, serial, sizeof(serial));
                serial[0] &= 0x7F;   // a positive integer
                serial[0] |= 0x01;
                if (mbedtls_x509write_crt_set_serial_raw(&crt, serial, sizeof(serial)) != 0) break;
                const std::time_t now = std::time(nullptr) - 3600;
                if (mbedtls_x509write_crt_set_validity(&crt, time_string(now).c_str(),
                        time_string(now + static_cast<std::time_t>(days) * 86400).c_str()) != 0) break;
                if (mbedtls_x509write_crt_set_basic_constraints(&crt, is_ca ? 1 : 0, is_ca ? 0 : -1) != 0) break;
                if (mbedtls_x509write_crt_set_subject_key_identifier(&crt) != 0) break;
                if (mbedtls_x509write_crt_set_authority_key_identifier(&crt) != 0) break;
                if (is_ca)
                {
                    if (mbedtls_x509write_crt_set_key_usage(&crt,
                        MBEDTLS_X509_KU_KEY_CERT_SIGN | MBEDTLS_X509_KU_CRL_SIGN) != 0)
                        break;
                    /* Marked non-critical, so that the verifiers that do not implement the
                       extension, Mbed TLS among them, still accept the authority; those of
                       the browsers and of the system apply it either way. */
                    const std::vector<unsigned char> nc = loopback_name_constraints();
                    static const char oid_name_constraints[] = MBEDTLS_OID_NAME_CONSTRAINTS;
                    if (mbedtls_x509write_crt_set_extension(&crt, oid_name_constraints,
                            MBEDTLS_OID_SIZE(MBEDTLS_OID_NAME_CONSTRAINTS), 0, nc.data(), nc.size()) != 0)
                        break;
                    /* The certificates it signs serve TLS servers alone: Windows, which applies the
                       extended key usage of an authority to the certificates below it, refuses one
                       of them for code or mail, whatever it declares. */
                    mbedtls_asn1_sequence eku;
                    std::memset(&eku, 0, sizeof(eku));
                    eku.buf.tag = MBEDTLS_ASN1_OID;
                    eku.buf.p = reinterpret_cast<unsigned char*>(const_cast<char*>(MBEDTLS_OID_SERVER_AUTH));
                    eku.buf.len = MBEDTLS_OID_SIZE(MBEDTLS_OID_SERVER_AUTH);
                    if (mbedtls_x509write_crt_set_ext_key_usage(&crt, &eku) != 0) break;
                }
                else
                {
                    if (mbedtls_x509write_crt_set_key_usage(&crt, MBEDTLS_X509_KU_DIGITAL_SIGNATURE) != 0) break;
                    mbedtls_asn1_sequence eku;
                    std::memset(&eku, 0, sizeof(eku));
                    eku.buf.tag = MBEDTLS_ASN1_OID;
                    eku.buf.p = reinterpret_cast<unsigned char*>(const_cast<char*>(MBEDTLS_OID_SERVER_AUTH));
                    eku.buf.len = MBEDTLS_OID_SIZE(MBEDTLS_OID_SERVER_AUTH);
                    if (mbedtls_x509write_crt_set_ext_key_usage(&crt, &eku) != 0) break;

                    static unsigned char ipv4[4] = { 127, 0, 0, 1 };
                    static unsigned char ipv6[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
                    static const char dns[] = "localhost";
                    mbedtls_x509_san_list n3, n2, n1;
                    std::memset(&n1, 0, sizeof(n1));
                    std::memset(&n2, 0, sizeof(n2));
                    std::memset(&n3, 0, sizeof(n3));
                    n1.node.type = MBEDTLS_X509_SAN_DNS_NAME;
                    n1.node.san.unstructured_name.p = reinterpret_cast<unsigned char*>(const_cast<char*>(dns));
                    n1.node.san.unstructured_name.len = sizeof(dns) - 1;
                    n1.next = &n2;
                    n2.node.type = MBEDTLS_X509_SAN_IP_ADDRESS;
                    n2.node.san.unstructured_name.p = ipv4;
                    n2.node.san.unstructured_name.len = sizeof(ipv4);
                    n2.next = &n3;
                    n3.node.type = MBEDTLS_X509_SAN_IP_ADDRESS;
                    n3.node.san.unstructured_name.p = ipv6;
                    n3.node.san.unstructured_name.len = sizeof(ipv6);
                    n3.next = nullptr;
                    if (mbedtls_x509write_crt_set_subject_alternative_name(&crt, &n1) != 0) break;
                }
                std::vector<unsigned char> buf(8192);
                if (mbedtls_x509write_crt_pem(&crt, buf.data(), buf.size(), mbedtls_ctr_drbg_random, &rng.drbg) != 0)
                    break;
                pem = reinterpret_cast<const char*>(buf.data());
                ok = true;
            } while (false);
            mbedtls_x509write_crt_free(&crt);
            return ok;
        }

        // The form of the private key in its file: encrypted for this account on Windows.
        bool seal_key(const std::string& pem, std::string& data)
        {
#if defined(_WIN32)
            DATA_BLOB in;
            in.cbData = static_cast<DWORD>(pem.size());
            in.pbData = reinterpret_cast<BYTE*>(const_cast<char*>(pem.data()));
            DATA_BLOB out = {};
            if (!CryptProtectData(&in, L"Albireo server key", nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN,
                    &out))
                return false;
            data.assign(reinterpret_cast<const char*>(out.pbData), out.cbData);
            LocalFree(out.pbData);
            return true;
#else
            data = pem;
            return true;
#endif
        }

        bool open_key(const std::string& data, std::string& pem)
        {
#if defined(_WIN32)
            DATA_BLOB in;
            in.cbData = static_cast<DWORD>(data.size());
            in.pbData = reinterpret_cast<BYTE*>(const_cast<char*>(data.data()));
            DATA_BLOB out = {};
            if (!CryptUnprotectData(&in, nullptr, nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out))
                return false;
            pem.assign(reinterpret_cast<const char*>(out.pbData), out.cbData);
            SecureZeroMemory(out.pbData, out.cbData);
            LocalFree(out.pbData);
            return true;
#else
            pem = data;
            return true;
#endif
        }

        bool write_key(const std::string& path, const std::string& pem)
        {
            std::string data;
            const bool ok = seal_key(pem, data) && write_file_restricted(path, data);
            wipe_secret(data);
            return ok;
        }

        /*
            The files of the versions before 1.0.8: the key of their authority, which is
            never kept now, and on Windows the key of the server in clear, which moves into
            its encrypted file.  Returns true when the key of the authority was found: a copy
            of it may have left the computer, and the authority is replaced at once.
        */
        bool migrate(const certificate_files& f)
        {
            const bool exposed = file_exists(f.ca_key);
            if (exposed)
            {
                if (remove_file(f.ca_key)) log_info("certificates: the key of the earlier local authority is deleted");
                else log_warning("certificates: the key of the earlier local authority cannot be deleted");
            }
            if (f.plain_key == f.server_key || !file_exists(f.plain_key)) return exposed;
            std::string pem;
            if (!file_exists(f.server_key) && read_file(f.plain_key, pem, 64 * 1024)
                && pem.compare(0, 11, "-----BEGIN ") == 0 && write_key(f.server_key, pem))
                log_info("certificates: the key of the server is now encrypted for the account of the service");
            wipe_secret(pem);
            remove_file(f.plain_key);
            return exposed;
        }
    }

    void wipe_secret(std::string& s)
    {
        if (!s.empty()) mbedtls_platform_zeroize(&s[0], s.size());
        s.clear();
    }

    certificate_files certificate_paths(const std::string& dir)
    {
        certificate_files f;
        f.ca_cert = join_path(dir, "albireo-authority.crt");
        f.ca_key = join_path(dir, "albireo-authority.key");
        f.server_cert = join_path(dir, "albireo-server.crt");
        f.plain_key = join_path(dir, "albireo-server.key");
#if defined(_WIN32)
        f.server_key = join_path(dir, "albireo-server.key.protected");
#else
        f.server_key = f.plain_key;
#endif
        return f;
    }

    bool read_server_key(const std::string& dir, std::string& pem)
    {
        std::string data;
        const bool ok = read_file(certificate_paths(dir).server_key, data, 64 * 1024) && open_key(data, pem);
        wipe_secret(data);
        return ok && !pem.empty();
    }

    bool ensure_certificates(const std::string& dir, bool& renewed, std::string& error)
    {
        renewed = false;
        psa_crypto_init();
        const certificate_files f = certificate_paths(dir);
        if (!make_dirs(dir))
        {
            error = "cannot create " + dir;
            return false;
        }
        random_source rng;
        if (!rng.ok)
        {
            error = "no entropy source";
            return false;
        }
        const bool exposed = migrate(f);
        if (exposed) log_warning("certificates: the local authority kept its key on disk, a new authority replaces it");

        // The certificates in place serve while they have time left and the key opens and matches.
        if (!exposed && file_exists(f.ca_cert) && file_exists(f.server_cert)
            && seconds_left(f.ca_cert) > renewal_seconds && seconds_left(f.server_cert) > renewal_seconds)
        {
            std::string pem;
            const bool ok = read_server_key(dir, pem) && key_matches(pem, f.server_cert, rng);
            wipe_secret(pem);
            if (ok) return true;
            log_warning("certificates: the key of the server cannot be read here, new certificates are made");
        }

        // A new authority signs a new certificate of the server, then its key is gone.
        std::string ca_pem, server_pem, key_text;
        bool ok = false;
        {
            pk_key ca_key, key;
            ok = generate_key(ca_key.ctx, rng) && generate_key(key.ctx, rng)
                && write_certificate(ca_key.ctx, ca_key.ctx, authority_name, authority_name, true, validity_days + 1,
                    rng, ca_pem)
                && write_certificate(key.ctx, ca_key.ctx, "CN=localhost,O=Cydral Technology", authority_name, false,
                    validity_days, rng, server_pem)
                && key_pem(key.ctx, key_text);
        }
        // The key first, then the certificate of the server, then the one of the authority.
        ok = ok && write_key(f.server_key, key_text) && write_file_atomic(f.server_cert, server_pem)
            && write_file_atomic(f.ca_cert, ca_pem);
        wipe_secret(key_text);
        if (!ok)
        {
            error = "cannot create the certificates of the local server in " + dir;
            return false;
        }
        renewed = true;
        log_info("certificates: new local authority and server certificate in " + dir);
        return true;
    }

    bool remove_certificate_files(const std::string& dir)
    {
        const certificate_files f = certificate_paths(dir);
        bool ok = true;
        for (const std::string& p : { f.server_key, f.plain_key, f.ca_key, f.server_cert, f.ca_cert })
            if (file_exists(p) && !remove_file(p)) ok = false;
        return ok;
    }

    std::string certificate_der(const std::string& pem_path)
    {
        std::string pem;
        if (!read_file(pem_path, pem)) return std::string();
        mbedtls_x509_crt crt;
        mbedtls_x509_crt_init(&crt);
        std::string der;
        if (mbedtls_x509_crt_parse(&crt, reinterpret_cast<const unsigned char*>(pem.c_str()), pem.size() + 1) == 0)
            der.assign(reinterpret_cast<const char*>(crt.raw.p), crt.raw.len);
        mbedtls_x509_crt_free(&crt);
        return der;
    }

    std::string certificate_thumbprint(const std::string& pem_path)
    {
        const std::string der = certificate_der(pem_path);
        if (der.empty()) return std::string();
        unsigned char h[32];
        mbedtls_sha256(reinterpret_cast<const unsigned char*>(der.data()), der.size(), h, 0);
        static const char* hex = "0123456789abcdef";
        std::string out;
        for (unsigned char b : h)
        {
            out += hex[b >> 4];
            out += hex[b & 15];
        }
        return out;
    }

#if defined(_WIN32)
    namespace
    {
        // Whether name constraints permit localhost and the loopback addresses, and nothing else.
        bool loopback_only(const CERT_EXTENSION* extension)
        {
            CERT_NAME_CONSTRAINTS_INFO* info = nullptr;
            DWORD size = 0;
            if (!CryptDecodeObjectEx(X509_ASN_ENCODING, X509_NAME_CONSTRAINTS, extension->Value.pbData,
                    extension->Value.cbData, CRYPT_DECODE_ALLOC_FLAG, nullptr, &info, &size))
                return false;
            // 127.0.0.0/8 and ::1/128, each address followed by its mask.
            const BYTE v4[8] = { 127, 0, 0, 0, 255, 0, 0, 0 };
            BYTE v6[32] = {};
            v6[15] = 1;
            std::memset(v6 + 16, 0xFF, 16);
            bool ok = info->cPermittedSubtree > 0;
            for (DWORD i = 0; ok && i < info->cPermittedSubtree; ++i)
            {
                const CERT_ALT_NAME_ENTRY& name = info->rgPermittedSubtree[i].Base;
                const CRYPT_DATA_BLOB& ip = name.IPAddress;
                if (name.dwAltNameChoice == CERT_ALT_NAME_DNS_NAME)
                    ok = name.pwszDNSName && _wcsicmp(name.pwszDNSName, L"localhost") == 0;
                else if (name.dwAltNameChoice == CERT_ALT_NAME_IP_ADDRESS)
                    ok = (ip.cbData == sizeof(v4) && std::memcmp(ip.pbData, v4, sizeof(v4)) == 0)
                        || (ip.cbData == sizeof(v6) && std::memcmp(ip.pbData, v6, sizeof(v6)) == 0);
                else
                    ok = false;
            }
            LocalFree(info);
            return ok;
        }

        /*
            Whether a certificate is the authority that Albireo makes: signed by its own key,
            named "Albireo Local Authority", and limited by its name constraints to localhost
            and the loopback addresses, so that the trust of the machine never reaches
            another site.
        */
        bool local_authority(const std::string& der)
        {
            PCCERT_CONTEXT c = CertCreateCertificateContext(X509_ASN_ENCODING,
                reinterpret_cast<const BYTE*>(der.data()), static_cast<DWORD>(der.size()));
            if (!c) return false;
            void* self = const_cast<CERT_CONTEXT*>(c);
            wchar_t name[128] = {};
            bool ok = CryptVerifyCertificateSignatureEx(0, X509_ASN_ENCODING, CRYPT_VERIFY_CERT_SIGN_SUBJECT_CERT,
                    self, CRYPT_VERIFY_CERT_SIGN_ISSUER_CERT, self, 0, nullptr)
                && CertGetNameStringW(c, CERT_NAME_ATTR_TYPE, 0, const_cast<char*>(szOID_COMMON_NAME), name, 128) > 1
                && std::wcscmp(name, L"Albireo Local Authority") == 0;
            const CERT_EXTENSION* constraints = ok ? CertFindExtension(szOID_NAME_CONSTRAINTS,
                c->pCertInfo->cExtension, c->pCertInfo->rgExtension) : nullptr;
            ok = ok && constraints && loopback_only(constraints);
            CertFreeCertificateContext(c);
            return ok;
        }

        void remove_authority_certificates(HCERTSTORE store)
        {
            for (;;)
            {
                PCCERT_CONTEXT c = CertFindCertificateInStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
                    CERT_FIND_SUBJECT_STR_W, L"Albireo Local Authority", nullptr);
                if (!c) break;
                CertDeleteCertificateFromStore(c);
            }
        }

        std::string system_error(const char* what)
        {
            return std::string(what) + " (error " + std::to_string(GetLastError()) + ")";
        }
    }

    bool authority_trusted()
    {
        HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0,
            CERT_SYSTEM_STORE_LOCAL_MACHINE | CERT_STORE_OPEN_EXISTING_FLAG | CERT_STORE_READONLY_FLAG, L"ROOT");
        if (!store) return false;
        PCCERT_CONTEXT c = CertFindCertificateInStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING, 0,
            CERT_FIND_SUBJECT_STR_W, L"Albireo Local Authority", nullptr);
        if (c) CertFreeCertificateContext(c);
        CertCloseStore(store, 0);
        return c != nullptr;
    }

    bool trust_authority(const std::string& dir, std::string& error)
    {
        const std::string der = certificate_der(certificate_paths(dir).ca_cert);
        if (der.empty())
        {
            error = "the certificate of the local authority cannot be read";
            return false;
        }
        if (!local_authority(der))
        {
            error = "the certificate of the local authority is not one that Albireo makes; it is not trusted";
            return false;
        }
        HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0,
            CERT_SYSTEM_STORE_LOCAL_MACHINE | CERT_STORE_OPEN_EXISTING_FLAG, L"ROOT");
        if (!store)
        {
            error = system_error("cannot open the trusted roots of the machine");
            return false;
        }
        remove_authority_certificates(store);
        const bool ok = CertAddEncodedCertificateToStore(store, X509_ASN_ENCODING | PKCS_7_ASN_ENCODING,
            reinterpret_cast<const BYTE*>(der.data()), static_cast<DWORD>(der.size()), CERT_STORE_ADD_REPLACE_EXISTING,
            nullptr) != FALSE;
        if (!ok) error = system_error("cannot add the local authority to the trusted roots");
        CertCloseStore(store, 0);
        return ok;
    }

    void untrust_authority()
    {
        HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_SYSTEM_W, 0, 0,
            CERT_SYSTEM_STORE_LOCAL_MACHINE | CERT_STORE_OPEN_EXISTING_FLAG, L"ROOT");
        if (!store) return;
        remove_authority_certificates(store);
        CertCloseStore(store, 0);
    }
#endif

}
