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
#include <mbedtls/sha256.h>
#include <mbedtls/x509_crt.h>
#include <psa/crypto.h>

#include "../common/log.h"
#include "../common/platform.h"

namespace albireo
{

    namespace
    {
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

        bool generate_key(mbedtls_pk_context& key, random_source& rng, std::string& pem)
        {
            if (mbedtls_pk_setup(&key, mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY)) != 0) return false;
            if (mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(key), mbedtls_ctr_drbg_random,
                &rng.drbg) != 0)
                return false;
            std::vector<unsigned char> buf(4096);
            if (mbedtls_pk_write_key_pem(&key, buf.data(), buf.size()) != 0) return false;
            pem = reinterpret_cast<const char*>(buf.data());
            return true;
        }

        bool load_key(mbedtls_pk_context& key, const std::string& path, random_source& rng)
        {
            std::string pem;
            if (!read_file(path, pem)) return false;
            return mbedtls_pk_parse_key(&key, reinterpret_cast<const unsigned char*>(pem.c_str()), pem.size() + 1,
                nullptr, 0, mbedtls_ctr_drbg_random, &rng.drbg) == 0;
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
            trusted by the whole machine, and the constraint is what keeps its key from
            ever vouching for another site, whatever happens to that key.  */
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
    }

    certificate_files certificate_paths(const std::string& dir)
    {
        certificate_files f;
        f.ca_cert = join_path(dir, "albireo-authority.crt");
        f.ca_key = join_path(dir, "albireo-authority.key");
        f.server_cert = join_path(dir, "albireo-server.crt");
        f.server_key = join_path(dir, "albireo-server.key");
        return f;
    }

    bool ensure_certificates(const std::string& dir, std::string& error)
    {
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

        static const char* ca_name = "CN=Albireo Local Authority,O=Cydral Technology,C=FR";
        mbedtls_pk_context ca_key;
        mbedtls_pk_init(&ca_key);
        bool new_ca = false;
        if (!(file_exists(f.ca_cert) && file_exists(f.ca_key) && load_key(ca_key, f.ca_key, rng)
              && seconds_left(f.ca_cert) > 86400LL * 60))
        {
            mbedtls_pk_free(&ca_key);
            mbedtls_pk_init(&ca_key);
            std::string key_pem, crt_pem;
            if (!generate_key(ca_key, rng, key_pem) ||
                !write_certificate(ca_key, ca_key, ca_name, ca_name, true, 3650, rng, crt_pem) ||
                !write_file_restricted(f.ca_key, key_pem) || !write_file_atomic(f.ca_cert, crt_pem))
            {
                mbedtls_pk_free(&ca_key);
                error = "cannot create the local certification authority";
                return false;
            }
            new_ca = true;
            log_info("certificates: new local authority " + f.ca_cert);
        }

        if (new_ca || !file_exists(f.server_cert) || !file_exists(f.server_key)
            || seconds_left(f.server_cert) < 86400LL * 30)
        {
            mbedtls_pk_context key;
            mbedtls_pk_init(&key);
            std::string key_pem, crt_pem;
            const bool ok = generate_key(key, rng, key_pem) &&
                write_certificate(key, ca_key, "CN=localhost,O=Cydral Technology", ca_name, false, 397, rng, crt_pem) &&
                write_file_restricted(f.server_key, key_pem) && write_file_atomic(f.server_cert, crt_pem);
            mbedtls_pk_free(&key);
            if (!ok)
            {
                mbedtls_pk_free(&ca_key);
                error = "cannot issue the certificate of the local server";
                return false;
            }
            log_info("certificates: new server certificate " + f.server_cert);
        }
        mbedtls_pk_free(&ca_key);
        return true;
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

}
