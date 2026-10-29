// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef ALBIREO_HOST_CERTIFICATES_H_
#define ALBIREO_HOST_CERTIFICATES_H_

#include <string>

namespace albireo
{

    struct certificate_files
    {
        std::string ca_cert;       // PEM of the local authority
        std::string ca_key;
        std::string server_cert;   // PEM of the certificate of the local server
        std::string server_key;
    };

    certificate_files certificate_paths(const std::string& dir);

    /*!
        ensures
            - Makes sure dir holds a local certification authority and a certificate for
              the local server signed by it, valid for localhost, 127.0.0.1 and ::1.
            - The authority is created once, for ten years, with an elliptic curve key
              (P-256).  It is the certificate the installer adds to the trusted roots of
              the machine, so that the browsers accept the local server without warning.
            - The server certificate is valid for 397 days and is issued again, by the
              same authority, when less than 30 days remain: renewing it never requires
              installing anything.
            - Returns false, with the reason in error, when the files cannot be created.
    !*/
    bool ensure_certificates(const std::string& dir, std::string& error);

    // SHA-256 of the DER form of a PEM certificate, in lower case hexadecimal.
    std::string certificate_thumbprint(const std::string& pem_path);

    // DER bytes of the first certificate of a PEM file, empty on failure.
    std::string certificate_der(const std::string& pem_path);

}

#endif // ALBIREO_HOST_CERTIFICATES_H_
