// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef PSEUDALIS_HOST_CERTIFICATES_H_
#define PSEUDALIS_HOST_CERTIFICATES_H_

#include <string>

namespace pseudalis
{

    struct certificate_files
    {
        std::string ca_cert;       // PEM of the local authority
        std::string server_cert;   // PEM of the certificate of the local server
        std::string server_key;    // private key of the server: encrypted on Windows, PEM elsewhere
        std::string plain_key;     // the key of the server in clear of the earlier design, on Windows
        std::string ca_key;        // the key of the authority that the earlier design kept
    };

    certificate_files certificate_paths(const std::string& dir);

    /*!
        ensures
            - Makes sure dir holds the certificate of a local certification authority, a
              certificate of the local server signed by it, valid for localhost, 127.0.0.1
              and ::1, and the private key of that server.
            - The authority signs the certificate of the server once, when both are made,
              and its private key is erased from memory then: it is never written.  Its
              name constraints limit it to localhost and the loopback addresses, and its
              extended key usage to the authentication of servers.  It is the certificate
              the trusted roots of the machine receive (trust_authority()), so that the
              browsers accept the local server without warning.
            - Both certificates are valid for 397 days.  When less than 30 days remain, when
              the key of the server does not read or does not match its certificate, or when
              dir holds the key of an authority kept by the earlier design of these files,
              which may have been copied, a new authority and a new certificate of the server
              replace them, and renewed is set to true.
            - On Windows, the private key of the server is encrypted with DPAPI for the
              account that runs this function, LocalSystem for the service, in a file that
              SYSTEM and the administrators alone may open: a copy of that file serves no
              other account and no other computer.  Elsewhere it is a PEM file that its
              owner alone may read.  A key left in clear by an earlier version is encrypted
              in place of its file, and the key of an earlier authority is deleted.
            - Returns false, with the reason in error, when the files cannot be created.
    !*/
    bool ensure_certificates(const std::string& dir, bool& renewed, std::string& error);

    /*!
        ensures
            - Reads the private key of the server of dir in PEM form, decrypted in memory for
              the TLS server.  Returns false when it cannot be read or decrypted, by another
              account or on another computer for instance.
    !*/
    bool read_server_key(const std::string& dir, std::string& pem);

    // Erases the characters of a secret, then empties it.
    void wipe_secret(std::string& s);

    // Deletes the certificates and the keys of dir; false when one of them remains.
    bool remove_certificate_files(const std::string& dir);

    // DER bytes of the first certificate of a PEM file, empty on failure.
    std::string certificate_der(const std::string& pem_path);

#if defined(_WIN32)
    /*!
        The local authority in the trusted roots of the machine (LocalMachine\Root), which
        only an administrator or LocalSystem may change.
            - authority_trusted() tells whether an authority of Pseudalis is there.
            - trust_authority() puts the authority of dir there, in place of any earlier
              authority of Pseudalis, once it has checked that it is the one Pseudalis makes:
              signed by its own key, named "Pseudalis Local Authority" and limited by its name
              constraints to localhost and the loopback addresses.
            - untrust_authority() removes every authority of Pseudalis.
    !*/
    bool authority_trusted();
    bool trust_authority(const std::string& dir, std::string& error);
    void untrust_authority();
#endif

}

#endif // PSEUDALIS_HOST_CERTIFICATES_H_
