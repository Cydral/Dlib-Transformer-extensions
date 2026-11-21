// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_PRIVACY_SECRET_FORMATS_ABSTRACT_H_
#ifdef DLIB_PRIVACY_SECRET_FORMATS_ABSTRACT_H_

#include <string>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct secret_statistics
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                What the characters of a value tell of it, read on its bytes.

                length  - its number of bytes
                lower   - its small letters of the Latin alphabet without accents
                upper   - its capitals of the same alphabet
                digits  - its digits
                joiners - its hyphens, underscores, periods and apostrophes, which words
                          hold too
                symbols - its other characters below 0x80: signs and punctuation
                other   - its bytes of 0x80 and above: letters with accents, other scripts
                entropy - the Shannon entropy of its bytes, in bits per byte: 0 for one
                          byte repeated, log2(length) at most
        !*/
        std::size_t length = 0;
        std::size_t lower = 0;
        std::size_t upper = 0;
        std::size_t digits = 0;
        std::size_t joiners = 0;
        std::size_t symbols = 0;
        std::size_t other = 0;
        double entropy = 0;

        std::size_t letters() const;
        /*!
            ensures
                - Returns lower + upper + other.
        !*/
    };

    secret_statistics secret_stats(
        const std::string& text,
        std::size_t begin,
        std::size_t end
    );
    /*!
        ensures
            - Returns the statistics of the bytes of text from begin to end, end taken
              within the text, and those of an empty value when begin is not before it.
    !*/

    bool secret_like(
        const secret_statistics& st
    );
    /*!
        ensures
            - Returns true when a value of these statistics has the look of a value chosen
              to be hard to guess: four bytes at least, with letters and digits
              ("Tournesol2024"), with a sign among letters or digits ("p@ssword"), or
              sixteen letters and digits at least that repeat little (an entropy of 3.5
              bits at least). A word of a sentence, hyphens and apostrophes included, has
              none of these.
    !*/

    bool secret_placeholder(
        const std::string& value
    );
    /*!
        ensures
            - Returns true when value stands where a secret would be and is none: an empty
              value, a masked one ("********", "xxxxxxxx", "••••••"), a field left to fill
              ("<password>", "[password]", "{{ db_password }}", "%PASSWORD%"), a reference
              to a value kept elsewhere ("$DB_PASSWORD", "${DB_PASSWORD}",
              "os.environ[...]", "getpass.getpass()", "vault:secret/db", "*default",
              "!vault"), the path of a file ("/run/secrets/db_password", "~/.ssh/id_rsa",
              "C:\keys\service.pem"), a word that tells the state of a secret in French,
              English, Italian, Spanish or German ("required", "oublié", "none", "null",
              "true"), or the name of a secret in place of its value ("password",
              "YOUR_API_KEY", "my-secret-token").
    !*/

    int secret_key_strength(
        const std::string& name
    );
    /*!
        ensures
            - Reads name as the key of a setting, the name of a variable, of a parameter, of
              a header or of an option, its words joined by underscores, hyphens, periods
              or capitals ("DB_PASSWORD", "client-secret", "spring.datasource.password",
              "apiKey", "X-Api-Key"), and returns what it tells of the value it holds:
                0 - nothing: "username", "key", "public_key", "primary_key", "max_tokens",
                    "password_policy", "secret_name"
                1 - a secret or another value, which the look of the value decides:
                    "secret", "token", "auth", "session", "sig", "clave"
                2 - a secret: "password", "passwd", "pwd", "passphrase", "mdp",
                    "mot_de_passe", "passwort", "kennwort", "contrasena", "apikey",
                    "api_key", "secret_key", "private_key", "access_key", "client_secret",
                    "access_token", "refresh_token", "db_pass", "psk", "authorization",
                    and the names that end with them in one word ("PGPASSWORD")
                3 - a secret that may be a short code of digits: "pin", "puk", "otp", "cvv",
                    "cvc", "access_code", "security_code", "verification_code"
            - A final number and a final word of confirmation leave the name what it is:
              "password2", "password_confirmation" and "new_password" name a password.
    !*/

// ----------------------------------------------------------------------------------------

    enum class secret_evidence
    {
        /*!
            WHAT THIS ENUM REPRESENTS
                What told a secret.
                format     - the form of the value itself: no other value is written so
                assignment - the key, the option, the element or the header that names it
        !*/
        format,
        assignment
    };

    struct secret_match
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A secret found in a text.

                begin, end      - its byte range, without the quotes around it
                evidence        - what told it
                confidence      - between 0 and 1
                web_credentials - the value is the password of a web address ("http" or
                                  "https"), which a reader of web addresses reports whole
        !*/
        std::size_t begin = 0;
        std::size_t end = 0;
        secret_evidence evidence = secret_evidence::format;
        double confidence = 0;
        bool web_credentials = false;
    };

    std::vector<secret_match> find_secret_formats(
        const std::string& text
    );
    /*!
        ensures
            - Returns the secrets of text that their form tells, in the order of the text:
                - the lines of a private key between "-----BEGIN ... PRIVATE KEY-----" and
                  its closing line, each line a value, the lines that name the key and its
                  cipher left out; a key written on one line, with escaped line breaks as a
                  JSON value holds it or with spaces, gives its pieces. A line that is
                  neither of the alphabet of base 64 nor the name of the key or of its
                  cipher ends a key whose closing line is missing. A part of less than 16
                  characters is one when it ends a key whose closing line is read;
                - the tokens that a prefix of their issuer opens, of the length and the
                  alphabet it writes, 1024 characters at most: code hosting ("ghp_", "github_pat_", "glpat-"), cloud
                  ("AKIA" and "ASIA" with the secret key of 40 characters written on their
                  line, "AIza", "ya29."), payment ("sk_live_", "rk_live_", "whsec_"),
                  messaging ("xoxb-", "xoxp-", "SG."), packages ("npm_", "pypi-"), models
                  ("sk-", "hf_") and others;
                - JSON Web Tokens, three parts joined by periods whose first two open as
                  objects ("eyJ");
                - the hashes of passwords in the forms of account files ("$2b$12$...",
                  "$6$...", "$argon2id$...");
                - the password of an address written with its credentials
                  ("postgres://app:S3cret@db:5432/crm"), marked web_credentials for "http"
                  and "https";
                - the token that "Bearer" presents, of twenty characters at least with
                  letters and digits.
            - The text is read once: what is read again for a value is bounded (512 bytes for
              the credentials of an address, 1024 for a token), so that the time grows with
              the length of the text whatever it holds.
    !*/

    std::vector<secret_match> find_secret_assignments(
        const std::string& text
    );
    /*!
        ensures
            - Returns the values of text that a key names as secrets
              (secret_key_strength()), in the order of the text:
                - after "=", ":", "=>" or ":=", the key between quotes or not, as the
                  settings of a program, the variables of an environment, the keys of a
                  JSON or YAML document, the parameters of a query and the headers of a
                  request write them ("DB_PASSWORD=...", "\"client_secret\": \"...\"",
                  "?token=...&", "Authorization: Bearer ...", the scheme left out);
                - after an option of a command ("--password VALUE", "-Password 'VALUE'"),
                  and after "curl -u user:" for the password of its user;
                - between quotes after the key ("PASSWORD 'VALUE'", "IDENTIFIED BY
                  'VALUE'");
                - as the content of an element ("<password>VALUE</password>").
            - A value runs between its quotes, or to the first space; in a query it stops at
              the next parameter, in a list of settings joined by semicolons at the next
              one. The comma, the semicolon, the period and the bracket that close what
              holds the value stay outside. A remark of several words between parentheses
              written in place of the value is read whole, and is no value ("password:
              (sent apart)").
            - A placeholder (secret_placeholder()) is no value. After a key of strength 1
              the value has eight characters at least and the look of a secret
              (secret_like()); after a key of strength 3 it is a code of three to ten digits
              or has that look; after a key of strength 2 it has four characters at least.
              A word of letters alone, written without quotes before another word of its
              line, belongs to a sentence or to a list and is no value ("passwd: files
              systemd", "password: see below"). A value of more than 4096 bytes is none.
    !*/

    std::vector<secret_match> find_secrets(
        const std::string& text
    );
    /*!
        ensures
            - Returns the matches of find_secret_formats() and of find_secret_assignments(),
              sorted by position, none overlapping another: where a format is read at the
              start of the value of an assignment and runs past its end, the bounds of the
              format are kept.
    !*/

    void secret_chunk(
        const std::string& text,
        std::size_t& begin,
        std::size_t& end
    );
    /*!
        requires
            - begin <= end
        ensures
            - Grows the byte range begin to end to the limits of the value written there, as
              a secret is written whole: to the space, the quote, the bracket, the comma,
              the semicolon, the equals sign or the colon before it, and to the space, the
              quote, the closing bracket, the comma or the semicolon after it, by 1024 bytes
              at most on each side; a period or a colon that ends it is left out. An empty
              range stays as it is.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_PRIVACY_SECRET_FORMATS_ABSTRACT_H_
