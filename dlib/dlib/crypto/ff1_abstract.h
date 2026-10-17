// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FF1_ABSTRACT_H_
#ifdef DLIB_FF1_ABSTRACT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class ff1_cipher
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The FF1 format-preserving encryption method of NIST SP 800-38G (Rev. 1,
                second public draft, February 2025), instantiated with AES. FF1 is a ten
                round Feistel network whose round function is an AES based PRF; it maps a
                string of n numerals in base radix to another string of n numerals in the
                same base, under a key and a public tweak.

                Encrypting a value yields a value of the same shape: ten digits stay ten
                digits, a word over a 26 letter alphabet stays such a word. The mapping is
                a permutation for every (key, tweak, n), so decryption recovers the input
                exactly. Equal inputs under equal tweaks give equal outputs, so equality
                comparisons and joins still apply to protected data.

            PARAMETER CONSTRAINTS (enforced)
                - 2 <= radix <= 65536
                - n >= 2 and radix^n >= 1,000,000. Below this domain size the published
                  attacks on small-domain Feistel ciphers become practical, so the
                  revision of the standard forbids it. min_length() returns the smallest
                  admissible n for the radix.
                - every numeral < radix

            IMPLEMENTATION NOTES
                The conversions NUM_radix and STR_radix are carried out with exact integer
                arithmetic; the draft disallows floating point. The byte count b is
                derived from the bit length of radix^v - 1, which equals
                ceil(v * log2(radix)) without rounding error.

            THREAD SAFETY
                encrypt() and decrypt() are const and may run concurrently.
        !*/
    public:
        ff1_cipher(
            const void* key,
            std::size_t key_size,
            std::uint32_t radix
        );
        /*!
            ensures
                - #get_radix() == radix
                - The cipher uses AES with the given key.
            throws
                - crypto_error if key_size is not 16, 24 or 32, or if radix is outside
                  [2, 65536].
        !*/

        std::uint32_t get_radix() const;
        /*!
            ensures
                - Returns the radix given to the constructor.
        !*/

        std::size_t min_length() const;
        /*!
            ensures
                - Returns the smallest n >= 2 such that radix^n >= 1,000,000.
        !*/

        std::vector<std::uint32_t> encrypt(
            const std::vector<std::uint32_t>& numerals,
            const std::string& tweak
        ) const;
        /*!
            ensures
                - Returns FF1.Encrypt(K, tweak, numerals).
                - The result has numerals.size() elements, each < get_radix().
            throws
                - crypto_error if the parameter constraints above are violated.
        !*/

        std::vector<std::uint32_t> decrypt(
            const std::vector<std::uint32_t>& numerals,
            const std::string& tweak
        ) const;
        /*!
            ensures
                - Returns FF1.Decrypt(K, tweak, numerals), so that
                  decrypt(encrypt(x, t), t) == x.
            throws
                - crypto_error if the parameter constraints above are violated.
        !*/

        std::string encrypt(
            const std::string& text,
            const std::string& alphabet,
            const std::string& tweak
        ) const;
        std::string decrypt(
            const std::string& text,
            const std::string& alphabet,
            const std::string& tweak
        ) const;
        /*!
            requires
                - alphabet.size() == get_radix() and its characters are distinct.
            ensures
                - Character level convenience wrappers: each character of text is mapped
                  to its index in alphabet, the numeral string is processed, and the
                  result is mapped back through alphabet.
            throws
                - crypto_error if text contains a character missing from alphabet or if
                  the constraints above are violated.
        !*/
    };

// ----------------------------------------------------------------------------------------

    std::vector<std::uint32_t> bytes_to_numerals(
        const std::string& bytes,
        std::uint32_t radix,
        std::size_t length
    );
    /*!
        ensures
            - Interprets bytes as a big-endian unsigned integer and returns its base radix
              representation with exactly length numerals, most significant first.
        throws
            - crypto_error if radix is outside [2, 65536] or if the integer does not fit
              in length numerals.
    !*/

    bool numerals_to_bytes(
        const std::vector<std::uint32_t>& numerals,
        std::uint32_t radix,
        std::size_t length,
        std::string& bytes
    );
    /*!
        requires
            - 2 <= radix <= 65536
            - every element of numerals is < radix
        ensures
            - Interprets numerals as a base radix integer, most significant first.
            - If the integer fits in length bytes, #bytes holds its big-endian
              representation on exactly length bytes and true is returned.
            - Otherwise false is returned and bytes is unspecified.
    !*/

    std::size_t numerals_needed(
        std::size_t byte_count,
        std::uint32_t radix
    );
    /*!
        ensures
            - Returns the smallest m such that radix^m >= 256^byte_count, that is the
              number of base radix numerals able to hold any byte_count byte integer.
        throws
            - crypto_error if radix is outside [2, 65536].
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_FF1_ABSTRACT_H_
