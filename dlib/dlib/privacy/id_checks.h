// Copyright (C) 2026 Cydral Technology (cydraltechnology@gmail.com)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PRIVACY_ID_CHECKS_H_
#define DLIB_PRIVACY_ID_CHECKS_H_

#include <cstddef>
#include <ctime>
#include <string>

namespace dlib
{
    namespace privacy_impl
    {
        /*
            Check digit algorithms of the identifiers recognized by pii_detector. Each
            function takes the compact form of a value, its digits and capital letters
            without separators, and returns true when the value is well formed and its
            control characters verify. The detector accepts a candidate only when its
            check holds: alone for the distinctive layouts, after the usual label of the
            number for the others.
        */

        inline bool id_all_digits(const std::string& s, std::size_t b = 0, std::size_t e = std::string::npos)
        {
            if (e == std::string::npos) e = s.size();
            if (b >= e) return false;
            for (std::size_t i = b; i < e; ++i)
                if (s[i] < '0' || s[i] > '9') return false;
            return true;
        }

        inline int id_digit(const std::string& s, std::size_t i) { return s[i] - '0'; }

        inline long long id_number(const std::string& s, std::size_t b, std::size_t e)
        {
            long long v = 0;
            for (std::size_t i = b; i < e; ++i) v = v * 10 + id_digit(s, i);
            return v;
        }

        inline bool luhn_ok(const std::string& d)
        {
            if (!id_all_digits(d)) return false;
            int sum = 0;
            bool dbl = false;
            for (std::size_t i = d.size(); i-- > 0;)
            {
                int v = id_digit(d, i);
                if (dbl) { v *= 2; if (v > 9) v -= 9; }
                sum += v;
                dbl = !dbl;
            }
            return sum % 10 == 0;
        }

        inline int luhn_digit(const std::string& payload)
        {
            int sum = 0;
            bool dbl = true;
            for (std::size_t i = payload.size(); i-- > 0;)
            {
                int v = id_digit(payload, i);
                if (dbl) { v *= 2; if (v > 9) v -= 9; }
                sum += v;
                dbl = !dbl;
            }
            return (10 - sum % 10) % 10;
        }

        inline int verhoeff_digit(const std::string& payload)
        {
            static const int d[10][10] = {
                {0,1,2,3,4,5,6,7,8,9}, {1,2,3,4,0,6,7,8,9,5}, {2,3,4,0,1,7,8,9,5,6}, {3,4,0,1,2,8,9,5,6,7},
                {4,0,1,2,3,9,5,6,7,8}, {5,9,8,7,6,0,4,3,2,1}, {6,5,9,8,7,1,0,4,3,2}, {7,6,5,9,8,2,1,0,4,3},
                {8,7,6,5,9,3,2,1,0,4}, {9,8,7,6,5,4,3,2,1,0} };
            static const int p[8][10] = {
                {0,1,2,3,4,5,6,7,8,9}, {1,5,7,6,2,8,3,0,9,4}, {5,8,0,3,7,9,6,1,4,2}, {8,9,1,6,0,4,3,5,2,7},
                {9,4,5,3,1,2,6,8,7,0}, {4,2,8,6,5,7,3,9,0,1}, {2,7,9,3,8,0,6,4,1,5}, {7,0,4,6,9,1,3,2,5,8} };
            static const int inv[10] = {0,4,3,2,1,5,6,7,8,9};
            int c = 0;
            std::size_t i = 0;
            for (std::size_t k = payload.size(); k-- > 0; ++i)
                c = d[c][p[(i + 1) % 8][id_digit(payload, k)]];
            return inv[c];
        }

        inline bool valid_date_digits(int year, int month, int day)
        {
            if (month < 1 || month > 12 || day < 1) return false;
            static const int days[12] = { 31,29,31,30,31,30,31,31,30,31,30,31 };
            if (day > days[month - 1]) return false;
            if (month == 2 && day == 29)
                return (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            return true;
        }

        inline bool is_hex_char(char c)
        {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        }

        // 8-4-4-4-12 hexadecimal digits, in one case.
        inline bool is_uuid_shape(const std::string& s)
        {
            if (s.size() != 36) return false;
            bool upper = false, lower = false;
            for (std::size_t i = 0; i < 36; ++i)
            {
                const char c = s[i];
                if (i == 8 || i == 13 || i == 18 || i == 23)
                {
                    if (c != '-') return false;
                    continue;
                }
                if (!is_hex_char(c)) return false;
                if (c >= 'a' && c <= 'f') lower = true;
                if (c >= 'A' && c <= 'F') upper = true;
            }
            return !(upper && lower);
        }

    // ------------------------------------------------------------------------------------

        // Belgian national register number: YYMMDD SSS CC, 97 - n mod 97, n prefixed with
        // a 2 for people born from 2000. Bis numbers add 20 or 40 to the month.
        inline bool be_national_number_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d)) return false;
            const int month = id_digit(d, 2) * 10 + id_digit(d, 3);
            const int day = id_digit(d, 4) * 10 + id_digit(d, 5);
            if (!(month <= 12 || (month >= 20 && month <= 32) || (month >= 40 && month <= 52)) || day > 31)
                return false;
            const long long n = id_number(d, 0, 9);
            const int check = id_digit(d, 9) * 10 + id_digit(d, 10);
            return 97 - static_cast<int>(n % 97) == check || 97 - static_cast<int>((2000000000LL + n) % 97) == check;
        }

        // Spanish DNI (8 digits and a letter) and NIE (X, Y or Z, 7 digits and a letter).
        inline bool es_dni_ok(const std::string& s)
        {
            static const char* letters = "TRWAGMYFPDXBNJZSQVHLCKE";
            if (s.size() != 9) return false;
            std::string d = s.substr(0, 8);
            if (d[0] == 'X') d[0] = '0';
            else if (d[0] == 'Y') d[0] = '1';
            else if (d[0] == 'Z') d[0] = '2';
            if (!id_all_digits(d)) return false;
            return letters[id_number(d, 0, 8) % 23] == s[8];
        }

        // Italian codice fiscale, omocodia substitutions included.
        inline bool it_fiscal_code_ok(const std::string& s)
        {
            if (s.size() != 16) return false;
            static const char* omocodia = "LMNPQRSTUV";
            auto digit_like = [&](char c) {
                return (c >= '0' && c <= '9') || (c != 0 && std::string(omocodia).find(c) != std::string::npos);
            };
            auto letter = [](char c) { return c >= 'A' && c <= 'Z'; };
            for (std::size_t i : { 0, 1, 2, 3, 4, 5, 11, 15 })
                if (!letter(s[i])) return false;
            for (std::size_t i : { 6, 7, 9, 10, 12, 13, 14 })
                if (!digit_like(s[i])) return false;
            if (std::string("ABCDEHLMPRST").find(s[8]) == std::string::npos) return false;
            static const int odd[26] = { 1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14,
                                         16, 10, 22, 25, 24, 23 };
            int sum = 0;
            for (std::size_t i = 0; i < 15; ++i)
            {
                const char c = s[i];
                const int v = (c >= '0' && c <= '9') ? c - '0' : c - 'A';
                sum += (i % 2 == 0) ? odd[v] : v;
            }
            return s[15] == static_cast<char>('A' + sum % 26);
        }

        // United Kingdom National Insurance number: two letters, six digits, A to D.
        inline bool uk_nino_ok(const std::string& s)
        {
            if (s.size() != 9 || !id_all_digits(s, 2, 8)) return false;
            const char a = s[0], b = s[1];
            if (a < 'A' || a > 'Z' || b < 'A' || b > 'Z' || s[8] < 'A' || s[8] > 'D') return false;
            if (std::string("DFIQUV").find(a) != std::string::npos) return false;
            if (std::string("DFIOQUV").find(b) != std::string::npos) return false;
            const std::string prefix = s.substr(0, 2);
            return prefix != "BG" && prefix != "GB" && prefix != "KN" && prefix != "NK" && prefix != "NT" &&
                   prefix != "TN" && prefix != "ZZ";
        }

        // United States Social Security number, area, group and serial rules.
        inline bool us_ssn_ok(const std::string& d)
        {
            if (d.size() != 9 || !id_all_digits(d)) return false;
            const int area = static_cast<int>(id_number(d, 0, 3));
            const int group = static_cast<int>(id_number(d, 3, 5));
            const int serial = static_cast<int>(id_number(d, 5, 9));
            return area != 0 && area != 666 && area < 900 && group != 0 && serial != 0;
        }

        // Swiss AHV/AVS number: 756 and ten digits, EAN-13 check digit.
        inline bool ch_avs_ok(const std::string& d)
        {
            if (d.size() != 13 || !id_all_digits(d) || d.compare(0, 3, "756") != 0) return false;
            int sum = 0;
            for (std::size_t i = 0; i < 12; ++i) sum += id_digit(d, i) * (i % 2 == 0 ? 1 : 3);
            return (10 - sum % 10) % 10 == id_digit(d, 12);
        }

        // Luxembourg national identification number: birth date, three digits, a Luhn and
        // a Verhoeff check digit, both computed over the first eleven digits.
        inline bool lu_national_number_ok(const std::string& d)
        {
            if (d.size() != 13 || !id_all_digits(d)) return false;
            const int year = static_cast<int>(id_number(d, 0, 4));
            if (year < 1850 || year > 2100 ||
                !valid_date_digits(year, static_cast<int>(id_number(d, 4, 6)), static_cast<int>(id_number(d, 6, 8))))
                return false;
            const std::string payload = d.substr(0, 11);
            return luhn_digit(payload) == id_digit(d, 11) && verhoeff_digit(payload) == id_digit(d, 12);
        }

        // Dutch citizen service number (BSN): the eleven test.
        inline bool nl_bsn_ok(const std::string& d)
        {
            if (d.size() != 9 || !id_all_digits(d)) return false;
            int sum = 0;
            for (std::size_t i = 0; i < 8; ++i) sum += id_digit(d, i) * static_cast<int>(9 - i);
            sum -= id_digit(d, 8);
            return sum != 0 && sum % 11 == 0;
        }

        // German tax identification number (Steuer-IdNr): ISO 7064 MOD 11,10.
        inline bool de_tax_id_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d) || d[0] == '0') return false;
            int product = 10;
            for (std::size_t i = 0; i < 10; ++i)
            {
                int sum = (id_digit(d, i) + product) % 10;
                if (sum == 0) sum = 10;
                product = (2 * sum) % 11;
            }
            int check = 11 - product;
            if (check == 10) check = 0;
            return check == id_digit(d, 10);
        }

        // Polish PESEL: weights 1 3 7 9.
        inline bool pl_pesel_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d)) return false;
            static const int w[10] = { 1, 3, 7, 9, 1, 3, 7, 9, 1, 3 };
            int sum = 0;
            for (std::size_t i = 0; i < 10; ++i) sum += id_digit(d, i) * w[i];
            return (10 - sum % 10) % 10 == id_digit(d, 10);
        }

        // Portuguese tax number (NIF): modulo 11.
        inline bool pt_nif_ok(const std::string& d)
        {
            // First digit 0 is not issued, and 4 only as the prefix 45 of non-residents.
            if (d.size() != 9 || !id_all_digits(d) || d[0] == '0' || (d[0] == '4' && d[1] != '5')) return false;
            int sum = 0;
            for (std::size_t i = 0; i < 8; ++i) sum += id_digit(d, i) * static_cast<int>(9 - i);
            const int r = sum % 11;
            return (r < 2 ? 0 : 11 - r) == id_digit(d, 8);
        }

        // French tax number (numero fiscal, SPI): 13 digits, the last three equal to the
        // first ten modulo 511.
        inline bool fr_tax_number_ok(const std::string& d)
        {
            if (d.size() != 13 || !id_all_digits(d) || d[0] > '3') return false;
            return id_number(d, 0, 10) % 511 == id_number(d, 10, 13);
        }

        // French bank details (RIB): bank, branch, account and key, letters of the account
        // number converted as the banks do.
        inline bool fr_rib_ok(const std::string& s)
        {
            if (s.size() != 23 || !id_all_digits(s, 0, 10) || !id_all_digits(s, 21, 23)) return false;
            static const char* conv = "12345678912345678923456789";
            long long account = 0;
            for (std::size_t i = 10; i < 21; ++i)
            {
                const char c = s[i];
                int v;
                if (c >= '0' && c <= '9') v = c - '0';
                else if (c >= 'A' && c <= 'Z') v = conv[c - 'A'] - '0';
                else return false;
                account = (account * 10 + v) % 97;
            }
            const long long bank = id_number(s, 0, 5) % 97, branch = id_number(s, 5, 10) % 97;
            const long long key = 97 - (89 * bank + 15 * branch + 3 * account) % 97;
            return key == id_number(s, 21, 23);
        }

        // Vehicle identification number: 17 characters without I, O and Q.
        inline bool vin_shape_ok(const std::string& s)
        {
            if (s.size() != 17) return false;
            int letters = 0, digits = 0;
            for (char c : s)
            {
                if (c >= '0' && c <= '9') ++digits;
                else if (c >= 'A' && c <= 'Z' && c != 'I' && c != 'O' && c != 'Q') ++letters;
                else return false;
            }
            return letters >= 1 && digits >= 3;
        }

        // The North American check digit in position 9, also used by many European makers.
        inline bool vin_check_digit_ok(const std::string& s)
        {
            if (!vin_shape_ok(s)) return false;
            static const int w[17] = { 8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2 };
            static const int tr[26] = { 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 0, 7, 0, 9, 2, 3, 4, 5, 6, 7, 8, 9 };
            int sum = 0;
            for (std::size_t i = 0; i < 17; ++i)
            {
                const char c = s[i];
                sum += ((c >= '0' && c <= '9') ? c - '0' : tr[c - 'A']) * w[i];
            }
            const int r = sum % 11;
            return s[8] == (r == 10 ? 'X' : static_cast<char>('0' + r));
        }

    // ------------------------------------------------------------------------------------
    // National numbers of other countries. The detector accepts most of them only after
    // their usual label, since a check digit alone leaves one chance in ten or eleven.

        inline int weighted_sum(const std::string& d, const int* weights, std::size_t count, std::size_t from = 0)
        {
            int sum = 0;
            for (std::size_t i = 0; i < count; ++i)
                sum += weights[i] * id_digit(d, from + i);
            return sum;
        }

        // Austrian social security number (SVNR): check digit in position 4.
        inline bool at_svnr_ok(const std::string& d)
        {
            if (d.size() != 10 || !id_all_digits(d) || d[0] == '0') return false;
            static const int w[10] = { 3, 7, 9, 0, 5, 8, 4, 2, 1, 6 };
            return weighted_sum(d, w, 10) % 11 == id_digit(d, 3);
        }

        // Czech and Slovak birth number (rodne cislo): YYMMDD then 3 or 4 digits.
        inline bool cz_birth_number_ok(const std::string& d)
        {
            if ((d.size() != 9 && d.size() != 10) || !id_all_digits(d)) return false;
            int year = 1900 + static_cast<int>(id_number(d, 0, 2));
            const int month = static_cast<int>(id_number(d, 2, 4)) % 50 % 20;
            const int day = static_cast<int>(id_number(d, 4, 6));
            if (d.size() == 9)
            {
                if (year >= 1980) year -= 100;
                if (year > 1953) return false;
            }
            else if (year < 1954)
            {
                year += 100;
            }
            if (!valid_date_digits(year, month, day)) return false;
            return d.size() == 9 || static_cast<int>(id_number(d, 0, 9) % 11 % 10) == id_digit(d, 9);
        }

        inline int current_year()
        {
            const std::time_t now = std::time(nullptr);
            std::tm parts;
#if defined(_WIN32)
            gmtime_s(&parts, &now);
#else
            gmtime_r(&now, &parts);
#endif
            return parts.tm_year + 1900;
        }

        // Danish CPR number: DDMMYY and four digits, without check digit since 2007.
        inline bool dk_cpr_ok(const std::string& d)
        {
            if (d.size() != 10 || !id_all_digits(d)) return false;
            const int day = static_cast<int>(id_number(d, 0, 2)), month = static_cast<int>(id_number(d, 2, 4));
            int year = static_cast<int>(id_number(d, 4, 6));
            const char c = d[6];
            if ((c >= '5' && c <= '8') && year >= 58) year += 1800;
            else if ((c >= '0' && c <= '3') || ((c == '4' || c == '9') && year >= 37)) year += 1900;
            else year += 2000;
            return year <= current_year() && valid_date_digits(year, month, day);
        }

        // Finnish personal identity code (HETU): DDMMYY, century sign, 3 digits, check character.
        inline bool fi_hetu_ok(const std::string& s)
        {
            if (s.size() != 11 || !id_all_digits(s.substr(0, 6)) || !id_all_digits(s.substr(7, 3))) return false;
            int century = 0;
            const char c = s[6];
            if (c == '+') century = 1800;
            else if (c == '-' || c == 'Y' || c == 'X' || c == 'W' || c == 'V' || c == 'U') century = 1900;
            else if (c >= 'A' && c <= 'F') century = 2000;
            else return false;
            const int day = static_cast<int>(id_number(s, 0, 2)), month = static_cast<int>(id_number(s, 2, 4));
            const int year = century + static_cast<int>(id_number(s, 4, 6));
            const int individual = static_cast<int>(id_number(s, 7, 10));
            if (!valid_date_digits(year, month, day) || individual < 2 || individual >= 900) return false;
            static const char* control = "0123456789ABCDEFHJKLMNPRSTUVWXY";
            const long long n = id_number(s, 0, 6) * 1000 + individual;
            return s[10] == control[n % 31];
        }

        // Swedish personal identity number (personnummer): YYMMDD and four digits, Luhn.
        inline bool se_personnummer_ok(const std::string& d)
        {
            if ((d.size() != 10 && d.size() != 12) || !id_all_digits(d)) return false;
            const std::string tail = d.substr(d.size() - 10);
            const int month = static_cast<int>(id_number(tail, 2, 4));
            int day = static_cast<int>(id_number(tail, 4, 6));
            if (day > 60) day -= 60;      // coordination numbers
            const int year = d.size() == 12 ? static_cast<int>(id_number(d, 0, 4)) : 2000;
            return valid_date_digits(year, month, day) && luhn_ok(tail);
        }

        // Norwegian national identity number (fodselsnummer): two mod 11 check digits.
        inline bool no_fodselsnummer_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d)) return false;
            static const int w1[9] = { 3, 7, 6, 1, 8, 9, 4, 5, 2 };
            static const int w2[10] = { 5, 4, 3, 2, 7, 6, 5, 4, 3, 2 };
            const int k1 = (11 - weighted_sum(d, w1, 9) % 11) % 11, k2 = (11 - weighted_sum(d, w2, 10) % 11) % 11;
            if (k1 == 10 || k2 == 10 || k1 != id_digit(d, 9) || k2 != id_digit(d, 10)) return false;
            int day = static_cast<int>(id_number(d, 0, 2)), month = static_cast<int>(id_number(d, 2, 4));
            if (day >= 80) return false;
            if (day > 40) day -= 40;
            if (month > 40) month -= 40;
            return valid_date_digits(2000, month, day);
        }

        // Irish personal public service number (PPSN): seven digits, a check letter, an optional letter.
        inline bool ie_pps_ok(const std::string& s)
        {
            if ((s.size() != 8 && s.size() != 9) || !id_all_digits(s.substr(0, 7))) return false;
            static const char* alphabet = "WABCDEFGHIJKLMNOPQRSTUV";
            auto index = [&](char c) {
                for (int k = 0; alphabet[k]; ++k) if (alphabet[k] == c) return k;
                return -1;
            };
            if (s[7] < 'A' || s[7] > 'W') return false;
            int extra = 0;
            if (s.size() == 9)
            {
                const char c = s[8];
                if (c != 'A' && c != 'B' && c != 'H' && c != 'W' && c != 'T' && c != 'X') return false;
                if (c == 'A' || c == 'B' || c == 'H') extra = index(c);
            }
            int sum = 9 * extra;
            for (std::size_t i = 0; i < 7; ++i) sum += static_cast<int>(8 - i) * id_digit(s, i);
            return s[7] == alphabet[sum % 23];
        }

        // Romanian personal numeric code (CNP).
        inline bool ro_cnp_ok(const std::string& d)
        {
            if (d.size() != 13 || !id_all_digits(d) || d[0] == '0') return false;
            static const int w[12] = { 2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9 };
            const int check = weighted_sum(d, w, 12) % 11;
            if ((check == 10 ? 1 : check) != id_digit(d, 12)) return false;
            int century = 1900;
            if (d[0] == '3' || d[0] == '4') century = 1800;
            else if (d[0] == '5' || d[0] == '6') century = 2000;
            const int county = static_cast<int>(id_number(d, 7, 9));
            if (!((county >= 1 && county <= 48) || county == 51 || county == 52 || county == 70 ||
                  (county >= 80 && county <= 83)))
                return false;
            return valid_date_digits(century + static_cast<int>(id_number(d, 1, 3)), static_cast<int>(id_number(d, 3, 5)),
                                     static_cast<int>(id_number(d, 5, 7)));
        }

        // Bulgarian uniform civil number (EGN).
        inline bool bg_egn_ok(const std::string& d)
        {
            if (d.size() != 10 || !id_all_digits(d)) return false;
            static const int w[9] = { 2, 4, 8, 5, 10, 9, 7, 3, 6 };
            if (weighted_sum(d, w, 9) % 11 % 10 != id_digit(d, 9)) return false;
            int year = 1900 + static_cast<int>(id_number(d, 0, 2)), month = static_cast<int>(id_number(d, 2, 4));
            if (month > 40) { year += 100; month -= 40; }
            else if (month > 20) { year -= 100; month -= 20; }
            return valid_date_digits(year, month, static_cast<int>(id_number(d, 4, 6)));
        }

        // ISO 7064 MOD 11,10, the check of the Croatian OIB and of the German tax number.
        inline bool iso7064_mod11_10_ok(const std::string& d)
        {
            if (d.size() < 2 || !id_all_digits(d)) return false;
            int product = 10;
            for (std::size_t i = 0; i + 1 < d.size(); ++i)
            {
                int sum = (id_digit(d, i) + product) % 10;
                if (sum == 0) sum = 10;
                product = (2 * sum) % 11;
            }
            return (11 - product) % 10 == id_digit(d, d.size() - 1);
        }

        // Croatian personal identification number (OIB).
        inline bool hr_oib_ok(const std::string& d)
        {
            return d.size() == 11 && iso7064_mod11_10_ok(d);
        }

        // Estonian (isikukood) and Lithuanian (asmens kodas) personal codes.
        inline bool ee_lt_personal_code_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d)) return false;
            int check = 0;
            for (std::size_t i = 0; i < 10; ++i) check += static_cast<int>(i % 9 + 1) * id_digit(d, i);
            check %= 11;
            if (check == 10)
            {
                check = 0;
                for (std::size_t i = 0; i < 10; ++i) check += static_cast<int>((i + 2) % 9 + 1) * id_digit(d, i);
                check %= 11;
            }
            if (check % 10 != id_digit(d, 10)) return false;
            if (d[0] == '9') return true;
            if (d[0] < '1' || d[0] > '8') return false;
            const int century = 1800 + 100 * ((d[0] - '1') / 2);
            return valid_date_digits(century + static_cast<int>(id_number(d, 1, 3)), static_cast<int>(id_number(d, 3, 5)),
                                     static_cast<int>(id_number(d, 5, 7)));
        }

        // Slovenian unique master citizen number (EMSO).
        inline bool si_emso_ok(const std::string& d)
        {
            if (d.size() != 13 || !id_all_digits(d)) return false;
            static const int w[12] = { 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2 };
            const int check = (11 - weighted_sum(d, w, 12) % 11) % 11 % 10;
            if (check != id_digit(d, 12)) return false;
            int year = static_cast<int>(id_number(d, 4, 7));
            year += year < 800 ? 2000 : 1000;
            return valid_date_digits(year, static_cast<int>(id_number(d, 2, 4)), static_cast<int>(id_number(d, 0, 2)));
        }

        // Greek social security number (AMKA): DDMMYY, Luhn.
        inline bool gr_amka_ok(const std::string& d)
        {
            if (d.size() != 11 || !luhn_ok(d)) return false;
            return valid_date_digits(2000, static_cast<int>(id_number(d, 2, 4)), static_cast<int>(id_number(d, 0, 2)));
        }

        // British NHS number: mod 11 over ten digits.
        inline bool gb_nhs_ok(const std::string& d)
        {
            if (d.size() != 10 || !id_all_digits(d)) return false;
            int sum = 0;
            for (std::size_t i = 0; i < 10; ++i) sum += static_cast<int>(10 - i) * id_digit(d, i);
            return sum % 11 == 0;
        }

        // Canadian social insurance number (SIN, NAS): Luhn, not starting with 0 or 8.
        inline bool ca_sin_ok(const std::string& d)
        {
            return d.size() == 9 && d[0] != '0' && d[0] != '8' && luhn_ok(d);
        }

        // Brazilian individual taxpayer number (CPF): two mod 11 check digits.
        inline bool br_cpf_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d) || d.find_first_not_of('0') == std::string::npos) return false;
            int s1 = 0, s2 = 0;
            for (std::size_t i = 0; i < 9; ++i)
            {
                s1 += static_cast<int>(10 - i) * id_digit(d, i);
                s2 += static_cast<int>(11 - i) * id_digit(d, i);
            }
            const int d1 = (11 - s1 % 11) % 11 % 10;
            const int d2 = (11 - (s2 + 2 * d1) % 11) % 11 % 10;
            return d1 == id_digit(d, 9) && d2 == id_digit(d, 10);
        }

        // Indian Aadhaar number: twelve digits, Verhoeff, not starting with 0 or 1.
        inline bool in_aadhaar_ok(const std::string& d)
        {
            if (d.size() != 12 || !id_all_digits(d) || d[0] < '2') return false;
            if (std::string(d.rbegin(), d.rend()) == d) return false;
            return verhoeff_digit(d.substr(0, 11)) == id_digit(d, 11);
        }

        // Mexican population registry code (CURP).
        inline bool mx_curp_ok(const std::string& s)
        {
            if (s.size() != 18) return false;
            for (std::size_t i = 0; i < 18; ++i)
            {
                const char c = s[i];
                const bool letter = c >= 'A' && c <= 'Z', digit = c >= '0' && c <= '9';
                if ((i < 4 || (i >= 10 && i < 16)) && !letter) return false;
                if (i >= 4 && i < 10 && !digit) return false;
                if (i == 16 && !letter && !digit) return false;
                if (i == 17 && !digit) return false;
            }
            if (s[10] != 'H' && s[10] != 'M') return false;
            static const char* states = "AS BC BS CC CL CM CS CH DF DG GT GR HG JC MC MN MS NT NL OC PL QT QR SP SL SR TC "
                                        "TS TL VZ YN ZS NE";
            if (std::string(states).find(s.substr(11, 2)) == std::string::npos) return false;
            const int year = (s[16] >= '0' && s[16] <= '9' ? 1900 : 2000) + static_cast<int>(id_number(s, 4, 6));
            if (!valid_date_digits(year, static_cast<int>(id_number(s, 6, 8)), static_cast<int>(id_number(s, 8, 10))))
                return false;
            static const char* alphabet = "0123456789ABCDEFGHIJKLMN&OPQRSTUVWXYZ";
            int sum = 0;
            for (std::size_t i = 0; i < 17; ++i)
            {
                int v = 0;
                for (int k = 0; alphabet[k]; ++k) if (alphabet[k] == s[i]) v = k;
                sum += v * static_cast<int>(18 - i);
            }
            return (10 - sum % 10) % 10 == id_digit(s, 17);
        }

        // Chilean national number (RUT, RUN): digits and a check digit or K.
        inline bool cl_rut_ok(const std::string& s)
        {
            if ((s.size() != 8 && s.size() != 9) || !id_all_digits(s, 0, s.size() - 1)) return false;
            int sum = 0;
            for (std::size_t i = 0; i + 1 < s.size(); ++i)
                sum += id_digit(s, s.size() - 2 - i) * static_cast<int>(4 + (5 - i % 6) % 6);
            return s.back() == "0123456789K"[sum % 11];
        }

        // Argentinian tax and labour numbers (CUIT, CUIL).
        inline bool ar_cuit_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d)) return false;
            static const char* types[] = { "20", "23", "24", "27", "30", "33", "34", "50", "51", "55" };
            bool known = false;
            for (const char* t : types) if (d.compare(0, 2, t) == 0) known = true;
            if (!known) return false;
            static const int w[10] = { 5, 4, 3, 2, 7, 6, 5, 4, 3, 2 };
            return "012345678990"[11 - weighted_sum(d, w, 10) % 11] == d[10];
        }

        // Chinese resident identity card number: ISO 7064 MOD 11-2 with X, and a birth date.
        inline bool cn_resident_id_ok(const std::string& s)
        {
            if (s.size() != 18 || !id_all_digits(s, 0, 17)) return false;
            int sum = 0;
            for (std::size_t i = 0; i < 17; ++i)
            {
                int weight = 1;
                for (std::size_t k = 0; k < 17 - i; ++k) weight = weight * 2 % 11;
                sum += weight * id_digit(s, i);
            }
            const int check = (12 - sum % 11) % 11;
            if (s[17] != (check == 10 ? 'X' : static_cast<char>('0' + check))) return false;
            return valid_date_digits(static_cast<int>(id_number(s, 6, 10)), static_cast<int>(id_number(s, 10, 12)),
                                     static_cast<int>(id_number(s, 12, 14)));
        }

        // Turkish identification number (T.C. Kimlik No).
        inline bool tr_kimlik_ok(const std::string& d)
        {
            if (d.size() != 11 || !id_all_digits(d) || d[0] == '0') return false;
            int weighted = 0, plain = 0;
            for (std::size_t i = 0; i < 9; ++i)
            {
                weighted += (i % 2 == 0 ? 3 : 1) * id_digit(d, i);
                plain += id_digit(d, i);
            }
            const int c1 = (10 - weighted % 10) % 10, c2 = (c1 + plain) % 10;
            return c1 == id_digit(d, 9) && c2 == id_digit(d, 10);
        }

        // Israeli identity number (mispar zehut): nine digits, Luhn.
        inline bool il_idnr_ok(const std::string& d)
        {
            return d.size() == 9 && d.find_first_not_of('0') != std::string::npos && luhn_ok(d);
        }

        // South African identity number: YYMMDD, citizenship digit, Luhn.
        inline bool za_idnr_ok(const std::string& d)
        {
            if (d.size() != 13 || !luhn_ok(d) || (d[10] != '0' && d[10] != '1')) return false;
            return valid_date_digits(2000, static_cast<int>(id_number(d, 2, 4)), static_cast<int>(id_number(d, 4, 6)));
        }

        // Australian tax file number (TFN).
        inline bool au_tfn_ok(const std::string& d)
        {
            if ((d.size() != 8 && d.size() != 9) || !id_all_digits(d)) return false;
            static const int w[9] = { 1, 4, 3, 7, 5, 8, 6, 9, 10 };
            return weighted_sum(d, w, d.size()) % 11 == 0;
        }

        // South Korean resident registration number (RRN).
        inline bool kr_rrn_ok(const std::string& d)
        {
            if (d.size() != 13 || !id_all_digits(d)) return false;
            static const int w[12] = { 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5 };
            if ((11 - weighted_sum(d, w, 12) % 11) % 10 != id_digit(d, 12)) return false;
            const char c = d[6];
            int year = static_cast<int>(id_number(d, 0, 2));
            if (c == '1' || c == '2' || c == '5' || c == '6') year += 1900;
            else if (c == '3' || c == '4' || c == '7' || c == '8') year += 2000;
            else year += 1800;
            return id_number(d, 7, 9) <= 96 &&
                   valid_date_digits(year, static_cast<int>(id_number(d, 2, 4)), static_cast<int>(id_number(d, 4, 6)));
        }

        // Japanese individual number (My Number).
        inline bool jp_my_number_ok(const std::string& d)
        {
            if (d.size() != 12 || !id_all_digits(d)) return false;
            static const int w[11] = { 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2 };
            return (11 - weighted_sum(d, w, 11) % 11) % 11 % 10 == id_digit(d, 11);
        }

        // Icelandic identity code (kennitala).
        inline bool is_kennitala_ok(const std::string& d)
        {
            if (d.size() != 10 || !id_all_digits(d) || (d[9] != '0' && d[9] != '9')) return false;
            static const int w[10] = { 3, 2, 7, 6, 5, 4, 3, 2, 1, 0 };
            if (weighted_sum(d, w, 10) % 11 != 0) return false;
            int day = static_cast<int>(id_number(d, 0, 2));
            if (day >= 40) day -= 40;
            const int year = (d[9] == '9' ? 1900 : 2000) + static_cast<int>(id_number(d, 4, 6));
            return valid_date_digits(year, static_cast<int>(id_number(d, 2, 4)), day);
        }

        // American individual taxpayer identification number (ITIN).
        inline bool us_itin_ok(const std::string& d)
        {
            if (d.size() != 9 || !id_all_digits(d) || d[0] != '9') return false;
            const int group = static_cast<int>(id_number(d, 3, 5));
            return (group >= 70 && group <= 88) || (group >= 90 && group <= 92) || (group >= 94 && group <= 99);
        }
    }
}

#endif // DLIB_PRIVACY_ID_CHECKS_H_
