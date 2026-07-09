#ifndef UTEC_CRYPTO_MULTILAYER_CIPHER_DETECTOR_H
#define UTEC_CRYPTO_MULTILAYER_CIPHER_DETECTOR_H

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "utec/crypto/CipherDetector.h"

namespace utec::crypto {

struct LayerCandidate {
    std::string chain;
    double score = 0.0;
    std::string possible_plaintext;
};

class MultiLayerCipherDetector {
public:
    explicit MultiLayerCipherDetector(const CipherDetector& base_detector) : base_(base_detector) {
        chains_ = {
            {"Caesar", "Base64"},
            {"ROT13", "Base64"},
            {"Atbash", "Base64"},
            {"Caesar", "Hexadecimal"},
            {"ROT13", "Hexadecimal"},
            {"Atbash", "Hexadecimal"},
            {"Morse", "Base64"},
            {"RailFence", "Base64"}
        };
    }

    std::vector<LayerCandidate> analyze_double(const std::string& encrypted, int top = 3) const {
        if (encrypted.empty()) throw std::invalid_argument("texto vacio");
        std::vector<LayerCandidate> out;
        for (const auto& chain : chains_) {
            const std::string& first = chain.first;
            const std::string& second = chain.second;

            std::string after_second;
            double outer_score = reverse_outer(encrypted, second, after_second);
            if (outer_score <= 0.0) continue;

            auto inner_report = base_.analyze(after_second);
            double inner_score = probability_for(inner_report, first);
            std::string plain = decode_first(after_second, first);
            double read_score = readability_score(plain);

            double score = 0.55 * outer_score + 0.30 * inner_score + 0.15 * read_score;
            out.push_back({first + " -> " + second, score, plain});
        }
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){ return a.score > b.score; });
        if (static_cast<int>(out.size()) > top) out.resize(top);
        normalize_scores(out);
        return out;
    }

    static void print_double_report(const std::vector<LayerCandidate>& candidates) {
        std::cout << "\n=== Top cadenas de cifrado doble ===\n";
        if (candidates.empty()) {
            std::cout << "No se encontraron combinaciones compatibles.\n";
            return;
        }
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            std::cout << (i + 1) << ". " << std::setw(24) << std::left << candidates[i].chain
                      << " prob=" << std::fixed << std::setprecision(2) << candidates[i].score * 100.0 << "%\n";
            std::cout << "   Posible mensaje: " << shorten(candidates[i].possible_plaintext, 180) << "\n";
        }
        std::cout << "\nNota: si el cifrado requiere clave desconocida, el texto descifrado puede ser aproximado.\n";
    }

private:
    const CipherDetector& base_;
    std::vector<std::pair<std::string, std::string>> chains_;

    static std::string trim_spaces(const std::string& s) {
        std::string r;
        for (unsigned char c : s) if (!std::isspace(c)) r.push_back(static_cast<char>(c));
        return r;
    }

    static std::string caesar_decode(const std::string& s, int shift) {
        std::string r = s;
        shift = ((shift % 26) + 26) % 26;
        for (char& ch : r) {
            unsigned char uc = static_cast<unsigned char>(ch);
            if (std::isalpha(uc)) {
                char base = std::isupper(uc) ? 'A' : 'a';
                ch = static_cast<char>(base + (ch - base - shift + 26) % 26);
            }
        }
        return r;
    }

    static std::string rot13(const std::string& s) { return caesar_decode(s, 13); }

    static std::string atbash(const std::string& s) {
        std::string r = s;
        for (char& ch : r) {
            unsigned char uc = static_cast<unsigned char>(ch);
            if (std::isalpha(uc)) {
                char base = std::isupper(uc) ? 'A' : 'a';
                ch = static_cast<char>(base + 25 - (ch - base));
            }
        }
        return r;
    }

    static int from_hex_digit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + c - 'a';
        if (c >= 'A' && c <= 'F') return 10 + c - 'A';
        return -1;
    }

    static bool hex_decode(const std::string& s, std::string& out) {
        std::string t = trim_spaces(s);
        if (t.size() < 2 || t.size() % 2 != 0) return false;
        out.clear();
        for (std::size_t i = 0; i < t.size(); i += 2) {
            int a = from_hex_digit(t[i]);
            int b = from_hex_digit(t[i + 1]);
            if (a < 0 || b < 0) return false;
            out.push_back(static_cast<char>((a << 4) | b));
        }
        return true;
    }

    static int base64_value(char c) {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return 26 + c - 'a';
        if (c >= '0' && c <= '9') return 52 + c - '0';
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    }

    static bool base64_decode(const std::string& s, std::string& out) {
        std::string t = trim_spaces(s);
        if (t.size() < 4 || t.size() % 4 != 0) return false;
        out.clear();
        int val = 0, valb = -8;
        for (char c : t) {
            if (c == '=') break;
            int d = base64_value(c);
            if (d == -1) return false;
            val = (val << 6) + d;
            valb += 6;
            if (valb >= 0) {
                out.push_back(static_cast<char>((val >> valb) & 0xFF));
                valb -= 8;
            }
        }
        return !out.empty();
    }

    static std::string morse_decode(const std::string& s) {
        static const std::map<std::string, char> inv = {
            {".-",'A'},{"-...",'B'},{"-.-.",'C'},{"-..",'D'},{".",'E'},{"..-.",'F'},{"--.",'G'},
            {"....",'H'},{"..",'I'},{".---",'J'},{"-.-",'K'},{".-..",'L'},{"--",'M'},{"-.",'N'},
            {"---",'O'},{".--.",'P'},{"--.-",'Q'},{".-.",'R'},{"...",'S'},{"-",'T'},{"..-",'U'},
            {"...-",'V'},{".--",'W'},{"-..-",'X'},{"-.--",'Y'},{"--..",'Z'},{"-----",'0'},{".----",'1'},
            {"..---",'2'},{"...--",'3'},{"....-",'4'},{".....",'5'},{"-....",'6'},{"--...",'7'},
            {"---..",'8'},{"----.",'9'}
        };
        std::string out, token;
        std::istringstream is(s);
        while (is >> token) {
            if (token == "/") { out.push_back(' '); continue; }
            auto it = inv.find(token);
            if (it != inv.end()) out.push_back(it->second);
        }
        return out;
    }

    static std::string rail_fence_decode_2(const std::string& s) {
        std::size_t a_len = (s.size() + 1) / 2;
        std::string a = s.substr(0, a_len);
        std::string b = s.substr(a_len);
        std::string out;
        for (std::size_t i = 0; i < a_len; ++i) {
            out.push_back(a[i]);
            if (i < b.size()) out.push_back(b[i]);
        }
        return out;
    }

    static double printable_ratio(const std::string& s) {
        if (s.empty()) return 0.0;
        int ok = 0;
        for (unsigned char c : s) if (std::isprint(c) || std::isspace(c)) ++ok;
        return static_cast<double>(ok) / static_cast<double>(s.size());
    }

    static double reverse_outer(const std::string& encrypted, const std::string& outer, std::string& decoded) {
        bool ok = false;
        if (outer == "Base64") ok = base64_decode(encrypted, decoded);
        else if (outer == "Hexadecimal") ok = hex_decode(encrypted, decoded);
        if (!ok) return 0.0;
        return 0.70 + 0.30 * printable_ratio(decoded);
    }

    static double probability_for(const AnalysisReport& r, const std::string& label) {
        for (const auto& p : r.predictions) {
            if (p.label == label) return p.probability;
            if (label == "Hexadecimal" && p.label == "Hexadecimal") return p.probability;
        }
        return 0.0;
    }

    static std::string best_caesar_decode(const std::string& s) {
        std::string best = s;
        double best_score = -1.0;
        for (int sh = 1; sh <= 25; ++sh) {
            std::string cand = caesar_decode(s, sh);
            double sc = readability_score(cand);
            if (sc > best_score) { best_score = sc; best = cand; }
        }
        return best;
    }

    static std::string decode_first(const std::string& inner, const std::string& first) {
        if (first == "Caesar") return best_caesar_decode(inner);
        if (first == "ROT13") return rot13(inner);
        if (first == "Atbash") return atbash(inner);
        if (first == "Morse") return morse_decode(inner);
        if (first == "RailFence") return rail_fence_decode_2(inner);
        return inner;
    }

    static double readability_score(const std::string& s) {
        if (s.empty()) return 0.0;
        double printable = printable_ratio(s);
        int letters = 0, spaces = 0, vowels = 0;
        for (unsigned char c : s) {
            char u = static_cast<char>(std::toupper(c));
            if (std::isalpha(c)) ++letters;
            if (std::isspace(c)) ++spaces;
            if (u=='A'||u=='E'||u=='I'||u=='O'||u=='U') ++vowels;
        }
        double n = static_cast<double>(s.size());
        double letter_ratio = letters / n;
        double space_ratio = spaces / n;
        double vowel_ratio = letters ? static_cast<double>(vowels) / letters : 0.0;
        double english_bonus = 0.0;
        std::string up = s;
        for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        const std::vector<std::string> common = {"THE", "AND", "HELLO", "WORLD", "ATTACK", "PROJECT", "SECURITY", "CRYPTO"};
        for (const auto& w : common) if (up.find(w) != std::string::npos) english_bonus += 0.08;
        return std::clamp(0.35*printable + 0.25*letter_ratio + 0.20*space_ratio + 0.20*(1.0-std::abs(vowel_ratio-0.40)) + english_bonus, 0.0, 1.0);
    }

    static void normalize_scores(std::vector<LayerCandidate>& c) {
        double sum = 0.0;
        for (const auto& x : c) sum += std::max(0.0, x.score);
        if (sum <= 0.0) return;
        for (auto& x : c) x.score = std::max(0.0, x.score) / sum;
    }

    static std::string shorten(const std::string& s, std::size_t n) {
        std::string clean;
        for (unsigned char c : s) clean.push_back((std::isprint(c) || std::isspace(c)) ? static_cast<char>(c) : '?');
        if (clean.size() <= n) return clean;
        return clean.substr(0, n) + "...";
    }
};

} // namespace utec::crypto

#endif
