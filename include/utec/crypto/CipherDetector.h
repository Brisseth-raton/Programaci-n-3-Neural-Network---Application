#ifndef UTEC_CRYPTO_CIPHER_DETECTOR_H
#define UTEC_CRYPTO_CIPHER_DETECTOR_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "utec/io/TextDatasetLoader.h"
#include "utec/monitoring/TrainingMonitor.h"

namespace utec::crypto {

struct Prediction {
    std::string label;
    double probability = 0.0;
};

struct AnalysisReport {
    double entropy = 0.0;
    double index_of_coincidence = 0.0;
    double letter_ratio = 0.0;
    double digit_ratio = 0.0;
    double symbol_ratio = 0.0;
    double hex_ratio = 0.0;
    bool looks_base64 = false;
    bool looks_hex = false;
    bool looks_morse = false;
    std::vector<Prediction> predictions;
};

struct TrainSummary {
    int epochs = 0;
    int examples = 0;
    double final_loss = 0.0;
    double final_accuracy = 0.0;
};

class CipherDetector {
public:
    CipherDetector() {
        labels_ = {
            "Caesar", "ROT13", "Vigenere", "XOR", "Base64", "Hexadecimal",
            "Morse", "Atbash", "RailFence", "ColumnarTransposition",
            "SHA-256/SHA-512", "AES/RSA-like"
        };
        initialize_network();
    }

    const std::vector<std::string>& labels() const { return labels_; }
    int trained_examples() const { return trained_examples_; }
    const monitoring::TrainingMonitor& monitor() const { return monitor_; }

    TrainSummary train_from_txt(const std::string& samples_path, const std::string& labels_path,
                                int epochs = 20, double lr = 0.03) {
        auto ds = io::TextDatasetLoader::from_txt(samples_path, labels_path);
        std::vector<std::pair<std::string, int>> data;
        data.reserve(ds.samples.size());
        for (std::size_t i = 0; i < ds.samples.size(); ++i) data.push_back({ds.samples[i], label_to_id(ds.labels[i])});
        return train_dataset(data, epochs, lr);
    }

    TrainSummary train_synthetic(int samples_per_class = 150, int epochs = 10, double lr = 0.03) {
        if (samples_per_class <= 0) throw std::invalid_argument("samples_per_class invalido");
        std::mt19937 rng(20260702);
        std::vector<std::pair<std::string, int>> data;
        for (int c = 0; c < static_cast<int>(labels_.size()); ++c) {
            for (int i = 0; i < samples_per_class; ++i) data.push_back({make_sample(c, rng), c});
        }
        return train_dataset(data, epochs, lr);
    }

    void train_one(const std::string& text, int label_id, double lr = 0.03) {
        if (label_id < 0 || label_id >= static_cast<int>(labels_.size())) throw std::invalid_argument("label_id invalido");
        train_example(extract_features(text), label_id, lr);
        ++trained_examples_;
    }

    AnalysisReport analyze(const std::string& text) const {
        if (text.empty()) throw std::invalid_argument("texto vacio");
        auto x = extract_features(text);
        auto p = predict_proba(x);
        apply_crypto_rules(text, p);
        normalize(p);

        AnalysisReport r;
        r.entropy = shannon_entropy(text);
        r.index_of_coincidence = index_of_coincidence(text);
        r.letter_ratio = ratio_letters(text);
        r.digit_ratio = ratio_digits(text);
        r.symbol_ratio = ratio_symbols(text);
        r.hex_ratio = ratio_hex(text);
        r.looks_base64 = is_base64_like(text);
        r.looks_hex = is_hex_like(text);
        r.looks_morse = is_morse_like(text);
        for (std::size_t i = 0; i < labels_.size(); ++i) r.predictions.push_back({labels_[i], p[i]});
        std::sort(r.predictions.begin(), r.predictions.end(), [](const auto& a, const auto& b){ return a.probability > b.probability; });
        return r;
    }

    void save_model(const std::string& path) const {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream out(path);
        if (!out) throw std::runtime_error("no se pudo guardar el modelo");
        out << "CRYPTOVISION_AI_V1\n";
        out << input_size_ << ' ' << hidden_size_ << ' ' << labels_.size() << ' ' << trained_examples_ << '\n';
        write_matrix(out, W1_); write_vector(out, b1_); write_matrix(out, W2_); write_vector(out, b2_);
    }

    void load_model(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("no se pudo cargar el modelo");
        std::string magic; std::getline(in, magic);
        if (magic != "CRYPTOVISION_AI_V1") throw std::runtime_error("archivo de modelo invalido");
        std::size_t in_size = 0, hidden = 0, classes = 0;
        in >> in_size >> hidden >> classes >> trained_examples_;
        if (in_size != input_size_ || hidden != hidden_size_ || classes != labels_.size()) throw std::runtime_error("modelo incompatible");
        read_matrix(in, W1_); read_vector(in, b1_); read_matrix(in, W2_); read_vector(in, b2_);
    }

    void export_history_csv(const std::string& path) const { monitor_.write_csv(path); }

    int label_to_id(std::string label) const {
        label = normalize_label(label);
        static const std::unordered_map<std::string, int> aliases = {
            {"caesar",0},{"cesar",0},{"césar",0},{"shift",0},{"cifradocesar",0},
            {"rot13",1},{"rot-13",1},
            {"vigenere",2},{"vigenère",2},{"vigenerecipher",2},
            {"xor",3},{"xorcipher",3},
            {"base64",4},{"base 64",4},{"b64",4},
            {"hex",5},{"hexadecimal",5},
            {"morse",6},{"codigomorse",6},{"morsecode",6},
            {"atbash",7},
            {"railfence",8},{"rail fence",8},{"zigzag",8},
            {"columnar",9},{"columnartransposition",9},{"transposicioncolumnar",9},{"transposicióncolumnar",9},
            {"sha",10},{"sha256",10},{"sha-256",10},{"sha512",10},{"sha-512",10},{"sha-2",10},{"sha-3",10},
            {"aes",11},{"rsa",11},{"random",11},{"randomlike",11},{"aesrsa",11},{"aes/rsa-like",11}
        };
        auto it = aliases.find(label);
        if (it == aliases.end()) throw std::invalid_argument("Etiqueta no reconocida: " + label);
        return it->second;
    }

private:
    static constexpr std::size_t input_size_ = 42;
    static constexpr std::size_t hidden_size_ = 32;

    std::vector<std::string> labels_;
    std::vector<std::vector<double>> W1_, W2_;
    std::vector<double> b1_, b2_;
    int trained_examples_ = 0;
    monitoring::TrainingMonitor monitor_;

    void initialize_network() {
        std::mt19937 rng(42);
        std::normal_distribution<double> nd(0.0, 0.08);
        W1_.assign(hidden_size_, std::vector<double>(input_size_));
        b1_.assign(hidden_size_, 0.0);
        W2_.assign(labels_.size(), std::vector<double>(hidden_size_));
        b2_.assign(labels_.size(), 0.0);
        for (auto& row : W1_) for (double& v : row) v = nd(rng);
        for (auto& row : W2_) for (double& v : row) v = nd(rng);
    }

    TrainSummary train_dataset(std::vector<std::pair<std::string, int>>& data, int epochs, double lr) {
        if (data.empty()) throw std::invalid_argument("dataset vacio");
        if (epochs <= 0) throw std::invalid_argument("epochs debe ser mayor a cero");
        std::mt19937 rng(1234);
        TrainSummary summary{epochs, static_cast<int>(data.size()), 0.0, 0.0};
        monitor_ = monitoring::TrainingMonitor{};
        for (int e = 1; e <= epochs; ++e) {
            auto start = std::chrono::steady_clock::now();
            std::shuffle(data.begin(), data.end(), rng);
            double loss_sum = 0.0;
            int correct = 0;
            for (const auto& [text, y] : data) {
                auto x = extract_features(text);
                auto p_before = predict_proba(x);
                if (argmax(p_before) == y) ++correct;
                loss_sum += -std::log(std::max(1e-12, p_before[static_cast<std::size_t>(y)]));
                train_example(x, y, lr);
            }
            auto end = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            summary.final_loss = loss_sum / static_cast<double>(data.size());
            summary.final_accuracy = static_cast<double>(correct) / static_cast<double>(data.size());
            monitor_.on_epoch_end(e, summary.final_loss, summary.final_accuracy, ms);
            std::cout << "Epoca " << e << "/" << epochs
                      << " | loss=" << std::fixed << std::setprecision(4) << summary.final_loss
                      << " | accuracy=" << std::setprecision(2) << summary.final_accuracy * 100.0 << "%\n";
        }
        trained_examples_ += static_cast<int>(data.size()) * epochs;
        return summary;
    }

    void train_example(const std::vector<double>& x, int y, double lr) {
        std::vector<double> z1(hidden_size_), h(hidden_size_);
        for (std::size_t i = 0; i < hidden_size_; ++i) {
            z1[i] = b1_[i] + dot(W1_[i], x);
            h[i] = std::max(0.0, z1[i]);
        }
        std::vector<double> z2(labels_.size());
        for (std::size_t k = 0; k < labels_.size(); ++k) z2[k] = b2_[k] + dot(W2_[k], h);
        auto p = softmax(z2);

        std::vector<double> dz2 = p;
        dz2[static_cast<std::size_t>(y)] -= 1.0;
        std::vector<double> dh(hidden_size_, 0.0);
        for (std::size_t k = 0; k < labels_.size(); ++k) {
            for (std::size_t i = 0; i < hidden_size_; ++i) dh[i] += dz2[k] * W2_[k][i];
        }

        for (std::size_t k = 0; k < labels_.size(); ++k) {
            for (std::size_t i = 0; i < hidden_size_; ++i) W2_[k][i] -= lr * dz2[k] * h[i];
            b2_[k] -= lr * dz2[k];
        }
        for (std::size_t i = 0; i < hidden_size_; ++i) {
            double dz1 = (z1[i] > 0.0 ? dh[i] : 0.0);
            for (std::size_t j = 0; j < input_size_; ++j) W1_[i][j] -= lr * dz1 * x[j];
            b1_[i] -= lr * dz1;
        }
    }

    std::vector<double> predict_proba(const std::vector<double>& x) const {
        std::vector<double> h(hidden_size_);
        for (std::size_t i = 0; i < hidden_size_; ++i) h[i] = std::max(0.0, b1_[i] + dot(W1_[i], x));
        std::vector<double> z(labels_.size());
        for (std::size_t k = 0; k < labels_.size(); ++k) z[k] = b2_[k] + dot(W2_[k], h);
        return softmax(z);
    }

    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }

    static int argmax(const std::vector<double>& p) {
        return static_cast<int>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
    }

    static std::string normalize_label(std::string s) {
        std::string out;
        for (unsigned char c : s) {
            if (std::isspace(c) || c == '_' || c == '-') continue;
            out.push_back(static_cast<char>(std::tolower(c)));
        }
        return out;
    }

    static double safe_div(double a, double b) { return b == 0.0 ? 0.0 : a / b; }
    static std::string upper_letters(const std::string& s) {
        std::string r;
        for (unsigned char c : s) if (std::isalpha(c)) r.push_back(static_cast<char>(std::toupper(c)));
        return r;
    }
    static double ratio_letters(const std::string& s) { return safe_div(std::count_if(s.begin(), s.end(), [](unsigned char c){return std::isalpha(c);}), s.size()); }
    static double ratio_digits(const std::string& s) { return safe_div(std::count_if(s.begin(), s.end(), [](unsigned char c){return std::isdigit(c);}), s.size()); }
    static double ratio_spaces(const std::string& s) { return safe_div(std::count_if(s.begin(), s.end(), [](unsigned char c){return std::isspace(c);}), s.size()); }
    static double ratio_symbols(const std::string& s) { return safe_div(std::count_if(s.begin(), s.end(), [](unsigned char c){return !std::isalnum(c) && !std::isspace(c);}), s.size()); }
    static double ratio_hex(const std::string& s) { return safe_div(std::count_if(s.begin(), s.end(), [](unsigned char c){return std::isxdigit(c);}), s.size()); }

    static double shannon_entropy(const std::string& s) {
        if (s.empty()) return 0.0;
        std::array<int, 256> cnt{};
        for (unsigned char c : s) cnt[c]++;
        double h = 0.0;
        for (int n : cnt) if (n) { double p = static_cast<double>(n) / s.size(); h -= p * std::log2(p); }
        return h;
    }

    static double index_of_coincidence(const std::string& s) {
        std::string u = upper_letters(s);
        if (u.size() < 2) return 0.0;
        std::array<int, 26> cnt{};
        for (char c : u) cnt[c - 'A']++;
        double num = 0.0;
        for (int n : cnt) num += n * (n - 1);
        return num / (static_cast<double>(u.size()) * (u.size() - 1));
    }

    static bool is_hex_like(const std::string& s) {
        std::string t;
        for (unsigned char c : s) if (!std::isspace(c)) t.push_back(static_cast<char>(c));
        return t.size() >= 16 && t.size() % 2 == 0 && ratio_hex(t) > 0.98;
    }

    static bool is_base64_like(const std::string& s) {
        std::string t;
        for (unsigned char c : s) if (!std::isspace(c)) t.push_back(static_cast<char>(c));
        if (t.size() < 8 || t.size() % 4 != 0) return false;
        for (unsigned char c : t) if (!(std::isalnum(c) || c == '+' || c == '/' || c == '=')) return false;
        return true;
    }

    static bool is_morse_like(const std::string& s) {
        int total = 0, valid = 0;
        for (unsigned char c : s) {
            if (!std::isspace(c)) ++total;
            if (c == '.' || c == '-' || c == '/' || std::isspace(c)) ++valid;
        }
        return total >= 4 && safe_div(valid, s.size()) > 0.95;
    }

    std::vector<double> extract_features(const std::string& s) const {
        std::vector<double> x;
        x.reserve(input_size_);
        x.push_back(std::min(static_cast<double>(s.size()) / 256.0, 1.0));
        x.push_back(ratio_letters(s)); x.push_back(ratio_digits(s)); x.push_back(ratio_spaces(s));
        x.push_back(ratio_symbols(s)); x.push_back(ratio_hex(s));
        x.push_back(shannon_entropy(s) / 8.0); x.push_back(index_of_coincidence(s));
        x.push_back(is_base64_like(s) ? 1.0 : 0.0); x.push_back(is_hex_like(s) ? 1.0 : 0.0);
        x.push_back(is_morse_like(s) ? 1.0 : 0.0); x.push_back(s.find('=') != std::string::npos ? 1.0 : 0.0);
        std::string u = upper_letters(s);
        std::array<int, 26> cnt{}; for (char c : u) cnt[c - 'A']++;
        for (int i = 0; i < 26; ++i) x.push_back(safe_div(cnt[i], u.size()));
        x.push_back(safe_div(std::count(u.begin(), u.end(), 'E'), u.size()));
        x.push_back(safe_div(std::count(u.begin(), u.end(), 'T'), u.size()));
        x.push_back(safe_div(std::count(u.begin(), u.end(), 'A'), u.size()));
        x.push_back(safe_div(std::count(u.begin(), u.end(), 'O'), u.size()));
        while (x.size() < input_size_) x.push_back(0.0);
        return x;
    }

    static std::vector<double> softmax(std::vector<double> z) {
        double m = *std::max_element(z.begin(), z.end());
        double sum = 0.0;
        for (double& v : z) { v = std::exp(v - m); sum += v; }
        for (double& v : z) v /= sum;
        return z;
    }
    static void normalize(std::vector<double>& p) {
        double s = std::accumulate(p.begin(), p.end(), 0.0);
        if (s > 0) for (double& v : p) v /= s;
    }

    void apply_crypto_rules(const std::string& text, std::vector<double>& p) const {
        const int CAESAR=0, ROT13=1, VIG=2, XOR=3, B64=4, HEX=5, MORSE=6, ATBASH=7, RAIL=8, COL=9, SHA=10, RANDOM=11;
        if (is_morse_like(text)) p[MORSE] += 2.0;
        if (is_base64_like(text)) p[B64] += 1.4;
        if (is_hex_like(text)) {
            std::string t; for (unsigned char c : text) if (!std::isspace(c)) t.push_back(static_cast<char>(c));
            if (t.size() == 64 || t.size() == 128) p[SHA] += 2.2;
            else { p[HEX] += 1.0; if (shannon_entropy(t) > 3.2) p[RANDOM] += 0.7; }
        }
        if (ratio_symbols(text) > 0.35 && !is_morse_like(text)) p[XOR] += 0.8;
        const double ic = index_of_coincidence(text);
        if (ratio_letters(text) > 0.70 && ic > 0.055) { p[CAESAR] += 0.5; p[ROT13] += 0.4; p[ATBASH] += 0.4; p[RAIL] += 0.2; p[COL] += 0.2; }
        if (ratio_letters(text) > 0.70 && ic < 0.052) p[VIG] += 0.6;
    }

    static void write_vector(std::ofstream& out, const std::vector<double>& v) {
        out << v.size() << '\n'; for (double x : v) out << std::setprecision(17) << x << ' '; out << '\n';
    }
    static void read_vector(std::ifstream& in, std::vector<double>& v) {
        std::size_t n = 0; in >> n; v.assign(n, 0.0); for (double& x : v) in >> x;
    }
    static void write_matrix(std::ofstream& out, const std::vector<std::vector<double>>& M) {
        out << M.size() << ' ' << (M.empty()?0:M[0].size()) << '\n';
        for (const auto& row : M) { for (double x : row) out << std::setprecision(17) << x << ' '; out << '\n'; }
    }
    static void read_matrix(std::ifstream& in, std::vector<std::vector<double>>& M) {
        std::size_t r = 0, c = 0; in >> r >> c; M.assign(r, std::vector<double>(c));
        for (auto& row : M) for (double& x : row) in >> x;
    }

    static std::string caesar(const std::string& s, int shift) {
        std::string r=s; shift = ((shift%26)+26)%26;
        for(char& ch:r) if(std::isalpha(static_cast<unsigned char>(ch))) { char base=std::isupper(static_cast<unsigned char>(ch))?'A':'a'; ch=static_cast<char>(base+(ch-base+shift)%26); }
        return r;
    }
    static std::string atbash(const std::string& s) {
        std::string r=s;
        for(char& ch:r) if(std::isalpha(static_cast<unsigned char>(ch))) { char base=std::isupper(static_cast<unsigned char>(ch))?'A':'a'; ch=static_cast<char>(base+25-(ch-base)); }
        return r;
    }
    static std::string vigenere(const std::string& s, const std::string& key) {
        std::string r=s; int j=0;
        for(char& ch:r) if(std::isalpha(static_cast<unsigned char>(ch))) { char base=std::isupper(static_cast<unsigned char>(ch))?'A':'a'; int sh=std::toupper(key[j++%key.size()])- 'A'; ch=static_cast<char>(base+(ch-base+sh)%26); }
        return r;
    }
    static std::string xor_hex(const std::string& s, unsigned char key) {
        std::ostringstream os; for (unsigned char c : s) os << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c ^ key); return os.str();
    }
    static std::string hex_encode(const std::string& s) {
        std::ostringstream os; for (unsigned char c : s) os << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c); return os.str();
    }
    static std::string base64_encode(const std::string& in) {
        static const char* tbl="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out; int val=0, valb=-6;
        for (unsigned char c : in) { val=(val<<8)+c; valb+=8; while(valb>=0){ out.push_back(tbl[(val>>valb)&0x3F]); valb-=6; } }
        if(valb>-6) out.push_back(tbl[((val<<8)>>(valb+8))&0x3F]); while(out.size()%4) out.push_back('='); return out;
    }
    static std::string morse(const std::string& s) {
        static const std::map<char,std::string> m={{'A',".-"},{'B',"-..."},{'C',"-.-."},{'D',"-.."},{'E',"."},{'F',"..-."},{'G',"--."},{'H',"...."},{'I',".."},{'J',".---"},{'K',"-.-"},{'L',".-.."},{'M',"--"},{'N',"-."},{'O',"---"},{'P',".--."},{'Q',"--.-"},{'R',".-."},{'S',"..."},{'T',"-"},{'U',"..-"},{'V',"...-"},{'W',".--"},{'X',"-..-"},{'Y',"-.--"},{'Z',"--.."},{'0',"-----"},{'1',".----"},{'2',"..---"},{'3',"...--"},{'4',"....-"},{'5',"....."},{'6',"-...."},{'7',"--..."},{'8',"---.."},{'9',"----."}};
        std::string out; for (unsigned char c : s) { if (std::isspace(c)) out += "/ "; else { auto it=m.find(std::toupper(c)); if(it!=m.end()) out += it->second + " "; } } return out;
    }
    static std::string rail_fence(const std::string& s) { std::string a,b; bool top=true; for(char c:s){ (top?a:b).push_back(c); top=!top; } return a+b; }
    static std::string columnar(const std::string& s, int cols=5) { std::string out; for(int c=0;c<cols;++c) for(size_t i=c;i<s.size();i+=cols) out.push_back(s[i]); return out; }
    static std::string random_hex(std::mt19937& rng, int n) { std::uniform_int_distribution<int> d(0,15); const char* h="0123456789abcdef"; std::string s; for(int i=0;i<n;i++) s.push_back(h[d(rng)]); return s; }
    static std::string make_sample(int label, std::mt19937& rng) {
        static const std::vector<std::string> plain = {
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "ATTACK AT DAWN WITH ALL AVAILABLE FORCES",
            "PROGRAMMING THREE FINAL PROJECT NEURAL NETWORK", "CRYPTOGRAPHY IS THE SCIENCE OF SECRET WRITING",
            "HELLO WORLD THIS IS A SIMPLE MESSAGE", "SECURITY ANALYSIS FREQUENCY ENTROPY AND INDEX",
            "UNIVERSITY PROJECT USING C PLUS PLUS AND EIGEN", "CIPHER IDENTIFICATION HELPS CRYPTOANALYSIS"
        };
        std::uniform_int_distribution<int> pick(0, static_cast<int>(plain.size()-1));
        std::uniform_int_distribution<int> sh(1,25); std::uniform_int_distribution<int> key(1,255);
        std::string p = plain[pick(rng)];
        switch(label) {
            case 0: return caesar(p, sh(rng)); case 1: return caesar(p, 13); case 2: return vigenere(p, "LEMON");
            case 3: return xor_hex(p, static_cast<unsigned char>(key(rng))); case 4: return base64_encode(p); case 5: return hex_encode(p);
            case 6: return morse(p); case 7: return atbash(p); case 8: return rail_fence(p); case 9: return columnar(p);
            case 10: return random_hex(rng, (key(rng)%2)?64:128); default: return random_hex(rng, 96 + (key(rng)%96));
        }
    }
};

inline void print_report(const AnalysisReport& r, int top = 8) {
    std::cout << "\n=== Analisis criptografico ===\n";
    std::cout << "Entropia Shannon: " << std::fixed << std::setprecision(3) << r.entropy << "\n";
    std::cout << "Indice de coincidencia: " << std::fixed << std::setprecision(4) << r.index_of_coincidence << "\n";
    std::cout << "Letras: " << std::setprecision(2) << r.letter_ratio * 100 << "%  Digitos: " << r.digit_ratio * 100 << "%  Simbolos: " << r.symbol_ratio * 100 << "%\n";
    std::cout << "Firma Base64: " << (r.looks_base64 ? "si" : "no") << " | Firma Hex: " << (r.looks_hex ? "si" : "no") << " | Firma Morse: " << (r.looks_morse ? "si" : "no") << "\n";
    std::cout << "\n=== Probabilidades ===\n";
    for (int i = 0; i < std::min(top, static_cast<int>(r.predictions.size())); ++i) {
        std::cout << std::setw(24) << std::left << r.predictions[i].label << " "
                  << std::setw(6) << std::right << std::fixed << std::setprecision(2)
                  << r.predictions[i].probability * 100.0 << "%\n";
    }
}

} // namespace utec::crypto

#endif
