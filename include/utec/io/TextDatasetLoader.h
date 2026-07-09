#ifndef UTEC_IO_TEXT_DATASET_LOADER_H
#define UTEC_IO_TEXT_DATASET_LOADER_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::io {

struct TextDataset {
    std::vector<std::string> samples;
    std::vector<std::string> labels;
};

class TextDatasetLoader {
public:
    static std::vector<std::string> read_lines(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("No se pudo abrir: " + path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) lines.push_back(line);
        }
        return lines;
    }

    static TextDataset from_txt(const std::string& samples_path, const std::string& labels_path) {
        TextDataset ds{read_lines(samples_path), read_lines(labels_path)};
        if (ds.samples.empty()) throw std::invalid_argument("El archivo de textos cifrados esta vacio");
        if (ds.samples.size() != ds.labels.size()) {
            throw std::invalid_argument("Los TXT deben tener la misma cantidad de filas");
        }
        return ds;
    }
};

} // namespace utec::io

#endif
