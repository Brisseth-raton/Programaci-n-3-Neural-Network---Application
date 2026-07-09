//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/io/TextDatasetLoader.h"

#include <fstream>
#include <filesystem>

using namespace utec::io;

int main() {

    TestRunner test;

    std::filesystem::create_directories("artifacts");

    // =====================================
    // Crear archivos de prueba
    // =====================================

    {
        std::ofstream samples("artifacts/samples.txt");
        samples << "ABC\n";
        samples << "DEF\n";

        std::ofstream labels("artifacts/labels.txt");
        labels << "Caesar\n";
        labels << "Vigenere\n";
    }

    // =====================================
    // Leer líneas
    // =====================================

    auto lines = TextDatasetLoader::read_lines("artifacts/samples.txt");

    CHECK(test, lines.size() == 2);
    CHECK(test, lines[0] == "ABC");
    CHECK(test, lines[1] == "DEF");

    // =====================================
    // Cargar dataset
    // =====================================

    auto ds = TextDatasetLoader::from_txt(
        "artifacts/samples.txt",
        "artifacts/labels.txt"
    );

    CHECK(test, ds.samples.size() == 2);
    CHECK(test, ds.labels.size() == 2);

    CHECK(test, ds.samples[0] == "ABC");
    CHECK(test, ds.labels[1] == "Vigenere");

    // =====================================
    // Archivo inexistente
    // =====================================

    CHECK_THROWS(test,
        TextDatasetLoader::read_lines("archivo_inexistente.txt")
    );

    // =====================================
    // Cantidad distinta de líneas
    // =====================================

    {
        std::ofstream labels("artifacts/bad_labels.txt");
        labels << "SoloUna\n";
    }

    CHECK_THROWS(test,
        TextDatasetLoader::from_txt(
            "artifacts/samples.txt",
            "artifacts/bad_labels.txt"
        )
    );

    // =====================================
    // Archivo vacío
    // =====================================

    {
        std::ofstream empty("artifacts/empty.txt");
    }

    CHECK_THROWS(test,
        TextDatasetLoader::from_txt(
            "artifacts/empty.txt",
            "artifacts/empty.txt"
        )
    );

    return test.summary();
}