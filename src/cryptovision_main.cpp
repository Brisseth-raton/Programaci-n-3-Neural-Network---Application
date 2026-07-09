#include <iostream>
#include <string>

#include "utec/crypto/CipherDetector.h"
#include "utec/crypto/MultiLayerCipherDetector.h"

using utec::crypto::CipherDetector;
using utec::crypto::MultiLayerCipherDetector;
using utec::crypto::print_report;

static std::string read_line(const std::string& prompt) {
    std::cout << prompt;
    std::string s;
    std::getline(std::cin, s);
    return s;
}

static int read_int(const std::string& prompt, int def) {
    std::cout << prompt;
    std::string s;
    std::getline(std::cin, s);
    if (s.empty()) return def;
    try { return std::stoi(s); } catch (...) { return def; }
}

static double read_double(const std::string& prompt, double def) {
    std::cout << prompt;
    std::string s;
    std::getline(std::cin, s);
    if (s.empty()) return def;
    try { return std::stod(s); } catch (...) { return def; }
}

static void show_classes(const CipherDetector& detector) {
    std::cout << "\n=== Algoritmos simples soportados ===\n";
    const auto& labels = detector.labels();
    for (std::size_t i = 0; i < labels.size(); ++i) std::cout << i << ". " << labels[i] << '\n';

    std::cout << "\n=== Cadenas dobles soportadas en v2 ===\n";
    std::cout << "1. Caesar -> Base64\n";
    std::cout << "2. ROT13 -> Base64\n";
    std::cout << "3. Atbash -> Base64\n";
    std::cout << "4. Caesar -> Hexadecimal\n";
    std::cout << "5. ROT13 -> Hexadecimal\n";
    std::cout << "6. Atbash -> Hexadecimal\n";
    std::cout << "7. Morse -> Base64\n";
    std::cout << "8. RailFence -> Base64\n";
}

int main() {
    CipherDetector detector;
    bool running = true;

    std::cout << "========================================\n";
    std::cout << "          CryptoVision AI v2.0\n";
    std::cout << "  Identificador de cifrado simple/doble\n";
    std::cout << "========================================\n";

    while (running) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "1. Entrenar modelo desde TXT\n";
        std::cout << "2. Identificar tipo de cifrado simple\n";
        std::cout << "3. Identificar doble cifrado y descifrar Top 3\n";
        std::cout << "4. Ver algoritmos soportados\n";
        std::cout << "5. Guardar modelo entrenado\n";
        std::cout << "6. Cargar modelo entrenado\n";
        std::cout << "7. Exportar historial CSV\n";
        std::cout << "8. Entrenamiento sintetico de prueba\n";
        std::cout << "9. Salir\n";
        std::cout << "Ejemplos entrenados acumulados: " << detector.trained_examples() << "\n";

        int op = read_int("Elige una opcion: ", 0);

        try {
            if (op == 1) {
                std::cout << "\nFormato esperado:\n";
                std::cout << "  cifrados.txt -> una muestra cifrada por fila\n";
                std::cout << "  labels.txt   -> respuesta correcta de cada fila\n";
                std::cout << "Ambos archivos deben tener la misma cantidad de filas.\n\n";
                std::string x_path = read_line("Ruta del TXT de textos cifrados [examples/cifrados.txt]: ");
                if (x_path.empty()) x_path = "examples/cifrados.txt";
                std::string y_path = read_line("Ruta del TXT de respuestas [examples/labels.txt]: ");
                if (y_path.empty()) y_path = "examples/labels.txt";
                int epochs = read_int("Epocas [20]: ", 20);
                double lr = read_double("Learning rate [0.03]: ", 0.03);
                auto summary = detector.train_from_txt(x_path, y_path, epochs, lr);
                std::cout << "\nEntrenamiento terminado.\n";
                std::cout << "Ejemplos por epoca: " << summary.examples << "\n";
                std::cout << "Loss final: " << summary.final_loss << "\n";
                std::cout << "Accuracy final aproximada: " << summary.final_accuracy * 100.0 << "%\n";
            } else if (op == 2) {
                std::string text = read_line("Pega el texto cifrado: ");
                auto report = detector.analyze(text);
                print_report(report, 12);
            } else if (op == 3) {
                std::cout << "\nCombinaciones soportadas:\n";
                std::cout << "  Caesar -> Base64 | ROT13 -> Base64 | Atbash -> Base64\n";
                std::cout << "  Caesar -> Hexadecimal | ROT13 -> Hexadecimal | Atbash -> Hexadecimal\n";
                std::cout << "  Morse -> Base64 | RailFence -> Base64\n\n";
                std::string text = read_line("Pega el texto con doble cifrado: ");
                MultiLayerCipherDetector multi(detector);
                auto candidates = multi.analyze_double(text, 3);
                MultiLayerCipherDetector::print_double_report(candidates);
            } else if (op == 4) {
                show_classes(detector);
                std::cout << "\nEn labels.txt puedes usar alias como cesar, base64, hex, sha256, aes o rsa.\n";
            } else if (op == 5) {
                std::string path = read_line("Ruta para guardar [artifacts/cryptovision_model.txt] (0 para cancelar): ");
                if (path == "0") {
                    std::cout << "Guardado cancelado.\n";
                    continue;
                }
                if (path.empty()) path = "artifacts/cryptovision_model.txt";
                detector.save_model(path);
                std::cout << "Modelo guardado en: " << path << "\n";
            } else if (op == 6) {
                std::string path = read_line("Ruta para cargar [artifacts/cryptovision_model.txt] (0 para cancelar): ");
                if (path == "0") {
                    std::cout << "Subida cancelado.\n";
                    continue;
                }
                if (path.empty()) path = "artifacts/cryptovision_model.txt";
                detector.load_model(path);
                std::cout << "Modelo cargado correctamente.\n";
            } else if (op == 7) {
                std::string path = read_line("Ruta CSV [artifacts/training_history.csv]: ");
                if (path.empty()) path = "artifacts/training_history.csv";
                detector.export_history_csv(path);
                std::cout << "Historial exportado en: " << path << "\n";
            } else if (op == 8) {
                int n = read_int("Ejemplos sinteticos por clase [150]: ", 150);
                int epochs = read_int("Epocas [10]: ", 10);
                double lr = read_double("Learning rate [0.03]: ", 0.03);
                auto summary = detector.train_synthetic(n, epochs, lr);
                std::cout << "Entrenamiento sintetico listo. Accuracy final: " << summary.final_accuracy * 100.0 << "%\n";
            } else if (op == 9) {
                running = false;
            } else {
                std::cout << "Opcion invalida.\n";
            }
        } catch (const std::exception& ex) {
            std::cout << "Error: " << ex.what() << "\n";
        }
    }

    std::cout << "Hasta luego.\n";
    return 0;
}
