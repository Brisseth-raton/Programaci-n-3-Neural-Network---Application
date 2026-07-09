//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/crypto/CipherDetector.h"

#include <filesystem>

using namespace utec::crypto;

int main() {

    TestRunner test;

    CipherDetector detector;

    // =====================================
    // Constructor
    // =====================================

    CHECK(test, detector.labels().size() == 12);
    CHECK(test, detector.trained_examples() == 0);

    // =====================================
    // Analizar texto
    // =====================================

    auto report = detector.analyze("SGVsbG8gV29ybGQ=");

    CHECK(test, report.predictions.size() == 12);
    CHECK(test, report.looks_base64);

    // =====================================
    // Texto vacío
    // =====================================

    CHECK_THROWS(test,
        detector.analyze("")
    );

    // =====================================
    // label_to_id
    // =====================================

    CHECK(test, detector.label_to_id("Caesar") == 0);
    CHECK(test, detector.label_to_id("ROT13") == 1);
    CHECK(test, detector.label_to_id("Hexadecimal") == 5);

    CHECK_THROWS(test,
        detector.label_to_id("NO_EXISTE")
    );

    // =====================================
    // train_one
    // =====================================

    detector.train_one("HELLO WORLD",0);

    CHECK(test, detector.trained_examples() == 1);

    CHECK_THROWS(test,
        detector.train_one("HELLO",-1)
    );

    CHECK_THROWS(test,
        detector.train_one("HELLO",100)
    );

    // =====================================
    // train_synthetic
    // =====================================

    auto summary = detector.train_synthetic(1,1);

    CHECK(test, summary.epochs == 1);
    CHECK(test, summary.examples == 12);

    CHECK(test, detector.trained_examples() == 13);

    CHECK_THROWS(test,
        detector.train_synthetic(0)
    );

    // =====================================
    // Guardar modelo
    // =====================================

    detector.save_model("artifacts/model_test.txt");

    CHECK(test,
        std::filesystem::exists("artifacts/model_test.txt")
    );

    // =====================================
    // Cargar modelo
    // =====================================

    CipherDetector detector2;

    detector2.load_model("artifacts/model_test.txt");

    CHECK(test,
        detector2.trained_examples() == detector.trained_examples()
    );

    // =====================================
    // Archivo inexistente
    // =====================================

    CHECK_THROWS(test,
        detector2.load_model("archivo_inexistente.txt")
    );

    return test.summary();
}