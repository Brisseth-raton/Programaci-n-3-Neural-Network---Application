//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/algebra/tensor_backend.h"

using namespace utec::tf;

int main() {

    TestRunner test;

    // Constructor
    Tensor<float> t(Shape{2,3});

    CHECK(test, t.rank() == 2);
    CHECK(test, t.numel() == 6);
    CHECK(test, (t.shape() == Shape{2,3}));

    // Constructor con valor
    Tensor<float> a(Shape{2,2}, 5.0f);

    CHECK(test, a[0] == 5.0f);
    CHECK(test, a[3] == 5.0f);

    // Escritura y lectura
    a[1] = 10.0f;

    CHECK(test, a[1] == 10.0f);

    // Acceso multidimensional
    Tensor<int> b(Shape{2,2});

    b(0,0) = 1;
    b(0,1) = 2;
    b(1,0) = 3;
    b(1,1) = 4;

    CHECK(test, b(1,1) == 4);
    CHECK(test, b.at({1,0}) == 3);

    // Reshape válido
    b.reshape(Shape{4});

    CHECK(test, b.rank() == 1);
    CHECK(test, b.numel() == 4);

    // Reshape inválido
    CHECK_THROWS(test, b.reshape(Shape{5}));

    // Constructor con cantidad incorrecta de datos
    CHECK_THROWS(test,
        Tensor<float>(
            Shape{2,2},
            std::vector<float>{1,2,3}
        )
    );

    return test.summary();
}