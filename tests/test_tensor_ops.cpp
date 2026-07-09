//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/algebra/tensor_ops.h"
#include "utec/nn/nn_ops.h"

using namespace utec::tf;

int main() {

    TestRunner test;

    // =====================================================
    // Suma
    // =====================================================

    Tensor<float> a(Shape{2,2},{1,2,3,4});
    Tensor<float> b(Shape{2,2},{5,6,7,8});

    auto suma = a + b;

    CHECK(test, suma(0,0) == 6);
    CHECK(test, suma(0,1) == 8);
    CHECK(test, suma(1,0) == 10);
    CHECK(test, suma(1,1) == 12);

    // =====================================================
    // Resta
    // =====================================================

    auto resta = b - a;

    CHECK(test, resta(0,0) == 4);
    CHECK(test, resta(0,1) == 4);
    CHECK(test, resta(1,0) == 4);
    CHECK(test, resta(1,1) == 4);

    // =====================================================
    // Multiply elemento a elemento
    // =====================================================

    auto mult = ops::multiply(a,b);

    CHECK(test, mult(0,0) == 5);
    CHECK(test, mult(0,1) == 12);
    CHECK(test, mult(1,0) == 21);
    CHECK(test, mult(1,1) == 32);

    // =====================================================
    // MatMul
    // =====================================================

    Tensor<float> m1(Shape{2,2},{1,2,3,4});
    Tensor<float> m2(Shape{2,2},{5,6,7,8});

    auto c = ops::matmul(m1,m2);

    CHECK(test, c(0,0) == 19);
    CHECK(test, c(0,1) == 22);
    CHECK(test, c(1,0) == 43);
    CHECK(test, c(1,1) == 50);

    // =====================================================
    // Transpuesta
    // =====================================================

    auto t = ops::transpose2d(m1);

    CHECK(test, (t.shape() == Shape{2,2}));
    CHECK(test, t(0,1) == 3);
    CHECK(test, t(1,0) == 2);

    // =====================================================
    // Casos inválidos
    // =====================================================

    CHECK_THROWS(test,
        ops::matmul(
            Tensor<float>(Shape{2,3}),
            Tensor<float>(Shape{2,2})
        )
    );

    CHECK_THROWS(test,
        ops::transpose2d(
            Tensor<float>(Shape{2,2,2})
        )
    );

    CHECK_THROWS(test,
        ops::multiply(
            Tensor<float>(Shape{2,2}),
            Tensor<float>(Shape{3,2})
        )
    );

    return test.summary();
}