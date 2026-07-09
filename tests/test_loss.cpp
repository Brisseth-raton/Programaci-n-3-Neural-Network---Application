//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/nn/nn_loss.h"

#include <cmath>

using namespace utec::tf;
using namespace utec::tf::losses;

int main() {

    TestRunner test;

    CategoricalCrossentropy loss;

    // =====================================
    // Loss correcta
    // =====================================

    Tensor<float> y_true(Shape{1,3},{0,1,0});
    Tensor<float> y_pred(Shape{1,3},{0.1f,0.8f,0.1f});

    float l = loss(y_true,y_pred);

    CHECK(test, l > 0.0f);
    CHECK(test, std::abs(l - (-std::log(0.8f))) < 1e-5f);

    // =====================================
    // Gradient
    // =====================================

    auto grad = loss.gradient(y_true,y_pred);

    CHECK(test, grad.shape() == y_true.shape());

    CHECK(test, std::abs(grad(0,0)-0.1f)<1e-5f);
    CHECK(test, std::abs(grad(0,1)+0.2f)<1e-5f);
    CHECK(test, std::abs(grad(0,2)-0.1f)<1e-5f);

    // =====================================
    // Shapes incompatibles
    // =====================================

    CHECK_THROWS(test,
        loss(
            Tensor<float>(Shape{1,2}),
            Tensor<float>(Shape{1,3})
        )
    );

    CHECK_THROWS(test,
        loss.gradient(
            Tensor<float>(Shape{1,2}),
            Tensor<float>(Shape{1,3})
        )
    );

    // =====================================
    // Rank inválido
    // =====================================

    CHECK_THROWS(test,
        loss(
            Tensor<float>(Shape{3}),
            Tensor<float>(Shape{3})
        )
    );

    return test.summary();
}