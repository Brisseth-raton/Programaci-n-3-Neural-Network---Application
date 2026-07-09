#include "check.h"

#include "utec/nn/nn_optimizer.h"

#include <cmath>

using namespace utec::tf;
using namespace utec::tf::optimizers;

int main() {

    TestRunner test;

    // =====================================
    // Constructor válido
    // =====================================

    SGD optimizer(0.1f);

    // =====================================
    // Update
    // =====================================

    Tensor<float> weights(Shape{2}, {1.0f, 2.0f});
    Tensor<float> grads(Shape{2}, {0.5f, 1.0f});

    optimizer.update(weights, grads);

    CHECK(test, std::abs(weights[0] - 0.95f) < 1e-5f);
    CHECK(test, std::abs(weights[1] - 1.90f) < 1e-5f);

    // =====================================
    // Learning rate inválido
    // =====================================

    CHECK_THROWS(test,
        SGD(0.0f)
    );

    CHECK_THROWS(test,
        SGD(-0.1f)
    );

    // =====================================
    // Shape incompatible
    // =====================================

    Tensor<float> p(Shape{2});
    Tensor<float> g(Shape{3});

    CHECK_THROWS(test,
        optimizer.update(p, g)
    );

    return test.summary();
}