//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/nn/nn_flatten.h"

using namespace utec::tf;
using namespace utec::tf::layers;

int main() {

    TestRunner test;

    // =====================================
    // Build
    // =====================================

    Flatten flatten;

    flatten.build(Shape{4,4,2});

    Shape esperado{32};

    CHECK(test, flatten.output_shape() == esperado);

    // =====================================
    // Forward
    // =====================================

    Tensor<float> input(Shape{1,4,4,2},1.0f);

    auto out = flatten.forward(input);

    Shape salida{1,32};

    CHECK(test, out.rank() == 2);
    CHECK(test, out.shape() == salida);
    CHECK(test, out.numel() == input.numel());

    // =====================================
    // Backward
    // =====================================

    Tensor<float> grad(Shape{1,32},2.0f);

    auto back = flatten.backward(grad);

    CHECK(test, back.shape() == input.shape());
    CHECK(test, back.numel() == input.numel());

    // =====================================
    // Backward sin forward
    // =====================================

    Flatten f2;

    f2.build(Shape{2,2,1});

    CHECK_THROWS(test,
        f2.backward(
            Tensor<float>(Shape{1,4})
        )
    );

    // =====================================
    // Gradiente incompatible
    // =====================================

    CHECK_THROWS(test,
        flatten.backward(
            Tensor<float>(Shape{1,10})
        )
    );

    return test.summary();
}