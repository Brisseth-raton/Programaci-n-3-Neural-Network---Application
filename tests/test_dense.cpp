#include "check.h"

#include "utec/nn/nn_dense.h"

using namespace utec::tf;
using namespace utec::tf::layers;

int main() {

    TestRunner test;

    // ======================================
    // Constructor
    // ======================================

    Dense dense(3);

    CHECK(test, dense.output_shape().empty());

    // ======================================
    // Build
    // ======================================

    dense.build(Shape{2});

    CHECK(test, dense.output_shape() == Shape{3});

    // ======================================
    // Forward
    // ======================================

    Tensor<float> x(Shape{1,2});

    x(0,0)=1.0f;
    x(0,1)=2.0f;

    auto y = dense.forward(x);

    CHECK(test, y.rank()==2);
    CHECK(test, (y.shape()==Shape{1,3}));

    // ======================================
    // Dense sin build
    // ======================================

    Dense d2(4);

    CHECK_THROWS(test,d2.forward(x));

    // ======================================
    // Constructor inválido
    // ======================================

    CHECK_THROWS(test,
        Dense(0)
    );

    // ======================================
    // Input incorrecto
    // ======================================

    CHECK_THROWS(test,
        dense.forward(
            Tensor<float>(Shape{2,2,2})
        )
    );

    return test.summary();
}