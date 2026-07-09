#include "check.h"

#include "utec/nn/nn_convolution.h"

using namespace utec::tf;
using namespace utec::tf::layers;

int main() {

    TestRunner test;

    // =====================================
    // Constructor válido
    // =====================================

    Conv2D conv(2, {3,3}, Activation::Linear);

    CHECK(test, conv.output_shape().empty());

    // =====================================
    // Build correcto
    // =====================================

    conv.build(Shape{5,5,1});

    Shape esperado{3,3,2};

    CHECK(test, conv.output_shape() == esperado);

    // =====================================
    // Forward correcto
    // =====================================

    Tensor<float> input(
        Shape{1,5,5,1},
        1.0f
    );

    auto out = conv.forward(input);

    Shape salida{1,3,3,2};

    CHECK(test, out.rank() == 4);
    CHECK(test, out.shape() == salida);

    // =====================================
    // Constructor inválido
    // =====================================

    CHECK_THROWS(test,
        Conv2D(
            0,
            {3,3},
            Activation::Linear
        )
    );

    // =====================================
    // Kernel inválido
    // =====================================

    CHECK_THROWS(test,
        Conv2D(
            2,
            {0,3},
            Activation::Linear
        )
    );

    // =====================================
    // Build con shape incorrecto
    // =====================================

    CHECK_THROWS(test,
        conv.build(
            Shape{5,5}
        )
    );

    // =====================================
    // Kernel mayor que la entrada
    // =====================================

    Conv2D conv2(
        2,
        {7,7},
        Activation::Linear
    );

    CHECK_THROWS(test,
        conv2.build(
            Shape{5,5,1}
        )
    );

    // =====================================
    // Forward sin build
    // =====================================

    Conv2D conv3(
        2,
        {3,3},
        Activation::Linear
    );

    CHECK_THROWS(test,
        conv3.forward(input)
    );

    // =====================================
    // Entrada con rango incorrecto
    // =====================================

    CHECK_THROWS(test,
        conv.forward(
            Tensor<float>(
                Shape{5,5,1}
            )
        )
    );

    // =====================================
    // Entrada incompatible
    // =====================================

    CHECK_THROWS(test,
        conv.forward(
            Tensor<float>(
                Shape{1,4,4,1}
            )
        )
    );

    return test.summary();
}