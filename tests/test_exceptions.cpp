#include "check.h"

#include "utec/algebra/shape.h"
#include "utec/algebra/tensor_backend.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_convolution.h"
#include "utec/nn/nn_pooling.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/crypto/CipherDetector.h"

using namespace utec::tf;
using namespace utec::tf::layers;
using namespace utec::tf::losses;
using namespace utec::tf::optimizers;
using namespace utec::crypto;

int main() {

    TestRunner test;

    // =====================================
    // Shape
    // =====================================

    CHECK_THROWS(test,(Shape{-1,2}));

    // =====================================
    // Tensor
    // =====================================

    Tensor<float> t(Shape{2,2});

    CHECK_THROWS(test,
        t.at({2,0})
    );

    // =====================================
    // Dense
    // =====================================

    CHECK_THROWS(test,
        Dense(0)
    );

    // =====================================
    // Conv2D
    // =====================================

    CHECK_THROWS(test,
        Conv2D(0,{3,3})
    );

    // =====================================
    // MaxPooling2D
    // =====================================

    CHECK_THROWS(test,
        MaxPooling2D({0,2})
    );

    // =====================================
    // Loss
    // =====================================

    CategoricalCrossentropy loss;

    CHECK_THROWS(test,
        loss(
            Tensor<float>(Shape{1,2}),
            Tensor<float>(Shape{1,3})
        )
    );

    // =====================================
    // Optimizer
    // =====================================

    SGD opt(0.1f);

    Tensor<float> p(Shape{2});
    Tensor<float> g(Shape{3});

    CHECK_THROWS(test,
        opt.update(p,g)
    );

    // =====================================
    // CipherDetector
    // =====================================

    CipherDetector detector;

    CHECK_THROWS(test,
        detector.analyze("")
    );

    CHECK_THROWS(test,
        detector.label_to_id("DESCONOCIDO")
    );

    CHECK_THROWS(test,
        detector.train_synthetic(0)
    );

    CHECK_THROWS(test,
        detector.load_model("archivo_inexistente.txt")
    );

    return test.summary();
}