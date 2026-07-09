//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"

using namespace utec::tf;
using namespace utec::tf::layers;
using namespace utec::tf::optimizers;
using namespace utec::tf::losses;

int main() {

    TestRunner test;

    // =====================================
    // Modelo vacío
    // =====================================

    Sequential model;

    Tensor<float> input(Shape{1,2}, 1.0f);

    CHECK(test, !model.compiled());

    CHECK_THROWS(test,
        model.predict(input)
    );

    // =====================================
    // Crear modelo
    // =====================================

    model.add(Input(Shape{2}));
    model.add(Dense(3, Activation::Relu));
    model.add(Dense(2, Activation::Softmax));

    CHECK(test, !model.compiled());

    // =====================================
    // Compile
    // =====================================

    SGD optimizer(0.01f);
    CategoricalCrossentropy loss;

    model.compile(optimizer, loss);

    CHECK(test, model.compiled());

    // =====================================
    // Predict
    // =====================================

    auto output = model.predict(input);

    Shape esperado{1,2};

    CHECK(test, output.shape() == esperado);

    // =====================================
    // Backward
    // =====================================

    Tensor<float> grad(Shape{1,2}, 1.0f);

    auto back = model.backward(grad);

    CHECK(test, back.shape() == input.shape());

    // =====================================
    // Fit inválido
    // =====================================

    Tensor<float> x(Shape{2,2}, 1.0f);
    Tensor<float> y(Shape{3,2}, 0.0f);

    CHECK_THROWS(test,
        model.fit(x, y)
    );

    // =====================================
    // Epochs inválidos
    // =====================================

    FitOptions opt;
    opt.epochs = 0;

    CHECK_THROWS(test,
        model.fit(
            Tensor<float>(Shape{1,2}),
            Tensor<float>(Shape{1,2}),
            opt
        )
    );

    // =====================================
    // Batch inválido
    // =====================================

    opt.epochs = 1;
    opt.batch_size = 0;

    CHECK_THROWS(test,
        model.fit(
            Tensor<float>(Shape{1,2}),
            Tensor<float>(Shape{1,2}),
            opt
        )
    );

    return test.summary();
}