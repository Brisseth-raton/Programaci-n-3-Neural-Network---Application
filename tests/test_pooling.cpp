//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/nn/nn_pooling.h"

using namespace utec::tf;
using namespace utec::tf::layers;

int main() {

    TestRunner test;

    // =====================================
    // Constructor
    // =====================================

    MaxPooling2D pool({2,2});

    CHECK(test, pool.output_shape().empty());

    // =====================================
    // Build
    // =====================================

    pool.build(Shape{4,4,1});

    Shape esperado{2,2,1};

    CHECK(test, pool.output_shape() == esperado);

    // =====================================
    // Forward
    // =====================================

    Tensor<float> input(Shape{1,4,4,1});

    int valor = 1;

    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            input(0,i,j,0)=valor++;
        }
    }

    auto out = pool.forward(input);

    CHECK(test, out.rank()==4);

    Shape salida{1,2,2,1};

    CHECK(test, out.shape()==salida);

    CHECK(test, out(0,0,0,0)==6);
    CHECK(test, out(0,0,1,0)==8);
    CHECK(test, out(0,1,0,0)==14);
    CHECK(test, out(0,1,1,0)==16);

    // =====================================
    // Backward
    // =====================================

    Tensor<float> grad(Shape{1,2,2,1},1.0f);

    auto back = pool.backward(grad);

    CHECK(test, back.shape()==input.shape());

    // =====================================
    // Constructor inválido
    // =====================================

    CHECK_THROWS(test,
        MaxPooling2D({0,2})
    );

    // =====================================
    // Forward sin build
    // =====================================

    MaxPooling2D pool2({2,2});

    CHECK_THROWS(test,
        pool2.forward(input)
    );

    // =====================================
    // Backward sin forward
    // =====================================

    MaxPooling2D pool3({2,2});

    pool3.build(Shape{4,4,1});

    CHECK_THROWS(test,
        pool3.backward(grad)
    );

    // =====================================
    // Build inválido
    // =====================================

    CHECK_THROWS(test,
        pool.build(Shape{4,4})
    );

    return test.summary();
}