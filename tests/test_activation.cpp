//
// Created by luisf on 7/07/2026.
//
#include "check.h"

#include "utec/nn/nn_activation.h"

using namespace utec::tf;

int main() {

    TestRunner test;

    // ==========================
    // Linear
    // ==========================

    {
        Tensor<float> x(Shape{2,2});

        x(0,0)=1;
        x(0,1)=2;
        x(1,0)=3;
        x(1,1)=4;

        auto y = apply_activation(x,Activation::Linear);

        CHECK(test,y(0,0)==1);
        CHECK(test,y(1,1)==4);
    }

    // ==========================
    // ReLU
    // ==========================

    {
        Tensor<float> x(Shape{2,2});

        x(0,0)=-2;
        x(0,1)=0;
        x(1,0)=3;
        x(1,1)=-1;

        auto y = apply_activation(x,Activation::Relu);

        CHECK(test,y(0,0)==0);
        CHECK(test,y(0,1)==0);
        CHECK(test,y(1,0)==3);
        CHECK(test,y(1,1)==0);
    }

    // ==========================
    // Softmax
    // ==========================

    {
        Tensor<float> x(Shape{1,3});

        x(0,0)=1;
        x(0,1)=2;
        x(0,2)=3;

        auto y = apply_activation(x,Activation::Softmax);

        CHECK(test,std::abs(
            y(0,0)+y(0,1)+y(0,2)-1.0f
        )<1e-5f);

        CHECK(test,y(0,2)>y(0,1));
        CHECK(test,y(0,1)>y(0,0));
    }

    // ==========================
    // Softmax rango inválido
    // ==========================

    CHECK_THROWS(test,
        apply_activation(
            Tensor<float>(Shape{2,2,2}),
            Activation::Softmax
        )
    );

    return test.summary();
}