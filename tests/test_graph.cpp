#include "check.h"

#include "utec/nn/nn_graph.h"
#include "utec/nn/nn_dense.h"

using namespace utec::tf;
using namespace utec::tf::layers;

int main() {

    TestRunner test;

    // =====================================
    // Grafo vacío
    // =====================================

    SequentialGraph graph;

    CHECK(test, graph.empty());

    Tensor<float> input(Shape{1,2}, 1.0f);
    Tensor<float> grad(Shape{1,2}, 1.0f);

    CHECK_THROWS(test,
        graph.forward(input)
    );

    CHECK_THROWS(test,
        graph.backward(grad)
    );

    // =====================================
    // Agregar una capa
    // =====================================

    Dense dense(3);
    dense.build(Shape{2});

    graph.add(dense);

    CHECK(test, !graph.empty());

    // =====================================
    // Forward
    // =====================================

    auto out = graph.forward(input);

    Shape esperado{1,3};

    CHECK(test, out.shape() == esperado);

    // =====================================
    // Backward
    // =====================================

    Tensor<float> grad_out(Shape{1,3}, 1.0f);

    auto grad_in = graph.backward(grad_out);

    Shape esperado_grad{1,2};

    CHECK(test, grad_in.shape() == esperado_grad);

    return test.summary();
}