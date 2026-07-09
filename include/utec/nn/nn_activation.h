//
// Created by HP on 18/05/2026.
//

#ifndef PROYECTO_NN_ACTIVATION_H
#define PROYECTO_NN_ACTIVATION_H

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "utec/algebra/tensor_backend.h"

namespace utec::tf {

enum class Activation {
    Linear,
    Relu,
    Softmax
};

template <typename T>
Tensor<T> apply_activation(const Tensor<T>& input,Activation activation) {
    Tensor<T> output(input.shape());

    for (std::size_t i = 0; i < input.size(); ++i) { output[i] = input[i];}

    if (activation == Activation::Linear) { return output; }

    if (activation == Activation::Relu) {
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (output[i] < static_cast<T>(0)) { output[i] = static_cast<T>(0); }
        }
        return output;
    }

    if (activation == Activation::Softmax) {
        if (input.rank() != 2) { throw std::invalid_argument("softmax requiere tensores de rango 2"); }

        int batch = input.shape()[0];
        int classes = input.shape()[1];

        for (int b = 0; b < batch; ++b) {
            T maximo = input(b, 0);

            for (int j = 1; j < classes; ++j) {maximo = std::max(maximo, input(b, j));}

            T suma = static_cast<T>(0);

            for (int j = 0; j < classes; ++j) {
                output(b, j) = std::exp(input(b, j) - maximo);
                suma += output(b, j);
            }

            for (int j = 0; j < classes; ++j) {output(b, j) /= suma;}
        }
        return output;
    }
    return output;
}

template <typename T>
Tensor<T> activation_backward(
    const Tensor<T>& grad_output,
    const Tensor<T>& last_linear,
    Activation activation
) {
    Tensor<T> grad = Tensor<T>::zeros(grad_output.shape());

    if (activation == Activation::Linear) {
        for (std::size_t i = 0; i < grad_output.size(); ++i) {grad[i] = grad_output[i];}
        return grad;
    }

    if (activation == Activation::Relu) {
        for (std::size_t i = 0; i < grad_output.size(); ++i) {
            if (last_linear[i] > static_cast<T>(0)) {
                grad[i] = grad_output[i];
            } else { grad[i] = static_cast<T>(0); }
        }
        return grad;
    }

    if (activation == Activation::Softmax) {
        for (std::size_t i = 0; i < grad_output.size(); ++i) { grad[i] = grad_output[i]; }
        return grad;
    }
    return grad;
}
}

#endif