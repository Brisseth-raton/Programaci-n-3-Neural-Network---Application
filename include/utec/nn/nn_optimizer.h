//
// Created by HP on 18/05/2026.
//
#ifndef PROYECTO_NN_OPTIMIZER_H
#define PROYECTO_NN_OPTIMIZER_H

#include <stdexcept>

#include "utec/algebra/tensor_backend.h"

namespace utec::tf::optimizers {

    class SGD {
    public:
        explicit SGD(float lr): learning_rate(lr) {
            if (lr <= 0.0f) { throw std::invalid_argument("el learning rate debe ser mayor que cero"); }
        }

        void update(Tensor<float>& parameter,const Tensor<float>& gradient) const {
            if (parameter.shape() != gradient.shape()) { throw std::invalid_argument("parameter y gradient deben tener la misma forma"); }

            for (std::size_t i = 0;i < parameter.size();++i) {
                parameter[i] -= learning_rate * gradient[i];
            }
        }
        float learning_rate;
    };
}
#endif