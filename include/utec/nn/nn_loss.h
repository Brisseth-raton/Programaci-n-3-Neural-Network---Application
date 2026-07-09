#ifndef PROYECTO_NN_LOSS_H
#define PROYECTO_NN_LOSS_H

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "utec/algebra/tensor_backend.h"

namespace utec::tf::losses {

    class CategoricalCrossentropy {
    public:
        float operator()(const Tensor<float>& y_true,const Tensor<float>& y_pred) const {
            if (y_true.shape() != y_pred.shape()) { throw std::invalid_argument("las formas de y_true y y_pred no coinciden"); }
            if (y_true.rank() != 2) { throw std::invalid_argument("categorical crossentropy requiere tensores de rango 2"); }

            float loss = 0.0f;

            constexpr float epsilon = 1e-7f;

            for (std::size_t i = 0; i < y_true.size(); ++i) {
                float pred = std::max(y_pred[i], epsilon);
                loss -= y_true[i] * std::log(pred);
            }
            return loss / static_cast<float>(y_true.shape()[0]);
        }

        Tensor<float> gradient(const Tensor<float>& y_true,const Tensor<float>& y_pred) const {
            if (y_true.shape() != y_pred.shape()) { throw std::invalid_argument("las formas de y_true y y_pred no coinciden"); }

            Tensor<float> grad = Tensor<float>::zeros(y_pred.shape());

            float batch = static_cast<float>(y_true.shape()[0]);

            for (std::size_t i = 0; i < y_pred.size(); ++i) { grad[i] = (y_pred[i] - y_true[i]) / batch; }
            return grad;
        }
    };
}
#endif