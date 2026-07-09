//
// Created by HP on 18/05/2026.
//

#ifndef PROYECTO_NN_FLATTEN_H
#define PROYECTO_NN_FLATTEN_H

#include <memory>
#include <stdexcept>

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_ops.h"

namespace utec::tf::layers {

    class Flatten final : public utec::tf::Layer {
    public:
        void build(const Shape& input_shape) override {
            input_shape_ = input_shape;

            int total = 1;

            for (std::size_t i = 0;i < input_shape.rank();++i) {
                total *= input_shape[i];
            }
            output_shape_ = Shape{total};
        }

        Tensor<float> forward(const Tensor<float>& input) override {
            last_input_shape_ = input.shape();
            forward_called_ = true;

            return ops::flatten_batch(input);
        }

        Tensor<float> backward(const Tensor<float>& grad_output) override {

            if (!forward_called_) { throw std::logic_error("flatten backward requiere forward previo"); }
            if (grad_output.numel() != last_input_shape_.numel()) { throw std::invalid_argument("grad_output incompatible con flatten"); }

            return grad_output.reshaped( last_input_shape_ );
        }

        [[nodiscard]]
        Shape output_shape() const override { return output_shape_; }
        [[nodiscard]]
        std::unique_ptr<utec::tf::Layer>
        clone() const override {
            return std::unique_ptr<utec::tf::Layer>(static_cast<utec::tf::Layer*>(new Flatten(*this))
            );
        }

    private:

        Shape input_shape_;
        Shape output_shape_;
        Shape last_input_shape_;

        bool forward_called_ = false;
    };
}

#endif