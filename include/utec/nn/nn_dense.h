#ifndef PROYECTO_NN_DENSE_H
#define PROYECTO_NN_DENSE_H

#include <map>
#include <string>
#include <stdexcept>
#include <memory>

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_ops.h"

namespace utec::tf::layers {

class Dense final :
    public utec::tf::Layer {

public:

    explicit Dense(int units,Activation activation = Activation::Linear)
    :units_(units),activation_(activation) {
        if (units <= 0) { throw std::invalid_argument("la cantidad de unidades debe ser mayor que cero"); }
    }

    void build(const Shape& input_shape) override {

        if (input_shape.rank() != 1) { throw std::invalid_argument("dense requiere una entrada de rango 1"); }

        input_shape_ = input_shape;

        int input_features = input_shape[0];

        params_["weights"] = Tensor<float>::zeros(Shape{input_features, units_});

        for (int i = 0; i < input_features; ++i) {
            for (int j = 0; j < units_; ++j) {
                params_["weights"](i, j) = 0.05f * static_cast<float>((i + 1) + (j + 1));
            }
        }

        params_["bias"] = Tensor<float>( Shape{units_}, 0.0f );

        grads_["weights"] = Tensor<float>::zeros(Shape{input_features, units_});
        grads_["bias"] = Tensor<float>::zeros(Shape{units_});

        output_shape_ = Shape{units_};

        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& input) override {

        if (!built_) { throw std::runtime_error("la capa dense no fue construida"); }
        if (input.rank() != 2) { throw std::invalid_argument("dense espera un tensor de rango 2"); }
        if (input.shape()[1] != input_shape_[0]) { throw std::invalid_argument("la entrada no coincide con la forma esperada"); }

        last_input_ = input;

        auto output = ops::matmul(input,params_.at("weights"));

        for (int b = 0; b < output.shape()[0]; ++b) {
            for (int j = 0; j < units_; ++j) { output(b, j) += params_.at("bias")[j]; }
        }

        last_linear_ = output;

        last_output_ = apply_activation(output,activation_);
        return last_output_;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {

        if (!built_) { throw std::runtime_error("la capa dense no fue construida"); }
        if (grad_output.rank() != 2) { throw std::invalid_argument("dense backward espera un tensor de rango 2"); }
        if (grad_output.shape()[1] != units_) { throw std::invalid_argument("grad_output incompatible con dense"); }

        auto grad = activation_backward(
            grad_output,
            last_linear_,
            activation_
        );

        auto input_t = ops::transpose2d( last_input_ );

        grads_["weights"] = ops::matmul(input_t,grad);

        grads_["bias"] = Tensor<float>::zeros(Shape{units_});

        for (int b = 0; b < grad.shape()[0]; ++b) {
            for (int j = 0; j < units_; ++j) {
                grads_["bias"][j] += grad(b, j);
            }
        }

        auto weights_t = ops::transpose2d(params_.at("weights"));

        auto grad_input = ops::matmul(
            grad,
            weights_t
        );
        return grad_input;
    }

    void set_weights(const Tensor<float>& weights) {

        if (!built_) { throw std::runtime_error("la capa dense no fue construida"); }
        if (weights.shape() != params_.at("weights").shape()) { throw std::invalid_argument("shape incompatible para weights"); }

        params_["weights"] = weights;
    }

    void set_bias(const Tensor<float>& bias) {

        if (!built_) { throw std::runtime_error("la capa dense no fue construida"); }
        if (bias.shape() != params_.at("bias").shape()) { throw std::invalid_argument("shape incompatible para bias"); }

        params_["bias"] = bias;
    }

    [[nodiscard]]
    Shape output_shape() const override {return output_shape_;}
    [[nodiscard]]
    std::map<std::string, Tensor<float>>
    parameters() const override {return params_;}
    [[nodiscard]]
    std::map<std::string, Tensor<float>>
    gradients() const override {return grads_;}

    std::map<std::string, Tensor<float>>&
    parameters_ref() override {return params_;}

    [[nodiscard]]
    std::unique_ptr<utec::tf::Layer>
    clone() const override {
        return std::unique_ptr<utec::tf::Layer>(
            static_cast<utec::tf::Layer*>(
                new Dense(*this)
            )
        );
    }

private:

    int units_;

    Activation activation_;

    bool built_ = false;

    Shape input_shape_;
    Shape output_shape_;

    Tensor<float> last_input_;
    Tensor<float> last_linear_;
    Tensor<float> last_output_;

    std::map<std::string, Tensor<float>> params_;
    std::map<std::string, Tensor<float>> grads_;
};
}
#endif