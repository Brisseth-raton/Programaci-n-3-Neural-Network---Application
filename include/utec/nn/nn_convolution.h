//
// Created by HP on 18/05/2026.
//

#ifndef PROYECTO_NN_CONVOLUTION_H
#define PROYECTO_NN_CONVOLUTION_H

#include <map>
#include <string>
#include <utility>
#include <stdexcept>
#include <memory>

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_ops.h"

namespace utec::tf::layers {

class Conv2D final :
    public utec::tf::Layer {

public:

    Conv2D(
        int filters,
        std::pair<int, int> kernel_size,
        Activation activation = Activation::Linear
    ):filters_(filters),kernel_size_(kernel_size),activation_(activation) {
        if (filters <= 0) { throw std::invalid_argument("la cantidad de filtros debe ser mayor que cero"); }
        if (kernel_size.first <= 0 || kernel_size.second <= 0) { throw std::invalid_argument("el kernel debe tener dimensiones válidas"); }
    }

    void build(const Shape& input_shape) override {

        if (input_shape.rank() != 3) { throw std::invalid_argument("conv2d requiere entrada de rango 3"); }

        input_shape_ = input_shape;

        int height = input_shape[0];
        int width = input_shape[1];
        int channels = input_shape[2];

        if (kernel_size_.first > height || kernel_size_.second > width) { throw std::invalid_argument("el kernel es más grande que la entrada"); }

        params_["weights"] = Tensor<float>::zeros(
            Shape{
                kernel_size_.first,
                kernel_size_.second,
                channels,
                filters_
            }
            );
        for (int kh = 0; kh < kernel_size_.first; ++kh) {
            for (int kw = 0; kw < kernel_size_.second; ++kw) {
                for (int c = 0; c < channels; ++c) {
                    for (int f = 0; f < filters_; ++f) {
                        params_["weights"](kh, kw, c, f) = 0.05f * static_cast<float>(
                                (kh + 1) + (kw + 1) + (c + 1) + (f + 1)
                            );
                    }
                }
            }
        }

        params_["bias"] = Tensor<float>( Shape{filters_}, 0.0f );

        grads_["weights"] = Tensor<float>::zeros( params_.at("weights").shape() );
        grads_["bias"] = Tensor<float>::zeros( params_.at("bias").shape() );

        int out_h = height - kernel_size_.first + 1;
        int out_w = width - kernel_size_.second + 1;

        output_shape_ = Shape{
            out_h,
            out_w,
            filters_
        };
        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& input) override {

        if (!built_) { throw std::runtime_error("la capa conv2d no fue construida"); }
        if (input.rank() != 4) { throw std::invalid_argument("conv2d espera entrada NHWC de rango 4"); }
        if (
            input.shape()[1] != input_shape_[0] ||
            input.shape()[2] != input_shape_[1] ||
            input.shape()[3] != input_shape_[2]
        ) { throw std::invalid_argument("la entrada no coincide con la forma esperada"); }

        last_input_ = input;

        auto output = ops::conv2d(
            input,
            params_.at("weights"),
            {1, 1},
            Padding::Valid
        );

        int batch = output.shape()[0];
        int out_h = output.shape()[1];
        int out_w = output.shape()[2];

        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    for (int f = 0; f < filters_; ++f) {
                        output(b, h, w, f) += params_.at("bias")[f];
                    }
                }
            }
        }

        last_linear_ = output;
        last_output_ = apply_activation(output,activation_);

        return last_output_;
    }

    Tensor<float> backward(
        const Tensor<float>& grad_output
    ) override {

        if (!built_) { throw std::runtime_error("la capa conv2d no fue construida"); }
        if (grad_output.rank() != 4) { throw std::invalid_argument("conv2d backward espera tensor de rango 4"); }

        auto grad = activation_backward(
            grad_output,
            last_linear_,
            activation_
        );

        const auto& weights = params_.at("weights");

        int batch = last_input_.shape()[0];
        int height = last_input_.shape()[1];
        int width = last_input_.shape()[2];
        int channels = last_input_.shape()[3];

        int kernel_h = kernel_size_.first;
        int kernel_w = kernel_size_.second;

        int out_h = grad.shape()[1];
        int out_w = grad.shape()[2];

        grads_["weights"] = Tensor<float>::zeros(weights.shape());

        grads_["bias"] = Tensor<float>::zeros(Shape{filters_});

        Tensor<float> grad_input = Tensor<float>::zeros(last_input_.shape());

        for (int b = 0; b < batch; ++b) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    for (int f = 0; f < filters_; ++f) {

                        float g = grad(b, oh, ow, f);

                        grads_["bias"][f] += g;

                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                for (int c = 0; c < channels; ++c) {

                                    int ih = oh + kh;
                                    int iw = ow + kw;

                                    grads_["weights"](kh, kw, c, f) += last_input_(b, ih, iw, c) * g;

                                    grad_input(b, ih, iw, c) += weights(kh, kw, c, f) * g;
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    void set_weights(const Tensor<float>& weights) {
        if (!built_) { throw std::runtime_error("la capa conv2d no fue construida"); }
        if (weights.shape() != params_.at("weights").shape()) { throw std::invalid_argument("shape incompatible para weights"); }
        params_["weights"] = weights;
    }

    void set_bias(const Tensor<float>& bias) {

        if (!built_) { throw std::runtime_error("la capa conv2d no fue construida"); }
        if (bias.shape() != params_.at("bias").shape()) { throw std::invalid_argument("shape incompatible para bias"); }
        params_["bias"] = bias;
    }

    [[nodiscard]]
    Shape output_shape() const override { return output_shape_; }

    [[nodiscard]]
    std::map<std::string, Tensor<float>>
    parameters() const override { return params_; }

    [[nodiscard]]
    std::map<std::string, Tensor<float>>
    gradients() const override { return grads_; }

    std::map<std::string, Tensor<float>>&
    parameters_ref() override { return params_; }

    [[nodiscard]]
    std::unique_ptr<utec::tf::Layer>
    clone() const override {
        return std::unique_ptr<utec::tf::Layer>(
            static_cast<utec::tf::Layer*>(
                new Conv2D(*this)
            )
        );
    }

private:

    int filters_;

    std::pair<int, int> kernel_size_;

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