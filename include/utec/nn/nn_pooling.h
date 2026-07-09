#ifndef PROYECTO_NN_POOLING_H
#define PROYECTO_NN_POOLING_H

#include <stdexcept>
#include <algorithm>
#include <utility>
#include <vector>
#include <memory>

#include "utec/nn/nn_interfaces.h"

namespace utec::tf::layers {

class MaxPooling2D final :
    public utec::tf::Layer {

public:

    explicit MaxPooling2D( std::pair<int, int> pool_size ):pool_size_(pool_size) {
        if (pool_size.first <= 0 || pool_size.second <= 0) {
            throw std::invalid_argument("el tamaño de ventana debe ser mayor que cero");
        }
    }

    void build(const Shape& input_shape) override {

        if (input_shape.rank() != 3) { throw std::invalid_argument("maxpooling2d requiere entrada de rango 3"); }

        input_shape_ = input_shape;

        int height = input_shape[0];
        int width = input_shape[1];
        int channels = input_shape[2];

        if (pool_size_.first > height || pool_size_.second > width) { throw std::invalid_argument("la ventana es más grande que la entrada"); }
        if (height % pool_size_.first != 0 || width % pool_size_.second != 0) { throw std::invalid_argument("la ventana debe dividir exactamente la entrada"); }

        int out_h = height / pool_size_.first;
        int out_w = width / pool_size_.second;

        output_shape_ = Shape{
            out_h,
            out_w,
            channels
        };
        built_ = true;
    }

    Tensor<float> forward(const Tensor<float>& input) override {

        if (!built_) { throw std::runtime_error("la capa maxpooling2d no fue construida"); }
        if (input.rank() != 4) { throw std::invalid_argument("maxpooling2d espera entrada NHWC de rango 4"); }

        last_input_shape_ = input.shape();

        int batch = input.shape()[0];
        int channels = input.shape()[3];

        int out_h = output_shape_[0];
        int out_w = output_shape_[1];

        Tensor<float> output =
            Tensor<float>::zeros(
                Shape{
                    batch,
                    out_h,
                    out_w,
                    channels
                }
            );

        indices_.assign(static_cast<std::size_t>(batch * out_h * out_w * channels),0);

        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    for (int c = 0; c < channels; ++c) {

                        int mejor_fila = h * pool_size_.first;
                        int mejor_colum = w * pool_size_.second;

                        float maximo = input( b,mejor_fila,mejor_colum,c );

                        for (int filap = 0; filap < pool_size_.first; ++filap) {
                            for (int colump = 0; colump < pool_size_.second; ++colump) {

                                int fila = h * pool_size_.first + filap;
                                int colum = w * pool_size_.second + colump;

                                float valor = input( b,fila,colum,c );

                                if (valor > maximo) {
                                    maximo = valor;
                                    mejor_fila = fila;
                                    mejor_colum = colum;
                                }
                            }
                        }

                        output(b, h, w, c) = maximo;

                        std::size_t pos = static_cast<std::size_t>(((b * out_h + h) * out_w + w) * channels + c);

                        indices_[pos] =linear_index(b,mejor_fila,mejor_colum,c,last_input_shape_);
                    }
                }
            }
        }
        forward_called_ = true;
        return output;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {

        if (!built_) { throw std::runtime_error("la capa maxpooling2d no fue construida"); }
        if (!forward_called_) { throw std::logic_error("maxpooling2d backward requiere forward previo"); }
        if (grad_output.rank() != 4) { throw std::invalid_argument("maxpooling2d backward espera tensor de rango 4"); }
        if (
            grad_output.shape()[1] != output_shape_[0] ||
            grad_output.shape()[2] != output_shape_[1] ||
            grad_output.shape()[3] != output_shape_[2]
        ) { throw std::invalid_argument("grad_output incompatible con maxpooling2d"); }

        Tensor<float> grad_input = Tensor<float>::zeros( last_input_shape_ );

        int batch = grad_output.shape()[0];
        int out_h = grad_output.shape()[1];
        int out_w = grad_output.shape()[2];
        int channels = grad_output.shape()[3];

        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    for (int c = 0; c < channels; ++c) {

                        std::size_t pos =
                            static_cast<std::size_t>(
                                ((b * out_h + h) * out_w + w) * channels + c
                            );

                        grad_input[indices_[pos]] += grad_output( b,h,w,c );
                    }
                }
            }
        }
        return grad_input;
    }

    [[nodiscard]]
    Shape output_shape() const override {return output_shape_;}
    [[nodiscard]]
    std::unique_ptr<utec::tf::Layer>
    clone() const override {
        return std::unique_ptr<utec::tf::Layer>(
            static_cast<utec::tf::Layer*>(
                new MaxPooling2D(*this)
            )
        );
    }

private:

    std::pair<int, int> pool_size_;

    bool built_ = false;
    bool forward_called_ = false;

    Shape input_shape_;
    Shape output_shape_;
    Shape last_input_shape_;

    std::vector<std::size_t> indices_;

    std::size_t linear_index(
        int b,
        int h,
        int w,
        int c,
        const Shape& shape
    ) const {
        return static_cast<std::size_t>(
            ((b * shape[1] + h) * shape[2] + w) * shape[3] + c
        );
    }
};

}

#endif