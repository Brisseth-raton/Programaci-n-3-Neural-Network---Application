#ifndef PROYECTO_NN_INTERFACES_H
#define PROYECTO_NN_INTERFACES_H

#include <memory>
#include <map>
#include <string>
#include <stdexcept>

#include "utec/algebra/tensor_backend.h"

namespace utec::tf {

    class Layer {
    public:
        virtual ~Layer() = default;
        virtual void build(const Shape& input_shape) = 0;
        virtual Tensor<float> forward(const Tensor<float>& input) = 0;
        virtual Tensor<float> backward(const Tensor<float>& grad_output) = 0;

        [[nodiscard]]
        virtual Shape output_shape() const = 0;
        [[nodiscard]]
        virtual std::map<std::string, Tensor<float>> parameters() const { return {}; }
        [[nodiscard]]
        virtual std::map<std::string, Tensor<float>> gradients() const { return {}; }
        virtual std::map<std::string, Tensor<float>>& parameters_ref() { throw std::logic_error("esta capa no tiene parámetros entrenables"); }
        [[nodiscard]]
        virtual std::unique_ptr<utec::tf::Layer> clone() const = 0;
    };

    namespace layers {

        class Input final : public utec::tf::Layer {
        public:
            explicit Input(const Shape& sample_shape)
                : sample_shape_(sample_shape) {
                if (sample_shape.empty()) { throw std::invalid_argument("el shape de entrada no puede estar vacío"); }
            }

            void build(const Shape&) override {}

            Tensor<float> forward(const Tensor<float>& input) override { return input; }
            Tensor<float> backward(const Tensor<float>& grad_output) override { return grad_output; }

            [[nodiscard]]
            Shape output_shape() const override { return sample_shape_; }

            [[nodiscard]]
            std::unique_ptr<utec::tf::Layer> clone() const override {
                return std::unique_ptr<utec::tf::Layer>(
                    static_cast<utec::tf::Layer*>(
                        new utec::tf::layers::Input(sample_shape_)
                    )
                );
            }
        private:
            Shape sample_shape_;
        };
    }
}
#endif