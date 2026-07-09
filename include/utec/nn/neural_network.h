#ifndef PROYECTO_NEURAL_NETWORK_H
#define PROYECTO_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <stdexcept>
#include <optional>
#include <algorithm>

#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_convolution.h"
#include "utec/nn/nn_pooling.h"
#include "utec/nn/nn_flatten.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_graph.h"

namespace utec::tf {

struct FitOptions {
    int epochs = 1;
    int batch_size = 1;
};

struct History {std::vector<float> loss;};
struct EvaluationResult {float loss = 0.0f;};

class Sequential {
public:
    Sequential() = default;
    void add(const Layer& layer) {
        auto cloned = layer.clone();
        if (!layers_.empty()) { cloned->build(layers_.back()->output_shape());}

        layers_.push_back(std::move(cloned));
    }

    void compile(const optimizers::SGD& optimizer,const losses::CategoricalCrossentropy& loss) {
        optimizer_ = optimizer;
        loss_ = loss;
        compiled_ = true;
    }

    void compile(const losses::CategoricalCrossentropy& loss,const optimizers::SGD& optimizer) {
        compile(optimizer,loss);
    }

    [[nodiscard]]
    bool compiled() const noexcept {return compiled_;}

    Tensor<float> predict(const Tensor<float>& input) {
        if (layers_.empty()) {throw std::runtime_error("el modelo no tiene capas");}

        Tensor<float> output = input;

        for (auto& layer : layers_) {output = layer->forward(output);}

        last_prediction_ = output;

        return output;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) {

        if (!compiled_) {throw std::logic_error("el modelo no fue compilado");}

        Tensor<float> grad = grad_output;

        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {grad = (*it)->backward(grad);}

        registrar_gradientes();

        return grad;
    }

    Tensor<float> backward() {

        if (!compiled_) {throw std::logic_error("el modelo no fue compilado");}
        if (!last_backward_ready_) {throw std::logic_error("no hay gradiente registrado para backward");}
        return backward( last_loss_gradient_ );
    }

    History fit(
        const Tensor<float>& x,
        const Tensor<float>& y,
        const FitOptions& options = FitOptions{}
    ) {

        if (!compiled_) {throw std::logic_error("el modelo no fue compilado");}
        if (options.epochs <= 0) {throw std::invalid_argument("epochs debe ser mayor que cero");}
        if (options.batch_size <= 0) {throw std::invalid_argument("batch_size debe ser mayor que cero");}
        if (x.rank() < 2 || y.rank() < 2) {throw std::invalid_argument("x e y deben tener dimensión batch");}
        if (x.shape()[0] != y.shape()[0]) {throw std::invalid_argument("x e y deben tener el mismo batch");}

        History history;

        int total = x.shape()[0];

        for (int epoch = 0; epoch < options.epochs; ++epoch) {

            float suma_loss = 0.0f;
            int cantidad_batches = 0;

            for (int start = 0; start < total; start += options.batch_size) {

                int current_batch =std::min(options.batch_size,total - start);

                Tensor<float> xb =batch_slice(x,start,current_batch);

                Tensor<float> yb =batch_slice(y,start,current_batch);

                Tensor<float> pred =predict(xb);

                if (pred.shape() != yb.shape()) { throw std::invalid_argument("las etiquetas no coinciden con la salida del modelo"); }

                suma_loss += (*loss_)(yb,pred);

                last_loss_gradient_ = loss_->gradient(yb,pred);
                last_backward_ready_ = true;

                backward(last_loss_gradient_);

                actualizar_parametros();

                ++cantidad_batches;
            }

            history.loss.push_back(suma_loss /static_cast<float>(cantidad_batches));
        }
        return history;
    }

    EvaluationResult evaluate(const Tensor<float>& x,const Tensor<float>& y) {

        if (!compiled_) {throw std::logic_error("el modelo no fue compilado");}

        Tensor<float> pred = predict( x );

        if (pred.shape() != y.shape()) {throw std::invalid_argument("las etiquetas no coinciden con la salida del modelo");}

        EvaluationResult result;

        result.loss = (*loss_)( y,pred );
        return result;
    }

    [[nodiscard]]
    std::map<std::string, Tensor<float>>
    parameters() const {

        std::map<std::string, Tensor<float>> params;

        int dense_count = 0;
        int conv_count = 0;

        for (const auto& layer : layers_) {

            auto layer_params = layer->parameters();

            if (layer_params.empty()) {continue;}

            std::string prefix =
                nombre_capa(
                    layer.get(),
                    dense_count,
                    conv_count
                );

            for (const auto& [name, tensor] : layer_params) {
                params[prefix + "/" + name] = copiar_tensor(tensor);
            }
        }
        return params;
    }

    [[nodiscard]]
    std::map<std::string, Tensor<float>>
    last_gradients() const {return last_gradients_;}

private:

    std::vector<std::unique_ptr<Layer>> layers_;

    bool compiled_ = false;
    bool last_backward_ready_ = false;

    std::optional<optimizers::SGD> optimizer_;
    std::optional<losses::CategoricalCrossentropy> loss_;

    Tensor<float> last_prediction_;
    Tensor<float> last_loss_gradient_;

    std::map<std::string, Tensor<float>> last_gradients_;

    Tensor<float> batch_slice(const Tensor<float>& tensor, int start, int cantidad) const {

        std::vector<int> dims = tensor.shape().dims();

        dims[0] = cantidad;

        Tensor<float> out = Tensor<float>::zeros(Shape(dims));

        int por_muestra = 1;

        for (std::size_t i = 1; i < tensor.rank(); ++i) {por_muestra *= tensor.shape()[i];}

        for (int b = 0; b < cantidad; ++b) {
            for (int j = 0; j < por_muestra; ++j) {
                out[ static_cast<std::size_t>(b * por_muestra + j) ] = tensor[ static_cast<std::size_t>((start + b) * por_muestra + j) ];
            }
        }
        return out;
    }

    static Tensor<float> copiar_tensor(const Tensor<float>& tensor) {
        Tensor<float> copia = Tensor<float>::zeros(tensor.shape());

        for (std::size_t i = 0;i < tensor.size();++i) {
            copia[i] = tensor[i];
        }
        return copia;
    }

    static std::string nombre_capa(
        const Layer* layer,
        int& dense_count,
        int& conv_count
    ) {
        if (dynamic_cast<const layers::Dense*>(layer)) {
            return "dense_" + std::to_string(dense_count++);
        }

        if (dynamic_cast<const layers::Conv2D*>(layer)) {
            return "conv2d_" + std::to_string(conv_count++);
        }

        return"layer_" + std::to_string( dense_count + conv_count );
    }

    void registrar_gradientes() {

        last_gradients_.clear();

        int dense_count = 0;
        int conv_count = 0;

        for (const auto& layer : layers_) {

            auto layer_grads = layer->gradients();

            if (layer_grads.empty()) {continue;}

            std::string prefix =
                nombre_capa(
                    layer.get(),
                    dense_count,
                    conv_count
                );

            for (const auto& [name, tensor] : layer_grads) {
                last_gradients_[prefix + "/" + name] = copiar_tensor(tensor);
            }
        }
    }

    void actualizar_parametros() {

        if (!optimizer_) {throw std::logic_error("el modelo no tiene optimizador");}

        for (auto& layer : layers_) {
            auto grads = layer->gradients();

            if (grads.empty()) {continue;}

            auto& params = layer->parameters_ref();

            for (const auto& [name, grad] : grads) {
                optimizer_->update(params.at(name),grad);
            }
        }
    }
};

}

#endif