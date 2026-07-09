//
// Created by HP on 25/06/2026.
//

#ifndef PROYECTO_NN_GRAPH_H
#define PROYECTO_NN_GRAPH_H

#include <vector>
#include <memory>
#include <stdexcept>

#include "utec/nn/nn_interfaces.h"

namespace utec::tf {

    class SequentialGraph {
    public:

        SequentialGraph() = default;

        [[nodiscard]]
        bool empty() const noexcept {return layers_.empty();}

        template <typename LayerType>
        void add(const LayerType& layer) {
            layers_.push_back(layer.clone());
        }

        Tensor<float> forward(const Tensor<float>& input) {
            if (layers_.empty()) { throw std::logic_error("el grafo no tiene capas"); }

            Tensor<float> output = input;

            for (auto& layer : layers_) { output = layer->forward(output); }
            return output;
        }

        Tensor<float> backward(const Tensor<float>& grad_output) {
            if (layers_.empty()) { throw std::logic_error("el grafo no tiene capas"); }

            Tensor<float> grad = grad_output;

            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) { grad = (*it)->backward(grad); }
            return grad;
        }
    private:
        std::vector<std::unique_ptr<utec::tf::Layer>> layers_;
    };
}

#endif //PROYECTO_NN_GRAPH_H