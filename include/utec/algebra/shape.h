#ifndef PROYECTO_SHAPE_H
#define PROYECTO_SHAPE_H

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <cstddef>
#include <utility>
#include <ostream>

namespace utec::tf {

class Shape {
public:
    Shape() = default;

    Shape(std::initializer_list<int> ejes) : ejes_(ejes) { validar(); }

    Shape(std::initializer_list<std::size_t> ejes) {
        for (auto eje : ejes) { ejes_.push_back(static_cast<int>(eje)); }
        validar();
    }

    explicit Shape(std::vector<int> ejes) : ejes_(std::move(ejes)) { validar(); }

    explicit Shape(const std::vector<std::size_t>& ejes) {
        for (auto eje : ejes) { ejes_.push_back(static_cast<int>(eje)); }
        validar();
    }

    [[nodiscard]] std::size_t rank() const noexcept { return ejes_.size(); }
    [[nodiscard]] std::size_t size() const noexcept { return ejes_.size(); }

    [[nodiscard]] std::size_t numel() const {
        std::size_t total = 1;
        for (int eje : ejes_) { total *= static_cast<std::size_t>(eje); }
        return total;
    }

    [[nodiscard]] std::size_t total_size() const { return numel(); }
    [[nodiscard]] bool empty() const noexcept { return ejes_.empty(); }
    [[nodiscard]] const std::vector<int>& dims() const noexcept { return ejes_; }
    [[nodiscard]] int operator[](std::size_t i) const { return ejes_[i]; }

    friend bool operator==(const Shape&, const Shape&) = default;

    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "{";
        for (std::size_t i = 0; i < shape.ejes_.size(); ++i) {
            os << shape.ejes_[i];
            if (i + 1 < shape.ejes_.size()) { os << ", "; }
        }
        os << "}";
        return os;
    }

private:
    std::vector<int> ejes_;

    void validar() const {
        for (int eje : ejes_) { if (eje <= 0) { throw std::invalid_argument("dimensiones deben ser > 0"); } }
    }
};
}

using utec::tf::Shape;

#ifndef PROYECTO_TENSOR_BACKEND_H
#include "utec/algebra/tensor_backend.h"
using utec::tf::Tensor;
#endif

#endif