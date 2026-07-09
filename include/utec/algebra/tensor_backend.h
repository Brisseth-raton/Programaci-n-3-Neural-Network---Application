#ifndef PROYECTO_TENSOR_BACKEND_H
#define PROYECTO_TENSOR_BACKEND_H

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <span>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <memory>

#include "utec/algebra/shape.h"

namespace utec::tf {

template <typename T>
class Tensor {
public:

    Tensor(): forma_(),datos_(std::make_shared<Eigen::Array<T, Eigen::Dynamic, 1>>()) {}

    explicit Tensor(const Shape& forma): forma_(forma),datos_(std::make_shared<Eigen::Array<T, Eigen::Dynamic, 1>>(static_cast<Eigen::Index>(forma.numel()))) { datos_->setZero(); }

    Tensor(const Shape& forma,const T& valor): forma_(forma),datos_(std::make_shared<Eigen::Array<T, Eigen::Dynamic, 1>>(static_cast<Eigen::Index>(forma.numel()))) { datos_->setConstant(valor); }

    Tensor(const Shape& forma,const std::vector<T>& valores): forma_(forma),datos_(
    std::make_shared<Eigen::Array<T, Eigen::Dynamic, 1>
    >(static_cast<Eigen::Index>(forma.numel()))) {
        if (valores.size()!= forma.numel()) {
            throw std::invalid_argument("cantidad de datos incompatible con forma");
        }

        for (std::size_t i = 0;i < valores.size();++i) {
            (*datos_)(static_cast<Eigen::Index>(i)) = valores[i];
        }
    }

    Tensor& operator=(const Tensor&) = default;

    [[nodiscard]]
    std::size_t rank() const noexcept { return forma_.rank(); }
    [[nodiscard]]
    std::size_t numel() const noexcept { return forma_.numel(); }
    [[nodiscard]]
    std::size_t size() const noexcept { return numel(); }
    [[nodiscard]]
    const Shape& shape() const noexcept { return forma_; }

    static Tensor zeros(const Shape& forma) {
        Tensor t(forma);
        t.datos_->setZero();
        return t;
    }

    static Tensor ones(const Shape& forma) {
        Tensor t(forma);
        t.datos_->setOnes();
        return t;
    }

    static Tensor from_data(const Shape& forma,const std::vector<T>& v) {
        return Tensor(forma, v);
    }

    Tensor operator+(const Tensor& otro) const {
        if (forma_ != otro.forma_) { throw std::invalid_argument("formas incompatibles"); }

        Tensor resultado(forma_);

        *(resultado.datos_) = (*datos_) + (*otro.datos_);
        return resultado;
    }

    Tensor operator-(const Tensor& otro) const {
        if (forma_!=otro.forma_) { throw std::invalid_argument("formas incompatibles"); }

        Tensor resultado( forma_ );

        for (Eigen::Index i = 0;i < datos_->size();++i) {
            resultado.datos_->operator()(i) = datos_->operator()(i) - otro.datos_->operator()(i);
        } return resultado;
    }

    void reshape(const Shape& forma_nueva) {
        if (forma_nueva.numel() !=numel()) {
            throw std::invalid_argument("reshape incompatible");
        }
        forma_ = forma_nueva;
    }

    Tensor reshaped(const Shape& forma_nueva) const {
        if (forma_nueva.numel() != numel()) { throw std::invalid_argument("reshape incompatible");}

        Tensor resultado( forma_nueva );

        *(resultado.datos_) = *datos_;

        return resultado;
    }

    T& operator[]( std::size_t i ) { return (*datos_)(i); }

    const T& operator[]( std::size_t i ) const { return (*datos_)(i); }

    T& at(std::initializer_list<int> coords) {
        return (*datos_)(
            static_cast<Eigen::Index>(
                lin_idx(std::span<const int>(
                        coords.begin(),
                        coords.size()
                    )
                )
            )
        );
    }

    const T& at(std::initializer_list<int> coords
    ) const {
        return (*datos_)(
            static_cast<Eigen::Index>(lin_idx(std::span<const int>(
                        coords.begin(),
                        coords.size()
                    )
                )
            )
        );
    }

    template <typename... Ix>
    T& operator()(Ix... ix) {

        int indices[] = { static_cast<int>(ix)... };

        return (*datos_)(
            static_cast<Eigen::Index>(lin_idx(std::span<const int>(
                        indices, sizeof...(ix)
                    )
                )
            )
        );
    }

    template <typename... Ix>
    const T& operator()(Ix... ix) const {

        int indices[] = { static_cast<int>(ix)... };

        return (*datos_)(
            static_cast<Eigen::Index>(lin_idx(std::span<const int >( indices, sizeof...(ix)
                    )
                )
            )
        );
    }

private:

    Shape forma_;

    std::shared_ptr<Eigen::Array<T,Eigen::Dynamic,1>> datos_;

    std::size_t lin_idx(std::span<const int> coords) const {

        if (coords.size() != forma_.rank() ) { throw std::invalid_argument("rango de índice inválido");}

        std::size_t indice_lineal = 0;
        std::size_t salto = 1;

        for (std::size_t i = forma_.rank();i-- > 0;) {

            if ( coords[i] < 0 || coords[i] >= forma_[i] ) { throw std::out_of_range("índice fuera de rango"); }

            indice_lineal +=static_cast<std::size_t>(coords[i]) *salto;

            salto*=static_cast<std::size_t>(forma_[i]);
        }
        return indice_lineal;
    }
};

    template <typename T>
    bool allclose(const Tensor<T>& a,const Tensor<T>& b,T atol = static_cast<T>(1e-5)) {
        if (a.shape() != b.shape()) {return false;}
        for (std::size_t i = 0; i < a.size(); ++i) {
            T diff = a[i] - b[i];
            if (diff < static_cast<T>(0)) {diff = -diff;}
            if (diff > atol) {return false;}
        }
        return true;
    }
}

using utec::tf::Shape;
using utec::tf::Tensor;

#endif