#ifndef PROYECTO_TENSOR_OPS_H
#define PROYECTO_TENSOR_OPS_H

#include <cmath>

#include "utec/algebra/tensor_backend.h"

namespace utec::tf {

    template <typename T>
    Tensor<T> matmul( const Tensor<T>& a, const Tensor<T>& b );
    template <typename T>
    Tensor<T> multiply( const Tensor<T>& a, const Tensor<T>& b );
    template <typename T>
    Tensor<T> transpose2d( const Tensor<T>& x );
    template <typename T>
    Tensor<T> conv2d( const Tensor<T>& input, const Tensor<T>& kernel );

}

#endif