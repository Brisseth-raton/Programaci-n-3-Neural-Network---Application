#ifndef PROYECTO_NN_OPS_H
#define PROYECTO_NN_OPS_H

#include <stdexcept>
#include <vector>
#include "utec/algebra/tensor_backend.h"

namespace utec::tf {
struct Strides {
    int filas;
    int colums;
};
enum class Padding { Valid };

namespace ops {

template <typename T>
Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape() != b.shape()) { throw std::invalid_argument("las formas de los tensores deben coincidir"); }
    Tensor<T> out = Tensor<T>::zeros(a.shape());

    for (std::size_t i = 0; i < a.size(); ++i) { out[i] = a[i] * b[i]; }
    return out;
}

template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.rank() != 2 || b.rank() != 2) { throw std::invalid_argument( "matmul requiere tensores de rango 2" ); }

    int m = a.shape()[0];
    int k = a.shape()[1];
    int k2 = b.shape()[0];
    int n = b.shape()[1];

    if (k != k2) { throw std::invalid_argument( "las dimensiones de matmul son incompatibles" ); }

    Tensor<T> out = Tensor<T>::zeros(Shape{m, n});

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T acld = T{};
            for (int t = 0; t < k; ++t) { acld += a(i, t) * b(t, j); }
            out(i, j) = acld;
        }
    }
    return out;
}

template <typename T>
Tensor<T> transpose2d(const Tensor<T>& x) {
    if (x.rank() != 2) { throw std::invalid_argument( "transpose2d requiere un tensor de rango 2" ); }

    int filas = x.shape()[0];
    int colums = x.shape()[1];

    Tensor<T> out = Tensor<T>::zeros( Shape{colums, filas} );

    for (int i = 0; i < filas; ++i) { for (int j = 0; j < colums; ++j) { out(j, i) = x(i, j); } }
    return out;
}

template <typename T>
Tensor<T> flatten_batch(const Tensor<T>& x) {
    if (x.rank() < 2) { throw std::invalid_argument( "flatten_batch requiere tensores de rango mayor o igual a 2" ); }

    int batch = x.shape()[0];
    int features = 1;

    for (std::size_t i = 1; i < x.rank(); ++i) { features *= x.shape()[i]; }

    Tensor<T> out = x;
    out.reshape(Shape{batch, features});
    return out;
}

template <typename T>
Tensor<T> conv2d(
    const Tensor<T>& input,
    const Tensor<T>& kernel,
    Strides strides = {1, 1},
    Padding padding = Padding::Valid
) {
    if (padding != Padding::Valid) { throw std::invalid_argument( "solo se soporta padding válido" ); }
    if (input.rank() != 4 || kernel.rank() != 4) { throw std::invalid_argument( "conv2d requiere tensores de rango 4" ); }
    if (strides.filas <= 0 || strides.colums <= 0) { throw std::invalid_argument("los strides deben ser mayores que cero"); }

    int batch = input.shape()[0];
    int hetd = input.shape()[1];
    int aetd = input.shape()[2];
    int cetd = input.shape()[3];

    int alkern = kernel.shape()[0];
    int ankern = kernel.shape()[1];
    int cankern = kernel.shape()[2];
    int cansld = kernel.shape()[3];

    if (cetd != cankern) { throw std::invalid_argument("la cantidad de canales no coincide"); }
    if (alkern > hetd || ankern > aetd) { throw std::invalid_argument( "el kernel es más grande que la entrada" ); }

    int alsld = (hetd - alkern) / strides.filas + 1;
    int ansld = (aetd - ankern) / strides.colums + 1;

    Tensor<T> out = Tensor<T>::zeros( Shape{batch, alsld, ansld, cansld} );

    for (int e = 0; e < batch; ++e) {
        for (int fsld = 0; fsld < alsld; ++fsld) {
            for (int colsld = 0; colsld < ansld; ++colsld) {
                for (int flto = 0; flto < cansld; ++flto) {

                    T acld = T{};

                    for (int fkern = 0; fkern < alkern; ++fkern) {
                        for (int colkern = 0; colkern < ankern; ++colkern) {
                            for (int can = 0; can < cetd; ++can) {

                                int ih = fsld * strides.filas + fkern;
                                int iw = colsld * strides.colums + colkern;

                                acld += input(e, ih, iw, can) * kernel(fkern, colkern, can, flto);
                            }
                        }
                    }
                    out(e, fsld, colsld, flto) = acld;
                }
            }
        }
    }
    return out;
}
}
}
#endif