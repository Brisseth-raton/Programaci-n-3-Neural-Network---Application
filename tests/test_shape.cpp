#include "check.h"

#include "utec/algebra/shape.h"

#include <vector>

using namespace utec::tf;

int main() {

    TestRunner test;

    // =====================================================
    // Constructor y propiedades básicas
    // =====================================================

    {
        Shape s{2,3,4};

        CHECK(test, s.rank() == 3);
        CHECK(test, s.size() == 3);
        CHECK(test, s.numel() == 24);
        CHECK(test, s.total_size() == 24);
        CHECK(test, !s.empty());
    }

    // =====================================================
    // Acceso por índice
    // =====================================================

    {
        Shape s{5,7,9};

        CHECK(test, s[0] == 5);
        CHECK(test, s[1] == 7);
        CHECK(test, s[2] == 9);
    }

    // =====================================================
    // dims()
    // =====================================================

    {
        Shape s{4,8,2};

        std::vector<int> esperado = {4,8,2};

        CHECK(test, s.dims() == esperado);
    }

    // =====================================================
    // Igualdad
    // =====================================================

    {
        Shape a{2,3};
        Shape b{2,3};
        Shape c{3,2};

        CHECK(test, a == b);
        CHECK(test, !(a == c));
    }

    // =====================================================
    // Shape vacío
    // =====================================================

    {
        Shape s;

        CHECK(test, s.empty());
        CHECK(test, s.rank() == 0);
        CHECK(test, s.size() == 0);
        CHECK(test, s.numel() == 1);
    }

    // =====================================================
    // Constructor desde vector<int>
    // =====================================================

    {
        std::vector<int> dims = {3,4,5};

        Shape s(dims);

        CHECK(test, s.rank() == 3);
        CHECK(test, s.numel() == 60);
        CHECK(test, s[0] == 3);
        CHECK(test, s[1] == 4);
        CHECK(test, s[2] == 5);
    }

    // =====================================================
    // Constructor desde vector<size_t>
    // =====================================================

    {
        std::vector<size_t> dims = {2,2,5};

        Shape s(dims);

        CHECK(test, s.rank() == 3);
        CHECK(test, s.numel() == 20);
    }

    // =====================================================
    // Una dimensión
    // =====================================================

    {
        Shape s{8};

        CHECK(test, s.rank() == 1);
        CHECK(test, s.numel() == 8);
        CHECK(test, s[0] == 8);
    }

    // =====================================================
    // Excepciones
    // =====================================================

    CHECK_THROWS(test, Shape{0});
    CHECK_THROWS(test, Shape{-1});

    return test.summary();
}