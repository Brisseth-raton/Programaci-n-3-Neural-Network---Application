cmake_minimum_required(VERSION 3.20)
project(CryptoVisionAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

add_executable(cryptovision
        src/cryptovision_main.cpp
)

target_include_directories(cryptovision PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)





add_executable(test_shape
        tests/test_shape.cpp
        tests/check.h
)
add_test(NAME test_shape COMMAND test_shape)


target_include_directories(test_shape PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_tensor
        tests/test_tensor.cpp
        tests/check.h
)
add_test(NAME test_tensor COMMAND test_tensor)


target_include_directories(test_tensor PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_tensor_ops
        tests/test_tensor_ops.cpp
        tests/check.h
)
add_test(NAME test_tensor_ops COMMAND test_tensor_ops)


target_include_directories(test_tensor_ops PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_activation
        tests/test_activation.cpp
        tests/check.h
)
add_test(NAME test_activation COMMAND test_activation)


target_include_directories(test_activation PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)


add_executable(test_dense
        tests/test_dense.cpp
        tests/check.h
)
add_test(NAME test_dense COMMAND test_dense)


target_include_directories(test_dense PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_conv2d
        tests/test_conv2d.cpp
        tests/check.h
)
add_test(NAME test_conv2d COMMAND test_conv2d)


target_include_directories(test_conv2d PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_flatten
        tests/test_flatten.cpp
        tests/check.h
)
add_test(NAME test_flatten COMMAND test_flatten)


target_include_directories(test_flatten PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_pooling
        tests/test_pooling.cpp
        tests/check.h
)
add_test(NAME test_pooling COMMAND test_pooling)


target_include_directories(test_pooling PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_loss
        tests/test_loss.cpp
        tests/check.h
)
add_test(NAME test_loss COMMAND test_loss)


target_include_directories(test_loss PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_optimizer
        tests/test_optimizer.cpp
        tests/check.h
)
add_test(NAME test_optimizer COMMAND test_optimizer)


target_include_directories(test_optimizer PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_graph
        tests/test_graph.cpp
        tests/check.h
)
add_test(NAME test_graph COMMAND test_graph)


target_include_directories(test_graph PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_neural_network
        tests/test_neural_network.cpp
        tests/check.h
)
add_test(NAME test_neural_network COMMAND test_neural_network)


target_include_directories(test_neural_network PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_io
        tests/test_io.cpp
        tests/check.h
)
add_test(NAME test_io COMMAND test_io)


target_include_directories(test_io PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_cryptovision
        tests/test_cryptovision.cpp
        tests/check.h
)
add_test(NAME test_cryptovision COMMAND test_cryptovision)


target_include_directories(test_cryptovision PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)

add_executable(test_exceptions
        tests/test_exceptions.cpp
        tests/check.h
)
add_test(NAME test_exceptions COMMAND test_exceptions)

target_include_directories(test_exceptions PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/eigen
)






if(MSVC)
    target_compile_options(cryptovision PRIVATE /W4)
    target_compile_options(test_shape PRIVATE /W4)
else()
    target_compile_options(cryptovision PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_shape PRIVATE -Wall -Wextra -Wpedantic)
endif()