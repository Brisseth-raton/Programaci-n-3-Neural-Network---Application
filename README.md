# CryptoVision
## Neural Network Library and Cryptocurrency Pattern Classifier

**Curso:** CS2023 – Programación III  
**Proyecto Final – 2026-1**

---

# Integrantes

- Angel Flores 
- Carolina Castro Gomez
- Luis Fabian
- Brisseth Surquislla Gamarra

---

# 1. Resumen

## Problema

El análisis manual de textos cifrados es un proceso lento y complejo debido a la gran variedad de algoritmos y la dificultad para reconocer sus patrones visualmente. Esta limitación afecta diariamente tanto a profesionales de la seguridad de datos como a participantes de competencias de criptografía y hacking (CTFs), donde el tiempo es un factor crítico.

## Objetivo

Desarrollar una biblioteca de Redes Neuronales desde cero en **C++23**, utilizando **Eigen** como backend numérico, con una API inspirada en TensorFlow/Keras, capaz de entrenar modelos y aplicarlos a un problema real.

Como aplicación práctica se implementó CryptoVision, una herramienta basada en redes neuronales capaz de reconocer patrones criptográficos y clasificar textos cifrados utilizando uno o dos algoritmos de cifrado aprendidos durante el entrenamiento.

---

# Objetivos del proyecto

Implementar una biblioteca que incluya:

- Tensor
- Shape
- Sequential
- Dense
- Conv2D
- MaxPooling2D
- Flatten
- Funciones de activación
- Funciones de pérdida
- SGD
- Forward Propagation
- Backpropagation
- Gradientes
- Entrenamiento
- Inferencia
- Evaluación
- Persistencia del modelo

Finalmente utilizar esta biblioteca para resolver un problema práctico.

---

# Tecnologías utilizadas

- C++23
- Eigen
- CMake
- STL
- Catch2 (pruebas)
- CSV
- Git

---

# 2. Estructura del proyecto

```
.
├── include/
│   ├── algebra/
│   ├── io/
│   ├── monitoring/
│   ├── nn/
│   └── utec/
│
├── src/
│
├── tests/
│
├── docs/
│
├── data/
│
├── models/
│
├── reports/
│
├── CMakeLists.txt
└── README.md
```

---

# Descripción de carpetas

## include/

Contiene toda la biblioteca pública.

### algebra/

Backend numérico basado completamente en Eigen.

Incluye:

- Tensor
- Shape
- Operaciones matemáticas
- Backend tensorial

---

### nn/

Implementación de la red neuronal.

Contiene:

- Dense
- Conv2D
- Flatten
- MaxPooling2D
- Funciones de activación
- Funciones de pérdida
- SGD
- Sequential
- Grafo computacional
- NeuralNetwork

---

### io/

Manejo de archivos.

Incluye:

- lectura de datasets
- serialización
- carga de modelos

---

### monitoring/

Herramientas de monitoreo.

Incluye:

- métricas
- historial de entrenamiento
- exportación de resultados

---

### utec/

Interfaces públicas del proyecto.

---

## src/

Implementación de todas las clases.

---

## tests/

Pruebas unitarias por feature.

---

## docs/

Documentación adicional.

---

## data/

Dataset utilizado para entrenamiento.

---

## models/

Modelos entrenados.

---

## reports/

Resultados y métricas.

---

# 3. Arquitectura

La arquitectura sigue una organización modular inspirada en bibliotecas modernas de Deep Learning.

```
Dataset
    │
    ▼
DatasetLoader
    │
    ▼
Tensor
    │
    ▼
Sequential
    │
 ┌──┴────────────┐
 │               │
 ▼               ▼
Dense         Conv2D
 │               │
 ▼               ▼
Activations  MaxPooling
 │
 ▼
Flatten
 │
 ▼
Loss
 │
 ▼
Backpropagation
 │
 ▼
SGD Optimizer
 │
 ▼
Updated Model
 │
 ▼
Serializer
```

---

# Flujo general

1. Leer textos cifrados y etiquetas.
2. Convertir datos a Tensor.
3. Construir la red.
4. Forward Propagation.
5. Calcular pérdida.
6. Backpropagation.
7. Actualizar parámetros.
8. Evaluar.
9. Guardar modelo.
10. Realizar inferencia.

---

# API tipo Keras

La biblioteca fue diseñada con una API similar a TensorFlow/Keras.

Ejemplo:

```cpp
Sequential model;

model.add(Dense(64));
model.add(ReLU());

model.add(Dense(32));
model.add(ReLU());

model.add(Dense(3));
model.add(Softmax());

model.compile(
    CrossEntropy(),
    SGD(0.01)
);

model.fit(X_train, y_train, 100);

model.evaluate(X_test, y_test);

model.predict(sample);
```

---

# 4. Componentes implementados

## Backend numérico

✔ Shape

✔ Tensor

✔ Operaciones matemáticas

✔ Multiplicación matricial

✔ Broadcasting

✔ Convolución

---

## Capas

✔ Input

✔ Dense

✔ Conv2D

✔ MaxPooling2D

✔ Flatten

---

## Activaciones

✔ ReLU

✔ Sigmoid

✔ Tanh

✔ Softmax

---

## Funciones de pérdida

✔ Mean Squared Error

✔ Cross Entropy

---

## Optimizador

✔ SGD

---

## Entrenamiento

Implementado mediante:

- Forward Propagation
- Backward Propagation
- Cálculo de gradientes
- Actualización de pesos
- Bias
- Learning Rate

---

# Persistencia

El modelo puede:

- Guardarse en disco.
- Cargarse posteriormente.
- Utilizarse para inferencia sin volver a entrenar.

---

# 5. Aplicación práctica

## CryptoVision

La aplicación utiliza la biblioteca desarrollada para analizar textos cifrados y reconocer automáticamente patrones criptográficos, clasificando mensajes protegidos mediante un algoritmo de cifrado o una combinación de dos algoritmos.

El flujo es:

```
Textos cifrados (.txt)
          │
          ▼
 TextDatasetLoader
          │
          ▼
       Tensor
          │
          ▼
   Neural Network
          │
          ▼
   Clasificación
```

---

# Resultados

Durante el entrenamiento se registran:

- Loss
- Accuracy
- Epochs
- Tiempo de entrenamiento

Además se exportan:

- Historial de entrenamiento
- Modelo entrenado
- Resultados finales

---

# Evidencias

El proyecto genera automáticamente:

- Modelo serializado
- CSV de entrenamiento
- Métricas
- Resultados de inferencia

---

# 6. Pruebas

Se implementaron pruebas unitarias para verificar cada feature principal.

Incluyen:

- Tensor
- Shape
- Dense
- Activaciones
- Conv2D
- MaxPooling
- Flatten
- Loss
- Optimizer
- Dataset Loader
- Serialización
- Neural Network

También se probaron casos de error:

- Dimensiones incompatibles
- Tensores vacíos
- Parámetros inválidos
- Accesos fuera de rango
- Errores de carga de archivos

---

# Ejecución

## Requisitos

- C++23
- CMake 3.20+
- Eigen

---

## Compilar

```bash
mkdir build

cd build

cmake ..

cmake --build .
```

---

## Ejecutar aplicación

```bash
./cryptovision
```

En Windows:

```bash
cryptovision.exe
```

---

## Ejecutar pruebas

```bash
ctest
```

o

```bash
./test_tensor

./test_dense

./test_conv

...
```

---

# Dataset

El proyecto utiliza dos archivos de texto:

• cifrados.txt
• labels.txt

Cada línea del archivo de textos cifrados corresponde a una muestra, mientras que cada línea del archivo de etiquetas contiene el algoritmo de cifrado esperado para esa muestra.

---

# 7. Conclusiones

Se desarrolló exitosamente una biblioteca básica de Redes Neuronales en C++23 utilizando Eigen como backend numérico.

La biblioteca permite entrenar modelos completamente desde cero mediante Forward Propagation, Backpropagation y SGD.

Como validación práctica se implementó CryptoVision, demostrando que la biblioteca desarrollada puede aplicarse al reconocimiento automático de patrones criptográficos y a la clasificación de textos cifrados.

---

# Limitaciones

Actualmente el proyecto:

- utiliza únicamente SGD.
- no implementa GPU.
- no soporta entrenamiento distribuido.
- no posee optimizaciones SIMD avanzadas.
- utiliza una CNN simplificada para fines educativos.

---

# Mejoras futuras

- Adam Optimizer
- Dropout
- Batch Normalization
- Mini Batch Training
- Early Stopping
- GPU Backend
- Exportación ONNX
- Visualización del entrenamiento

---

# Referencias

- Ian Goodfellow, Deep Learning.
- Michael Nielsen, Neural Networks and Deep Learning. 
- Eigen Official Documentation
  https://eigen.tuxfamily.org
- TensorFlow
  https://tensorflow.org
- Keras Documentation.
- C++ Reference.
- Dcode

---