# Lambda AI 🚀

**Lambda AI** is a lightweight, pure, and fully functional Deep Learning framework built from scratch in **Haskell**. Designed with a strong emphasis on **Fully Convolutional Networks (FCNs)**, it bridges the gap between the rigorous functional paradigms of Haskell and the flexible architecture paradigms popularized by modern frameworks like **TensorFlow** and **PyTorch**.

---

## 🌟 Key Features

* **Fully Functional Design:** Implemented entirely using immutable data structures, pure functions, and explicit tensor/gradient transformations without relying on mutable state or hidden side effects.
* **Focus on Fully Convolutional Networks (FCNs):** Optimized for spatial data transformations, supporting multidimensional convolutions, custom padding, and structural layer composition.
* **Modular Meta-Layers:** Easily define complex network architectures using declarative meta-layers (e.g., `MetaConvolution`, `MetaActivation`).
* **Flexible Activation Functions:** Out-of-the-box support for multiple standard activation functions with built-in gradient evaluation.
* **Inspired by PyTorch & TensorFlow:** Familiar API concepts adapted to idiomatic Haskell, making it intuitive for deep learning practitioners transitioning to functional programming.

---

## ⚙️ Supported Activation Functions

Lambda AI comes equipped with a diverse set of activation functions and their respective derivatives for backpropagation:

* **Sigmoid** (`sigmoid`)
* **ReLU** (`relu`)
* **Leaky ReLU** (`leakyrelu`)
* **Tanh** (`tanh`)
* **ELU** (`elu`)

---

## 🛠️ Project Architecture

The framework is organized into modular core components:
* **`Architeture`**: Core type definitions (`Inputs`, `Results`, `Weights`, `Gradients`, `Dimensions`) and tensor convolution/padding operations.
* **`Convolutional`**: Implementation of convolutional perceptrons, layer generation, forward passes, and backpropagation slices.
* **`Layers`**: High-level wrapper mapping meta-layers to operational layers (Convolutional & Activation).
* **`Model`**: Pipeline orchestration, batch processing, loss calculations, and training loops (`train`, `trainEpoch`, `trainBatch`).
* **`ActivationFunction`**: Collection of activation formulas and derivatives.

---

## 📦 Installation & Getting Started

Lambda AI uses **Stack** for dependency management and reproducible builds.

### Prerequisites
* [Haskell Cabal](https://www.haskell.org/cabal/index.html)
* [Optional - Haskell Stack](https://docs.haskellstack.org/en/stable/README/)
* [For use the jupyter environment](https://github.com/DL-2026-1/jupyter-haskell)

### Local Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/DL-2026-1/lambda-ai.git](https://github.com/DL-2026-1/lambda-ai.git)
   cd lambda-ai
   ```
2. Build the project:
    ```bash
   cabal build
   ```
3. Test it:
    ```bash
   cabal test
   ```