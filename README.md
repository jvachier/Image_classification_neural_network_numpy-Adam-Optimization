# Neural Network for Digit Classification with Adam Optimization

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Implementation-013243.svg)](https://numpy.org/)

A from-scratch implementation of a neural network for handwritten digit classification (0-9) using only NumPy, featuring the Adam optimization algorithm. This project demonstrates the fundamentals of deep learning by building a complete neural network without high-level frameworks.

## Overview

This project implements a 3-layer neural network that classifies handwritten digits from the MNIST dataset. The implementation compares the performance of Adam optimization against standard gradient descent, showcasing the effectiveness of adaptive learning rate methods.

### Key Features

- **Pure NumPy Implementation**: No deep learning frameworks (TensorFlow, PyTorch) used for the core network
- **Adam Optimization**: Full implementation of the Adam optimizer as described in the original paper
- **Performance Comparison**: Side-by-side comparison of Adam-optimized vs. standard gradient descent
- **High Accuracy**: Achieves 100% accuracy on the training set with Adam optimization
- **Educational**: Clear, documented code ideal for learning neural network fundamentals

## Project Structure

```
├── Image_classification_neural_network_numpy-Adam Optimization.ipynb
├── README.md
└── LICENCE
```

## Neural Network Architecture

The network consists of three layers:

| Layer | Type | Neurons | Activation |
|-------|------|---------|------------|
| Input | Dense | 784 (28×28 pixels) | - |
| Hidden 1 | Dense | 128 | ReLU |
| Hidden 2 | Dense | 40 | ReLU |
| Output | Dense | 10 (digits 0-9) | Softmax |

**Loss Function**: Mean Squared Error (MSE)  
**Optimization**: Adam (β₁=0.9, β₂=0.99, ε=1e-8)

## Dataset

- **Source**: [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer)
- **Training Set**: 42,000 labeled images
- **Test Set**: 28,000 unlabeled images
- **Image Format**: 28×28 grayscale pixels (784 features)

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
tensorflow  # Only used for validation metrics
pillow
```

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/jvachier/Image_classification_neural_network_numpy-Adam-Optimization.git
   cd Image_classification_neural_network_numpy-Adam-Optimization
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow pillow
   ```

3. **Download the dataset**
   - Download `train.csv` and `test.csv` from [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
   - Place them in the project directory

4. **Run the notebook**
   ```bash
   jupyter notebook "Image_classification_neural_network_numpy-Adam Optimization.ipynb"
   ```

## Implementation Details

### Adam Optimization Algorithm

The Adam (Adaptive Moment Estimation) optimizer combines the advantages of two popular methods:
- **RMSprop**: Uses adaptive learning rates
- **Momentum**: Accelerates convergence in relevant directions

The update rules are:

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
```

```math
\hat{m}_t = \frac{m_t}{1-\beta_1^t}
```

```math
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
```

```math
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

### Forward Propagation

1. **Layer 1**: `Z[1] = W[1]X + b[1]`, `A[1] = ReLU(Z[1])`
2. **Layer 2**: `Z[2] = W[2]A[1] + b[2]`, `A[2] = ReLU(Z[2])`
3. **Layer 3**: `Z[3] = W[3]A[2] + b[3]`, `A[3] = Softmax(Z[3])`

### Backpropagation

Gradients are computed using the chain rule and used to update weights and biases through the Adam optimizer.

## Results

- **Training Accuracy**: 100% (with Adam optimization)
- **Convergence**: Significantly faster with Adam compared to standard gradient descent
- **Visualization**: Includes training curves for loss, accuracy, MSE, and R² score

## References

- Kingma, D. P., & Ba, J. (2014). [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf). *arXiv preprint arXiv:1412.6980*.

## Links

- **Kaggle Notebook**: [Classification with Neural Network - Adam - NumPy](https://www.kaggle.com/code/jvachier/classification-with-neural-network-adam-numpy)
- **Dataset**: [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See the [LICENCE](LICENCE) file for details.

## Author

**jvachier**  
*Created: July 2022*

---

### Acknowledgments

This project was created as an educational exercise to understand the inner workings of neural networks and optimization algorithms by implementing them from scratch.
