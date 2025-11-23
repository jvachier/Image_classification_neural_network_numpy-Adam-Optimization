# Mathematical Documentation: Neural Network with Adam Optimization

This document provides a comprehensive mathematical breakdown of all equations and operations implemented in the neural network for digit classification.

---

## Table of Contents

1. [Activation Functions](#1-activation-functions)
2. [Loss and Cost Functions](#2-loss-and-cost-functions)
3. [Network Architecture](#3-network-architecture)
4. [Forward Propagation](#4-forward-propagation)
5. [Backpropagation](#5-backpropagation)
6. [Adam Optimization Algorithm](#6-adam-optimization-algorithm)
7. [Parameter Updates](#7-parameter-updates)
8. [Performance Metrics](#8-performance-metrics)

---

## 1. Activation Functions

### 1.1 ReLU (Rectified Linear Unit)

**Function:**
```math
\text{ReLU}(x) = \max(0, x)
```

**Derivative:**
```math
\frac{d\text{ReLU}(x)}{dx} = 
\begin{cases}
0 & \text{if } x \leq 0 \\
1 & \text{if } x > 0
\end{cases}
```

**Implementation:**
- `ReLU(x)` returns `max(0, x)` for each element
- `ReLU_derive(x)` returns 0 for negative values, 1 for positive values

**Purpose:** Introduces non-linearity while avoiding vanishing gradient problem, commonly used in hidden layers.

---

### 1.2 Sigmoid

**Function:**
```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

**Derivative:**
```math
\frac{d\sigma(x)}{dx} = \sigma(x) \cdot (1 - \sigma(x))
```

**Range:** (0, 1)

**Purpose:** Maps inputs to probability-like values, though not used in this specific implementation.

---

### 1.3 Softmax

**Function:**
```math
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
```

Where:
- `x_i` is the i-th element of the input vector
- `n` is the total number of classes (10 digits)

**Properties:**
- Output sums to 1: `∑ Softmax(x_i) = 1`
- Each output is in range (0, 1)
- Converts raw scores (logits) to probability distribution

**Purpose:** Used in output layer for multi-class classification, produces class probabilities.

---

## 2. Loss and Cost Functions

### 2.1 Mean Squared Error (MSE) - Element-wise

**Per-sample error:**
```math
\text{MSE}_{\text{sample}} = (y_{\text{true}} - y_{\text{pred}})^2
```

**Purpose:** Measures the squared difference between predicted and true values for backpropagation.

---

### 2.2 Cost Function (Overall MSE)

**Total cost:**
```math
J = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2
```

Where:
- `N` is the total number of samples
- `y_true` is the one-hot encoded true label
- `y_pred` is the predicted probability distribution from Softmax

**Implementation:**
```python
cost = sum((y_data - y_prediction)^2) / y_data.size
```

**Purpose:** Quantifies overall network performance across all training samples.

---

## 3. Network Architecture

### 3.1 Layer Configuration

| Layer | Type | Input Size | Output Size | Activation | Parameters |
|-------|------|------------|-------------|------------|------------|
| Input | - | 784 | 784 | - | 0 |
| Hidden 1 | Dense | 784 | 128 | ReLU | W₁(128×784), b₁(128×1) |
| Hidden 2 | Dense | 128 | 40 | ReLU | W₂(40×128), b₂(40×1) |
| Output | Dense | 40 | 10 | Softmax | W₃(10×40), b₃(10×1) |

### 3.2 Parameter Initialization

All weights and biases initialized from uniform distribution U(-0.5, 0.5):

```math
W^{[l]} \sim U(-0.5, 0.5)
```
```math
b^{[l]} \sim U(-0.5, 0.5)
```

This initialization:
- Centers values around 0
- Prevents symmetry breaking issues
- Provides reasonable starting gradients

---

## 4. Forward Propagation

### 4.1 Layer 1 (Input → Hidden 1)

**Linear transformation:**
```math
Z^{[1]} = W^{[1]} X + b^{[1]}
```

**Activation:**
```math
A^{[1]} = \text{ReLU}(Z^{[1]})
```

**Dimensions:**
- `X`: (784 × m) where m is batch size
- `W[1]`: (128 × 784)
- `b[1]`: (128 × 1), broadcasted to (128 × m)
- `Z[1]`: (128 × m)
- `A[1]`: (128 × m)

---

### 4.2 Layer 2 (Hidden 1 → Hidden 2)

**Linear transformation:**
```math
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}
```

**Activation:**
```math
A^{[2]} = \text{ReLU}(Z^{[2]})
```

**Dimensions:**
- `W[2]`: (40 × 128)
- `b[2]`: (40 × 1)
- `Z[2]`: (40 × m)
- `A[2]`: (40 × m)

---

### 4.3 Layer 3 (Hidden 2 → Output)

**Linear transformation:**
```math
Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}
```

**Activation (Softmax):**
```math
A^{[3]} = \text{Softmax}(Z^{[3]})
```

**Dimensions:**
- `W[3]`: (10 × 40)
- `b[3]`: (10 × 1)
- `Z[3]`: (10 × m)
- `A[3]`: (10 × m) - final predictions

---

## 5. Backpropagation

### 5.1 Output Layer Gradient (Layer 3)

**Error at output:**
```math
\delta^{[3]} = A^{[3]} - Y
```

Where:
- `A[3]` is the predicted probability distribution (from Softmax)
- `Y` is the true one-hot encoded labels
- This is the derivative of MSE loss with respect to the output

**Weight gradient:**
```math
\frac{\partial J}{\partial W^{[3]}} = \delta^{[3]} \cdot (A^{[2]})^T
```

**Bias gradient:**
```math
\frac{\partial J}{\partial b^{[3]}} = \frac{1}{m}\sum_{i=1}^{m} \delta_i^{[3]}
```

---

### 5.2 Hidden Layer 2 Gradient (Layer 2)

**Error propagation:**
```math
\delta^{[2]} = (W^{[3]})^T \delta^{[3]} \odot \text{ReLU}'(Z^{[2]})
```

Where:
- `⊙` denotes element-wise multiplication (Hadamard product)
- `ReLU'(Z[2])` is the derivative of ReLU

**Weight gradient:**
```math
\frac{\partial J}{\partial W^{[2]}} = \delta^{[2]} \cdot (A^{[1]})^T
```

**Bias gradient:**
```math
\frac{\partial J}{\partial b^{[2]}} = \frac{1}{m}\sum_{i=1}^{m} \delta_i^{[2]}
```

---

### 5.3 Hidden Layer 1 Gradient (Layer 1)

**Error propagation:**
```math
\delta^{[1]} = (W^{[2]})^T \delta^{[2]} \odot \text{ReLU}'(Z^{[1]})
```

**Weight gradient:**
```math
\frac{\partial J}{\partial W^{[1]}} = \delta^{[1]} \cdot X^T
```

**Bias gradient:**
```math
\frac{\partial J}{\partial b^{[1]}} = \frac{1}{m}\sum_{i=1}^{m} \delta_i^{[1]}
```

---

## 6. Adam Optimization Algorithm

Adam (Adaptive Moment Estimation) combines momentum and RMSprop for efficient optimization.

### 6.1 Hyperparameters

```math
\beta_1 = 0.9 \quad \text{(momentum decay rate)}
```
```math
\beta_2 = 0.99 \quad \text{(RMSprop decay rate)}
```
```math
\epsilon = 10^{-8} \quad \text{(numerical stability)}
```
```math
\alpha = \text{learning rate}
```

---

### 6.2 First Moment (Momentum) - Weights

For each layer l and weight matrix W:

```math
m_t^{[l]} = \beta_1 \cdot m_{t-1}^{[l]} + (1 - \beta_1) \cdot \frac{\partial J}{\partial W^{[l]}}
```

**Physical interpretation:** Exponentially weighted average of past gradients (velocity).

**Implementation variables:**
- `m_now_weight1`, `m_now_weight2`, `m_now_weight3`
- Updated each iteration using past values `m_past_weight*`

---

### 6.3 Second Moment (RMSprop) - Weights

```math
v_t^{[l]} = \beta_2 \cdot v_{t-1}^{[l]} + (1 - \beta_2) \cdot \left(\frac{\partial J}{\partial W^{[l]}}\right)^2
```

**Physical interpretation:** Exponentially weighted average of squared gradients (acceleration).

**Implementation variables:**
- `v_now_weight1`, `v_now_weight2`, `v_now_weight3`

---

### 6.4 Bias Correction

Since `m_t` and `v_t` are initialized at 0, they are biased toward zero in early iterations. Adam corrects this:

**Corrected first moment:**
```math
\hat{m}_t^{[l]} = \frac{m_t^{[l]}}{1 - \beta_1^t}
```

**Corrected second moment:**
```math
\hat{v}_t^{[l]} = \frac{v_t^{[l]}}{1 - \beta_2^t}
```

**Implementation:**
```python
invBETA1 = 1.0 / (1.0 - beta1)  # Approximation for large t
invBETA2 = 1.0 / (1.0 - beta2)
hm_now_weight = m_now_weight * invBETA1
hv_now_weight = v_now_weight * invBETA2
```

**Note:** The code uses a simplified approximation assuming large t, where `1-β₁^t ≈ 1-β₁`.

---

### 6.5 First Moment (Momentum) - Biases

```math
m_t^{b^{[l]}} = \beta_1 \cdot m_{t-1}^{b^{[l]}} + (1 - \beta_1) \cdot \frac{\partial J}{\partial b^{[l]}}
```

**Implementation variables:**
- `B_now_bias1`, `B_now_bias2`, `B_now_bias3`

---

### 6.6 Second Moment (RMSprop) - Biases

```math
v_t^{b^{[l]}} = \beta_2 \cdot v_{t-1}^{b^{[l]}} + (1 - \beta_2) \cdot \left(\frac{\partial J}{\partial b^{[l]}}\right)^2
```

**Implementation variables:**
- `vB_now_bias1`, `vB_now_bias2`, `vB_now_bias3`

---

## 7. Parameter Updates

### 7.1 Weight Update Rule

```math
W_t^{[l]} = W_{t-1}^{[l]} - \alpha \cdot \frac{\hat{m}_t^{[l]}}{\sqrt{\hat{v}_t^{[l]}} + \epsilon}
```

**Step-by-step:**
1. Compute correction term:
   ```math
   \text{correction} = \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   ```

2. Update weights:
   ```math
   W^{[l]} \leftarrow W^{[l]} - \text{correction}
   ```

**Implementation:**
```python
correction_weight = hm_now_weight * (learning_rate / (np.sqrt(hv_now_weight) + epsilon))
weights_layer = weights_layer - correction_weight
```

---

### 7.2 Bias Update Rule

```math
b_t^{[l]} = b_{t-1}^{[l]} - \alpha \cdot \frac{\hat{m}_t^{b^{[l]}}}{\sqrt{\hat{v}_t^{b^{[l]}}} + \epsilon}
```

**Implementation:**
```python
correction_bias = hB_now_bias * (learning_rate / (np.sqrt(hvB_now_bias) + epsilon))
biases_layer = biases_layer - correction_bias
```

---

### 7.3 Update Order

Parameters are updated in reverse order (output to input):
1. Layer 3: `W[3]`, `b[3]`
2. Layer 2: `W[2]`, `b[2]`
3. Layer 1: `W[1]`, `b[1]`

---

## 8. Performance Metrics

### 8.1 Categorical Accuracy

```math
\text{Accuracy} = \frac{1}{m}\sum_{i=1}^{m} \mathbb{1}[\arg\max(y_{\text{true}}^{(i)}) = \arg\max(y_{\text{pred}}^{(i)})]
```

Where:
- `𝟙[·]` is the indicator function (1 if true, 0 if false)
- `argmax` returns the index of the maximum value (predicted class)

**Implementation:**
```python
yTrue = [argmax(z) for z in y_true.T]
yPred = [argmax(z) for z in y_pred.T]
accuracy = sum(yPred == yTrue) / len(yPred)
```

---

### 8.2 Root Mean Squared Error (RMSE)

```math
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2}
```

Where N is the total number of elements (samples × classes).

---

### 8.3 R² Score (Coefficient of Determination)

```math
R^2 = 1 - \frac{\sum_i (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2}{\sum_i (y_{\text{true}}^{(i)} - \bar{y}_{\text{true}})^2}
```

Where:
- `ȳ_true` is the mean of true values
- Values close to 1 indicate good fit
- Values close to 0 indicate poor fit

---

## Summary of Complete Training Loop

For each iteration t = 1, 2, ..., T:

1. **Forward Pass**: Compute `Z[l]` and `A[l]` for l = 1, 2, 3
2. **Compute Loss**: `Loss = MSE(A[3], Y)`
3. **Backward Pass**: Compute gradients `∂J/∂W[l]` and `∂J/∂b[l]` for l = 3, 2, 1
4. **Adam Update**:
   - Update first moments: `m_t = β₁m_{t-1} + (1-β₁)∇J`
   - Update second moments: `v_t = β₂v_{t-1} + (1-β₂)(∇J)²`
   - Bias correction: `m̂_t = m_t/(1-β₁^t)`, `v̂_t = v_t/(1-β₂^t)`
   - Parameter update: `θ_t = θ_{t-1} - α·m̂_t/(√v̂_t + ε)`
5. **Metrics**: Calculate accuracy, RMSE, R², and cost

---

## Mathematical Advantages of Adam

1. **Adaptive Learning Rates**: Each parameter has its own effective learning rate based on gradient history
2. **Momentum**: Helps escape local minima and accelerates convergence
3. **Bias Correction**: Ensures proper updates even in early training stages
4. **Robust to Hyperparameters**: Default values (β₁=0.9, β₂=0.99) work well for most problems
5. **Efficient**: Computationally similar to standard SGD with minimal memory overhead

---

**Reference:**
Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. *arXiv preprint arXiv:1412.6980*.
