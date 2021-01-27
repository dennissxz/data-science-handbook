# Neural Networks

## vs Machine Learning: Linear vs Hierarchical

Most machine learning relies almost entirely on linear predictors, where predictors can be non-linear features of the data.

Feature transform $\boldsymbol{\phi} : \mathcal{X} \rightarrow \mathbb{R}^{d}$

$$
f_{y}(\mathbf{x} ; \mathbf{w}, \mathbf{b})=\mathbf{w}_{y} \cdot \boldsymbol{\phi} (\mathbf{x})+b_{y}
$$

Shallow learning: hand-crafted, non-hierarchical $\boldsymbol{\phi}$.

Basic example:
- polynomial regression: $\phi_{j}(x)=x^{j}, j=0, \ldots, d$
- Kernel SVM: employing kernel $K$ corresponds to (some) feature space such that $K\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\phi\left(\mathbf{x}_{i}\right) \cdot \phi\left(\mathbf{x}_{j}\right)$. SVM is just a linear classifier in that space.

In **deep** learning, a predictor that uses a **hierarchy** of features of the input, typically (but not always) learned end-to-end jointly with the predictor.

$$
f_{y}(\mathbf{x})=F_{L}\left(F_{L-1}\left(F_{L-2}\left(\cdots F_{1}(\mathbf{x}) \cdots\right)\right)\right)
$$

A 2-layer neural network can be represented by

$$
f_{y}(\mathbf{x})=\sum_{j=1}^{m} w_{j, y}^{(2)} h\left(\sum_{i=1}^{d} w_{i, j}^{(1)} x_{i}+b_{j}^{(1)}\right)+b_{y}^{(2)}
$$

In matrix form,

$$
\mathbf{f}(\mathbf{x})=\mathbf{W}_{2} \cdot h\left(\mathbf{W}_{1} \cdot \mathbf{x}+\mathbf{b}_{1}\right)+\mathbf{b}_{2}
$$

where $h$ is applied elementwise; $\mathbf{x} \in \mathbb{R}^{d}, \mathbf{W}_{1} \in \mathbb{R}^{m \times d}, \mathbf{W}_{2} \in \mathbb{R}^{C \times m}, \mathbf{b}_{2} \in \mathbb{R}^{C}, \mathbf{b}_{1} \in \mathbb{R}^{m}$.

Theorem (nn approximate)
: 2-layer net with linear output (sigmoid hidden units) can approximate any continuous function over compact domain to arbitrary accuracy (given enough hidden units!) [Cybenko 1998].

[img36]


## Pros of Deep

Example: parity of n-bit numbers, with AND, OR, NOT, XOR gates

Trivial shallow architecture: express parity as DNF or CNF. They are shallow functions, and need exponential number of gates!

Deep architecture: a tree of XOR gates.
