# Principal Component Analysis

Proposed by Pearson in 1901 and further deveoped by Hotelling in 1993.

## Background

### Problem

### Goal

minimize the sum of residuals subject to orthogonality of projection directions.


$$
\begin{aligned}
\mathbf{W}^{*} &=\underset{\mathbf{W}}{\operatorname{argmin}} \sum_{i=1}^{N}\left\|\mathbf{x}_{i}-\hat{\mathbf{x}}_{i}\right\|^{2} \\
\text { s.t. } \mathbf{W}^{T} \mathbf{W} &=I
\end{aligned}
$$

### Idea

### Used when


### Alternative Formulation

$$
\begin{aligned}
\mathbf{W}^{*} &=\underset{\mathbf{W}}{\operatorname{argmax}} \operatorname{tr}\left(\mathbf{W}^{T} \mathbf{X} \mathbf{X}^{T} \mathbf{W}\right) \\
\text { s.t. } \mathbf{W}^{T} \mathbf{W} &=I
\end{aligned}
$$

which leads to the same solutions as the minimum-residual formulation

*Proof of equivalence*

## Cons

PCA can be fooled by the scale of the input dimensions. pg.29

So it is common practice to standardize the data to unit variance.


- reconstruction
  - $\hat{\mathbf{x}}=\mu_{\mathbf{x}}+\mathbf{W} \mathbf{z}$


## Choosing the dimensionality of $k$


## Relationship with SVD

## PCA and Gaussians

PCA only looks at variance in the data
can be described by Gaussians

Probabilistic PCA
