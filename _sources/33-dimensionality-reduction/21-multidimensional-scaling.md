# Multidimensional Scaling

In this section we introduce (metric) multidimensional scaling (MDS).

Instead of a $n \times p$ data matrix, MDS looks at pairwise similarity measures of data points. For instance,

- a matrix of Euclidean distances between points,
- co-occurrence counts of words between documents
- an adjacency matrix of web pages

Non-metric MDS applies ranks, KL divergences, etc (rather than numerical similarities).


## Objective

MDS seeks a $k$-dimensional representation $\boldsymbol{z} \in \mathbb{R} ^k$ of a data set that preserves inner products between pairs of data points $(\boldsymbol{x_i}, \boldsymbol{x}_j)$

$$
\min \sum_{i, j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}-\boldsymbol{z}_{i} \cdot \boldsymbol{z}_{j}\right)^{2}
$$

or equivalently

$$
\min \left\Vert \boldsymbol{X} \boldsymbol{X} ^\top  - \boldsymbol{Z} \boldsymbol{Z} ^\top    \right\Vert _F^2
$$

## Learning

```{margin}
Note $\boldsymbol{G}_{n\times n} = \boldsymbol{X} \boldsymbol{X} ^\top  $ it is different from the data covariance matrix $\boldsymbol{S}_{d\times d} = \boldsymbol{X} ^\top \boldsymbol{X}$.
```

The solution can be obtained from the $n\times n$ Gram matrix of inner products

$$
\boldsymbol{G}=\boldsymbol{X} \boldsymbol{X} ^\top
$$

where

$$
\begin{equation}
g_{i j}=\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}
\end{equation}
$$

The projected data matrix is

$$
\boldsymbol{Z}_{n \times k} = \left[  \boldsymbol{V} \operatorname{diag}\left( \boldsymbol{\Lambda} ^{1/2}  \right) \right]_{[:k]}
$$

where $\boldsymbol{V}_{n \times 1}$ is the eigenvector matrix and $\boldsymbol{\Lambda}_{n \times n}$ is the eigenvalue matrix of $\boldsymbol{G}$.

## Special Cases

### Input is a Euclidean Distance Matrix

If the input is not a data matrix $\boldsymbol{X}$ but a Euclidean distances matrix $\boldsymbol{F}$

$$
\begin{equation}
f_{i j}=\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}=\left\|\boldsymbol{x}_{i}\right\|^{2}-2 \boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{j}+\left\|\boldsymbol{x}_{j}\right\|^{2}
\end{equation}
$$

If $\boldsymbol{x} _i$ are **centered** (zero-mean), we can convert the Euclidean distance matrix $\boldsymbol{F}$ to the Gram matrix $\boldsymbol{G}$ of inner product by left- and right-multiplying by the centering matrix $C = \left(\boldsymbol{I}-\frac{1}{n} \boldsymbol{1} \boldsymbol{1}^{\top}\right)$,

$$
\begin{equation}
\boldsymbol{G} = - \frac{1}{2} \boldsymbol{C}  \boldsymbol{F}\boldsymbol{C} ^{\top}
\end{equation}
$$

And then we can run MDS to $\boldsymbol{G}$.

In this case, for each eigenvector $\boldsymbol{u}_i$ of the data covariance matrix $\boldsymbol{S} = \frac{1}{n}  \boldsymbol{X} ^\top \boldsymbol{X}$, there is a corresponding eigenvector $\boldsymbol{v} _i$ of the Gram matrix $\boldsymbol{G} = \boldsymbol{X}  \boldsymbol{X} ^\top$ such that

$$
\boldsymbol{v} _i = \boldsymbol{X} ^\top \boldsymbol{u} _i
$$

which means that the PCA projections $\boldsymbol{z}_{i}^{PCA} = \boldsymbol{X} ^\top \boldsymbol{u} _i$ are the same as those of the MDS projections $\boldsymbol{v} _i$, ??lambda?? i.e., the first $k$ vectors $\boldsymbol{v} _i$ gives the projected  data in both PCA and MDS.

But unlike PCA, MDS only gives projections for the training set; it does not give us a way to project a new data point.

Many non-linear dimensionality reduction methods are extension to MDS. MDS is a clue to link linear and non-linear dimensionality reduction.
