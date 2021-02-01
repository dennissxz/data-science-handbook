# Multidimensional Scaling

In this section we introduce (metric) multidimensional scaling (MDS).

Instead of a $n \times p$ data matrix, MDS looks at pairwise similarity measures of data points. For instance,

- a matrix of Euclidean distances between points,
- co-occurrence counts of words between documents
- an adjacency matrix of web pages

Non-metric MDS applies ranks, KL divergences, etc (rather than numerical similarities).

Many non-linear dimensionality reduction methods are extension to MDS. MDS is a clue to link linear and non-linear dimensionality reduction.

## Objective

MDS seeks a $k$-dimensional representation $\boldsymbol{z} \in \mathbb{R} ^k$ of a data set that preserves inner products (or similarity/distance) between pairs of data points $(\boldsymbol{x_i}, \boldsymbol{x}_j)$

$$
\min \sum_{i, j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}-\boldsymbol{z}_{i} \cdot \boldsymbol{z}_{j}\right)^{2}
$$

or equivalently

$$
\min \left\Vert \boldsymbol{X} \boldsymbol{X} ^\top  - \boldsymbol{Z} \boldsymbol{Z} ^\top    \right\Vert _F^2
$$

## Learning

```{margin}
Note the inner product matrix $\boldsymbol{G}_{n\times n} = \boldsymbol{X} \boldsymbol{X} ^\top$ is different from the data covariance matrix $\boldsymbol{S}_{d\times d} = \boldsymbol{X} ^\top \boldsymbol{X}$.
```

The solution can be obtained from the $n\times n$ Gram matrix of inner products

$$
\boldsymbol{G}=\boldsymbol{X} \boldsymbol{X} ^\top
$$

where

$$
g_{i j}=\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}
$$

Suppose the spectral decomposition of $\boldsymbol{G}$ is

$$
\boldsymbol{G} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V} ^\top  
$$

The projected data matrix is given by

$$
\boldsymbol{Z}_{n \times k} = \left[  \boldsymbol{V}  \boldsymbol{\Lambda} ^{1/2} \right]_{[:k]} = \boldsymbol{V}_{[: k]} \mathbf{\Lambda}_{[: k,: k]}^{1 / 2}
$$

The reconstruction of $\boldsymbol{G}$ from $\boldsymbol{Z}$ is then

$$
\widehat{\boldsymbol{G}} = \boldsymbol{Z} \boldsymbol{Z} ^\top
$$

## Relation to PCA

A difference is that, unlike PCA which gives a projection equation $\boldsymbol{z} = \boldsymbol{U} ^\top \boldsymbol{x}$, MDS only gives a projected result for the training set. It does not give us a way to project a new data point.

A connection is that, the two projected data matrices have a deterministic relation. Suppose the data matrix $\boldsymbol{X}$ is centered. Let $\boldsymbol{Z} _{PCA}$ be the $n\times d$ projected matrix by PCA and $\boldsymbol{Z} _{MDS}$ be that by MDS. Then it can be shown that

$$
\boldsymbol{Z} _{PCA} = \boldsymbol{Z} _{MDS} \boldsymbol{D} ^ {1/2}\\
$$

where $\boldsymbol{D}_{d\times d}$ is the eigenvalue matrix of $\boldsymbol{X} ^\top \boldsymbol{X}$.

That is, the $j$-th projected column vector by MDS has the same direction as that by PCA, but scaled by the square root of the $j$-th eigenvalue $\sqrt{\lambda_j}$ of matrix $\boldsymbol{X} ^\top \boldsymbol{X}$.

:::{admonition,dropdown,seealso} *Proof*

Consider the SVD of the **centered** data matrix

$$\boldsymbol{X}_{n\times d} = \boldsymbol{V} \boldsymbol{\Sigma} \boldsymbol{U} ^\top$$

- In PCA, the EVD of $n$ times the data covariance matrix is

    $$n \boldsymbol{S}_{d \times d} = \boldsymbol{X} ^\top \boldsymbol{X} = \boldsymbol{U} \boldsymbol{\Sigma} ^\top \boldsymbol{\Sigma} \boldsymbol{U} = \boldsymbol{U} \boldsymbol{D} \boldsymbol{U}$$

    where the diagonal entries in $\boldsymbol{D}$ are the squared singular values $\sigma^2 _j$ for $j=1,\ldots, d$.

    The $n\times k$ projected matrix $(k\le d)$ is

    $$
    \boldsymbol{Z}_{PCA} = \boldsymbol{X} \boldsymbol{U} _{[:k]}
    $$

- In MDS, the EVD of the inner product matrix $\boldsymbol{G}$ is

    $$
    \boldsymbol{G}_{n \times n} = \boldsymbol{X} \boldsymbol{X} ^\top = \boldsymbol{V} \boldsymbol{\Sigma} \boldsymbol{\Sigma} ^\top \boldsymbol{V} ^\top = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V} ^\top = \boldsymbol{V} _{[:d]} \boldsymbol{D} \boldsymbol{V} _{[:d]} ^\top
    $$

    where

    $$
    \boldsymbol{\Lambda} _{n \times n} = \left[\begin{array}{cc}
    \boldsymbol{D} _{d \times d} & \boldsymbol{0}  \\
    \boldsymbol{0}  & \boldsymbol{0}_{(n-d) \times (n-d)}
    \end{array}\right]
    $$

    The $n\times k$ projected matrix $(k\le d)$ is

    $$
    \boldsymbol{Z} _{MDS} = \boldsymbol{V}_{[:k]} \boldsymbol{\Lambda} ^{1/2}_{[:k, :k]} = \boldsymbol{V}_{[:k]} \boldsymbol{D} ^{1/2}_{[:k, :k]}
    $$

Let $\boldsymbol{v} _j$ be an eigenvector of $\boldsymbol{G}$ with eigenvalue $\sigma^2 _j$. Pre-multiplying $\boldsymbol{G} \boldsymbol{v}_j = \sigma^2 _j \boldsymbol{v} _j$ by $\boldsymbol{X} ^\top$ yields

$$\begin{aligned}
\boldsymbol{X} ^\top (\boldsymbol{X} \boldsymbol{X} ^\top) \boldsymbol{v} _j &= \boldsymbol{X} ^\top (\sigma^2 _j  \boldsymbol{v} _j) \\
\Rightarrow \qquad  n\boldsymbol{S} (\boldsymbol{X} ^\top \boldsymbol{v} _j) &= \sigma^2 _j (\boldsymbol{X} ^\top \boldsymbol{v} _j)
\end{aligned}$$

Hence, we found that $\boldsymbol{X} ^\top \boldsymbol{v} _j$ is an eigenvector of $n \boldsymbol{S}$ with eigenvalue $\sigma^2 _j$, denoted $\boldsymbol{u} _j$,

$$
\boldsymbol{u} _j = \boldsymbol{X} ^\top \boldsymbol{v} _j
$$

That is, there is a one-one correspondence between the first $d$ eigenvectors of $\boldsymbol{G}$ and $n \boldsymbol{S}$. More specifically, we have,

$$
\boldsymbol{U} _{[:d]} = \boldsymbol{X} ^\top \boldsymbol{V} _{[:d]}
$$

Substituting this relation to the $n\times d$ projected matrix by PCA gives


$$\begin{aligned}
\boldsymbol{Z} _{PCA}
&= \boldsymbol{X} \boldsymbol{U} _{[:d]}\\
&= \boldsymbol{X} \boldsymbol{X} ^\top \boldsymbol{V}  _{[:d]}\\
&= \boldsymbol{V} _{[:d]} \boldsymbol{D} \boldsymbol{V} ^\top _{[:d]} \boldsymbol{V}  _{[:d]}\\
&= \boldsymbol{V} _{[:d]} \boldsymbol{D} \\
&= \boldsymbol{Z} _{MDS} \boldsymbol{D} ^ {1/2}\\
\end{aligned}$$

:::


## Special Cases

### Input is a Euclidean Distance Matrix

If the input is not a data matrix $\boldsymbol{X}$ but a Euclidean distances matrix $\boldsymbol{F}$

$$
f_{i j}=\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}=\left\|\boldsymbol{x}_{i}\right\|^{2}-2 \boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{j}+\left\|\boldsymbol{x}_{j}\right\|^{2}
$$

If $\boldsymbol{x} _i$ are **centered** (zero-mean), we can convert the Euclidean distance matrix $\boldsymbol{F}$ to the Gram matrix $\boldsymbol{G}$ of inner product by left- and right-multiplying by the centering matrix $\boldsymbol{C}  = \left(\boldsymbol{I}-\frac{1}{n} \boldsymbol{1} \boldsymbol{1}^{\top}\right)$,

$$
\boldsymbol{G} = - \frac{1}{2} \boldsymbol{C}  \boldsymbol{F}\boldsymbol{C} ^{\top}
$$

And then we can run MDS over $\boldsymbol{G}$.
