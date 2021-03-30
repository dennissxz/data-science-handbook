# Multidimensional Scaling

In this section we introduce (metric) multidimensional scaling (MDS).

Instead of a $n \times p$ data matrix, MDS looks at pairwise similarity measures of data points. For instance,

- a matrix of Euclidean distances between points,
- co-occurrence counts of words between documents
- an adjacency matrix of web pages

Non-metric MDS applies ranks, KL divergences, etc (rather than numerical similarities).

Many non-linear dimensionality reduction methods are extension to MDS. MDS is a clue to link linear and non-linear dimensionality reduction.

## Objective


Question: If we are given a distance matrix $\boldsymbol{D}\in \mathbb{R} ^{n \times n}$, then we can find a $2$-dimensional representation of it $\boldsymbol{Z} \in \mathbb{R} ^{n \times k}$, to visualize them in 2-D plane?

Essentially, MDS seeks a $k$-dimensional representation $\boldsymbol{z} \in \mathbb{R} ^k$ of a data set that preserves inner products (or similarity/distance) between pairs of data points $(\boldsymbol{x_i}, \boldsymbol{x}_j)$

$$
\min \sum_{i, j}\left(\boldsymbol{x}_{i} ^\top  \boldsymbol{x}_{j}-\boldsymbol{z}_{i} ^\top  \boldsymbol{z}_{j}\right)^{2}
$$

or equivalently

$$
\min \left\Vert \boldsymbol{X} \boldsymbol{X} ^\top  - \boldsymbol{Z} \boldsymbol{Z} ^\top    \right\Vert _F^2
$$

The input to MDS can be one of the following: data matrix $\boldsymbol{X}$, gram matrix $\boldsymbol{X} \boldsymbol{X} ^{\top}$, or distance matrix $\boldsymbol{D}$.

## Learning

```{margin}
Note the inner product matrix $\boldsymbol{G}_{n\times n} = \boldsymbol{X} \boldsymbol{X} ^\top$ is different from the data covariance matrix $\boldsymbol{S}_{d\times d} = \boldsymbol{X} ^\top \boldsymbol{X}$.
```

### Input is Data or Gram Matrix


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
\boldsymbol{Z}_{n \times k} = \left[  \boldsymbol{V}  \boldsymbol{\Lambda} ^{1/2} \right]_{[:k]} = \boldsymbol{V}_{[: k]} \boldsymbol{\Lambda}_{[: k,: k]}^{1 / 2}
$$

The reconstruction of $\boldsymbol{G}$ from $\boldsymbol{Z}$ is then

$$
\widehat{\boldsymbol{G}} = \boldsymbol{Z} \boldsymbol{Z} ^\top
$$

## Input is an Euclidean Distance Matrix

If the input is not a data matrix $\boldsymbol{X}$ but a Euclidean distances matrix $\boldsymbol{D}$

$$
d_{i j}=\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}=\left\|\boldsymbol{x}_{i}\right\|^{2}-2 \boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{j}+\left\|\boldsymbol{x}_{j}\right\|^{2}
$$

We can convert the Euclidean distance matrix $\boldsymbol{D}$ to the Gram matrix $\boldsymbol{G}$ for centered $\boldsymbol{X}$ by left- and right-multiplying by the centering matrix $\boldsymbol{C}  = \left(\boldsymbol{I}-\frac{1}{n} \boldsymbol{1} \boldsymbol{1}^{\top}\right)$,

$$
\boldsymbol{G} = - \frac{1}{2} \boldsymbol{C}  \boldsymbol{D}\boldsymbol{C} ^{\top}
$$

And then we can run MDS over $\boldsymbol{G}$ to obtain $k$-dimensional representation.

:::{admonition,dropdown,seealso} *Proof*

$$\begin{aligned}
d_{ij}
&= \left\| x_i \right\|  + \left\| x_j \right\|  - 2 \boldsymbol{x}_i ^{\top} \boldsymbol{x}_j  \\
\Rightarrow \boldsymbol{D} &= \boldsymbol{v} \boldsymbol{1} ^{\top} + \boldsymbol{1} \boldsymbol{v} ^{\top} - 2 \boldsymbol{X}  \boldsymbol{X} ^{\top}\text{ where }  \boldsymbol{v} = \left\| \boldsymbol{x}_i  \right\| ^2  \\
\Rightarrow \boldsymbol{C} \boldsymbol{D} \boldsymbol{C} &= -2 \boldsymbol{C} \boldsymbol{X}  \boldsymbol{X} ^{\top} \boldsymbol{C} \quad \because \boldsymbol{C} (\boldsymbol{v} \boldsymbol{1} ^{\top}) \boldsymbol{C} = 0\\
\Rightarrow -\frac{1}{2} \boldsymbol{C} \boldsymbol{D} \boldsymbol{C} &= (\boldsymbol{C} \boldsymbol{X} )(\boldsymbol{C} \boldsymbol{X}) ^{\top}  \\
&= \boldsymbol{G} \quad \text{where $\boldsymbol{C} \boldsymbol{X}$ is column-centered $\boldsymbol{X}$} \\
\end{aligned}$$

$\square$
:::


## Model Selection

noise vs $k$


## Relation to PCA

A difference is that, unlike PCA which gives a projection equation $\boldsymbol{z} = \boldsymbol{U} ^\top \boldsymbol{x}$, MDS only gives a projected result for the training set. It does not give us a way to project a new data point.

A connection is that, the two projected data matrices are exactly the **same**. Suppose the data matrix $\boldsymbol{X}$ is centered. Let $\boldsymbol{Z} _{PCA}$ be the $n\times k$ projected matrix by PCA and $\boldsymbol{Z} _{MDS}$ be that by MDS. Then it can be shown that

$$
\boldsymbol{Z} _{PCA} = \boldsymbol{Z} _{MDS}\\
$$

This also implies that, to obtain PCA projections, we can use the covariance matrix $\boldsymbol{S}$, or the Gram matrix $\boldsymbol{G}$, or the Euclidean distances matrix $\boldsymbol{F}$.

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

But note that $\boldsymbol{u} _j$ is not normalized, since $\left\| \boldsymbol{u} _j \right\|^2 = \boldsymbol{v} _j ^\top \boldsymbol{X} \boldsymbol{X} ^\top \boldsymbol{v} _j = \sigma^2 _j$. After normalization, we have,

$$
\boldsymbol{U} _{[:d]} = \boldsymbol{X} ^\top \boldsymbol{V} _{[:d]} \boldsymbol{D} ^ {-1/2}
$$

Substituting this relation to the $n\times d$ projected matrix by PCA gives

$$\begin{aligned}
\boldsymbol{Z} _{PCA}
&= \boldsymbol{X} \boldsymbol{U} _{[:d]} \boldsymbol{D} ^ {-1/2}\\
&= \boldsymbol{X} \boldsymbol{X} ^\top \boldsymbol{V}  _{[:d]} \boldsymbol{D} ^ {-1/2}\\
&= \boldsymbol{V} _{[:d]} \boldsymbol{D} \boldsymbol{V} ^\top _{[:d]} \boldsymbol{V}  _{[:d]} \boldsymbol{D} ^ {-1/2} \quad \because \text{EVD of } \boldsymbol{X} \boldsymbol{X} ^\top  \\
&= \boldsymbol{V} _{[:d]} \boldsymbol{D} \boldsymbol{D} ^ {-1/2} \quad \because \boldsymbol{V} \text{ is orthogonal} \\
&= \boldsymbol{V} _{[:d]}\boldsymbol{D} ^ {1/2} \\
&= \boldsymbol{Z} _{MDS} \\
\end{aligned}$$

:::
