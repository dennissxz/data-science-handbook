# Multidimensional Scaling

In this section we introduce (metric) multidimensional scaling (MDS). It finds some representation of input data, which can be used for dimension reduction or visualization. In addition to a $n \times d$ data matrix $\boldsymbol{X}$, MDS can also take pairwise relations, aka **proximity** data, as input, such as

- pairwise dissimilarity measures of data points as input, denoted $\boldsymbol{D} \in \mathbb{R} ^{n \times n}$
  - a matrix of Euclidean distances between points
  - survey results of customers' perception of dissimilarity between products

  Note that the input dissimilarity matrix $\boldsymbol{D}$ may not be exactly euclidean distance, hence it does not enjoy all properties that a distance matrix has.

- pairwise similarity measures of data points as input, denoted $\boldsymbol{S} \in \mathbb{R} ^{n \times n}$

  - co-occurrence counts of words between documents
  - an adjacency matrix of web pages (MDS as a graph layout technique)
  - survey results of customers' perception of similarity between products

- Usually the diagonal entires of a dissimilarity matrix are $0$, and those of a similarity matrix are 1. A dissimilarity measure can be created based on the given similarity measure,
and vice versa.

In all, there are four types of MDS: {metric, non-metric} $\times$ {distance, classical}.

- **distance scaling**: fit dissimilarity by Euclidean distance $d_{ij} \approx \left\| \boldsymbol{z} _i - \boldsymbol{z} _j \right\|$
- **classical scaling**: transform dissimilarity to some form of 'similarity' and then fit by inner product  $s_{ij} \approx \langle \boldsymbol{z}_i , \boldsymbol{z} _j \rangle$. Note the identity $\left\| \boldsymbol{z}_i - \boldsymbol{z}_j  \right\| ^2 = \left\| \boldsymbol{z} _i \right\|^2 + \left\| z
_j \right\| ^2 - 2 \langle \boldsymbol{z}_i , \boldsymbol{z}_j  \rangle$.
- **metric scaling** uses actual numerical values of dissimilarities
- **non-metric scaling** applies ranks, KL divergences, etc. rather than numerical dissimilarities.

Here, we introduce metric classical scaling. For others, see this [paper](http://www.stat.yale.edu/~lc436/papers/JCGS-mds.pdf).

Many non-linear dimensionality reduction methods are extension to MDS. MDS is a clue to link linear and non-linear dimensionality reduction.

## Objective

- Given a distance matrix $\boldsymbol{D}\in \mathbb{R} ^{n \times n}$, can we find a $2$-dimensional representation of it $\boldsymbol{Z} \in \mathbb{R} ^{n \times k}$, to visualize them in 2-D plane?

  More generally, suppose $\boldsymbol{Z} \in \mathbb{R} ^{n \times k}$ are the underlying embeddings, MDS finds $\boldsymbol{Z}$ such that the distance is nearly preserved: $\left\| \boldsymbol{z} _i - \boldsymbol{z} _j \right\| \approx d_{ij}$. Hence the objective is to find embeddings $\boldsymbol{Z}$ that minimizes **stress**

  $$
  \operatorname{Stress}_{D}\left(\boldsymbol{z}_{1}, \ldots, \boldsymbol{z}_{N}\right)=\left(\sum_{i \neq j=1}^n\left(d_{ij}-\left\|\boldsymbol{z}_{i}-\boldsymbol{z}_{j}\right\|\right)^{2}\right)^{1 / 2}
  $$

- Find coordinates for the $n$ points in a low dimensional space such that the corresponding distances maintain the ordering in of pairwise distance in $\boldsymbol{D}$.

<!--
This can be solved by gradient descent. However, we can also transform $\boldsymbol{D}$ to a form $\boldsymbol{B}$ that is naturally fitted by inner product. The transformation satisfies $d_{ij} = b_{ii} - 2b_{ij} + b_{ij}$, thereby mimicking the corresponding identities for $\left\| \boldsymbol{x}_i - \boldsymbol{x}_j  \right\|$ and $\langle \boldsymbol{x} _i, \boldsymbol{x} _j \rangle$.


If the input is similarity matrix $\boldsymbol{S}$, for the conversion of similarities $s_{ij}$ to dissimilarities $d_{ij}$, one could in principle use any monotone decreasing transformation, but the following conversion is preferred.

$$
d_{ij} = s_{ii} - 2 s_{i, j} + s_{jj}
$$

It interprets the similarities as inner product data, and guarantee $d_{ii} = 0$. -->


- Given data matrix $\boldsymbol{X} \in \mathbb{R} ^{n \times d}$, as a dimensionality reduction method, MDS seeks a $k$-dimensional representation $\boldsymbol{Z} \in \mathbb{R} ^{n \times k}$ that preserves inner products (a similarity measure) between pairs of data points $(\boldsymbol{x_i}, \boldsymbol{x}_j)$

  $$
  \min \sum_{i, j}\left(\boldsymbol{x}_{i} ^\top  \boldsymbol{x}_{j}-\boldsymbol{z}_{i} ^\top  \boldsymbol{z}_{j}\right)^{2}
  $$

  or equivalently

  $$
  \min \left\Vert \boldsymbol{X} \boldsymbol{X} ^\top  - \boldsymbol{Z} \boldsymbol{Z} ^\top    \right\Vert _F^2
  $$

  The lower-dimensional embeddings can be then used for downstream tasks, including clustering, classification, etc. It worths mentioning that the embeddings are exactly the same as PCA's. See the last section for proof.


## Learning

Let $\boldsymbol{C}  = \boldsymbol{I}-\frac{1}{n} \boldsymbol{1} \boldsymbol{1}^{\top}$ be a centering matrix.

```{margin}
Note the inner product matrix $\boldsymbol{G}_{n\times n} = \boldsymbol{X} \boldsymbol{X} ^\top$ is different from the data covariance matrix $\boldsymbol{S}_{d\times d} = \boldsymbol{X} ^\top \boldsymbol{X}$.
```

- If the input matrix is data matrix $\boldsymbol{X}$, the solution can be obtained from the $n\times n$ Gram matrix of inner products of centered data $\boldsymbol{C} \boldsymbol{X}$

  $$
  \boldsymbol{G} := (\boldsymbol{C} \boldsymbol{X} ) (\boldsymbol{C} \boldsymbol{X}) ^\top
  $$

  where $g_{i j}=\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}$. Suppose the spectral decomposition of $\boldsymbol{G}$ is $\boldsymbol{G} = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V} ^\top$, then the representation is given by

  $$
  \boldsymbol{Z}_{n \times k} = \left[  \boldsymbol{V}  \boldsymbol{\Lambda} ^{1/2} \right]_{[:k]} = \boldsymbol{V}_{[: k]} \boldsymbol{\Lambda}_{[: k,: k]}^{1 / 2}
  $$

  For derivation see [here](http://www.math.uwaterloo.ca/~aghodsib/courses/f10stat946/notes/lec10-11.pdf). The reconstruction of $\boldsymbol{G}$ from $\boldsymbol{Z}$ is then $\widehat{\boldsymbol{G}} = \boldsymbol{Z} \boldsymbol{Z} ^\top$.


- If the input is a similarity matrix $\boldsymbol{S}$, we can run the above algorithm by treating $\boldsymbol{S}$ as $\boldsymbol{G}$. Some other methods first convert a similarity measure to a distance measure and then analyze it. There are many ways for such conversion, e.g. $d = 10 \sqrt{2 (1-s)}$, which guarantees that the obtained dissimilarity order is exactly the inversion of the similarly order.

- If the input is an **Euclidean** distance matrix $\boldsymbol{D}$, suppose the true data is $\boldsymbol{X}$, then by definition we have

  $$
  d_{i j}=\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}=\left\|\boldsymbol{x}_{i}\right\|^{2}-2 \boldsymbol{x}_{i}^{\top} \boldsymbol{x}_{j}+\left\|\boldsymbol{x}_{j}\right\|^{2}
  $$

  We can convert the Euclidean distance matrix $\boldsymbol{D}$ to the Gram matrix $\boldsymbol{G}$ for centered $\boldsymbol{X}$ by left- and right-multiplying $\boldsymbol{C}$

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

- Motivated by this, if the input matrix is dissimilarity (not necessarily Euclidean distance) matrix $\boldsymbol{D}$, we can run the above algorithm over $\boldsymbol{B} = - \frac{1}{2} \boldsymbol{C} \boldsymbol{D} \boldsymbol{C}$, where $\boldsymbol{B}$ approximates $\boldsymbol{G}$.


### Tuning

How to choose $k$?

$k \le N-1$??, related to the rank of the distance matrix.

### Goodness of Fit

Scale stress to 1.

Stress

SStress

Stress plot

common practice: below 10%. May becomes higher if $k$ goes large.

fitted vs recovered distance plot.

## Relation to PCA

We compare PCA and MDS in terms of finding representation $\boldsymbol{Z}$ given $\boldsymbol{X}$,

- Difference: unlike PCA which gives a projection equation $\boldsymbol{z} = \boldsymbol{U} ^\top \boldsymbol{x}$, MDS only gives a projected result for the training set. It does not give us a way to project a new data point.

- Connections

  - the two representation are exactly the **same**. Suppose the data matrix $\boldsymbol{X}$ is centered. Let $\boldsymbol{Z} _{PCA}$ be the $n\times k$ projected matrix by PCA and $\boldsymbol{Z} _{MDS}$ be that by MDS. Then it can be shown that

    $$
    \boldsymbol{Z} _{PCA} = \boldsymbol{Z} _{MDS}\\
    $$

    This also implies that, to obtain PCA projections, we can use the covariance matrix, the Gram matrix $\boldsymbol{G}$, or the Euclidean distances matrix $\boldsymbol{F}$.

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

  - PCA finds basis $\boldsymbol{u} \in \mathbb{R} ^n$ (principle directions) for spanning $\boldsymbol{X}$, and MDS finds the coordinates $\boldsymbol{z} \in \mathbb{R} ^d$ of the embeddings associated with the PCA basis.

    $$\begin{aligned}
    \boldsymbol{X} ^{\top}  
    &= \boldsymbol{U} (\boldsymbol{V} \boldsymbol{\Sigma} ) ^{\top} \\
    &= \boldsymbol{U} (\boldsymbol{V}_{[:d]} \boldsymbol{\Sigma}_{[:d]} ) ^{\top} \\
    &= \boldsymbol{U} \boldsymbol{Z} _{MDS} ^{\top} \\
    [\boldsymbol{x} _i \ \ldots \ \boldsymbol{x} _n]&= [\boldsymbol{u}_1 \ \ldots \ \boldsymbol{u} _d ] [\boldsymbol{z} _1 \ \ldots \ \boldsymbol{z} _n] \\
    \boldsymbol{x} _i&= \sum_{j=1}^d
    z^{MDS}_{ij} \boldsymbol{u} _j\\
    \end{aligned}$$


Reference: Davis-kahan theorem
