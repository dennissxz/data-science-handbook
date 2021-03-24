# Dimensionality Reduction

Aka representation learning.


## Motivation

Although the data may appear high dimensional, there may only be a small number of degrees of variability, corresponding to latent factors. Low dimensional representations are useful for enabling fast nearest neighbor searches and 2-d projections are useful for visualization. It is also useful for  compression and downstream learning tasks.

The question to ponder before making dimensionality reduction: If I take away some information, am I able to reconstruct it?

## Objective

We want to reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the **essence** of the data, i.e. find a projection $\mathcal{X} \rightarrow \mathcal{Z}$ such that $\mathrm{dim}  \left( \mathcal{Z} \right)  \ll \mathrm{dim} \left( \mathcal{X} \right)$

More specifically, we seek a projection

$$
P(\boldsymbol{x} ; \boldsymbol{\theta}): \mathbb{R}^{d} \rightarrow \mathbb{R}^{k}, \quad k \ll d
$$

by optimizing some goodness of fit criteria $J$

$$
\min _{\boldsymbol{\theta}} J\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n}, P\left(\boldsymbol{x}_{1}\right), \ldots, P\left(\boldsymbol{x}_{n}\right)\right)
$$

- For linear dimensionality models, the projection $P(\boldsymbol{x}; \boldsymbol{\theta} )$ can be represented by a $d \times k$ matrix $\boldsymbol{W}$. The low dimensional representation $\boldsymbol{z}$ is computed by

    $$
    \boldsymbol{z} = \boldsymbol{W} ^\top \boldsymbol{x}  
    $$

    and the reconstruction of $\boldsymbol{x}$ is a weighted combination of the vectors in $\boldsymbol{W}$:

    $$
    \hat{\boldsymbol{x}} = \boldsymbol{W} \boldsymbol{z} = \boldsymbol{W} \boldsymbol{W} ^\top \boldsymbol{x}
    $$

    For the whole data matrix


    $$\begin{aligned}
    \boldsymbol{Z} _{n \times k} &= \boldsymbol{X} \boldsymbol{W}  \\
    \widehat{\boldsymbol{X}} &= \boldsymbol{X} \boldsymbol{W} \boldsymbol{W} ^\top  \\
    \end{aligned}$$

- For non-linear dimensionality reduction models, projection $P$ can be quite different.


## Metrics

The performance of a dimensionality reduction method can be measured by

- qualitative evaluation
  - does the visualization looks better?
- evaluation on supervised downstream tasks
- intrinsic evaluation
  - cluster purity
  - degree of compression

- evaluate it on labeled data

- reconstruction error
  - need to evaluate on hold out data
  - but some model do not work with out-of-sample data

- KL distance between kernel density estimators on training set and embedding data set.



## Multi-view Dimensionality Reduction

Unsupervised dimensionality reduction is very challenging. If we happen to have multiple data views, it can be easier.

View means the number of ways we describe a data object.

- picture: pixel value + captions
- picture: one with rotation noise, the other with pixel noise
- wiki webpage: page texts + hyper link structure
- articles: two languages
- speech: voice wave + video of mouth

For instance, in two-view representation learning, training data consists of samples of a $d$-dimensional random vector that has some natural split into two sub-vectors.

$$
\left[\begin{array}{l}
\boldsymbol{x} \\
\boldsymbol{y}
\end{array}\right], \boldsymbol{x} \in \boldsymbol{R}^{d_{x}}, \boldsymbol{y} \in \boldsymbol{R}^{d_{y}}, d_{x}+d_{y}=d
$$

The task is to learn useful features/subspaces from such two-view data. Typically it involves learning representations of one view that are **predictive** of the other. The idea is, if some transformation of the two views to make them "similar", then the transformation is meaningful, and the transformed vectors are representations.

One example of linear multi-view representation learning is canonical correlation analysis. Other variants include Kernel CCA, variational CCA, etc.

## Summary

### Linear Models

| Model | Input | Objective | Solution | $\qquad \qquad \text{Remarks}\qquad \qquad$|
| - | - | - | - | :- |
| PCA   | $\boldsymbol{X} ^\top \boldsymbol{X}$  | $\underset{\boldsymbol{w}}{\max} \boldsymbol{w}^{\top} \boldsymbol{\Sigma} \boldsymbol{w}$ <br> $\underset{\boldsymbol{W}}{\operatorname{min}} \sum_{i}^{n}\left\|\boldsymbol{x}_{i}-\boldsymbol{W} \boldsymbol{z}_{i}\right\|^{2}$ | $\boldsymbol{X} ^\top \boldsymbol{X} = \boldsymbol{U} \boldsymbol{D} \boldsymbol{U} ^\top$ <br> $\boldsymbol{z} = \boldsymbol{U}_{[:k]} ^\top \boldsymbol{x}$  | Standardize $\boldsymbol{X}$ before running  |
| Kernel PCA  | $\boldsymbol{X}$, kernel  | $\underset{\boldsymbol{\alpha}}{\max} \boldsymbol{\alpha}^{\top} \boldsymbol{K}^2 \boldsymbol{\alpha}$  |  $\boldsymbol{K} \boldsymbol{\alpha}_{j}=n \lambda_{j} \boldsymbol{\alpha}_{j}$ <br> $\boldsymbol{z} = \boldsymbol{A} ^\top _{[:k]} \boldsymbol{\Phi} \boldsymbol{\phi} (\boldsymbol{x}),  z_j = \sum_{i=1}^n \alpha_{ji} k(\boldsymbol{x}_i , \boldsymbol{x})$ | 1. Center $\boldsymbol{K}$ <br> 2. Linear kernel reduces to PCA |
| Probabilistic PCA  | $\boldsymbol{X} ^\top \boldsymbol{X}$   |  $\boldsymbol{z} \sim \mathcal{N}( \boldsymbol{0}, \boldsymbol{I})$ <br> $\boldsymbol{x} \mid \boldsymbol{z} \sim \mathcal{N}\left( \boldsymbol{W} \boldsymbol{z}+\boldsymbol{\mu} , \sigma^{2} \boldsymbol{I}\right)$ <br> ML $\boldsymbol{W} , \boldsymbol{\mu} , \sigma$  |  $\boldsymbol{W}_{M L} =\boldsymbol{U}_{d \times k}\left(\boldsymbol{\Lambda} _{k}-\sigma^{2} \boldsymbol{I}_k\right)^{1 / 2} \boldsymbol{R}_k$ <br> $\sigma_{M L}^{2} =\frac{1}{d-k} \sum_{j=k+1}^{d} \lambda_{j}$ <br>  $\widehat{\operatorname{E}}\left( \boldsymbol{z} \mid \boldsymbol{x}   \right) = \boldsymbol{M}  ^{-1} _{ML} \boldsymbol{W} ^\top _{ML}(\boldsymbol{x} - \bar{\boldsymbol{x}})$   | 1. $\boldsymbol{W} _{ML}$ not unique <br> 2. $\sigma^2 \rightarrow 0$ reduces to PCA  |
| CCA  | $\boldsymbol{X} , \boldsymbol{Y}$  | $\max _{\boldsymbol{\alpha}, \boldsymbol{\beta} } \operatorname{Corr}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{x} , \boldsymbol{\beta}^{\prime} \boldsymbol{y} \right)$  | $\boldsymbol{\Sigma}_{x x}^{-1} \boldsymbol{\Sigma}_{x y} \boldsymbol{\Sigma}_{y y}^{-1} \boldsymbol{\Sigma}_{y x} \boldsymbol{\alpha} = \rho^2 \boldsymbol{\alpha}$ <br> $\boldsymbol{\beta} \propto \boldsymbol{\Sigma}_{y y}^{-1} \boldsymbol{\Sigma}_{y x} \boldsymbol{\alpha}$ | $\rho$ is invariant of scaling of linear transformation of $\boldsymbol{X}$ or $\boldsymbol{Y}$  |
| Regularized CCA  | ''  |  '' | $\boldsymbol{\Sigma}_{x x} \leftarrow\boldsymbol{\Sigma}_{x x}+r_{x} I$ <br> $\boldsymbol{\Sigma}_{y y} \leftarrow \boldsymbol{\Sigma}_{y y}+r_{y} I$    | Add spherical noise $rI$ to the covariance matrices  |
| (Regularized) Kernel CCA  | $\boldsymbol{X}, \boldsymbol{Y}$, $r$, kernel  | $\max _{\boldsymbol{\alpha}, \boldsymbol{\beta}} \frac{\boldsymbol{\alpha} ^{\top} \boldsymbol{K} _{x} \boldsymbol{K} _{y} \boldsymbol{\beta} }{\sqrt{\boldsymbol{\alpha} ^{\top} \boldsymbol{K} _{x}^{2} \boldsymbol{\alpha} \cdot \boldsymbol{\beta} ^{\top} \boldsymbol{K} _{y}^{2} \boldsymbol{\beta} }}$  |  $\left(\boldsymbol{K}_{x}+r I\right)^{-1} \boldsymbol{K}_{y}\left(\boldsymbol{K}_{y}+r I\right)^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha}=\lambda^{2} \boldsymbol{\alpha}$ <br> $\boldsymbol{\beta} = \left(\boldsymbol{K}_{y}+r I\right)^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha} /\lambda$ |   To avoid trivial solution|
|MDS   |  $\boldsymbol{X} \boldsymbol{X} ^\top$ or $\boldsymbol{F}$  | $\min \sum_{i, j}\left(\boldsymbol{x}_{i} ^\top  \boldsymbol{x}_{j}-\boldsymbol{z}_{i} ^\top\boldsymbol{z}_{j}\right)^2$  | $\boldsymbol{G}  = \boldsymbol{X} \boldsymbol{X} ^\top = \boldsymbol{V} \boldsymbol{\Lambda} \boldsymbol{V}$ <br> $\boldsymbol{Z}=\boldsymbol{V}_{[: k]} \boldsymbol{\Lambda}_{[: k, k]}^{1 / 2}$  | 1. Retain inner product <br> 2. Gives exactly the embeddings as PCA: $\boldsymbol{Z}_{P C A}=\boldsymbol{Z}_{M D S}$ |

**Review**

| Model | Pros | Cons |
| - | -| - |
|PCA |   | 1. Sensitive to variable scale <br> 2. Principal direction may not be discriminative |
|Probabilistic PCA| 1. Fewer variance parameters <br> 2. Enable sampling  |  |
| Kernel PCA   |  1. Enable non-linear feature <br> 2. Enable out-of-sample projection <br> 3. Work well when the distribution fits the kernel | Computational issue (sol: subset, approximation $\boldsymbol{K} \approx \boldsymbol{F} ^\top \boldsymbol{F}$, random Fourier)  |
|CCA   |  Have discriminative power in some cases that PCA doesn't   | Easy to overfit tiny signals  |
| Regularized CCA   |  Avoid overfitting of CCA |   |
| (Regularized) Kernel CCA   |  1. Enable non-linear feature <br> 2. Enable out-of-sample projection |   |
| MDS   | Can be obtain from Euclidean distance matrix $\boldsymbol{F}$. For instance, from a survey "how do you compare the two items?" |   |

### Non-linear Models

In previous linear models, the lower-dimensional linear projection preserves distances
between **all** points. In non-linear models below, they only preserve local distance/neighborhood information along nonlinear manifold.

In general, there are three steps

1. Define some similarity/distance measure between data points $d(\boldsymbol{x}_i ,\boldsymbol{x}_j)$.
2. Induce a graph from the $n \times d$ data set $\boldsymbol{X}$
   - nodes are the data points
   - add edge $e(i,j)$ if the distance $d(\boldsymbol{x}_i,\boldsymbol{x}_j)$ satisfy certain criteria, e.g.  $d<\varepsilon$, $k$-NN, or mutual $k$-NN.
   - edge weights are the distance measures $w_{ij} = d_{ij}$
3. Perform spectral methods on some graph matrices, such as adjacency matrix, weights matrix, Laplacian matrix, etc, to obtain the embedding $\boldsymbol{Z} _{n \times d}$ that preserve some property in the original space.

| Model | Similarity/Distance Measure| Objective | Solution | $\qquad \qquad \text{Remarks}\qquad \qquad$|
| - | - | - | - | :- |
| Isomap   | geodesic distance along manifold  | retain geodesic distances $\left\|\boldsymbol{z}_{i}-\boldsymbol{z}_{j}\right\|^{2} \approx \Delta_{i j}^{2}$  | Run MDS with geodesic distance $\Delta$ matrix  |  Unfold manifold |
| Laplacian Eigenmaps  | Gaussian kernel similarity <br> $\exp \left(-\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2} / t\right)$ | total weights of a node $d_{ii}= \sum_j w_{ij}$ <br> $\min \sum_{i j} \frac{w_{i j} \Vert \boldsymbol{z}_{i}-\boldsymbol{z}_{j}\Vert^{2}}{\sqrt{d_{i i} d_{j j}}}$   | $k$ bottom eigenvectors of $\boldsymbol{L}=\boldsymbol{I}-\boldsymbol{D}^{-\frac{1}{2}} \boldsymbol{W} \boldsymbol{D}^{-\frac{1}{2}}$  | Retain weighted distance |  
| SNE   | $p_{j\mid i} = \frac{\exp \left( \left(-|| \boldsymbol{x}_{i}-\boldsymbol{x}_{j} \|^{2}\right) / 2 \sigma_{i}^{2} \right)}{\sum_{k \neq i} \exp \left( \left(-|| \boldsymbol{x}_{i}-\boldsymbol{x}_{k} \|^{2}\right) / 2 \sigma_{i}^{2} \right)}$ <br> $q_{j \mid i} = \frac{\exp \left(-|| \boldsymbol{z}_{i}-\boldsymbol{z}_{j} \|^{2}\right)}{\sum_{k \neq i} \exp \left(-|| \boldsymbol{z}_{i}-\boldsymbol{z}_{k}||^{2}\right)}$ | Retain neighborhood conditional probability for every point $i$ <br> $\min \sum_{i} \mathrm{KL}\left(P_{i}, Q_{i}\right) = \sum_{i,j} p_{j \mid i} \log \frac{p_{j \mid i}}{q_{j \mid i}}$  | gradient descent    | Computationally expensive  |
| $t$-SNE  | $p_{i j}=\frac{p_{j \mid i}+p_{i \mid j}}{2 n}$ <br> $q_{i j}=\frac{\left(1+\left\|\boldsymbol{z}_{i}-\boldsymbol{z}_{j}\right\|^{2}\right)^{-(df+1)/2}}{\sum_{k \neq l}\left(1+\left\|\boldsymbol{z}_{k}-\boldsymbol{z}_{l}\right\|^{2}\right)^{-(df+1)/2}}$  |  Retain neighborhood joint probability $\min \operatorname{KL}(P, Q)=\sum_{i,j} p_{i j} \log \frac{p_{i j}}{q_{i j}}$ |  '' | 1. Use joint probability <br> 2. Use $t$-distribution, usually $df=1$ |

**Review**

| Model | Pros | Cons |
| - | -| - |
| Isomap  | Recovers the manifold well | 1. Sensitive to neighborhood size / noise <br> 2. Can't handle holes in the manifold |
| Laplacian Eigenmaps   |   |   |
| SNE   |   | 1. Difficult to optimize <br> 2. Suffers from the “crowding problem” due to the asymmetric property as KL divergence  |
| $t$-SNE   |  Works especially well for data with clustering nature |  1. Perplexity hyperparameter is important <br> 2. Fail if no clustering nature  |
| Parametric $t$-SNE  | Enable out-of-sample projection  |   |
