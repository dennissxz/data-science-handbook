# Dimensionality Reduction

Aka representation learning.


## Motivation

- although the data may appear high dimensional, there may only be a small number of degrees of variability, corresponding to latent factors
- low dimensional representations are useful for enabling fast nearest neighbor searches and 2-d projections are useful for visualization

- for compression
- for visualization
- for downstream learning tasks


If I take away some information, am I able to reconstruct it?

## Model

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$
- goals
  - reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the **essence** of the data, i.e. find a projection $\mathcal{X} \rightarrow \mathcal{Z}$ such that $\mathrm{dim}  \left( \mathcal{Z} \right)  \ll \mathrm{dim} \left( \mathcal{X} \right)$



Seek a projection

$$
P(\mathbf{x} ; \boldsymbol{\theta}): \mathbb{R}^{d} \rightarrow \mathbb{R}^{k}, \quad k \ll d
$$

Fit $P$ by optimize some goodness of fit $J$

$$
\min _{\boldsymbol{\theta}} J\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}, P\left(\mathbf{x}_{1}\right), \ldots, P\left(\mathbf{x}_{N}\right)\right)
$$

Linear subspace is defined by a $m \ times p$ matrix $\boldsymbol{W}$.

Approximation of $\boldsymbol{x}$ as a weighted combination of the vectors in $\boldsymbol{W}$:

$$
\hat{\boldsymbol{x}} = \boldsymbol{W} \boldsymbol{y}
$$

where $\boldsymbol{y}$ can be seen as a low-dimensional representation of $\boldsymbol{x}$


### Multi-view



## Applications

- models
  - principal components analysis
  - autoencoders
  - find latent factors
  - word embeddings, contextual word embeddings

- matrix
  - minimize the reconstruction cost

    $$
    \min \sum_i J(\boldsymbol{x}_i, \boldsymbol{z}_i)
    $$

## Metrics

- qualitative evaluation
  - does the visualization looks better?
- evaluation on supervised downstream tasks
- intrinsic evaluation
  - cluster purity
  - degree of compression


### Linear Dimensionality Reduction

Linear subspace is defined by a $d\times k$ matrix $W$ containing $k$.

$$
\hat{\boldsymbol{x}_i} = W \boldsymbol{z}_i
$$

Reconstruction error

$$
\boldsymbol{x}_i - \hat{\boldsymbol{x}}_i
$$
