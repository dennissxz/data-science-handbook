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
\min _{\boldsymbol{\theta}} J\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}, P\left(\mathbf{x}_{1}\right), \ldots, P\left(\mathbf{x}_{n}\right)\right)
$$

- For linear dimensionality models, the projection $P(\boldsymbol{x}; \boldsymbol{\theta} )$ can be represented by a $d \times k$ matrix $\boldsymbol{W}$. The low dimensional representation $\boldsymbol{z}$ is computed by

    $$
    \boldsymbol{z} = \boldsymbol{W} ^\top \boldsymbol{x}  
    $$

    and the reconstruction of $\boldsymbol{x}$ is a weighted combination of the vectors in $\boldsymbol{W}$:

    $$
    \hat{\boldsymbol{x}} = \boldsymbol{W} \boldsymbol{z} = \boldsymbol{W} \boldsymbol{W} ^\top \boldsymbol{x}
    $$

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
- wiki webpage: page texts + hyper link structure
- articles: two languages
- speech: voice wave + video of mouth

For instance, in two-view representation learning, training data consists of samples of a $d$-dimensional random vector that has some natural split into two sub-vectors.

$$
\left[\begin{array}{l}
\mathbf{x} \\
\mathbf{y}
\end{array}\right], \mathbf{x} \in \mathbf{R}^{d_{x}}, \mathbf{y} \in \mathbf{R}^{d_{y}}, d_{x}+d_{y}=d
$$

The task is to learn useful features/subspaces from such two-view data. Typically involves learning representations of one view that are **predictive** of the other.

One example of linear multi-view representation learning is canonical correlation anlaysis.
