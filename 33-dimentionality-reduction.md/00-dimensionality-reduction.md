# Dimensionality Reduction


Aka representation learning.

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$
- goals
  - reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the **essence** of the data, i.e. find a projection $\mathcal{X} \rightarrow \mathcal{Z}$ such that $\mathrm{dim}  \left( \mathcal{Z} \right)  \ll \mathrm{dim} \left( \mathcal{X} \right)$
  - for compression
  - for visualization
  - for downstream learning tasks

- motivation
  - although the data may appear high dimensional, there may only be a small number of degrees of variability, corresponding to latent factors
  - low dimensional representations are useful for enabling fast nearest neighbor searches and 2-d projections are useful for visualization

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



- application
  - latent semantic analysis (a variant of PCA) for document retrieval in natural language processing
  - ICA (a variant of PCA) to separate signals into different sources in signal processing
  - discovering units of speech in a language

If I take away some information, am I able to reconstruct it?


## Taxonomy

### linear Dimensionality Reduction

Linear subspace is defined by a $d\times k$ matrix $W$ containing k.

$$
\hat{\boldsymbol{x}_i} = W \boldsymbol{z}_i
$$

Reconstruction error

$$
\boldsymbol{x}_i - \hat{\boldsymbol{x}}_i
$$
