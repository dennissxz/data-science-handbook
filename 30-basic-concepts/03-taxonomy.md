# Taxonomy

There are several kinds of machine learning tasks. For each learning task, we introduce
- input
- goals/output
- metrics

- The boundary between supervised and unsupervised learning is fuzzy. "differently supervised".
- supervised learning usually doesn't optimize the loss we really care about.


## Supervised Learning

- input
  - $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$
  - $\boldsymbol{x_i}$ is called features, attributes or covariates
  - $y_i$ is a categorical or real-valued variable

- goals
  - given a new $\boldsymbol{x_i}$, predict $y_i$


- model selection


### Classification
- input
  - $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$, where $y_i$ is categorical

- goals
  - given a new $\boldsymbol{x_i}$, predict the probabilities in each category $p(y_i=c\vert \boldsymbol{x}_i, \mathcal{D})$
  - MAP estimate: $\hat y_i = \underset{c}{\mathrm{argmax}} \, p(y_i=c\vert \boldsymbol{x}_i, \mathcal{D})$

- metrics
  - AUC
  - F1 score
  - see [confusion-matrix](../13-statistics/33-confusion-matrix.md)

### Regression
- input
  - $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$ where $y_i\in \mathbb{R}$

- output
  - predicted value $\hat y_i$

## Unsupervised Learning


- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$, no labels

- goals
  - find interesting pattern in the data, aka knowledge discovery

- metrics
  - qualitative evaluation
    - does the visualization looks better?
  - evaluation on supervised downstream tasks
  - intrinsic evaluation
    - cluster purity
    - degree of compression




### Clustering

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$, no labels

- goals
  - estimate the distribution over the number of clusters $p(C \vert \mathcal{D})$
  - determine which cluster each point belongs to, i.e. $z_i^* = \underset{c}{\mathrm{argmax}} \, p(z_i=c\vert \boldsymbol{x}_i \mathcal{D})$

- models
  - k-means
  - spectral clustering

- application
  - cluster users into groups



### Dimensionality Reduction

Aka representation learning.

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$
- goals
  - reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the **essence** of the data, i.e. find a projection $\mathcal{X} \rightarrow \mathcal{Z}$ such that $\mathrm{dim}  \left( \mathcal{Z} \right)  < \mathrm{dim} \left( \mathcal{X} \right)$
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

- application
  - latent semantic analysis (a variant of PCA) for document retrieval in natural language processing
  - ICA (a variant of PCA) to separate signals into different sources in signal processing
  - discovering units of speech in a language



### Discovering Graph Structure

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$

- goals
  - discover which variables are most correlated with which others
  - get better joint probability density estimators
  - estimate the most probable graph $\hat G = \underset{}{\mathrm{argmax}} \, p(G\vert \mathcal{D})$

- application
  - portfolio management
  - predicting traffic flow
  - discovering topic hierarchies in Wikipedia

### Matrix Completion

Aka imputation.

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$ with missing entries

- goals
  - infer plausible values for the missing entries

- application
  - image inpainting, e.g. impute the pixels hidden behind the occlusion
  - collaborative filtering
    - row: user
    - column: item
    - entry: rating
  - market basket analysis
    - row: transaction
    - column: item
    - entry: binary, $x_{ij}=1$ if item $j$ was purchased on transaction $i$
    - like collaborative filtering,
          the matrix is sparse. but not considered as missing

### Generation / Density Estimation

- goal
  - estimate density
  - generate new data

- application
  - generate images
  - anomaly detection


## Semi-supervised Learning

- input
  - Lots of unlabeled data $\boldsymbol{X}_U$ and a little labeled data $\boldsymbol{X}_L$

- goals
  - learn representation from unlabeled data
  - use learned representation to improve performance of supervised learner on labeled data


## Reinforcement Learning

### Multi-agent Learning


### Parametric vs Non-parametric Models

- parametric models
  - definition: have a fixed number of parameters
  - pros: faster to use
  - cons: stronger assumptions

- non-parametric models
  - definition: the number of parameters grow with the amount of training data
  - pros: flexible
  - cons: computationally intractable for large datasets
