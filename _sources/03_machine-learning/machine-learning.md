
# Machine Learning

<!-- TOC -->

- [Problems and Metrics](#problems-and-metrics)
  - [Supervised](#supervised)
    - [Regression](#regression)
    - [Classification](#classification)
  - [Unsupervised](#unsupervised)
    - [Dimensionality Reduction](#dimensionality-reduction)
    - [Clustering](#clustering)
- [Techniques](#techniques)

<!-- /TOC -->





For each learning task, we introduce
- input
- goals/output
- metrics




## Supervised learning

- input
  - $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$
  - $\boldsymbol{x_i}$ is called features, attributes or covariates
  - $y_i$ is a categorical or real-valued variable

- goals
  - given a new $\boldsymbol{x_i}$, predict $y_i$

### Classification
- input
  - $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$, where $y_i$ is categorical

- goals
  - given a new $\boldsymbol{x_i}$, predict the probabilities in each category $p(y_i=c\vert \boldsymbol{x}_i, \mathcal{D})$
  - MAP estimate: $\hat y_i = \underset{c}{\mathrm{argmax}} \, p(y_i=c\vert \boldsymbol{x}_i, \mathcal{D})$

- metrics
  - AUC
  - F1 score
  - see [confusion-matrix](../01_statistics/confusion-matrix.md)

### Regression
- input
  - $\mathcal{D} = \left\{ \left(\boldsymbol{x} _i, y_i  \right)  \right\}_{i=1}^N$ where $y_i\in \mathbb{R}$

- output
  - predicted value $\hat y_i$

## Unsupervised learning


- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$, no labels

- goals
  - find interesting pattern in the data, aka knowledge discovery

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


### Dimensionality reduction

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$
- goals
  - reduce the dimensionality by projecting the data to a lower dimensional subspace which captures the **essence** of the data, i.e. find a projection $\mathcal{X} \rightarrow \mathcal{Z}$ such that $\mathrm{dim}  \left( \mathcal{Z} \right)  < \mathrm{dim} \left( \mathcal{X} \right)$

- motivation
  - although the data may appear high dimensional, there may only be a small number of degrees of variability, corresponding to latent factors
  - low dimensional representations are useful for enabling fast nearest neighbor searches and 2-d projections are useful for visualization

- models
  - principal components analysis
  - autoencoders
  - latent factors

- application
  - latent semantic analysis (a variant of PCA) for document retrieval in natural language processing
  - ICA (a variant of PCA) to separate signals into different sources in signal processing

### Discovering graph structure

- input
  - $\mathcal{D} = \left\{ \boldsymbol{x} _i  \right\}_{i=1}^N$

- goals
  - discover which variables are most correlated with which others
  - get better joint probability density estimators
  - estimate the most probable graph $\hat G = \underset{}{\mathrm{argmax}} \, p(G\vert \mathcal{D})$

- application
  - portfolio manamenge
  - predicting traffic flow

### Matrix completion

Aka imputation

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

## Basic concepts

### Parametric vs non-parametric models

- parametric models
  - definition: have a fixed number of parameters
  - pros: faster to use
  - cons: stronger assumptions
- non-parametric models
  - definition: the number of parameters grow with the amount of training data
  - pros: flexible
  - cons: computationally intractable for large datasets

### $K$-nearest neightbors

- task: classification
- parametric: no
- prediction: $p(y=c\vert \boldsymbol{x}, \mathcal{D}, K) = \frac{1}{K} \sum_{j \in N_K (\boldsymbol{x}, \mathcal{D})} \mathbb{I}\left( y_j =c \right)$
- description
  - it simply look at the $K$ points in the training set that are nearest to the test input $\boldsymbol{x}_i$
  - it is an example of **memory-based learning** or **instance-based learning**
  - it requires a distance measure $d(\boldsymbol{x}_i, \boldsymbol{x}_j)$, such as Euclidean distance, to identity $N_K \left( \boldsymbol{x} \mathcal{D} \right)$, the $K$ nearest points to $\boldsymbol{x}$
- cons:
  - poor performance in high dimensional settings

### The curse of dimentionality



- parametric models for classification and regression
- curse of dimensionality
- non-parametric: K-nearest neighbors

###  overfitting

###  model selection

### no free lunch theorem
