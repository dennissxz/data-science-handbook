
# Introduction to Machine Learning

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






## Basic Concepts

### Parametric vs Non-parametric Models

- parametric models
  - definition: have a fixed number of parameters
  - pros: faster to use
  - cons: stronger assumptions

- non-parametric models
  - definition: the number of parameters grow with the amount of training data
  - pros: flexible
  - cons: computationally intractable for large datasets

### $K$-nearest Neighbors

- task: classification
- parametric: no
- prediction: $p(y=c\vert \boldsymbol{x}, \mathcal{D}, K) = \frac{1}{K} \sum_{j \in N_K (\boldsymbol{x}, \mathcal{D})} \mathbb{I}\left( y_j =c \right)$
- description
  - it simply look at the $K$ points in the training set that are nearest to the test input $\boldsymbol{x}_i$
  - it is an example of **memory-based learning** or **instance-based learning**
  - it requires a distance measure $d(\boldsymbol{x}_i, \boldsymbol{x}_j)$, such as Euclidean distance, to identity $N_K \left( \boldsymbol{x} \mathcal{D} \right)$, the $K$ nearest points to $\boldsymbol{x}$
- cons:
  - poor performance in high dimensional settings

### The Curse of Dimensionality

- parametric models for classification and regression
- curse of dimensionality
- non-parametric: K-nearest neighbors

###  Overfitting

###  Model Selection

### No Free Lunch Theorem
