# Clustering

Many data sets naturally consist of multiple natural subsets, e.g. multiple digits, object classes in images, topics, phonemes in speech, ...

Clustering aims to identifying these subsets and assigning data points to the correct one.

:::{admonition,warning} Warning
Almost no theoretical guarantees of clustering algorithms.
:::

If the ultimate goal of clustering is to represent data points by its cluster "center", then the methods are called "vector quantization".

## Vector Quantization

```{margin}
Quantization = discretization = compression
```

VQ is a clustering-based discrete representation learning method.

The feature vectors in many data sets are often a vector of real values per data sample, but for some applications, compact representation may be important. For instance, compression, coding, etc. Quantization code the real-valued $\boldsymbol{x}$ to codeword, and represent the vector $\boldsymbol{x}$ by the closest codeword. Of course, quantization incurs some loss of information.

The simplest VQ approach is $k$-means clustering.

:::{figure} clustering-vq
<img src="../imgs/clustering-vq.png" width = "80%" alt=""/>

VQ codebooks for 2 dimensional vectors [Livescue 2021]
:::

:::{figure} k-means-img-compression
<img src="../imgs/k-means-img-compression.png" width = "80%" alt=""/>

$k$-means for image compression [Livescue 2021]
:::

## Comparison

We will introduce several methods.

- **$k$-means clustering** is an iterative algorithm for clustering. It initializes $k$ cluster centers, assign each example to its **closest** center, and re-compute the center, until there is no changes in assignment.

- **Agglomerative methods** view each data point as a cluster, and then iteratively merge two **closest** clusters according to some distance measure, until some convergence criterion is achieved. They do not get to see all data points at once – might miss some important pattern,

- **Decision tree** can also be used in clustering. It is a top-down approach that start from one cluster and iteratively partition every cluster to two smaller clusters until come convergence criterion is achieved.

- **Spectral clustering** methods analyze the $n\times n$ similarity matrix of a data set and use spectral (eigenvector-based) methods to divide the graph into connected sub-graphs.

Most of the above clustering methods have some weakness:

- make a "hard" (discrete) decision about cluster membership for each data point, like SVM.

- can’t always do out-of-sample extension
  - K-means: Can map new points to cluster with nearest cluster mean
  - No obvious way to do this for hierarchical clustering, spectral clustering

- need assumptions that the data points are separable

As a result, rather than clustering, we can view such label-assignment problems are density estimation, which can improve the above weakness ("soft clustering"). The basic one is Gaussian mixture.

- **Gaussian mixtures** are mixture models that represent the density of the data as a mixture of component densities.