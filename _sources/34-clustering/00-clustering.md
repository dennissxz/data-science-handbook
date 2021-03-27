(clustering)=
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

## Flat vs Hierarchical

In general, we have two types of clustering algorithms.

- **flat**: specify the number of clusters at the beginnings, and the output are the assignment of data points to clusters.

- **hierarchical**: do not specify the number of cluster at the beginnings.
  - bottom-up (agglomerative): view each single data point as a cluster, iteratively merge two smaller clusters to a larger cluster by some rule.
  - top-down (divisive): start from a single cluster consisting all data points, iteratively split a large clusters into two smaller clusters by some rule.

We will introduce several methods.

- **$k$-means clustering** is an iterative algorithm for flat clustering. It initializes $k$ cluster centers, assign each example to its **closest** center, and re-compute the center, until there is no changes in assignment.

- **Agglomerative methods** are bottom-up hierarchical clustering methods. The methods view each data point as a cluster, and then iteratively merge two **closest** clusters according to some distance measure, until some convergence criterion is achieved. They do not get to see all data points at once – might miss some important pattern,

- **Decision tree** is a top-down hierarchical clustering method, which can also be used in classification. It starts from one cluster and iteratively partition every cluster to two smaller clusters until come convergence criterion is achieved. See the [decision tree](decision-tree) section under classification for details.

- **Spectral clustering** is also a top-down approach. It analyzes the $n\times n$ similarity matrix of a data set and use spectral (eigenvector-based) methods to divide the graph into connected sub-graphs.

Most of the above clustering methods have some weakness:

- make a "hard" (discrete) decision about cluster membership for each data point, rather than probability, like SVM.

- can’t always do out-of-sample extension
  - $k$-means: can map new points to cluster with nearest cluster mean
  - No obvious way to do this for hierarchical clustering, spectral clustering, etc

- need assumptions that the data points are separable

As a result, rather than clustering, we can view such label-assignment problems are density estimation, which can improve the above weakness ("soft clustering"). The basic one is Gaussian mixture.

- **Gaussian mixtures** are mixture models that represent the density of the data as a mixture of component densities.




## Clustering for Classification

Clustering can be used for classification in the sense of label propagation. See [cluster-then-label](cluster-then-label).

A deep version: deep spectral clustering [Hershey et al. 2016]
