# Graph-based Spectral Methods

PCA, CCA, MDS are linear dimensionality reduction methods. If out data lies on a nonlinear manifold (a topological space locally resembling Euclidean space), then we need non-linear dimensionality reduction methods. Many of them are extended from MDS.

Graph-based spectral methods takes inspiration from MDS, extends to a variety of distance/similarity measures.

In general, there are three steps:

- Construct a graph with nodes being the data points. Define a similarity measure between neighbors. Assign the similarity measure value to edge weights.
- Construct a $n \times n$ matrix summarization of the data set / graph, where the entries are pairwise weights (usually sparse).
- Perform operations on the wight matrix to find the data projections (e.g. spectral decomposition).

Examples: Isomap, maximum variance unfolding, locally linear embedding, Laplacian eigenmaps.

## Isomap

[Tenenbaum et al. 2000]

For a manifold in a high-dimensional space, it may NOT be reasonable to use Euclidean distance to measure the pairwise dissimilarity of points. Geodesic distance along the manifold may be more appropriate.

Isomap is a direct extension of MDS where Euclidean distances are replaced with *estimated geodesic distances* along the manifold, i.e. it tries to "unfold" a manifold and preserve the pairwise geodesic distances.


### Learning

Isomap algorithm:

1. Construct a graph where node $i$ corresponds to input example $\boldsymbol{x}_i$, and nodes $i$ and $j$ are connected by an edge if $\boldsymbol{x}_i, \boldsymbol{x}_j$ are nearest neighbors (by some definition) with edge weight being the Euclidean distance $d_{ij} = \left\| \boldsymbol{x}_i -\boldsymbol{x}_j  \right\|$.

1. For each $i,j$ in the graph, compute pairwise distance $\Delta_{i j}$ along the shortest path in the graph (e.g., using Dijkstra’s shortest path algorithm) as the estimated geodesic distance.

1. Perform MDS for some dimensionality $k$, taking the estimated geodesic distances $\Delta_{i j}$ as input in place of Euclidean distances.

The output is the $k$-dimensional projections of all of the data points $\boldsymbol{z}_i$ such that the Euclidean distances in the projected space approximate the geodesic distances on the manifold.

$$
\left\|\mathbf{z}_{i}-\mathbf{z}_{j}\right\|^{2} \approx \Delta_{i j}^{2}
$$

:::{figure} isomap-geodesic-distance
<img src="../imgs/gb-isomap-d3k2.png" width = "80%" alt=""/>

Isomap with $d=3,k=2$. The blue line is the real geodesic distance and the red line is estimated.  [Livescu 2021]
:::

### Pros Cons

**Pros**

- As the data set size increases, isomap is guaranteed to converge to the correct manifold that the data was drawn from, under certain conditions (e.g. no holes)


**Cons**

- Can be sensitive to the **neighborhood size** used in graph construction, or equivalently to the noise in the data.

    :::{figure} isomap-noise
    <img src="../imgs/gb-isomap-failure.png" width = "70%" alt=""/>

    Isomap fails when there are noises [Livescu 2021]
    :::

- Can't handle **holes** in the manifold. Geodesic distance computation can break. This is because the two points (even with the same color/label) sit in opposite to the hole have large geodesic distance on the manifold, which leads to large Euclidean distance in the projected space.

    :::{figure} gb-isomap-holes
    <img src="../imgs/gb-isomap-holes.png" width = "80%" alt=""/>

    Isomap fails when there are holes [Livescu 2021]
    :::


## Laplacian Eigenmaps

[Belkin & Niyogi 2003]

Unlike isomap where the edge weights are local Euclidean distances, Laplacian eigenmaps define edge weights in another way.

### Objective

1. Define a weight matrix $\boldsymbol{W}$ with pairwise edge weights as

    $$
    w_{i j}=\left\{\begin{array}{ll}
    \exp \left(-\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|^{2} / t\right), & \left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|<\epsilon \\
    0 & \text { otherwise }
    \end{array}\right.
    $$

    where

    - $\epsilon$ is a hyperparameter used to define nearest neighbors (can also use counts)
    - $t$ is a hyperparameter like temperature.

1. Then we define a diagonal matrix $\boldsymbol{D}$ with $d_{ii} = \sum_j w_{ij}$. This can be seen as the density around the node $i$.

1. Find centered and unit-covariance projections $\boldsymbol{z}_i$ that solve the total projected pairwise distances weighted by $w_{ij}$ and scaled by $d_{ii}d_{jj}$


    $$\begin{aligned}
    \min &\ \sum_{i j} \frac{w_{i j}|| \mathbf{z}_{i}-\mathbf{z}_{j}||^{2}}{\sqrt{d_{i i} d_{j j}}} \\
    \text{s.t.} &\ \boldsymbol{Z} \text{ is centered and has unit covariance} \\
    \end{aligned}$$


### Learning

The solution is given by the $k$ **bottom** eigenvectors of

$$
\boldsymbol{L} =\boldsymbol{I}  - \boldsymbol{D} ^{-\frac{1}{2}} \boldsymbol{W}  \boldsymbol{D} ^{-\frac{1}{2}}
$$

excluding the bottom (constant) eigenvector??. This is a symmetrized, normalized form of the graph Laplacian $\boldsymbol{D} - \boldsymbol{W}$.

:::{figure} gb-laplacian-eigenmap-Nt
<img src="../imgs/gb-laplacian-eigenmap-Nt.png" width = "50%" alt=""/>

Laplacian eigenmap with varing N-nearest-neighbors and temperature $t$ [Livescu 2021]

:::


## Locally Linear Embedding

Locally linear embedding learns a mapping in which each point can be expressed as a **linear function** of its nearest neighbors.

## Maximum Variance Unfolding

Maximum variance unfolding tries to maximize the variance of the data (like PCA) while respecting neighborhood relationships.
