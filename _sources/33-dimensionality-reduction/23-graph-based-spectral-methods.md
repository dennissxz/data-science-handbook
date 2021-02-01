# Graph-based Spectral Methods

Taks inspiration from MDS, extend to a variety of distance/similarity measures.

- Construct a graph where each node corresponds to a data point and edges corresponds to neighbor relationship
- Construct a matrix summarization of the graph


# Isomap

A direct extension of MDS where Euclidean distances are replaced by estimated ??geodesic distance along the manifold.



## Learning

1. Construct a graph where node $i$ corresponds to input example $\boldsymbol{x}_i$, and nodes $i$ and $j$ are connected by an edge if $\boldsymbol{x}_i, \boldsymbol{x}_j$ are $N$-nearest neighbors (by some definition).

1. For each $i,j$ compute pairwise distance $\Delta_{i j}^{2}$ along the shortest path in the graph (e.g., using Dijkstra’s shortest path algorithm)

1. Perform MDS for some dimensionality $k$, taking the distances $\Delta_{i j}^{2}$ as input in place of Euclidean distances.

The output is the k-dimensional projections of all of the data points zi, which satisfy

$$
\left\|\mathbf{z}_{i}-\mathbf{z}_{j}\right\|^{2} \approx \Delta_{i j}^{2}
$$

:::{figure}
<img src="../imgs/gb-isomap-d3k2.png" width = "80%" alt=""/>

Iso map with $d=3,k=2$ [Livescu 2021]
:::

## Pros Cons

Pros

- As the data set size increases, Isomap is guaranteed to converge to the correct manifold that the data was drawn from


Cons

- Can be sensitive to the neighborhood **size** used in graph construction, or equivalently to the noise in the data.

:::{figure}
<img src="../imgs/gb-isomap-failure.png" width = "70%" alt=""/>

Isomap fails when there are noises [Livescu 2021]
:::

- Can't handle holes in the manifold. Geodesic distance computation can break)

:::{figure}
<img src="../imgs/gb-isomap-holes.png" width = "80%" alt=""/>

Isomap fails when there are holds [Livescu 2021]
:::

This is because the two points of the color sit in opposite to the hole have large distance on the manifold.

# Laplacian Eigenmaps

# Locally Linear Embedding

# Maximum Variance Unfolding
