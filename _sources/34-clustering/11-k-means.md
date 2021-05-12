# $k$-means clustering

Originated by MacQueen (1967), $k$-means clustering is an iterative algorithm for clustering. It initialize $k$ cluster centers, assign each example to its closest center, and re-compute the center, until there is no changes in assignment.


## Objective

For a given $k$, the objective is to form clusters $C_1, \ldots, C_k$ that minimize the within cluster sum of squares (aka total distortion)

$$
D=\sum_{c=1}^{k} \sum_{\boldsymbol{x} \in \text { cluster } c}\left\|\boldsymbol{x}- \boldsymbol{\mu}_{c}\right\|^{2}
$$

where $\boldsymbol{\mu}$ is the mean or centroid of objects in cluster $C_i$.

$k$-means is NP hard.

## Algorithm

A standard algorithm for $k$-means is Lloyd-Forgy algorithm, which is an heuristic method.

---
**Lloyd-Forgy algorithm for $k$-means clustering**

---

- Initialize $k$ cluster centers $\boldsymbol{\mu} _1, \ldots, \boldsymbol{\mu} _k$ at random locations.

- While True

  - Assignment step: assign each example $\boldsymbol{x} _i$ to the closest mean

      $$
      y_{i}=\operatorname{argmin}_{c}\left\|\boldsymbol{x}_{i}- \boldsymbol{\mu} _{c}\right\|
      $$

  - Update step: re-estimate each mean based on examples assigned to it

      $$
      \mu_{c}=\frac{1}{n_{c}} \sum_{y_{i}=c} \boldsymbol{x}_{i}
      $$

      where $n_{c}= \left\vert \left\{\boldsymbol{x}_{i}: y_{i}=c\right\} \right\vert$

  - repeat until there are no changes in assignment

---



With each iteration step of the K-means algorithm, the within-cluster variations (or centered sums of squares) decrease and the algorithm converges.

The standard algorithm often converges to a local minimum, rather than global minimum. The results are affected by initialization.

:::{figure} k-means-example
<img src="../imgs/k-means-example.png" width = "70%" alt=""/>

Iterations in $k$-means example [Livescu 2021]
:::


## Pros Cons

Pros

- Work well for clusters with spherical shape and similar size

Cons

- Affected by random initialization.
  - heuristic remedy: try several initializations, keep the result with lowest total distortion $D$

- Work bad for clusters with non-ideal attributes

:::{figure} k-means-bad
<img src="../imgs/k-means-bad.png" width = "50%" alt=""/>

$k$-means fail for clusters with non-ideal shapes [Livescue 2021]
:::

## Relation to

Geometrically, $k$-means method is closely related to the Voronoi tessellation partition of the data space with respect to cluster centers

:::{figure} kmeans-voronoi
<img src="../imgs/kmeans-voronoi.png" width = "50%" alt=""/>

Illustration of Voronoi tessellation [DJ Srolovitz]
:::

Various modifications of $k$-means such as spherical $k$-means and $k$-medoids have been proposed to allow using other distance measures.
