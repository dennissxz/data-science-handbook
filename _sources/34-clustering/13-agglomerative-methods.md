# Agglomerative Methods

Unlike $k$-means which specify $k$ at start, agglomerative methods view each data point as a cluster, and then iteratively merge two **closest** clusters according to some distance measure, until some convergence criterion.

## Distance Measure

Let $A$ and $B$ be two clusters and let $a,b$ be individual data points.

- **Single-linkage** uses the minimum distance

    $$\operatorname{dist}(A, B)=\min _{a \in A, b \in B} \operatorname{dist}(a, b)$$

    Single-linkage tends to yield **long**, “stringy” clusters.

- **Complete-linkage** uses the maximal distance

    $$\operatorname{dist}(A, B)=\max _{a \in A, b \in B} \operatorname{dist}(a, b)$$

    Complete-linkage tends to yield **compact**, “round” clusters.

- **Average-linkage** uses the average distance

    $$\operatorname{dist}(A, B)=\operatorname{mean}  _{a \in A, b \in B} \operatorname{dist}(a, b)$$

- **Wald's methods** uses the increment in total within-group difference if we merge them

    $$
    \begin{aligned}
    \operatorname{dist}(A, B) &=\sum_{i \in A \cup B}\left\|\mathbf{x}_{i}-\mathbf{\mu}_{A \cup B}\right\|^{2}-\sum_{i \in A}\left\|\mathbf{x}_{i}-\mathbf{\mu}_{A}\right\|^{2}-\sum_{i \in B}\left\|\mathbf{x}_{i}-\mathbf{\mu}_{B}\right\|^2 \\
    &=\frac{|A||B|}{|A|+|B|}|| \mathbf{\mu}_{A}-\mathbf{\mu}_{B}||^{2}
    \end{aligned}
    $$


    :::{admonition,note} Wald's method vs $k$-means

    - At each merge step, Ward’s method minimizes the same sum-of-squares criterion as k-means, but constrained by choices in previous iterations, so the total sum-of-squares for a given k is normally larger for Ward’s method than for k-means.

    - A common trick: Use Ward’s method to pick k, then run k-means starting from the Ward cluster

    :::

In the examples below, $k$-means tends to produce clusters with spherical shapes, and we can see how single-linkage is good or bad.

:::{figure} clustering-comparison-1
<img src="../imgs/clustering-comparison-1.png" width = "80%" alt=""/>

Comparison of clustering algorithms [Livescue 2021]
:::

:::{figure} clustering-comparison-2
<img src="../imgs/clustering-comparison-2.png" width = "80%" alt=""/>

Comparison of clustering algorithms [Livescue 2021]
:::


## Convergence Criterion

```{margin}
This is one advantage of hierarchical clustering over “flat” clustering like $k$-means
```

A good representation of clustering process is dendrogram. The $x$-axis represents items, and the $y$-axis is distance. It provides visual guidance to a good choice for the number of clusters

Stop merging when the merge cost (distance between merged clusters) would be much larger than in previous iterations (for some precise definition of “much larger”)

:::{figure} clustering-dendrogram
<img src="../imgs/clustering-dendrogram.png" width = "50%" alt=""/>

Representing clustering with dendrograms
:::

For instance, given a matrix of phonemes and electrodes, we are interested in discovering cohesive neural regions/firing patterns and relating them to clusters of stimuli.

:::{figure} clustering-neural-regions
<img src="../imgs/clustering-neural-regions.png" width = "50%" alt=""/>

Cohesive neural regions/firing patterns [Bouchard et al. 2013]
:::
