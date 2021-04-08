# Graph-based Spectral Methods

PCA, CCA, MDS are linear dimensionality reduction methods. The lower-dimensional linear projection preserves distances between **all** points.

If our data lies on a nonlinear manifold (a topological space locally resembling Euclidean space), then we need non-linear dimensionality reduction methods. Many of them are extended from MDS, that extends to a variety of distance/similarity measures. They only preserve **local** distance/neighborhood information along nonlinear manifold.

In general, there are three steps:

1. Define some similarity/distance measure between data points $d(\boldsymbol{x}_i ,\boldsymbol{x}_j)$.
2. Induce a graph from the $n \times d$ data set $\boldsymbol{X}$
   - nodes are the data points
   - add edge $e(i,j)$ if the distance $d(\boldsymbol{x}_i,\boldsymbol{x}_j)$ satisfy certain criteria, e.g.  $d<\varepsilon$, $k$-NN, or mutual $k$-NN.
3. Perform spectral methods on some graph matrices, such as adjacency matrix, weights matrix, Laplacian matrix, etc, to obtain the embedding $\boldsymbol{Z} _{n \times d}$ that preserve some property in the original space.


Examples: Isomap, maximum variance unfolding, locally linear embedding, Laplacian eigenmaps.

The obtained embeddings can then be used as input for clustering, e.g. $k$-means. Essentially, we look for clustering on the manifold, rather than the original space, which makes more sense.

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
\left\|\boldsymbol{z}_{i}-\boldsymbol{z}_{j}\right\|^{2} \approx \Delta_{i j}^{2}
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

[[Belkin & Niyogi 2003](https://web.cse.ohio-state.edu/~belkin.8/papers/LEM_NC_03.pdf)]

Unlike isomap where the edge weights are local Euclidean distances, Laplacian eigenmaps define edge weights in another way.


### Learning

1. Construct an graph $G = (V, E)$ from data $\boldsymbol{X}$. Add edge $(i, j)$ if $\boldsymbol{x}_i$ and $\boldsymbol{x}_j$ are close, in the sense that
   - $\left\| \boldsymbol{x}_i  - \boldsymbol{x}_j  \right\| < \epsilon$, or
   - $n$-nearest-neighbors

1. Define edge weights as

    $$
    w_{i j}=\left\{\begin{array}{ll}
    \exp \left(-\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2} / t\right), & (i, j) \in E \\
    0 & \text { otherwise }
    \end{array}\right.
    $$

    where $t$ is a hyperparameter like temperature. As $t = \infty$, $w_{ij} = a_{ij}$.

1. Define a diagonal matrix $\boldsymbol{D}$ with $d_{ii} = \sum_j w_{ij}$. This can be seen as the density around the node $i$. The graph Laplacian is $\boldsymbol{L} = \boldsymbol{D} - \boldsymbol{W}$. The $k$-dimensional representation $\boldsymbol{Z}$ is given by the $k$ bottom eigenvectors (excluding the smallest one, which is $\boldsymbol{1}$) for the generalized eigenvector problem

    $$
    \boldsymbol{L} \boldsymbol{v} = \lambda \boldsymbol{D} \boldsymbol{v}
    $$

    If $G$ is not connected, run this step for each connected component in $G$.

:::{admonition,dropdown,seealso} *Derivation*

We want to preserve locality: if two data points $\boldsymbol{x}_i , \boldsymbol{x}_j$ are close, then their embeddings $\boldsymbol{z}_i , \boldsymbol{z}_j$ are also close. To ensure this, the loss function is formulated as

$$
\sum_{i,j=1}^n w_{ij} \left\| \boldsymbol{z}_i - \boldsymbol{z}_j  \right\|  ^2
$$

where $w_{ij}$ measures the closeness of $i$ and $j$ in $\boldsymbol{X}$. If $i$ and $j$ are close in $\boldsymbol{X}$, then $w_{ij}$ is large, which force $\left\| \boldsymbol{z}_i - \boldsymbol{z}_j  \right\|$ to be small, i.e. $i$ and $j$ are close in $\boldsymbol{Z}$. For $i$ and $j$ that are far away in $\boldsymbol{X}$, don't care. Recall that the objective is to maintain locality.

It can be shown that

$$\begin{aligned}
\sum_{i,j=1}^n w_{ij} \left\| \boldsymbol{z}_i - \boldsymbol{z}_j  \right\|  ^2
&= \sum_{i,j=1}^n w_{ij} \left\| \boldsymbol{Z}^{\top} (\boldsymbol{e} _i - \boldsymbol{e} _j) \right\|  ^2\\
&= 2 \operatorname{tr} \left( \boldsymbol{Z} \boldsymbol{Z} ^{\top} \underbrace{\sum_{ij} w_{ij}(\boldsymbol{e} _i - \boldsymbol{e} _j) (\boldsymbol{e} _i - \boldsymbol{e} _j) ^{\top}}_{=\boldsymbol{L}} \right) \\
&= 2 \operatorname{tr}\left( \boldsymbol{Z} ^{\top} \boldsymbol{L} \boldsymbol{Z} \right) \\
\end{aligned}$$


Hence, our objective is now

$$\begin{aligned}
\min && \operatorname{tr}\left( \boldsymbol{Z} ^{\top} \boldsymbol{L} \boldsymbol{Z} \right) & &&\\
\mathrm{s.t.}
&& \boldsymbol{Z} ^{\top} \boldsymbol{D} \boldsymbol{Z} &= \boldsymbol{I} \\ && \boldsymbol{Z} ^{\top} \boldsymbol{D} \boldsymbol{1} &= \boldsymbol{0} \\
\end{aligned}$$

where the first constraint prevents trivial solution $\boldsymbol{Z} = \boldsymbol{0}$. Actually $\boldsymbol{Z} ^{\top} \boldsymbol{D} \boldsymbol{1} = \boldsymbol{0}$ is not necessary.

The solution ($k$ columns of $\boldsymbol{Z}$) is given by the bottom $k$ eigenvectors (excluding $\boldsymbol{1}$) of the generalized eigenvalue problem

$$
\boldsymbol{L} \boldsymbol{v} = \lambda \boldsymbol{D} \boldsymbol{v}
$$

```{margin}
See [graph Laplacians](graph-laplacian) for details about $\boldsymbol{L} ^\mathrm{rw}$ and $\boldsymbol{L} ^\mathrm{sym}$.
```

Or equivalently, the eigenvectors of random-walk graph Laplacian: $\boldsymbol{L} ^{\mathrm{rw}} = \boldsymbol{D} ^{-1} \boldsymbol{L}$.

To see why the two constraints come from, we can first see a $k=1$ example, i.e. projection onto a line. Suppose the projections are $z_1, \ldots, z_n$, the problem is

$$
\min _{\boldsymbol{z}} \boldsymbol{z} ^{\top} \boldsymbol{L} \boldsymbol{z}
$$

Note that there are two issues
- arbitrary scaling: if $\boldsymbol{z}^*$ is an optimal solution, then a new solution $c\boldsymbol{z}^*$ where $0<c<1$ gives a smaller function value, contradiction. Or we say $\boldsymbol{z} = \boldsymbol{0}$ is a trivial solution.
- translational invariance: if $\boldsymbol{z} ^*$ is an optimal solution, then a new solution $\boldsymbol{z} ^* + c\boldsymbol{1}$ gives the same function value.

```{margin}
The matrix $\boldsymbol{D}$ here is introduced by the authors in the original paper to reflect vertex importance. Actually replacing $\boldsymbol{D}$ by $\boldsymbol{I}$ also solve these two issues.
```

To solve these two issues, we add two constraints $\boldsymbol{z} ^{\top} \boldsymbol{D} \boldsymbol{z} = 1$ and $\boldsymbol{z} ^{\top} \boldsymbol{D} \boldsymbol{1} = 0$ respectively. The second constraint also removes a trivial solution $\boldsymbol{z} = c\boldsymbol{1}$, to be introduced soon. The problem becomes

$$\begin{aligned}
\min && \boldsymbol{z} ^{\top} \boldsymbol{L} \boldsymbol{z}  & &&\\
\mathrm{s.t.}
&& \boldsymbol{z} ^{\top} \boldsymbol{D} \boldsymbol{z} &= 1 &&  \\
&& \boldsymbol{z} ^{\top} \boldsymbol{D} \boldsymbol{1} &= 0 && \\
\end{aligned}$$

the solution is given by the 2nd smallest eigenvector of the generalized eigenproblem

$$
\boldsymbol{L} \boldsymbol{v} = \lambda \boldsymbol{D} \boldsymbol{v}
$$


Note that $\boldsymbol{v} = c\boldsymbol{1}$ is an eigenvector of $\boldsymbol{L}$ but the constraint $\boldsymbol{z} ^{\top} \boldsymbol{D} \boldsymbol{1} =0$ removes that.

To generalize to $k\ge 2$, we generalize the constraints to $\boldsymbol{Z} ^{\top} \boldsymbol{D} \boldsymbol{Z} = \boldsymbol{I}$ and $\boldsymbol{Z} ^{\top} \boldsymbol{D} \boldsymbol{1} = \boldsymbol{0}$ shown above. Note that if we move the second constraint (as in the paper), then the embedding in one of the $k$ dimensions will be $c \boldsymbol{1}$, hence we actually obtain $(k-1)$-dimensional embedding, but it is also helpful to distinguish points.



:::{figure} gb-laplacian-eigenmap-Nt
<img src="../imgs/gb-laplacian-eigenmap-Nt.png" width = "50%" alt=""/>

Laplacian eigenmap with varing $N$-nearest-neighbors and temperature $t$ [Livescu 2021]

:::

<!-- Other formulation: find centered and unit-covariance projections $\boldsymbol{z}_i$ that solve the total projected pairwise distances weighted by $w_{ij}$ and scaled by $d_{ii}d_{jj}$

$$\begin{aligned}
\min &\ \sum_{i j} \frac{w_{i j}|| \boldsymbol{z}_{i}-\boldsymbol{z}_{j}||^{2}}{\sqrt{d_{i i} d_{j j}}} \\
\text{s.t.} &\ \boldsymbol{Z} \text{ is centered and has unit covariance} \\
\end{aligned}$$

The solution $\boldsymbol{Z}$ is given by the $k$ bottom eigenvectors (excluding the smallest one) of the symmetrized normalized Laplacian defined as

$$
\boldsymbol{L}^{\mathrm{sym}} = \boldsymbol{I}  - \boldsymbol{D} ^{-\frac{1}{2}} \boldsymbol{W}  \boldsymbol{D} ^{-\frac{1}{2}}
$$ -->

### Relation to Spectral Clustering

Laplacian eigenmaps, as a dimension reduction that preserves locality, yields the same solution as [normalized cut](Ncut) in spectral clustering. By setting

$$
x_{i}=\left\{\begin{array}{c}
\frac{1}{\operatorname{vol}(A)}, \text { if } V_{i} \in A \\
-\frac{1}{\operatorname{vol}(B)}, \text { if } V_{i} \in B
\end{array}\right.
$$

We can show that $\boldsymbol{x} ^{\top} \boldsymbol{D} \boldsymbol{1} = \boldsymbol{0}$ and

$$
\frac{\boldsymbol{x}^{\top} \boldsymbol{L} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{D}  \boldsymbol{x}}=W(A, B)\left(\frac{1}{\operatorname{vol}(A) }+\frac{1}{\operatorname{vol}(B) }\right)=\operatorname{Ncut}(A, B)
$$

The relaxed problem is

$$\begin{aligned}
\min_{\boldsymbol{x}} && \frac{\boldsymbol{x}^{\top} \boldsymbol{L} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{D}  \boldsymbol{x}} & &&\\
\mathrm{s.t.}
&& \boldsymbol{x} ^{\top} \boldsymbol{D} \boldsymbol{1}  &= 0  && \\
\end{aligned}$$

To solve this, let $\boldsymbol{y} = \boldsymbol{D} ^{1/2} \boldsymbol{x}$, where $\boldsymbol{D}$ is invertible if $G$ has no isolated vertices. Then $\boldsymbol{y} ^{\top} \boldsymbol{D}^{1/2} \boldsymbol{1} =0$ and

$$
\frac{\boldsymbol{x}^{\top} \boldsymbol{L} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{D}}  = \frac{\boldsymbol{y} \boldsymbol{D} ^{-1/2}\boldsymbol{L} \boldsymbol{D} ^{-1/2}\boldsymbol{y} }{\boldsymbol{y} ^{\top} \boldsymbol{y}}
$$

Note that $\boldsymbol{D} ^{-1/2}\boldsymbol{L} \boldsymbol{D} ^{-1/2} = \boldsymbol{L} ^{\mathrm{sym}}$. The problem is then

$$\begin{aligned}
\min_{\boldsymbol{y}} && \frac{\boldsymbol{y} \boldsymbol{L}^{\mathrm{sym}} \boldsymbol{y} }{\boldsymbol{y} ^{\top} \boldsymbol{y}}& &&\\
\mathrm{s.t.}
&& \boldsymbol{y} ^{\top} \boldsymbol{D}^{1/2} \boldsymbol{1} &=0 && \\
\end{aligned}$$

The solution is given by the second smallest eigenvalue of $\boldsymbol{L}^{\mathrm{sym}}$, when $\boldsymbol{y}$ is the corresponding eigenvector.


<!-- ### Interpretation

Consider data points on a circle.

some choice of $\epsilon$

$$
\approx \boldsymbol{L}
$$

$\boldsymbol{L} \boldsymbol{U}  = \boldsymbol{U} \boldsymbol{\Lambda}$ is discretization of the following:

$$
\frac{\partial^2 \boldsymbol{U}}{\partial x^2}  = \lambda \boldsymbol{U}
$$

$$U(0) = U(1)$$ -->

<!-- ### Diffusion Map

Interpretation: weighting $\boldsymbol{L}$ by connectivity (low weight for low connectivity) $\boldsymbol{D} ^{-1} \boldsymbol{L} \boldsymbol{v} = \lambda \boldsymbol{v}$, recall $\boldsymbol{L} = \boldsymbol{D} - \boldsymbol{W}$, then $\boldsymbol{D} ^{-1} \boldsymbol{W} \boldsymbol{v} = (1-\lambda)\boldsymbol{z}$.

$\mathbb{M}  = \boldsymbol{D} ^{-1} \boldsymbol{W}$ is a rwo stochastic matrix.

where $\mathbb{M} \boldsymbol{1} = \boldsymbol{1}, \mathbb{M} \ge 0, \lambda(\mathbb{M}) \le 1$. -->


### Seriation Problem

[Wikipedia](https://en.wikipedia.org/wiki/Seriation_(archaeology))

Now there are some archeological pieces $i = 1, \ldots, n$. Want to time order them $\pi(i)$. Have in hand is there similarity $w_{ij}$.

The problem can be formulated as

$$

$$

Permutation is hard. Spectral relaxation drop the permutation constraint.

Hope that the relative order in $f^*$ can tell $\boldsymbol{\pi}$.

What if we have some information, say $i$ is in some year for some $i$'s? Can we use this information?



## Randomized SVD-based Methods


### Johnson-Lindenstrauss Lemma

Lemma (Johnson-Lindenstrauss)
: For data vectors be $\boldsymbol{x} _1, \boldsymbol{x} _2, \ldots, \boldsymbol{x} _n \in \mathbb{R} ^d$ and  tolerance $\epsilon \in (0, \frac{1}{2} )$, there exists a Lipschitz mapping $f: \mathbb{R} ^d \rightarrow \mathbb{R} ^k$, where $k = \lfloor \frac{24 \log n}{\epsilon^2} \rfloor$ such that

  $$
  (1 - \epsilon) \left\| \boldsymbol{x}_i - \boldsymbol{x}_j  \right\| ^2 \le \left\| f(\boldsymbol{x}_i ) - f(\boldsymbol{x}_j )\right\| \le (1 + \epsilon) \left\| \boldsymbol{x}_i - \boldsymbol{x}_j  \right\| ^2
  $$

How do we construct $f$? Consider a random linear mapping: $f(\boldsymbol{u}) = \frac{1}{\sqrt{k}} \boldsymbol{A} \boldsymbol{u}$ for some $\boldsymbol{A} \in \mathbb{R} ^{k \times d}$ where $k < d$ and $a_{ij} \overset{\text{iid}}{\sim} \mathcal{N} (0, 1)$. The intuition: the columns of $\boldsymbol{A}$ are orthogonal to each other in expectation. If indeed orthogonal, then $\left\| \frac{1}{\sqrt{k}} \boldsymbol{A} \boldsymbol{u}  \right\| = \left\| \boldsymbol{u}  \right\|$.

To prove it, we need the following lemma.

Lemma (Norm preserving)
: Fix a vector $\boldsymbol{u} \in \mathbb{R} ^d$, then $\boldsymbol{A}$ preserves its norm in expectation.

  $$
  \mathbb{E} \left[ \left\| \frac{1}{\sqrt{k}} \boldsymbol{A} \boldsymbol{u}   \right\|^2 \right]  = \mathbb{E} [\left\| \boldsymbol{u}  \right\|^2]
  $$

  :::{admonition,dropdown,seealso} *Proof*


  $$\begin{aligned}
  \frac{1}{k} \mathbb{E} [\left\| \boldsymbol{A} \boldsymbol{u}  \right\| ^2]  
  &= \frac{1}{k} \boldsymbol{u} ^{\top} \mathbb{E} [\boldsymbol{A} ^{\top} \boldsymbol{A} ] \boldsymbol{u}    \\
  &= \frac{1}{k} \boldsymbol{u} ^{\top} k \boldsymbol{I}_{n \times n} \boldsymbol{u}     \\
  &= \left\| \boldsymbol{u}  \right\|   ^2 \\
  \end{aligned}$$

  The second equality holds since


  $$
  \mathbb{E} [\boldsymbol{a} _i ^{\top} \boldsymbol{a} _j] = \left\{\begin{array}{ll}
  \sum_{p=1}^k \mathbb{E} [a_{ik}^2] = \sum_{p=1}^k 1 =k , & \text { if } i=j \\
  0, & \text { otherwise }
  \end{array}\right.
  $$

  :::

Lemma (Concentration)
: Blessing of high dimensionality: things concentrate around mean. The probability of deviation is bounded. We first prove one-side deviation probability. The proof for the other side is similar.

  $$
  \mathbb{P} \left( \left\| \frac{1}{\sqrt{k}} \boldsymbol{A} \boldsymbol{u}   \right\| ^2 > (1 + \epsilon) \left\| \boldsymbol{u}  \right\|  ^2 \right)  \le \exp \left( \frac{k}{2} \left( \frac{\epsilon^2}{2} - \frac{\epsilon^3}{2}  \right) \right)
  $$

  :::{admonition,dropdown,seealso} *Proof*

  Let $\boldsymbol{v} = \frac{\boldsymbol{A} \boldsymbol{u} }{\left\| \boldsymbol{u}  \right\| } \in \mathbb{R} ^k$, it is easy to see $V_i \sim \mathcal{N} (0, 1)$. In this case,

  $$\begin{aligned}
  \mathbb{P}\left( \left\| \boldsymbol{v} \right\| ^2 > (1 + \epsilon) k\right)
  &= \mathbb{P}\left( \exp (\lambda \left\| \boldsymbol{v}  \right\| ^2) > \exp (1+ \epsilon) k \lambda \right)  \\
  &\le \frac{\mathbb{E} [\exp (\lambda \left\| \boldsymbol{v}  \right\| ^2)] }{\exp [ (1+ \epsilon) k\lambda]}  \quad \because \text{Markov inequality} \\
  &\le \frac{[\mathbb{E} [\exp (\lambda V_i^2)]]^k }{\exp [ (1+ \epsilon) k\lambda]}  \quad \because V_i \text{ are i.i.d.}  \\
  &=  \exp [-(1 + \epsilon) k \lambda] \left( \frac{1}{1-2\lambda}  \right)^{k/2} \\
  \end{aligned}$$

  The last equality holds since by moment generating function $\mathbb{E} [e^{tX}] = \frac{1}{\sqrt{1- 2t} }$ for $X \sim \chi ^2 _1$.

  If we choose $\lambda = \frac{\epsilon}{2(1+\epsilon)} < \frac{1}{2}$, then


  $$
  \mathbb{P} (\left\| \boldsymbol{v}  \right\| > (1 + \epsilon)k)  \le \left[ (1+\epsilon)e^{- \epsilon} \right]^{k/2}.
  $$

  Then it remains to show $1+\epsilon \le \exp(\epsilon - \frac{\epsilon^2}{2} +  \frac{\epsilon^3}{2})$ for $\epsilon > 0$, which is true by derivative test. Plug in this inequality we get the required inequality.


  Then by union bound,

  $$
  \mathbb{P}\left( \left\| \boldsymbol{v}  \right\| > (1 + \epsilon) k \text{ or }  \left\| \boldsymbol{v}  \right\| < (1 - \epsilon) k \right) \le 2 \exp \left(\frac{k}{2}\left(\frac{\epsilon^{2}}{2}-\frac{\epsilon^{3}}{2}\right)\right)
  $$

  :::

Now we prove the JL lemma.

:::{admonition,dropdown,seealso} *Proof of JL*

The probability we fail to find an $\epsilon$-distortion map for any $(i, j)$ pair is

$$\begin{aligned}
&= \mathbb{P} \left( \exists i, j: \left\| \boldsymbol{A} \boldsymbol{x}_i - \boldsymbol{A} \boldsymbol{x}_j  \right\|^2 > (1 + \epsilon) \left\| \boldsymbol{x}_i - \boldsymbol{x}_j  \right\|  ^2  \text{ or } < (1 - \epsilon) \left\| \boldsymbol{x}_i - \boldsymbol{x}_j  \right\|  ^2 \right)   \\
&= \mathbb{P} \left( \cup_{(i,j)} \right)  \\
&\le \binom{n}{2} 2  \exp \left(\frac{k}{2}\left(\frac{\epsilon^{2}}{2}-\frac{\epsilon^{3}}{2}\right)\right)\quad \because \text{union bound} \\
&\le 2 n^2 \exp \left(\frac{k}{2}\left(\frac{\epsilon^{2}}{2}-\frac{\epsilon^{3}}{2}\right)\right)\\
\end{aligned}$$

With some choice of $k$, this upper bound is $1 - \frac{1}{n}$, i.e. there is an $\frac{1}{n}$ chance we get a map with $\epsilon$ distortion. What if we want a higher probability?

For some $\alpha$, if we set, $k \ge (4 + 2\alpha) \left( \frac{\epsilon^{2}}{2}-\frac{\epsilon^{3}}{2} \right) ^{-1} \log(n)$, then the embedding $f(\boldsymbol{x} ) = \frac{1}{\sqrt{k}} \boldsymbol{A} \boldsymbol{x}$ succeeds with probability at least $1 - \frac{1}{n^\alpha}$.

:::


## Locally Linear Embedding

Locally linear embedding learns a mapping in which each point can be expressed as a **linear function** of its nearest neighbors.

## Maximum Variance Unfolding

Maximum variance unfolding tries to maximize the variance of the data (like PCA) while respecting neighborhood relationships.

ref: https://www.youtube.com/watch?v=DW3lSYltfzo


.


.


.


.


.


.


.


.
