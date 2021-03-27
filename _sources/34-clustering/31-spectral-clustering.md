# Spectral Clustering

Spectral clustering methods use a similarity graph (as in graph-based dimensionality reduction) to represents the data points, then apply spectral (eigenvector-based) methods on certain graph matrices associated with the connectivity of $G$, to divide the graph into connected sub-graphs.

Similar to graph-based representation learning, there are three steps for spectral clustering. Given a data set.

1. Define a similarity measure
2. Construct a similarity graph, and obtain
   - Adjacency matrix
   - Laplacian matrix
   - ...
4. Run graph cut algorithm on graph matrices with some objectives.


:::{figure} spectral-clustering-ep-1
<img src="../imgs/spectral-clustering-ep-1.png" width = "80%" alt=""/>

Divide the graph into sub-graphs [D. Sontag]
:::

## Similarity Graphs

Formally, we have data points $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n}$, Consider some similarity measures $s_{i j}$ or dissimilarity measures $d_{ij}$. We create a graph $G=(V,E)$, add one node $v_i$ for each data point $\boldsymbol{x}_i$, and unweighted or weighted edges by some criterions.

- **$\varepsilon$-neighborhood graph**: add edge $(v_i, v_j)$ if $d_{ij} \le \varepsilon$

- **$k$-NN**: add edge $(v_i,v_j)$ if $v_j$ is a $k$-NN of $v_i$ **or** vice versa, according to $d_{ij}$

- **Mutual $k$-NN**: add edge $(v_i,v_j)$ if $v_i$ is a $k$-NN of $v_j$ **and** vice versa. It works well for sub-graphs with different densities.

- **Fully connected**: add edge $(v_i,v_j)$ with weight $w_{ij} = s_{ij}$ or some function of $s_{ij}$.

:::{figure} spectral-clustering-types
<img src="../imgs/spectral-clustering-types.png" width = "70%" alt=""/>

Comparison of three types of graphs [von Luxburg]
:::

We mainly consider bisection ($K=2$) and unweighted case.

## Adjacency Matrix-based

Recall that an adjacency matrix contains binary entries of connection relation between any two nodes. Sometimes we can extend it to be the matrix of edge weights (similarities).

- **Degree** (considering weights) $\operatorname{deg} (i)=\sum_{j} a_{i j} = \operatorname{RowSum}_i (A)$
- **Volume** of a set $S$ of nodes $\operatorname{vol}(S)=\sum_{i \in S} \operatorname{deg} _{i}$.

### Property

Let the eigen-pairs of an binary adjacency matrix $\boldsymbol{A}$ be $(\lambda_i, \boldsymbol{v}_i)$, where $\lambda_1 \le \ldots \le \lambda_{N_v}$ (not necessarily distinct).

Fact (Spectrum of graph adjacency matrices)
: In the case of a graph $G$ consisting of two $d$-regular graphs joined to each other by just a handful of vertices,
  - the two largest eigenvalues $\lambda_1, \lambda_2$ will be roughly equal to $d$, and the remaining eigenvalues will be of only $\mathcal{O} (d^{1/2})$ in magnitude. Hence, there is a gap in the spectrum of eigenvalues, namely 'spectral gap'.
  - the two corresponding eigenvectors $\boldsymbol{v} _1, \boldsymbol{v} _2$ are expected two have large positive entires on vertices of one $d$-'regular' graphs, and large negative entires on the vertices of the other.

### Bisection

Using this fact, to find two clusters in the data set, we can compute eigenvalues and eigenvectors of $\boldsymbol{A}$, then find the largest positive and largest negative entries in the two eigenvectors. Their respective neighbors are declared to be two clusters.

For instance, in the plots below, We see that
- The first two eigenvalues are fairly distinct from the rest, indicating the possible presence of two sub-graphs.
- The first eigenvector appears to suggest a separation of the 1st and 34th actors, and some of their nearest neighbors, from the rest of the actors.
- The second eigenvector in turn provides evidence that these two actors, and certain of their neighbors, should themselves be separated.

:::{figure} spectral-clustering-ex1
<img src="../imgs/spectral-clustering-ex1.png" width = "70%" alt=""/>

Spectral analysis of the karate club network. Left: $\left\vert \lambda_i \right\vert$. Right: $\boldsymbol{v} _1, \boldsymbol{v} _2$, colored by subgroups. [Kolaczyk 2009]
:::

For $K \ge 3$, we can
- Look for a spectral gap to determine $K$
- Run $K$-NN using the first $K$ eigenvectors to determine assignment

### Pros and Cons

Cons
- In reality, graphs are far from regular, so this method does not work well
- The partitions found through spectral analysis will tend to be ordered and separated by vertex degree, since the eigenvalues will mirror quite closely the underlying degree distribution. Normalizing the adjacency matrix to have unit row sums is a commonly proposed solution.


## Laplacian Matrix-based

Recall that the graph Laplacian matrix is defined as $\boldsymbol{L} = \boldsymbol{D} -\boldsymbol{A}$ where $\boldsymbol{D}$ is diagonal matrix of degrees.

Let the eigen-pairs of $\boldsymbol{L}$ be $(\lambda_i, \boldsymbol{v}_i)$, where $\lambda_1 \le \ldots \le \lambda_{N_v}$ (not necessarily distinct).

### Property

Fact (Spectrum of graph Laplacian matrices)
: A graph $G$ will consist of $K$ connected components if and only $\lambda_1 = \ldots = \lambda_K = 0$ and $\lambda_{K+1} > 0$.

Therefore, if we suspect a graph $G$ to consist of nearly $K=2$ components, then we expect $\lambda_2$ to be close to zero.

Definitions
: - The **ratio** of the cut defined by $(S, \bar{S})$ is the ratio between the number of across-part edges and the number of vertices in the smaller component.

    $$\phi(S, \bar{S}) = \frac{\left\vert E(S, \bar{S}) \right\vert}{ \left\vert S \right\vert}$$

  - The **isoperimetric number** of a graph $G$ is defined as the smallest ratio of all cuts.

    $$\phi(G)=\min _{S \subset V:|S| \leq N_{v} / 2} \phi(S, \bar{S})$$

The ratio is a natural quantity to minimize in seeking a good bisection of $G$. Unfortunately, this minimization problem to find $S$ for $\phi(G)$, aka **ratio cut**, is NP-hard. But the isoperimetric number is closely related to $\lambda_2$

Property
: The isoperimetric number $\phi(G)$ is bounded by $\lambda_2$ as

  $$
  \frac{\lambda_{2}}{2} \leq \phi(G) \leq \sqrt{\lambda_{2}}\left(2 \operatorname{deg} _{\max }-\lambda_{2}\right)
  $$

Thus, $\phi(G)$ will be small when $\lambda_2$ is small and vice versa.

### Bisection

Fiedler [SAND 144] associate $\lambda_2$ with the connectivity of a graph. We partition vertices according to the sign of their entires in $\boldsymbol{v} _2$:

$$
S_F=\left\{u \in V: \boldsymbol{v} _{2}(u) \geq 0\right\} \quad \text { and } \quad \bar{S}_F=\left\{u \in V: \boldsymbol{v} _{2}(u)<0\right\}
$$

- The eigenvector $\boldsymbol{v} _2$ is hence often called the Fiedler vector
- The eigenvalue $\lambda_2$ is often called the Fiedler value, which is also the algebraic connectivity of the graph.

This method is often called **spectral bisection**. It can be shown that

$$
\phi(G) \leq \phi(S_F, \bar{S}_F) \leq \frac{\phi^{2}(G)}{\operatorname{deg} _{\max }} \leq \lambda_{2}
$$

Fiedler bisection can be viewed as a computationally efficient approximation to finding a best cut achieving $\phi(G)$. In the example below, we see that is gives a good result, classifying all but the 3rd actor correctly.

:::{figure} spectral-clustering-ex2
<img src="../imgs/spectral-clustering-ex2.png" width = "50%" alt=""/>

Fiedler vector $\boldsymbol{v} _2$ of the karate club graph. Color and shape indicate subgroups. [Kolaczyk 2009]
:::

For $K \ge 3$, apply bisection recursively.

### Pros and Cons

Pros
- Work well on bounded-degree planar graphs and certain mesh graphs [SAND 366]

## Computation

For both of the two spectral partitioning methods above, the computational
overhead is in principle determined by the cost of computing the spectral decomposition of an $N_v \times N_v$ matrix $\boldsymbol{A}$ or $\boldsymbol{L}$, which takes $\mathcal{O} (N_v^2)$ time. However, realistically,
- only a small subset of extreme eigenvalues and eigenvectors are needed.
- the matrices $\boldsymbol{A}$ and $\boldsymbol{L}$ will typically be quite sparse in practice.

Lanczos algorithm can efficiently solve such settings. If the graph between $\lambda_2$ and $\lambda_3$ is sufficiently large (nearly $K=2$), the spectral bisection takes $\mathcal{O} (\frac{1}{\lambda_3 - \lambda_2} N_e)$, i.e. almost linear.

## Weighted Case

Now we consider the weighted case from optimization point of view, where weights $w_{ij} = s_{ij}$.

We define the (unnormalized) graph Laplacian  as

$$\boldsymbol{L}_w = \boldsymbol{D} - \boldsymbol{W}$$

where

- $\boldsymbol{W}$ is the similarity matrix of $s_{ij}$
- $\boldsymbol{D}$ is the diagonal matrix of $\boldsymbol{W} \boldsymbol{1}$

The volume of a set $S$ of vertices extends to $\operatorname{vol}(S)=\sum_{i \in S} \sum_{j \in N(i)} w_{ij}$

### Objectives

Definition (Value of a cut)
: A cut is a partition of the graph into two sub-graphs $S$ and $\bar{S}$. The value of a cut is defined as the sum of total edge weights between the two subgraphs

$$W(S, \bar{S})=\sum_{i \in A, j \in B} w_{i j}$$

To partition the graph into $K$ subgraphs, we would like to minimize the sum of cut values between **each** subgraph $A_i$ and the rest of the graph:

$$
\underset{S_{1}, \ldots, S_{K}}{\operatorname{argmin}} \frac{1}{2} \sum_{k=1}^{K} W\left(S_{k}, \bar{S}_{k}\right)
$$

:::{figure} spectral-clustering-cuts
<img src="../imgs/spectral-clustering-cuts.png" width = "70%" alt=""/>

A graph cut with $W(S,\bar{S})=0.3$ [Hamad & Biela]
:::

The algorithm with the above objective function is called **MinCut**, which favors **isolated** nodes.

Other methods with modified/normalized objectives include:

- **Ratio cuts** $(\operatorname{RatioCut} )$ normalizes by cardinality: $\frac{1}{2} \sum_{k=1}^{K} \frac{W\left(S_{k}, \bar{S}_{k}\right)}{\left|S_{k}\right|}$
- **Normalized cuts** $(\operatorname{Ncut})$ normalizes by volume: $\frac{1}{2} \sum_{i=1}^{k} \frac{W\left(S_{i}, \bar{S}_{i}\right)}{\operatorname{vol}\left(S_{i}\right)}$.

### Bisection Normalized Cut

We introduce how to solve normalized cuts. The solution is related to the first $K$ eigenvectors of the Laplacian matrix. We define a indicator vector $\boldsymbol{c} \in\{-1,1\}^{n}$, where $c_i = 1$ means data point $i$ is in cluster/subgraph $A$; otherwise cluster $B$.

We can show that the $\operatorname{Ncut}$ problem to find $\boldsymbol{c}$ is equivalent to the optimization problem

$$
\begin{aligned}
\min _{\boldsymbol{c}} \operatorname{Ncut} (\boldsymbol{c})
=\min _{\boldsymbol{c}} &\ \frac{\boldsymbol{c}^{\top}\boldsymbol{L} _w \boldsymbol{c}}{\boldsymbol{c}^{\top} \boldsymbol{D}  \boldsymbol{c}} \\
 \text { s.t. } &\ \boldsymbol{c}^{\top} \boldsymbol{D}  \boldsymbol{1}  =0 \\
&\ \boldsymbol{c} \in\{-1,1\}^{n}  \\
\end{aligned}
$$

However, solving for discrete combinatorial values is hard. The optimization problem is relaxed to solve for a continuous $\boldsymbol{c}$ vector instead:

$$
\begin{array}{cl}
\min _{\boldsymbol{c}} & \boldsymbol{c}^{\top}\boldsymbol{L} _w \boldsymbol{c} \\
\text { s.t. } & \boldsymbol{c}^{\top} \boldsymbol{D}  \boldsymbol{c}= \boldsymbol{1} \\
\Longrightarrow & \boldsymbol{L} _w \boldsymbol{c}=\lambda \boldsymbol{D}  \boldsymbol{c}
\end{array}
$$

The first eigenvector of $\boldsymbol{L} _w$ is all ones (all data in a single cluster). We take the 2nd eigenvector as the real-valued solution. As shown below, the solved eigenvector (right picture) has positive and negative values, which can be used for assignment.

:::{figure} spectral-clustering-egvector
<img src="../imgs/spectral-clustering-egvector.png" width = "80%" alt=""/>

$\operatorname{Ncut}$ for a data set of $40$ points.
:::


### More Clusters

- Option 1: Recursively apply the 2-cluster algorithm
- Option 2: Treat the first $K$ eigenvectors as a reduced-dimensionality representation of the data, and cluster these eigenvectors. The task will be easier if the values in the eigenvectors are close to discrete, like the above case.

```{margin} Relation to Representation Learning

Spectral clustering is like representation learning, but push the representation to be discrete. We can then apply some simple clustering examples, like $k$-means.

```


## Applications

### Comparison to $k$-means

:::{figure} spectral-clustering-vs-k-means
<img src="../imgs/spectral-clustering-vs-k-means.png" width = "80%" alt=""/>

$k$-means vs spectral clustering on double rings data set [Ng, Jordan, & Weiss]
:::

### Image Segmentation

For an image, we can view each pixel $i$ as a graph vertex, and the similarity depends on pixel content (color intensity) $f_i$ and pixel location $x_i$:

$$
w_{i j}= \left\{\begin{array}{ll}
\exp \left\{ -\left(\frac{1}{2 \pi \sigma^{2}}\|f_i-f_j\|^{2}+\frac{1}{2 \pi \tau^{2}}\|x_i- x_j\|^{2}\right) \right\}, & \text { if } \left\vert x_i-x_j \right\vert< \varepsilon \\
0, & \text { otherwise }
\end{array}\right.
$$

:::{figure} spectral-clustering-img-seg
<img src="../imgs/spectral-clustering-img-seg.png" width = "80%" alt=""/>

Spectral clustering for image segmentation [Shi & Malik]
:::

:::{figure} spectral-clustering-img-seg-2
<img src="../imgs/spectral-clustering-img-seg-2.png" width = "50%" alt=""/>

Spectral clustering for image segmentation [Arbel ́aez et al.]
:::

### Speech Separation

Speech separation (into speakers): Similar to image segmentation, where a “pixel” is a cell of a spectrogram. Darkness corresponds to amount of frequency at that time.

:::{figure} spectral-clustering-speech-sep
<img src="../imgs/spectral-clustering-speech-sep.png" width = "80%" alt=""/>

Spectral clustering for speech separation [Bach & Jordan]
:::
