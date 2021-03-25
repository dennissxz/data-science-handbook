# Descriptive Analysis

## Vertex and Edge Characteristics

How to measure the importance of a vertex or an edge?

### Degree

#### Degree Distributions

Given a network graph $G$, define $f(d)$ to be the fraction of vertices $v \in V$ with degree $d_v = d$. The collection $\left\{ f(d) \right\}$ is called the degree distribution of $G$, which is simply the histogram formed from the degree sequence, with bins of size one.

For directed graphs, degree distributions may be defined analogously for in- and out-degrees.

:::{figure} graph-deg-dist
<img src="../imgs/graph-deg-dist.png" width = "80%" alt=""/>

Degree distributions in base-2 logarithmic scale [Kolaczyk 2009]
:::

In each plot, we can see that majority of vertices are of very low degree, a nevertheless non-trivial number of vertices are of much higher degree. The distribution is right skewed. There is roughly a linear decay, which suggests the presence of a power-law component to these distributions:

$$
f(d) \propto d^{-\alpha}
$$

There are several methods to estimate $\alpha$, or in general, fitting power-law-like distributions. For details see Mitzenmacher [SAND 280].

- Linear regression

  The above relation implies
  $$\log f(d) \sim C-\alpha \log d$$

  Hence we can fit a line to the above plots, and $\hat{\alpha} = - \hat{\beta}$ where $\hat{\beta}$ is the estimated slope.

  However, this method is not advisable due to the disproportionate level of 'noise' in the data at the high degrees.

- Linear regression using cumulative frequencies

  To smooth the noise, we use cumulative frequencies rather than raw frequencies. Note that

  $$\bar{F}(d)=1-F(d) \sim d^{-(\alpha-1)}$$

  We can then fit a line to the plot of $\log \bar{F}(d)$ over $d$.

- Use relative frequencies calculated on intervals of log increasing size (i.e. logarithmic binning).

- Hill estimator of $\gamma = (\alpha - 1) ^{-1}$

  $$
  \hat{\alpha}_{k}=1+\hat{\gamma}_{k}^{-1}, \quad \text { with } \quad \hat{\gamma}_{k}=\frac{1}{k} \sum_{i=0}^{k-1} \log \frac{d_{\left(N_{v}-i\right)}}{d_{\left(N_{v}-k\right)}}
  $$

  where $d_{(1)} \leq \cdots \leq d_{\left(N_{v}\right)}$ are the sorted vertex degrees and $k$ is a value chosen by users. Typically, one can plot $\hat{\alpha}_k$ for a range of $k$, and look for an area where the plot settle down to some stable values of $\hat{\alpha}$. If the decay is sharp and there is no flatten area, then it suggests that a simple power-law-like model is inappropriate.

  :::{figure} graph-deg-hill-plot
  <img src="../imgs/graph-deg-hill-plot.png" width = "80%" alt=""/>

  Hill plots of $\hat{\alpha}_k$ over $k$ for the two datasets above. Note different decay shapes. [Kolaczyk 2009]
  :::

- mixtures of power-laws

- power-law + exponential truncation

  $$f(d) \propto d^{-\alpha} \exp \left(-d / d^{*}\right)$$

#### Joint Degree Distribution

Two graphs may have identical degree sequences and yet otherwise differ noticeably in the way their vertices are paired. To capture information of this sort, we consider a two-dimensional analogue of the degree distribution, i.e. joint degree distribution $f(d_1, d_2)$, which is symmetric.

- For a directed graph, it equals the frequency of an arc $(u, v)$ such that $d_u = d_1, d_v = d_2$.
- For an undirected graph,
  - if $d_1 < d_2$ then $f(d_1, d_2) = f(d_2, d_1) = \frac{1}{2} \times$ frequency of edge such that one end has degree $d_1$ and the other has $d_2$.
  - if $d_1 = d_2 =d$ then $f(d, d) =$ frequency of edge $(u, v)$ such that $d_u = d_v = d$.

In the plot below, we see that the joint distribution concentrate primarily where pairs $(d_1, d_2)$ are both low. However, we can see there is also noticeable tendency for the vertices of largest degree to be connected to low-degree vertices.


:::{figure} graph-deg-dist-joint
<img src="../imgs/graph-deg-dist-joint.png" width = "80%" alt=""/>

Joint degree distributions for the two datasets. Colors range from blue (low relative frequency) to red (high relative frequency), with white indicating areas with no data. [Kolaczyk 2009]
:::

#### Conditional Degree Distribution

From the joint degree distribution we can define conditional degree distribution $f_{d ^\prime \vert d}$: given a vertex of degree $d$, what is the relative frequency of its neighbor that has degree $d ^\prime$?

$$f_{d ^\prime \vert d} = \mathbb{P}\left( D_v = d ^\prime \vert D_u = d, (u,v) \in E \right)$$

We can also defined the conditional mean

$$\bar{d}(d)=\sum_{d^{\prime}} d^{\prime} f_{d^{\prime} \mid d}$$

A negative trend has been observed in $\bar{d}(d)$ as $d$ increases.

#### Degree Correlation

Analogously, we can define correlation $\operatorname{Corr}\left( D, D ^\prime  \right)$ by the joint degree distribution $f(d_1, d_2)$ and its marginals.

For the two data sets above, the degree correlation is 0.023 and -0.093 respectively. Though they are small, the difference in sign reinforces our observation of high-low degree pair in the second data set.

A closely related concept is [assortativity](graph-assortativity).

### Centrality

The importance of a vertex $v$ can be measured by centrality $c(v)$. There are many kinds of centrality measures. Degree is one of them. Deciding which are most appropriate for a given application clearly requires consideration of the context.

#### Closeness Centrality

Closeness centrality measures how close a vertex is to other vertices.

$$c_{cl}(v) = \frac{1}{\sum_{u \in V} \operatorname{dist}(v, u) }$$

where $\operatorname{dist} (v, u)$ is the distance between $u, v$.

Note
- The graph is assumed to be connected. If not, we can define centrality for each connected component, or set a finite upper limit on distances, e.g $N_v$.
- To compute $c_{cl}(v)$, we need to compute single-source shortest paths from $v$ to all other vertices $u \in V$.
- Often, for comparison across graphs and with other centrality measures, this measure is normalized to lie in the interval $[0,1]$, through multiplication by a factor $N_v - 1$. It is 1 if $v$ is the center of a star.

#### Betweenness Centrality

Betweenness centrality relates 'importance' to where a vertex is located with respect to the paths in the graph.

$$
c_{bet}(v) = \sum_{s,t \in V, s\ne v, t \ne v}\frac{\sigma(s,t \mid v)}{\sigma(s,t)}
$$

where
- $\sigma(s,t \mid v)$ is the total number of shortest paths between $s$ and $t$ that pass through $v$
- $\sigma(s,t)$ is the total number of shortest paths between $s$ and $t$

Note

- If all shortest paths are unique, i.e. $\sigma(s,t)=1$, then $c_{bet}(v)$ simple counts how many shortest paths going through $v$.
- It can be normalized to $[0,1]$ through division by $(N_v - 1) (N_v - 2)/2$. For instance, it is 1 if $v$ is the center of a star.

#### Eigenvector Centrality

A vertex's importance may depends on its neighbors' importance. Eigenvector centrality captures this,

$$
c_{eig}(v) = \alpha \sum_{(u,v) \in E} c_{eig}(u)
$$

The vector $\boldsymbol{c} _{eig} = [c_{eig}(1), \ldots, c_{eig}(N_v)] ^{\top}$ is the solution to the eigenvalue problem

$$
\boldsymbol{A} \boldsymbol{c} _{eig} = \alpha ^{-1} \boldsymbol{c} _{eig}
$$

Bonacich [SAND 37] argues that an optimal choice of $\alpha ^{-1}$ is the largest eigenvalue of $\boldsymbol{A}$, and hence $\boldsymbol{c} _{eig}$ is the corresponding eigenvector.

When $G$ is undirected an connected, the largest eigenvector of $\boldsymbol{A}$ is simple: entries are non-zero and share the same sign. Convention is to report the absolute values of these entries.


:::{admonition,note} Computation

Calculation of the largest eigenvalue of a matrix and its eigenvector is a standard problem. The power method is generally used. This method is iterative and is guaranteed to converge under various conditions, such as when the matrix is symmetric, which $\boldsymbol{A}$ will be for undirected graphs. The rate of convergence to $\boldsymbol{c} _{eig}$ will behave like a power, in the number of iterations, of the ratio of the second largest eigenvalue of $\boldsymbol{A}$ to the first.

:::

#### Hubs and Authorities (HITS) Algorithms

Given an adjacency matrix $\boldsymbol{A}$ for a directed web graph,
- hubs are determined by the eigenvector centrality of the matrix $\boldsymbol{M}_{hub} = \boldsymbol{A} \boldsymbol{A} ^{\top}$, where $[\boldsymbol{M}_{hub}]_{ij}=$ the number of vertices that both $i$ and $j$ point to.

  $$[\boldsymbol{M}_{hub}]_{ij}= \langle\boldsymbol{a}_{i\cdot}, \boldsymbol{a} _{j \cdot} \rangle = \sum_{v \in V} \mathbb{I} \left\{ i \rightarrow v \leftarrow j \right\}$$

- authorities are determined by the eigenvector centrality of the matrix $\boldsymbol{M}_{auth} = \boldsymbol{A} ^{\top}\boldsymbol{A}$, where $[\boldsymbol{M}_{auth}]_{ij} =$ the number of vertices that point to both $i$ and $j$.

  $$[\boldsymbol{M}_{hub}]_{ij}= \langle\boldsymbol{a}_{\cdot i}, \boldsymbol{a} _{\cdot j} \rangle = \sum_{v \in V} \mathbb{I} \left\{ i \leftarrow v \rightarrow j \right\}$$

#### Centrality of Edges

Betweenness centrality extends to edges in a straightforward manner. For other measures, we can apply them to the vertices in the edge-to-vertex dual graph (line graph) of $G$.

#### Graph-level Summaries

Once we compute $c(v)$ for all $v$, we can look for graph-level summaries, e.g. the distribution of $c(v)$, in analogy to the degree distribution, as well as its moments and quantiles.

For instance, centralization index is defined as

$$
c = \frac{\sum_{v \in V} c^* - c(v)}{\max \sum_{v \in V} c^* - c(v)}
$$

where
- $c^* = \max_{v \in V} c(v)$
- $\max$ in the denominator is over all possible graphs of order $N_v$, which is not easy to compute outside of certain special cases.

There are many other extension of the above centrality measures to different levels.

## Network Cohesion

Are some subsets of vertices cohesive? Many measures differ from local (triads) to global (giant components), and explicitly (cliques) or implicitly (clusters).

### Local Density

#### Cliques

Recall that a clique is a complete subgraph $H$ of $G$. A common case is that of 3-cliques, i.e. triangles. In practice, large cliques are rare. A sufficient condition for a clique of size $n$ to exist in $G$ is $N_e > \frac{n-2}{n-1}\frac{N_v^2}{2}$. But in real-world networks, $N_e \sim N_v$.

- P:
  - whether a specific subset of nodes $U\subseteq V$ is a clique and whether it is maximal. $O(N_v + N_e)$.
- NP-complete:
  - whether a graph $G$ has a maximal clique of ata least size $n$

#### Plexes

Plexes are weakened version of cliques: A subgraph $H$ consisting of $m$ vertices is called an $n$-plex for $m > n$ if no vertex has degree less than $m-n$. In other words, no vertex is missing more than $n$ of its possible $m-1$ edges with other vertices in the subgraph.

Computation problems tend to scale like those involving cliques.

#### Cores

A $k$-core of a graph $G$ is a subgraph $H$ for which all vertices have degree at least $k$.

A maximal $k$-core subgraph may be computed in $O(N_v +N_e)$. The algorithm computes the shell indices for all $v$. The shell index of $v$ is the largest value $c$ such that $v$ belongs to the $c$-core of $G$ but not its $(C+1)$-core.

#### Density

The density of a subgraph $H = (V_H, E_H)$ is defined as the realized fraction of total possible edges in this subgraph

$$
\operatorname{den}(H)=\frac{\left|E_{H}\right|}{\left|V_{H}\right|\left(\left|V_{H}\right|-1\right) / 2} \in [0,1]
$$

It measures how $H$ is close to a clique. If $\operatorname{den}(H) =1$ then $H$ is a clique. Note that it is a rescaling of the average degree, $\operatorname{den}(H) = \frac{1}{\left\vert V_H \right\vert-1} \bar{d}(H)$

- If $H=G$, then $\operatorname{den}(G)$ gives the density of the overall graph
- If $H=N(v)$ is the set of neighbors of a vertex $v$ and the edges between them, then $\operatorname{den}(N(v))$ gives a measure of density in the immediate neighborhood of $v$, which is called the Watts-Strogatz local clustering coefficient. The average of $\operatorname{den}(N(v))$ can be used as a clustering coefficient for the overall graph.

#### Clustering Coefficient

A measure of density in the immediate neighborhood of $v$ can be the answer to the question: among all pairs of neighbors of $v$, how many of them are connected? Or, what's the proportion that two of my friends are also friends of each other? To answer this, we first formally define triangles and connected triples.

- A triangle is a complete subgraph of order three: $\Delta$
- A connected triple is a subgraph of three vertices connected by two edges (i.e. 2-star): $\land$

Let
- $\tau_\Delta(v)$ be the number of triangles in $G$ such that $v \in V$ falls into
- $\tau_{\land}(v)$ be the number of connected triples in $G$ such that both edges incident to $v\in V$.

We can then define a local clustering coefficient as to answer the questions above. For $v$ with at least two neighbors, i.e. $\tau_{\land }(v) > 0$,

$$
\operatorname{clus} (v) = \frac{\tau_{\Delta}(v)}{\tau_{\land }(v)}
$$

Note that $\tau_{\land }(v) = C_{d_v}^2$ and hence $\operatorname{clus} (v) = \operatorname{den}(N(v))$.

The clustering coefficient for $G$ is the average over "eligible" vertices in $G$,

$$
\operatorname{clus} (G) = \frac{1}{\left\vert V ^\prime  \right\vert}  \sum_{v \in \boldsymbol{V} ^\prime } \operatorname{clus}(v)
$$

where $\boldsymbol{V} ^\prime \subseteq V$ is the set of vertices $v$ with $d_v \ge 2$.

#### Transitivity

The above definition is an simple average of $\operatorname{clus}(v)$, which treat $v$ with different $d_v$ equally. A more informative measure is the weighted average by $\tau_{\land }(v)$,

$$
\frac{\sum_{v \in V ^\prime} \tau_{\land }(v) \operatorname{clus} (v) }{\sum_{v \in V ^\prime} \tau_{\land }(v)} = \frac{\sum_{v \in V ^\prime} \tau_{\Delta}(v)}{\sum_{v \in V ^\prime} \tau_{\land }(v)}
$$

Let
- $\tau_\Delta (G) = \frac{1}{3} \sum_{v \in V} \tau_\Delta(v)$ be the number of triangles in the graph
- $\tau_ \land (G) = \sum_{v \in V} \tau _ \land (v)$ be the number of connected triples in the graph

For $v \ne V ^\prime$, we have $\tau_\Delta(v) = \tau_ \land  (v) = 0$, hence the above fraction equals to

$$\operatorname{clus}_T (G) = \frac{3 \tau_\Delta(G)}{\tau_ \land (G)}$$

which is the transitivity of graph $G$, a standard quantity in the social network literature. Transitivity in this context refers to, for example, the case where the friend of your friend is also a friend of yours.

The two clustering measures $\operatorname{clus}(G)$ and $\operatorname{clus}_T(G)$ can differ. It is possible to define highly imbalanced graphs so that $\operatorname{clus}(G) \rightarrow 1$ while $\operatorname{clus}_T(G) \rightarrow 0$ as $N_v$ grows.

Clustering coefficients have become a standard quantity used in the analysis of network structure. Interestingly,
- their values have typically been found to be quite large in real-world networks, in comparison to what otherwise might be expected based on classical random graph models.
- in large-scale networks with broad degree distributions, it has frequently been found that the local clustering coefficient $\operatorname{clus}(v)$ varies inversely with vertex degree.

Higher-order clustering coefficients have also been proposed, involving cycles of length greater than three.

### Connectivity




### Graph Partitioning

(graph-assortativity)=
### Assortativity and Mixing


## Dynamic Networks


.


.


.


.


.


.


.


.
