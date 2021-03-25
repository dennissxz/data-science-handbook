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

## Network Cohesion

### Local Density

### Connectivity

### Graph Partitioning

(graph-assortativity)=
### Assortativity and Mixing







## Dynamic Networks
