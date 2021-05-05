# Graph Representation Learning

Graph representation learning alleviates the need to do feature engineering every single time. We don't need to design task-dependent features for node, link, graph, etc.

:::{figure} graph-rep-learning
<img src="../imgs/graph-rep-learning.png" width = "70%" alt=""/>

Workflow of graph machine learning
:::

After embeddings are obtained, we can use it for downstream tasks. For instance, node embeddings $\boldsymbol{z} _i$ can be used for clustering, node classification. For link prediction $(i,j)$, we can use
- Concatenate: $f(\boldsymbol{z} _i, \boldsymbol{z} _j) = g([\boldsymbol{z} _i; \boldsymbol{z} _j])$. Good for directed graphs.
- Hadamard: $f(\boldsymbol{z} _i, \boldsymbol{z} _j) = g(\boldsymbol{z} _i * \boldsymbol{z} _j)$
- Sum/Avg: $f(\boldsymbol{z} _i, \boldsymbol{z} _j) = g(\boldsymbol{z} _i + \boldsymbol{z} _j)$
- Distance: $f(\boldsymbol{z} _i, \boldsymbol{z} _j) = g(\left\| \boldsymbol{z} _i - \boldsymbol{z} _j \right\| _2)$

## Node Embeddings

For node embeddings, we want to learn a function, aka encoder, $\operatorname{ENC} : V \rightarrow \mathbb{R} ^d$, such that
- For two nodes $u, v \in V$, similarity of embeddings $\boldsymbol{z}_u = \operatorname{ENC}(u), \boldsymbol{z} _v=\operatorname{ENC}(v)$ in embedding space $\mathbb{R} ^d$ indicates similarity of nodes $u, v$ in graph $G$
- Embeddings encode graph information
- Embeddings are useful for downstream predictions

Specifically, we need to define
- A measure of similarity of $u, v$ in $G$, denoted $\operatorname{sim}(u, v)$
  - are adjacent
  - share common neighbors
  - have similar structural roles
- A measure of similarity of $\boldsymbol{z} _u, \boldsymbol{z} _v$ in $\mathbb{R} ^d$, aka **decoder**, denoted $\operatorname{DEC}: \mathbb{R} ^d \times \mathbb{R} ^d \rightarrow \mathbb{R}$. For instance, cosine similarity $\boldsymbol{z} _u ^{\top} \boldsymbol{z} _v$.

Different models use different measures. The goal is to preserve similarity: if $\operatorname{sim}(u, v)$ is small/large, we want $\operatorname{DEC}(\boldsymbol{z} _u, \boldsymbol{z} _v)$ to be small/large.


The simplest encoding approach is embedding-lookup.

$$
\operatorname{ENC}(v) = \boldsymbol{z} _v = \boldsymbol{Z} \boldsymbol{v}
$$

- $\boldsymbol{Z} \in \mathbb{R} ^{d\times N_v}$ is a matrix where each column is a node embedding, to be learned/optimized.
- $\boldsymbol{v} \in \mathbb{I} ^{N_v}$ is an indicator vector, with all zeros except a one indicating node $v$

Obviously, the number of parameters $d\times N_v$ can be large if $N_v$ is large. This method is not scalable. Examples include DeepWalk, node2vec.

### DeepWalk

[Perozzl 2014]

Measures
- $\operatorname{sim}(u, v)$: We introduce a similarity definition that uses random walks. Specifically, in graph $G$, let $N_R(u)$ be the sequence of nodes visited on a random walk starting at $u$ according to some random walk strategy $R$ (e.g. uniform). We say $v \in N_R(u)$ is similar to $u$. $N_R(u)$ is aka random walk neighborhood of $u$.
- $\operatorname{DEC}(\boldsymbol{z} _u, \boldsymbol{z} _v)$: cosine similarity, but normalized over all nodes $\boldsymbol{z} _w, w\in V$ by sigmoid function,

  $$
  \operatorname{DEC}(\boldsymbol{z} _u, \boldsymbol{z} _v) = \frac{\exp( \boldsymbol{z} _u ^{\top} \boldsymbol{z} _v)}{ \sum_{w \in V} \exp (\boldsymbol{z} _u ^{\top} \boldsymbol{z} _w)}
  $$

  This can be interpreted as the probability of visiting $v$ on a random walk starting from $u$, i.e.

  $$\mathbb{P} (v \in N_R(u)) = \operatorname{DEC}(\boldsymbol{z} _u, \boldsymbol{z} _v)$$

  note that it is not symmetric.

Pros of using random walks
- Expressivity: incorporates local and higher-order neighborhood information. If a random walk starting from node $\boldsymbol{u}$ visits $\boldsymbol{v}$ w.h.p, then $\boldsymbol{u}$ and $\boldsymbol{v}$ are similar, in terms of high-order multi-hop information
- Efficiency: do not need to consider all node pairs when training; only need to consider pairs that co-occur on random walks


Objective
- Given fixed $u$, we run a random walk from $u$ and obtain $N_R(u)$, then the likelihood can be formulated as

  $$
  \prod_{v \in N_R(u)}\mathbb{P} (v \in N_R(u))
  $$

- If we run random walk for each $u \in V$, the joint log-likelihood is

  $$
  \sum_{u \in V}\sum_{v \in N_R(u)} \log \mathbb{P} (v \in N_R(u))
  $$

- substituting $\mathbb{P} (v \in N_R(u)) = \operatorname{DEC} (\boldsymbol{z} _u, \boldsymbol{z} _v)$, the loss function is then

  $$
  L(\boldsymbol{Z}) = -  \sum_{u\in V} \sum_{v \in N_R (u)} \log \left( \frac{\exp( \boldsymbol{z} _u ^{\top} \boldsymbol{z} _v)}{ \sum_{w \in V} \exp (\boldsymbol{z} _u ^{\top} \boldsymbol{z} _w)} \right)
  $$


- Intuition: learn feature representations $\boldsymbol{z} _u$ that are predictive of the nodes in its random walk neighborhood $N_R(u)$

Learning
- Run **short fixed-length** random walks starting from each node $u \in V$ using some random walk strategy $R$
- Obtain $N_R(u)$ for each $u\in V$. Note that $N_R(u)$ can be a multiset (repeat elements)
- Optimize embeddings that minimize the loss function $L(\boldsymbol{Z})$


Computation
- Two nested summations $u \in V$ and $w \in V$ indicates $\mathcal{O} (N_v ^2)$ complexity. To alleviate this, we approximate the second summation by negative sampling.
- **negative sampling**: instead of normalizing w.r.t. all nodes $w \in V$, we just normalize against $k$ random nodes $w_1, \ldots w_{k}$ sampled from $V$ according to some distribution $\mathbb{P} _V$ over nodes.

  $$
  \log \left( \frac{\exp( \boldsymbol{z} _u ^{\top} \boldsymbol{z} _v)}{ \sum_{w \in V} \exp (\boldsymbol{z} _u ^{\top} \boldsymbol{z} _w)} \right) \approx \log \left( \sigma (\boldsymbol{z} _u ^{\top} \boldsymbol{z} _v) \right) - \sum_{i=1}^k \log \left( \sigma (\boldsymbol{z} _u ^{\top} \boldsymbol{z} _{w_i}) \right)
  $$

  - In $\mathbb{P}_V$, the probability is proportional to its degree
  - Higher $k$ gives more robust estimates, but corresponds to higher bias on negative events. In practice $k=5 \sim 20$.

  :::{admonition,note,dropdown} Why is the approximation valid?

  This is a different objective, but negative sampling is a form of noise contrastive estimation which approximately maximizes the log probability of softmax function. The new formulation stands for a logistic regression (sigmoid function) to distinguish the target node $v$ from nodes $w_i \sim \mathbb{P}_V$. See arxiv.1402.3722.

  :::


- Then solve by SGD.


DeepWalk is an unsupervised way since it does not utilizing node labels or node features.

How to choose strategy $R$? If uniform from neighbors, then it might be too constrained. Node2vec generalizes this.

### Node2vec

[Grover & Leskovec, 2016]

In node2vec, we use biased random walks that can trade off between local and global views of the network. Local ~ BFS, global ~ DFS.

:::{figure} node-emb-node2vec-ep
<img src="../imgs/node-emb-node2vec-ep.png" width = "70%" alt=""/>

Micro- vs Macro-view of neighbourhood [Leskovec 2021]
:::

Two parameters

- Return parameter $p$: return back to the previous node
- In-out parameter $q$ moving outwards (DFS) vs inwards (BFS). Intuitively, $q$ is the ratio of BFS vs DFS

:::{figure} node-emb-node2vec-walk
<img src="../imgs/node-emb-node2vec-walk.png" width = "70%" alt=""/>

Walker moved from $S_1$ to $W$, what's next? [Leskovec 2021]
:::

Learning
- compute random walk probabilities
- simulate $r$ biased random walks of length $\ell$ starting from each node $u \in V$, obtain $N_R(u)$
- use the same objective function, negative sampling, SGD, as DeepWalk

Computation
- Linear-time complexity
- All 3 steps are individually parallelizable

Extensions
- different kinds of biased random walks
  - based on node attributes
  - based on learned weights
- run random walks on modified versions of the original network

Remarks
- node2vec performs better on node classification, while alternative methods perform better on link prediction [Goyal & Ferrara 2017]
- in practice, choose definition of node similarity that matches application


### PageRank


In WWWW, Consider page as nodes and hyperlinks as directed edges. We want to rank the importance of the pages. The importance can be seen as 1-dimensional node embeddings.

#### Model

Assumption
- a page is more important if it has more in-coming links
- links from important pages worth more
- all pages have at least one out-going links, $d_i \ge 1$
- if a page $i$ with importance $r_i$ has $d_i$ out-links, each link gets $r_i/d_i$ importance.
- page $j$'s own importance $r_j$ is the sum of the votes on its in-links. $r_j = \sum_{i: i \rightarrow  j} r_i/d_i$.
- $\sum_{i=1}^{N_v} r_i =1$.

Define a matrix $\boldsymbol{M}$ such that $M_{ij} = \frac{u, v}{d_j}$ if $j \rightarrow i$. We can see it is a column stochastic matrix. The above assumptions leads to the flow equation

$$
\boldsymbol{r} = \boldsymbol{M} \boldsymbol{r}  
$$

#### Computation

To solve $\boldsymbol{r}$, we can use a linear system, but not scalable.

Note that the column stochastic matrix $\boldsymbol{M}$ can define a [random walk over graphs](rw-graph): a walker at $i$ follows an out-link from $i$ uniformly at random. Since $\boldsymbol{r} = \boldsymbol{M} \boldsymbol{r}$, we know that $\boldsymbol{r}$ is a stationary distribution for the random walk.

:::{admonition,seealso} R.t. Eigenvector centrality

Recall [eigenvector centrality](eig-centrality) $\boldsymbol{c}$ can be solved by the first eigenvector of adjacency matrix $\boldsymbol{A}$

$$
\boldsymbol{A} \boldsymbol{c} = \lambda \boldsymbol{c}
$$

In this problem, $\boldsymbol{M} = \boldsymbol{A} \boldsymbol{D} ^{-1}$ and $\boldsymbol{r}$ is the principal eigenvector (i.e. with eigenvalue 1) of $\boldsymbol{M}$.

:::

We can use power iteration to find $\boldsymbol{r}$. But
- if there is a page $i$ without out-going links (aka 'dead end'), then $M_{\cdot i} = 0$, $\boldsymbol{M}$ is not column stochastic which violates the assumption;
- if $i$ only has a self-loop (aka 'spider traps'), i.e. $M_{ii}=1$, then $\boldsymbol{r}$ is degenerate: $r_i = 1$ in $\boldsymbol{r}$, this page has dominates importance.

To overcome these issues, we use teleport trick:
- w.p. $\beta$ follow a link uniformly at random, usually $\beta = 0.8, 0.9$
- w.p. $1-\beta$ jumpy to a random page in $V$

Hence, the PageRank equation [Brin-Page 98] is

$$
r_j = \sum_{i: i \rightarrow j} \beta \frac{r_i}{d_i}  + (1-\beta) \frac{u, v}{N_v}
$$

or

$$
\boldsymbol{r} = \beta \boldsymbol{M} \boldsymbol{r} + (1-\beta)/N_v  \boldsymbol{u, v}  
$$

Note that this formulation assumes that $\boldsymbol{M}$ has no dead ends. We can either preprocess matrix $\boldsymbol{M}$ to remove all dead ends or explicitly follow random teleport links with probability 1 from dead-ends.

Let $\boldsymbol{P} = \beta \boldsymbol{M} + (1- \boldsymbol{\beta} )/N_v \boldsymbol{u, v} \boldsymbol{u, v} ^{\top}$, then we have $\boldsymbol{r}  = \boldsymbol{P} \boldsymbol{r}$. The random walk characterized by column-stochastic matrix $\boldsymbol{P}$ has no dead ends or spider traps, hence we can use the power method over $\boldsymbol{P}$.


#### Personalized PageRank

Aka Topic-specific PageRank

In personalized PageRank, a walker does not teleport to all nodes $V$, but to some subset $S$ of nodes.

If $S$ is the start node, then we call this **random walks with restarts**. We can then use this kind of random walk to form a proximity measure of two nodes: simulate multiple random walk starting from node $s$ with restarts, and then count the number of visits to other nodes. Nodes with higher visit count have higher proximity.

The relative frequencies can also be found using power iteration. The uniform teleport probability $(1-\beta)/N_v  \boldsymbol{u, v}$ in PageRank now becomes $\boldsymbol{e} _s$.

This method also applies to a subset $S$ of multiple nodes. The teleport probability is non-zero for $v \in S$ but 0 otherwise.


Random walks with restarts can be used in recommender systems. The user-item networks can be viewed as a bipartite graph. We can let $S$ be a subset of items, and run the above algorithm, but only count the number of visits to items. Than we obtain a proximity measure of items.

pseudo-code:

```python
item = QUERY_NODES.sample_by_weight()
for i in range( N_STEPS ):
  user = item.get_random_neighbor()
  item = user.get_random_neighbor()
  item.visit_count += 1
  if random( ) > beta:
    item = QUERY_NODES.sample.by_weight ()
```

The similarity consider
- multiple connections
- multiple paths
- direct and indirect connections
- degree of the node


### R.t. Matrix Factorization

Consider two simple measures
- $\operatorname{similarity}(u, v)$: two nodes are similar if they are adjacent.
- $\operatorname{DEC(\boldsymbol{z} _u, \boldsymbol{z} _v)} = \boldsymbol{z} _u ^{\top} \boldsymbol{z} _v$

Let our $d$-dimensional embeddings be $\boldsymbol{Z} \in \mathbb{R} ^{d \times n}$. Then we want to find $\boldsymbol{Z}$ such that $\boldsymbol{A} = \boldsymbol{Z} ^{\top} \boldsymbol{Z}$. However, exact factorization of $\boldsymbol{A}$ is generally impossible (unless it is p.s.d.). We can then approximate the adjacency matrix $\boldsymbol{A}$ by $\boldsymbol{Z} ^{\top} \boldsymbol{Z}$. The problem can be formulated as

$$
\min\ \left\| \boldsymbol{A} - \boldsymbol{Z} ^{\top} \boldsymbol{Z}  \right\| _F
$$

Conclusion: inner product decoder with node similarity defined by edge connectivity is equivalent to matrix factorization of $\boldsymbol{A}$.

DeepWalk and node2vec can also be formulated as matrix factorization problem.


$$
\boldsymbol{M} = \log \left(\operatorname{vol}(G)\left(\frac{u, v}{T} \sum_{r=1}^{T}\left(D^{-1} A\right)^{r}\right) D^{-1}\right)-\log b
$$

- $\operatorname{vol}(G) = \sum_{i,j}^n a_{ij}$
- $T = \left\vert N_R(u) \right\vert$ is the length of random walks
- $b$ is the number of negative samples.

The matrix for node2vec is more complex.

Hence rather than simulating random walks and then use SGD to find $\boldsymbol{Z}$, we can solve the minimization problem $\min \left\| \boldsymbol{M} - \boldsymbol{Z} ^{\top} \boldsymbol{Z}  \right\| _F$.

See Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec [WSDM 18]


### Limitations

There are limitations of node embeddings via matrix factorization and random walks.

- They learn the embeddings, not a function for embedding. Hence, they cannot obtain embeddings for nodes not in the training set. (similar problem happens in PCA, MDS).
- Cannot capture structural similarity of two nodes. Hard to define $\operatorname{similarity} (u, v)$ to measure structural similarity of two nodes, e.g. both are a vertex in a triangle in two different subgraphs in the graph. Anonymous random walk introduced below solves this problem.
- Cannot utilize node, edge and graph features. Sol: graph neural networks.




## Graph Embeddings

Can we entire an entire graph $G$ or some subgraph? For instance, classification of molecules, or identifying anomalous graphs.

Simple ideas:
- to embed an entire graph, first obtain node embeddings, then take sum or average. [Duvenaud+ 2016]
- to embed a subgraph $S$ of $G$, add a virtual node $v$ to $G$ with edges $(v, u)$ for all $u \in H$. Obtain node embedding of $v$ in $\left\{ G \cup \left\{ v \right\} \right\}$, use it as the embedding of $S$. [Li+ 2016]

:::{figure} graph-emb-2
<img src="../imgs/graph-emb-2.png" width = "70%" alt=""/>

Graph embedding by virtual node [Leskovec 2021]
:::

Another example use anonymous walk over graphs.

### Anonymous Walk Embeddings

arxiv.1805.11921 2018

Definition (Anonymous walks)
: An anonymous walk is a special type of random walk where the states correspond to the index of the first time we visited the node, rather than the node label itself.

The states are agnostic to the identity of the nodes visited (hence anonymous)

:::{figure} graph-emb-anon-walk
<img src="../imgs/graph-emb-anon-walk.png" width = "70%" alt=""/>

Anonymous walks
:::

Let $\eta_\ell$ be the number of distinct anonymous walks of length $\ell$. It is easy to see then when length $\ell$ of a anonymous walk is $3$, there are 5 anonymous walks

$$
w_{u, v}=111, w_{2}=112, w_{3}=121, w_{4}=122, w_{5}=123
$$

The number $\eta_\ell$ grows exponentially wicvvc th $\ell$.

:::{figure} graph-emb-anon-walk-number
<img src="../imgs/graph-emb-anon-walk-number.png" width = "70%" alt=""/>

Number of anonymous walks
:::

Learning
- Simulate independently a set of $m$ anonymous walks $w$ of $\ell$ steps and record their counts
- Use the sample distribution of the walks as $\boldsymbol{z} _G$. For instance, if $\ell =3$, then $\boldsymbol{z} _G \in \mathbb{R}^5$.

Computation
- How many anonymous walks $m$ do we need? If we want the distribution has error $\epsilon$ w.p. less than $\delta$, then

  $$
  m=\left[\frac{2}{\varepsilon^{2}}\left(\log \left(2^{\eta_\ell}-2\right)-\log (\delta)\right)\right]
  $$

### Anonymous+

We learn $\boldsymbol{z} _G$ together with anonymous walk embeddings $\boldsymbol{z} _i$ for $i = 1, \ldots, \eta$ where $\eta$ is the number of simulated distinct anonymous walks.

The intuition is, for $T$ independently simulated anonymous walks of length $\ell$ starting from the same node $u$, denoted $w_1^u, w_2^u, \ldots, w_T^u$, they should be 'similar'. The embeddings can be optimized such that the walk $w_t^u$ can be predicted by its left and right walk 'neighbors' in a $\Delta$-size window: $w_s^u$ for $s=t-\Delta, t-\Delta+1, \ldots, t-1, t+1, \ldots, t+\Delta$.

Objective:


$$
\max _{\mathrm{\boldsymbol{Z}}, \mathrm{d}} \sum_{u \in V} \frac{u, v}{T} \sum_{t=\Delta}^{T-\Delta} \log \mathbb{P} \left(w^u_{t} \mid\left\{w^u_{t-\Delta}, \ldots, w^u_{t+\Delta}, \boldsymbol{z} _{G}\right\}\right)
$$

- $\mathbb{P} \left(w_{t} \mid\left\{w_{t-\Delta}, \ldots, w_{t+\Delta}, \boldsymbol{z}_{\boldsymbol{G}}\right\}\right)=\frac{\exp \left(y\left(w_{t}\right)\right)}{\sum_{i=1}^{\eta} \exp \left(y\left(w_{i}\right)\right)}$. Note the denominator is over $\eta$ distinct sampled walks (require negative sampling)
- $y\left(w_{t}\right)=\beta_0 + \boldsymbol{\beta} ^{\top} [\frac{u, v}{2 \Delta} \sum_{i=-\Delta}^{\Delta} \boldsymbol{z}_{i}; \boldsymbol{z}_{\boldsymbol{G}}]$, where $\beta_0, \boldsymbol{\beta}$ are learnable parameters. ';' stands for vertical concatenation This step represents a linear layer.

### Hierarchical Embeddings

We can hierarchically cluster nodes in graphs, and sum/avg the node embeddings according to these clusters.

:::{figure} graph-hier-emb
<img src="../imgs/graph-hier-emb.png" width = "80%" alt=""/>

Hierarchical Embeddings
:::

## Message Passing and Node Classification

Consider a classification problem: Given a network with node features. Some node are labeled. How do we assign labels to all other non-labeled nodes in the network?

One may use node embeddings to build a classifier. We also introduce a method called message passing.

:::{figure} graph-node-classification
<img src="../imgs/graph-node-classification.png" width = "50%" alt=""/>

Node classification [Leskovec 2021]
:::

Notation
- Labeled data of size $\ell$: $(\mathcal{X}_\ell, \mathcal{Y}_\ell) = \left\{ (x_{1:\ell}, y_{1:\ell}) \right\}$
- Unlabeled data $\mathcal{X}_u = \left\{ x_{\ell + 1:n} \right\}$
- adjacency matrix of $n$ nodes $\boldsymbol{A}$, from which we can have $\boldsymbol{A} _\ell$ and $\boldsymbol{A} _u$

This can be viewed as a [semi-supervised](semi-supervised) method, where we use $\left\{ \mathcal{X} _\ell, \mathcal{Y} _\ell, \mathcal{X} _u, \boldsymbol{A} \right\}$ to predict $\mathcal{Y} _u$. It is also a collective classification method, which assigns labels to all nodes simultaneously.

Key observation: correlation exists in networks, nearby nodes have the same characteristics. In social science, there are two concepts
- Homophily: individual characteristics affect social connection. Individuals with similar characteristics tend to be close.
- Influence: social connection affects individual characteristics. One node's characteristics can affect that of nearby nodes.

Hence, the label of $v$ may depend on
- its feature $x_v$
- its nearby nodes' features $x_u$
- its nearby nodes' labels $y_u$

In general, collective classification has three steps
1. Learn a **local** classifier to assign initial labels, using $\left\{ \mathcal{X} _\ell, \mathcal{Y} _\ell, \mathcal{X} _u \right\}$ without using network information $\boldsymbol{A}$
2. Learn a **relational** classifier to label one node based on the labels and/or features of its neighbors. This step uses $\boldsymbol{A}$, and captures correlation between nodes.
3. Collective inference: apply relational classifier to each node iteratively, until convergence of $\mathcal{Y}_\ell$.

In the following we introduce some traditional methods, which are motivation for graphical neural networks.

### Relational Classification

Model: Class probability of a node equals the weighted average of class probability of its neighbors.


$$\begin{aligned}
\mathbb{P} \left(Y_{v}=c\right)
&= \frac{u, v}{d_v} \sum_{u \in \mathscr{N} (v)} \mathbb{P} (Y_u = c) \\  
&=\frac{u, v}{\sum_{(v, u) \in E} A_{v, u}} \sum_{(v, u) \in E} A_{v, u} \mathbb{P} \left(Y_{u}=c\right)
\end{aligned}$$

Algorithm
- Initialize
  - for labeled nodes, use ground-truth label $y_v$
  - for unlabeled nodes, use $Y_v = 0.5$
- Run iterations until convergence of labels or maximum number of iterations achieved
  - Update labels of all non-labeled nodes in a random order

:::{figure} graph-relational-classification
<img src="../imgs/graph-relational-classification.png" width = "70%" alt=""/>

Relational classification [Leskovec 2021]
:::

If edge weights is provided, we can replace $\boldsymbol{A}$ by $\boldsymbol{W}$

Cons
- Convergence is not guaranteed
- This method do not use features $\mathcal{X}$.

### Iterative Classification

Iterative classification uses both features and labels.

Train two classifiers
- $\phi_1 (x _v)$ to predict node label $y_v$ based on node feature vector $x_v$
- $\phi_w (f_v, z_v)$ to predict node label $y_v$ based on node feature vector $f_v$ and summary $z_v$ of labels of its neighbors $\mathscr{N} (v)$. $z_v$ can be
  - relative frequencies of the number of each label in $\mathscr{N} (v)$
  - most common label in $\mathscr{N} (v)$
  - number of different labels in $\mathscr{N} (v)$

Learning

- Phase 1: train classifiers on a training set $\left\{ \mathcal{X} _\ell, \mathcal{Y} _\ell \right\}$
  - $\phi_1 (x)$ using $\left\{ \mathcal{X} _\ell, \mathcal{Y} _\ell \right\}$
  - compute $\mathcal{Z}_\ell$ using $\mathcal{Y} _\ell$ and network information $\boldsymbol{A} _\ell$
  - $\phi_2 (x, z)$ using $\left\{ \mathcal{X} _\ell, \mathcal{Z}_\ell, \mathcal{Y} _\ell \right\}$

- Phase 2: iteration
  - on test set $\left\{ \mathcal{X} _u \right\}$
    - initialize label $\hat{\mathcal{Y}}_{u, 1}$ by $\phi_1 (x_u)$
    - compute $z_u$ by $\hat{\mathcal{Y}}_{u, 1}$ and network information $\boldsymbol{A} _u$
    - update label $\hat{\mathcal{Y}}_{u, 2}$ by $\phi_2 (x_u, z_u)$
  - repeat for **each** node until labels $\hat{\mathcal{Y}}_{u, 2}$ stabilize or max number of iterations is reached
    - compute $z_u$ by $\hat{\mathcal{Y}}_{u, 2}$ and network information $\boldsymbol{A} _u$
    - update label $\hat{\mathcal{Y}}_{u, 2}$ by $\phi_2 (x_u, z_u)$

Remarks
- training set is only used for training $\phi_1, \phi_2$, not involved in iteration
- $\phi_1$ is used to initialize labels $\hat{\mathcal{Y}}_{u, 1}$, which is then used in iteration
- the output $\hat{\mathcal{Y}}_{u, 2}$, obtained from $\phi_2$, use information from both node features and labels.
- convergence is not guaranteed.

### Belief Propagation

Belief propagation is a dynamic programming approach to answering probability queries in a graph. It is an iterative process of passing messages to neighbors. The message sent from $i$ to $j$
- depends on messages $i$ received from its neighbors
- contains $i$'s belief of the state of $j$, e.g. when the state is label, the belief can be 'node $i$ believes node $j$ belong to class 1 with likelihood ...'.

When consensus is reached, we can calculate final belief.

#### In Acyclic Graphs

We introduce belief on labels in acyclic graphs as an example. Define
- $\mathcal{L}$ is the set of all classes/labels

- **Label-label potential matrix** $\boldsymbol{\psi}$ over $\mathcal{L} \times \mathcal{L}$. The entry is

  $$\psi(Y_i, Y_j) \propto \mathbb{P} (Y_j \mid Y_i)$$

  is proportional to the probability of a node $j$ being in class $Y_j$ given that it has neighbor $i$ in class $Y_i$.
- **Prior belief** $\phi$ over $\mathcal{L}$:

  $$\phi(Y_i)\propto \mathbb{P} (Y_i)$$

  is proportional to the probability that node $i$ being in class $Y_i$.

- $m_{i \rightarrow j}(Y_j)$ is $i$'s belief/**message**/estimate of $j$ being in class $Y_j$, which can be compute by

  $$
  m_{i \rightarrow j}(Y_j) = \sum_{Y_i \in \mathcal{L}} \left[ \psi(Y_i, Y_j) \phi (Y_i) \prod_{k \in N_i \setminus j} m_{k \rightarrow i} (Y_i) \right] \quad \forall Y_j \in \mathcal{L}
  $$

  This message is a 'combination' of conditional probabilities, prior probabilities, and 'prior' message from $i$'s neighbors.

:::{figure} graph-belief-prop
<img src="../imgs/graph-belief-prop.png" width = "40%" alt=""/>

Belief propagation [Leskovec 2021]
:::

Learning
- Learn $\boldsymbol{\Psi}$ and $\boldsymbol{\phi}$ by some methods.
- Initialize all messages $m$ to $1$
- Since the graph is acyclic, we can define an ordering of nodes. Start from some node, we follow this ordering to compute $m$ for each node. Repeat until convergence
- Compute self belief as output: node $i$'s belief of being in class $Y_i$
  $$b_i (Y_i) = \phi(Y_i) \prod_{k \in N_i} m_{k \rightarrow  i} (Y_i)\quad \forall Y_j \in \mathcal{L}$$

The messages in the starting node can be viewed as **separate evidence**, since they do not depend on each other.


#### In Cyclic Graphs

It is also called loopy belief propagation since people also used it over graphs with cycles.

Problems in cyclic graphs
- Messages from different subgraphs are no longer independent. There is no 'separate' evidence.
- The initial belief of $i$ (which could be incorrect) is reinforced/amplified by a cycle, e.g. $i \rightarrow j \rightarrow k \rightarrow  u \rightarrow i$
- convergence guarantee and the previous interpretation may be lost

:::{figure} graph-belief-prop-cycle
<img src="../imgs/graph-belief-prop-cycle.png" width = "50%" alt=""/>

Loopy belief propagation on a cyclic graph
:::

In practice, Loopy BP is still a good heuristic for complex graphs which contain many branches, few cycles, or long cycles (weak ).

Since there is no ordering, some modification of the algorithm is necessary.
- start from arbitrary nodes.
- follow the edges to update the neighboring nodes, like a random walker.

#### Review

Advantages:
- Easy to program & parallelize
- Generalize: can apply to any graph model with any form of potentials
  - e.g. higher order: e.g. $\phi (Y_i, Y_j, Y_k)$

- Challenges:
  - Convergence is not guaranteed (when to stop?), especially if many closed loops
  - Potential functions (parameters) need to be estimated


## Graphical Neural Networks

Recall the limitations of shallow embedding methods:
- $\mathcal{O} (N_v)$ parameters are needed
  - no sharing of parameters between nodes
- transductive, no out-of-sample prediction.
- do not incorporate node features

Now we introduce deep graph encoders for node embeddings, where $\operatorname{ENC}(v)$ are deep neural networks. Note that the $\operatorname{similarity}(u,v)$ measures can also be incorporated into GNN. The output can be node embeddings, sub graph embeddings, or entire graph embeddings, to be used for downstream tasks.

:::{figure} gnn-pipeline
<img src="../imgs/gnn-pipeline.png" width = "70%" alt=""/>

Pipeline of graphical neural networks
:::


Let
- $G$ be an undirected graph of order $N_v$
- $\boldsymbol{A}$ be the $N_v \times N_v$ adjacency matrix
- $\boldsymbol{X}$ be the $N_v \times d$ node features

A naive idea to represent the graph with features is to use the $N_v \times (N_v + d)$ matrix $[\boldsymbol{A} , \boldsymbol{X}]$, and then feed into NN. However, there are some issues
- for node-level task
  - $N_v + d$ parameters > $N_v$ examples (nodes), easy overfit
- for graph-level task:
  - not applicable to graphs of different sizes
  - sensitive to node ordering

### Structure

To solve the above issues, GNN borrows idea of CNN filters (hence GNN is also called graphical convolutional neural networks GCN).

#### Computation Graphs

In CNN, a convolutional operator can be viewed as an operator over lattices. Can we generalize it to subgraphs? How to define sliding windows?

```{margin}
Bipartite graph can be 'projected' to obtain 'neighbors'. Useful in recommender systems.
```

Consider a 3 by 3 filter in CNN. We aggregate information in 9 cells and and then output one cell. In a graph, we can aggregate information in a neighborhood $\mathscr{N} (v)$ and output one 'node'. For instance, given messages $h_j$ from neighbor $j$ with weight $w_j$, we can output new message $\sum_{j \in \mathcal{N} (v)} w_j h_j$. In GNN, every node defines a **computation graph** based on its neighborhood

:::{figure} gnn-cnn-filter
<img src="../imgs/gnn-cnn-filter.png" width = "50%" alt=""/>

CNN filter and GNN
:::

It also borrows idea from message passing. The information in GNN propagate through neighbors like those in belief networks. The number of hops determines the number of layers of GNN. For a GNN designed to find node embeddings, the information at each layer is node embeddings.

:::{figure} gnn-aggregate
<img src="../imgs/gnn-aggregate.png" width = "70%" alt=""/>

Aggregation of neighbors information
:::

#### Neurons

The block in the previous computation graph represents an aggregation-and-transform step, where we use neighbors' embeddings $\boldsymbol{h} _u ^{(\ell)}$'s and self embedding $\boldsymbol{h} _v ^{(\ell)}$ to obtain $\boldsymbol{h} _v ^{(\ell + 1)}$. This step work like a neuron. Different GNN models differ in this step.


:::{admonition,note} Note
One important property of aggregation is that the aggregation operator should be permutation invariant, since neighbors have no orders.
:::

A basic approach of aggregation-and-transform is to average last layer information, take linear transformation, and then non-linear activation. Consider an $L$-layer GNN to obtain $k$-dimensional embeddings

- $\boldsymbol{h} _v ^{(0)} = \boldsymbol{x} _v$: initial $0$-th layer embeddings, equal to node features
- $\boldsymbol{h} _v ^{(\ell +1)} = \sigma \left( \boldsymbol{W} _\ell \frac{u, v}{d_v}\sum_{u \in \mathscr{N} (v)} \boldsymbol{h} _u ^ {(\ell)} + \boldsymbol{B} _\ell \boldsymbol{h} _v ^{(\ell)} \right)$ for $\ell = \left\{ 0, \ldots, L-1 \right\}$
  - average last layer (its neighbors') hidden embeddings $\boldsymbol{h} _u ^{(\ell)}$, linearly transformed by $k \times k$ weight matrix $\boldsymbol{W}_ \ell$
  - also take as input its hidden embedding $\boldsymbol{h} _v ^{(\ell)}$ at last layer (last updated embedding?? stored in $\boldsymbol{H}$??), linearly transformed by $k\times k$ weight matrix $\boldsymbol{B}_ \ell$
  - finally activated by $\sigma$.
- $\boldsymbol{z} _v = \boldsymbol{h} _v ^{(L)}$ final output embeddings.

The weight parameters in layer $\ell$ are $\boldsymbol{W} _\ell$ and $\boldsymbol{B} _\ell$, which are shared across neurons in layer $\ell$. Hence, the number of model parameters is sub-linear in $N_v$.

If we write the hidden embeddings $\boldsymbol{h} _v ^{(\ell)}$ as rows in a matrix $\boldsymbol{H}^{(\ell)}$, then the aggregation step can be written as


$$
\boldsymbol{H} ^{(\ell+1)}=\sigma\left(\boldsymbol{D} ^{-1} \boldsymbol{A} \boldsymbol{H} ^{(\ell)} \boldsymbol{W} _{\ell}^{\top}+\boldsymbol{H} ^{(\ell)} \boldsymbol{B} _{\ell}^{\top}\right)
$$

where we update the $v$-th row at a neuron for node $v$, or several rows corresponding to neurons in a layer.

In practice, $\boldsymbol{A}$ is sparse, hence some sparse matrix multiplication can be used. But not all GNNs can be expressed in matrix form, when
aggregation function is complex.

### Training

#### Supervised

After build GNN layers, to train it, we compute loss and do SGD. The pipeine is

:::{figure} gnn-training-pipeline
<img src="../imgs/gnn-training-pipeline.png" width = "50%" alt=""/>

GNN Training Piepline
:::

The prediction heads depend on whether the task is at node-level, edge-level, or graph-level.

##### Node-level

If we have some node label, we can minimize the loss

$$
\min\ \mathcal{L} (y_v, f(\boldsymbol{z} _v))
$$

- if $y$ is a real number, then $f$ maps embedding from $\mathbb{R} ^d$ to $\mathbb{R}$, and $\mathcal{L}$ is L2 loss

- if $y$ is $C$-way categorical, $f$ maps embedding from $\mathbb{R} ^d$ to $\mathbb{R} ^C$, and $\mathcal{L}$ is cross-entropy.

##### Edge-level

If we have some edge label $y_{uv}$, then the loss is

$$
\min\ \mathcal{L} (y_{uv}, f(\boldsymbol{z} _u, \boldsymbol{z} _v))
$$

To aggregate the two embedding vectors, $f$ can be
- concatenation and then linear transformation
- inner product $\boldsymbol{z} _u ^{\top} \boldsymbol{z} _v$ (for 1-way prediction)
- $y_{uv} ^{(r)} = \boldsymbol{z} _u ^{\top} \boldsymbol{W} ^{(r)} \boldsymbol{z} _v$ then $\hat{\boldsymbol{y}} _{uv} = [y_{uv} ^{(1)}, \ldots , y_{uv} ^{(R)}]$. The weights $\boldsymbol{W} ^{(r)}$ are trainable.

##### Graph-level

For graph-level task, we make prediction using all the node embeddings in our graph. The loss is

$$
\min\ \mathcal{L} (y_{G}, f \left( \left\{ \boldsymbol{z} _v, \forall v \in V \right\} \right))
$$

where $f \left( \left\{ \boldsymbol{z} _v, \forall v \in V \right\} \right)$ is similar to $\operatorname{AGG}$ in a GNN layer
- global pooling: $f = \operatorname{max}, \operatorname{mean}, \operatorname{sum}$
- hierarchical pooling: global pooling may lose information. We can apply pooling to some subgraphs to obtain subgraph embeddings, and then pool these subgraph embeddings.

  :::{figure}
  <img src="../imgs/graph-hier-emb.png" width = "80%" alt=""/>

  Hierarchical Pooling
  :::

  To decide subgraph assignment ??

#### Unsupervised

In unsupervised setting, we use information from the graph itself as labels.

- Node-level: some node statistics, e.g. clustering coefficient, PageRank
- Edge-level: $y_{u, v} = 1$ when node $u$ and $v$ are similar. Similarity can be defined using random walks, node proximity, etc.
- Graph-level: some graph statistic, e.g. predict if two graphs are isomorphic


#### Batch

We also use batch gradient descent. In each iteration, we train on a set of nodes, i.e., a batch of compute graphs.

#### Train-test Splitting

Given a graph input with features and labels $\boldsymbol{G} = (V, E, \boldsymbol{X} , \boldsymbol{y})$, how do we split it into train / validation / test set? The speciality of graph is that nodes as observations are connected by edges, they are not independent due to message passing.

##### Node-level

```{margin}
“transductive” means the entire graph can be observed in all dataset splits
```

Transductive setting
- training: hide some nodes' label $\boldsymbol{y} _h$, use all the remaining information $(V, E, \boldsymbol{X} , \boldsymbol{y} \setminus \boldsymbol{y} _h)$ to train a GNN. Of course, the computation graphs are for those labeled nodes.
- test: evaluate on $\boldsymbol{y} _h$, i.e. the computation graphs are for those unlabeled nodes.

Inductive setting
- partition the graph into training subgraph $G_{\text{train} }$ and test subgraph $G_{\text{test} }$, remove across-subgraph edges. Then the two subgraphs are independent.
- training: use $G_{\text{train} }$
- test: use the trained model, evaluate on $G_{\text{test} }$
- applicable to node / edge / graph tasks

:::{figure} gnn-train-test-split
<img src="../imgs/gnn-train-test-split.png" width = "70%" alt=""/>

Splitting graph, transductive (left) and inductive (right)
:::

In the first layer only features not labels are fed into GNN???

##### Link-level

For link-prediction task, we first

Inductive setting
- Partition edges $E$ into
  - message edges $E_m$, used for GNN message passing, and
  - supervision edges $E_s$, use for computing objective, not fed into GNN
- Partition graph into training subgraph $G_{\text{train} }$ and test subgraph $G_{\text{test} }$, remove across-subgraph edges. Each subgraph will have some $E_m$ and some $E_s$.
- Training on $G_{\text{train} }$
- Test on $G_{\text{test} }$

:::{figure} gnn-split-edge-ind
<img src="../imgs/gnn-split-edge-ind.png" width = "70%" alt=""/>

Inductive splitting for link prediction
:::

Transductive setting (common setting)
- Partition the edges into
  - training message edges $E_{\text{train, m}}$
  - training supervision edges $E_{\text{train, s}}$
  - test edges $E_{\text{test}}$
- Training: use training message edges $E_{\text{train, m}}$ to predict training supervision edges $E_{\text{train, s}}$
- Test: Use $E_{\text{train, m}}$, $E_{\text{train, s}}$ to predict $E_{\text{test}}$

After training, supervision edges are **known** to GNN. Therefore, an ideal model should use supervision edges $E_{\text{train, s}}$ in message passing at test time. If there is a validation step, then the validation edges are also used to predict $E_{\text{test}}$.

:::{figure} gnn-split-edge-tran
<img src="../imgs/gnn-split-edge-tran.png" width = "70%" alt=""/>

Transductive splitting for link prediction
:::

### Pros

Inductive capability

- new graph: after training GNN on one graph, we can generalize it to an unseen graph. For instance, train on protein interaction graph from model organism A and generate embeddings on newly collected data about organism B.
- new nodes: if an unseen node is added to the graph, we can directly run forward propagation to obtain its embedding.

### Variants

As said, different GNN models mainly differ in the aggregation-and-transform step. Let's write the aggregation step as

$$
\operatorname{AGG} \left( \left\{ \boldsymbol{h} _u ^{(\ell)}, \forall u \in \mathscr{N} (v)  \right\} \right)
$$

In basic GNN, the aggregation function is just average. And the update function is

$$\boldsymbol{h} _v ^{(\ell +1)} = \sigma \left( \boldsymbol{W} _\ell \frac{u, v}{d_v}\sum_{u \in \mathscr{N} (v)} \boldsymbol{h} _u ^ {(\ell)} + \boldsymbol{B} _\ell \boldsymbol{h} _v ^{(\ell)} \right)$$

This is called graph convolutional networks [Kipf and Welling ICLR 2017].

There are many variants and extensions to this update function. Before aggregation, there can be some transformation of the neighbor embeddings. The aggregation-and-transform step then becomes transform-aggregation-transform.

#### GraphSAGE

[Hamilton et al., NIPS 2017]

In GraphSAGE, the update function is more general,

$$\boldsymbol{h} _v ^{(\ell + 1)} = \sigma \left( \left[ \boldsymbol{W} _\ell \operatorname{AGG} \left( \left\{ \boldsymbol{h} _u ^{(\ell)}, \forall u \in \mathscr{N} (v)  \right\} \right), \boldsymbol{B} _\ell \boldsymbol{h} _v ^{(\ell)}\right] \right)$$

where we concatenate two vectors instead of summation, and $\operatorname{AGG}$ is a flexible aggregation function. L2 normalization of $\boldsymbol{h} _v ^{(\ell + 1)}$ to a unit length embedding vector can also be applied. In some cases (not always), normalization of embedding results in performance improvement

AGG can be

- Mean

  $$
  \operatorname{AGG} = \frac{u, v}{d_v}\sum_{u \in \mathscr{N} (v)} \boldsymbol{h} _u ^ {(\ell)}
  $$

- Pool: Transform neighbor vectors and apply symmetric vector function $\gamma$, e.g. mean, max

  $$\operatorname{AGG} = \gamma \left( \left\{ \operatorname{MLP} (\boldsymbol{h} _u ^{(\ell)} ), \forall u \in \mathscr{N} (v)   \right\} \right)$$

- LSTM: apply LSTM, but to make it permutation invariant, we reshuffle the neighbors (some random order)

  $$
  \operatorname{AGG}  = \operatorname{LSTM} \left( \left[ \boldsymbol{h} _u ^{(\ell)} , \forall u \in \pi (\mathscr{N} (v)) \right] \right)
  $$

#### Matrix Operations

If we use mean, then the aggregation step (ignore $\boldsymbol{B}_\ell$) can be written as

$$
\boldsymbol{H} ^{(\ell+1)}  = \boldsymbol{D} ^{-1} \boldsymbol{A} \boldsymbol{H}  ^{(\ell)}
$$

A variant [Kipf+ 2017]

$$
\boldsymbol{H} ^{(\ell+1)}  = \boldsymbol{D} ^{-1/2} \boldsymbol{A} \boldsymbol{D} ^{-1/2} \boldsymbol{H}  ^{(\ell)}
$$

!!Laplacian

#### Graph Attention Networks

If AGG is mean, then the message from neighbors are of equal importance with the same weight $\frac{u, v}{d_v}$. Can we specify some unequal weight/attention $\alpha_{vu}$?

$$
\operatorname{AGG} = \sum_{u \in \mathscr{N} (v)} \alpha_{vu} \boldsymbol{h} _u ^ {(\ell)}
$$

- Compute **attention coefficients** $e_{vu}$ across pairs of nodes $u, v$ based on their messages, by some **attention mechanism** $a$

  $$
  e_{v u}=a\left(\mathbf{W}^{(l)} \mathbf{h}_{u}^{(l-1)}, \mathbf{W}^{(l)} \boldsymbol{h}_{v}^{(l-1)}\right)
  $$

  The attention coefficients $e_{vu}$ indicates the importance of $u$'s message to node $v$.

- Normalize $e_vu$ into the final **attention weight** $\alpha_{vu}$ by softmax function

  $$
  \alpha_{v u}=\frac{\exp \left(e_{v u}\right)}{\sum_{k \in N(v)} \exp \left(e_{v k}\right)}
  $$

How to design attention mechanism $a$? For instance, $a$ can be some MLP. The parameters in $a$ are trained jointly.

We can generalize to use multi-head attention, which are shown to stabilizes the learning process of attention mechanism [Velickovic+ 2018]. There are $R$ independent attention mechanisms are used. Each one of them, namely a 'heard', computes a set of attention weight $\boldsymbol{\alpha} _{v} ^{(r)}$. Finally, we aggregate the $\boldsymbol{h} _v$ again, by concatenation or summation.
- $\boldsymbol{h} _v ^{(\ell+1, \color{red}{r})} = \sigma \left( \boldsymbol{W} ^{(\ell) }\sum_{u \in \mathscr{N} (v)} \alpha_{vu}^ {(\color{red}{r})} \boldsymbol{h} _u ^ {(\ell)}  \right)$
- $\boldsymbol{h} _v ^{(\ell + 1)}  = \operatorname{AGG} \left( \left\{ \boldsymbol{h} _v ^{(\ell+1, \color{red}{r})}, r = 1, \ldots, R \right\}  \right)$

Pros
- allow for implicitly specifying weights
- computationally efficient, parallelizable
- storage efficient, total number of parameters $\mathcal{O} (N_v + N_e)$

If edge weights $w_{vu}$ are given, we can
- use it as weights $\alpha_vu$, e.g. by softmax function
- incorporate it into the design of attention mechanism $a$.

In many cases, attention leads to performance gains.

#### GIN

##### Expressiveness

Consider a special case that node features are the same $\boldsymbol{x} _{v_1} = \boldsymbol{x} _{v_2} = \ldots$, represented by colors in the discussion below. Then if the computational graph are exactly the same for two nodes, then they have the same embeddings.

:::{figure} gnn-expr-same
<img src="../imgs/gnn-expr-same.png" width = "50%" alt=""/>

Same computational graphs of two nodes
:::

Computational graphs are identical to **rooted subtree** structures around each node. GNN's node embeddings capture rooted subtree structures. Most expressive GNN maps different rooted subtrees into different node embeddings, i.e. should be like an injective function.

:::{figure} gnn-injective
<img src="../imgs/gnn-injective.png" width = "50%" alt=""/>

Injective mapping from computational graph to embeddings
:::

Some of the previously seen models do not use injective function at the neighbor aggregation step. Note that neighbor aggregation is a function over multi-sets (sets with repeating elements). They are not maximally powerful GNNs in terms of expressiveness.
- GCN (mean-pool)
- GraphSAGE aggregation function (MLP + max-pool) cannot distinguish different multi-sets with the same set of distinct colors.

:::{figure} gnn-expr-fail
<img src="../imgs/gnn-expr-fail.png" width = "70%" alt=""/>

Mean and max pooling failure cases
:::

##### GIN

How to design maximally powerful GNNs? Can we design a neural network in the aggregation step that can model injective multi-set function.

Theorem [Xu et al. ICLR 2019]: Any injective multi-set function can be expressed
as:

$$
\Phi \left( \sum_{x \in S} f(x)  \right)
$$

where $f, \Phi$ are non-linear functions. To model them, we can use MLP which have universal approximation power.

$$
\operatorname{MLP}_{\Phi}  \left( \sum_{x \in S} \operatorname{MLP}_{f} (x)  \right)
$$

In practice, MLP hidden dimensionality of 100 to 500 is sufficient. The model is called Graph Isomorphism Network (GIN) [Xu+ 2019].

GIN‘s neighbor aggregation function is injective. It is the most expressive GNN in the class of message-passing GNNs. The key is to use element-wise sum pooling, instead of mean-/max-pooling.

##### R.t. WL Kernel

It is a “neural network” version of the [WL graph kernel](wl-kernel), where the color update function is

$$
c^{(k+1)}(v)=\operatorname{HASH}\left(c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u \in N(v)}\right)
$$

Xu proved that any injective function over the tuple $\left(c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u \in N(v)}\right)$ can be modeled as

$$
\operatorname{GINConv}\left(c^{(k)}(v),\left\{c^{(k)}(u)\right\}_{u \in N(v)}\right) =  \left.\mathrm{MLP}_{\Phi}\left((1+\epsilon) \cdot \mathrm{MLP}_{f}\left(c^{(k)}(v)\right)\right)+\sum_{u \in N(v)} \mathrm{MLP}_{f}\left(c^{(k)}(u)\right)\right)
$$

where $\varepsilon$ is a learnable scalar.

Advantages of GIN over the WL graph kernel are:
- Node embeddings are low-dimensional; hence, they can capture the fine-grained similarity of different nodes.
- Parameters of the update function can be learned for the downstream tasks.

Because of the relation between GIN and the WL graph kernel, their expressive is exactly the same. If two graphs can be distinguished by GIN, they can be also distinguished by the WL kernel, and vice versa. They are powerful to distinguish most of the real graphs!

#### Improvement

There are basic graph structures that existing GNN framework cannot distinguish, such as difference in cycles. One node in 3-node-cycle and the other in 4-node cycle have the same computational graph when layer. GNNs’ expressive power can be improved to resolve the above problem. [You et al. AAAI 2021, Li et al. NeurIPS 2020]

### Layer Design

#### Modules

We can include modern deep learning modules that proved to be useful in many domains, including
- BatchNorm to stabilize neural network training, used to normalize embeddings.
- Dropout to prevent overfitting, used in linear layers $\boldsymbol{W} _\ell \boldsymbol{h} _u ^{(\ell)}$.

:::{figure} gnn-layer
<img src="../imgs/gnn-layer.png" width = "20%" alt=""/>

A GNN Layer
:::

Try design ideas in [GraphGym](https://github.com/snap-stanford/GraphGym).

We then introduce how to add skip connection across layers.

#### Issue with Deep GNN

Unlike some other NN models, adding more GNN layers do not always help. The issue of stacking many GNN layers is that GNN suffers from the **over-smoothing** problem, where all the node embeddings converge to the same value.

Let receptive field be the set of nodes that determine the embedding of a node of interest. In a $L$-layer GNN, this set is the union of $\ell$-hop neighborhood for $\ell=1, 2, \ldots, L$. As the number of layers $L$ goes large, the $L$-hop neighborhood increases exponentially. The receptive filed quickly covers all nodes in the graph.

:::{figure} gnn-receptive-filed
<img src="../imgs/gnn-receptive-filed.png" width = "80%" alt=""/>

Receptive field for different layers of GNN
:::

For different nodes, the **shared** neighbors quickly grow when we
increase the number of hops (num of GNN layers). Their computation graphs are similar, hence similar embeddings.

#### Solution

How to enhance the expressive power of a shallow GNN?
- Increase the expressive power within each GNN layer
  - make $\operatorname{AGG}$ a neural network
- Add layers that do not pass messages
  - MLP before and after GNN layers, as pre-process layers (image, text) and post-process layers for downstream tasks.

We can also add shortcut connections (aka skip connections) in GNN. Then we automatically get a mixture of shallow GNNs and deep GNNs. We can even add shortcuts from each layer to the final layer, then final layer directly aggregates from the all the node embeddings in the previous layers

:::{figure} gnn-skip-connection
<img src="../imgs/gnn-skip-connection.png" width = "70%" alt=""/>

Skip connection in GNN
:::

### Graph Manipulation

It is unlikely that the input graph happens to be the optimal computation graph for embeddings.

Issues and solutions:

- Graph Feature manipulation
  - prob: the input graph lacks features.
  - sol: feature augmentation
- Graph Structure manipulation if the graph is
  - too sparse
    - prob: inefficient message passing
    - sol: add virtual nodes / edges
  - too dense
    - prob: message passing is too costly
    - sol: sample neighbors when doing message passing
  - too large
    - prob: cannot fit the computational graph into a GPU
    - sol: sample subgraphs to compute embeddings

#### Feature Augmentation

Sometimes input graph does not have node features, e.g. we only have the adjacency matrix.

Standard approaches to augment node features
- assign constant values to nodes
- assign unique IDs to nodes, in the form of $N_v$-dimensional one-hot vectors
- assign other node-level descriptive features, that are hard to learn by GNN
  - centrality
  - clustering coefficients
  - PageRank
  - cycle count (as one-hot encoding vector)

:::{figure} gnn-feature-aug
<img src="../imgs/gnn-feature-aug.png" width = "80%" alt=""/>

Comparison of feature augmentation methods
:::

#### Virtual Nodes/Edges

If the graph is too sparse, then the receptive field of a node covers small number of nodes. The message passing is then inefficient.

We can add a virtual node, and connected it to all $N_v$ nodes in the graph. Hence all nodes will have a distance at most 2. (too dense?? if $L=2$ then the input layer covers all nodes??)

#### Neighborhood Sampling

In the standard setting, for a node $v$, all the nodes in $\mathscr{N} _(v)$ are used for message passing. We can actually randomly sample a subset of a node’s neighborhood for message passing.

Next time when we compute the embeddings, we can sample **different** neighbors. In expectation, we will still use all neighbors vectors.

Benefits: greatly reduce computational cost. Allows for scaling to large graphs.

### Graph Generative Models

Types
- Realistic graph generation: generate graphs that are similar to a given set of graphs. (our focus)
- goal-directed graph generation: generate graphs that optimize given objectives/constraints, e.g. molecules chemical properties


Given a set of graph, we want to
- **Density estimation**: model the distribution of these graphs by $p_{\text{model} }$, hope it is close to $p_{\text{data} }$, e.g. maximum likelihood
- **Sampling**: sample from $p_{\text{model} }$. A common approach is
  - sample  from noise $z_i \sim \mathcal{N} (0, 1)$
  - transform the noise $z_i$ via $f$ to obtain $x_i$, where $f$ can be trained NN.

Challenge
- large output space: $N^2$ bits of adjacency matrix?
- variable output space: how to generate graph of different sizes?
- non-unique representation: for a fixed graph, its adjacency matrix has $N_v !$ ways of representation.
- dependency: existence of an edge may depend on the entire graph

#### GraphRNN

[You+ 2018](https://cs.stanford.edu/people/jure/pubs/graphrnn-icml18.pdf)

Recall that by chain rule, a joint distribution can be factorized as

$$
p_{\text {model }}(\boldsymbol{x} ; \theta)=\prod_{t=1}^{n} p_{\text {model }}\left(x_{t} \mid x_{u, v}, \ldots, x_{t-1} ; \theta\right)
$$

In our case, $x_t$ will be the $t$-th action (add node, add edge). The ordering of nodes is a random ordering $\pi$. For a fixed ordering $\pi$ of nodes, for each node, we add a sequence of edges.


:::{figure} gnn-gen
<img src="../imgs/gnn-gen.png" width = "50%" alt=""/>

Generation process of a graph
:::

It is like propagating along columns of upper-triangularized adjacency matrix corresponding to the ordering $\pi$. Essentially, we have transformed graph generation problem into a sequence generation problem. We need to model two processes
1. Generate a state for a new node (Node-level sequence)
2. Generate edges for the new node based on its state (Edge-level sequence)

To model these two sequences, we can use [RNNs](rnn).

GraphRNN has a node-level RNN and an edge-level RNN.
- Node-level RNN generates the initial state for edge-level RNN
- Edge-level RNN sequentially predict the probability that the new node will connect to each of the previous node. Then the last hidden state is used to run node RNN for another step.

:::{figure} graph-rnn-pipeline
<img src="../imgs/graph-rnn-pipeline.png" width = "70%" alt=""/>

GraphRNN Pipeline (node RNN + edge RNN)
:::

The training and test pipeline is:


:::{figure} graph-rnn-train-test
<img src="../imgs/graph-rnn-train-test.png" width = "90%" alt=""/>

GraphRNN training and test example
:::

- SOS means start of sequence, EOS means end of sequence.
- If Edge RNN outputs EOS at step 1, we know no edges are connected to the new node. We stop the graph generation. At test time, this means node sequence length is not fixed.
- In training of edge RNN, teacher enforcing of edge existence is applied. The loss is binary cross entropy

  $$L=- \sum _{u, v} \left[y_{u, v} \log \left(\hat{y}_{u, v}\right)+\left(1-y_{u, v}\right) \log \left(1-\hat{y}_{u, v}\right)\right]$$

  where $\hat{y}_{u, v}$ is predicted probability of existence of edge $(u, v)$ and $y_{u,v}$ is ground truth 1 or 0.

#### Tractability

In the structure introduced above, each edge RNN can have at most $n-1$ step. How to limit this?

The answer is to use Breadth-First search node ordering rather than random ordering of nodes.

:::{figure} graph-rnn-ordering
<img src="../imgs/graph-rnn-ordering.png" width = "50%" alt=""/>

Random ordering and BFS ordering
:::

Illustrated in adjacency matrix, we only look at connectivity with nodes in the BFS frontier, rather than all previous nodes.

:::{figure} graph-rnn-ordering-2
<img src="../imgs/graph-rnn-ordering-2.png" width = "70%" alt=""/>

Random ordering and BFS ordering in adjacency matrices
:::

#### Similarity of Graphs

How to compare two graphs? Qualitatively, we can compare visual similarity. Quantitatively, we can compare graph statistics distribution such as
- degree distribution
- clustering coefficient distribution
- [Graphlet-based](https://en.wikipedia.org/wiki/Graphlets#Graphlet-based_network_properties) distribution
The distance between two distributions can be measured by [earth mover distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) (aka Wasserstein metric in math), which measure the minimum effort that move earth from one pile to the other.

:::{figure} emd
<img src="../imgs/emd.png" width = "70%" alt=""/>

Earth mover distance illustration
:::

How two compare two sets of graphs by some set distance? Each graph element may have different $N_v, N_e$. First we compute a statistic for each graph element, then compute Maximum Mean Discrepancy (MDD). If $\mathcal{X} = \mathcal{H} = \mathbb{R} ^d$ and we choose $\phi: \mathcal{X} \rightarrow \mathcal{H}$ to be $\phi(x)=x$, then it becomes

$$
\begin{aligned}
\mathrm{MMD}(P, Q) &=\left\|\mathbb{E}_{X \sim P}[\varphi(X)]-\mathbb{E}_{Y \sim Q}[\varphi(Y)]\right\|_{\mathcal{H}} \\
&=\left\|\mathbb{E}_{X \sim P}[X]-\mathbb{E}_{Y \sim Q}[Y]\right\|_{\mathbb{R}^{d}} \\
&=\left\|\mu_{P}-\mu_{Q}\right\|_{\mathbb{R}^{d}}
\end{aligned}
$$

Given a set of input graphs, we can generate a set of graphs using some algorithms. Then compare the set distance. Many algorithms are particularly designed to generate certain graphs, but GraphRNN can learn from the input and generate any types of graphs (e.g. grid).

#### Variants

Graph convolutional policy network
- use GNN to predict the generation action
- further uses RL to direct graph generation to certain goals

Hierarchical generation: generate subgraphs at each step



### Limitations

For a perfect GNN:
1. If two nodes have the same neighborhood structure, they must have the same embedding
2. If two nodes have different neighborhood structure, they must have different embeddings

However,
- point 1 may not be practical, sometimes we want to assign different embeddings to two nodes with the same neighborhood structure. Solution: position-aware GNNs.
- point 2 often cannot be satisfied. Nodes on two rings have the same computational graphs. Sol: idendity-aware GNNs

#### Position-aware GNNs

[J. You, R. Ying, J. Leskovec. Position-aware Graph Neural Networks, ICML 2019](https://arxiv.org/abs/1906.04817)

- structure-aware task: nodes are labeled by their structural roles in the graph
- position-aware task: nodes are labeled by their positions in the graph

:::{figure} gnn-struc-posi-aware
<img src="../imgs/gnn-struc-posi-aware.png" width = "70%" alt=""/>

Two types of labels.
:::

GNNs differentiate nodes by their computational graphs. Thus, they often work well for structure-aware task, but fail for position-aware tasks (but node features are different??)

To solve this, we introduce anchors. Randomly pick some nodes or some set of nodes in the graph as **anchor-sets**. Then we compute the relative distances from every nodes to these anchor-sets.

:::{figure} gnn-anchor-set
<img src="../imgs/gnn-anchor-set.png" width = "60%" alt=""/>

Anchors
:::

The distance can then be used as **position encoding**, which represent a node’s position by its distance to randomly selected anchor-set.

Note that each dimension of the position encoding is tied to an anchor-set. Permutation of the order does not change the meaning of the encoding. Thus, we cannot directly use this encoding as augmented feature.

We require a special NN that can maintain the permutation invariant property of position encoding. Permuting the input feature dimension will only result in the permutation of the output dimension, the **value** in each dimension won’t change.

#### Identity-aware GNN

[J. You, J. Gomes-Selman, R. Ying, J. Leskovec. Identity-aware Graph Neural Networks, AAAI 2021]

Heterogenous: different types of message passing is applied to different nodes. Suppose two nodes $v_1, v_2$ have the same computational graph structure, but have different node colorings. Since we will apply different neural network for embedding computation, their embeddings will be different.

:::{figure} gnn-idgnn
<img src="../imgs/gnn-idgnn.png" width = "50%" alt=""/>

Identity-aware GNN
:::

ID-GNN can count cycles originating from a given node, but GNN cannot.

Rather than to heterogenous message passing, we can include identity information as an augmented node feature (no need to do heterogenous message passing). Use cycle counts in each layer as an augmented node feature.

:::{figure} gnn-idgnn-cycle
<img src="../imgs/gnn-idgnn-cycle.png" width = "70%" alt=""/>

Cycle count at each level forms a vector
:::

ID-GNN is more expressive than their GNN counterparts. ID-GNN is the first message passing GNN that is more expressive than 1-WL test.

### Reference

- Tutorials and overviews:
  - Relational inductive biases and graph networks (Battaglia et al., 2018)
  - Representation learning on graphs: Methods and applications (Hamilton et al., 2017)
- Attention-based neighborhood aggregation:   
  - Graph attention networks (Hoshen, 2017; Velickovic et al., 2018; Liu et al., 2018)
- Embedding entire graphs:
  - Graph neural nets with edge embeddings (Battaglia et al., 2016; Gilmer et. al., 2017)
  - Embedding entire graphs (Duvenaud et al., 2015; Dai et al., 2016; Li et al., 2018) and graph pooling (Ying et al., 2018, Zhang et al., 2018)
  - Graph generation and relational inference (You et al., 2018; Kipf et al., 2018)
  - How powerful are graph neural networks(Xu et al., 2017)
- Embedding nodes:
  - Varying neighborhood: Jumping knowledge networks (Xu et al., 2018), GeniePath (Liu et al., 2018)
  - Position-aware GNN (You et al. 2019)
- Spectral approaches to graph neural networks:
  - Spectral graph CNN & ChebNet (Bruna et al., 2015; Defferrard et al., 2016)
  - Geometric deep learning (Bronstein et al., 2017; Monti et al., 2017)
- Other GNN techniques:
  - Pre-training Graph Neural Networks (Hu et al., 2019)
  - GNNExplainer: Generating Explanations for Graph Neural Networks (Ying et al., 2019)
