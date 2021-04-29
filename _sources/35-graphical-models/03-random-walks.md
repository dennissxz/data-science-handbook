(rw-graph)=
# Random Walks in Graphs

Review [Markov Chains](markov-chains).

## Regular Random Walks

### Model

In this section we consider random walks over undirected connected unweighted graphs. Let $\boldsymbol{A}$ be the ajacency matrix and $\boldsymbol{D}$ be the diagonal matrix of degrees, the transition matrix is defined as

$$
\boldsymbol{P}  = \boldsymbol{D} ^{-1} \boldsymbol{A}
$$

where

$$
p_{ij} = \left\{\begin{array}{ll}
1/d_i, & \text { if } (i,j) \in E \\
0, & \text { otherwise }
\end{array}\right.
$$

Let $\boldsymbol{\pi} ^{(0)}$ be the initial distribution, and $\boldsymbol{\pi} ^{(t)}$ be the distribution after $t$ steps. Both are **row** vectors. We have

$$\pi ^{(t+1)}_j =  \sum_{i} \pi^{(t)}_i \cdot \frac{1}{d_i} A_{ij}$$

In matrix form,

$$
\boldsymbol{\pi}  ^{(t)} = \boldsymbol{\pi}  ^{(t-1)} \boldsymbol{P}= \boldsymbol{\pi}  ^{(0)} \boldsymbol{P}^t.
$$

Theorem (limiting distribution)
: A random walk on connected non-bipartite graphs converges to limiting distribution

  $$
  \lim _{t \rightarrow \infty} \boldsymbol{\pi} ^{(t)} =\lim _{t \rightarrow \infty} \boldsymbol{\pi}  ^{(0)}\boldsymbol{P}^{t}= \boldsymbol{\pi}
  $$

If limiting distribution $\boldsymbol{\pi}$ exists, then we have

$$\begin{aligned}
\lim _{t \rightarrow \infty} \boldsymbol{\pi} ^{(t + 1)}  
&=   \lim _{t \rightarrow \infty} \boldsymbol{\pi} ^{(t)} \boldsymbol{P} \\
\Rightarrow \boldsymbol{\pi} &= \boldsymbol{\pi} \boldsymbol{P}  \\
\end{aligned}$$

Hence $\boldsymbol{\pi}$ is also a stationary distribution of $\boldsymbol{P}$. Moreover, $\boldsymbol{\pi}$ is a **left** eigenvector of $\boldsymbol{P} = \boldsymbol{D} ^{-1} \boldsymbol{A}$ with eigenvalue $\lambda = 1$.

The 'connected' condition corresponds to 'irreducible', and the 'non-bipartite' condition corresponds to 'aperiodic', in Markov Chains. If the two conditions fail, then we may have some stationary distribution $\boldsymbol{\pi}: \boldsymbol{\pi} = \boldsymbol{\pi} \boldsymbol{P}$, but no limiting distribution.

### Computation

Power iteration
: Since $\boldsymbol{\pi} = \boldsymbol{\pi} ^{(0)}\lim _{t \rightarrow \infty} \boldsymbol{P} ^{(t)}$, we can start from arbitrary $\boldsymbol{\pi} ^{(0)}$, and then repeat $\boldsymbol{\pi} ^{(t)} \leftarrow \boldsymbol{\pi} ^{(t-1)} \boldsymbol{P}$ until convergence $\left\| \boldsymbol{\pi} ^{(t)}  - \boldsymbol{\pi} ^{(t-1)}  \right\| _1 < \epsilon$. Can also use $L_2$ norm. $t=50$ is sufficient.

Definition (Reversible random walks)
: A random walk with limiting distribution $\boldsymbol{\pi}$ is called reversible if for all $i, j:\pi_i p_{ij} = \pi _j p_{ji}$.

This means that $\pi_i \frac{a_{ij}}{d_i}  = \pi_j \frac{a_{ji}}{d_j}$. For undirected graphs, since, $a_{ij} = a_{ji}$, we have $\frac{\pi_i}{d_i} = \frac{\pi_j}{d_j}$, i.e., $\frac{\pi_i}{d_i}$ is a constant, or $\pi_i \propto d_i$. Since $\sum_{i=1}^n \pi_i = 1$, we have $\pi_i = \frac{d_i}{\sum_{i=1}^n d_i} = \frac{d_i}{2 N_e}$. This gives a easy way to compute stationary distribution $\boldsymbol{\pi}$ for reversible random walks.

## Lazy Random Walk

Unlike random walk that a walker cannot stay, in lazy random walk he stays with probability $\frac{1}{2}$, or transit to its neighbors with probability $\frac{1}{d_i}$.

$$\pi ^{(t+1)}_j = \frac{1}{2} \pi ^{(t)}_j + \frac{1}{2} \sum_{i} \pi^{(t)}_i \cdot \frac{1}{d_i} A_{ij}$$

In matrix form,

$$
\boldsymbol{\pi}  ^{(t+1)} = \frac{1}{2} \boldsymbol{\pi}  ^{(t)} (\boldsymbol{I} + \boldsymbol{D} ^{-1} \boldsymbol{A} )
$$

If the limiting distribution exists, taking limit on both sides gives the equality

$$
\boldsymbol{\pi} = \boldsymbol{\pi}  (\boldsymbol{D} ^{-1} \boldsymbol{A})
$$

so the limiting distribution are is the same as that in regular random walk.

## Rate of Convergence

Theorem
: Let $\lambda_2$ denote second largest eigenvalue of transition matrix $\boldsymbol{P} = \boldsymbol{D} ^{-1} \boldsymbol{A}$, $\boldsymbol{\pi} ^{(t)}$ probability distribution vector and $\boldsymbol{\pi}$ stationary distribution. If walk starts from the vertex $i$, $\pi_i^{(0)} = 1$, then after $t$ steps for every vertex:

  $$
  \left|\pi_{j}^{(t)}-\pi_{j}\right| \leq \sqrt{\frac{d_{j}}{d_{i}}} \lambda_{2}^{t}
  $$

Note that the first largest eigenvalue of $\boldsymbol{D} ^{-1} \boldsymbol{A}$ is $1$.

- For regular random walk, $\boldsymbol{P} = \boldsymbol{D} ^{-1} \boldsymbol{A}, \lambda_1 = 1, \lambda_2 < 1$
- For lazy random walk, $\boldsymbol{P} ^\prime  = \frac{1}{2} ( \boldsymbol{I}  + \boldsymbol{D} ^{-1} \boldsymbol{A}) , \lambda_2 = \frac{1}{2}(1 + \lambda_2)$


.


.


.


.


.


.


.


.
