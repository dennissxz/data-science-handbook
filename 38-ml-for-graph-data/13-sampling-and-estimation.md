# Sampling and Estimation

Like other statistical data, we usually only observe a sample from a larger underlying graph. We introduce sampling and estimation in graphs.

Graph sampling designs are somewhat distinct from typical sampling designs in non-network contexts, in that there are effectively two inter-related sets $V$ and $E$ of units being sampled. Often these designs can be characterized as having two stages:
1. a selection stage, among one set (e.g. vertices)
2. an observation stage, among the other or both

It's also important to discuss the inclusion probabilities of a vertex and an edge in each sampling design, denoted $\pi_i$ for vertex $i \in E$ and $\pi_{(i, j)}$ for edge $(i,j) \in V^{(2)}$, where $V^{(2)}$ is the set of all unordered pairs of vertices.


## Induced Subgraph Sampling

The two stages are
1. Select a simple random sample of $n$ vertices $V^{*}=\left\{i_{1}, \ldots, i_{n}\right\}$ from $V$ without replacement
2. Observe a set $S^*$ of edges in their induced subgraphs: for $n(n-1)$ pairs of $(i, j)$ for $i,j \in V^*$, check whether $(i,j)\in E$.

For instance, in social networks, we an sample a group of individuals, and then ask their relation or some measure of contact, e.g. friendship, likes or dislike.

The inclusion probabilities are uniformly equal to

$$\begin{aligned}

\pi_{i}&= \mathbb{P}\left( v_i \text{ is selected}  \right) \\
&= \frac{n}{N_v} \\
\pi_{(i,j)}&= \mathbb{P}\left( \text{both $v_i$ and $v_j$ are selected}  \right) \\
&= \mathbb{P}\left( \text{$v_i$ is selected}  \right) \mathbb{P}\left( \text{$v_j$ is selected} \mid \text{$v_i$ is selected}  \right)\\
&= \frac{n}{N_v} \cdot \frac{n-1}{N_v - 1} \quad \because \text{without replacement}
\end{aligned}$$

Note that $N_v$ is necessary to compute these probabilities.

:::{figure} graph-sampling-induced-incident
<img src="../imgs/graph-sampling-induced-incident.png" width = "50%" alt=""/>

Induced (left) and incident (right) subgraph sampling. Selected vertices/edges are shown in yellow, while observed edges/vertices are shown in orange.
:::


## Incident Subgraph Sampling

Complementary to induced subgraph sampling is incident subgraph sampling.
Instead of selecting $n$ vertices in the initial stage, $n$ edges are selected:
1. Select a simple random sample $E^*$ of $n$ edges from $E$ without replacement
2. All vertices incident to the selected edges are then observed, thus providing $V^*$.

For instance, we sample email correspondence from a database, and observe the sender and receiver.

Inclusion probabilities:

$$\begin{aligned}
\pi_{(i, j)} &= \frac{n}{N_e} \\
\pi_i&= 1-\mathbb{P}(\text{no edge incident to $v_i$ is selected}) \\
&=\left\{\begin{array}{ll}\frac{\left(\begin{array}{c}
N_e-d_{i} \\
n
\end{array}\right)}{\left(\begin{array}{c}
N_{e} \\
n
\end{array}\right)} & \text { if } n \leq N_{e}-d_{i} \\ 1, & \text { if } n>N_{e}-d_{i}\end{array}\right. \\
\end{aligned}$$

Hence, in incident subgraph sampling, while the edges are included in the sample graph $G^*$ with equal probability, the vertices are included with unequal probabilities depending on their degrees.

Note that $N_e$ and $d_i$ are necessary to compute the inclusion probabilities. In the example of sampling email correspondence graph, this would require having access to marginal summaries of the total number of emails (say, in a given month) as well as the number of emails in which a
given sender had participated.

### Star Sampling

As its name suggests, we



.


.


.


.


.


.


.


.
