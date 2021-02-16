# Graphs

A graph $G = (V, E)$ contains vertex set $V$ and edge set $E$, with size $\left\vert V \right\vert = n, \left\vert E \right\vert = m$.


## Data Structure to Represent a Graph

The most intuitional representation is by an $n\times n$ **adjacency matrix**. It contains binary entries, which is $1$ if there is an edge between vertices $i$ and $j$, and zero otherwise.

:::{figure} graph-adjacency-matrix
<img src="../imgs/graph-adjacency-matrix.png" width = "50%" alt=""/>

Adjacency matrix for undirected and directed graph.
:::

Most real-world networks are sparse, $\left\vert E \right\vert\ll \left\vert E _\max \right\vert$, or $\bar{d} \ll n-1$, so the adjacency matrix is quite sparse.

:::{figure} graph-adjacency-sparse
<img src="../imgs/graph-adjacency-sparse.png" width = "40%" alt=""/>

Sparse adjacency matrix
:::

Adjacency matrix can also store weights.

:::{figure} graph-weights
<img src="../imgs/graph-weights.png" width = "50%" alt=""/>

Adjacency matrix with weights
:::

Other data structures include

- **Edge List**

  $m$ objects in the list, each object represents an edge, and stores the pair of vertices of that edge.

- **Adjacency List**

  $n$ objects in the list, each object represents a vertex, and stores a list of neighbors of that vertex.

  - Easy to work with large and sparse graphs.
  - Allows us to quickly retrieve all neighbors of a given vertex

  :::{figure} graph-edge-adjacency-list
  <img src="../imgs/graph-edge-adjacency-list.png" width = "50%" alt=""/>

  Edge list and adjacency list
  :::


## More Types of graphs

- **Self-edges** (self-loops): allow self-edge $(u,u)$

- **Multi-graph**: allow multiple edges between two vertices.

  :::{figure} graph-self-edge-multigraph
  <img src="../imgs/graph-self-edge-multigraph.png" width = "50%" alt=""/>

  Self-edges graph and multigraph
  :::


- **Connected graph**

  There is a path between every pairs of vertices.

  - Bridge edge: if we erase the edge, the graph becomes disconnected.
  - Articulation node: if we erase the node, the graph becomes disconnected.

  To identify connectivity, we can check the adjacency matrix. The adjacency matrix of a network with several components can be written in a block-diagonal form, so that nonzero elements are confined to squares, with all other elements being zero.

- **Connected directed graph**:

  - **Strong connected**: has a path from each node to every other node and vice versa ($a-b$ and $b-a$).

  - **Weakly connected**: connected if we disregard the edge directions.

  - **Strongly connected components**: a component that is strongly connected.
    - **in-component**: nodes that can reach SCC
    - **out-component**: nodes that can be reached from SCC

- **Folded/projected bipartite graphs**

  In the projection of a bipartite graph $G=(X,Y,E)$ over $X$, $x_1, x_2 \in X$ is connected if they are both connected to $y \in Y$.

  For instance, in the authors-papers network, we can find co-authorship network from projection on authors.

  :::{figure} graph-folded
  <img src="../imgs/graph-folded.png" width = "40%" alt=""/>

  Folded/projected bipartite graphs
  :::


- **Hyterogeneous graphs** (node are different kinds of objects)
- **Multimodal graphs** (topics, papers, authors, institutions)

  :::{figure} mlg-more-types
  <img src="../imgs/mlg-more-types.png" width = "60%" alt=""/>

  More types of graphs
  :::
