# Minimum Spanning Tree

## Problem Setup

Input
: Graph $G=(V,E)$, with
  - $\left\vert V \right\vert = n$
  - $\left\vert E \right\vert = m$
  - weights $w(e)\ge 0$

Goal
: Select minimum-weight edge set $E^\prime \subseteq E$.

Constraint
: $G(V, E^\prime)$ is connected.

Notations:
: - We denoted $G(V, E ^\prime)$ by $G(E ^\prime)$.
 - We denoted the total cost $\sum_{e \in E} w(e)$ by $w(E)$.

## Analysis

Try to think some reasonable assumptions to transform the original problem setting into a simpler setting.

1. The input graph is connected.

    Otherwise the output graph cannot be connected.

1. All edge weights are positive $w(e)>0$.

    For $e$ with zero weight, we can include it to $E ^\prime$ for free so there is no problem of connection of this edge, and merge the two vertices into one. We solve the problem on the new graph, and the optimal solution in the new graph with the zero edges give the optimal solution on the original graph.

1. All edge weights are different $w(e_i)\ne w(e_j)$. If not, we can add a tiny amount $w(e) \leftarrow w(e) + \frac{1}{n^k}$ to achieve that without changing the optimal solution.

Analysis of OPT
- The optimal solution must be a **tree** (a graph without cycles).

  If there exists cycles, then removing the additional edges in the cycle keeps the graph connected and reduces the weight sum.

- Since the tree include every vertex in the graph, i.e. $\left\vert V^\prime \right\vert = n$, it's a **spanning tree** (a subgraph of $G$ that is a tree which includes all vertices of $G$).

- For an OPT (a spanning tree), the number of edges is always $(n-1)$

- If you add an additional edge $e(u,v)$ to a spanning tree (connected graph), you must close a cycle, since the vertices $(u, v)$ are already connected. If you delete an arbitrary edge on the cycle created, then the resulting graph is still a spanning tree.

Therefore, to generate a spanning tree (a feasible solution), one algorithm can be
- choose one edge according to some rule at a time that does not close a cycle
- repeat until there are $(n-1)$ edges.

Then we will have a graph with $(n-1)$ edges and no cycles, which must be a spanning tree. The question is, what rule should we use to minimize the total cost $w(E)$?


### Kruskal's Algorithm

Kruskal's Algorithm use a greedy rule: add a **minimum-weight** edge connecting **different connected components** of the graph.


```{note}
Can it happen that there is no edge to add but the graph is still not connected? If the original graph is not connected, then this happens. But we assume the original graph is connected.
```

### Correctness Proof

#### Feasibility
 Immediate.

#### Optimality

We use the conclusion from the analysis part.

Claim 1
: An optimal solution is a spanning tree.

Then we do reasoning.

Definition (Cut)
: A cut of a graph is a partition of the vertices $V$ into 2 non-empty sets $A$ and $B$. The edges that go across the two sets are denoted by $E(A,B)$, called cut edges in short.

Claim 2
: For every cut $(A,B)$ of $G$, if $e$ is the smallest weight edge in $E(A,B)$, then $e$ lies in every minimum spanning tree. This smallest weight edge is called **a special edge**.

  ```{dropdown} Proof by contradiction
  Assume that there exist
  - a cut $(A, B)$
  - an edge $e$ that is the smallest-weight edge in $E(A, B)$, with vertices $u \in A$ and $v \in B$.
  - a minimum spanning tree, denoted by $T$

  such that $e \notin T$.

  Since $u$ and $v$ are connected in $T$, there must exists a path in $T$ that goes across $A$ and $B$. Let $e ^\prime$ be the edge that is in that path and goes across $A$ and $B$. Since $w(e) < w(e ^\prime)$, we can replace $e ^\prime$ by $e$ and obtain a new graph $T^\prime = \left( T \cup {e} \right) \backslash \left\{ e^\prime \right\}$.

  Note that $T \cup \{e\}$ is a connected graph. After deleting an edge $e ^\prime$ that lies on a cycle of $T \cup \{e\}$, the the remaining graph $T ^\prime = T \cup \left\{ e \right\} \backslash \left\{ e ^\prime \right\} ^\prime$ graph is still connected, i.e. a feasible solution. Moreover, the total cost is reduced, $w(T ^\prime) < w(T)$.

  Hence, $T$ is **not** a minimum spanning tree, contradiction.
  ```

Corollary 2
: If $T$ is a spanning tree of $G$, then it must contains **all** special edges of $G$.

Claim 3
: There are $(n-1)$ special edges in a graph.

Claim 4
: If $G ^\prime$ is an optimal solution, then it has $(n-1)$ edges and all of them are special edges.
: Proved by Claim 1, Claim 2, Corollary 2, Claim 3

Corollary 4
: If a tree has and only has $(n-1)$ special edges, then it is the unique optimal solution.
: Proved by Claim 3

Since every edge we add according to the greedy rule is a special edge, our solution $G(E ^\prime)$ has $(n-1)$ special edges. By Corollary 4, $G(E ^\prime)$ is the optimal solution.


#### Extension

What if not all edge weights are distinct? There will be multiple optimal solutions. We'll still get an optimal solutions by this algorithm. Just choose an arbitrary edge if there are multiple smallest weight edges.

### Implementation

- Sort edges by their weights to $E = \left\{ e_1, e_2, \ldots, e_m \right\}$ such that $w(e_1) < w(e_2)<\dots < w(e_m)$.

- Start with $E = \emptyset$

- For $i=1$ to $m$

  - if endpoints of $e_i$ lie in different connected component of $G(E^\prime)$, then add this edge $E^\prime \leftarrow E^\prime \cup {e_i}$

### Complexity

- Sorting $O(m\log m)$

- Iteration: $m$ iteration, for each iteration
  - $O(\left\vert E ^\prime \right\vert) = O(n)$: BFS/DFS to check if two vertices are in the same connected component of $G(E^\prime)$ (check if there is a path). Total $O(mn)$

  - Better? Can maintain data structure that responds to connected queries in time $O(\log n)$ per iteration. So total run time $O(m\log n)$

## Reversed Kruskal

- Start $E^\prime = E$, so $G(E^\prime)$ is always connected.

- While $\exists \, e: G(E^\prime) \backslash \left\{ e \right\}$ is connected
  - delete max-weight $e$

## Prim's Algorithms

Just grow a single connected component. Every time, add the cheapest edge reaching out from a connected component.
