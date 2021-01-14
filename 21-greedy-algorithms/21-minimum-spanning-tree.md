# Minimum Spanning Tree

## Problem

- Input
  - Graph $G=(V,E)$, with
      - $\left\vert V \right\vert = n$
      - $\left\vert E \right\vert = m$
      - weights $w(e)\ge 0$

- Goal
  - select minimum-weight set $E^\prime \subseteq E$ such that
    - $G(V, E^\prime)$ is connected

## Analysis

Try to think some simplifying assumption.

- all edge weights are positive $w(e)>0$
- all edge weights are integral ?? $w(e) \leftarrow w(e) + \frac{1}{n^2}$
- all edge weights are different $w(e_i)\ne w(e_j)$

Why called minimum spanning **tree**? Since if there is no edge with zero-weight, then the solution is a tree (no cycle). Because if there is a cycle, then we can delete an arbitrary edge with non-zero weight.

Why called **spanning**? Every vertex is in the tree, i.e. $\left\vert V^\prime \right\vert = n$. Note that if you add an additional edge to a spanning tree, you must create a cycle.

So we have: the number of edges in the solution tree is $n-1$

### Kruskal's Algorithm

Start with $E^\prime=\emptyset$
  while $G(E^\prime)$ is not connected
    add some edge according to some greedy rule to $E^\prime$ that does **not** close a cycle

>?? Result: $G(E^\prime)$ is always a forest (collection of trees).

One greedy rule can be: minimum weight

Can it happen that there is no edge to add but the graph is still not connected? If the original graph is not connected, then this happens.

### Feasibility

Immediate

### Optimality

**Cut**: a cut of a graph is a partition of $V$ into 2 non-empty sets $A$ and $B$. The edges between the two sets are denoted by $E(A,B)$.

Claim: For every cut $(A,B)$ of $G$, if $e$ is the smallest weight edge in $E(A,B)$, then $e$ lies in every MST.

**Proof** by contradiction.

Assume that there exists a cut $(A, B)$ and $e$ is the smallest-weight edge in $E(A, B)$, and there is a MST, denoted by $T$, such that $e \not \in T$.

If we add $e$ to $T$, then there is a cycle. There will some path between $U$ and $V$, and hence there is another edge, $e^\prime \in E(A,B)$ such that ??

Claim: $T^\prime = \left( T \cup {e} \right) \backslash \left\{ e^\prime \right\}$ is a feasible solution.

Definition: $e$ is a special edge iff $e$ is the smallest-weight edge in $E(A, B)$ in a cut $(A,B)$

**Proof**

Claim: Every solution we are going to add is a special edge, which lies in OPT. So we obtain an optimal solution.

-> OPT only contains special edges
-> OPT is unique (assumed that all edge weights are distinct)



What if not all edge weights are distinct?

Yes, we'll still get optimal solutions.

### Implementation

Sort edges by their weights to $E = \left\{ e_1, e_2, \ldots, e_m \right\}$ such that $w(e_1) < w(e_2)<\dots < l(e_m)$.

Start with $E \emptyset$

For $i=1$ to $m$

- if endpoints of $e_i$ lie in different connected component of $G(E^\prime)$, then $E^\prime \leftarrow E^\prime U {e_i}$

Complexity

Sorting $O(m\log m)$

Iteration: $m$ iteration, each iteration
- $O(n)$: BFS/DFS to check if two vertices are in the same connected component of $G(E^\prime)$

so total $O(mn)$

Better?

Can maintain data structure that responds to connected queries in time $O(\log n)$ per iteration.

So total run time $O(m\log n)$

## Prim's Algorithms

## Reversed Kruskal

Start $E^\prime = E$, so $G(E^\prime)$ is always connected.

While $\exists \, e: G(E^\prime) \backslash \left\{ e \right\}$ is onnected
- delete ...
