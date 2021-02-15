# Matching

Definition (Matching)
: In graph $G=(V,E)$, an edge set $M \subseteq E$ is a matching in $G$, if no two edges in $M$ share a common vertex. That is, every vertex in $(V, M)$ can only have one edge.

We call a vertex $v\in V$ **matched** w.r.t. $M$ if it is an endpoint of some edge $e\in M$.

A matching $M$ is called **perfect**, if all vertices $V$ of $G$ is matched. That is, every vertex $v \in V$ is an endpoint of one and only one edge $e \in M$.

Definition (Maximum Matching)
: A matching $M$ is a maximum matching in $G$, it it contains the largest possible number of edges among all matchings of $G$.

Many matching problems can reduce to max-flow problems.

## Maximum Bipartite Matching

Definition (Bipartite graph)
: A graph is called a bipartite graph if its vertices are partitioned into two subsets, and every edge has one endpoint in one subset, and on in the other.

### Problem

```{margin}
Note this problem is different from interval scheduling. Here one machine only process one job.
```

Consider a job assignment problem. There are a set $J$ of jobs and set $M$ of machines. For each job $j \in J$, we are given a subset $M_j\subseteq M$ of machines that are capable to process the job $j$. We would like to assign as many jobs as possible to machines, subject to the constraint that each machine is assigned at most one job.

### Reduction to Bipartite Matching

We can build a graph $G=(V,E)$ modeling this problem.

- $V$ consists of two disjoint subsets $X$ and $Y$
  - $X$ contains vertex $x_j$ for each job $j$
  - $Y$ contains vertex $y_i$ for each machine $i$
- There is an edge $e(i,j)$ if machine $i$ can process job $j$

Claim
: An optimal assignment of jobs to machines is a maximum matching in $G$

Indeed, every assignment of jobs to machines assigns every job to at most one machine, and
every machine processes at most one job, which translates the assignment directly to a matching
of $G$. As a result, if we solve the maximum matching problem in bipartite graphs, we will solve
the assignment problem too.

### Reduction to Max-flow

We are going to solve the maximum bipartite matching problem by reducing it to an instance
of maximum flow problem. First, construct a graph $G_2$ as follows

- direct all edges from $X$ towards $Y$, assign capacity $1$
- add a source vertex $s$, and directed edges $e(s, x_j)$ with capacity $1$
- add a sink vertex $t$, and directed edges $e(y_i, t)$ with capacity $1$


Claims
: - Any matching of size $z$ in $G$ gives a flow of value $z$ in $G_2$
  - Any integral flow of value $z$ in $G_2$ gives a matching of size $z$ in $G$

Due to this one-one correspondence, as a corollary, a maximum integral flow in $G_2$ gives a maximum bipartite matching in $G$. And, since we can compute the maximum integral flow in $G_2$ efficiently, this gives us an algorithm for computing maximum bipartite matching in $G$.

## Bipartite $b$-matching

### Problem

Now suppose every vertex $x \in X$ can take $b(x) \ge 1$ edges towards $Y$.

Definition (Bipartite $b$-matching)
: Given
  - a bipartite graph $G = (V, E)$ where $V = (X,Y)$
  - a function $b: X \rightarrow \mathbb{Z} _{\ge 0}$

  a subset $M \subseteq E$ of edges is called a $b$-matching, if

  - every vertex $y\in Y$ is an endpoint of at most 1 edge in $M$
  - every vertex $x\in X$ is an endpoint of at most $b(x)$ edges in $M$

### Reduction to Max-flow

Likewise, we can construct a graph $G_2$, but this time we change the capacities of edges that go from source $s$ to vertex $x \in X$ from $1$ to $b(x)$, for every $x \in X$.

Claims
: - Any $b$-matching of size $z$ in $G$ gives a flow of value $z$ in $G_2$
  - Any integral flow of value $z$ in $G_2$ gives a $b$-matching of size $z$ in $G$

### Extension

We can let each $y\in Y$ take up to $b(y)$ edges from $X$. That is, the function $b$ becomes $b: V \rightarrow \mathbb{Z}_{\ge 0}$. Consequently, to solve this, in $G_2$ we change the capacities of edges that go from vertex $y \in Y$ to sink $t$ from $1$ to $b(y)$, for every $y \in Y$.

## Matching in $d$-regular Bipartite Graph

Definition ($d$-regular graph)
: A graph $G=(V,E)$ is called $d$-regular for $d \ge 0$, if every vertex has the same degree $\operatorname{deg}(v) = d$.

A $d$-regular bipartite graph is a bipartite graph where every vertex $x \in X$ take $d$ edges towards $Y$, and vice versa. It turns out that a $d$-regular bipartite graph has **a lot of** different perfect matchings.

Claim (Existence of perfect matching)
: Let $G = (X, Y, E)$ be a $d$-regular bipartite graph with $\left\vert X \right\vert = \left\vert Y \right\vert = n$. Then $G$ contains at least one perfect matching.

*Proof*: A perfect matching can be obtained by constructing $G_2$ with all capacities being 1. can run Ford-Fulkerson algorithm on $G_2$ like in the above sections to obtain a flow of value $n$, which corresponds to a perfect matching.

Claim (Union of matchings)
: Every $d$-regular bipartite graph $G=(X,Y,E)$ is a union of $d$ disjoint perfect matchings.

*Proof*: We can use the above method to find a perfect matching $M$, and delete all its edges from $G$, then obtain $G-M$. It  is easy to see that $G-M$ is a $(d-1)$ bipartite graph. We can repeat this process until no edges are left, and then we get $d$ disjoint perfect matchings in total.
