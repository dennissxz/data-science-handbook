# Shortest Path

## Problem

```{margin}
If the graph is not complete, we can assign arbitrary large edge length to the non-existing edge to make the graph complete.
```

Input
: - A directed complete graph with non-negative edge length $\ell(e)$.
  - A source vertex $s$.


Goal
: Find the shortest path $P$ between $s$ and all other vertices


## Dijkstra's Algorithm

Dijkstra's algorithm is a greedy algorithm, which maintains a set of vertices $S$ and update it iteratively. At start, $S={s}$. If we found a shortest path between $s$ to a vertex $u$, then $u$ is added to $S$. If $S=V$, then the algorithm terminates.

Definition (Shortest path from $s$ to $u$ w.r.t $S$)
: The shortest path from $s$ to $u$ w.r.t. $S$ is a shortest path such that all vertices along the it $(s,u)$ are in $S$. The distance is denoted as $d(u)$.

Let $d^{\text{opt}}(u)$ be the unconstrained shortest path from $s$ to $u$. Then we have

$$
d(u) \ge d^{\text{opt}}(u)
$$

The algorithm add $u$ to $S$ if the equality is achieved.

Implementation:

---
**Dijkstra's algorithm**

---
- Initialize

  - $S = \left\{ s \right\}$
  - $d(s)=0$
  - For all other nodes $u \in V \backslash \left\{ s \right\}$,
    - if $u$ is an neighbor of $s$, set $d(u) = \ell(s,u)$
    - else set $d(u) = \infty$

```{margin}
If we only want to find the shortest path between $s$ and a specific target vertex $t$, then the loop condition can be changed to WHILE $t \notin S$.
```

- While $S \ne V$,

  - Find $u^* = \arg\min _{u \notin S} d(u)$
  - Add $u^*$ to $S$
  - For $u \notin S$,
    - If $d(u^*) + \ell(u^*, u) < d(u)$, then update $d(u) \leftarrow d(u^*) + \ell(u^*, u)$

- For $v \in S$, the shortest path from $s$ to $v$ are is the collection of edges to compute $d(v)$.

---

### Correctness

Claim: For every $u\in S$, we have $d(u) = d^{\text{opt}}(u)$

***Proof by induction***

- Base: at initialization, $S = \left\{ s \right\}, d(s) = d^{\text{opt}}(s) = 0$

- Step: if the claim holds before adding $u^*$, then it also holds after adding $u^*$

For a newly added note $u^*$, let $P_{s,u^*}^{y}$ be any path from $s$ to $u^*$ through $y\notin S$. Since the algorithm select by $u^* = \arg\min _{u \notin S} d(u)$, we have

$$
d(u^*) \le d(y)
$$

By step assumption,

$$
d(y) \le \ell(\text{any other paths from $s$ to $y$})
$$

Let $\ell_{P}(y,u^*)$ be the path length from $y$ to $u^*$ along path $P_{s,u^*}^{y}$

$$\begin{aligned}
d(y) + \ell_{P}(y,u^*)
&\le \ell(\text{any path from $s$ to $y$}) + \ell_{P}(y,u^*)  \\
&\le \ell(\text{any path from $s$ to $y$, then $y$ to $u^*$}) \\
&\le \ell(P_{s,u^*}^{y}) \\
\end{aligned}$$


Therefore,

$$
d(u^*) \le d(y) \le d(y)+ \ell_{P}(y,u^*) \le \ell(P_{s,u^*}^{y})
$$

for any $P_{s,u^*}^{y}$. Hence, $d(u^*) = d^{\text{opt}}(u^*)$

$\square$

### Complexity

There are $n-1$ iterations. In every iteration, the $\arg\min$ step takes $O(n)$, and the update takes $O(n)$.

So total $O(n^3)$.


### Negative Edge

Definition (Negative cycle)
: We call a cycle negative if the sum of the weights of the edges in a cycle is negative.

This leads to a problem

$$
A - negative\ cycle - B
$$

Then the path found by Dijkstra's algorithm goes around the negative cycle for infinite times.

## DP-based Algorithm (Bellman-Fosd)

Definition (Simple Path)
: A path is a simple path if each vertex appears only once, i.e., no cycles.

Observations
: - If there are no negative cycles, then for every vertex $v$, there exists a shortest path $s-v$ that a simple path.

*Proof*

Let $P$ be any shortest $s-v$ path. If $p$ is simple then done. Otherwise, some vertex $x$ appears more than once on $P$. Then we can just delete all vertices between any appearances of $x$ on this path. Since there are no negative cycles, the length of $P$ does not increase.

$$
s - v_1 - x - [v_2 - \ldots - v_k - x] - v_{k+1} - v
$$

Moreover, such path can only contains at most $\left\vert V \right\vert - 1$ vertices.

### Algorithm

#### DP-table

For every $v \in V(G)$ and $0 \le i \le n$, $T_{[i,v]}$ will store the length of shortest $s-v$ path containing at most m$i$ edges.

The final solution is the entry $T_{[n,t]}$.



Either $P$ contains less than $i-1$ edges such that $T_{[i-1, v]}$, or $P$ contains $i$ edges, such that $(u,v)$ i the last edge on $P$, then $T_{[i-1,u]} + \ell(u,v)$.


$$
T_{[i,v]} = \min \left\{\begin{array}{ll}
T_{[i-1, v]} & \text { if } ??? \\
\min_{u\ne v} \left\{ T_{[i-1, u]} + \ell(u,v) \right\} & \text { otherwise }
\end{array}\right.
$$

Base
:
$$
i=0 \\
T_{[0,s]} = 0 \\
T_{[0,v]} = \infty ??
$$

Step
:

### Run Time

$O(n^2)$ entries in the DP table, $O(n)$ table to compute each entry.

Total $O(n^2)$.

### Correctness

Claim
: For all $i, v$, entry $T_{i,v}$ contains the length of shortest $s-v$ path with less than $i$ vertices.

Base
: $i=0$

Step
: Assume correctness for all vertices below $i$...
