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


## Greedy Algorithm (Dijkstra)

Dijkstra's algorithm is a greedy algorithm, which maintains a set of vertices $S$ and update it iteratively. At start, $S={s}$. If we found a shortest path between $s$ to a vertex $u$, then $u$ is added to $S$. If $S=V$, then the algorithm terminates.

Definition (Shortest path from $s$ to $u$ w.r.t $S$)
: The shortest path from $s$ to $u$ w.r.t. $S$ is a shortest path such that all vertices along the it $(s,u)$ are in $S$. The distance is denoted as $d(u)$.

Let $d^{\text{opt}}(u)$ be the unconstrained shortest path from $s$ to $u$. Then we have

$$
d(u) \ge d^{\text{opt}}(u)
$$

The algorithm add $u$ to $S$ if the equality is achieved.

### Algorithm

The algorithm is

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

- For $v \in S$, the shortest path from $s$ to $v$ are the collection of edges to compute $d(v)$.

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

More generally, negative edge are allowed. Dijkstra's algorithm does not work. The greedy edge may have a larger length than the sum of other two edges.

$$
s \quad \overset{\quad 1 \quad }{\longrightarrow} \quad v
$$

$$
1000 \searrow \qquad \nearrow -1000
$$

$$
u
$$


## DP-based Algorithm (Bellman-Fosd)

If there are negative edges but no negative cycles, then the shortest path problem is not well defined due to negative cycles.

Definition (Negative cycle)
: We call a cycle negative if the sum of the weights of the edges in a cycle is negative.

$$
A - \text{negative cycle}  - B
$$

Then the path can go around the negative cycle for **infinite** times and have infinitely small length. Hence, out goal change to find the shortest **simple** path.

Definition (Simple Path)
: A path is a simple path if each vertex appears only once, i.e., no cycles.


### Analysis

#### An Observation

Observation
: - If there are no negative cycles, then for every vertex $v$, there exists a shortest path $s-v$ that a simple path.

***Proof***

Let $P$ be any shortest $s-v$ path. If $p$ is simple then done. Otherwise, some vertex $x$ appears more than once on $P$. Then we can just delete all vertices between any appearances of $x$ on this path. Since there are no negative cycles, the length of $P$ does not increase.

$$
s - v_1 - x - [v_2 - \ldots - v_k - x] - v_{k+1} - v
$$

Moreover, such path can only contains at most $\left\vert V \right\vert - 1$ vertices.


#### DP-table


We want to store the length and the number of edges from $s$ to any vertex $v$ along the shortest path.

For every $v \in V(G)$ and $0 \le i \le n$, $T_{i,v}$ will store the **length** of shortest $s-v$ path containing **at most** $i$ edges.

#### Iterative Relation

For $T_{i,v}$, there are two cases

- $P$ contains less than $i-1$ edges. Then $T_{i,v} = T_{i-1, v}$

- $P$ contains $i$ edges. Let the last edge be $(u,v)$, then $T_{i,v} = \min_{u\ne v} \left\{ T_{i-1, u} + \ell(u,v) \right\}$.

Hence, we have

$$
T_{i,v} = \min \left\{\begin{array}{ll}
T_{i-1, v}  \\
\min_{u\ne v} \left\{ T_{i-1, u} + \ell(u,v) \right\}
\end{array}\right\}
$$

Base cases are

$$\begin{aligned}
T_{0,s} &= 0 \\
T_{0,v} &= \infty \ \forall v \ne s
\end{aligned}$$


#### Computational Order

For $i=1, \ldots, n-1$, then for each vertex $v$.


### Run Time

$O(n^2)$ entries in the DP table, $O(n)$ table to compute each entry due to $\min _{u\ne v}$.

Total $O(n^3)$.


:::{admonition,note} If there are negative cycles

If there are negative cycles, then the Bellman-Ford algorithm still works but it will not stop in $n-1$ iterations.

It can also be used to detect negative cycle by checking whether $T_{n,v} = T_{n+1,v}$, and can be used to find the negative cycles.

:::
