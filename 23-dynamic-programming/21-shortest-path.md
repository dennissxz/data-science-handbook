# Shortest Path


## Problem

Input
: Connected graph with edge length $\ell(e)$.


Goal
: Find the shortest path
  $$
  \ell(p) = \sum _{e \in E(P)} \ell(e)
  $$


# Dijkstra's Algorithm

If all edge length are positive.

Maintain a set of vertices $S$.


$$
\forall v \in S, \ d(v) = dist(s,v) \\
d(v) = d(u) = \ell(u,v)
$$

---

What if there are negative edge weights?

Definition (Negative cycle)
: We call a cycle negative if the sum of the weights of the edges in a cycle is negative.

This leads to a problem

$$
A - negative\ cycle - B
$$

Then the path found by Dijkstra's algorithm goes around the negative cycle for infinite times.

# DP-based Algorithm (Bellman-Fosd)

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

## Algorithm

### DP-table

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

## Run Time

$O(n^2)$ entries in the DP table, $O(n)$ table to compute each entry.

Total $O(n^2)$.

## Correctness

Claim
: For all $i, v$, entry $T_{i,v}$ contains the length of shortest $s-v$ path with less than $i$ vertices.

Base
: $i=0$

Step
: Assume correctness for all vertices below $i$...
