# Maximum Flow

Widely used in algorithms on graphs.

## Problem

Send something from a place to another over some networks, e.g. information, package, oil.

Each edge has capacity limit.

Input
: directed graph $G=(V,E)$, capacities $c(e)\ge 0$, source vertex $s\in V$, destination vertex $t \in V$.

Goal
: find the maximum volume flowing from $s$ to $t$ subject to the capacity constraint.


## Analysis

Assume

- all capacities are integers (capacity is a finite number. if not integer, scale.)

- no edges enter $s$ or leave $t$ (makes no sense to use those edges)

- call all edges entering $v$ by $\delta ^- (v)$

- call all edges leaving $v$ by $\delta ^+ (v)$

```{margin}
If the two vertices of an edge are $e=(u,v)$, sometimes we write $f(u,v)$ for convenience. If there is no edge between two vertices $u$ and $v$, then $f(u,v)=0$.
```

Definition (Flow)
: A function $f:E \rightarrow \mathbb{R}$ which assign a value $f(e)$ to every edge $e \in E$, subject to

- capacity constraints: edge flow less than edge capacity

  $$\forall e: \quad 0\le f(e) \le c(e)$$

- flow conservation constraint: in-flow = out-flow for all intermediate nodes.

  $$\forall v \in V \backslash \left\{ s,t \right\}: \quad \sum_ {e\in \delta^- (v)} f(e) = \sum_ {e\in \delta^+ (v)} f(e) \quad$$

  or

  $$f^{\text{in}}(v) = f^{\text{out}}(v)$$

Definition (Value of flow)
: The value of a flow is the amount of out-flow from source node $s$ (assuming no in-flow to $s$).

  $$
  \operatorname{val}(f) = f^{\text{out}}(s) = \sum_ {e\in \delta^+ (s)} f(e)
  $$


Observation (Flow cancelation operation on anti-parallel edges)
: For two anti-parallel edges $e = (u,v)$ and $e ^\prime  = (v,u)$. Suppose $f(e), f(e ^\prime ) >0$.

  Let $\Delta = \min \left\{ f(e), f(e ^\prime ) \right\}$ and assign new flow

  $$
  f ^\prime (e) = f(e) - \Delta \\
  f ^\prime (e ^\prime ) = f(e ^\prime ) - \Delta
  $$

  Then the new flow $f ^\prime$ is still a feasible flow with the same value of flow $f$.

  $$
  \operatorname{val}(f ^\prime ) = \operatorname{val}(f)
  $$



:::{admonition,dropdown,note} Greedy algorithms not optimal

Given a simple $s$-$t$ path $P$ and a flow $f$, how much available capacity left?

$$
\Delta(P) = \min _{e \in E(P)} \left\{ c(e) - f(e) \right\}
$$

- Start: $\forall e \in E, f(e) = 0$

- While there is a simple $s$-$t$ path $P$ with $\Delta(P)>0$, set for every $e \in E(P)$.

$$
f(e)\leftarrow f(e) + \Delta(p)
$$

This gives a feasible solution. Optimal? No, depends on the order of $s$-$t$ path in `WHILE`.

:::

## Algorithm


### Algorithm

We first make an additional assumption and define residual flow networks.


```{margin}
This assumption is not necessary for the algorithm to run, but make the algorithm easier to understand.
```

Assumption (No anti-parallel edges)
: Input $G$ has no anti-parallel edges. If there is, say $e = (u,v)$ and $e ^\prime  = (v,u)$, then add a node $x$ in-between $e = (u,v)$ and the new edges have the same capacity $c(u,x) = c(x,v) = c(u,v)$.

Definition (Residual Flow Network)
: Given a graph $G$ and a feasible flow $f$ in $G$, let $G_f$ be a new graph with the same vertices but new edges.

```{margin}
One can check if the residual capacity assigned to a pair of forward edge and backward edge in the residual flow network are correct by checking their sum of residual capacity $c_f(e) + c_f(e ^\prime)$, which should be $c(e)$.
```

  - For every $e(u,v) \in G$, add edges and assign capacity (called **residual capacity**)

    - add a forward edge to reflect **unused** capacity of $e$

        $$
        c_f (u,v) = c(e) - f(e)
        $$

    - add a backward edge $(v,u)$ to reflect **used** capacity

        $$
        c_f (v,u) = f(e)
        $$

  - For every edge $e ^\prime$ in $G_f$ with zero residual capacity $c_f (e ^\prime ) = 0$, delete.

:::{figure} max-flow-three-edges
<img src="../imgs/max-flow-three-edges.png" width = "40%" alt=""/>

Three kinds of edge in $G$ [Chuzhoy 2021]
:::


Ford-Fulkerson algorithm is an iterative algorithm. In each iteration, we compute the residual floe network of the current graph and use that to improve the original graph. Note that flow $f()$ only exists in the original graph.

---
**Ford-Fulkerson's algorithm**

---
- Start:

    - For all edge $e \in E(G)$, initialize zero flow $f(e)=0$.

    - Compute the residual flow network $G_f$. For every $e(u,v) \in G$, add edges and assign capacity (called **residual capacity**)

      - add a forward edge to reflect **unused** capacity of $e$

          $$
          c_f (u,v) = c(e) - f(e)
          $$

      - add a backward edge $(v,u)$ to reflect **used** capacity

          $$
          c_f (v,u) = f(e)
          $$

      - delete edges with zero residual capacity

```{margin}
An augmenting path in residual graph can be found using DFS or BFS.
```

- While $\exists$ a simple path $s$-$t$ path $P$ in the residual flow network $G_f$, we "push" flow along this path (aka *augmenting path*).

  - Compute the smallest residual capacity along that path

      $$\Delta \leftarrow \min _{e \in E(P)} \left\{ c_f(e) \right\}$$

  - For each edge $e=(u,v) \in P$, check whether it is a forward edge or a backward edge w.r.t. the original graph.

    - If $e=(u,v)$ is a forward edge, then $(u,v) \in G$, and we **increase** the flow of that edge by $\Delta$,

        $$f(u,v)\leftarrow f(u,v) + \Delta$$

    - Else, $e=(u,v)$ is a backward edge, then $(v,u) \in G$, and we **decrease** the flow of $(v,u)$ by $\Delta$,

        $$f(v,u) \leftarrow f(v,u) - \Delta$$

  - Re-compute $G_f$.
---


### Feasibility

Claim (Stops in finit time)
: The FF algorithm stops after at most $\sum_{v\in \operatorname{succ}(s)} c(s, v)$ iterations.

:::{admonition,dropdown,seealso} *Proof*

Upon initialization, $f(e)=0$ are integers. In each iteration, in $G_f$, all residual capacities $c_f(e)$ are integers and at least 1. The smallest residual capacity is also integer and at least 1. So the update flow of each edge in $G$ is $f(e)\leftarrow f(e) \pm \Delta$.

Note that along the augmenting path $P$, the first edge out from $s$ is always a forward edge (by assumption). As a result, the updated flow of that edge is $f(e) \mathrel{+}= \Delta$. Hence, the updated value of the flow is $\operatorname{val}(f) \mathrel{+}= \Delta$, which is at least 1.

Therefore, the algorithm stops after at most $\sum_{v\in \operatorname{succ}(s)} c(s, v)$ iterations.

$\square$

:::


Claim 2 (Always a valid flow)
: Flow $f$ always remains a valid flow. That is, the flow always satisfies the capacity constraints and the conservation constraints.

:::{admonition,dropdown,seealso} *Proof by induction*

- Base: $\forall e: f(e)=0$ at initialization is a valid flow

- Step: if at the beginning of an iteration, $f$ is valid, then after the iteration, it remains valid.

We now prove the capacity constraints and the conservation of flow constraints.

**Capacity constraints**

For every edge $e=(u,v)$ in the augmenting path $P$,

  - If $e$ is a forward edge, $e \in G$, then we increases $f(e) \mathrel{+}= \Delta$. By definition, $\Delta \le c_f (e) =  c(e) - f(e)$. Hence $f(e)+\Delta \le c(e)$.

  - If $e$ is a backward edge, then $e ^\prime = (v,u)\in G$, we decrease $f(e ^\prime) \mathrel{-}= \Delta$. Will it be negative? No, since by definition, $\Delta \le c_f(e) = f(e ^\prime)$, so $f(e ^\prime) - \Delta \ge 0$.

$\square$

**Conservation of flow constraints**

By definition, we want to prove $\forall v \in V \backslash \left\{ s,t \right\}$

$$f^{\text{in}}(v) = f^{\text{out}}(v)$$

Suppose along $P$ the two edges of $v$ are $e_1, e_2$

$$ \text{along } P: \quad \ldots \overset{e_1}{\longrightarrow} v \overset{e_2}{\longrightarrow} \ldots$$

We have the following 3 possible situations for $e_1$ and $e_2$,

1.  both forward edges, then we increase both $f(e_1)$ and $f(e_2)$ by $\Delta$, so the in- and out-flow of $v$ both increases by $\Delta$.

    $$ \text{in } G: \quad \ldots \overset{e_1}{\longrightarrow} v \overset{e_2}{\longrightarrow} \ldots$$

1. both backward edges, then we decrease $f(e_1 ^\prime)$ and $f(e_2 ^\prime)$ by $\Delta$, so the in- and out-flow of $v$ both decreases by $\Delta$.

    $$\text{in } G: \quad \ldots \overset{e_1 ^\prime }{\longleftarrow} v \overset{e_2 ^\prime }{\longleftarrow} \ldots$$

1. $e_1$ forward, $e_2$ backward, then $f(e_1)$ increases by $\Delta$, and $f(e_2 ^\prime)$ decreases by $\Delta$, so the two changes to in-follow of $v$ cancel out. No change to out-flow.

    $$\text{in } G: \quad \ldots \overset{e_1}{\longrightarrow} v \overset{e_2 ^\prime }{\longleftarrow} \ldots$$

1. $e_2$ forward, $e_1$ backward, similar to the case 3.

$\square$

:::


Therefore, we show that after an iteration is completed, the constraints remain to be satisfied, so the feasibility is guaranteed.

### Optimality

To prove optimality of the FF algorithm, we first introduce some definitions.

Recall the definition of in-flow to and out-flow from a node $v$

$$
f^{\text{in}}(v) = \sum_{e \in \delta^{-}(v)} f(e) \\
f^{\text{out}}(v) =\sum_{e \in \delta^{+}(v)} f(e)
$$

We define similar quantities for a set of vertices.

Definition (In- and out-flow of a set of vertices)
: For a set of vertices $S \subseteq V$, ,

  $$\begin{aligned}
  f^{\text{in}}(S) &= \sum_{u\notin S, v \in S} f(u, v) \\
  f^{\text{out}}(S) &= \sum_{u\in S, v \notin S} f(u, v)
  \end{aligned}$$

Definition ($s$-$t$ cut)
: An $s$-$t$ cut $(A,B)$ is a cut in $G$ such that the source node $s\in A$ and the destination node $t\in B$. The in- and out-flow of $A$ and $B$ have the relations

  $$\begin{aligned}
  f^{\text{in}}(A) &= f^{\text{out}}(B) \\
  f^{\text{out}}(A) &= f^{\text{in}}(B)
  \end{aligned}$$


Definition (Capacity of an $s$-$t$ cut)
: The capacity of an $s$-$t$ cut $(A,B)$ is defined as the sum of capacities of the edges from $A$ to $B$

  $$
  c(A,B) = \sum _{u\in A, v \in B} c(u, v)
  $$

Property (Compute flow value from a cut)
: Let $f$ be any flow in $G$, recall that the definition of flow value $\operatorname{val}(f)=f^{\text{out}}(s)$. For any $s$-$t$ cut $c(A,B)$ in $G$, the value of the flow $f$ can be computed as

  $$
  \operatorname{val}(f) = f^{\text{out}}(A) - f^{\text{in}}(A)
  $$

:::{admonition,dropdown,seealso} *Proof*

Let $f(u,v)=0$ if there is no edge between two vertices $u$ and $v$.

$$\begin{aligned}
\operatorname{val}(f)
& = f^{\text{out} }(s) \\
&= \sum_{u \in A} \left[ f^{\text{out}}(u) - f^{\text{in}}(u) \right] \\
&= \sum_{u \in A} \left[ \sum_{v} f(u,v) - \sum_{v} f(v,u) \right] \\\\
&= \sum_{u \in A, v \in A}  f(u,v) + \sum_{u \in A, v \in B}  f(u,v) - \sum_{u \in A, v \in A} f(v,u) - \sum_{u \in A, v \in B} f(v,u)  \\\\
&= \sum_{u \in A, v \in B}  f(u,v) - \sum_{u \in A, v \in B} f(v,u)  \\\\
&= f^{\text{out}}(A)- f^{\text{in}}(A)  \\\\
\end{aligned}$$

$\square$

:::



```{margin}
This corollary is the key for subsequent analysis
```

Corollary
: $\operatorname{val}(f) \le c(A,B)$, with equality iff $f^{\text{in}}(A) = 0$ and $f^{\text{out}}(A) = c(A,B)$.


Theorem
: If $f$ is any $s$-$t$ flow and $(A,B)$ is any $s$-$t$ cut, and $\operatorname{val}(f) = c(A,B)$, then $f$ is a maximum flow, by Corollary.

How about existence?

Claim (Optimality)
: If $f$ is the flow returned by FF algorithm, then there exists an $s$-$t$ cut $(A,B)$ such that $\operatorname{val}(f) = c(A,B)$. So $f$ is optimal by the above theorem.

:::{admonition,dropdown,seealso} *Proof*

Recall that FF algorithm stops if there is no $s$-$t$ path. After it stops, consider a cut $(A,B)$ in $G_f$, where $A$ is the set of all vertices $v \in V$ such that there is an $s-v$ path in $G_f$, and all other vertices (e.g., $t$) are in $B$. By this definition, there is no edge from $A$ to $B$.

Now, for the cut $(A,B)$ in $G$, we want to prove

$$
\operatorname{val}(f) = c(A,B)
$$

By Corollary, this holds iff $f^{\text{in}}(A) = 0$ and $f^{\text{out}}(A) = c(A,B)$. Equivalently,

1. $\forall e^+ \in \delta^+(A), f(e^+) = c(e^+)$

1. $\forall e^- \in \delta^-(A), f(e^-) = 0 \\$

These two conditions are indeed satisfied when FF algorithm stops.

1. If there exists $e^+ = (a,b): f(e^+) < c(e^+)$, then there is an forward edge $(a,b)$ from $A$ to $B$ in $G_f$ with residual capacity $c_f(a,b) = c(e^+) - f(e^+)>0$, contradiction to the stoping condition of $G_f$.

1. If there exists $e^- = (b,a): f(e^-) > 0$, then there is an edge $(a,b)$ from $A$ to $B$ in $G_f$ with residual capacity $c_f(a,b) > 0$, contradiction to the stoping condition of $G_f$.

$\square$

:::


### Complexity

Let $m$ be the number of edges in the graph $G$.

Recall that there are at most $\sum_{v\in \operatorname{succ}(s)} c(s, v)$ iterations. The bound is upper bounded by $n \times c _{\max}(e)$.

 In each iteration

- Finding augmenting path $P$ takes $O(m)$
- Pushing flow along $P$ takes $O(m)$
- Recompute $G_f$ takes $O(m)$

So the total running time is $O(m\cdot n\cdot c _{\max})$


:::{admonition,note} Is FF algorithm efficient?

There are two inputs.

- Graph, which is a combinatorial part of size $(n,m)$
- Capacities, which is a numerical part of size $m$

Recall different running time

- strong-polynomial time: $Poly(\text{input size of the combinatorial part})$, e.g. $O(n)$
- weak-polynomial time: $Poly(\text{input sizes of both parts})$, e.g. $O(n \log c_\max)$
- pseudo-polynomial time: $Poly(\text{the largest integer present in the input})$, e.g. $O(c_\max)$

:::


### Improvement: Edmonds-Karp Algorithm

Instead of using an arbitrary augmenting path, we use the **shortest** path $s$-$t$ in $G$ that minimizes number of edges. This work takes $O(m)$ by BFS or DFS, so each iteration still takes $O(m)$. But it reduces the number of iterations from $O(nc_\max)$ to $O(nm)$, this leads to the Edmonds-Karp algorithm with complexity $O(nm^2)$.

To show that, we first run that algorithm, record the length of the chosen shortest path in each iteration, and then partition these the execution into phases, where each phase lasts as long as the lengths of augmenting paths chosen remains the same.


$$\begin{aligned}
\text{iteration} &\quad 1 \quad2 \ \quad 3 \  \quad4 \ \quad 5 \quad6 \quad 7\\
\text{shortest path length} &\quad \underbrace{2 \quad 2}_{\text{phase 1} } \quad \underbrace{3\quad 3}_{\text{phase 2} } \quad \underbrace{5 \quad 5 \quad 5}_{\text{phase 3} }  \\
\end{aligned}$$


Claim (Non-decreasing shortest path length)
: From iteration to iteration, the length of the augmenting path is non-decreasing. Hence, the number of phases is at most $n$.

Claim ($O(m)$ iterations in each phase)
: Every phase covers at most $O(m)$ iterations.

::::{admonition,dropdown,seealso} *Proof*

To prove them, let $G_f$ be the residual graph at the *start* of iteration $i$, and $G_f ^\prime$ be the residual path at the *end* of iteration $i$, and $P$ be the augmenting path in iteration $i$. From the algorithm we observe that

- if $e \in E(G_f)$ but $e \notin E(G_f ^\prime)$, then $e \in E(P)$
- at least one edge $e\in P$ has to disappear in $G ^\prime _f$
- if $e \notin E(G_f)$ but $e \in E(G_f ^\prime)$ then its anti-parallel edge $e ^\prime  \in E(P)$.

Now consider using BFS from to find a shortest path $s$-$t$ in $G_f$. Suppose the path length is $d$, then there are $d+1$ layers. The first layers only contains $s$, and the last layer contains $t$. In each iteration, we delete some forward-looking edge, and add a backward-looking edge or sideways-looking edge, but **no** shortcut edge. So the shortest path is non-decreasing. Beside, there are at most $m$ layers to delete in a phase with path length $d$, so at most $O(m)$ iterations in that phase.

:::{figure} max-flow-bfs
<img src="../imgs/max-flow-bfs.png" width = "100%" alt=""/>

BFS in $G_f$ [Chuzhoy 2021]
:::

$\square$

::::

### Approximation

$(1+\epsilon)$-approximation returns a flow of value at least $\frac{OPT}{1+\epsilon}$.


### Other Properties

Theorem (Integrality of flow)
: If all capacities are integrals, then the Ford-Fulkerson algorithm finds a maximum flow where $f(e)$ is integral for all $e$.



## Minimum Cut

### Problem

**Input**

- A directed graph $G=(V,E)$.
- Capacity $c(e)$.
- Two special vertices $s$ and $t$.

**Goal**

Find an $s$-$t$ cut $(A,B)$ where $s \in A, t \in B$, with minimal cut capacity $c(A,B)$, which is the sum of capacities of edge from $A$ to $B$, $c(A, B) = \sum_{u\in A, v\in B} c(u,v)$.

In other words, we want to remove some edges to **disconnect** $s$ and $t$, and minimize the total capacities of these removed edges. The vertex partition's perspective and edge removal's perspective are equivalent.



### Analysis

Theorem (Equivalency of maximum flow and minimum cut)
: In any flow network $G$, the value of a maximum $s$-$t$ flow is equal to the capacity of a minimum $s$-$t$ cut.

The proof is simply from the Corollary.

Thus, FF algorithm also gives an algorithm for finding a minimum $s$-$t$ cut: after the algorithm stops, in the residual graph $G_f$, find the set of vertices reachable from $s$, then $(A, V\setminus B)$ is a minimum $s$-$t$ cut.





## Extension

$O(m n c_\max)$ is not efficient. There are alternative algorithms to improve this.

### Max-flow and Min-cut in Undirected Graphs

To find maximum $s-t$ flow in undirected graph with capacities $c(e)>0$, we can make the graph directed, and run Ford-Fulkerson algorithm on the directed counterpart.

- Convert every undirected edge to two anti-parallel directed edges with the same capacity as the undirected edge.

  $$u - v \quad \Rightarrow \quad  u \leftrightarrows v$$

- Run the Ford-Fulkerson algorithm for directed graph.
- Finally, run flow cancelation for two anti-parallel edges, such that one of the two anti-parallel edges is reduced to 0.

  $$u \leftrightarrows v \quad \Rightarrow \quad  u \rightarrow v \text{ or } u \leftarrow v$$

The direction of an edge indicates the direction of flow (e.g. pipeline) in the undirected graph. The final flow value are the same, and the integrality of max-flow also holds.

Meanwhile, we can find a minimum cut on a undirected graph, the capacity/cost of the cut is the sum of the capacities of the edges across $A$ and $B$.

$$
\sum _{e \in E(A,B)} c(e)
$$

Likewise, we convert every undirected edge to two anti-parallel directed edges, run Ford-Fulkerson algorithm to find a $s$-$t$ cut $(A,B)$. The cut values are the same.

Since in the directed graph we have max-flow $=$ min-cut, in the undirected graph we have max-flow $=$ min-cut.

### Path-based Flow

```{margin}
Aka flow-paths.
```

Recall the flow $f: V\rightarrow \mathbb{R} _{\ge 0}$ is defined for edges. We can consider a path-based flow $f_{p} : \mathcal{P}\rightarrow \mathbb{R} _{\ge 0}$, where $\mathcal{P}$ is the set of some $s$-$t$ paths. Let $f _{p} (P)$ be the flow of a path $P \in \mathcal{P}$. It is valid if the edge capacity constraint holds.

$$
\forall e:\quad \sum_{P: P \in \mathcal{P} \text{ and } E(P) \ni e} f _{p} (P) \le c(e)
$$

The set $\mathcal{P}$ and the flow assignment $f_p$ together are called a flow-paths solution, with value $\sum_{P \in \mathcal{P}}f _{p} (P)$.

Theorem (Equivalence of edge-flow and path-flow)
: Given an edge-based flow $\left\{ f(e) \right\}_{e \in E}$, we can compute the path-based flow $\left\{ f _{p} (P) \right\}_{P \in \mathcal{P}}$ efficiently, and

  - They have the same flow values $\sum_{e \in \delta^+(s)} f(e)=\sum_{P \in \mathcal{P}}f _{p} (P)$
  - If $f$ is feasible, then , and $f _{p}$ is also feasible.
  - If $f$ is integral, then $f _{p}$ is also integral.

  And the reverse also hold.

Theorem (Flow decomposition)
: Any $s$-$t$ flow can decompose into at most $m$ number of cycles and $s$-$t$ paths.

:::{admonition,dropdown,seealso} *Proof*

We provide an efficient algorithm for such decomposition.

Let $(a,b)$ be an edge with $f(a,b)>0$. Then trace backward from $a$, and trace forward from $b$, along edges with positive flow $f(e)>0$.

$$\cdots a \rightarrow b \cdots$$

Until we reach
- a cycle (along $a$, along $b$, or involve edge $(a,b)$), or
- both $s$ and $t$

Let $W$ be the edges in cycle or $s$-$t$ path. Compute $\Delta = \min_{e \in E(W)} f(e)$. Subtract $\Delta$ from all flow of edges in $W$. The resulted flow is also valid, and the flow value remains unchanged.

We repeat this process until there is no cycle or path. Each iteration reduces flow to one edge to 0, so the number of iterations is at most $m$. Each iteration gives a cycle or $s$-$t$ path with flow $\Delta$. Hence, the number of cycles and $s$-$t$ paths is at most $m$.

Each iteration takes $O(m)$, so total running time is $O(m^2)$.

:::

Note that dropping the cycles does not affect the flow value. So the paths obtained from this algorithm together with their $\Delta$ form a valid flow-paths solution.

### Edge-Disjoint Paths

Ford-Fulkerson algorithm is an efficient algorithm for finding edge-disjoint paths problem in directed graph.

```{margin}
Edge-disjoint path = EDP
```

**Problem**: For a directed graph with two disjoint **sets** of vertices $S$ and $T$, we want to find a maximum-cardinality set $\mathcal{P}$ of $S$-$T$ paths that are edge-disjoint, i.e. no path in $\mathcal{P}$ can share any edges.

To solve this,

1. For every edge $e\in G$, set capacity $c(e)=1$. Add one node $s$ that connects to every vertex $u$ in $S$ with capacity $\infty$. Add one node $t$ that connects to every vertex $v$ in $T$ with capacity $\infty$. Call this flow network $H$.

    $$
    s \overset{\infty}{-} S \cdots T \overset{\infty}{-} t
    $$

1. Run Ford-Fulkerson algorithm on $H$ to obtain a flow $f$. Since $f(e)\le c(e)=1$ and is integral, it is 1.

1. Run flow-path decomposition, then each path also carries path-flow value 1. When we delete the path, we actually remove all edges along the path since $c(e)=1$. Moreover, the subsequent paths must be disjoint with this one since $c(e)=1$. We will get a collection of EDP from $S$ to $T$. The number of paths equals to the flow value.


### $S$-$T$ Cut

**Problem**: Given two sets of vertices $S$ and $T$ in a directed graph $G$, what is the minimum number of edges needed to disconnect $S$ from $T$? Formally, find a minimum-cardinality edge set $E ^\prime \subseteq E$ such that in the remaining graph $G \setminus E ^\prime$, there is **no** path from a vertex of $S$ to a vertex of $T$.

Menger's Theorem
: The maximum number of EDPs connecting $S$ to $T$ is equal to the minimum number of edges needed to disconnect $S$ from $T$.

The same can be done for undirected graphs.

### Vertex-capacitated Max Flow

In reality, capacities are often defined on vertices, such as computer networks. Each vertex has capacity constraint $c(v)$. The capacity constraints are on vertices: the total inflow into any vertex $v$ is at most $c(v)$. The conservation becomes: for each vertex, outflow = inflow. How to find a maximum flow?

We reduce this problem to the usual max flow problem buy convert an vertex-capacitated max flow problem instance $I_V$ into an edge-capacitated problem instance $I_E$, and show that we can solve $I_V$ by solving $I_E$.

Assign infinite capacity to all edges. Convert each vertex to two vertices connected by an edge, with edge weight $c(e) = c(v)$. Equivalent.


Vertex-disjoint path problem: find maximum number of vertex-disjoint paths (no two paths share vertices) connecting $S$ to $T$.

Recall Menger’s Theorem:
- The maximum number of EDPs connecting $S$ to $T$ is equal to the minimum number of edges needed to disconnect $S$ from $T$.

The corresponding version in this setting is:
- The maximum number of **VDPs** connecting $S$ to $T$ is equal to the minimum number of **vertices** needed to disconnect $S$ from $T$.


## Applications


### Image Segmentation

An image can be viewed as a vertex. We want to partition an image into two parts, e.g. foreground and background. For pixel/vertex $s$, let $a_v$ be how likely $v$ to be in a part, and $b_v$ be how likely $v$ to be in the other part.

To solve this, we define strength/similarity for every pair of pixels $s_{u,v}$. The ultimate task is to partition the pixels into two sets $X$ and $Y$. The similarity of two pixels from different partition should be small. The objective is

$$
\max \left\{ \sum_{v \in X} a_v  + \sum_{u \in Y} b_u  - \sum_{v \in X, u\in Y} s_{v,u}  \right\}
$$

which is equivalent to

$$
\min \left\{ \sum_{v \in X, u\in Y} P_{v,u}  - \sum_{v \in X} a_v  - \sum_{u \in Y} b_u  \right\}
$$

which is equivalent to

$$
\min \left\{ \sum_{v \in X, u\in Y} s_{v,u}  - \sum_{v \in X} a_v  - \sum_{u \in Y} b_u + \sum_{w \in V} (a_w + b_w) \right\}
$$

which is

$$
\min \left\{ \sum_{v \in X, u\in Y} s_{v,u}  + \sum_{v \in Y} a_v  + \sum_{u \in X} b_u \right\}
$$

We can solve this with minimum cut on undirected graph. The capacity of an edge is the strength of that edge. For every vertex $v$, add edge $(s,v)$ of capacity $a_v$, and edge $(v, t)$ of capacity $b_v$. Also for add edge $e(v,u)$ of capacity $s_{v,u}$ for $u,v \ne s,t$. Consider an $s$-$t$ cut $(A,B)$, denote $X = A \backslash \left\{ s \right\}$ and $Y = B \backslash \left\{ t \right\}$.

Claim
: The capacity of the cut equals the value of the objective function. So the optimization problem in image segmentation can be solved by the minimum cut problem.

$$
c(A,B) = \sum _{e \in E(A,B)} c(e) = f(X,Y)
$$

***Proof***

There are 3 kinds of across-set edges in $E(A, B)$

1. $e=(u,v), u\ne s, v\ne t$, contribute $s_e$
2. $e=(s,x), x\in B \backslash \left\{ t \right\}$ with edge capacity $a_x$. Total contribute $\sum_{x \in Y} a_x$
1. $e=(y,t), y\in A \backslash \left\{ s \right\}$ with edge capacity $a_x$. Total contribute $\sum_{b \in X} b_y$

Hence

$$
c(A,B) = \sum_{v \in X, u\in Y} s_{v,u}  + \sum_{v \in Y} a_v  + \sum_{u \in X} b_u
$$

which is exactly the objective function.

$\square$



## Exercise

Let $G$ be an arbitrary (directed) flow network with integral edge capacities



1. [**Change of capacity**] Prove or disapprove:

    1. Let $e=(u,v)$ be an edge of $G$ with capacity $c(e) ≥ 1$. **Decreasing** the capacity of an edge $e$ by $1$ decreases the maximum flow value by at most $1$.

    2. Let $e=(u,v)$ be an edge of $G$ with capacity $c(e) ≥ k$ where $k$ is a positive integer. **Decreasing** the capacity of an edge $e$ by $k$ decreases the maximum flow value by at most $k$.

    3. **Increasing** the capacity of an edge $e$ by $1$ increases the maximum flow value by at most $1$.

    4. **Increasing** the capacity of an edge $e$ by a positive integer $k$ increases  the maximum flow value by at most $k$.

    5. Let $(A,B)$ be a minimum $s$-$t$ cut in G. If we **increase** the capacity of **each** edge in $E(G)$ by $1$, then $(A,B)$ remains a minimum $s$-$t$ cut in the new flow network.

    :::{admonition,dropdown,seealso} *Solution*

    1. True.
        - If $e$ is in some minimum cut, then decreasing the capacity of $e$ decreases the min cut value by 1.
        - Else, $e$ is not in every min cut, then decreasing the capacity of $e$ by 1 leaves the min cut value unchanged. Either way, the capacity decreases by at most 1. The claim follows from the max-flow-min-cut theorem.

        Algorithm to update the flow:  

        - If $c(e) \ge f(e) + 1$, then the max flow remains the same.
        - Else $c(e) = f(e)$ (saturated edge), then to satisfy the constraint, we need to remove one unit of flow from $s$ to $t$ that goes through edge $e$. The algorithm is

          - Find a path $s$-$u$ and a path $v$-$t$ that contain only edges of positive flow. Remove $1$ unit of flow for each edge on path $s-u$, and $v-t$. This step takes $O(m)$.
          - Run Ford-Fulkerson. There is at most one iteration since the flow will increases by at most $1$. One iteration takes $O(m)$.

    2. True.
        Decreasing by $k$ is the same as decreasing in $k$ steps of 1. Each such step decreases the max flow by at most 1. So the total decrease is at most $k$.

    3. True.    
        - If edge $e$ is in every min cut, then increasing the capacity of $e$ by $1$ increases the min cut value by $1$.
        - Else, edge $e$ is not in every min cut, then increasing the capacity of $e$ by $1$ leaves the min cut value unchanged.

    4. True. Increasing by $k$ is the same as increasing in $k$ steps of 1. By the previous solution, each such step increases the max flow by at most $1$. So the total increase is at most $k$.

        Algorithm to update the flow

        - Repeat at most $k$ times:
          - Look for an augmenting path in the residual network by BFS/DFS.
          - If there is one, then update the existing flow, else terminate.

        Each pass takes $O(m)$, total $O(km)$.

    5. False. More specifically,
        - If some edge has different capacity, increasing capacity might have resulted in different minimum cut, as the graph below.
        - Else all edges have same capacity then minimum cut would remain same. To see this, if all edges have the same capacity $c$, then the capacity of any cut is $xc$ where $x$ is the number of edges cut. So a min-cut has $x_\min$. After $c$ becomes $c+1$, it is still a min-cut since it has $x_\min$.

        :::{figure} max-flow-ex-1
        <img src="../imgs/max-flow-ex-1.png" width = "30%" alt=""/>

        New min-cut becomes $S$-$A$ with cut capacity $5$.
        :::

    ::::

1. [**Edges in residual graph**] Suppose we are given a flow network $G$, and a valid flow $f$ in that network. Let $G_f$ be the corresponding residual graph, $P$ a shortest $s$-$t$ path in $G_f$, and $f ^\prime$ a new flow obtained from $f$ after performing a single iteration of the Ford-Fulkerson algorithm, with $P$ as the augmenting path. Prove each one of the following statements.

    1. If $e \in G_f$ but $e \notin G _{f ^\prime}$, then $e\in P$.

    2. At least one edge of $P$ does not belong to $G_{f ^\prime}$.

    3. If $e=(u,v) \in G_{f ^\prime }$ but $e \notin G_f$, then edge $(v,u)$ belongs to path $P$.

    :::{admonition,dropdown,seealso} *Solution*

    1. - If $e$ is a forward edge in $G_f$, it disappears in $G_{f ^\prime }$ iff we increase the flow $f(e) < c(e)$ to $c(e)$ in $G$, such that in $G_{f ^\prime }: c_{f ^\prime }(e) = c(e)-f(e) = 0$ and hence $e$ disappears. According to the algorithm, if we increases $f(e)$ in $G$, then $e\in P$ in that iteration.
       - If $e$ is a backward edge in $G_f$, it disappears in $G_{f ^\prime }$ iff we decrease the flow $f(e ^\prime)>0$ to 0, such that in $G_{f ^\prime }: c_{f ^\prime }(e) = f(e) = 0$ and hence $e$ disappears. According to the algorithm, if we decrease $f(e ^\prime )$ in $G$, then $e\in P$ in that iteration.


    2. Since we increase of decrease $\Delta = \min _{e \in E(P)}\left\{c_{f}(e)\right\}$ for all corresponding edges in $G$, one edge's flow $f(e)$ must be increased to $c(e)$ or decreased to $0$.

        - If $f(e)$ is increased to $c(e)$, then the forward edge $e$ disappears in $G_{f ^\prime }$ since $c_{f ^\prime } (e) = c(e) - f(e)$ changes from positive to $0$.
        - If $f(e)$ is decreased to $0$, then the backward edge $e ^\prime$ disappears in $G_{f ^\prime }$ since $c_{f ^\prime } (e ^\prime ) = f(e)$ changes from positive to $0$.

    3. - If $e$ is a forward edge, then we must decrease $f(e)$ from $c(e)$ in $G$. According to the algorithm, if we decrease $f(e)$ in $G$, then the backward edge $e ^\prime  = (v,u)\in P$ in that iteration.
       - If $e$ is a backward edge, then we must increase $f(e ^\prime)$ from $0$ in $G$. According to the algorithm, if we increase $f(e ^\prime)$ in $G$, then the forward edge $e ^\prime  = (v,u)\in P$ in that iteration.

    :::


1. T/F: If $f$ is a valid $s$-$t$ flow in graph $G$ of value $v_f$, and $f ^\prime$ is a valid $s$-$t$ flow in the residual graph $G_f$ of value $v(f ^\prime)$, then there is a valid $s$-$t$ flow in graph G of value $v(f) + v(f ^\prime)$.

    :::{admonition,dropdown,seealso} *Solution*

    True. Moreover, let $v (f_\max)$ be the value of a max-flow in $G$ and $v (f ^\prime _\max)$ be the value of a max-flow in residual graph $G_f$, then we have $v(f) + v(f ^\prime) \le v (f_\max)$ with equality iff $v(f ^\prime) = v(f ^\prime _\max)$.

    :::



1. [**Acyclic flow**] Show an efficient algorithm to find a maximum $s$-$t$ flow in $G$ which is integral and acyclic.

    :::{admonition,dropdown,seealso} *Solution*
    Algorithm (Eliminating flow cycles)

    1. Use Edmonds-Karp algorithm to compute a maximum flow $f$
    2. Delete every edge $e$ with $f(e)=0$ in $G$ and obtain a new flow network $H$.
    3. While True:
        - Find a simple cycle $C$ in $H$, or correctly establishes that no such cycle exists. To to so, for each vertex $v \in V(H)$, perform BFS from $v$. If BFS ever discovers a vertex $u$, such that an edge $(u,v)\in E(H)$, then we can use the BFS to compute a simple path connecting $v$ to $u$ in $H$, which, together with edge $(u,v)$ provides a simple cycle in $C$ containing $v$.
          - If the BFS never discovers a vertex $u$ for which edge $(u,v)$ exists, then no simple cycle in $H$ may contain $v$. Terminate.
        - Otherwise, $C$ exists. Select an edge $e\in E(C)$ that carries the smallest amount of flow among all edges in $C$; denote this flow amount by $\Delta$. We then decrease the flow on every edge of $C$ by $\Delta$, and recompute $H$. This is guaranteed to produce a feasible flow, whose value is exactly the same as the value of the original flow (since vertex $s$ may not lie on cycle $C$).

    Running Time $O(m^2 n)$

    1. $O(m^2 n)$
    2. $O(m)$
    3. $O(m\cdot mn)$
       - In every iteration of the algorithm, the flow on at least one edge decreases to $0$, and we never increase the flow on any edge. Therefore, the total number of iterations is $O(m)$.
       - Finding a simple cycle takes $O(n\cdot m)$
       - Updating flow and recomputing graph $H$ takes $O(m)$

    :::

1. [**Incoming edges to source and outgoing edge from sink**] Assume now that vertex $s$ may have incoming edges and $t$ may have outgoing edges. Recall that in such a case, the value of a flow $f$ is dened to be $\sum_{e \in \delta^{+}(s)} f(e)-\sum_{e \in \delta^{-}(s)} f(e)$. Prove that there exists a maximum flow $f$, such that $f(e)=0$ for every edge $e \in \delta^{-}(s) \cup \delta^{+}(t)$.

    :::{admonition,dropdown,seealso} *Solution*

    First, we run Edmonds-Karp algorithm to obtain a maximum $s$-$t$ flow $f$.

    In $f$, if there is an edge $e \in \delta^- (s)$ such that $f(e)>0$, then $s$ must be in a cycle that every edge contains positive flow (why??). For each of such cycle $C$, we subtract $\Delta = \min _{e \in C} f(e)$ from all flows of edges in $C$ to eliminate it. At most $m$ iterations, we will have $f(e)=0$ for all $e \in \delta^- (s)$. Note that by the flow conservation constraints, the resulting flow is also a maximum flow and the flow value is unchanged.

    The operation to the outgoing edges from $t$ is similar. Finally, we obtain a required maximum flow.

    $\square$

    An alternative approach is to prove the output graph in the previous question has no positive flow for edges entering $s$ and leaving $t$.

    Note that this proves that in the maximum flow problem, we can assume without loss of generality that no edge enters s and no edge leaves t, as deleting all such edges does not change the maximum flow value.

    :::


### More

http://www.cim.mcgill.ca/~langer/251/E11-networkflow-2.pdf

http://web.stanford.edu/class/archive/cs/cs161/cs161.1176/maxflow_problems.pdf

https://courses.engr.illinois.edu/cs573/fa2012/hw/files/hw_3_pract.pdf

https://www.cs.cornell.edu/courses/cs6820/2016fa/handouts/flows.pdf
