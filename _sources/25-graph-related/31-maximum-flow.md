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

Given a simple $s-t$ path $P$ and a flow $f$, how much available capacity left?

$$
\Delta(P) = \min _{e \in E(P)} \left\{ c(e) - f(e) \right\}
$$

- Start: $\forall e \in E, f(e) = 0$

- While there is a simple $s-t$ path $P$ with $\Delta(P)>0$, set for every $e \in E(P)$.

$$
f(e)\leftarrow f(e) + \Delta(p)
$$

This gives a feasible solution. Optimal? No, depends on the order of $s-t$ path in `WHILE`.

:::

## Solution

We first make an additional assumption and define residual flow networks.


```{margin}
This assumption is not necessary for the algorithm to run, but make the algorithm easier to understand.
```

Assumption (No anti-parallel edges)
: Input $G$ has no anti-parallel edges. If there is, say $e = (u,v)$ and $e ^\prime  = (v,u)$, then add a node $x$ in-between $e = (u,v)$ and the new edges have the same capacity$c((u,x)) = c((x,v)) = c((u,v))$.

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


Ford-Fulkerson algorithm is an iterative algorithm. In each iteration, we compute the residual floe network of the current graph and use that to improve the original graph. Note that flow $f()$ only exists in the original graph.

---
**Ford-Fulkerson's algorithm**

---
- Start:

    - For all edge $e \in E(G)$, initialize zero flow $f(e)=0$.

    - Compute the residual flow network $G_f$.

```{margin}
An augmenting path in residual graph can be found using DFS or BFS.
```

- While $\exists$ a simple path $s-t$ path $P$ in the residual flow network $G_f$, we "push" flow along this path (aka *augmenting path*).

  - Compute the smallest residual capacity along that path

      $$\Delta \leftarrow \min _{e \in E(P)} \left\{ c_f(e) \right\}$$

  - For each edge $e=(u,v) \in P$, check whether it is a forward edge or a backward edge w.r.t. the original graph.

    - If $e=(u,v)$ is a forward edge, then $(u,v) \in G$, and we **increase** the flow of that edge by $\Delta$,

        $$f(u,v)\leftarrow f(u,v) + \Delta$$

    - Else, $e=(u,v)$ is a backward edge, then $(v,u) \in G$, and we **decrease** the flow of $(v,u)$ by $\Delta$,

        $$f(v,u) \leftarrow f(v,u) - \Delta$$

  - Re-compute $G_f$.
---


## Correctness

### Feasibility

Claim 1 (Stops)
: The FF algorithm stops after at most $\sum_{v\in \operatorname{succ}(s)} c(s, v)$ iterations.

***Proof***

Upon initialization, $f(e)=0$ are integers. In each iteration, in $G_f$, all residual capacities $c_f(e)$ are integers and at least 1. The smallest residual capacity is also integer and at least 1. So the update flow of each edge in $G$ is $f(e)\leftarrow f(e) \pm \Delta$.

Note that along the augmenting path $P$, the first edge out from $s$ is always a forward edge (by assumption). As a result, the updated flow of that edge is $f(e) \mathrel{+}= \Delta$. Hence, the updated value of the flow is $\operatorname{val}(f) \mathrel{+}= \Delta$, which is at least 1.

Therefore, the algorithm stops after at most $\sum_{v\in \operatorname{succ}(s)} c(s, v)$ iterations.

$\square$

Claim 2 (Always a valid flow)
: Flow $f$ always remains a valid flow.

***Proof by Induction***

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

  $$
  f^{\text{in}}(S) = \sum_{u\notin S, v \in S} f(u, v) \\
  f^{\text{out}}(S) = \sum_{u\in S, v \notin S} f(u, v) \\
  $$


Definition ($s-t$ cut)
: An $s-t$ cut $(A,B)$ is a cut in $G$ such that the source node $s\in A$ and the destination node $t\in B$. The in- and out-flow of $A$ and $B$ have the relations

  $$
  f^{\text{in}}(A) = f^{\text{out}}(B) \\
  f^{\text{out}}(A) = f^{\text{in}}(B)
  $$


Definition (Capacity of an $s-t$ cut)
: The capacity of an $s-t$ cut $(A,B)$ is defined as the sum of capacities of the edges from $A$ to $B$

  $$
  c(A,B) = \sum _{u\in A, v \in B} c(u, v)
  $$

Property (Compute flow value from a cut)
: Let $f$ be any flow in $G$, recall that the definition of flow value $\operatorname{val}(f)=f^{\text{out}}(s)$. For any $s-t$ cut $c(A,B)$ in $G$, the value of the flow $f$ can be computed as

  $$
  \operatorname{val}(f) = f^{\text{out}}(A) - f^{\text{in}}(A)
  $$


***Proof***

$$\begin{aligned}
\operatorname{val}(f)
&= \sum_{u \in A} \left[ f^{\text{out}}(u) - f^{\text{in}}(u) \right] \\
&= \sum_{u \in A} \left[ \sum_{v} f(u,v) - \sum_{v} f(v,u) \right] \\\\
&= \sum_{u \in A, v \in A}  f(u,v) + \sum_{u \in A, v \in B}  f(u,v) - \sum_{u \in A, v \in A} f(v,u) - \sum_{u \in A, v \in B} f(v,u)  \\\\
&= \sum_{u \in A, v \in B}  f(u,v) - \sum_{u \in A, v \in B} f(v,u)  \\\\
&= f^{\text{out}}(A)- f^{\text{in}}(A)  \\\\
\end{aligned}$$

$\square$


**Corollary**

1. $\operatorname{val}(f) = f^{\text{in}}(B) - f^{\text{out}}(B)$

1. $\operatorname{val}(f) = f^{\text{in}}(t)$

1. $\operatorname{val}(f) \le c(A,B)$, with equality iff $f^{\text{in}}(A) = 0$ and $f^{\text{out}}(A) = c(A,B)$.


Theorem
: If $f$ is any $s-t$ flow and $(A,B)$ is any $s-t$ cut, and $val(f) = c(A,B)$, then $f$ is a maximum flow, by Corollary 3.

How about existence?

Claim (Optimality)
: If $f$ is the flow returned by FF algorithm, then there exists an $s-t$ cut $(A,B)$ such that $val(f) = c(A,B)$. So $f$ is optimal by the above theorem.

***Proof***

Recall that FF algorithm stops if there is no $s-t$ path. After it stops, consider a cut $(A,B)$ in $G_f$, where $A$ is the set of all vertices $v \in V$ such that there is an $s-v$ path in $G_f$, and all other vertices (e.g., $t$) are in $B$. By this definition, there is no edge from $A$ to $B$.

Now, for the cut $(A,B)$ in $G$, we want to prove

$$
val(f) = c(A,B)
$$

By Corollary 3, this holds iff $f^{\text{in}}(A) = 0$ and $f^{\text{out}}(A) = c(A,B)$. Equivalently,

1. $\forall e^- \in \delta^-(A), f(e^-) = 0 \\$

1. $\forall e^+ \in \delta^+(A), f(e^+) = c(e^+)$

These two conditions are indeed satisfied when FF algorithm stops.

1. If there exists $e^- = (u,v): f(e^-) > 0$, then there is an edge $(v,u)$ from $A$ to $B$ in $G_f$ with residual capacity $c_f(v,u) > 0$, contradiction to the property of $G_f$.

1. If there exists $e^+ = (u,v): f(e^+) < c(e^+)$, then there is an forward edge $(u,v)$ from $A$ to $B$ in $G_f$ with residual capacity $c_f(v,u) = c(e^+) - f(e^+)$, contradiction to the property of $G_f$.

$\square$

## Minimum Cut


### Problem

**Input**

- A directed graph $G=(V,E)$.
- Capacity $c(e)$.
- Two special vertices $s$ and $t$.

**Goal**

Find an $s-t$ cut $(A,B)$ that minimizes cut capacity $c(A,B)$, called minimum cut.

### Analysis

Theorem (Equivalency of maximum flow and minimum cut)
: In any flow network $G$, the value of a maximum $s-t$ flow is equal to value of a minimum $s-t$ cut.

The proof is simply from the Corollary.

Thus, FF algorithm also gives an algorithm for finding a minimum $s-t$ cut.

## Complexity

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

Recall different running time??

- strong-polynomial tim#e: $Poly(\text{size of the combinatorial part})$, e.g. $O(n)$
- weak-polynomial time: $Poly(\text{sizes of both parts})$, e.g. $O(mn \times m)$
- pseudo-polynomial time: $Poly(\text{the largest integer present in the input})$, e.g. $O(c_\max)$

:::

## Improvement

We want to bound the number of iterations in Edmonds-Korp algorithm.

To find an augmenting path, use the shortest path $s-t$ in $G$.

Partition the algorithm's execution into phases. Number of phase lasts as low as the lengths of augmenting paths chosen in each iteration remain the same.

Number of iteration is $O(mn)$. Total run time is $O(m^2 n)$.

Observation: Let $G_f$ be the residual graph at the start of iteration $i$, and $G_f ^\prime$ be the residual path at the end of iteration $i$,.

- if $e \in E(G_f) \backslash E(G_f ^\prime)$ then $e \in E(P)$ where $P$ is the augmenting path in iteration $i$.
- $\left\vert E(G_f) \backslash E(G_f ^\prime) \right\vert \ge 1$.
- if $e = (u,v) \in E(G_f) \backslash E(G_f ^\prime)$ then $e \in E(P)$.

Claim
:


## Exercise

http://www.cim.mcgill.ca/~langer/251/E11-networkflow-2.pdf

http://web.stanford.edu/class/archive/cs/cs161/cs161.1176/maxflow_problems.pdf

https://courses.engr.illinois.edu/cs573/fa2012/hw/files/hw_3_pract.pdf
