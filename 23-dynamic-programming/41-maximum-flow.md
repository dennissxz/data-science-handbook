# Maximum Flow

Widely used in algorithms on graphs.

## Problem

Send something from a place to another over some networks, e.g. information, package, oil.

Each edge has capacity limit.

Input
: directed graph $G=(V,E)$, capacities $c(e)\ge 0$, source vertex $s\in V$, destination vertex $t \in V$.

## Analysis

Assume

- all capacities are integers (capacity is a finite number. if not integer, scale.)
- no edges enter $s$ or leave $t$ (makes no sense to use those edges)
- call all edges entering $v$ by $\delta ^- (v)$
- call all edges leaving $v$ by $\delta ^+ (v)$

Definition (Flow)
: A function $f:E \rightarrow \mathbb{R}$ which assign amount of flow $f(e)$ to every edge $e \in E$, subject to

- capacity constraints: edge flow less than edge capacity

  $$\forall e: \quad 0\le f(e) \le c(e)$$

- flow conservation constraint: in-flow = out-flow for all intermediate nodes.

  $$\forall v \in V \backslash \left\{ s,t \right\}: \quad \sum_ {e\in \delta^- (v)} f(e) = \sum_ {e\in \delta^+ (v)} f(e) \quad$$

  or

  $$\quad f^{in}(v) = f^{out}(v)$$

Definition (Value of flow)
: The value of a flow is the amount of out-flow from source node $s$ (assuming no in-flow to $s$).

$$
val(f) = f^{out}(s) = \sum_ {e\in \delta^+ (s)} f(e)
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
val(f ^\prime ) = val(f)
$$


Assumption

- Input $G$ has no anti-parallel edges. If there is, say $e = (u,v)$ and $e ^\prime  = (v,u)$, then add a node $x$ in-between $e = (u,v)$ and the new edges have the same capacity$c((u,x)) = c((x,v)) = c((u,v))$.


## Greedy Algorithms

Given a simple $s-t$ path $P$ and a flow $f$, how much available capacity left?

$$
\Delta(p) = \min _{e \in E(P)} \left\{ c(e) - f(e) \right\}
$$

- Start: $\forall e \in E, f(e) = 0$

- While there is a simple $s-t$ path $p$ with $\Delta(p)>0$, set for every $e \in E(P)$.

$$
f(e)\leftarrow f(e) + \Delta(p)
$$

This gives a feasible solution. Optimal? No, depends on the order of $s-t$ path in WHILE.

## Ford-Fulkerson Algorithms

Definition (Residual Flow Network)
: Given a graph and a feasible flow in $G$, let $G_f$ be a new graph with the same vertices but new edges. For every $e(u,v) \in G$,

- add forward edge

    $$
    c_f (u,v) = c(e) - f(e)
    $$

- and a backward edge (v,u)

    $$
    c_f (v,u) = f(e)
    $$

Delete from $G_f$ every edge $e$ with $c_f (e) = 0$

d
