# Linear Programming

## Canonical Form

A linear programming problem consists of three elements

- $n$ variables $x_1, x_2, \ldots, x_n$

- one objective function that is linear in variables

$$
\max/\min \ c_1 x_1 + c_2 x_2 \ldots c_n x_n
$$

- $m$ constraints that are linear in variables, for $j = 1, 2, \ldots, m$

$$a_{j1}x_1 + a_{j2}x_2 \ldots + a_{jn}x_n \ (\ge \text{or}  \le \text{or}  =) \ b_j$$

```{margin} Integer programming
Sometimes we want $x_i$ to have integer value, then it's called integer programming. But we cannot solve this kind of problem efficiently.
```

The canonical form of a linear programming is

$$\begin{aligned}
\max && \sum_{i=1}^n c_i x_i &&& \\
\text{s.t.} && \sum_{i=1}^n a_{ji} x_i &\le b_j &&\forall\ 1 \le j \le m\\
&& x_i &\ge 0 &&\forall\ 1 \le i \le n \\
\end{aligned}$$

Or in matrix form,

$$\begin{aligned}
\max
&& \boldsymbol{c} ^\top \boldsymbol{x} && \\
\text{s.t.}
&&\boldsymbol{A} \boldsymbol{x} &\le \boldsymbol{b} &\\
&& \boldsymbol{x} &\ge \boldsymbol{0} &
\end{aligned}$$

To convert a problem into the canonical form,

1. if it's $\min$, negate the objective function to get $\max$

1. replace $=$ with simultaneous $\ge$ and $\le$.

1. replace $\ge$ with $\le$ by negation.

1. if some variable $x_i$ is unconstrained, replace $x_i$ with by $x_i^+ - x_i^-$, where the two variables $x_i^+,x_i^- \ge 0$.


:::{admonition,note} Geometry's perspective

From geometry's perspective, every constraint is a hyperplane that partitions the space into half-spaces. The intersection of half-spaces defined by the constraints is called a feasible region, usually it is a polyhedron. If the feasible region is bounded, then it's a polytope.

:::


## Algorithm

In every LP, one of the following holds

1. no feasible solution
1. finite optimal solutions
1. optimal solution is unbounded

There are many algorithms to find a solution.

Let $L$ be the maximal coefficients.

Ellipsoid method $O(n^b L)$. Slow but useful.

Interior point method $O(n^{3.5}L)$.

## Examples

### Line Fitting

Input
: $n$ points $\boldsymbol{x}_i \in \mathbb{R}^d$ with label $y_i \in \mathbb{R}$.

```{margin} Mind the notation
Here $\boldsymbol{x}_i$ are constant data, while $a_j$'s and $b$ are variables.
```

Objective
: Find a linear function $h: \mathbb{R} ^d \rightarrow \mathbb{R}$ parameterized by $\boldsymbol{a} \in \mathbb{R}^d$ and $b \in \mathbb{R}$, such that the value $h(\boldsymbol{x}_i) = \boldsymbol{a} ^\top \boldsymbol{x}_i + b$ is close to the label $y_i$.

In other words, we want $h$ to minimize the total error

$$
\sum_{i=1}^n \left\vert h(\boldsymbol{x}_i ) - y_i \right\vert
$$

To convert this problem into the canonical form of LP, we need to get rid of the absolute value operation. We introduce LP variables $\boldsymbol{z}_i$ to represent the error terms.

$$\begin{aligned}
z_i &\ge \boldsymbol{a} ^\top \boldsymbol{x}_i + b - y_i  \\
z_i &\ge - (\boldsymbol{a} ^\top  \boldsymbol{x}_i + b - y_i)
\end{aligned}$$

The objective is to minimize $\sum_{i=1}^n z_i$.

In all, the LP problem in the canonical form is

$$\begin{aligned}
\max
&&- \sum_{i=1}^n z_i && \\
\text{s.t.}
&&  z_i - \boldsymbol{a} ^\top \boldsymbol{x}_i - b &\ge -y_i &\forall\ 1 \le i \le n\\
&&  z_i + \boldsymbol{a} ^\top \boldsymbol{x}_i + b &\ge y_i &\forall\ 1 \le i \le n\\
\end{aligned}$$

with variables $\left\{ \boldsymbol{a} ,b , \boldsymbol{z}  \right\}$.

### Binary Classifier

Now suppose the label is binary.


Input
: $n$ points $\boldsymbol{x}_i \in \mathbb{R}^d$ with label $y_i \in \left\{ -1, 1 \right\}$.

Objective
: Find a linear function $h: \mathbb{R} ^d \rightarrow \mathbb{R}$ parameterized by $\boldsymbol{a} \in \mathbb{R}^d$ and $b \in \mathbb{R}$, such that the value $h(\boldsymbol{x}_i) = \boldsymbol{a} ^\top \boldsymbol{x}_i + b$ has the same sign with $y_i$.

Let $i^+$ be the index of points with positive label $y_i = 1$ and let $i^-$ be the index of points with negative label $y_i = -1$. The objective is

$$\begin{aligned}
h(\boldsymbol{x}_{i^+}) &> 0 &&\forall \ i^+\\
h(\boldsymbol{x}_{i^-}) &< 0 &&\forall \ i^-\\
\end{aligned}$$

#### Separable

We first assume such hyperplane exists, i.e. the two kinds of points are separable.

To convert this to problem into the canonical form, we need to get rid of the $>$ and $<$ signs. We introduce a **slack** variable $\delta$. The new problem is

$$\begin{aligned}
\max &&\delta && \\
\text{s.t.}
&&\  h(\boldsymbol{x}_{i^+}) - \delta &\ge 0 &&\forall \ i^+\\
&&\  h(\boldsymbol{x}_{i^-}) + \delta &\le 0 &&\forall \ i^-\\
\end{aligned}$$

with variables $\left\{ \boldsymbol{a} , b, \delta \right\}$.

Actually the $\max$ does not matter. If we can find a LP solution with $\delta > 0$, then the original $>$ and $<$ conditions must hold.

#### Non-Separable

If there is no hyperplane to perfectly separate the points, we introduce buffer, say $d>0$, such that beyond the buffer region all points are correctly classified. That is, the following conditions always holds

$$\begin{aligned}
i \in \left\{ i^+ \right\} \quad \forall \ i: h(\boldsymbol{x}_{i}) &\ge d \\
i \in \left\{ i^- \right\} \quad \forall \ i: h(\boldsymbol{x}_{i}) &\le -d \\
\end{aligned}$$

Note that if there exists a solution $\left\{ \boldsymbol{a} ,b \right\}$, then we can scale the solution down by $\frac{1}{d}$ so the solution remains valid. So w.l.o.g., the buffer can simply be $d=1$.

Then, we introduce an error term to quantify the mis-classification error. It measures the absolute deviation between the value $h(\boldsymbol{x}_i )$ to the buffer line $(1/-1)$ in its correct region.

- If a data point with index $i^+$ is mis-classified, the absolute deviation is $1-h(\boldsymbol{x}_i)$.
- If a data point with index $i^-$ is mis-classified, the absolute deviation is $h(\boldsymbol{x}_i) - (-1) = 1+ h(\boldsymbol{x}_i)$.

In sum, we have

$$
e_i = \left\{\begin{array}{ll}
1 - h(\boldsymbol{x}_i) & \text { for } i \in \left\{ i^+ \right\} \text{ but } h(\boldsymbol{x}_i) < -1 \\
1 + h(\boldsymbol{x}_i) & \text { for } i \in \left\{ i^- \right\} \text{ but } h(\boldsymbol{x}_i) > 1 \\
\end{array}\right.
$$

To incorporate the correctly classified points into account, the error is

$$
e_i = \left\{\begin{array}{ll}
\max \left\{ 1 - h(\boldsymbol{x}_i), 0 \right\} & \text { for } i \in \left\{ i^+ \right\} \\
\max \left\{ 1 + h(\boldsymbol{x}_i), 0 \right\} & \text { for } i \in \left\{ i^- \right\} \\
\end{array}\right.
$$

The objective is to minimize the total mis-classification error $\sum {e_i}$.

The original conditions for the correctly classified points

$$\begin{aligned}
i \in \left\{ i^+ \right\} \quad \forall \ i: h(\boldsymbol{x}_{i}) &\ge 1 \\
i \in \left\{ i^- \right\} \quad \forall \ i: h(\boldsymbol{x}_{i}) &\le -1 \\
\end{aligned}$$

now can be rewritten as

$$\begin{aligned}
e_{i^+} + h(\boldsymbol{x}_{i^+}) &\ge 1 \quad &&\forall \ i^+\\
e_{i^-} - h(\boldsymbol{x}_{i^+}) &\le -1 \quad &&\forall \ i^-\\
\end{aligned}$$

which can be interpreted as we pull the mis-classified points back into their correct region.

Therefore, the LP problem is

$$\begin{aligned}
\max && -\sum_{i} e_i && \\
\text{s.t.}
&& e_{i^+} +  h(\boldsymbol{x}_{i^+}) &\ge 1 &&\forall \ i^+\\
&& e_{i^-} - h(\boldsymbol{x}_{i^+}) &\le -1 &&\forall \ i^-\\
&& e_i &\ge 0 &&\forall \ i\\
\end{aligned}$$

with variables $\left\{ \boldsymbol{a} ,b , \boldsymbol{e}  \right\}$.

#### Non-linear Separable

If the two kinds of points are non-linearly separable, we can consider feature transformation $\boldsymbol{x}  \rightarrow \boldsymbol{\phi}(\boldsymbol{x} )$ and then solve the LP problem.

The feature transformation can involve non-linear terms such as $x_1^2, x_1 x_2$ etc.

## Duality

### Primal and Dual

Consider an LP minimization problem of the form

$$\begin{aligned}
\min && f(\boldsymbol{x} ) &= \boldsymbol{c}^{\top} \boldsymbol{x} \\
\text { s.t.} && \boldsymbol{A} \boldsymbol{x} &\ge \boldsymbol{b} \\
&& \boldsymbol{x} &\geq \mathbf{0}
\end{aligned}$$

where $\boldsymbol{A} \in \mathbb{R} ^{m \times n}, \boldsymbol{b} \in \mathbb{R} ^{m}$.

Before solving it, we consider a lower bound $\ell$ for $f(\boldsymbol{x} ^*)$, by linearly combine the constraints $\boldsymbol{a} _{1 \cdot} ^\top \boldsymbol{x} \ge b_1$, $\boldsymbol{a} _{2 \cdot} ^\top \boldsymbol{x} \ge b_2$, etc, where $\boldsymbol{a} _{i\cdot}$ is $i$-th row of $\boldsymbol{A}$. Let $y_1, y_2, \ldots$ be the corresponding non-negative multipliers. The linear combination of the constraints is

$$
y_1(\boldsymbol{a} _{1 \cdot} ^\top \boldsymbol{x} ) + y_2(\boldsymbol{a} _{2 \cdot} ^\top \boldsymbol{x} ) + \ldots + y_m (\boldsymbol{a} _{m \cdot} ^\top \boldsymbol{x}) \ge y_1 b_1 + y_2 b_2 + \ldots + y_m b_m
$$

or

$$
\boldsymbol{y} ^\top \boldsymbol{A} \boldsymbol{x} \ge \boldsymbol{y} ^\top \boldsymbol{b}
$$

If the LHS's coefficients of $x_i$ is **smaller** than $c_i$, i.e., $\sum_{j=1}^m y_j a_{ji} = \boldsymbol{a} _{\cdot i} ^\top \boldsymbol{y} < c_i$ where $\boldsymbol{a} _{\cdot j}$ is the $j$-th column of $\boldsymbol{A}$, then due to non-negativity of $x_i$, we always have

$$\boldsymbol{c}  ^\top \boldsymbol{x} \ge [\boldsymbol{a} _{\cdot 1} ^\top \boldsymbol{y} \quad \boldsymbol{a} _{\cdot 2} ^\top \boldsymbol{y} \quad \ldots \quad \boldsymbol{a} _{\cdot m} ^\top \boldsymbol{y}] \ \boldsymbol{x} = \boldsymbol{y} ^\top \boldsymbol{A} \boldsymbol{x}  \ge \boldsymbol{b} ^\top \boldsymbol{y}$$

So $\boldsymbol{b} ^\top \boldsymbol{y}$ is always a lower bound of $f(\boldsymbol{x})$. Moreover, we want the lower bound to be as larger as possible so that we can have a good estimate of the minimum value  $\min _ \boldsymbol{c} \boldsymbol{c} ^\top \boldsymbol{x}$. To sum up, we want to find $\boldsymbol{y}$ of the following maximization problem


$$\begin{aligned}
\max && g(\boldsymbol{y} ) &= \boldsymbol{b}^{\top} \boldsymbol{y} \\
\text { s.t. } && \boldsymbol{A} ^\top  \boldsymbol{y} &\le \boldsymbol{c} \\
&& \boldsymbol{y} &\geq \mathbf{0}
\end{aligned}$$

Any solution to this $\max g(\boldsymbol{y})$ problem provides a lower bound of $f(\boldsymbol{x} ^*)$.

$$
g(\boldsymbol{y}) \le f(\boldsymbol{x}^*)
$$

The problem $\max f(\boldsymbol{x} )$ is called **primal**, the problem $\min g(\boldsymbol{y} )$ is called **dual**.

Likewise, we can find an upper bound of $g(\boldsymbol{y} )$ by similar operation. Denote the multipliers by $\boldsymbol{z}$,

$$\begin{aligned}
\min && h(\boldsymbol{z} ) &= \boldsymbol{c}^{\top} \boldsymbol{z} \\
\text { s.t. } && \boldsymbol{A} \boldsymbol{z} &\ge \boldsymbol{b} \\
&& \boldsymbol{z} &\geq \mathbf{0}
\end{aligned}$$

Note that $h(\boldsymbol{z} )$ has exactly the same form with $f(\boldsymbol{x} )$. Hence, the dual of dual is primal.

### Duality Theorem

The inequality $g(\boldsymbol{y}) \le f(\boldsymbol{x})$ is called weak duality theorem.

The equality $g(\boldsymbol{y}^*) = f(\boldsymbol{x}^*)$ is called the strong duality theorem.

## Max-flow and Min-Cut from LP

### LP-max and Dual

Consider the path-defined flow. We can view each $f(P)$ as a variable. Then the optimization problem

$$\begin{aligned}
\max && \sum _{P \in \mathcal{P}} f(P) &&& \\
\text { s.t. }
&& \sum_{P: e \in e(P)} f(P) &\le c(e)  &&\forall e \\
&& f(P) &\geq 0  &&\forall P
\end{aligned}$$

is equivalent to the max-flow problem. Call this LP-flow problem.

Note the number of paths $\left\vert \mathcal{P} \right\vert$ is exponential to the graph size. Let's consider the dual.

Let the multipliers be $\boldsymbol{y}$, where $y_e$ is the multiplier for constraint of edge $e$. Note that in the primal, the coefficients of $f(P)$ in the objective function and each constraint are $1$. Hence, the dual is

$$\begin{aligned}
\min && \sum _{e} c(e) y_e &&& \\
\text { s.t. }
&& \sum_{e: e \in e(P)} y_e &\ge 1  &&\forall P \in \mathcal{P} \\
&& y_e &\geq 0  &&\forall e
\end{aligned}$$

One can image there is a matrix $\boldsymbol{A}$ with $\left\vert E \right\vert = m$ rows and $\left\vert \mathcal{P} \right\vert = p$ columns. Each entry $a_{ij}=\mathbb{I} [e_i \in e(P_j)]$. Let $\boldsymbol{f} \in \mathbb{R} ^{p}$ be the path flow vector, $\boldsymbol{c} \in \mathbb{R} ^ m$ be the edge capacity vector, then the primal is,

$$\begin{aligned}
\max && \boldsymbol{1}_p ^\top \boldsymbol{f}   &&& \\
\text { s.t. }
&& \boldsymbol{A} \boldsymbol{f}  &\le \boldsymbol{c}\\
&& \boldsymbol{f}  &\geq \boldsymbol{0}  &&
\end{aligned}$$

Let $\boldsymbol{y} \in \mathbb{R} ^{m}$ be the vector of $y_e$'s, then the dual is

$$\begin{aligned}
\min && \boldsymbol{c}^\top  \boldsymbol{y} &&& \\
\text { s.t. }
&& \boldsymbol{A} ^\top \boldsymbol{y}  &\ge \boldsymbol{1}_p   &&\\
&& \boldsymbol{y}  &\geq \boldsymbol{0}   &&
\end{aligned}$$

which is consistent with our notation in the last section.


### Relaxation

If we add integer constraint $y_e \in \left\{ 0,1 \right\}$, then the objective $\sum _{e} c(e) y_e$ is a edge selection problem to minimize the total capacity, and the constraints $\sum_{e: e \in e(P)} y_e \le 1$ implies that at least one edge is selected along every $s-t$ path. In other words, we want to find minimum number of edges to disconnect $s$ and $t$, which is exactly the min-cut problem.

If $y_e \in \mathbb{R}$, then it is a relaxation to the integer constraint $y_e \in \left\{ 0, 1 \right\}$. Call this problem LP-cut, we say LP-cut is a relaxation of min-cut.

Definition (Relaxation)
: Consider two problem $P$ and $P_r$, if any solution to $P$ corresponds to a solution to $P_r$ with the **same** value of the objective function in $P$, then we say problem $P_r$ is a relaxation to problem $P$. In this sense, $OPT_r$ is always better than or equal to $OPT$.

Since the feasible solutions to min-cut are integers, we call them integral solutions. The solutions to the LP-cut are called fractional solutions.

By the strong duality theorem, we have $OPT(\text{LP-flow} ) = OPT(\text{LP-cut})$. To sum up, we have

$$
OPT(\text{max-flow} ) = OPT(\text{LP-flow} ) = OPT(\text{LP-cut}) \le OPT(\text{min-cut} )
$$

We have shown that $OPT(\text{max-flow} ) = OPT(\text{min-cut} )$ in the [max-flow](../25-graph-related/31-maximum-flow) section. Hence the inequality $\le$ above should be equality $=$. Let's not use this fact, but just analyze this inequality itself.

### Integrality Gap

Definition (Integrality Gap)
: Integrality gap measures the largest deviation between the objective value of a fractional solution and that of a integral solution.

  - For minimization problem, its is $\frac{OPT}{OPT_{LP}} \ge 1$,
  - For maximization problem, its is $\frac{OPT_{LP}}{OPT} \ge 1$,

  If the gap is one, then an optimal integral solution and an optimal fractional solution gives the same objective value.

Claim: The integrality gap between min-cut and LP-cut is 1

Prove by providing an efficient algo that given any optimal fractional solution to $OPT_{LP}$, it returns an integral feasible solution whose cost is **not** higher than $OPT_{LP}$. (LP-rounding algorithm).

View $y_e \in \mathbb{R}$ as the length of edge $e$. The distance $d(u,v)$ is the length of shortest $u-v$ path under $y_e$ edge length. Recall the LP-cut problem

$$\begin{aligned}
\min && \sum _{e} c(e) y_e &&& \\
\text { s.t. }
&& \sum_{e: e \in e(P)} y_e &\ge 1  &&\forall P \in \mathcal{P} \\
&& y_e &\geq 0  &&\forall e
\end{aligned}$$

The constraints says that the distance of any $s-t$ path is at least one.

Consider a value $\rho \in (0,1)$, which defines a cut $(A_\rho, B_\rho)$, such that
- $A_\rho = \left\{ v \mid d(s,v) \le \rho \right\}$
- $B_\rho = \left\{ v \mid d(s,v) > \rho \right\} = V \backslash A_\rho$

Since $d(s,t) \ge 1 > \rho$, then $t \in B_\rho$.

Define
- $v(\rho) = c(A_\rho, B_\rho) = \sum _{u \in A_\rho, v\in B_\rho} c(u,v)$.
- $\rho^* = \arg\min _\rho v(\rho)$
- $(A^*, B^*) = (A_{\rho ^*}, B_{\rho ^*})$

Question: how to find $\rho^*$ efficiently? Note that most of the cuts are the same, though they corresponds to different $\rho$ values. We can sort vertices by $d(s,v)$ such that

$$
d(s,v_1) \le d(s,v_2) \le \ldots \le d(s, v_n)
$$

The number of different cuts is actually $n-1$. We can set cutoffs to be $\rho_i = d(s,v_i)$ where $i = 1, 2, \ldots, n-1$. Then any $\rho \in [\rho_i, \rho_{i+1})$ always gives the same cut.









.
