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
