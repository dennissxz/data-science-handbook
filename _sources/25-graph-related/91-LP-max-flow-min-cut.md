# LP on Max-flow and Min-cut

## Duality


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



## Relaxation

If we add integer constraint $y_e \in \left\{ 0,1 \right\}$, then the objective $\sum _{e} c(e) y_e$ is a edge selection problem to minimize the total capacity, and the constraints $\sum_{e: e \in e(P)} y_e \le 1$ implies that at least one edge is selected along every $s-t$ path. In other words, we want to find minimum number of edges to disconnect $s$ and $t$, which is exactly the min-cut problem.

If $y_e \in \mathbb{R}$, then it is a relaxation to the integer constraint $y_e \in \left\{ 0, 1 \right\}$. Call this problem LP-cut, we say LP-cut is a relaxation of min-cut.



Since the feasible solutions to min-cut are integers, we call them integral solutions. The solutions to the LP-cut are called fractional solutions.

By the strong duality theorem, we have $OPT(\text{LP-flow} ) = OPT(\text{LP-cut})$. To sum up, we have

$$
OPT(\text{max-flow} ) = OPT(\text{LP-flow} ) = OPT(\text{LP-cut}) \le OPT(\text{min-cut} )
$$

We have shown that $OPT(\text{max-flow} ) = OPT(\text{min-cut} )$ in the [max-flow](../25-graph-related/31-maximum-flow) section. Hence the inequality $\le$ above should be equality $=$. Let's not use this fact, but just analyze this inequality itself.



## Integrality Gap

Claim: The integrality gap between min-cut and LP-cut is 1

Prove by providing an efficient algo that given any optimal fractional solution to $OPT_{LP}$, it returns an integral feasible solution whose cost is **not** higher than $OPT_{LP}$. (LP-rounding algorithm).

:::{admonition,dropdown,seealso} *Proof*

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

The number of different cuts is actually $n-1$. We can set cutoffs to be $\rho_i = d(s,v_i)$ where $i = 1, 2, \ldots$ until it is closest to 1. Then any $\rho \in [\rho_i, \rho_{i+1})$ always gives the same cut.

We want to prove $c(A^*, B^*) \le \sum_{e} c(e)y_e$. A sufficient condition is

$$
\operatorname{E} [v(\rho) ]\le \sum_{e} c(e)y_e
$$

where $\rho$ is chosen uniformly at random from $(0,1)$. Now we prove this sufficient condition.


$$\begin{aligned}
\operatorname{E}_\rho \left[ v(\rho) \right]
&= \operatorname{E}\left[ \sum _{u \in A_\rho, v\in B_\rho} c(u,v) \right]\\
&= \sum_e c(e) \operatorname{E} \left\{ \mathbb{I}\left[ e \in E(A_\rho, B_\rho) \right] \right\}\\
&= \sum_e c(e) \operatorname{P} \left\{ e \in E(A_\rho, B_\rho)  \right\}\\
\end{aligned}$$

Thus, it remains to show $\operatorname{P} \left\{ e \in E(A_\rho, B_\rho)  \right\}\le y_e$. Suppose $e=(u,v)$, consider the probability.

- If $d(s,u) < d(s,v)$, then it contributes to the cut iff $d(s,u)\le \rho < d(s,v)$, with probability $d(s,v)-d(s,u)$, which is less than or equal to the edge length $y_e$.
- If $d(s,u) \ge d(s,v)$, then it cannot contribute to the cut, i.e. $\operatorname{P} \left\{ e \in E(A_\rho, B_\rho)  \right\}=0$

Hence, the probability is always less than $y_e$, and it completes proof of integrality gap > 1.

:::
