# Randomized Algorithms

Randomized algorithms requires no assumption of distribution of input instance. For any input instance, a randomized algorithm is good in the sense that its expected running time is polynomial, and it gives a near optimal solution with high probability.

Reference
- UBC Randomized Algorithms (Winter 2015) [link](https://www.cs.ubc.ca/~nickhar/W15/)


## Max Exact 3SAT

### Problem

Input
- E3SAT formula $\phi$
  - $n$ boolean variables
  - $m$ clauses
    - every clause $c$ consist of three literals from **different** variables.

Optimization version
- find an assignment to all variables that satisfies as many clauses as possible.

### Randomized Algorithm

A randomized algorithm is to assign to every variable a value T/F uniformly at random **independently**.

Consider a clause, $c_i = l_{i1} \vee l_{i2} \vee l_{i3}$. Then
- $\operatorname{P}\left( c_i = T \right) = \frac{7}{8}$
- $\operatorname{E}\left( \# \text{satisfied clauses}  \right) = \sum_{i=1}^m \operatorname{E} \left[ \mathbb{I} (c_i = T) \right] = \sum_{i=1}^m \operatorname{P}\left( c_i = T \right) = \frac{7}{8} m$. Note that the first equality holds due to linearity of expectation, though the two events $\left\{ c_i = T \right\}$ and $\left\{ c_j= T \right\}$ are not independent.


Corollary
: For any E3SAT formula $\phi$  on $m$ clauses, there exists an assignment satisfying at least $\frac{7}{8} m$ clauses, i.e. $\lceil \frac{7}{8} m \rceil$ clauses. Moreover, if $m\le 7$ then the formula is satisfiable, since $\lceil \frac{7}{8} m \rceil = m$.

To ensure that in the output of the randomized algorithm, the number of clauses satisfied is at least $\frac{7}{8} m$, we just run the randomized algorithm multiple times, until we obtain an assignment satisfying at least $\frac{7}{8} m$ clauses.

### Expected Number of Iterations

If a random assignment satisfies at least $\frac{7}{8} m$ clauses, we call it a success.

What is the expected number of iterations to obtain a success?

Claim
: Let $p^*$ be the probability of success, then $p^* \ge \frac{1}{8m}$.

:::{admonition,dropdown,seealso} *Proof*

Let

- $p_j$ be the probability that exactly $j$ clauses are satisfied.
- $k$ be the largest integer smaller than $\frac{7}{8}m$

Then we have


$$\begin{aligned}
\operatorname{E}\left( \# \text{satisfied clauses}  \right)
&= \sum_{j=0}^m j \cdot p_j \\
&= \sum_{j=0}^k j \cdot p_j + \sum_{j=k+1}^m j \cdot p_j\\
&\le \sum_{j=0}^k k \cdot p_j + \sum_{j=k+1}^m m \cdot p_j\\
&\le k + m p^*\\
\end{aligned}$$

Hence $k+mp^* \ge \operatorname{E}\left( \# \text{satisfied clauses}  \right) = \frac{7}{8}m$, and then $p^* \ge \frac{1}{m} \left( \frac{7}{8}m - k  \right)$. Since $\frac{7}{8}m - k > 0$, then the integer $7m-8k \ge 1$, then $\frac{7}{8}m - k \ge \frac{1}{8}$ and $p^* \ge \frac{1}{8m}$.

:::

Therefore, the number of random runs until a success follows a geometric distribution with probability $p^*$. Hence, to obtain a success the expected number of runs is $\frac{1}{p^*}$ which is smaller than $8m$. But it may happen that there is no success solution after many runs, so make sure there is some appropriate stoping criteria.

## Monte Carlo and Las Vegas

### Monte Carlo Algorithms

```{margin}
w.h.p. = with high probability
w.p.1 = with probability 1
```

Monte Carlo algorithms are a family of algorithms that

- provide a good (optimal, near-optimal, etc.) solution w.h.p., or
- the expected solution value is close to $OPT$.

In the first case w.h.p., if we obtain a unsatisfactory solution, then we can repeat running until find a satisfactory solution, like what we did in the previous example.


### Las Vegas Algorithm

Las Vegas algorithms always produce a good solution with polynomial expected run time.

Generally, if we have a Las Vegas algorithm then we can get a Monte Carlo algorithm: run LV , if there is no optimal solution in some polynomial time, then output some random solution and then re-run.

The reverse does not always work. Cannot know whether optimal in MC??

Question: If there is an efficient randomized algorithm for a problem, is there always an efficient deterministic algorithm for it? i.e., $RP = P$? Believed to be true, no proof yet.

## Global Minimum Cut

A global minimum cut is a partition of vertices $(A,B)$ of $V$ that minimizes the number of across-set edges $\left\vert E(A,B) \right\vert$.

We've introduced minimum $s-t$ cuts. Apply it to this problem: assign every edge the same capacity, and go over all vertices pairs $(u,v)$, find the global minimum $u$-$v$ cut. This takes $O(n^2)$ time which is slow.

There is a randomized algorithm to solve this in $O(n)$. The idea is: iteratively contraction of an edge $e=(u,v)$, i.e., merge two adjacent vertices $u,v$ into one new vertex $w$.

$$
> u - v < \quad \Longrightarrow \quad > w <
$$

- Delete edge $e=(u,v)$
- All edges that are adjacent to $u,v$ now becomes adjacent to $w$.
- If there is a vertex $a$ that are adjacent to both of $u$ and $v$, we keep the two parallel edges $(a,w)$.

Lemma
: Let $c$ be the minimal cost. Then any vertex must have degree greater than $c$ (otherwise that vertex has a cut smaller than $c$).

---
Algorithm

---
Repeat

  - Choose one edge $e=(u,v)$ from the current graph uniformly at random.
  - Contract $e$, keep parallel edges, delete self loops. Note that the number of vertices decreases by 1.

After $n-2$ iterations, then there are two vertices left. Each vertex stands for some vertices of original $V$. These two vertices defines a cut $(A,B)$.

---

Theorem (Global minimum cut probability)
: The cut $(A,B)$ returned by the above algorithm is a *specific* global minimum cut with probability $\ge \frac{1}{C_n^2}$.

:::{admonition,dropdown,seealso} *Proof*

Let $(A^*,B^*)$ be a global minimum cut. Let $E ^* = E(A ^*, B^*)$ be the across-set edges. The cut returned by the algorithm is this minimal cut $E^*$ if no edge in $E^*$ is contracted.

Let $A_i$ be the event that all edges in $E ^*$ survive in iteration $i$. If $A_{n-2}$ happens then the output cut is $E^*$.

Suppose $\left\vert E ^*  \right\vert = c$. Then in the original graph, $\operatorname{deg}(v) \ge c$. So the total number of edges $\left\vert E(G) \right\vert  =  \frac{1}{2} \sum_{v\in V} \operatorname{deg}(v)  \ge \frac{nc}{2}$.

It is easy to see in the first iteration

$$\mathbb{P} \left( A_1 \right) = 1 - \mathbb{P} \left( \text{one edge of $E ^*$ is chosen}  \right) = 1 - \frac{\left\vert E ^* \right\vert}{\left\vert E(G) \right\vert} \ge 1 - \frac{c}{nc/2} = 1 - \frac{2}{n} = \frac{n-2}{n}$$

Given all edges survive in the previous $(i-1)$- iterations, the probability that they survive the $i$-th iteration is

$$
\mathbb{P} \left( A_i \mid A_{i-1} \right) = 1 - \frac{\left\vert E^* \right\vert}{ \left\vert E(G_{i-1}) \right\vert} \ge 1 - \frac{c}{(n-i +1)c/2} = \frac{n-i-1}{n-i+1}
$$


By chain rule, we have

$$\begin{aligned}
\mathbb{P} (A_{n-2})
&= \mathbb{P} (A_1) \prod _{i=2}^{n-2} \mathbb{P}\left( A_i \mid A_{i-1} \right) \\
&\ge \frac{n-2}{2}   \prod _{i=2}^{n-2} \frac{n-i-1}{n-i+1}  \\
&= \frac{n-2}{n} \cdot \frac{n-3}{n-1} \cdot \frac{n-4}{n-2} \ldots \frac{1}{3}    \\
&= \frac{(n-2)!2!}{n!}   \\
&= \frac{1}{C_n^2}  \\
\end{aligned}$$

What if a deleted parallel edge is in $E ^*$? This cannot happen since if we do not select an edge from $E^*$, then no parallel edge can be in $E^*$. In other words, either all parallel edges are in the output cut, or none of them are in the output cut.

:::

Corollary (Largest number of global minimum cut)
: The number of global minimum cut in a graph is at most $C_n^2$.

  Suppose the total number of global minimum cut is $K$, then

  $$\mathbb{P} (\text{the output is not any g.m.c.} ) = 1 - \sum_{k=1}^K \mathbb{P} (\text{the output is g.m.c $k$} ) = 1 - \frac{K}{C_n^2} \ge 0 \Rightarrow K \le C_n^2$$

An equivalent algorithm (Karger's) is

- For each edge $e$, assign random weight $w_e \sim \operatorname{U}(0,1]$.
- Run Kruskal's algorithm to find a minimum spanning tree.
- Remove the heaviest edge from this MST to obtain a cut.

Run time of these steps is $\mathcal{O}(m \log m)$, and the output cut is a specific gobal minimum cut with probability at least $\frac{1}{C_n^2}$. To find a global minimum cut with constant probability (not depend on $n$), we need $\mathcal{O}(mn^2 \log m)$.

Improvement:

In the algorithm, in early iterations (small $i$), the probability of survival in each iteration $\frac{n-i-1}{n-i-1}$ is high. We can take advantage of this.

- Run the algorithm for $\frac{n}{2}$ iterations.
- Then make 2 copies of the graph, run 2 instances of the algorithm in parallel for $\frac{n}{4}$ iterations.
- Then make 2 copies of each graph, run 4 instances of the algorithm in parallel for $\frac{n}{8}$ iterations.
- $\ldots$

This reduces the running time to $\mathcal{O}(n^2 \log n)$ with success probability $\mathcal{\Omega} \left( \frac{1}{\log n}  \right)$.

Other improvements:
- approximate global minimum cut
- sparsify the graph

## Polynomial Identity Testing


A moninomial of $n$ variables has the form $c x_1 ^{d_1} x_2 ^{d_2} \ldots x_n ^{d^n}$ where $d_i \ge 0$. The degree of a monomial is $\sum_i d_i$.

A polynomial $P(x_1, x_2, \ldots, x_n)$ of $n$ variables is a sum of moninomials of $n$ variables. The degree of a polynomial is the largest degree of its monomials.

A polynomial $P$ is said to be identically zero, denoted $P \equiv 0$ if the coefficients of its monomials are all 0.

Question: given a polynomial $P$ in $n$ variables over field $f$ of degree $d$ in the following forms, is $P \equiv 0$?


$$
P(x_1, x_2, \ldots, x_n) = (x_1 + x_2)(x_3 ^2 + 2x_4 - x_5) \ldots
$$

$$
A = \left[\begin{array}{ccc}
x_1 & 0 & \cdots \\
2x_2 & -1 & \cdots \\
\vdots & \vdots & \cdots
\end{array}\right]
$$

If we unpack the parentheses or compute the determinant of $A$ to find the monomial representation of $P$ and determine whether $P \equiv 0$, the computation is hard. But if we know the values of variables, it is easy to verify the value of $P$.

In particular,

- If $\mathbb{F} = \mathbb{R}$, then $P \equiv 0$ if and only if $P(x_1, \ldots x_n)=0$ for all $x_1, \ldots, x_n \in \mathbb{R}$.
- But for some other fields this is not true: if $\mathbb{F} = \left\vert 0,1 \right\vert$ and $P(x)=x^2 -x$, then $P(x)=0$ for all $x\in \left\{ 0,1 \right\}$ but it is not identically 0.

Lemma (Schwartz-Zippel)
: Let
- $P(x_1, \ldots, x_n)$ be a multivariate polynomial of degree $d\ge 0$ over some field $\mathbb{F}$
- $S \subseteq \mathbb{F}$ be a finite subset of $\mathbb{F}$
- $r_1, \ldots, r_n$ be independent and uniform random draws from $S$

If $P(x_1, \ldots, x_n)$ is not identically to $0$, then it evaluates to 0 with probability

$$
\mathbb{P} \left\{ P(r_1, \ldots , r_n) =0 \right\} \le \frac{d}{\left\vert s \right\vert}
$$

For instance, we can choose $\left\vert S \right\vert = nd$, such that the upper bound is $\frac{1}{n}$. If $\mathbb{F}$ is finite then we choose $S = \mathbb{F}$.
