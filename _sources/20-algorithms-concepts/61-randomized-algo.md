# Randomized Algorithms

Randomized algorithms requires no assumption of distribution of input instance. For any input instance, a randomized algorithm is good in the sense that its expected running time is polynomial, and it gives a near optimal solution with high probability.


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

What is the expected number of iterations?

Claim
: Let $p^*$ be the probability that a random assignment satisfies at least $\frac{7}{8} m$ clauses, then $p^* \ge \frac{1}{8m}$.

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


.


.


.


.


.


.


.


.
