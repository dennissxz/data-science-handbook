# Randomized Algorithms

No assumption of distribution of input. For any instance, the algo is good

- optimal solution
- near optimal solution
- expected running time is polynomial


### Max Exact 3SAT

Input
- E3SAt formula $\phi$
  - $n$ boolean variables, $m$ clauses $c$ and every $c$ consist of three literals from different variables.

Optimization version
- find an assignment to all variables satisfying as many clauses as possible.

Algo: assign to every variable a value T/F uniformly at random independently.

Consider a clause, $c_i = l_{i_1} \vee l_{i_2} \vee l_{i_3}$. Then
- $\operatorname{P}\left( c_i = T \right) = \frac{7}{8}$.
- $\operatorname{E}\left( \# \text{satisfied clauses}  \right) = \sum_{i=1}^m \operatorname{P}\left( c_i = T \right) = \frac{7}{8} m$

Corollary
: For any E3SAT formula $\phi$  on $m$ clauses, there exists an assignment satisfying at least $\frac{7}{8} m$ clauses. Moreover, if $m\le 7$ then the formula is satisfiable, since $\lceil \frac{7}{8} m \rceil = m$.

To ensure that the number of clauses satisfied is at least $\frac{7}{8} m$, just run the above algorithm multiple times, until we obtain an assignment satisfying at least $\frac{7}{8} m$ clauses.

What is the expected number of iterations?


Claim
: Let $P$ be the probability that a random assignment satisfies at least $\frac{7}{8} m$ clauses. $P \ge \frac{1}{8m}$.

:::{admonition,dropdown,seealso} *Proof*


:::
