# Polynomial Reduction

In reality, we have a huge class of problems from various areas. If we solve one of them, we solve all of them. If we prove one of them does not have an efficient algorithm, then all of them do not. This is the class of NP-hard problems.

We first introduce polynomial reduction between problems, then define NP-hardness and NP-completeness.

```{margin}
Sometimes people say "$y$ is easier than x" or "x is harder than $y$".
```

Definition (Polynomial reduction)
: We say there exists a polynomial-reduction from problem $y$ to $x$, denoted $y \le _p x$, if there exists an algorithm for problem $y$ that solves it efficiently (time for writing inputs to $x$ + time for obtaining solution to $x$), given a black-box access to problem $x$.

Note that to solve $y$, suppose there is a black-box that solves problem $x$.

- Algorithm for problem $y$ can write inputs to problem $x$. For each such input, it query the black-box, the black-box provides a solution to problem $x$ in 1 time unit. Note that it may query many times, but polynomial.
- Time to write down input to problem $x$ is counted in the run time for problem $y$


Claim
: If $y \le _p x$ and there exists an efficient algorithm for $x$ (not black-box), then there exists an efficient algorithm for $y$.

:::{admonition,dropdown,seealso} *Proof*

Let $A$ be an efficient algorithm for $y$, given black-box access to $x$. Let $f(n)$ be its running time on input of length $n$. Since $A$ is efficient, we have $f(n)\le \operatorname{Poly}(n)$.

Let $A ^\prime$ be an efficient algorithm for $x$, with running time $g(N)$ of input length $N$. We have $g(N)\le \operatorname{Poly}(N)$.

To solve $y$, we run $A$, and whenever $A$ queries the black-box, we run algorithm $A ^\prime$ instead. Let this algorithm be $A^*$.
- By definition of $A$, we call black-box at most $f(n)$ times, hence in $A^*$ we call $A ^\prime$ at most $f(n)$ times.
- In each time, the length of input is also at most $f(n)$, so the running time of a single call to $A ^\prime$ is at most $g(f(n))$.
- Total time of all calls to $A ^\prime$ is at most $f(n)\times g(f(n))$.

The total runtime is $f(n)+f(n)\times g(f(n))$, which is also polynomial.

??plus
:::

Corollary (contrapositive to the claim)
: If $y \le _p x$ and there is no efficient algorithm for $y$, then there is no efficient algorithm for $x$.

Reduction can be used to solve problems (e.g. solve bipartite matching by max-flow), or used to prove hardness.


## Examples

We introduce some problems. All are NP-hard.

### Independent Set $\Leftrightarrow$ Vertex Cover

Definition (Independent set)
: A set $S \subseteq V$ is an independent set of graph $G=(V,E)$ if no pair or vertices in $S$ is connected by an edge.

Optimization version
- Given a graph $G=(V,E)$, find its max-cardinality independent set.

Decision version
- Given a graph $G=(V,E)$ and an integer $k$, is there an independent set of size $k$?

Note that

- If there is an algorithm to solve the optimization version, then we can use it for the decision version. Since once $k_\max$ is find, all $k \le k_\max$ is possible.

- If there is an algorithm to solve the decision version, then we can use it for the optimization version efficiently. First, use it to find the $k_\max$ (e.g. by binary search), then to find the IS of size $k_\max$, for each vertex
  - Remove it and its neighbors, ask the algorithm if there is an IS of size $k_\max-1$ in the remaining graph
    - If yes, then this vertex is in an IS of size $k_\max$, add it. Repeat this operation in the remaining graph.
    - Otherwise, some neighbors of this vertex are both in an IS of size $k_\max$. Go to next vertex.

Definition (Vertex cover)
: In an undirected graph $G=(V,E)$, a set $S\subseteq V$ is a vertex cover if for any edge in the graph, at least one of its endpoint is in $S$. Formally,  $\forall e = (u,v)\in E: u\in S$ or $v\in S$ or both.

Optimization version
- Given a graph $G=(V,E)$, find its min-cardinality vertex cover.

Decision version
- Given a graph $G=(V,E)$ and an integer $k$, is there an vertex cover of size $k$?

Likewise, if an algorithm for any one version exists, we can use it to solve the other version.

Now we show that the IS problem and the VC problem are polynomial reduction of each other.

Claim
: In any graph $G$, a set $S \subseteq V$ is an independent set iff $V\backslash S$ is a vertex cover. Moreover, for an max-cardinality independent set, its complement is the min-cardinality vertex cover.

:::{admonition,dropdown,seealso} *Proof*

tbd


:::



### 3SAT $\le_p$ Independent Set

SAT and 3SAT are constraint satisfaction problems.

Definition (SAT)
: For $n$ boolean variables $x_1, x_2, \ldots, x_n$ , we define
- literals (negation) $\left\{ x_i, \bar{x_i} \right\}_{1\le i \le n}$
- clause: OR of a number of literals, e.g. $x_1 \vee \bar{x}_2 \vee x_4$.

SAT formula: clause $c_1, c_2, \ldots, c_m$.

$$
\phi = c_1 \wedge c_2 \wedge \ldots \wedge c_m
$$

SAT (decision): Given a SAT formula $\phi$ as input, is there an assignment to the variable satisfying the formula?

3SAT: every clause $c$ has at most 3 literals.

E3SAT: every clause $c$ has exact 3 literals.

###  Vertex Cover $\le_p$ Set Cover

## Transitivity of Reduction


$$
\text{3SAT}  \le_p \text{Independent Set}  \le_p \text{Vertex Cover}  \le_p \text{Set Cover}  
$$
