# Polynomial Reduction

In reality, we have a huge class of problems from various areas. If we solve one of them, we solve all of them. If we prove one of them does not have an efficient algorithm, then all of them do not. This is the class of NP-hard problems.

We first introduce polynomial reduction between problems, then define NP-hardness and NP-completeness.

## Definition

```{margin}
Sometimes people say "$y$ is easier than $x$" or "$x$ is harder than $y$".
```

Definition (Polynomial reduction)
: We say there exists a polynomial-reduction from problem $y$ to problem $x$, denoted $y \le _p x$, if there exists an algorithm for problem $y$ that solves it efficiently (time for writing inputs to $x$ + time for obtaining solution to $x$), given a black-box access to problem $x$.

Note that to solve $y$, suppose there is a black-box that solves problem $x$.

- Algorithm for problem $y$ can write inputs to problem $x$. For each such input, it query the black-box, the black-box provides a solution to problem $x$ in 1 time unit. Note that it may query many times, but polynomial.
  - If it query just one time, we call one-shot reduction (most cases)
  - Otherwise, we call multi-shot reduction (few, but used to find approximation hardness)
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

We introduce some problems. Their reductions are all one-shot reductions.

### Independent Set $=_p$ Vertex Cover

#### Independent Set

Definition (Independent set)
: A set $S \subseteq V$ is an independent set of graph $G=(V,E)$ if no two vertices in $S$ are adjacent.

From this definition, there are two problems.

- Optimization version: Given a graph $G=(V,E)$, find its max-cardinality independent set.

- Decision version: Given a graph $G=(V,E)$ and an integer $k$, is there an independent set of size $k$?

Note that if we can solve one of them then we can solve the other.

- If there is an algorithm to solve the optimization version, then we can use it for the decision version. Since once $k_\max$ is find, all $k \le k_\max$ is possible.

- If there is an algorithm to solve the decision version, then we can use it for the optimization version efficiently. First, use it to find the $k_\max$ (e.g. by binary search), then to find the IS of size $k_\max$, for each vertex
  - Remove it and its neighbors, ask the algorithm if there is an IS of size $k_\max-1$ in the remaining graph
    - If yes, then this vertex is in an IS of size $k_\max$, add it. Repeat this operation in the remaining graph.
    - Otherwise, some neighbors of this vertex are both in an IS of size $k_\max$. Go to next vertex.

#### Vertex Cover

Definition (Vertex cover)
: In an undirected graph $G=(V,E)$, a set $S\subseteq V$ is a vertex cover if for any edge in graph $G$, at least one of its endpoint is in $S$. Formally,  $\forall e = (u,v)\in E: u\in S$ or $v\in S$ or both.

From this definition, there are two problems.

- Optimization version: Given a graph $G=(V,E)$, find its min-cardinality vertex cover.

- Decision version: Given a graph $G=(V,E)$ and an integer $k$, is there an vertex cover of size $k$?

Likewise, if we can solve one of them then we can solve the other.

#### Equivalency

Now we show that the independent set problem and the vertex cover problem are polynomial reduction of each other.

Claim (Connection between IS and VC)
: In any graph $G$, a set $S \subseteq V$ is an independent set iff $V\backslash S$ is a vertex cover. Moreover, for an max-cardinality independent set, its complement is the min-cardinality vertex cover.

:::{admonition,dropdown,seealso} *Proof from detinitions*

$(\Rightarrow)$ If $S$ is an independent set, then for any edge $e=(u,v)$ in graph $G$, we must have $u \in V \backslash S$ or $v \in V \backslash S$. Otherwise, $u,v \in S$ and they are adjacent, which contradicts to the fact that $S$ is an independent set. Hence, $V\backslash S$ is a vertex cover.

$(\Leftarrow)$ If $C$ is a vertex cover, then for any two vertices $u,v$ in $V \backslash C$, they cannot be adjacent. Otherwise, $e=(u,v)$ but $u,v \notin C$, which contradicts to the fact that $C$ is a vertex cover. Hence, $V\backslash C$ is an independent set.

:::

Therefore, once we find an independent set, we can find a vertex cover as its complement, and vice versa. Hence, they are polynomial reduction of each other.

For instance, if we want to answer the decision problem: *"Given a graph $G=(V,E)$ and an integer $k$, is there an independent set of size $k$?"*, suppose we have a black-box for vertex cover, we just ask it *"Is there a vertex cover of size $n-k$?"*. The answer from the black-box is the answer to the original decision problem.


### 3SAT $\le_p$ Independent Set

SAT and 3SAT are Boolean constraint satisfaction problems.

#### 3SAT

Definitions
: - A **Boolean variable** $x$ can only take one of the two values, TRUE or FALSE
  - A **literal** $l$ is either a Boolean variable $x$, called positive literal, or the negation of a variable $\neg x$, called negative literal.
  - A **clause** $c$ is a disjunction (OR) of one or several literals.

    For instance, $x_1$ is a positive literal, $\neg x_2$ is a negative literal, $x_1 \lor \neg x_2$ is a clause of two literals.

  - A propositional logic **formula**, also called Boolean expression, is built from variables $x$, operators AND (conjunction, also denoted by $\land$), OR (disjunction, $\lor$), NOT (negation, $\neg$), and parentheses $()$.
  - A formula is said to be **satisfiable** if it can be made TRUE by assigning appropriate logical values (i.e. TRUE, FALSE) to its variables.

    For instance, the formula $x \land \neg x$ consisting of two clauses of one literal, is unsatisfiable.

  - A **SAT formula** $\phi$ is a conjunction (AND) of one or more clauses $c_1 \land c_2 \land \ldots \land c_m$.
  - The Boolean satisfiability problem (**SAT**) is, given an SAT formula $\phi = c_1 \land c_2 \land \ldots \land c_m$, determine whether there is an assignment to the variables satisfying the formula. Note that every clause need to be satisfied.
    - If every clause consists of at most 3 variables, we say it is a **3SAT** problem.
    - If every clause consists of exactly 3 variables, we say it is a **E3SAT** problem.

#### Reduction to IS

Though 3SAT is an boolean satisfaction problem and independent set is a graph problem, they are related and 3SAT $\le _p$ IS.

Suppose $\phi = c_1 \land \ldots \land c_m$, and $c_1 = x_1 \lor x_2 \lor \neg x_3$. If $\phi$ is satisfiable, then one of the literals in $c_1$ must be true. Hence, one method of assignment is, for every clause $c$, assign one literal to be true, without conflicts (e.g. $x_3 = 1$ in $c_2$ and $\neg x_3=1$ in $c_3$). In other words, if one literal is chosen in some clause, never choose its negation in other clauses.

Now, we build a graph $G(\phi)$ induced from $\phi$. For each clause $c_i$ in $\phi$

- Add a vertex $v_{ij}$ where $1\le j \le 3$ to represent each literal $l_{ij}$ in $c_i$
- Add an edge between every pair of vertices $v_{ij}$. We call this object a gadget.
- If one vertex $v_{i_1 j_1}$ represents a literal that is the negation of a literal represented by another vertex $v_{i_2 j_2}$, add an edge between them $e = (v_{i_1 j_1}, v_{i_2 j_2})$.

In sum, for each clause, we build a gadget, and connect pairs of vertices across gadget if their literals are negation of each other.


Claim
: $\phi = c_1 \land \ldots \land c_m$ is satisfiable iff $G(\phi)$ has an independent set of size $m$.

:::{admonition,dropdown,seealso} *Proof*

$(\Rightarrow)$ If $\phi$ is satisfiable, then we do not choose any two literals that are negation of each other, hence the corresponding vertices are not adjacent, i.e. we find an independent set of size $m$ in $G(\phi)$.

$(\Leftarrow)$ If there is an independent set of size $m$ in $G(\phi)$, since the vertices in a gadget are adjacent and there are $m$ gadgets, then exactly one vertices in a gadget is in an independent set. Moreover, their corresponding literals are not negations.

:::

Therefore, to answer the question *"Is an 3SAT problem $\phi$ satisfiable?"*, we can query the black-box *"Is there an independent set of size $m$ in graph $G(\phi)$?"*, where $m$ is the size of clauses in $\phi$. Hence, 3SAT $\le _p$ IS.


###  Vertex Cover $\le_p$ Set Cover

#### Set Cover

Set cover is a general problem.

- Input
  - A collection of element $U = \left\{ 1, 2, \ldots, n \right\}$ (aka universe)
  - A set of its subsets $F = \left\{ S_1, S_2, \ldots, S_m \right\}$ such that $S_i \subseteq U$.

- Goal
  - Select minimum-cardinality $F ^\prime \subseteq F$ such that $F ^\prime$ covers all elements in $U$, i.e. $\cup _{S \in F} = U$.

In other words, any element $u \in U$ must be in at least one set $S \in F ^\prime$.

Decision version: Given universe $U$ and an integer $k$, is there a set cover of size $k$?

#### Reduction to VC

Claim
: In graph $G$, there exists a vertex cover of size $k$ iff there exists a set cover $F ^\prime$ of size $k$ for problem $(U, F)$.

:::{admonition,dropdown,seealso} *Proof*

For an instance of vertex cover problem, we can build an instance of set cover problem. Given $G=(V,E)$,
- $U$ is the set of all edges in $G$, i.e. $U = E$.
- $S_i$ is the set of all edges that incident to $v_i$, such that $\left\vert F \right\vert = \left\vert V \right\vert$.

:::

Therefore, to answer the question *"Given graph $G$, is an vertex cover of size $k$"*, we can query the black-box *"Given universe $U$ and an integer $k$, is there a set cover of size $k$?"*, where $U$ and $F$ are induced from $G$. Hence, VC $\le _p$ SC.


## Transitivity

Claim (Transitivity of reduction)
: If $z \le _p y$ and $y \le _p x$, then $z \le_p x$.

:::{admonition,dropdown,seealso} *Proof*

Short: polynomial of polynomial is still polynomial.

Long:

Since $y \le_p x$, there exists an efficient algorithm $A_1$ for problem $y$, given access to a black-box $BB(x)$. Let the running time of $A_1$ be $f_1(n_y)$ where $n_y$ is the input size of problem $y$. Since $z \le_p y$, there exists an efficient algorithm $A_2$ for problem $z$, given access to a black-box $BB(y)$. Let the running time of $A_2$ be $f_2(n_z)$.

Now we want to show there is an efficient algorithm $A^*$ for $z$, given access to the black-box $BB(x)$. Given a instance of $z$ of size $n_z$, we run $A_2$, and whenever it queries $BB(y)$, we run $A_1$ instead, which queries $BB(x)$. It is efficient since
- Running time of $A_2$ is $f_2(n_z)$
- Number of runs of $A_1$ is at most $f_2(n_z)$.
  - Each run takes $f_1(n_y)$.
  - To convert input $z$ to $y$, the conversion time is at most $f_2(n)$, hence $n_y = O(f_2(n_z))$.

In sum, the total run time is $f_2(n_z) + f_2(n_z) \cdot f_1 (f_2(n_z))$, which is polynomial since $f_1, f_2$ are polynomials.
:::


Applying the claim gives a chain of reduction

$$
\text{3SAT}  \le_p \text{Independent Set}  \le_p \text{Vertex Cover}  \le_p \text{Set Cover}  
$$
