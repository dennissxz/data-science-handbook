# $P$ and $NP$

We introduce class $P$, $NP$ and related concepts. We focus on decision version rather than optimization version of problems which are convenient for discussion.

Let $s$ be a string that encodes some problem input (e.g. a formula $\phi$ in SAT, a graph $G$ and integer $k$ in IS). For a problem $x$, some string leads to answer yes, others no. Let $X$ be a set of "yes" string:

$$X = \left\{ s \mid s \text{ is a valid encoding of a "yes-instance" of problem } x   \right\}$$

Now we define previous problems as a set of yes-instance strings.

$$SAT = \left\{ s \mid s \text{ encodes a satisfiable SAT formula $\phi$} \right\}$$

$$IS = \left\{ s \mid s \text{ encodes an integer $k$ and a graph $G$ who has an IS of size $k$ }\right\}$$

## $P$

Definition (Efficient algorithm for problem $X$)
: We say $A$ is an efficient algorithm for problem $X$ iff there is a function $f:\mathbb{R} ^+ \rightarrow \mathbb{R} ^+$,  $f(n)\le \operatorname{Poly}(n)$, such that on input $s$ of length $n_s$, algorithm $A$
- runs for at most $f(n_s)$ time steps, and
- returns "yes" iff $s$ is a valid encoding of a problem instance of $X$, i.e. $s\in X$.

Definition (Class $P$)
: Class $P$ is all problems that have efficient algorithms.

## $NP$

There is no efficient algorithm for problems 3SAT, IS, VC, SC, so they are not in $P$. But for a proposed solution to such problem, it is easy to check whether it is a valid solution or not.

Definition (Verifier for problem $X$)
: A verifier for problem $X$ is an efficient algorithm with inputs
  - string $s$ that encodes problem instance $x$, and
  - string $t$, aka **certificate** or **proof**, which is usually a proposed solution

  and returns *yes* or *no*. "Efficient" means the run time is $\operatorname{Poly}(\left\vert s \right\vert + \left\vert t \right\vert)$.

Definition (Valid verifier for problem $X$)
: An efficient algorithm $A$ is a valid verifier for problem $X$ if there is a function $g: \mathbb{R} ^+ \rightarrow \mathbb{R} ^+$, $g(n)\le \operatorname{Poly}(n)$ such that for all $s$:

  - if $s\in X$, then there exists a certificate $t$ with length $\left\vert t \right\vert \le g(\left\vert s \right\vert)$ such that given $s$ and $t$, algorithm $A(s,t)$ returns "yes" (accept);

  - else $s\notin X$, then for any certificate $t$ with length $\left\vert t \right\vert\le g(\left\vert s \right\vert)$, verifier $A$ returns "no" (reject).


Definition (Class $NP$)
: Non-deterministic polynomial, denoted $NP$, is a class of all problems for which an **valid** verifier exists. The name "non-deterministic" means a non-deterministic Turing machine can describe this class of problems in polynomial time.

For instance, 3SAT, IS, VC, and SC are in class $NP$.

Claim
: Every problem in $P$ belongs to $NP$, i.e. $P \subseteq NP$.

***Proof***: Given input $s$ and $t$, the verifier ignores $t$, solve problem instance $s$ and accepts or reject accordingly.

There are many $NP$ problems, 3SAT, IS, VC and $SC$ are some of them.

- Question 1: $NP \subseteq P$ ? Or $NP = P$?

  It is believed the answer is NO. Some intuitions:

  - If yes, then there are many good consequences that are to good to be true
  - Solving a problem should be harder than verifying its solution

- Question 2: We have seen polynomial reduction of $NP$ problems $\text{3SAT} \le_p \text{IS}  \le_p \text{VC}  \le_p \text{SC}$, where $\text{SC}$ is the "hardest" one among the four. Is there a "hardest" problem in $NP$?

  This leads to the definition of $NP$-hard problems and $NP$-complete problems.

# $NP$-hard and $NP$-complete

Definition (Class $NP$-hard)
: A problem $X$ is in class $NP$-hard if for any $Y\in NP$, it reduces to $X$. Formally, $\forall\, Y \in NP, Y \le_p X$.


Definition (Class $NP$-complete)
: A problem $X$ is in class $NP$-complete if $X \in NP$ and $X$ is $NP$-hard. Formally

  $$ X \in NP \text{ and } \forall\, Y \in NP, Y \le_p X$$

Note that there are some $NP$-hard problem $X$ that is not in NP. We can say that such $X$ is even "harder" then $NP$, such that any hard problem in $NP$ reduces to it, but itself is not in (or harder than) $NP$.

The below Venn's Diagram should be helpful to understand the relation of these definitions.


:::{figure} np-p-and-np
<img src="../imgs/np-p-and-np.png" width = "50%" alt=""/>

Relation between $P, NP$ and related concepts
:::

Claims
: - Let $X$ be an $NP$-complete problem, then $P=NP$ iff there is an efficient algorithm for $X$.
  - Let $X$ be an $NP$-hard problem, then $P=NP$ if there is an efficient algorithm for $X$. Reverse does not hold.


:::{admonition,dropdown,seealso} *Proof*

If $X$ is an $NP$-complete problem,

- $(\Rightarrow)$ If $P=NP$, then $X \in P$, so there is an efficient algorithm for $X$.

- $(\Leftarrow)$ If there is an efficient algorithm for $X$, then there is an efficient algorithm for all $Y \in NP$, then $P\subseteq NP$.

If $X$ be an $NP$-hard problem,

- $(\Leftarrow)$ If there is an efficient algorithm for $X$, then there is an efficient algorithm for all $Y \in NP$, then $P\subseteq NP$.

:::


To prove a problem $Y$ is $NP$-complete, we can use one of the following method,

- **Method 1**: Prove $Y\in NP$ and $Y$ is $NP$-hard (definition of $NP$-complete)
- **Method 2**: Prove $Y \in NP$, and find an $NP$-complete problem $X$ such that $X \le_p Y$ (use transitivity polynomial reduction)

Method 2 is usually used. But how to find an $NP$-complete problem $X$?

Theorem (Cook-Levin)
: SAT is a $NP$-complete problem.

Idea: use the SAT formula to encode the run of the non-deterministic Turing machine.

Therefore, SAT can be used as the $X$ in Method 2.


Claim
: 3SAT is $NP$-complete.

:::{admonition,dropdown,seealso} *Proof*

By method 2, we prove

1. 3SAT is $NP$: This can be proved if we can find a certificate and a verifier. Given input formula $\phi$, the certificate is an assignment to variables of $\phi$, and the verifier checks that the assignment satisfies the formula.

1. SAT $\le_p$ 3SAT: Given instance $\phi$ of SAT, we efficiently produce instance $\phi ^\prime$ of 3SAT such that $\phi$ is satisfiable iff $\phi ^\prime$ is satisfiable.

  Let $\phi$ be an instance of SAT formula, $\phi = c_1 \wedge c_2 \wedge \ldots \wedge c_m$.


:::

Corollary
: IS, VC, SC are $NP$-complete.







Theorem ($k$-coloring)
: $k$-coloring is $NP$-complete.


Given a graph $G$, assign to each vertex one of $k$ colors, such that no adjoint vertices are assigned different colors.

Decision version: Given a graph, is it $k$-colorable?

For instance, bipartite graph is 2-colorable. To find whether a graph is 2-colorable, ...



.


.


.


.


.


.


.


.
