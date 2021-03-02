# Polynomial Reduction

In reality, we have a huge class of problems from various areas. If we solve one of them, we solve all of them. If we prove one of them does not have an efficient algorithm, then all of them do not. This is the class of NP-hard problems.

We first introduce polynomial reduction between problems, then define NP-hardness and NP-completeness.


Definition (Polynomial reduction)
: Polynomial reduction from problem $y$ to problem $x$ means that if we solve $x$, then we solve $y$. It is denoted as $y \le _p x$.

To solve $y$, suppose there is a black-box that solves problem $x$.

- Algorithm for problem $y$ can write inputs to problem $x$. For each such input, the black-box gets a solution to problem $x$ in 1 time step.
- Time to write down input to problem $x$ is counted in the run time for problem $y$

We say there exists a polynomial-reduction from problem $y$ to $x$ iff there exists an algorithm for problem $y$ that solves it efficiently (time for writing inputs to $x$+ time for obtaining solution to $x$), given a black-box access to problem $x$.

If $y \le _p x$ and there exists an efficient algorithm $x$ (not black-box), then there exists an efficient algorithm for $y$.



## Examples

### Independent Set $\le_p$ Vertex Cover

Independent set

### 3SAT $\le_p$ Independent Set

SAT


###  Vertex Cover $\le_p$ Set Cover

## Transitivity of Reduction


$$
\text{3SAT}  \le_p \text{Independent Set}  \le_p \text{Vertex Cover}  \le_p \text{Set Cover}  
$$
