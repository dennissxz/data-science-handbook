# Weighted JISP

## Problem

Input
: Set $J$ of $j$ jobs, each with start time $s_i$ and finish time $f_j$, and profit $p_j \ge 0$.

Goal
: Maximize total profit  of scheduled jobs.

Constraint
: No two scheduled jobs can overlap.


## Algorithms

### Greedy Rule

Scheduling by $f_j$ may not be optimal.

### Dynamic Programming

Sort jobs by $f_j$ from smallest to biggest.

$J = \left\{ j_1, \ldots, j_n \right\}$ such that $f_{j_1} \le f_{j_2}, \ldots, \le f_{j_n}$.

For now, we compute solution value.


If $\left\vert J \right\vert = 1$, return $$

Let $j_n$ be the job with rightmost right endpoint.

Consider two cases.

- $j_n$ is not included, then total profit is
  $S_1 = JISP(J \backslash \left\{ j_n \right\})$

- $j_n$ is included. Let $J ^\prime \subseteq J$ be all jobs that don't overlap with $j_n$. Then the total profit is
$S_2 = JISP(J^\prime) + p_n$

Return $\max(S_1, S_2)$.

### Implementation

Add a job $j_0$ with $s_{j_0} = 0, f_{j_0} = 0, p_{j_0} = 0$.

For $0 \le i\le n$, let $\Pi(i)$ be consecutive job set $\left\{ j_0, \ldots, j_i \right\}$.

For $1 \le i\le n$, let $Prev(i)$ be the largest job index that does not overlap with $j_i$. [detail].

We have

$JISP(\Pi(n)) = \max \left\{ JISP(\Pi(n-1)), JISP(\Pi(Prev(n))) + p_{j_n} \right\}$

Let $T(i)$ be the optimal solution value to problem $T(i) = JISP(\Pi(i))$.

Base: $T(0) = 0$,
Step: $T(i) \leftarrow \max \left\{ T(i-1), T(Prev(i)) + p_{j_i} \right\}$

The final solution is in entry $T(n)$

### Running Time Analysis

- Sorting: $O(n \log n)$
- For every job $i$, compute $Prev(i)$. Total $O(n)$ ??
- Table
  - $O(n)$ entries
  - $O(1)$ time per entry to compute the entry value

Total $O(n \log n)$.

## Proof

### Optimal

Let $OPT(i)$ be the optimal solution value of job set $\left\{ j_0, \ldots, j_i \right\}$. Want to prove $T(i) = OPT(i)$.

Base: $i=0, T(0) = 0, OPT(0) = 0, correct$

Step: for $i >0$, assume correctness for all values, i.e., $T(i) = OPT(i)$.

Proof for step

If $j_i \in S$, then

$$
OPT(i) = Profit(S) = OPT(i-1)
$$

otherwise $f_i \in S$, then

$S \ {i}$ is an optimal solution to $\Pi(Prev(i))$.

We have

$$
OPT(i) = Profit(S) = OPT(Prev(i)) + p_{j_i}
$$

$T(i) \leftarrow \max \left\{ T(i-1), T(Prev(i)) + p_{j_i} \right\}$ is lower bounded by the optimal solution.

$T(i) \ge OPT(i)$

Note $T(i) \le OPT(i)$ since $T(i)$ is a valid solution.

### Find the Schedule

Trace back

e.g. store the solution to $\Pi(i)$ in $T(i)$. But waste space and time (copy paste).
