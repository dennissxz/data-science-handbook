# Weighted Interval Scheduling

## Problem

Input
: A set of $n$ jobs. For each job $j$, it has start time $s_j$ and finish time $f_j$, and profit $p_j \ge 0$.

Goal
: Maximize total profit of scheduled jobs.

Constraint
: No two scheduled jobs can overlap.


## Analysis

For now, we compute total profit of the solution.

Sort jobs by $f_j$ from smallest to biggest. We obtain a sorted set of job indices $J = \left\{ j_1, \ldots, j_n \right\}$ such that $f_{j_1} \le f_{j_2}, \ldots, \le f_{j_n}$.

The subproblem can be on $J_k = \left\{ {j_1, \ldots, j_k} \right\}$.

Let $P()$ be the total profit of a set of jobs. For the last job $j_k$ in $J_k$, consider two cases.

- $j_k$ is not included, then the total profit is $P(J_{k}) = P(J_{k-1})$

- $j_k$ is included. Let $J ^\prime _k \subseteq J_k$ be all jobs that don't overlap with $j_k$. Then the total profit is $P(J^\prime _k) + p_k$

Let $i_k$ be the largest job index in $J_k$ that does not overlap with $j_k$.

$$
i_k = \underset{f_i \le s_{j_k}}{\operatorname{argmax}} \, i
$$

The iterative relation is

$$P(J_k) = \max \left\{ P(J_{k-1}), P(J_{i_k}) + p_k \right\}$$

## Solution

Define a DP table with $k$-th entry $T_k$ being the optimal profit of to the problem on $J_k$.

- Initialize $T_0 = 0$

- For $1 \le k \le n$, find $i_k$

- For $1 \le k \le n$, compute

    $$T_k = \max \left\{ T_{k-1}, T_{i_k} + p_k \right\}$$

- Return $T_n$


## Complexity

- Sorting: $O(n \log n)$
- For every job $k$, compute $i_k$. Total $O(n^2)$, can be optimized to $O(n)$.
- Table
  - $O(n)$ entries
  - $O(1)$ time per entry to compute the entry value

Total $O(n \log n)$.


## Track Solution

One can store the solution in the algorithm. For instance, store the choice in the $\max$ comparison, if choose the second one then append job $p_k$ to a list, or just record $k$ in every step and then do indexing on $J$.

Another way is to start from $T_n$, and compare $T_{n-1}$,

- if same then go to entry $T_{n-1}$
- if different, pop out $p_n$, and then go to entry $T_{i_k}$

Repeat until go to $T_0$.
