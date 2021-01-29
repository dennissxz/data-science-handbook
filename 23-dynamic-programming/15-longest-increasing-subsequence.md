# Longest Increasing Subsequence


## Problem

Input
: A sequence of $n$ numbers $A = (a_1, a_2, \ldots, a_n)$.

Objective
: Find a subsequence that is monotonically increasing (non-decreasing).

## Algorithm

Let $L_n$ be the length of the longest increasing subsequence **ending** at $A_{[n]}$

$$
L_k = 1 + \max _{1 \le i \le k-1, A_{[i]}\le A_{[k]}} L_i
$$

Initialization

$L_1=1$


## Running Time

There are $n$ iteration. Each iteration $O(n)$.

Total $O(n^2)$.
