# Longest Increasing Subsequence


## Problem

Input
: A sequence of $n$ numbers $A = (a_1, a_2, \ldots, a_n)$.

Objective
: Find the longest monotonically increasing (non-decreasing)  subsequence.

## Analysis

The key is to fix the ending number.

Let $S_k$ be the longest increasing subsequence **ending** at number $a_k$. Then we have the iterative relation

$$
S_k = \left\{ \underset{i:\ 1 \le i \le k-1, a_i\le a_k}{\operatorname{longest}} S_i \right\} \cup {a_k}
$$

The corresponding iterative relation for the length of $S_k$, denoted $L_k$, is

$$
L_k = \left\{ \underset{i:\ 1 \le i \le k-1, a_i\le a_k}{\operatorname{max}} L_i \right\} + 1
$$

The DP table stores $L_k$.

## Solution

- Initialize $L_1 = 1$.

- For $k = 2$ to n, compute

    $$
    L_k = \left\{ \underset{i:\ 1 \le i \le k-1, a_i\le a_k}{\operatorname{max}} L_i \right\} + 1
    $$

- Return $L_n$

## Running Time

There are $O(n)$ iteration. Each iteration $O(n)$.

Total $O(n^2)$.
