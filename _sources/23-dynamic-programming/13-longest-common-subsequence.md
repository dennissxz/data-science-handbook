# Longest Common Subsequence



Definition (Subsequence)
: Let $S = (s_1, s_2, \ldots, s_n)$ and $x = (x_1, x_2, \ldots, x_n)$. We say $X$ is a sub sequence of $S$ iff $\forall 1 \le j \le k, \exists i_j$ such that $1 \le i_1 < i_2 < \ldots, < i_k \le n, x_j = s_{i_j}$.

## Problem

Input
: $A$ is a sequence of length $n$, $B$ is a sequence of length $m$.

Goal
: Find the longest sequence $X$ that is a subsequence of both $A$ and $B$. $X = LCS(A,B)$.

## Algorithms

- If the last two letters of the two sequences are the same, i.e., $A[n] = B[n] = z$, then $z$ is the last letter of $LCS(A, B)$ (proof by contradiction), and hence

$$LCS(A,B) = LCS(A[:n-1], B[:n-1]) \circ z$$

- If the last two letters of the two sequences are difference, i.e., $A[n] \ne B[n]$, then

$$
LCS(A,B) = \max \left\{  LCS(A, B[:n-1]),  LCS(A[:n-1], B) \right\}
$$

Therefore,
