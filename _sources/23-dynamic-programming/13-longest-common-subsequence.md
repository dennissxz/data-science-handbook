# Longest Common Subsequence



Definition (Subsequence)
: Let $S = (s_1, s_2, \ldots, s_n)$ and $x = (x_1, x_2, \ldots, x_n)$. We say $X$ is a sub sequence of $S$ iff $\forall 1 \le j \le k, \exists i_j$ such that $1 \le i_1 < i_2 < \ldots, < i_k \le n, x_j = s_{i_j}$.

## Problem

Input
: $A$ is a sequence of length $n$, $B$ is a sequence of length $m$.

Goal
: Find the longest sequence $X$ that is a subsequence of both $A$ and $B$. $X = LCS(A,B)$.

## Algorithms

For now, we only compute the length, not the exact LCS $X$.

- If the last two letters of the two sequences are the same, i.e., $A_{[-1]} = B_{[-1]} = z$, then $z$ is the last letter of $LCS(A, B)$ (proof by contradiction), and hence

$$LCS(A,B) = LCS(A_{[:-1]}, B_{[:-1]}) \circ z$$

- If the last two letters of the two sequences are difference, i.e., $A_{[-1]} \ne B_{[-1]}$, then


$$
LCS(A,B) = \arg \operatorname{longer}  \left\{\begin{array}{ll}
LCS(A, B_{[:-1]}) \\
LCS(A_{[:-1]}, B)
\end{array}\right\}
$$

### DP Table

For all $0 \le i \le n, 0 \le j \le m$, the table entry $T_{[i,j]}$ stores the length $\left\vert LCS(A_{[:i]}, B_{[:j]}) \right\vert$

### Order of computation

In non-decreasing order of $i+j$. Because $T_{[i,j]}$ depends on three entries $T_{[i-1, j-1]}, T_{[i, j-1]}, T_{[i-1, j]}$.

Another order can follow fixed $i$ and increasing $j$. Any order works as long as a new value is computed from existing values.

Base
: For all $0 \le i \le n, 0 \le j \le m$,
$$
T_{[i,0]} = 0 \\
T_{[0,j]} = 0
$$

Step
: Compute $T_{[i,j]}$ for $i,j>0$


$$
T_{[i,j]} =  \left\{\begin{array}{ll}
T_{[i-1,j-1]} + 1 & \text{if } A_{[i]} = B_{[j]}  \\
\max \left\{\begin{array}{ll}
T_{[i-1, j]} \\
T_{[i, j-1]}
\end{array}\right\}
& \text{otherwise}
\end{array}\right.
$$

### Backtrace

If the condition $A_{[i]} = B_{[j]}$ holds, this means we pop out $A_{[i]}$.

??30:00


## Running Time

In total $n\times m$ entires, time for each entry is $O(1)$, so total time is $O(nm)$.

## Correctness

Claim
: For all $0 \le i \le n, 0 \le j \le m$, the table entry $T_{[i,j]}$ stores the length $\left\vert LCS(A_{[:i]}, B_{[:j]}) \right\vert$. The final solution is $T_{[n,m]} = \left\vert LCS(A, B) \right\vert$

*Proof*
: Induction over $i+j$.
