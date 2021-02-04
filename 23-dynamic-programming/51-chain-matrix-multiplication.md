# Chain Matrix Multiplication

## Problem

Input
: $n$ matrices $A_1, \ldots, A_n$ with dimensions $(d_0, d_1), (d_1, d_2), \ldots, (d_{n-1}, d_n)$.

Goal
: The total product $A_1 \cdot A_2, \ldots, A_n$ with minimum number of scalar multiplications.




## Analysis

Recall the matrix multiplication $A_{m \times n} \times B_{n \times p}$ takes $m\times n\times p$ number of scalar multiplications.

For three matrices, there are two ways to compute their product

- $\left( A_{m \times n} \times B_{n \times p} \right) \times C_{p \times q}$ takes $mnp + mpq = mp(n+q)$ number of scalar multiplications

- $A_{m \times n} \times \left(  B_{n \times p} \times C_{p \times q} \right)$ takes $mnq + npq = nq (m+n)$ number of scalar multiplications

Basically, we want to construct an agglomerative tree that represents the order of computation.


**Trial 1**

Let $T_n$ be the optimal number of scalar multiplications for the sequence. Consider the last matrix in the sequence, $A_n$.

- If $A_n$ is the last to be multiplied, then $T_n = T_{n-1} + d_0 d_{n-1} d_n$

- Else $A_n$ is not the last to be multiplied, then $T_n = ...$

Hard to figure out the subproblem.

**Trial 2**

Let's try a two dimensional DP table. Let $T_{i,j}$ where $j\le j$ be the number of scalar multiplications in computing the product of the sequence $A_i, A_{i+1}, \ldots, A_j$.

It is easy to find the base cases,

$$\begin{aligned}
T_{i,i} &= 0 \\
T_{i,i+1} &= d_{i-1} d_{i} d_{i+1} \\
\end{aligned}$$

How about other cases? It must be $(A_i \ldots A_k)\times (A_{k+1} \ldots A_j)$ for some $k$, with $d_{i-1} d_k d_j$ times of multiplication. Hence, the iterative equation is

$$T_{i,j} = \min_{k \in \left\{ i, i+1, \ldots, j \right\}} \left\{ T_{i,k} + T_{k+1,j} + d_{i-1} d_k d_j  \right\}$$

## Solution


## Complexity

In total $O(n^2)$ entries. In each entry, the `min` step takes need $O(n) time.

Total $O(n^3)$.
