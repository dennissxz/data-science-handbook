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


### 1-d Table

Let $T_n$ be the optimal number of scalar multiplications for the sequence. Consider the last matrix in the sequence, $A_n$.

- If $A_n$ is the last to be multiplied, then $T_n = T_{n-1} + d_0 d_{n-1} d_n$

- Else $A_n$ is not the last to be multiplied, then $T_n = ...$

Hard to figure out the subproblem.

### 2-d Table

Let's try a two dimensional DP table. Let $T_{i,j}$ where $i\le j$ be the number of scalar multiplications in computing the product of the sequence $A_i, A_{i+1}, \ldots, A_j$.

It is easy to find the base cases,

$$\begin{aligned}
T_{i,i} &= 0 \\
\end{aligned}$$

How about other cases? It must be $(A_i \ldots A_k)\times (A_{k+1} \ldots A_j)$ for some $k$, with $d_{i-1} d_k d_j$ times of multiplication. Hence, the iterative equation is

$$T_{i,j} = \min_{k \in \left\{ i, i+1, \ldots, j \right\}} \left\{ T_{i,k} + T_{k+1,j} + d_{i-1} d_k d_j  \right\}$$

Note the computational order. To compute entry $T_{i,j}$, we must know all entries $(T_{i,1:j-1})$ on the **left**, and all $(T_{i:j-1,j})$ entries **below**. Therefore, to fill this upper triangular table, we can either

- start from bottom right $T_{n,n}$, fill by row from left to right, or
- start from top left $T_{1,1}$, fill by column from bottom to top, or
- fill by diagonal, from $+1$ diagonal to $+(n-1)$ diagonal, which is equivalent to fill by length of sub-array.

## Solution

For each pair $i,j$ such that $1 \le i \le j \le n$, we define $T_{i,j}$ be the minimum number of multiplications for computing $A_{i} \times A_{i+1} \times \cdots \times A_{j}$.

- Initialize $T_{i,i} = 0$ for each $i$.

- For $i=n-1$ to $1$ (fill by row from bottom)
  - For $j=i+1$ to $n$

      $$T_{i,j} = \min_{k \in \left\{ i, i+1, \ldots, j \right\}} \left\{ T_{i,k} + T_{k+1,j} + d_{i-1} d_k d_j  \right\}$$

- Return $T_{1,n}$

Filling by diagonal is:

- For $s=1$ to $n-1$ (diagonal)
  - For $i=1$ to $n-s$ (row)

    $$
    T_{i,i+s} = \min_{k \in \left\{ i, i+1, \ldots, i+s \right\}} \left\{ T_{i,k} + T_{k+1,i+s} + d_{i-1} d_k d_{i+s}  \right\}
    $$


To track the solution, we create a label function to store the chosen $k^*$ in each recursive step.

## Complexity

In total $O(n^2)$ entries. In each entry, the `min` step takes need $O(n) time.

Total $O(n^3)$.
