# Largest Sum Subsequence


## Problem

Input
: A sequence of $n$ numbers $A = (a_1, a_2, \ldots, a_n)$, where $a_n \in \mathbb{R}$.

Objective
: Find a **continuous** subsequence that has the largest sum.

## Analysis


Special cases:

- If the array contains all non-negative numbers, then the problem is trivial; a maximum subarray is the entire array.

- If the array contains all non-positive numbers, then a solution is any subarray of size 1 containing the maximal value of the array (or the empty subarray, if it is permitted).

A continuity constrain and the mixture of positive and negative makes this problem a little involved.

Let $S_k$ be a subsequence **ending** at $a_k$ that has the largest sum.

$$
S_k = \underset{1 \le i \le k}{\operatorname{maxsum}}   \left\{ S_{[i:k]}\right\}
$$


We can consider the last number $a_k$. To obtina $S_k$, we can either add it to $S_{k-1}$, or discard $S_{k-1}$ and keep only $a_k$, whichever has a larger sum.

```{margin} Takeaway
It can be the case that the value in the iterative relation is not OPT.
```

Hence, the iterative relation for $S_k$ is

$$
S_k = \operatorname{maxsum}  \left\{ S_{k-1} \circ a_k, \left\{ a_k \right\} \right\}
$$

The corresponding iterative relation for the sum is

$$
C_k = \max \left\{ C_{k-1} + a_k, a_k \right\}
$$

so the choice only depends on $C_{k-1} >0$ or not.

The base case is $S_1= \left\{ a_1 \right\}$ if $a_1 > 0$, or $0$ otherwise.

The final output is NOT $S_n$. The optimal sequence can end anywhere. Thus,

$$OPT = \max _{1 \le i \le n} \left\{ S_i \right\}$$

## Solution

The algorithm is

---
**Largest Sum Subsequence**

---
- Initialize $C=0, OPT=0$

- For $k=1$ to $n$, // find $C_k$

  - if $C>0$
    - $C = C + a_k$
  - else
    - $C = a_k$
    - $\text{start}  = k$ // record the start index
  - if $C>OPT$ // find a larger sum
    - $OPT = C$
    - $\text{end}  = k$ // record the ending index

- Return $OPT, \text{start} , \text{end}$
---

The optimal subsequence is $A_{[\text{start}: \text{end}]}$.

## Complexity

$O(n)$
