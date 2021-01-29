# Minimum Knapsack

## Problem

Input
: There are $n$ items $X_1, \ldots, X_n$. Each item has a value $v_i$ and weight $w_i$.

Objective
: Choose a subset of items that maximize value while adhering to weight constraint $W$.


$$\begin{aligned}
\max &\  _{S\in \{1, \ldots, n\}}  \sum_{i \in S} v_i \\
\text{s.t.} &\  \sum_{i \in S} w_i \le W
\end{aligned}$$



## Algorithm
