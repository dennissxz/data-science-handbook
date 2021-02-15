# Maximum Independent Set in Trees

Tags: dynamic programming.

Definition (Independent set of a graph)
: A set of vertices $S\subseteq V$ is an independent set of graph $G$ if there is no edge between any pair of vertices in $S$.

## Problem

Input
: Tree $T$

Output
: An independent set of $T$ with maximum size.


## Analysis

The subproblem should be on the subgraph.

*Q: For a tree, how to defined a subgraph? Is there any natural hierarchical relation in a tree?*

*A: subtree, child, grand-child, parent*

We guess that, for a tree with root $r$, the maximum independent tree, denoted $S(r)$, has relation to the subtrees rooted at the children of $r$.

Besides, it is easy to see that neighbors cannot be in the set. And grand-child reflects this. So $S(r)$ also has relation to the subtrees rooted at the grand-children of $r$.

Therefore, the iterative relation for $S(r)$ is

$$
S(r) = \operatorname{larger}  \left\{  \cup S(\text{child}) ,   (\cup S(\text{grand-child})) \cup \left\{ r \right\}  )  \right\}
$$

Note that in the first case, the root itself is not in the set.

The DP table has $n$ entries. Each entry corresponds to a node $u$, and stores the size of the maximum independent set of the tree rooted at $u$, i.e. $\left\vert S(u) \right\vert$.

## Algorithm

Let entry $A_u$ be the size of the largest independent set of subtree rooted at node $u$.

- Initialization: $A_u = 1$ for each leaf $u$.

- Bottom up in tree $T$, for each vertex $u$ such that $\left\{ A_v \right\}_{v \in T_u  }$ are already computed,

$$
A_u = \max \left\{ \sum_{c \in \operatorname{children} (u)} A_c, 1+\sum_{g \in \operatorname{grand-children} (u)} A_g, \right\}
$$

- Return $A_r$.

## Complexity

The table $A$ consists of $n$ entries. For each entry, it takes $O(n)$ time to compute.

So total is $O(n^2)$

There are methods to improve it to $O(n)$.
