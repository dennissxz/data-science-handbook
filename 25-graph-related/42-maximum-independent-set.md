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

For a tree, how to defined a subgraph? Is there any natural hierachical relation in a tree?

Ans: subtree.

First, we arbitrarily assign a node to be the root.

For a node $v$, the 
