
# Interval Scheduling

Aka. job interval selection problem (JISP)

How to schedule jobs

- input
  - a set $J$ of $n$ jobs
  - start time $s_j$ and finish time $f_j$ of each job $j$ so that job $j$'s interval is $(s_j, f_j]$

- goal
  - schedule as many jobs (intervals) as possible
  - aka. max independent set of intervals

- constraints
  - no two jobs (intervals) can overlap in time

## Greedy Algorithm Solutions

Start with an empty set $s=\emptyset$.
While $J\ne \emptyset$, do
- choose a job $j$ by a greedy rule, to be discussed below
- add $j$ to $S$
- delete from $J$ all jobs that overlap with $j$ (include $j$ itself)

What greedy rule should we use?
- shortest job
  - cons: the shortest job could be in the middle and rule out all of the others by chance, e.g. $(0.9, 1.1]$ rules out $(0,2], (1,2]$.
- leftmost left endpoint
  - cons: one job overlaps many jobs
- leftmost right endpoint
  - optimal, to be proved below

## Correctness Proof

Two steps in correctness proof:
  - feasible, i.e. satisfy the constraints
  - optimal. optimal set is denoted as $OPT$.

Feasibility is guaranteed since we delete all jobs that overlap with $j$ from $J$.

Optimality: $\vert S \vert = \vert OPT \vert$.

Intuition is that, we are pushing intervals to the left
Method: prove $S$ is approaching $OPT$ gradually.

### Proof 1: Exchange Argument

Want to prove $\vert S \vert \ge \vert OPT \vert$.

Suppose the first job in $OPT$ is $j_1^*$ and the first job in $S$ is $j_1$. According to the greedy rule, the finish time of $j_1$ is ealier than $j_1^*$, so substituting $j_1^*$ by $j_1$ in $OPT$ does not affect the optimality of $OPT$. Call the updated $OPT$ $OPT_1$. We can then substitute $j_2^*$ in $OPT_1$ by $j_2$ in $S$, and so on.

What if $\vert S \vert < \vert OPT_k \vert$, i.e. we run out of jobs in $S$ after $k$ substitutions? This is impossible. Our algorithm will select $j$ in $J$ until $J$ is empty. So if there is any jobs in $OPT_k$ that do not in $S$, they should be added to $S$ by our algorithm.


### Proof 2: one-to-one mapping from $OPT$ to $S$

Claim 1: $g(j_i^*) = j$

If $j_i^* \in S$ then $g(j_1^*) = j_1^*$. Otherwise there is a job $j^\prime \in S$ whose right endpoint lies in $(s_{j_i^*}, f_{j_i^*}]$.

Claim 2: each job in $S$ is responsible?? for at most one job in $OPT$.

In sum, the mapping is one-to-one, such that $\vert OPT \vert = \vert S \vert$

## Analysis of Running time

at most $n$ iterations

$O(n)$ time per iteration

so $O(n^2)$ total
