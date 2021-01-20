
# Interval Scheduling

Aka. job interval selection problem (JISP)

## Basic Problem


### Problem Setup

How to schedule jobs

Input
: - A set $J$ of $n$ jobs
  - The start time $s_j$ and finish time $f_j$ of each job $j$ so that job $j$'s interval $I(j)$ is $(s_j, f_j]$

Goal
: - Schedule as many jobs (intervals) as possible

Constraints
: - No two jobs (intervals) can overlap in time

### Greedy Algorithm Solution

A solution by greedy algorithms is

- Start with an empty set $S=\emptyset$.
- While $J\ne \emptyset$, do
  - Choose a job $j$ by a greedy rule, to be discussed below
  - Add $j$ to $S$
  - Delete from $J$ all jobs that overlap with $j$ (include $j$ itself)

What greedy rule should we use?
- shortest job $\operatorname{argmin}_{j \in J} \, f_j - s_j$
  - cons: the shortest job could be in the middle and rule out all of the others by chance, e.g. $(0.9, 1.1]$ rules out $(0,2], (1,2]$.
- leftmost left endpoint $\operatorname{argmin}_{j \in J} \, s_j$
  - cons: one job overlaps many jobs
- leftmost right endpoint $\operatorname{argmin}_{j \in J} \, f_j$
  - optimal, to be proved below

### Correctness Proof

Feasibility Claim: No two jobs in $S$ overlap.
:

Feasibility is guaranteed since we delete all jobs that overlap with $j$ from $J$.

Optimality: $\left\vert S \right\vert = \left\vert OPT \right\vert$
: Intuition is that, we are pushing intervals to the left
: formal proof are shown below

#### Exchange Argument

Want to prove $\vert S \vert \le \vert OPT \vert$. Try to show $OPT$ approaches our solution $S$.

***Proof***

Suppose the first job in $OPT$ is $j_1^*$ and the first job in $S$ is $j_1$. According to the greedy rule, the finish time of $j_1$ is ealier than $j_1^*$, so substituting $j_1^*$ by $j_1$ in $OPT$ does not affect the optimality of $OPT$. Call the updated $OPT$ by $OPT_1$. We can then substitute $j_2^*$ in $OPT_1$ by $j_2$ in $S$ (note that $f_{j_2} \le f_{j_2^*}$ by our algorithms), and so on.

What if $\vert S \vert < \vert OPT_k \vert$, i.e. we run out of jobs in $S$ after $k$ substitutions? This is impossible. Our algorithm will select $j$ in $J$ until $J$ is empty. So if there is any jobs in $OPT_k$ that is not in $S$, they should be in $J$ and be added to $S$ by our algorithm.

$\square$

#### Charging Scheme

Given $\left\vert S \right\vert \le \left\vert OPT \right\vert$, want to show $\left\vert S \right\vert \ge \left\vert OPT \right\vert$ such that $\left\vert S \right\vert = \left\vert OPT \right\vert$.

Let the jobs in $OPT$ be $j_1^*, \ldots, j_{\left\vert OPT \right\vert} ^*$ and the jobs in $S$ be  $j_1^*, \ldots, j_{\left\vert S \right\vert} ^*$

Define a one-to-one function from set $OPT$ to set $S$: $g(j_i^*) = j_\ell$. This means every job in $OPT$ is mapped to a job in $S$, but there can be jobs in $S$ that has not mapping from $OPT$. If such function exists then $\left\vert S \right\vert \ge \left\vert OPS \right\vert$. Now we prove such function exists.

Let's look at a job $j_i^*$ in $OPT$.
- If $j_1^* \in S$ then we let the mapping to be an identity mapping $g(j_i^*) = j_i^*$.
- If $j_i^*\notin OPT$, then we deleted it at some time in $J$, since it overlaps from a job in $S$. Let the corresponding job in be $j ^\prime$, then its end time must be earlier than the end time of $j_i^*$, otherwise we won't select $j ^\prime$ and delete $j_i^*$ from $J$. In short, the interval $(s_{j^*}, f_{j^*}]$ must contains $f_{j ^\prime}$.

    In this case, let the mapping be $g(j_i^*) = j ^\prime$.

Moreover, for each job $j ^\prime \in S$, it can take up to one mapping. If it takes two jobs $j_i^*, j_k^* \in OPT$, then both intervals of $j_i^*$ and $j_k^*$ should contain $f_{j ^\prime}$, which is impossible since the intervals in $OPT$ cannot overlap.

In sum, this function $g$ exists. Hence, $\left\vert S \right\vert \ge \left\vert OPT \right\vert$.

$\square$


### Analysis of Running time

There are at most $n$ iterations in `while`, and in every iteration we need to find $\operatorname{argmin}_j \, s_j$, which can be doen in $O(n)$. In sum, the overall time complexity is $O(n^2)$.

Another way of implementation is to sort the jobs by $s_j$. Then there are at most $n$ iterations. The overall time complexity is $O(n \log n)$.


## Extension 1:
