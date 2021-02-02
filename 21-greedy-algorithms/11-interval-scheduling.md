# Interval Scheduling

Aka. job interval selection problem (JISP).

## Maximize #Jobs

There are many jobs, but we only have one machine that can execute at most one job at a time. We want to execute as many jobs as possible.

### Problem

Input
: - A set $J$ of $n$ jobs
  - The start time $s_j$ and finish time $f_j$ of each job $j$ so that job $j$'s interval $I(j)$ is $(s_j, f_j]$

Goal
: - Schedule as many jobs (intervals) as possible

Constraint
: - No two job intervals can overlap in time

### Solution

A solution by greedy algorithms is

---
**Algorithm** Interval Scheduling (Maximize #Jobs)

---
- Start with an empty set $S=\emptyset$.
- While $J\ne \emptyset$, do
  - Choose a job $j$ by a greedy rule, to be discussed below
  - Add $j$ to $S$
  - Delete from $J$ all jobs that overlap with $j$ (include $j$ itself)
---


What greedy rule should we use?
- shortest job $\operatorname{argmin}_{j \in J} \, f_j - s_j$
  - cons: the shortest job could be in the middle and rule out all of the others by chance, e.g. $(0.9, 1.1]$ rules out $(0,2], (1,2]$.
- leftmost left endpoint $\operatorname{argmin}_{j \in J} \, s_j$
  - cons: one job overlaps many jobs
- leftmost right endpoint $\operatorname{argmin}_{j \in J} \, f_j$
  - optimal, to be proved below

### Correctness

Feasibility is guaranteed since we delete all jobs that overlap with $j$ from $J$.

Optimality: $\left\vert S \right\vert = \left\vert OPT \right\vert$. The intuition is that, we are pushing intervals to the left. It can be shown in two ways.

#### Exchange Argument

Want to prove $\vert S \vert \le \vert OPT \vert$. Try to show $OPT$ approaches our solution $S$.

***Proof***

Suppose the first job in $OPT$ is $j_1^*$ and the first job in $S$ is $j_1$. According to the greedy rule, the finish time of $j_1$ is ealier than $j_1^*$, so substituting $j_1^*$ by $j_1$ in $OPT$ does not affect the optimality of $OPT$. Call the updated $OPT$ by $OPT_1$. We can then substitute $j_2^*$ in $OPT_1$ by $j_2$ in $S$ (note that $f_{j_2} \le f_{j_2^*}$ by our algorithms), and so on.

What if $\vert S \vert < \vert OPT_k \vert$, i.e. we run out of jobs in $S$ after $k$ substitutions? This is impossible. Our algorithm will select $j$ in $J$ until $J$ is empty. So if there is any jobs in $OPT_k$ that is not in $S$, they should be in $J$ and be added to $S$ by our algorithm.

$\square$

#### Charging Scheme

Given $\left\vert S \right\vert \le \left\vert OPT \right\vert$, want to show $\left\vert S \right\vert \ge \left\vert OPT \right\vert$ such that $\left\vert S \right\vert = \left\vert OPT \right\vert$.

***Proof***

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


## Minimize #Machines Used


Instead of schedule jobs on one machine, now we can execute all jobs on multiple machines. We want to minimize the number of machines used.

### Problem

Input
: - A set $J$ of $n$ jobs
  - The start time $s_j$ and finish time $f_j$ of each job $j$ so that job $j$'s interval $I(j)$ is $(s_j, f_j]$

Goal
: - Schedule jobs on fewest machine.

Constraint
: - No two jobs (intervals) can overlap in time if they are scheduled to the same machine.

### Solution

We will schedule jobs to machines iteratively. Over the course of the algorithm, we say that a machine is *used* iff at least one job is assigned to it.

---
**Algorithm** Interval Scheduling (Maximize #Jobs)

---
- Sort all jobs in the order of increasing start time, breaking ties arbitrarily.

- Assume without loss of generality that $s_{1} \leq s_{2} \leq \ldots \leq s_{n}$. Process jobs $j_{1}, \ldots, j_{n}$ in this order. Upon processing job $j_i$, schedule it on any used machine that does not contain a job whose interval overlaps with $\left(s_{i}, f_{i}\right]$; if there is no such machine, schedule it on a new machine.
---

### Correctness

Feasibility is immediate. We prove optimality.

Claim 1
: If our algorithm returns a scheduling using $k$ machines, then any feasible schedule has to use at least $k$ machines.

***Proof***

If our algorithm returns a scheduling using k machines, then there are $k$ jobs $j_{i_{1}}, \ldots, j_{i_{k}}$ in $J$, such that some time-point $x$ belongs to every interval in set $\left\{\left(s_{i_{t}}, f_{i_{t}}\right] \mid 1 \leq t \leq k\right\}$. Therefore, these jobs have to be scheduled on different machines in any feasible solution, and the claim then follows.

$\square$


### Complexity

The algorithm consists of at most n iterations. In each iteration we need to schedule a job $j$ on some machine. All this can be done in time $O(n)$ (by checking which scheduled jobs overlap with the current one), so the running time of the algorithm is $O(n^2)$.

## Minimize #Machine Rented

Similar to last problem, but the machines are rented, which means we need to return it at some time points.

## Minimize Maximum Lateness

In this problem, each job has a deadline $d_i$, and we have one machine. We want to execute all jobs, and minimize the maximum lateness of a job.

### Problem

Input
: - A set $J$ of $n$ jobs
  - Duration $t_j$ and deadline $d_j$ of each job $j$.

Goal
: - Schedule all jobs and minimize the maximum lateness, which is $\max \left\{ 0, f_i - d_i \right\}$.

Constraint
: - No two job intervals can overlap.

### Solution

A greedy solution is to execute the job with the earliest deadline, like in real-life setting.

---
**Algorithm** Interval Scheduling (Minimize Maximum Lateness)

---
- Sort all jobs in the order of increasing deadline, breaking ties arbitrarily.

- Schedule them one-by-one in this order with no idle time.

---

### Correctness

Feasibility is immediate. We prove correctness by proving two claims.

Definition (Inverted Pair)
: A pair of indices $(i_1. i_j)$ from $\left\{ 1, 2, \ldots, n \right\}$ is an inverted pair in schedule $S$, iff $d_{i_1} < d_{i_2}$ and job $j_{i_2}$ is scheduled **immediate** before $j_{i_1}$ in $S$.

Claim 2
: All schedules with no inverted pairs and no idle time have the same maximum lateness.

**Proof**

Let $S$ and $S ^\prime$ be any two different schedules with no inversions or idle time. Then $S$ and $S ^\prime$ may only differ in the order between jobs with identical deadlines. Consider any deadline $d$ and let $J_d$ be the set of jobs with the **same** deadline $d$. In both $S$ and $S ^\prime$, jobs in $J_d$ are all scheduled consecutively, and start at the same time $T_d$. Among all jobs in $J_d$, the one that is scheduled last in $S$ has the greatest lateness in $S$, which is $\left(T_{d}+\sum_{j_{i} \in J_{d}} t_{i}\right)-d$. Same for the last job in $J_d$ in $S ^\prime$.

$\square$


Claim 3
: There is an optimal schedule that has no inverted pairs and no idle time.

First, it is easy to see that there is an optimal schedule with no idle time.

Second, if there is an inverted pair $(i_1, i_2)$, we prove that swapping only that pair does not increase maximum lateness. Let the start time of that pair be $T$.

- The original lateness of job $j_{i_1}$ is $\max\{0, T + t_{i_2} + t_{i_1} - d_{i_1}\}$, and after swapping it becomes $\max \left\{0, T + t_{i_1} - d_{i_1} \right\}$.

- The original lateness of job $j_{i_2}$ is $\max \left\{0, T + t_{i_2} - d_{i_2} \right\}$, and after swapping it becomes $\max \left\{ 0, T + t_{i_2}  + t_{i_1} - d_{i_2} \right\}$.

The maximum lateness of the two jobs changes from $\max \left\{ T + t_{i_2} + t_{i_1} - d_{i_1} \right\}$ to $\max \left\{ T + t_{i_2} + t_{i_1} - d_{i_2} \right\}$, which does not increase since $d_{i_1} < d_{i_2}$.

$\square$

It is easy to see that our algorithm computes a schedule with no inverted pairs. From the two claims above, we see the schedule computed by our algorithm is optimal.


### Complexity

We first sort all jobs by their deadlines in time $O(n\log n)$, and then schedule jobs according to this order in time $O(n)$, so the running time of the algorithm is $O(n\log n)$.
