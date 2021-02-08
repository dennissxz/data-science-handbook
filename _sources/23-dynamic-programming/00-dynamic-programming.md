# Dynamic Programming

Dynamic programming is a multi-stage decision process. In every stage, we solve a sub-problem, which can be used to solve the original problem.

Note that dynamic programming only applies to problems that satisfy the **optimization principle**:

- Any sub-output in an optimal output to the original input is optimal to the corresponding sub-input.

We rely this principle to solve the original problem by solving its subproblems.

Usually we use a table to store the optimal values of the subproblems, so the space complexity is larger than the recursive solution

## Fibonacci Example

We take Fibonacci numbers as an example, and compare three methods.

$$\begin{align}
F(1) &= 1\\
F(2) &= 1 \\
F(i) &= F(i-1) + F(i-2) \quad \text{for} \ i > 2
\end{align}$$

### Recursive

For the function $Fib(n)$

If $n=1$ or $n=2$, return $1$

Otherwise, return $Fib(n-1) + Fib(n-2)$

**Running time**

Let $N(i)$ be the number of function evaluations of $Fib(i)$ when we call $F(n)$.

$N(n)=1, N(n-2)=1, N(n-2)=2, N(i) = N(i+1) + N(i+2)$

It's a Fibonacci sequence again. So the running time is at least $O(F(n))$.

Since $F(i) \ge 2F(i-2)$, we get $F(n) \ge 2^ {\Omega(n)}$, i.e. exponential running time.

### Memoization

Memoization means we record the results and use it, so that we don't need to re-evaluate it again when it is needed.

In this case, for any $i$, after $Fib(i)$ is called for the first time, we record the result. In each subsequent call, return its recorded result.

As a result, $Fib(i)$ is evaluated only $O(1)$ time when we call $Fib(n)$. The running time for the memoization approach is $O(n)$.

### Dynamic Programing

In DP, there is a DP table $T$ that stores the computed value to avoid re-evaluation.

It is **iterative** (use stored results), not **recursive** (call itself).

In the Fibonacci case, the table has only one row (column) $T(i)$.

It's known that $T(1) = T(2) = 1$. Then for $i=3$ to $n$, compute

$$
T(i) = T(i-1) + T(i-2)
$$

with $T(1) = T(2) = 1$.


diff with memoization??

## Analysis

To design a dynamic programming algorithm, there are elements: Subproblems, Recursive relation, Computation

### Subproblems

Subproblems are characterized by some parameters of the input, such as length, size, constraints etc.

Think about

- Is there any hierarchical structure in the input?
- What is the last computation step?
- Does the subproblem satisfy the optimization principle?


:::{admonition,tip} Tip

1. Usually, for an input of a 1-d array $A_{1:n}$, the input to subproblem can be

    - a subarray from the beginning to entry $j<n$, $A_{1:j}$
    - a subarray from entry $i\ge 1$ to entry $j<n$, $A_{i:j}$

1. For an input that is a tree, the sub-input can be a subtree.

    For instance, if the input to the original problem is a tree rooted at note $r$, then the input to the subproblem is a tree rooted at node $u$, which is a child of $r$.

1. If the input has two parameters $a,b$, then the subproblem may also have two parameters $c,d$ where $c<a, d<b$.

:::


### Recursive Relation

Once we define a subproblem, we can figure out the iterative relation between the optimal solution to the subproblems and that to the original problem.

Also, we need to define the base cases, which are the smallest/simplest subproblems.


:::{admonition,tip} Tip

Usually, if the original problem involves "maximum", "largest" or "longest", then the iterative relation involves $\max$ comparison of value, size, length, etc.

:::

### Computation

To compute the final output, we need a dynamic programming table (DP table) to store the values. We need to figure out

- the values of the base cases.
- the order of filling the table, according to the iterative relation
- the required output from the table




## Algorithm

The general structure of a DP algorithm is

Algorithm

1. Define a DP table, $T_{i,j}$ ...

1. Initialize the entries of base cases

1. Compute other entries recursively by the iterative relation

1. Return the required output, an entry in the table

## Proof

Usually proof by induction


## Track Solution

Usually, the DP algorithm only gives the size/length to the solution (array, set, path). To find the actual solution, we need to track the which value is selected in the comparison iterative relation, using some label function.

## Complexity

For time complexity, we need to consider both the DP table and the tracking step, usually the latter is smaller than the former.

- How many entries in the DP table?
- In each entry, what's the complexity to compute the value according to the iterative relation?
  - Some iterative relation only require a two-value comparison, which is $O(1)$
  - Some iterative relation involve comparison over a bunch of values, whose size depends on some input parameter.

## Exercise

https://people.cs.clemson.edu/~bcdean/dp_practice/
