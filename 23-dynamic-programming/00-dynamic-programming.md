
# Dynamic Programming 

We take Fibonacci numbers as an example.

$$\begin{align}
&F(1) &= 1\\
&F(2) &= 1 \\
\text{for} \ i > 2,\, & F(i) &= F(i-1) + F(i-2)
\end{align}$$

Example: every rabbit needs one year to mature, and every matured rabbit produces a rabbit at a year.

## Functions

### Recursive

Fib(n)

If $n=1$ or $n=2$, return $1$

Otherwise, return $Fib(n-1) + Fib(n-2)$

Running time:
Let $N(i)$ be the number of function evaluations of $Fib(i)$ when we call $F(n)$.

$N(n)=1, N(n-2)=1, N(n-2)=2, N(i) = N(i+1) + N(i+2)$

It's a Fibonacci sequence again. So the running time is at least $O(F(n))$.

Since $F(i) \ge 2F(i-2)$, we get $F(n) \ge 2^ {\Omega(n)}$, i.e. exponential running time.

### Memoization

Memoization means we record the results and use it, so that we don't need to re-evaluate it again when it is needed.

In this case, for any $i$, after $Fib(i)$ is called for the first time, record the result. In each subsequent call, return its recorded result.

As a result, $Fib(i)$ is evaluated only $O(1)$ time when we call $Fib(n)$. The running time for the memoization approach is $O(n)$.

### Dynamic Programing

In DP, there is a DP table $T$ that stores the computed value to avoid re-evaluation.

It is iterative (use stored results), not recursive (call itself).

In the Fibonacci case, the table has only one row (column) $T(i)$.


## Summary

consider all sub-cases

find iterative formula, usually use $\max$ for maximization problem.

design table

order of entry filling that meet the iterative formula (already stored)

determine which entry should be the output

proof by induction

trace back to find schedule/path
