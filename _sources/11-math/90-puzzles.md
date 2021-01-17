# Puzzles

This section contains some interesting math puzzles.

## 10-digit Number Puzzle

Tags: Jane Street, Trader, 18Q4

There is a 10-digit number. From left to right, the first digit equals the number of 0's in that number, the second digit equals the number of 1's in that number, and so on. What is this number?

### Sol.1 Enumeration by Hand

Suppose this number is $\underline{d_0d_1d_2d_3d_4d_5d_6d_7d_8d_9}$, where

$$d_{i} = \sum_j\mathbb{I}[d_j=i]\in \{0,1,2,\dots,9\}$$

Consider a summation of the digits,

$$S_1 = \sum_{i=0}^{9} d_i$$

Since each digit counts the number of appearances, the summation should be $S=10$.

Note that the summation can also be written as

$$S_1 = \sum_{i=0}^{9} d_i \times i :=S_2$$

Thus, we have

$$S_1 = S_2 = 10$$

Then we find $d_i$ by trial and error by applying the above constraints and the definition. We start from the first digit and follow the reasoning: suppose $d_0=k$, then $d_k\ge 1$. If $d_k = l$, then $d_l\ge 1$.

- If $d_0=9$, then $d_9=1$, then $d_1\ge1$, violating the constraint $S_1=10$
- If $d_0=8$, then $d_8=1$, then we must have $d_1=1$ to satisfy the constraint $S_1=10$, but it violates the definition since 1 appears twice.
- If $d_0=7$, then $d_7=1$, then $d_1=2$, then $d_2\ge 1$, violating the constraint $S_1=10$
- If $d_0=6$, then $d_6=1$, then $d_1=2$ or 3
  - If $d_1=2$, then we have $d_2 = 1$ to satisfy $S_1=10$. Check and find that $S_2=10$, so 6210001000 is a **correct** number.
  - If $d_1=3$, then $d_3\ge1$, violating $S_1=10$

- If $d_0=5$, then $d_5=1$, then $d_1=2,3,4$
  - If $d_1=2$, then $d_2\ge1$
    - If $d_2=1$, then the sum is 9. Increasing other $d$ to satisfy $S=10$ will make $d_1=3$, contradiction
    - If $d_2=2$, then we need an additional $d$ to be 1, but the sum is already 10
  - If $d_1=3$, then $d_3=1$, violating $S_2=10$
  - If $d_1=4$, then $d_4=1$, violating $S_1=10$
- If $d_0=4$, then $d_4=1$, then $d_1=2,3,4,5$
  - If $d_1=2$, then $d_2 = 1, 2$
    - If $d_2=1$, then $S_1 = S_2 = 8$, it's impossible to increase other $d$'s such that the two constraints are satisfied since $3\times d_3\ge 3$
    - If $d_2=2$, then $S_2 = 10$, but $S_1=9$
  - If $d_1=3$, then $d_3=1$, then $S_2=10$ but $S_1=9$
  - If $d_1=4$, then 4 appears twice, contradiction
  - If $d_1=5$, then $S_1=10$ but $S_2=9$
- If $d_0=3$, then $d_3\ge1$, and $d_1\le 6$
  - If $d_1=6$, then $S_1=10$ but $S_2=9$
  - If $d_1=5$, then $d_5=1$, violating $S_2=10$
  - If $d_1\le 4$, then $\sum_{i=2}^9d_i\times i \ge 7$ and $S_2\ge11$


...

Enumeration by hand is tedious. We have found a solution 6210001000 but we have to enumerate all other cases to see if there exists any other solutions.


### Sol.2 Enumeration by Computer

It seems that the time complexity is $O(10^{n-1})$ for a $n$-digit number, but in fact we can optimize it by adding the constraints.

Since $\sum_i i\cdot d_i = 10$ we have $d_i \le \lfloor \frac{10}{i} \rfloor$ for $i \ge 1$. So the time complexity is greatly reduced to $O(\Pi_{i=1}^{n-1} \lfloor \frac{10}{i} \rfloor)$

A python script can be

```python
from itertools import product
for ds in product(range(1, 10), *(range(10 // i + 1) for i in range(1, 10))):
    if sum(ds) == 10 and sum(i * ds[i] for i in range(10)) == 10:
        for i in range(10):
            if ds.count(i) != ds[i]:
                break
        else: # no break
            print("".join(str(d) for d in ds))
```

It turns out that 6210001000 is the unique solution.

### Sol.3 Mathematical Reasoning

In the above solutions we did partial reasoning and partial enumeration. Now we try more rigorous reasoning.

Define a set $A = \{d_i\vert d_i\ge 1\}$. Then the cardinality is

$$\vert A \vert = \sum_i \mathbb{I}[d_i \ge 1] = n - \sum_i \mathbb{I}[d_i=0] =   n - d_0$$

and the sum of the elements in $A$ is  

$$\sum_{d_i \in A} d_i = \sum_i d_i = n$$

Define another set $B = A \backslash \{d_0\} = \{d_i\vert d_i \ge 1, i\ge 1 \}$. Then

$$\vert B \vert = \vert A \vert - 1 = n - d_0 - 1$$

and the sum of the elements in $B$ is

$$\sum_{d_i \in B} d_i = \sum_{d_i \in A}d_i - d_0 = n - d_0$$

Since all the $d_i \ge 1 \ \forall \  d_i\in B$, we derive that there are $n-d_0-2\ge0$ number of $1$'s and one $2$ in $B$. Hence,


$$
\begin{align}
d_1 &= n - d_0 - 2 + \mathbb{I}[d_0=1]  &(1)\\
d_2 &= 1 + \mathbb{I}[d_0=2] &(2) \\
d_i &\le 2 \ \  \forall \ i\ge1 &(3)
\end{align}
$$



We assuming that $n>6$.

- If $d_0=1$, then by $(1)$ we have $d_1=n-2 > 4$, contradicting with $(3)$.
- If $d_0=2$, then $d_1 = n-4 > 2$, contradicting with $(3)$.

Therefore, $d_0 \ge 3$ and $d_{d_0} = 1$. Hence,

$$A = \{d_0, d_1 = n-d_0-2, d_2=1, d_{d_0}=1 \}$$

Recall that $\vert A \vert = n-d_0$. As a result, $d_0 = n-4, d_1 = 2$, and the number is

$$(n-4)210\ldots010\ldots0$$

In particular, if $n=10$, the number is 6210001000.

When $n\le 6$, if $d_0 \ge 3$, then $\vert A \vert =4$ and $n=d_0+4>7$, contradiction. Hence $d_0 = 1$ or $2$.

- If $d_0=1$, then by $(1)-(2)$, $d_1 = n-2, d_2=1$.
  - When $n=4$, the number is 1210.
  - When $n=5$, $d_1=3, d_3=1, \sum_{i\le3}=6>n$, contradiction.
  - When $n=6$, $d_1=4, d_4=1, \sum_{i\le4}=7>n$, contradiction.
- If $d_0=2$, then by $(1)-(2)$, $d_1 = n-4, d_2=2$.
  - When $n-4$, $d_1=0$, the number is 2020.
  - When $n=5$, $d_1=1$, the number if 21200.
  - When $n=6$, $d_1=2$, so $2$ appears three times but $d_2=2$, contradiction.


## Trailing zeros in $n$ factorial

*Find the number of trailing zeros* in $n!$.

It is easy to see the trailing zeros are produced by prime factors $2$ and $5$. If there are $m$ number of factor $2$ and $n$ number of factor $5$ in $n!$, then there will be $\min(m,n)$ number of trailing zeros.

Also note that
- $m>n$, since every $5$ numbers produce a factor $5$ and at least $3$ factors of $2$.
- multiples of 5 like $25, 125$ bring more than one factor of $5$.

Therefore, we conclude that $\min(m,n) = n$. How to count $n$? The easiest way is $\lfloor \frac{n}{5} \rfloor$. But we have to take numbers like $25, 125$ into consideration. The solution is to add $\lfloor \frac{n}{25} \rfloor, \lfloor \frac{n}{125} \rfloor$, etc.

```python
def count_trailing_zeros(n):

    count = 0
    i = 5
    while (n/i >= 1):
        count += n // i
        i *= 5

    return int(count)
```
