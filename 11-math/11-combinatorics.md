# Combinatorics



*How many ways can we place $n$ balls to $k$ boxes?*

Well, the answer depends on whether the balls and boxes are distinct or identical, and whether we allow some boxes to be empty.

The solutions in various scenarios are summarized below. We will introduce the them one by one.

<p align="center">

| $n$ balls   | $k$ boxes | non-empty | empty | recurrence relation and hint|
| :-: | :-: | :-: | :-: | :-: |
| distinct    | identical       | $S(n,k)$ |  $\sum_{i=1}^k S(n,i)$ |  $S(n,k) = S(n-1, k-1) + kS(n-1, k)$, singleton
| distinct    | distinct      | $S(n,k) \cdot k!$ | $k^n$                   | /|
| identical   | identical     | $P(n,k)$          | $\sum_{i=1}^{k}P(n,i)$  | $P(n,k) = P(n-1,k-1) + P(n-k, k)$ , partner|
| identical   | distinct      | $C_{n-1}^{k-1}$   | $C_{n+k-1}^{k-1}$       | $C(n,k) = C(n-1, k-1) + C(n-1, k)$ , chosen|

</p>

In the "distinct + identical + non-empty" case, if the number of balls in each box, $m_j$, where $m_j \ge 1, \sum_{j=1}^k m_j = n$, are specified, and suppose there are $d$ distinct values of $m_j$, each corresponds to $c_l$ number of boxes, then the number of ways is

$$\frac{C_n^{m_1} C_{n-m_1}^{m_2} \cdots C_{m_k}^{m_k} }{\Pi_{l=1}^{d}\left(c_{l}!\right)} = \frac{n!}{\Pi_{j=1}^{k}\left(m_{j}!\right)}\cdot\frac{1}{\Pi_{l=1}^{d}\left(c_{l}!\right)}$$


The “distinct + identical + non-empty” case with $m_j$ specified and the “distinct + identical” case and their variants are the most frequently tested.

References:
-  https://www.cs.uleth.ca/~holzmann/notes/distributions.pdf



## Distinct Balls and Identical Boxes

The $n$ balls are all distinct (distinguishable).

We first consider the scenario where all $k$ boxes are identical, e.g., the arrangement of two boxes $(\text{box}_1, \text{box}_2)$ is the same as $(\text{box}_2, \text{box}_1)$. This is often called "indistinguishable" boxes.

### Non-empty Boxes

Let $S(n,k)$ where $n \ge k$ be the number of ways. There are two cases that can happen to the first ball:
- The first ball is placed into a box as a singleton. In this case, the remaining $n-1$ balls are placed into the remaining $k-1$ boxes. As a result, the number of ways is $S(n-1, k-1)$.

- The first ball is placed into a box with other balls. In this case, we can first place the remaining $n-1$ balls to the $k$ boxes and then place the first ball into any one of them. As a result, the number of ways is $S(n-1, k)\times k$.

Therefore, the recurrence relation is

$$S(n,k) = S(n-1, k-1) + kS(n-1, k)$$

These numbers $S(n,k)$ are called Stirling numbers of the second kind and are often written as $\left\{\begin{array}{l} n \\ k \end{array}\right\}$. The explicit formula is

$$\left\{\begin{array}{l}
n \\
k
\end{array}\right\}=\frac{1}{k !} \sum_{i=0}^{k}(-1)^{i}\left(\begin{array}{l}
k \\
i
\end{array}\right)(k-i)^{n}$$



In particular, if the numbers of balls in each box are specified, we can find the number of ways by the following analysis.

Let $m_j$ be the number of balls in box $j$.

First, we suppose all values $m_j$ are different. Without loss of generality we assume $m_1>m_2>\cdots>m_k$. Image that you first select $m_1$ balls from the $n$ balls, then select $m_2$ balls from the remaining $n-m_1$ balls, and so on. The number of ways is

$$C_n^{m_1} C_{n-m_1}^{m_2} \cdots C_{m_k}^{m_k} = \frac{n!}{\Pi_{j=1}^k (m_j!)}\cdot$$

Another way to understand this formula is that you first permutate all the $n$ balls. Since there is no order of balls within a group but you have counted them, now you roll out the duplicate counting, which are $m_1!, m_2!, \cdots, m_k!$ respectively.



Now, suppose $c$ boxes have the same number $m_j$ of balls, and the other $k-c$ boxes  have different number of balls. The number of ways is

$$\frac{n!}{\Pi_{j=1}^p (m_j!)!}\cdot \frac{1}{c!}$$

This is simply because the boxes are indistinguishable, so we need to roll out $c!$ number of duplicated counts.


More generally, suppose there are $d$ different values of $m_j$, each corresponds to $c_l$ number of boxes, then the number of ways is

$$\frac{n!}{\Pi_{j=1}^k (m_j!)} \cdot \frac{1}{\Pi_{l=1}^d (c_l!)}$$


### Empty Boxes

If empty boxes are allowed, then it is easy to derive the number of ways is

$$\sum_{i=1}^k S(n,i)$$

### Exercise

1. *How many ways are there to distribute 6 different books into 3 indistinguishable boxes, each of size 1, 2, and 3?*

    ```{dropdown} solution
    $\frac{6!}{3!2!1!}=60$
    ```

1. *How many ways are there to evenly distribute 6 different books into 3 non-empty indistinguishable boxes?*
    ```{dropdown} solution  
    It must be $(2,2,2)$, so $\frac{6!}{2!2!2!} \cdot \frac{1}{3!}=15$
    ```

1. *How many ways are there to distribute 6 different books to 3 non-empty indistinguishable groups?*
    ```{dropdown} solution
    We first find the 3 possible combinations of $(m_1, m_2, m_3)$: $(3,2,1), (2,2,2), (4,1,1)$. Then for each scenario, we count the number of ways.

    $$\frac{6!}{3!2!1!} + \frac{6!}{2!2!2!} \cdot \frac{1}{3!} + \frac{6!}{4!1!1!} \cdot \frac{1}{2!} = 90 = S(6,3)$$
    ```

1. *True/False: To count the number of ways to distribute 8 distinct balls to 3 identical boxes so that each box has at least 2 balls, we can place one ball in each box and this problem reduces to the regular case of distributing $8-3=5$ balls to 3 identical boxes, which is $S(5,3)$*.
    ```{dropdown} solution
    False. We can only do so if the balls are identical.
    ```


## Distinct Balls and Distinct Boxes

### Non-empty Boxes

Distinct boxes means the order maters, e.g., the arrangement of two boxes $(\text{box}_1, \text{box}_2)$ is different from $(\text{box}_2, \text{box}_1)$.

There are basically two steps
1. Separate the balls into $k$ indistinguishable boxes (solved in the previous section)
1. Label the $k$ boxes with order, which brings $k!$ ways of permutations.

In general, the number of ways is

$$S(n,k)\cdot k!$$

If $m_j$'s are specified, the number of ways is

$$\frac{n!}{\Pi_{j=1}^k (m_j!)} \cdot \frac{1}{\Pi_{l=1}^d (c_l!)} \cdot k!$$

### Empty boxes

If empty boxes are allowed, we solve by another way.

For each ball, it can go to one of the $k$ boxes, i.e., there are $k$ distinct options. As a result, consider all $n$ distinct balls, the number of ways is

$$k^n$$


### Exercise

1. *How many ways are there to deal hands of 2 cards to each of 5 players from a deck containing 52 cards?*
    ```{dropdown} solution
    In this case, cards are distinct balls and players are distinct boxes. Since there are $52-5\times2=42$ cards left, we can think that there is an 6-th player who gets that 42 cards. Note that the permutation of the 5 hands of 2 cards only happen to the first 5 players. Applying the formula gives

    $$\frac{52!}{2!2!2!2!2!42!} \cdot \frac{1}{5!} \cdot (6-1)!$$

    A more straightforward way is

    $$C_{52}^2 C_{50}^2 C_{48}^2 C_{46}^2 C_{44}^2$$
    ```

1. *How many ways are there to deal hands of 13 cards to each of 4 players from a deck containing 52 cards so that the youngest player gets the queen of spades $\spadesuit \text{Q}$?*
    ```{dropdown} solution
    We can first deal $\spadesuit \text{Q}$ to the youngest player and then deal the remaining 51 cards. The number of ways is

    $$C_{51}^{12} C_{39}^{13} C_{26}^{13} C_{13}^{13}$$
    ```



## Identical Balls and Identical Boxes

The $n$ balls are all identical (indistinguishable).

### Non-empty Boxes

Let $P(n,k)$ where $n \ge k$ be the number of ways. There are two cases:
- At least 1 box has exactly 1 ball. Imagine that we first exclude that box and that ball from analysis so that there are $n-1$ balls and $k-1$ boxes left. As a result, the number of ways is $P(n-1, k-1)$.
- All boxes have at least 2 balls. Imagine that we first place 1 ball to each box and then place the $n-k$ balls. As a result, the number of ways is $P(n-k, k)$.

Therefore, the recurrence equation is

$$P(n,k) = P(n-1,k-1) + P(n-k, k)$$

It is easy to find $P(i,i)=1, P(2,1)=1$.


### Empty Boxes

If empty boxes are allowed, it is easy to find the number of ways is

$$\sum_{i=1}^k P(n,i)$$


### Exercise

1. *How many ways are there to distribute 6 identical balls into 3 non-empty indistinguishable bins?*
    ```{dropdown} solution
    There are $P(6,3)=3$ possible ways: $(3,2,1), (2,2,2), (4,1,1)$.
    ```

1. *How many ways are there to distribute 20 identical balls to 3 identical boxes if each box have at least 4 balls?*
    ```{dropdown} solution
    We can first put 3 balls in each box and distribute the remaining 11 balls. The number of way is $P(11,3)$.
    ```

## Identical Balls and Distinct Boxes

We solve by the "stars and bars" method.

### Non-empty Boxes

Image that we put all the $n$ identical balls in a row. There are $n-1$ gaps. Now we are going to insert bars to the gaps. Each gap can accept **up to** 1 bar. If we want to place the balls into $k$ distinguishable boxes, we need $k-1$ bars.

Then we follow the arrangement:

- The balls on the left of the first bar goes to group $1$
- The balls between the 1st bar and the 2nd bar goest to group $2$
- ...
- The balls on the right of the $(k-1)$-th bar goes to group $p$


For instance, the $6$ identical balls below are placed into $3$ boxes $(2, 3, 1)$ by $3-1=2$ bars.

$$\bullet \bullet \vert \bullet \bullet \bullet \vert \bullet $$

It is easy to see that the number of ways is

$$C_{n-1}^{k-1}$$

This is also the number of positive integer solutions $x_j\ge 1$ to the equation

$$x_{1}+x_{2}+\ldots+x_{k}=n$$

Note the recurrence relation of the "choose" number $C_n^k=C(n,k)$:

$$C(n,k) = C(n-1, k-1) + C(n-1, k)$$

To understand that, consider the two cases that can happen to a particular ball $j$:
- The ball $j$ is chosen, so there are $k-1$ balls to be chosen from the remaining $n-1$ balls. The number of ways is $C(n-1,k-1)$.
- The ball $j$ is not chosen, so there are $k$ balls to be chosen from the remaining $n-1$ balls. The number of ways is $C(n-1,k)$.


### Empty Boxes

We can first suppose there are $n+k$ identical balls, and place them into $k$ **non-empty** boxes. Then we take one item out from each group, such that the total number of balls is still $n$ and some boxes can be empty. There is a one-to-one correspondence between the two cases (non-empty vs. empty) and hence the method is valid.

For instance, if the boxes are ordered, by applying the formula we obtain

$$C_{(n+k)-1}^{k-1}$$

Another way to understand this formula is that, we are arranging $n$ balls and $k-1$ bars in a row without any constraints (note that in the non-empty case, two bars cannot be adjacent). Thus, there are $n+k-1$ total positions in a row, where we need $n$ positions to put balls and the remaining $k-1$ positions to put bars. The number of ways is

$$C_{n+k-1}^{n}=C_{n+k-1}^{k-1}$$

This corresponds to the number of nonnegative integer solutions $x_j\ge 0$ to the equation

$$x_{1}+x_{2}+\ldots+x_{k}=n$$

### Exercise


1. *How many ways are there to distribute 6 identical books into 3 persons such that each person gets at least 1 book?*
    ```{dropdown} solution
    $C_{6-1}^{3-1}=C_5^2=10$.
    ```

1. *How many solutions are there to $x_{1}+x_{2}+x_{3}+x_{4}+x_{5}=10$* if all are positive integers and $x_1\le 4$?
    ```{dropdown} solution
    Imagine that there are 10 distinct balls and 5 identical boxes. Each boxes contains at least one ball and the first box contains no more than 4 balls. We can just solve by complementary counting: what if $x_1 \ge 5$? This is equivalent to $x_{1}+x_{2}+x_{3}+x_{4}+x_{5}=10 - (5-1) = 6$ with $x_i\ge 1$. The number of ways in this complementary case is $C_{6-1}^{5-1}$, so the number of ways required is $C_{10-1}^{5-1} - C_{6-1}^{5-1}$.
    ```

1. *How many 3-digit numbers have a sum of digits equal to 9*
    ```{dropdown} solution
    We can formulate this problem as $x_1 + x_2 + x_3 = 9$ and $x_1 \ge 1$. Solving by complementary counting gives $C_{9+3-1}^{3-1} - C_{9+2-1}^{2-1} = 10$.

    Another way is to convert the problem to $x_1 + x_2 + x_3 = 8$ without any constraints, which gives $C_{8+3-1}^{3-1} = C_{10}^2 = 10$.
    ```

1. *How many numbers less than 1000 have the sum of their digits equal to 10?*
    ```{dropdown} solution
    We can formulate this problem as $x_1 + x_2 + x_3 = 10$ with constraints $x_i \le 9$. Solving by complementary counting gives $C_{10+3-1}^{3-1} - 3$.
    ```

1. *How many 8-digit decreasing numbers are there? Suppose the $i$-th digit is $d_i$, a decreasing number means $d_i \ge d_{i+1}$. For instance, 99,765,111.*
    ```{dropdown} solution
    We can formulate this problem as placing $8$ identical balls to $10$ distinct bins. If there are $x_j$ balls in the $j$-th bin, then there are $x_j$ number of numeral $j$ in the decreasing number $d_1d_2\ldots d_8$.

    There is a one-one-correspondence between the decreasing numbers and the patterns of the placements. For example, $\vert \bullet \bullet \bullet \vert \vert \vert \vert \bullet \vert \bullet \vert \bullet \vert \vert  \bullet \bullet$ corresponds to 99,765,111 if the bins are increasingly numbered from left to right.

    Note that the case of all zeros is not valid. Therefore, the number of ways is $C_{8+10-1}^{10-1}-1$.
    ```

1. *In the question above, what if no digits have the same value, i.e. $d_i > d_{i+1}$*?
    ```{dropdown} solution
    This implies that each bin has at most 1 ball. So we are choose 8 bins from the 10 bins to place balls. The number of ways is $C_{10}^8$.
    ```

1. *In the question above, what if there is exactly one pair of digits has the same value $d_i = d_{i+1}$ while all other digits are different $d_j>d_k$ for $j<k, j\ne i$?*


    ```{dropdown} solution
    This means the $d_i$-th bin has two balls and other $9$ bins can have at most 1 ball. Thus, we can exclude the $d_i$-th ball into consideration so that we are placing $8-2=6$ balls to the remaining $9$ bins. Since there are $10$ ways to choose an $d_i$, the number of ways is $10C_9^6$.
    ```

    ```{warning}
    Using the balls and boxes approach for counting is convenient, but one should be cautious when it comes to **probability**. Consider a simpler example of 2 digits, each take value from 0, 1, or 2. This corresponds to 2 balls and 3 bins. The correspondence relations are listed below.


    |        No.       | two balls goes to | # balls in each bin |             pattern            | decreasing number |
    |:----------------:|:-----------------:|:-------------------:|:------------------------------:|:-----------------:|
    |         1        |        0,0        |        2,0,0        |  $\bullet \bullet \vert \vert$ |   00 (discarded)  |
    |         2        |        0,1        |        1,1,0        | $\bullet \vert \bullet  \vert$ |         10        |
    |         3        |        0,2        |        1,0,1        | $\bullet \vert \vert \bullet$ |         20        |
    |         4        |        1,0        |        1,1,0        |  $\bullet \vert \bullet \vert$ |         10        |
    |         5        |        1,1        |        0,2,0        |  $\vert \bullet \bullet \vert$ |         11        |
    |         6        |        1,2        |        0,1,1        |  $\vert \bullet \vert \bullet$ |         21        |
    |         7        |        2,0        |        1,0,1        |  $\bullet \vert \vert \bullet$ |         20        |
    |         8        |        2,1        |        0,1,1        | $\vert \bullet  \vert \bullet$ |         21        |
    |         9        |        2,2        |        0,0,2        |  $\vert \vert \bullet \bullet$ |         22        |
    | # distinct <br>  items |       9       |        6          |              6               |         5         |

    The result from the table is consistent with our previous approach $C_{2+3-1}^{3-1}-1=5$. Now what if you are asked what is $\mathrm{P}\left( d_i > d_{d+1} \right)$ in a 2-digit number?
    - From the table the answer is $\frac{9-3}{9-1} = \frac{6}{8}$, where $-1$ discards $00$ and $-3$ discards $00,11,22$.
    - But from our approach above, the number of ways to see $d_i>d_{i+1}$ is $C_3^2=3$ and the total possible 3-digit number is $3^2-1=8$, so the probability should be $\frac{3}{8}$

    What's wrong? The first method makes a mistake when understanding the random process of how the decreasing numbers are generated. The balls and boxes approach inherently mask all numbers into decreasing numbers, so we see two $20$, two $10$, and two $21$. These numbers are count twice so the final answer are doubled.
    ```
