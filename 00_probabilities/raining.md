# Raining

Tags: Jane Street, Quant, 20Q4


Suppose the probabilities of raining on Saturday and Sunday are $p$ and $q$ respectively. What is the probability of raining on weekend? What is the probability that it rains on either Saturday or Sunday?


## Solution

Note that the question does not specify the dependence of raining on Saturday and Sunday. To be rigorous, we introduce two indicator variables

$$X=\mathbb{I}[\text{raining on Saturday}]$$
$$Y=\mathbb{I}[\text{raining on Sunday}]$$

Hence $\mathrm{P}(X=1)=p$, $\mathrm{P}(Y=1)=q$.

Suppose $\mathrm{P}(X=1, Y=1)=a$, then the contingency table is

|$X$ \ $Y$| $0$ | $1$| total |
|:-: | :-:| :-:| :-: |
|$0$ | $1-p-q+a$|   $q-a$    | $1-p$ |
|$1$   |  $p-a$ | $a$  | $p$|
|total  | $1-q$  | $q$  | $1$  |

Note that there are constraints on $a$:

$$\begin{align}
1-p-q+a &\ge 0 \\
p-a & \ge 0 \\
q-a & \ge 0
\end{align}$$

So we can obtain the valid range for $a$:

$$\max(p+q-1,0) \le a \le \min(p,q)$$

The required answer is  

$$\begin{align}
p_1&= \mathrm{P}(\text{raining on weekend})  \\
& = 1 - \mathrm{P}(X=0, Y=0)  \\
 & = p+q-a \\
 & \in [\max(p,q), \min(p+q,1)] \\

p_2 &= \mathrm{P}(\text{raining on either Saturday or Sunday})  \\
& = \mathrm{P}(X=1, Y=0) + \mathrm{P}(X=0, Y=1)  \\
 & = p+q-2a \\
 & \in [\vert p-q\vert, \min(p+q, 2-p-q)]
\end{align}$$
