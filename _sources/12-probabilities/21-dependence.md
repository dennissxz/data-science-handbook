# Independence




## Raining

Tags: Jane Street, Quant, 20Q4

*Suppose the probabilities of raining on Saturday and Sunday are $p$ and $q$ respectively. What is the probability of raining on weekend? What is the probability that it rains on either Saturday or Sunday?*


### Solution

Note that the question does not specify the dependence of raining on Saturday and Sunday. To be rigorous, we introduce two indicator variables


$$\begin{align}
X &= \mathbb{I}[\text{raining on Saturday}] \\
Y &= \mathbb{I}[\text{raining on Sunday}] \\
\end{align}$$


Hence

$$\begin{align}
\mathrm{P}(X=1) &= p\\
\mathrm{P}(Y=1) &= q\\
\end{align}$$


Suppose $\mathrm{P}(X=1, Y=1)=a$, then the contingency table is

|$X \ \backslash \ Y$| $0$ | $1$| total |
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


## Expected Value of Maximum of Two Uniform Random Variables

*Suppose $X$ and $Y$ are two uniformly distributed random variables over the interval $[0,1]$. What is the expected value $\mathrm{E}[\max(X,Y)]$?*


### Solution

Let $Z=\max(X,Y)$. Since there is no dependence specified, we start from the special cases.

- If $X$ and $Y$ are independent, then

    $$\begin{align}
    \mathrm{P}(Z\le z) &= \mathrm{P}(\max (X, Y) \le z) \\
    &=\mathrm{P}(X \leqslant z) \mathrm{P}(Y \leqslant z) \\
    &= z^2 \\
    \mathrm{E}(Z) &= \int_{0}^{1}\mathrm{P}(Z\ge z)\mathrm{d}z \\
     &= \int_{0}^{1}\left(1-z^{2}\right) \mathrm{d}z \\
    &=\frac{2}{3}
    \end{align}$$

    Another way without finding the cumulative distribution function $\mathrm{P}\left( Z\le z \right)$:

    $$
    \begin{aligned}
    \mathrm{E}(\max (x, y)) &=\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} \max (x, y) p(x, y) \,\mathrm{d}x \,\mathrm{d}y \\
    &=\int_{0}^{1} \int_{0}^{1} \max (x, y) \,\mathrm{d}x \,\mathrm{d}y \\
    &=\int_{0}^{1} \int_{0}^{x} x \,\mathrm{d}y \,\mathrm{d}x+\int_{0}^{1} \int_{0}^{y} y \,\mathrm{d}x \,\mathrm{d}y \\
    &=\int_{0}^{1} x^{2} \,\mathrm{d}x+\int_{0}^{1} y^{2} \,\mathrm{d}y \\
    &=\left[\frac{x^{3}}{3}\right]_{0}^{1}+\left[\frac{y^{3}}{3}\right]_{0}^{1} \\
    &=\frac{1}{3}+\frac{1}{3} \\
    &=\frac{2}{3}
    \end{aligned}
    $$

    In particular, for $n$ independent uniform random variables,

    $$\begin{align}
    \mathrm{E}(Z) &= \int_{0}^{1}\mathrm{P}(Z\ge z)\mathrm{d}z\\
    &= \int_{0}^{1}\left(1-z^{n}\right) \mathrm{d}z\\
    &= \frac{n}{n+1}\\
    \end{align}$$


- If $X$ and $Y$ has the relation $X=Y$, then

    $$\mathrm{E}(Z)=\mathrm{E}(X)=\frac{1}{2}$$

  In this case

  $$\begin{align}
  \mathrm{Corr}\left( X,Y \right) &= \frac{\mathrm{Cov}\left( X,Y \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( Y \right)}} \\
  &= \frac{\mathrm{Cov}\left( X,X \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( X \right)}} \\
  &= \frac{\mathrm{Var}\left( X \right)}{\mathrm{Var}\left( X \right)}\\
  &= 1
  \end{align}$$


- If $X$ and $Y$ has the relation $X+Y=1$, then by the law of total expectation

    $$\begin{align}
    \mathrm{E}(Z) &=\mathrm{E}[\mathrm{E}(Z \,\vert\, X)]\\
    &=\mathrm{\mathrm{P}\left( X\le \frac{1}{2} \right)} \cdot \mathrm{E}\left( 1-X\,\vert\, X\le \frac{1}{2}  \right) + \mathrm{P}\left( X> \frac{1}{2} \right) \cdot \mathrm{E}\left( X\,\vert\, X > \frac{1}{2}\right)\\
     &= \frac{1}{2} \times \frac{3}{4}  + \frac{1}{2} \times \frac{3}{4}  \\
    &=\frac{3}{4}
    \end{align}$$

  In this case

  $$\begin{align}
  \mathrm{Corr}\left( X,Y \right) &= \frac{\mathrm{Cov}\left( X,Y \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( Y \right)}} \\
  &= \frac{\mathrm{Cov}\left( X,1-X \right)}{\sqrt{\mathrm{Var}\left( X \right)\mathrm{Var}\left( 1-X \right)}} \\
  &= \frac{\mathrm{-Var}\left( X \right)}{\mathrm{Var}\left( X \right)}\\
  &= - 1
  \end{align}$$


It seems that the range is $[\frac{1}{2}, \frac{3}{4}]$.
