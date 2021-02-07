# Correlation and Dependence

## Definitions

### Correlation


The correlation of two random variables measures how two or more variables are related or associated to one another. There are several correlation coefficients, and the most familiar one is Pearson correlation coefficient.


#### Pearson Correlation Coefficient

It is defined for two continuous variables $X,Y$ and only measure the **linear relationship** between them.

$$
\rho
= \frac{\operatorname{Cov}\left( X, Y \right)}{\sqrt{\operatorname{Var}\left( X \right)\operatorname{Var}\left( Y \right)}}
= \frac{\sigma_{X,Y}}{\sigma_X \sigma_Y}
= \frac{\mathrm{E}\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]}{\sigma_{X} \sigma_{Y}}
$$

If $X$ and $Y$ are more likely to have values larger or smallerthan their means $\mu_X, \mu_Y$ concurrently, then the product $\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)$ is more likely to be positive, which leads to a positive value of the correlation coefficient. On the other hand, the correlation coefficient is more likely to be negative.

By the Cauchy-Schwarz inequality we have

$$
\left\vert \operatorname{Cov}\left( X,Y \right) \right\vert^2 \le \operatorname{Var}\left( X \right) \operatorname{Var}\left( Y \right)
$$

and hence

$$
-1 \le \rho \le 1
$$

The equality holds iff there is a deterministic linear relation between $X$ and $Y$, $Y = aX + b$.

Given a sample of $n$ observed pairs $(x_i, y_i)$, the sample correlation coefficient is defined as

$$
r_{x y}
= \frac{s_{x,y}}{s_x, s_y}
= \frac{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2} \sum_{i=1}^{n}\left(y_{i}-\bar{y}\right)^{2}}},
$$

or equivalently,

$$
\begin{aligned}
r_{x y} &=\frac{\sum x_{i} y_{i}-n \bar{x} \bar{y}}{n s_{x}^{\prime} s_{y}^{\prime}} \\
&=\frac{n \sum x_{i} y_{i}-\sum x_{i} \sum y_{i}}{\sqrt{n \sum x_{i}^{2}-\left(\sum x_{i}\right)^{2}} \sqrt{n \sum y_{i}^{2}-\left(\sum y_{i}\right)^{2}}}
\end{aligned}
$$


#### Spearman Correlation Coefficient

Spearman's rank correlation is more robust than Pearson's to capture nonlinear relationships. In fact, it assesses monotonic relationships. For a sample of $n$ scores $X_i, Y_i$, they are first converted to ranks $\operatorname{rg}_{X_i}, \operatorname{rg}_{Y_i}$, and the Spearman correlation coefficient is defined as the Pearson correlation coefficient between the rank variables.µ

$$
r_{s}=\rho_{\mathrm{rg}_{X}, \mathrm{rg}_{Y}}=\frac{\operatorname{Cov}\left(\mathrm{rg}_{X}, \mathrm{rg}_{Y}\right)}{\sigma_{\mathrm{rg}_{X}} \sigma_{\mathrm{rg}_{Y}}}
$$

If there is **no ties,** then it can be computed by the formula

$$
r_{s}=1-\frac{6 \sum_i d_{i}^{2}}{n\left(n^{2}-1\right)}
$$

where $d_i = \operatorname{rg}_{X_i} - \operatorname{rg}_{Y_i}$ is the rank difference of each observation.

One can see that
- If two variables are monotonically related (even if their relationship is not linear), then $d_i = 0$ for all $i$, and therefore $r_s = 1$. For instance, $X\sim U(-1,1), Y=X^3$.
- If two variables are inversely monotonically related, then $d_i = n-1, n-3, \ldots, 3-n, 1-n$ and $\sum_i d_i ^2 = \frac{1}{3} n (n^2-1)$, and therefore $r_s = 1-2 = -1$


```{seealso}
Mutual information can also be applied to measure association between two variables, given their distribution functions.
```

### Correlated

Two random variables $X,Y$ are said to be
- correlated if $\operatorname{Cov}\left( X,Y \right) \ne 0$
- uncorrelated if $\operatorname{Cov}\left( X,Y \right) = 0$.

### Dependence

Two random variables $X,Y$ are independent iff the joints cumulative distribution function satisfies

$$
F_{X, Y}(x, y)=F_{X}(x) F_{Y}(y) \quad \text{for all}\ x, y
$$

or equivalently, the joint density satisfies

$$
f_{X, Y}(x, y)=f_{X}(x) f_{Y}(y) \quad \text{for all}\ x, y
$$

From this definition we have

$$
f_{X\mid Y}(x\mid y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}  = f_X(x)
$$

which can be interpreted as "knowing any information about $Y=y$ does not change our knowledge of $X$". If this is false then the two random variables are not independent.

## Comparison

### Independent $\Rightarrow$ Uncorrelated

If two random variables are independent, then $\operatorname{E}\left( XY \right) = \operatorname{E}\left( X \right) \operatorname{E}\left( Y \right)$, $\operatorname{Cov}\left( X,Y \right) = 0$, i.e., they are uncorrelated.

### Uncorrelated $\not \Rightarrow$ Independent

If $\operatorname{Cov}\left( X,Y \right) = 0$ or $\rho(X,Y) = 0$, then we CAN NOT say they are independent.

For instance, let $X\sim U(-1,1)$ and $Y = \left\vert X \right\vert$. Then $Y$ is completely dependent on $X$, but


$$\begin{align}
\operatorname{Cov}\left( X, Y \right)  
&= \operatorname{E}\left( XY \right) - \operatorname{E}\left( X \right) \operatorname{E}\left( Y \right)   \\
&= \operatorname{E}\left( X \left\vert X \right\vert \right) - \operatorname{E}\left( X \right) \operatorname{E}\left( \left\vert X \right\vert \right)\\
&= \operatorname{E}\left( X^2 \right)\cdot \frac{1}{2} + \operatorname{E}\left( -X^2 \right)\cdot \frac{1}{2} - 0 \cdot \frac{1}{2} \\
& = 0
\end{align}$$

so that $\rho(X,Y) = 0$

```{seealso}
- One special case is bivariate normal distribution. If two variables $X,Y$ follows a bivariate normal distribution, then $\operatorname{Cov}\left( X,Y \right) = 0$ implies their independence. However, this does not hold for two arbitrary normal variables. See __.

- For two random variables, if their mutual information is 0, then they are independent. See __.
```



## Simpson's Paradox

We have talked about the dependence between two random variables, which says that the information of a random variable $X$ depends on the information of another variable $Y$. In fact, here "the information of $X$" can be generalized to any probablistic objects, such as
- the distribution of a random vector $\boldsymbol{X}=\left[ X_1, \ldots, X_2 \right]$
- the correlation of two random variables $\mathrm{Corr}\left( X_1, X_2 \right)$
- some statistical conclusions.

Simpson's Paradox refers to this case. It is a phenomenon in probability and statistics, in which a trend appears in several different groups of data but disappears or reverses when these groups are combined.

The cause of Simpson's paradox is the existence of a confounding variable, which has dependence with the observed variables.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/f/fb/Simpsons_paradox_-_animation.gif" width="50%" align="center"/>
</p>



Mathematically, if the trends are correlations, it can be formulated as

$$\mathrm{Corr}(X,Y\,\vert\, Z) > 0 \not \Rightarrow \mathrm{Corr}(X,Y) > 0$$

where $Z$ is a confounding variable.

If the trends are proportions, we can illustrate them with vectors. In the below case, for a vector $v$, suppose the angle between it and the $x$-axis is $\theta$. The horizontal projection $\vert \overrightarrow{v}\cos\theta\vert$ is the number of applicants and the vertical projection $\vert \overrightarrow{v}\sin\theta\vert$ is the number of accepted candidates. A steeper vector then represents a larger acceptance rate.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/c/cd/Simpsons_paradox.jpg" width="50%" />
</p>


The Simpson's paradox says that, even if $\overrightarrow{L_{1}}$ has a smaller slope than $\overrightarrow{B_{1}}$ and $\overrightarrow{L_{2}}$ has a smaller slope than $\overrightarrow{B_{2}}$, the overall slop $\overrightarrow{L_{1}} + \overrightarrow{L_{2}}$ can be greater than $\overrightarrow{B_{1}}+\overrightarrow{B_{2}}$. For this to occur, one of the orange vectos must have a greater slope than one of the blue vectors (e.g. $\overrightarrow{L_{2}}$ vs. $\overrightarrow{B_{1}}$), and these will generally be **longer** than the alternatively subscripted vectors (i.e. imbalanced data) – thereby dominating the overall comparison.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Simpson_paradox_vectors.svg/1200px-Simpson_paradox_vectors.svg.png" width="50%">
</p>


## Exercise

1. (Raining, Tags: Jane Street, Quant, 20Q4)

    *Suppose the probabilities of raining on Saturday and Sunday are $p$ and $q$ respectively. What is the probability of raining on weekend? What is the probability that it rains on either Saturday or Sunday?*

    :::{admonition,dropdown,seealso} *Proof*

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

    :::  


1. (Expected Value of the Maximum of Two Uniform Random Variables)

    *Suppose $X$ and $Y$ are two uniformly distributed random variables over the interval $[0,1]$. What is the expected value $\mathrm{E}[\max(X,Y)]$?*

    :::{admonition,dropdown,seealso} *Proof*

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

    :::



1. (Lower Bound of Correlation for IID)

   *Suppose $X_1, X_2, \ldots, X_n$ where $n\ge 2$ are IID variables with common pairwise correlation $\rho = \operatorname{Corr}\left( X_i, X_j \right)$ for $i\ne j$. What is the lower bound of $r$ and when is it obtained?*

      :::{admonition,dropdown,seealso} *Proof*

      Since

      $$\begin{align}
      \operatorname{Var}\left( \sum_i X_i \right)
      &= \sum_i \operatorname{Var}\left( X_i \right) + \sum_{i=1}^n \sum_{j\ne i}^n \operatorname{Cov}\left( X_i, X_j \right) \\
      &= n \sigma^2 + n(n-1)\rho\sigma^2 \\
      \ge 0 \\
      \end{align}$$

      we have


      $$
      \rho \ge - \frac{1}{n-1}
      $$

      if $\sigma^2 > 0$, otherwise $\rho$ is undefined.

      The lower bound is obtained iff $\operatorname{Var}\left( \sum_i X_i \right) = 0$, i.e., $\sum_i X_i = \text{constant}$ almost surely.

      :::


1. *For three variables $X,Y,Z$, is it possible that $\operatorname{Cov}\left( X,Y \right) \ne 0, \operatorname{Cov}\left( Y, Z \right) \ne 0$ but $\operatorname{Cov}\left( X, Z \right) = 0$?*

    :::{admonition,dropdown,seealso} *Solution*

    Consider the correlation matrix

    $$
    \boldsymbol{C} = \left[\begin{array}{ccc}
    1 & a & 0\\
    a & 1 & b\\
    0 & b & 1
    \end{array}\right]
    $$

    Since it is positive semi definite, we must have

    $$
    \operatorname{det}(\boldsymbol{C}) = (1) - (a^2 + b^2) \ge 0
    $$

    which has infinite many solutions $(a,b)$.

    :::
