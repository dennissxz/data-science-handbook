# Expected Value of Maximum of Two Uniform Random Variables

*Suppose $X$ and $Y$ are two uniformly distributed random variables over the interval $[0,1]$. What is the expected value $\mathrm{E}[\max(X,Y)]$?*


## Solution

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


    In paricular, for $n$ independent uniform random variables, $\mathrm{E}(Z) = \frac{n}{n+1}$.

- If $X$ and $Y$ has the relation $X=Y$, then

    $$\mathrm{E}(Z)=\mathrm{E}(X)=\frac{1}{2}$$

- If $X$ and $Y$ has the relation $X+Y=1$, then by the law of total expectation


    $$\begin{align}
    \mathrm{E}(Z) &=\mathrm{E}[\mathrm{E}(Z\vert X)]\\
    &=\mathrm{P(X\le 1/2)} \cdot \mathrm{E}(1-X\vert X\le 1/2) + \mathrm{P(X> 1/2)} \cdot \mathrm{E}(X\vert X > 1/2)\\

     &= \frac{1}{2} \times \frac{3}{4}  + \frac{1}{2} \times \frac{3}{4}  \\

    &=\frac{3}{4}

    \end{align}$$

It seems that the range is $[\frac{1}{2}, \frac{3}{4}]$.
