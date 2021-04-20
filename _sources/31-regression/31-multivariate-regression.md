# Multivariate Regression

Review [MANOVA](manova).

In a multivariate regression model, the response for each observation is a $p$-dimensional random vector $\boldsymbol{y}$. The explanatory variables are of the same structure as univariate case for each component of the response vector, but the coefficients are different.

$$
\boldsymbol{y}=\boldsymbol{\beta}_{0}+\boldsymbol{x}_{1} \boldsymbol{\beta}_{1}+\cdots+\boldsymbol{x}_{r} \boldsymbol{\beta}_{r}+\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim N_{p}\left(0_{p}, \boldsymbol{\Sigma}_{p \times p}\right)
$$

or

$$
\boldsymbol{y}=\left[\begin{array}{c}
Y_{1} \\
Y_{2} \\
\vdots \\
Y_{p}
\end{array}\right]=\left[\begin{array}{c}
\beta_{01} \\
\beta_{02} \\
\vdots \\
\beta_{0 p}
\end{array}\right]+x_{1}\left[\begin{array}{c}
\beta_{11} \\
\beta_{12} \\
\vdots \\
\beta_{1 p}
\end{array}\right]+\cdots+x_{r}\left[\begin{array}{c}
\beta_{r 1} \\
\beta_{r 2} \\
\vdots \\
\beta_{r p}
\end{array}\right]+\left[\begin{array}{c}
\varepsilon_{1} \\
\epsilon_{2} \\
\vdots \\
\epsilon_{p}
\end{array}\right]
$$

In matrix form of $n$ observations,


$$
\boldsymbol{Y}_{n \times p}=\boldsymbol{X}_{n \times(1+r)} \boldsymbol{\beta}_{(1+r) \times p}+\boldsymbol{\epsilon}_{n \times p}
$$

- Each coefficient for covariate $x_r$ is a $p$-dimensional vector $\boldsymbol{\beta} _r$.
- Since the dependence among the $Y_j$’s is of main interests, the error $\varepsilon_j$, usually are not independent. Therefore, their covariance matrix $\boldsymbol{\Sigma} _{p \times p}$ is NOT a diagonal matrix.
