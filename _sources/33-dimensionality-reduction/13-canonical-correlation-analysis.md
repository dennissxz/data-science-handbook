
# Canonical Corerlation Analysis

## Objective

Hotelling (1936) extended the idea of multiple correlation to the problem of measuring linear association between two **groups** of variables, say $\boldsymbol{x} _1$ and $\boldsymbol{x} _2$. Thus we consider the simple correlation coefficient between every pair of **linear combinations** of elements of $x_1$ and of $\boldsymbol{x} _2$ and choose the maximum.

$$
\max _{\boldsymbol{\alpha}, \boldsymbol{\beta} } \, \operatorname{Corr}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{x}_{1}, \boldsymbol{\beta}^{\prime} \boldsymbol{x}_{2}\right)
$$

This maximum, $\rho_1$, is called the first canonical correlation and the corresponding pair of linear combinations, say $(U_1, V_1)$, is called the first pair of canonical variables.


Since each group has more than one variable, one single correlation coefficient may miss significant linear association in other dimensions. So we consider the subclass of **all** pairs of linear combinations of elements of $\boldsymbol{x}  _1$ and $\boldsymbol{x}  _2$ whose members are **uncorrelated** with $(U_1, V_1）$. The maximum simple correlation coefficient, $\rho _2$, of such pairs is called the second canonical correlation and the pair, say $(U_2, V_2)$, achieving this maximum is called the second pair of canonical variables.

 Continuing this way we shall get the $p$ canonical correlations and $p$ pairs of canonical variables, where $p$ is the smallest dimension of $\boldsymbol{x} _1$ and $\boldsymbol{x} _2$.

## Learning

Partition the covariance matrix of full rank in accordance with two groups of variables,


$$
\begin{equation}
\operatorname{Var}\left(\begin{array}{l}
\boldsymbol{x}_{1} \\
\boldsymbol{x}_{2}
\end{array}\right)=\left(\begin{array}{ll}
\boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22}
\end{array}\right), \quad \boldsymbol{x}_{1} \text { is } p \times 1, \boldsymbol{x}_{2} \text { is } q \times 1, p \leq q
\end{equation}
$$

### Sequential Optimization

The first canonical correlation, $\rho _1$, equals the maximum correlation between all pairs of linear combinations of $\boldsymbol{x} _1$ and $\boldsymbol{x} _2$ with unit variance. That is,


$$\begin{align}
\max _{\boldsymbol{\alpha}, \beta} \operatorname{Corr}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{x}_{1}, \boldsymbol{\beta}^{\prime} \boldsymbol{x}_{2}\right)
= \max &  _{\boldsymbol{\alpha}, \boldsymbol{\beta}}  \frac{\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}}{\sqrt{\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha} \boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}}} \\
\text{s.t.}  &  \ \quad \boldsymbol{\alpha} ^\top \boldsymbol{\Sigma} _{11} \boldsymbol{\alpha} =1 \\
  &  \ \quad \boldsymbol{\beta} ^\top \boldsymbol{\Sigma} _{22} \boldsymbol{\beta} =1 \\
\end{align}$$

If the maximum is achieved at $\boldsymbol{\alpha} _1$ and $\boldsymbol{\beta} _1$, then the first pair of canonical varibales are defined as


$$
U_{1}=\boldsymbol{\alpha}_{1}^{\prime} \boldsymbol{x}_{1}, \quad V_{1}=\boldsymbol{\beta}_{1}^{\prime} \boldsymbol{x}_{2}
$$

Successively, for $i = 2, \ldots, p$, the $i$-th canonical correlation $\rho_i$ is defined as


$$\begin{align}
\rho_{i}
 =\max _{\boldsymbol{\alpha}, \boldsymbol{\beta}} & \quad \operatorname{corr}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{x}_{1}, \boldsymbol{\beta}^{\prime} \boldsymbol{x}_{2}\right) \\
 \text{s.t.} &  \quad (\boldsymbol{\alpha} ^\top \boldsymbol{x} _1, \boldsymbol{\beta} ^\top \boldsymbol{x} _2) \text{ uncorrelated with } (U_1, V_1), \ldots, (U_{i-1}, V_{i-1}) \\
\end{align}$$

If the maximum is achived at $\boldsymbol{\alpha} _i$ and $\boldsymbol{\beta} _i$, then the first pair of canonical varibales are defined as


$$
U_{i}=\boldsymbol{\alpha}_{i}^{\prime} \boldsymbol{x}_{i}, \quad V_{i}=\boldsymbol{\beta}_{i}^{\prime} \boldsymbol{x}_{2}
$$

### Spectral Decomposition


Rather than obtaining pairs of canonical variables and canonical correlation sequentially, it can be shown that the canonical correlations $\rho$'s and hence pairs of canonical variables $(U,V)$’s can be obtained simultaneously by solving for
the eigenvalues $\rho^2$'s and eigenvectors $\boldsymbol{\alpha}$’s of the matrix


$$
\boldsymbol{\Sigma}_{11}^{-1} \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21}
$$

A difficulty of this problem is that the matrix $\boldsymbol{\Sigma}_{11}^{-1} \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21}$ is not symmetric. Consequently, the symmetric matrix

$$
\boldsymbol{\Sigma}_{11}^{-1/2} \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1/2}
$$

is considered instead for the computational efficiency. Note that the two matrices possess the **same** eigenvalues and their eigenvectors are linearly related.

It turns out that the canonical correlation $\rho_i$ and the canonical variables $(\boldsymbol{\alpha} _i, \boldsymbol{\beta} _i), i = 1, 2, \ldots, p$ are related to matrix eigen-analysis:


$$
\begin{equation}
\begin{aligned}
\rho_{1}^{2} \geq \rho_{2}^{2} \geq \cdots \geq \rho_{p}^{2} & \quad \text {eigenvalues of } \boldsymbol{\Sigma}_{11}^{-1 / 2} \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1 / 2}\\
\boldsymbol{\alpha}_{1}^{\star}\quad \boldsymbol{\alpha}_{2}^{\star} \quad\cdots \quad \boldsymbol{\alpha}_{s}^{\star} & \quad \text { associated unit-norm eigenvectors,}\left(\boldsymbol{\alpha}_{i}^{\star}\right)^{\prime} \boldsymbol{\alpha}_{i}^{\star}=1
\\
\boldsymbol{\alpha}_{i}=\boldsymbol{\alpha}_{i}^{\star} / \sqrt{\left(\boldsymbol{\alpha}_{i}^{\star}\right)^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}_{i}^{\star}} & \quad \text { coefficients of } U_{i}, \boldsymbol{\alpha}_{i}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}_{i}=1\\
\boldsymbol{\alpha}_{i}=\boldsymbol{\Sigma}_{11}^{-1 / 2} \boldsymbol{\alpha}_{i}^{\star} & \quad \text { 2nd formula for coefficients of } U_{i}\\
\boldsymbol{\beta}_{i}=\boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21} \boldsymbol{\alpha}_{i} & \quad \text { coefficients of } V_{i}, \boldsymbol{\beta}_{i}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}_{i}=1
\end{aligned}
\end{equation}
$$

where $s = \min(p, q)$


```{dropdown} Derivation

Recall [](Covariance-Matrix-of-Two-Vectors).

We consider the following maximization problem:


$$
\begin{equation}
\rho^{2} \equiv \max _{\boldsymbol{\alpha}, \boldsymbol{\beta}} \operatorname{Corr}^{2}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{x}_{1}, \boldsymbol{\beta}^{\prime} \boldsymbol{x}_{2}\right)=\max _{\boldsymbol{\alpha}, \boldsymbol{\beta}} \frac{\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right)^{2}}{\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}\right)\left(\boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}\right)} \quad \text { s.t. } \boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}=\boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}=1
\end{equation}
$$

The Lagrangean is


$$
\begin{equation}
L(\boldsymbol{\alpha}, \boldsymbol{\beta}, \lambda, \theta)=\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right)^{2}-\lambda\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}-1\right)-\theta\left(\boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}-1\right)
\end{equation}
$$

The first order conditions are


$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial \boldsymbol{\alpha}}=& 2\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right) \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}-2 \lambda \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}=\mathbf{0} \\
\Rightarrow \qquad \qquad &\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right) \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}=\lambda \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha} \\
\frac{\partial L}{\partial \boldsymbol{\beta}}=& 2\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right) \boldsymbol{\Sigma}_{21} \boldsymbol{\alpha}-2 \theta \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}=\mathbf{0} \\
\Rightarrow \qquad \qquad &\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right) \boldsymbol{\Sigma}_{21} \boldsymbol{\alpha}=\theta \boldsymbol{\Sigma}_{22} \boldsymbol{\beta} \\
\quad \frac{\partial L}{\partial \lambda}=& 1-\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}=0 \\
\Rightarrow \qquad \qquad & \boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha}=1 \\
\quad \frac{\partial L}{\partial \theta}=& 1-\boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}=0 \\
\Rightarrow \qquad \qquad & \boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}=1
\end{aligned}
\end{equation}
$$

Premultiply the first condition by $\boldsymbol{\alpha} T$ and the second condition by $\boldsymbol{\beta} ^\top$, we have


$$
\begin{equation}
\begin{array}{l}
\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right)^{2}=\lambda \boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha} = \lambda \\
\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right)^{2}=\theta \boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta} = \theta
\end{array}
\end{equation}
$$

Hence


$$
\begin{equation}
\lambda=\theta=\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{\Sigma}_{12} \boldsymbol{\beta}\right)^{2}
\end{equation}
$$

which implies that the Lagrangian multipliers are equal to the maximized value of squared correlation $\begin{equation}
\operatorname{Corr}^{2}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{x}_{1}, \boldsymbol{\beta}^{\prime} \boldsymbol{x}_{2}\right)
\end{equation}$, i.e., $\rho^2$.

Based on the above results, we can further simplify the first and second conditions by replacing $\boldsymbol{\alpha} ^\top \boldsymbol{\Sigma} _{12} \boldsymbol{\beta}$ by $\sqrt{\lambda}$ and $\sqrt{\theta}$ respectively.


$$
\begin{equation}
\begin{aligned}
\left\{\begin{array}{l}
\boldsymbol{\Sigma}_{12} \boldsymbol{\beta}-\sqrt{\lambda} \boldsymbol{\Sigma}_{11} \boldsymbol{\alpha} & =\mathbf{0} \\
\boldsymbol{\Sigma}_{21} \boldsymbol{\alpha}-\sqrt{\lambda} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta} & =\mathbf{0}
\end{array}\right.\\
\Rightarrow\left(\begin{array}{cc}
-\sqrt{\lambda} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & -\sqrt{\lambda} \boldsymbol{\Sigma}_{22}
\end{array}\right) \left(\begin{array}{l}
\boldsymbol{\alpha} \\
\boldsymbol{\beta}
\end{array}\right)& =\mathbf{0}
\end{aligned}
\end{equation}
$$

In order to obtain the non-trivial solutions for $\boldsymbol{\alpha}$  and $\boldsymbol{\beta}$, we require


$$
\begin{equation}
\left|\begin{array}{cc}
-\sqrt{\lambda} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & -\sqrt{\lambda} \boldsymbol{\Sigma}_{22}
\end{array}\right|=0
\end{equation}
$$

which gives


$$
\begin{equation}
\begin{array}{l}
\left|\begin{array}{cc}
-\sqrt{\lambda} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\
\boldsymbol{\Sigma}_{21} & -\sqrt{\lambda} \boldsymbol{\Sigma}_{22}
\end{array}\right|=\left|\sqrt{\lambda} \boldsymbol{\Sigma}_{22}\right|\left|\sqrt{\lambda} \boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12}\left(\sqrt{\lambda} \boldsymbol{\Sigma}_{22}\right)^{-1} \boldsymbol{\Sigma}_{21}\right|=0 \\
\Rightarrow\left|\sqrt{\lambda} \boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12}\left(\sqrt{\lambda} \boldsymbol{\Sigma}_{22}\right)^{-1} \boldsymbol{\Sigma}_{21}\right|=0 \quad \because\left|\boldsymbol{\Sigma}_{11}\right|>0 \\
\Rightarrow\left|(1 / \sqrt{\lambda})\left(\lambda \boldsymbol{\Sigma}_{11}-\boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21}\right)\right|=0 & \\
\Rightarrow\left|\boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21}-\lambda \boldsymbol{\Sigma}_{11}\right|=0 \quad \because \lambda>0
\end{array}
\end{equation}
$$

which indeed is the eigenvalue problem of


$$
\begin{equation}
\boldsymbol{\Sigma}_{11}^{-1} \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21} \boldsymbol{\alpha}=\lambda \boldsymbol{\alpha}
\end{equation}
$$

Once the solution for $\boldsymbol{\alpha}$  is obtained, the solution for $\boldsymbol{\beta}$  can be obtained by

$$
\begin{equation}
\boldsymbol{\beta} \propto \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21} \boldsymbol{\alpha}
\end{equation}
$$

with the normalized condition


$$
\begin{equation}
\boldsymbol{\beta}^{\prime} \boldsymbol{\Sigma}_{22} \boldsymbol{\beta}=1
\end{equation}
$$

## Properties

**Invariance property**
: Canonical corelations $\rho_i$'s between $\boldsymbol{x} _1$ and $\boldsymbol{x} _2$ are the same as those between $\boldsymbol{A} _1 \boldsymbol{x} _1 + \boldsymbol{c}_1$ and $\boldsymbol{A} _2 \boldsymbol{x} _2 + \boldsymbol{c} _2$, where both $\boldsymbol{A} _1$ and $\boldsymbol{A} _2$ are non-singular square matrices and their computation can be based on either the partitioned covariance matrix or the partitioned correlation matrix. However, the canonical coefficients contained in $\boldsymbol{\alpha} _k$ and $\boldsymbol{\beta} _k$ are **not** invariant under the same transforam, nor their estimates.

```


## Model Selection

Now let $\boldsymbol{S} _{11}, \boldsymbol{S} _{12}, \boldsymbol{S} _{22}$ and $\boldsymbol{S} _{21}$ be the corresponding sub-matrices of the sample covariance matrix $\boldsymbol{S} $. For $i = 1, 2, \ldots, p$, let $r_i ^2, \boldsymbol{a} _i$ and $\boldsymbol{b} _i$ be respectively the sample estimators of $\rho_i^2, \boldsymbol{\alpha} _i$ and $\boldsymbol{\beta} _i$, all based on

$$\boldsymbol{S}_{11}^{-1 / 2} \boldsymbol{S}_{12} \boldsymbol{S}_{22}^{-1} \boldsymbol{S}_{21} \boldsymbol{S}_{11}^{-1 / 2}$$

in parallel to $\boldsymbol{\Sigma}_{11}^{-1/2} \boldsymbol{\Sigma}_{12} \boldsymbol{\Sigma}_{22}^{-1} \boldsymbol{\Sigma}_{21} \boldsymbol{\Sigma}_{11}^{-1/2}$. Then the $i$-th pair of sample canonical variables $\hat{U}_i, \hat{V}_i$ is


$$
\begin{equation}
\left\{\begin{array}{l}
\hat{U}_{i}=a_{i}^{\prime} \boldsymbol{x}_{1} \\
\hat{V}_{i}=\boldsymbol{b}_{i}^{\prime} \boldsymbol{x}_{2}, \text { where } \boldsymbol{b}_{i}=\boldsymbol{S}_{22}^{-1} \boldsymbol{S}_{21} \boldsymbol{a}_{i}
\end{array}\right.
\end{equation}
$$

### Hypothesis Testing

Since these are not the population quantities and we don’t know whether some $\rho_i$ are zero, or equivalently how many pairs of the canonical variables based on the sample to be retained. This can be answered by testing a sequence of null hypotheses of the form

$$
H_{0}(k): \rho_{k+1}=\cdots=\rho_{p}=0, k=0,1, \cdots, p
$$


until we accept one of them. Note that the first hypothesis of retaining no pair,

$$
H_{0}(k=0): \rho_{1}=\cdots=\rho_{p}=0
$$

is equivalent to independence between $\boldsymbol{x} 1$ and $\boldsymbol{x} _2$, or $H_0: \boldsymbol{\Sigma} _{12} = 0$. If it is rejected, test

$$H_{0}(k=1): \rho_{2}=\cdots=\rho_{p}=0$$

i.e., retain only the first pair; if rejected, test

$$
H_{0}(k=2): \rho_{3}=\cdots=\rho_{p}=0
$$

i.e., retain only the first two pairs, etc., until we obtain $k$ such that

$$
H_{0}(k): \rho_{k+1}=\cdots=\rho_{p}=0
$$

is accepted. Then we shall retain only the first $k$ pairs of canonical variables to describe the linear association between $\boldsymbol{x} _1$ and $\boldsymbol{x} _2$.



## Interpretation


The meanings of the canonical variables are to be interpreted either
- in terms of the relative weighting of the coefficients associated with the original variables, or
- by comparison of the correlations of a canonical variable with original variables.

It is an art to provide a good name to a canonical variable that represents the interpretation and often requires subject-matter knowledge in the field.


## Extension

### Regularizing CCA
