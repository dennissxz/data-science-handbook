
# Canonical Corerlation Analysis

## Objective

Hotelling (1936) extended the idea of multiple correlation to the problem of measuring linear association between two **groups** of variables, say $\boldsymbol{x}$ and $\boldsymbol{y}$. Thus we consider the simple correlation coefficient between every pair of **linear combinations** of elements of $\boldsymbol{x}$ and of $\boldsymbol{y}$ and choose the maximum.

$$
\max _{\boldsymbol{v}, \boldsymbol{w} } \, \operatorname{Corr}\left(\boldsymbol{v}^{\top} \boldsymbol{x} , \boldsymbol{w}^{\top} \boldsymbol{y} \right)
$$

This maximum, $\rho_1$, is called the first canonical correlation and the corresponding pair of linear combinations, say $(U_1, V_1)$, is called the first pair of canonical variables.


Since each group has more than one variable, one single correlation coefficient may miss significant linear association in other dimensions. So we consider the subclass of **all** pairs of linear combinations of elements of $\boldsymbol{x}$ and $\boldsymbol{y}$ whose members are **uncorrelated** with $(U_1, V_1）$. The maximum simple correlation coefficient, $\rho _2$, of such pairs is called the second canonical correlation and the pair, say $(U_2, V_2)$, achieving this maximum is called the second pair of canonical variables.

Continuing this way we shall get the $k$ canonical correlations and the corresponding pairs of canonical variables, where $k=\min(d_x, d_y)$ is the smallest dimension of $\boldsymbol{x}$ and $\boldsymbol{y}$.

Alternative formulations include

- minimizing the difference between two projected spaces, i.e. we want to predict projected $\boldsymbol{y}$ by projected $\boldsymbol{x}$.

    $$
    \begin{equation}
    \begin{array}{ll}
    \min & \left\| \boldsymbol{X} \boldsymbol{V} - \boldsymbol{Y}\boldsymbol{W} \right\|_{F}^{2} \\
    \text {s.t.} & \boldsymbol{V} ^\top \boldsymbol{\Sigma} _{xx} \boldsymbol{V} = \boldsymbol{W} ^\top \boldsymbol{\Sigma} _{y y} \boldsymbol{W} = \boldsymbol{I}_k  \\
    & \boldsymbol{V} \in \mathbb{R} ^{d_x \times k} \quad \boldsymbol{W} \in \mathbb{R} ^{d_y \times k} \\
    \end{array}
    \end{equation}
    $$

- maximizing the trace

    $$
    \begin{equation}
    \begin{array}{ll}
    \max & \operatorname{tr} \left( \boldsymbol{V} ^\top \boldsymbol{\Sigma} _{x y} \boldsymbol{W}  \right) \\
    \text {s.t.} & \boldsymbol{V} ^\top \boldsymbol{\Sigma} _{x x} \boldsymbol{V} = \boldsymbol{W} ^\top \boldsymbol{\Sigma} _{y y} \boldsymbol{W} = \boldsymbol{I}_k \\
    & \boldsymbol{V} \in \mathbb{R} ^{d_x \times k} \quad \boldsymbol{W} \in \mathbb{R} ^{d_y \times k} \\
    \end{array}
    \end{equation}
    $$



## Learning

Partition the covariance matrix of full rank in accordance with two groups of variables,

$$
\begin{equation}
\operatorname{Var}\left(\begin{array}{l}
\boldsymbol{x}  \\
\boldsymbol{y}
\end{array}\right)=\left(\begin{array}{ll}
\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}
\end{array}\right)
\end{equation}
$$

### Sequential Optimization

The first canonical correlation, $\rho _1$, equals the maximum correlation between all pairs of linear combinations of $\boldsymbol{x}$ and $\boldsymbol{y}$ with unit variance. That is,


$$\begin{align}
\max _{\boldsymbol{v}, \boldsymbol{w} } \operatorname{Corr}\left(\boldsymbol{v}^{\top} \boldsymbol{x} , \boldsymbol{w}^{\top} \boldsymbol{y} \right)
= \max &  _{\boldsymbol{v}, \boldsymbol{w}}  \frac{\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}}{\sqrt{\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v} \boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}}} \\
\text{s.t.}  &  \ \quad \boldsymbol{v} ^\top \boldsymbol{\Sigma} _{11} \boldsymbol{v} =1 \\
  &  \ \quad \boldsymbol{w} ^\top \boldsymbol{\Sigma} _{22} \boldsymbol{w} =1 \\
\end{align}$$

If the maximum is achieved at $\boldsymbol{v} _1$ and $\boldsymbol{w} _1$, then the first pair of canonical varibales are defined as


$$
U_{1}=\boldsymbol{v}_{1}^{\top} \boldsymbol{x} , \quad V_{1}=\boldsymbol{w}_{1}^{\top} \boldsymbol{y}
$$

Successively, for $i = 2, \ldots, k$, the $i$-th canonical correlation $\rho_i$ is defined as


$$\begin{align}
\rho_{i}
 =\max _{\boldsymbol{v}, \boldsymbol{w}} & \quad \operatorname{corr}\left(\boldsymbol{v}^{\top} \boldsymbol{x} , \boldsymbol{w}^{\top} \boldsymbol{y} \right) \\
 \text{s.t.} &  \quad (\boldsymbol{v} ^\top \boldsymbol{x} , \boldsymbol{w} ^\top \boldsymbol{y} ) \text{ uncorrelated with } (U_1, V_1), \ldots, (U_{i-1}, V_{i-1}) \\
\end{align}$$

If the maximum is achived at $\boldsymbol{v} _i$ and $\boldsymbol{w} _i$, then the first pair of canonical varibales are defined as


$$
U_{i}=\boldsymbol{v}_{i}^{\top} \boldsymbol{x}_{i}, \quad V_{i}=\boldsymbol{w}_{i}^{\top} \boldsymbol{y}
$$

### Spectral Decomposition


Rather than obtaining pairs of canonical variables and canonical correlation sequentially, it can be shown that the canonical correlations $\rho$'s and hence pairs of canonical variables $(U,V)$’s can be obtained simultaneously by solving for
the eigenvalues $\rho^2$'s and eigenvectors $\boldsymbol{v}$’s of the matrix


$$
\boldsymbol{\Sigma}_{xx}^{-1} \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx}
$$

A difficulty of this problem is that the matrix $\boldsymbol{\Sigma}_{xx}^{-1} \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx}$ is not symmetric. Consequently, the symmetric matrix

$$
\boldsymbol{\Sigma}_{xx}^{-1/2} \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx} \boldsymbol{\Sigma}_{xx}^{-1/2}
$$

is considered instead for the computational efficiency. Note that the two matrices possess the **same** eigenvalues and their eigenvectors are linearly related.

It turns out that the canonical correlation $\rho_i$ and the canonical variables $(\boldsymbol{v} _i, \boldsymbol{w} _i), i = 1, 2, \ldots, k$ are related to matrix eigen-analysis:


$$
\begin{equation}
\begin{aligned}
\rho_{1}^{2} \geq \rho_{2}^{2} \geq \cdots \geq \rho_{k}^{2} & \quad \text {eigenvalues of } \boldsymbol{\Sigma}_{xx}^{-1 / 2} \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx} \boldsymbol{\Sigma}_{xx}^{-1 / 2}\\
\boldsymbol{v}_{1}^{\star}\quad \boldsymbol{v}_{2}^{\star} \quad\cdots \quad \boldsymbol{v}_{k}^{\star} & \quad \text { associated unit-norm eigenvectors,}\left(\boldsymbol{v}_{i}^{\star}\right)^{\top} \boldsymbol{v}_{i}^{\star}=1
\\
\boldsymbol{v}_{i}=\boldsymbol{v}_{i}^{\star} / \sqrt{\left(\boldsymbol{v}_{i}^{\star}\right)^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v}_{i}^{\star}} & \quad \text { coefficients of } U_{i}, \boldsymbol{v}_{i}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v}_{i}=1\\
\boldsymbol{v}_{i}=\boldsymbol{\Sigma}_{xx}^{-1 / 2} \boldsymbol{v}_{i}^{\star} & \quad \text { 2nd formula for coefficients of } U_{i}\\
\boldsymbol{w}_{i}=\boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx} \boldsymbol{v}_{i} & \quad \text { coefficients of } V_{i}, \boldsymbol{w}_{i}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}_{i}=1
\end{aligned}
\end{equation}
$$

where $k = \min(d_x, d_y)$

:::{admonition,dropdown,seealso}

Recall the formula for the [covariance matrix](prob-covariance-matrix-of-two-vectors) of two vectors.

We consider the following maximization problem:

$$
\begin{aligned}
\rho^{2} \equiv \max _{\boldsymbol{v}, \boldsymbol{w}} \operatorname{Corr}^{2}\left(\boldsymbol{v}^{\top} \boldsymbol{x} , \boldsymbol{w}^{\top} \boldsymbol{y} \right)=

\max &\  _{\boldsymbol{v}, \boldsymbol{w}} \frac{\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right)^{2}}{\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v}\right)\left(\boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}\right)} \quad  \\

\text {s.t.} &\ \boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v}=\boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}=1
\end{aligned}
$$

The Lagrangean is


$$
\begin{equation}
L(\boldsymbol{v}, \boldsymbol{w}, \lambda, \theta)=\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right)^{2}-\lambda\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v}-1\right)-\theta\left(\boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}-1\right)
\end{equation}
$$

The first order conditions are


$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial \boldsymbol{v}}=& 2\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right) \boldsymbol{\Sigma}_{xy} \boldsymbol{w}-2 \lambda \boldsymbol{\Sigma}_{xx} \boldsymbol{v}=\mathbf{0} \\
\Rightarrow \qquad \qquad &\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right) \boldsymbol{\Sigma}_{xy} \boldsymbol{w}=\lambda \boldsymbol{\Sigma}_{xx} \boldsymbol{v} \\
\frac{\partial L}{\partial \boldsymbol{w}}=& 2\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right) \boldsymbol{\Sigma}_{yx} \boldsymbol{v}-2 \theta \boldsymbol{\Sigma}_{yy} \boldsymbol{w}=\mathbf{0} \\
\Rightarrow \qquad \qquad &\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right) \boldsymbol{\Sigma}_{yx} \boldsymbol{v}=\theta \boldsymbol{\Sigma}_{yy} \boldsymbol{w} \\
\quad \frac{\partial L}{\partial \lambda}=& 1-\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v}=0 \\
\Rightarrow \qquad \qquad & \boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v}=1 \\
\quad \frac{\partial L}{\partial \theta}=& 1-\boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}=0 \\
\Rightarrow \qquad \qquad & \boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}=1
\end{aligned}
\end{equation}
$$

Premultiply the first condition by $\boldsymbol{v} T$ and the second condition by $\boldsymbol{w} ^\top$, we have


$$
\begin{equation}
\begin{array}{l}
\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right)^{2}=\lambda \boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v} = \lambda \\
\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right)^{2}=\theta \boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w} = \theta
\end{array}
\end{equation}
$$

Hence


$$
\begin{equation}
\lambda=\theta=\left(\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}\right)^{2}
\end{equation}
$$

which implies that the Lagrangian multipliers are equal to the maximized value of squared correlation $\begin{equation}
\operatorname{Corr}^{2}\left(\boldsymbol{v}^{\top} \boldsymbol{x} , \boldsymbol{w}^{\top} \boldsymbol{y} \right)
\end{equation}$, i.e., $\rho^2$.

Based on the above results, we can further simplify the first and second conditions by replacing $\boldsymbol{v} ^\top \boldsymbol{\Sigma} _{12} \boldsymbol{w}$ by $\sqrt{\lambda}$ and $\sqrt{\theta}$ respectively.


$$
\begin{equation}
\begin{aligned}
\left\{\begin{array}{l}
\boldsymbol{\Sigma}_{xy} \boldsymbol{w}-\sqrt{\lambda} \boldsymbol{\Sigma}_{xx} \boldsymbol{v} & =\mathbf{0} \\
\boldsymbol{\Sigma}_{yx} \boldsymbol{v}-\sqrt{\lambda} \boldsymbol{\Sigma}_{yy} \boldsymbol{w} & =\mathbf{0}
\end{array}\right.\\
\Rightarrow\left(\begin{array}{cc}
-\sqrt{\lambda} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & -\sqrt{\lambda} \boldsymbol{\Sigma}_{yy}
\end{array}\right) \left(\begin{array}{l}
\boldsymbol{v} \\
\boldsymbol{w}
\end{array}\right)& =\mathbf{0}
\end{aligned}
\end{equation}
$$

In order to obtain the non-trivial solutions for $\boldsymbol{v}$  and $\boldsymbol{w}$, we require


$$
\begin{equation}
\left|\begin{array}{cc}
-\sqrt{\lambda} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & -\sqrt{\lambda} \boldsymbol{\Sigma}_{yy}
\end{array}\right|=0
\end{equation}
$$

which gives


$$
\begin{equation}
\begin{array}{l}
\left|\begin{array}{cc}
-\sqrt{\lambda} \boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\
\boldsymbol{\Sigma}_{yx} & -\sqrt{\lambda} \boldsymbol{\Sigma}_{yy}
\end{array}\right|=\left|\sqrt{\lambda} \boldsymbol{\Sigma}_{yy}\right|\left|\sqrt{\lambda} \boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xy}\left(\sqrt{\lambda} \boldsymbol{\Sigma}_{yy}\right)^{-1} \boldsymbol{\Sigma}_{yx}\right|=0 \\
\Rightarrow\left|\sqrt{\lambda} \boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xy}\left(\sqrt{\lambda} \boldsymbol{\Sigma}_{yy}\right)^{-1} \boldsymbol{\Sigma}_{yx}\right|=0 \quad \because\left|\boldsymbol{\Sigma}_{xx}\right|>0 \\
\Rightarrow\left|(1 / \sqrt{\lambda})\left(\lambda \boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx}\right)\right|=0 & \\
\Rightarrow\left|\boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx}-\lambda \boldsymbol{\Sigma}_{xx}\right|=0 \quad \because \lambda>0
\end{array}
\end{equation}
$$

which indeed is the eigenvalue problem of


$$
\begin{equation}
\boldsymbol{\Sigma}_{xx}^{-1} \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx} \boldsymbol{v}=\lambda \boldsymbol{v}
\end{equation}
$$

Once the solution for $\boldsymbol{v}$  is obtained, the solution for $\boldsymbol{w}$  can be obtained by

$$
\begin{equation}
\boldsymbol{w} \propto \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx} \boldsymbol{v}
\end{equation}
$$

with the normalized condition


$$
\begin{equation}
\boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}=1
\end{equation}
$$
:::

## Properties

**Invariance property**
: Canonical corelations $\rho_i$'s between $\boldsymbol{x}$ and $\boldsymbol{y}$ are the same as those between $\boldsymbol{A} _1 \boldsymbol{x}  + \boldsymbol{c}_1$ and $\boldsymbol{A} _2 \boldsymbol{y}  + \boldsymbol{c} _2$, where both $\boldsymbol{A} _1$ and $\boldsymbol{A} _2$ are non-singular square matrices and their computation can be based on either the partitioned covariance matrix or the partitioned correlation matrix. However, the canonical coefficients contained in $\boldsymbol{v} _k$ and $\boldsymbol{w} _k$ are **not** invariant under the same transforam, nor their estimates.



## Model Selection

Now let $\boldsymbol{S} _{11}, \boldsymbol{S} _{12}, \boldsymbol{S} _{22}$ and $\boldsymbol{S} _{21}$ be the corresponding sub-matrices of the sample covariance matrix $\boldsymbol{S}$. For $i = 1, 2, \ldots, k$, let $r_i ^2, \boldsymbol{a} _i$ and $\boldsymbol{b} _i$ be respectively the sample estimators of $\rho_i^2, \boldsymbol{v} _i$ and $\boldsymbol{w} _i$, all based on

$$\boldsymbol{S}_{11}^{-1 / 2} \boldsymbol{S}_{12} \boldsymbol{S}_{22}^{-1} \boldsymbol{S}_{21} \boldsymbol{S}_{11}^{-1 / 2}$$

in parallel to $\boldsymbol{\Sigma}_{xx}^{-1/2} \boldsymbol{\Sigma}_{xy} \boldsymbol{\Sigma}_{yy}^{-1} \boldsymbol{\Sigma}_{yx} \boldsymbol{\Sigma}_{xx}^{-1/2}$. Then the $i$-th pair of sample canonical variables $\widehat{U}_i, \widehat{V}_i$ is


$$
\begin{equation}
\left\{\begin{array}{l}
\widehat{U}_{i}=\boldsymbol{a} _{i}^{\top} \boldsymbol{x}  \\
\widehat{V}_{i}=\boldsymbol{b}_{i}^{\top} \boldsymbol{y} , \text { where } \boldsymbol{b}_{i}=\boldsymbol{S}_{22}^{-1} \boldsymbol{S}_{21} \boldsymbol{a}_{i}
\end{array}\right.
\end{equation}
$$

### Hypothesis Testing

Since these are not the population quantities and we don’t know whether some $\rho_i$ are zero, or equivalently how many pairs of the canonical variables based on the sample to be retained. This can be answered by testing a sequence of null hypotheses of the form

$$
H_{0}(k): \rho_{k+1}=\cdots=\rho_{k}=0, k=0,1, \cdots, k
$$


until we accept one of them. Note that the first hypothesis of retaining no pair,

$$
H_{0}(k=0): \rho_{1}=\cdots=\rho_{k}=0
$$

is equivalent to independence between $\boldsymbol{x} 1$ and $\boldsymbol{y}$, or $H_0: \boldsymbol{\Sigma} _{12} = 0$. If it is rejected, test

$$H_{0}(k=1): \rho_{2}=\cdots=\rho_{k}=0$$

i.e., retain only the first pair; if rejected, test

$$
H_{0}(k=2): \rho_{3}=\cdots=\rho_{k}=0
$$

i.e., retain only the first two pairs, etc., until we obtain $k$ such that

$$
H_{0}(k): \rho_{k+1}=\cdots=\rho_{k}=0
$$

is accepted. Then we shall retain only the first $k$ pairs of canonical variables to describe the linear association between $\boldsymbol{x}$ and $\boldsymbol{y}$.


## Interpretation


The meanings of the canonical variables are to be interpreted either
- in terms of the relative weighting of the coefficients associated with the original variables, or
- by comparison of the correlations of a canonical variable with original variables.

It is an art to provide a good name to a canonical variable that represents the interpretation and often requires subject-matter knowledge in the field.


## Pros and Cons


### Discriminative Power

Unlike PCA, CCA has discriminative power in some cases. In the comparison below, in the first scatter-plot, the principal direction is the discriminative direction, while in the second plot it is not. The 3rd (same as the 2nd) and the 4th plots corresponds to $\boldsymbol{x} \in \mathbb{R} ^2$ and $\boldsymbol{y} \in \mathbb{R} ^2$. The two colors means two kinds of data points in the $n\times 4$ data set, but the color labels are shown to CCA. The CCA solutions to $\boldsymbol{x}$ (3rd plot) is the direction $(-1,1)$ and to $\boldsymbol{y}$ (4th plot) is the direction $(-1,-1)$. Because they are the highest correlation pair of directions.

:::{figure,myclass} cca-has-disc-power
<img src="../imgs/cca-has-disc-power.png" width = "90%" alt=""/>

CCA has discriminative power [Livescu 2021]
:::


### Overfitting

CCA tends to overfit, i.e. find spurious correlations in the training data. Solutions:

- regularize CCA
- do an initial dimensionality reduction via PCA to filter the tiny signals that are correlated.


## Extension: Regularized CCA

To regularize CCA, we can add small constant $r$ (noise) to the covariance matrices of $\boldsymbol{x}$ and $\boldsymbol{y}$.

$$
\begin{equation}
\boldsymbol{v}_{1}, \boldsymbol{w}_{1}=\underset{\boldsymbol{v}, \boldsymbol{w}}{\operatorname{argmax}} \frac{\boldsymbol{v} ^\top  \boldsymbol{\Sigma}_{x y} \boldsymbol{w}}{\sqrt{\boldsymbol{v} ^\top \left(\boldsymbol{\Sigma}_{x x}+r_{x} I\right) \boldsymbol{v} \boldsymbol{w} ^\top \left(\boldsymbol{\Sigma}_{y y}+r_{y} I\right) \boldsymbol{w}}}
\end{equation}
$$

Then we solve for the eigenvalues's and eigenvectors’s of the new matrix

$$
\begin{equation}
\left(\boldsymbol{\Sigma}_{x x}+r_{x} I\right)^{-1} \boldsymbol{\Sigma}_{x y}\left(\boldsymbol{\Sigma}_{y y}+r_{y} I\right)^{-1} \boldsymbol{\Sigma}_{y x}
\end{equation}
$$
