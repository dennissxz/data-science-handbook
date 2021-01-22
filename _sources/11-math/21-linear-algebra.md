# Linear Algebra


## Matrix Operations


### Transpose

Definition
: $\boldsymbol{a} ^\top =\left( a_{ji} \right)$

Properties
: $\ $

  $$
  \begin{aligned}
  \left(\boldsymbol{A}^{\top}\right)^{\top} &=\boldsymbol{A} \\
  (\boldsymbol{A}+\boldsymbol{B})^{\top} &=\boldsymbol{A}^{\top}+\boldsymbol{B}^{\top} \\
  (\boldsymbol{A} \boldsymbol{B})^{\top} &=\boldsymbol{B}^{\top} \boldsymbol{A}^{\top} \\
  \left(\begin{array}{ll}
  \boldsymbol{A} & \boldsymbol{B} \\
  \boldsymbol{C} & \boldsymbol{D}
  \end{array}\right)^{\top} &=\left(\begin{array}{l}
  \boldsymbol{A}^{\top} \boldsymbol{C}^{\top} \\
  \boldsymbol{B}^{\top} & \boldsymbol{D}^{\top}
  \end{array}\right)
  \end{aligned}
  $$


### Determinant

Definition
: The determinant is a scalar value function of a square matrix. The Leibniz formula is

  $$
  \operatorname{det}(A)=\sum_{\sigma \in S_{n}}\left(\operatorname{sgn}(\sigma) \prod_{i=1}^{n} a_{i, \sigma_{i}}\right)
  $$

  where
  - $\sigma$ is a permutation of set ${1, 2, \ldots, n}$.
  - $S_n$ is the set of all such permutations.
  - $\sigma_i$ is the value in the $i$-th position after the reordering $\sigma$.
  - $\operatorname{sgn}{\sigma}$ is the signature of $\sigma$, which is $1$ if the reordering given by $\sigma$ can be achieved by successively interchanging two entries an even number of times, and $-1$ otherwise.

Properties
: $\ $

  $$
  \begin{align}
  |\boldsymbol{A} \boldsymbol{B}|&=|\boldsymbol{A}||\boldsymbol{B}| \\
  \left\vert\begin{array}{l}
  \boldsymbol{A} \quad C \\
  0 \quad \boldsymbol{B}
  \end{array}\right\vert&=| \boldsymbol{A}|| \boldsymbol{B} \mid \\
  \left|\boldsymbol{I}_{p}+\boldsymbol{A} \boldsymbol{B}\right|&=\left|\boldsymbol{I}_{q}+\boldsymbol{B} \boldsymbol{A}\right|
  \end{align}
  $$

### Inverse

Definition
: The inverse of a square matrix is $\boldsymbol{A} ^{-1}$ such that $\boldsymbol{A} \boldsymbol{A} ^{-1} = \boldsymbol{I}$.

Properties
: $\ $

  $$
  \begin{align}
  \left(\boldsymbol{A}^{\top}\right)^{-1}&=\left(\boldsymbol{A}^{-1}\right)^{\top} \\
  (\boldsymbol{A} \boldsymbol{B})^{-1}&=\boldsymbol{B}^{-1} \boldsymbol{A}^{-1} \\
  \left|\boldsymbol{A}^{-1}\right|&=|\boldsymbol{A}|^{-1} \\
  (\boldsymbol{A}+\boldsymbol{C B D})^{-1}
  &=\boldsymbol{A}^{-1}-\boldsymbol{A}^{-1} \boldsymbol{C B}\left(\boldsymbol{B}+\boldsymbol{B D A}^{-1} \boldsymbol{C B}\right)^{-1} \boldsymbol{B D A}^{-1} \\
  \left(\boldsymbol{A}+\boldsymbol{c} \boldsymbol{d}^{\top}\right)^{-1}&=\boldsymbol{A}^{-1}-\frac{\boldsymbol{A}^{-1} \boldsymbol{c} \boldsymbol{d}^{\top} \boldsymbol{A}^{-1}}{1+\boldsymbol{d}^{\top} \boldsymbol{A}^{-1} \boldsymbol{c}} \\
  \left|\boldsymbol{A}+\boldsymbol{c} \boldsymbol{d}^{\top}\right|&=|\boldsymbol{A}|\left(1+\boldsymbol{d}^{\top} \boldsymbol{A}^{-1} \boldsymbol{c}\right)
  \end{align}
  $$

### Trace

Definition
: For a square matrix $A$, $\operatorname{tr}\left( \boldsymbol{A}  \right)$ is the sum of the diagonal elements

  $$
  \operatorname{tr}\left( \boldsymbol{A}  \right) = \sum_i a_{ii}
  $$

Properties
: $\ $

  $$\begin{align}
  \operatorname{tr}(\boldsymbol{A}+\boldsymbol{B})
  &=\operatorname{tr}(\boldsymbol{A})+\operatorname{tr}(\boldsymbol{B}) \\
  \operatorname{tr}(\boldsymbol{A B}) &=\operatorname{tr}(\boldsymbol{B} \boldsymbol{A}) \\
  \operatorname{tr}(\alpha \boldsymbol{A}) &=\alpha \operatorname{tr}(\boldsymbol{A}) \\
  \end{align}$$


### Eigenvalues

Definition
: Let $\boldsymbol{A}$ be an $n\times n$ square matrix and let $\boldsymbol{x}$ be an $n\times 1$ nonzero vector that $\boldsymbol{A} \boldsymbol{x} = \lambda \boldsymbol{x}$. Then, $\lambda$ is called an eigenvalue of $\boldsymbol{A}$ and $\boldsymbol{x}$ is called an eigenvector corresponding to eigenvalue $\lambda$. The eigenvalues are the solutions of the **characteristic function**

  $$
  \left\vert \boldsymbol{A} - \lambda \boldsymbol{I}  \right\vert = 0
  $$

Properties
: For $\boldsymbol{A}_{n\times n}$ with eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$, we have

  $$
  \begin{align}
  \operatorname{tr}(\boldsymbol{A}) &=\sum_{i=1}^{n} \lambda_{i} \\
  |\boldsymbol{A}|&=\prod_{i=1}^{n} \lambda_{i} \\
  \left|\boldsymbol{I}_{n} \pm \boldsymbol{A}\right|&=\prod_{i=1}^{n}\left(1 \pm \lambda_{i}\right)
  \end{align}
  $$

  The nonzero eigenvalues of $\boldsymbol{A} \boldsymbol{B}$ are the same as those of $\boldsymbol{B} \boldsymbol{A}$

## Special Matrices


### Symmetric Matrices

Definition
: A matrix $\boldsymbol{A}$ is **symmetric** if $\boldsymbol{A} ^\top =\boldsymbol{A}$. This is denoted by $\boldsymbol{A} \in \mathrm{Sym}$

### Orthogonal Matrices

Aka rotation matrices.

Definition
: A matrix $\boldsymbol{A}$ is **orthogonal** if $\boldsymbol{A} ^{-1} = \boldsymbol{A} ^\top$.

Properties
: Transformation by $\boldsymbol{U}$ preserves vector length an angle.

  $$
  \begin{equation}
  \|\mathbf{U x}\|=\|\mathbf{x}\|,(\mathbf{U x})^{T} \mathbf{U} \mathbf{y}=\mathbf{x}^{T} \mathbf{y}
  \end{equation}
  $$


### Idempotent Matrices

Definition
: A matrix $\boldsymbol{A}$ is idempotent if $\boldsymbol{A} ^2 = \boldsymbol{A}$.

Properties
: If $\boldsymbol{A}$ is idempotent,
  - $\boldsymbol{I} - \boldsymbol{A}$ is idempotent
  - $\boldsymbol{P} ^\top \boldsymbol{A} \boldsymbol{P} \boldsymbol{x}$  
  - and symmetric
    - $\operatorname{rank}\left( \boldsymbol{A} \right) = \operatorname{tr}\left( \boldsymbol{A}  \right)$
    - $\operatorname{rank}\left( \boldsymbol{A}  \right) = r \Rightarrow \boldsymbol{A}$ has $r$ eigevalues equal to 1 and $n-r$ equal to $0$.
    - $\operatorname{rank}\left( \boldsymbol{A}  \right) = n \Rightarrow \boldsymbol{A} = \boldsymbol{I} _n$


### Projection Matrices


### Positive Semi-Definite and Positive Definite

Definitions

: A symmetric matrix $\boldsymbol{A}$ is **positive and semi-defiinite** (p.s.d.) if $\boldsymbol{A}$ is symmetric and $\boldsymbol{c}^\top \boldsymbol{A} \boldsymbol{c} \ge 0$ for all $\boldsymbol{c}$. This is denoted by $\boldsymbol{A} \succ \boldsymbol{0}$ or $\boldsymbol{A} \in \mathrm{PD}$.

: A symmetric matrix $\boldsymbol{A}$ is **positive and definite** (p.d.) if $\boldsymbol{A}$ is symmetric and $\boldsymbol{c}^\top \boldsymbol{A} \boldsymbol{c} \ge 0$ for all $\boldsymbol{c}\ne \boldsymbol{0}$. This is denoted by $\boldsymbol{A} \succeq \boldsymbol{0}$ or $\boldsymbol{A} \in \mathrm{PSD}$.


Properties
: $\ $

$$\begin{align}
\boldsymbol{A} \in \mathrm{PD} &\Leftrightarrow \lambda_i(\boldsymbol{A}) > 0 \\
&\Leftrightarrow \exists \text{ non-singular } \boldsymbol{R}: \boldsymbol{A} = \boldsymbol{R} \boldsymbol{R} ^\top\\
&\Rightarrow \boldsymbol{A} \text{ is nonsingular} \\
\boldsymbol{A} \in \mathrm{PSD} &\Leftrightarrow \lambda_i(\boldsymbol{A}) \ge 0 \\
&\Leftrightarrow \exists \text{ square } \boldsymbol{R}, \operatorname{rank}\left( R \right) = \operatorname{rank}\left( \boldsymbol{A}  \right): \boldsymbol{A} = \boldsymbol{R} \boldsymbol{R} ^\top\\
\text{square } \boldsymbol{B} &\Rightarrow \boldsymbol{B} ^\top \boldsymbol{B} \in \mathrm{PSD}\\
\end{align}$$


```{note}
If $\boldsymbol{A}$ is p.s.d. (p.d.) there exists a p.s.d. (p.d.) matrix $\boldsymbol{B}$ such that $\boldsymbol{A} = \boldsymbol{B} ^2$. The matrix $\boldsymbol{B}$ is written as $\boldsymbol{A} ^{\frac{1}{2} }$ notationally.
```

Inequalities
: If $\boldsymbol{A}$ is p.d., then for all $\boldsymbol{a}$

  $$
  \frac{\left(\boldsymbol{a}^{\top} \boldsymbol{b} \right)^{2}}{\boldsymbol{a}^{\top} \boldsymbol{A} \boldsymbol{a}} \leq \boldsymbol{b} ^{\top} \boldsymbol{A}^{-1} \boldsymbol{b}
  $$

  The equality holds when $\boldsymbol{a} \propto \boldsymbol{R} ^{-1} \boldsymbol{b}$.

  The inequality can be proved by Cauchy-Schwarz inequality where $\boldsymbol{u} = \boldsymbol{R} ^\top \boldsymbol{a} , \boldsymbol{v} = \boldsymbol{R} ^{-1} \boldsymbol{b}$.

: If $\boldsymbol{A}$ is symmetric and $\boldsymbol{B}$ is p.d., both of size $n \times n$, then for all $\boldsymbol{a}$ ,

  $$
  \lambda_{\min}(\boldsymbol{B} ^{-1} \boldsymbol{A} )
  \le
  \frac{\boldsymbol{a} ^\top \boldsymbol{A} \boldsymbol{a} }{\boldsymbol{a} ^\top \boldsymbol{B} \boldsymbol{a} }  
  \le
  \lambda_{\max}(\boldsymbol{B} ^{-1} \boldsymbol{A} )
  $$

  The equality on either side holds when $\boldsymbol{a}$ is proportional to the corresponding eigenvector.

: If $\boldsymbol{A}$ and $\boldsymbol{B}$ are p.d.,

  $$
  \max _{a, b} \frac{\left(\boldsymbol{a}^{\top} \boldsymbol{D} \boldsymbol{b}\right)^{2}}{\boldsymbol{a}^{\top} \boldsymbol{A} \boldsymbol{a} \cdot \boldsymbol{b}^{\top} \boldsymbol{B} \boldsymbol{b}}=\theta
  $$

  where $\theta$ is the largest eigenvalue of $\boldsymbol{A} ^{-1} \boldsymbol{D} \boldsymbol{B} ^{-1} \boldsymbol{D} ^\top$ or $\boldsymbol{B} ^{-1} \boldsymbol{D} ^\top \boldsymbol{A} ^{-1} \boldsymbol{D}$.

  The maximum is obtained when $\boldsymbol{a}$ is proportional to an eigenvector of $\boldsymbol{A} ^{-1} \boldsymbol{D} \boldsymbol{B} ^{-1} \boldsymbol{D} ^\top$ corresponding to $\theta$, $\boldsymbol{b}$ is proportional to an eigenvector of $\boldsymbol{B} ^{-1} \boldsymbol{D} ^\top \boldsymbol{A} ^{-1} \boldsymbol{D}$ corresponding to $\theta$.

: If $\boldsymbol{A} , \boldsymbol{\Sigma}$ are p.d., then the function

  $$
  f(\boldsymbol{\Sigma} ) = \log \left\vert \boldsymbol{\Sigma}  \right\vert + \operatorname{tr}\left( \boldsymbol{\Sigma} ^{-1} \boldsymbol{A}  \right)
  $$

  is minimized uniquely at $\boldsymbol{\Sigma} =\boldsymbol{A}$.

## Matrix Differentiation

Definitions
: $\ $

  $$
  \begin{array}{l}
  \frac{\partial y}{\partial \boldsymbol{x}}=\left(\begin{array}{c}
  \frac{\partial y}{\partial x_{1}} \\
  \vdots \\
  \frac{\partial y}{\partial x_{n}}
  \end{array}\right) \\
  \frac{\partial y}{\partial \boldsymbol{X}}=\left(\begin{array}{ccc}
  \frac{\partial y}{\partial x_{11}} & \cdots & \frac{\partial y}{\partial x_{1 n}} \\
  \vdots & & \vdots \\
  \frac{\partial y}{\partial x_{n 1}} & \cdots & \frac{\partial y}{\partial x_{n n}}
  \end{array}\right)
  \end{array}
  $$

Properties
: $\ $

  $$
  \begin{aligned}
  \frac{\partial \boldsymbol{a}^{\top} \boldsymbol{x}}{\partial \boldsymbol{x}}&=\boldsymbol{a}\\
  \frac{\partial \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}}{\partial \boldsymbol{x}}&=2 \boldsymbol{A} \boldsymbol{x} \text { if } \boldsymbol{A} \text { is symmetric. }\\
  \frac{\partial \operatorname{tr}(\boldsymbol{X})}{\partial \boldsymbol{X}}&=\boldsymbol{I}\\
  \frac{\partial \operatorname{tr}(\boldsymbol{A} \boldsymbol{X})}{\partial \boldsymbol{X}}&=\left\{\begin{array}{ll}
  \boldsymbol{A}^{\top}  &\text { if all elements of } \boldsymbol{X} \text { are distinct } \\
  \boldsymbol{A}+\boldsymbol{A}^{\top}-\operatorname{diag}(\boldsymbol{A})  &\text { if } \boldsymbol{X} \text { is symmetric. }
  \end{array}\right.\\
  \frac{\partial|\boldsymbol{X}|}{\partial \boldsymbol{X}}&=\left\{\begin{array}{ll}
  |\boldsymbol{X}|\left(\boldsymbol{X}^{-1}\right)^{\top}  &\text { if all elements of } \boldsymbol{X} \text { are distinct } \\
  |\boldsymbol{X}|\left(2 \boldsymbol{X}^{-1}-\operatorname{diag}\left(\boldsymbol{X}^{-1}\right)\right)^{\top}  &\text { if } \boldsymbol{X} \text { is symmetric. }
  \end{array}\right.
  \end{aligned}
  $$

## Matrix Decomposition

Summary table


### Eigenvalue Decomposition

[detail] If $\boldsymbol{A}$  has $n$ independent eigenvectors, then is has EVD.

[detail] $\boldsymbol{A}$ is symmetric $\Leftrightarrow$ there exists an orthogonal matrix $\boldsymbol{U} = \left[ \boldsymbol{u} _1, \boldsymbol{u} _2, \ldots, \boldsymbol{u} _n \right]$ such that

$$
\boldsymbol{U}^{\top} \boldsymbol{A} \boldsymbol{U}=\boldsymbol{\Lambda} = \left[\begin{array}{cccc}
\lambda_{1} & 0 & \cdots & 0 \\
0 & \lambda_{2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_{n}
\end{array}\right]
$$

or

$$
\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U} ^\top
$$

where $\lambda_1, \lambda_2, \ldots, \lambda_n$ are eigenvalues of $\boldsymbol{A}$ and $\boldsymbol{u}  _i$ are their corresponding eigenvectors.

Corollary
: the eigenvalues of symmetric matrices are real.
Proof:

$$
\begin{equation}
\begin{array}{l}
\mathbf{A} \mathbf{u}=\lambda \mathbf{u} \Rightarrow \mathbf{u}^{\star T} \mathbf{A} \mathbf{u}=\lambda \mathbf{u}^{\star T} \mathbf{u} \\
\mathbf{A} \mathbf{u}^{\star}=\lambda^{\star} \mathbf{u}^{\star} \Rightarrow \mathbf{u}^{T} \mathbf{A} \mathbf{u}^{\star}=\lambda^{\star} \mathbf{u}^{T} \mathbf{u}^{\star} \Rightarrow \mathbf{u}^{\star T} \mathbf{A}^{T} \mathbf{u}=\lambda^{\star} \mathbf{u}^{\star T} \mathbf{u} \\
\mathbf{A}=\mathbf{A}^{T} \Rightarrow\left(\lambda-\lambda^{\star}\right) \mathbf{u}^{\star T} \mathbf{u}=0 \Rightarrow \lambda=\lambda^{\star}
\end{array}
\end{equation}
$$

### Cholesky Decomposition

If $\boldsymbol{A}$ is p.s.d. (p.d.), there exists a (unique) upper triangular matrix $\boldsymbol{U}$ with non-negative (positive) diagonal elements such that

$$
\boldsymbol{A} = \boldsymbol{U} ^\top \boldsymbol{U}
$$

Cholesky decomposition is a special case of LU decomposition.

### Canonical Decomposition

If $\boldsymbol{A}$ is symmetric and $\boldsymbol{B}$ is p.d., then there exists a non-singular matrix $\boldsymbol{P}$ such that

$$\boldsymbol{P} ^\top \boldsymbol{A} \boldsymbol{P} = \boldsymbol{\Lambda} \text{ and } \boldsymbol{P} ^\top \boldsymbol{B} \boldsymbol{P} = \boldsymbol{I} _n$$

where $\boldsymbol{\Lambda} =\operatorname{diag}\left( \lambda_1, \lambda_2, \ldots, \lambda_n \right)$ and the $\lambda_i$ are the eigenvalues of $\boldsymbol{B} ^{-1} \boldsymbol{A}$ or $\boldsymbol{A} \boldsymbol{B} ^{-1}$.

### Singular Value Decomposition

Definition
: For any matrix $\boldsymbol{A} \in \mathbb{R} ^{n \times p}$, we can write $\boldsymbol{A} = \boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V} ^\top$. where
- $\boldsymbol{U} \in \mathbb{R} ^{n \times n}$ and $\boldsymbol{V} \in \mathbb{R} ^{p\times p}$ are orthogonal matrices.
- $\\boldsymbol{\Sigma}$ is a diagonal matrix.


Properties
: We can also write SVD as


$$
\mathbf{A}=\sigma_{1} \mathbf{u}_{1} \mathbf{v}_{1}^{T}+\sigma_{2} \mathbf{u}_{2} \mathbf{v}_{2}^{T}+\ldots+\sigma_{r} \mathbf{u}_{r} \mathbf{v}_{r}^{T}
$$

where $r = \operatorname{rank}\left( \boldsymbol{A}  \right)$.

: As a result, $\mathbf{A} \mathbf{v}=\sigma \mathbf{u}, \mathbf{A}^{T} \mathbf{u}=\sigma \mathbf{v}$.

Theorem
: Every matrix has SVD.

### QR Decomposition

### LU Decomposition


### Schur Decomposition
