# Linear Algebra

<!-- TOC -->

- [Linear Algebra](#linear-algebra)
  - [Operations](#operations)
    - [Transpose](#transpose)
    - [Determinant](#determinant)
    - [Inverse](#inverse)
    - [Trace](#trace)
    - [Eigenvalues](#eigenvalues)
  - [Special Matrices](#special-matrices)
    - [Orthogonal Matrices](#orthogonal-matrices)
    - [Positive Semi-Definite and Positive Definite](#positive-semi-definite-and-positive-definite)
    - [Idempotent Matrices](#idempotent-matrices)
    - [Projection Matrices](#projection-matrices)
  - [Matrix Decomposition](#matrix-decomposition)
    - [Eigenvalue Decomposition](#eigenvalue-decomposition)
    - [Singular Value Decomposition](#singular-value-decomposition)
    - [QR Decomposition](#qr-decomposition)
    - [LU Decomposition](#lu-decomposition)
    - [Schur Decomposition](#schur-decomposition)

<!-- /TOC -->




## Operations



### Transpose

$$
\begin{aligned}
\left(\boldsymbol{A}^{\prime}\right)^{\prime} &=\boldsymbol{A} \\
(\boldsymbol{A}+\boldsymbol{B})^{\prime} &=\boldsymbol{A}^{\prime}+\boldsymbol{B}^{\prime} \\
(\boldsymbol{A} \boldsymbol{B})^{\prime} &=\boldsymbol{B}^{\prime} \boldsymbol{A}^{\prime} \\
\left(\begin{array}{ll}
\boldsymbol{A} & \boldsymbol{B} \\
\boldsymbol{C} & \boldsymbol{D}
\end{array}\right)^{\prime} &=\left(\begin{array}{l}
\boldsymbol{A}^{\prime} \boldsymbol{C}^{\prime} \\
\boldsymbol{B}^{\prime} & \boldsymbol{D}^{\prime}
\end{array}\right)
\end{aligned}
$$


### Determinant

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


$$
\begin{align}
\boldsymbol{A}^{-1} \boldsymbol{A}&=\boldsymbol{A} \boldsymbol{A}^{-1} \\
&=I \\
\left(\boldsymbol{A}^{\prime}\right)^{-1}&=\left(\boldsymbol{A}^{-1}\right)^{\prime} \\
(\boldsymbol{A} \boldsymbol{B})^{-1}&=\boldsymbol{B}^{-1} \boldsymbol{A}^{-1} \\
\left|\boldsymbol{A}^{-1}\right|&=|\boldsymbol{A}|^{-1} \\
(\boldsymbol{A}+\boldsymbol{C B D})^{-1}
&=\boldsymbol{A}^{-1}-\boldsymbol{A}^{-1} \boldsymbol{C B}\left(\boldsymbol{B}+\boldsymbol{B D A}^{-1} \boldsymbol{C B}\right)^{-1} \boldsymbol{B D A}^{-1} \\
\left(\boldsymbol{A}+\boldsymbol{c} \boldsymbol{d}^{\prime}\right)^{-1}&=\boldsymbol{A}^{-1}-\frac{\boldsymbol{A}^{-1} \boldsymbol{c} \boldsymbol{d}^{\prime} \boldsymbol{A}^{-1}}{1+\boldsymbol{d}^{\prime} \boldsymbol{A}^{-1} \boldsymbol{c}} \\
\left|\boldsymbol{A}+\boldsymbol{c} \boldsymbol{d}^{\prime}\right|&=|\boldsymbol{A}|\left(1+\boldsymbol{d}^{\prime} \boldsymbol{A}^{-1} \boldsymbol{c}\right)
\end{align}
$$

### Trace

For a square matrix $A$, $\operatorname{tr}\left( \boldsymbol{A}  \right)$ is the sum of the diagonal elements, $\operatorname{tr}\left( \boldsymbol{A}  \right) = \sum_i a_{ii}$

$$\begin{align}
\operatorname{tr}(\boldsymbol{A}+\boldsymbol{B})
&=\operatorname{tr}(\boldsymbol{A})+\operatorname{tr}(\boldsymbol{B}) \\
\operatorname{tr}(\boldsymbol{A B}) &=\operatorname{tr}(\boldsymbol{B} \boldsymbol{A}) \\
\operatorname{tr}(\alpha \boldsymbol{A}) &=\alpha \operatorname{tr}(\boldsymbol{A}) \\
\end{align}$$


### Eigenvalues

Let $\boldsymbol{A}$ be an $n\times n$ square matrix and let $\boldsymbol{x}$ be an $n\times 1$ nonzero vector that $\boldsymbol{A} \boldsymbol{x} = \lambda \boldsymbol{x}$. Then, $\lambda$ is called an eigenvalue of $\boldsymbol{A}$ and $\boldsymbol{x}$ is called an eigenvector corresponding to eigenvalue $\lambda$. The eigenvalues are the solutions of

$$
\left\vert \boldsymbol{A} - \lambda \boldsymbol{I}  \right\vert = 0
$$

For $\boldsymbol{A}_{n\times n}$ with eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$, we have

$$
\begin{align}
\operatorname{tr}(\boldsymbol{A}) &=\sum_{i=1}^{n} \lambda_{i} \\
|\boldsymbol{A}|&=\prod_{i=1}^{n} \lambda_{i} \\
\left|\boldsymbol{I}_{n} \pm \boldsymbol{A}\right|&=\prod_{i=1}^{n}\left(1 \pm \lambda_{i}\right)
\end{align}
$$

The nonzero eigenvalues of $\boldsymbol{A} \boldsymbol{B}$ are the same as those of $\boldsymbol{B} \boldsymbol{A}$

## Special Matrices

### Orthogonal Matrices




### Positive Semi-Definite and Positive Definite

### Idempotent Matrices

### Projection Matrices

## Matrix Decomposition

Summary table


### Eigenvalue Decomposition

Eigenvectors and Eigenvalues

### Singular Value Decomposition

### QR Decomposition

### LU Decomposition

Cholesky

### Schur Decomposition
