# Non-linear Programming

**TBC**

Primal and Dual short [intro](https://zhuanlan.zhihu.com/p/46944722).

Lagrange multiplier:

- geometric [motivation](https://www.youtube.com/watch?v=yuqB-d5MjZA&t=16s&ab_channel=KhanAcademy) of the method: align tangency of the objective function and the constraints.
- [formulation](https://www.youtube.com/watch?v=hQ4UNu1P2kw&t=311s&ab_channel=KhanAcademy) of Lagrangean $\mathcal{L}$: combining all equations to $\nabla\mathcal{L} = 0$.
- [interpretation](https://www.youtube.com/watch?v=m-G3K2GPmEQ&t=185s&ab_channel=KhanAcademy) and [proof](https://www.youtube.com/watch?v=b9B2FZ5cqbM&ab_channel=KhanAcademy) of the Lagrange multiplier $\lambda$ as $\frac{\partial f}{\partial c}$, e.g. if budget change, how much will revenue change?

(rayleigh-quotient)=
## Rayleigh Quotients

Consider the following constrained optimization:

$$\begin{aligned}
\max_{\boldsymbol{x}} && \boldsymbol{x} ^{\top} \boldsymbol{A} \boldsymbol{x}  & \\
\mathrm{s.t.}
&& \left\| \boldsymbol{x}  \right\|^2 &= 1  \\
\end{aligned}$$

An equivalent unconstrained problem is

$$\begin{aligned}
\max_{\boldsymbol{x} \ne \boldsymbol{0}} && \frac{\boldsymbol{x} ^{\top} \boldsymbol{A} \boldsymbol{x} }{\boldsymbol{x} ^{\top} \boldsymbol{x} }  & \\
\end{aligned}$$

which makes the objective function invariant to scaling of $\boldsymbol{x}$. How do we solve this?

Definition (Quadratic forms)
: Let $\boldsymbol{A}$ be a symmetric real matrix. A quadratic form corresponding to $\boldsymbol{A}$ is a function $Q: \mathbb{R} ^n \rightarrow \mathbb{R}$ with

  $$
  Q_{\boldsymbol{A}}(\boldsymbol{x}) = \boldsymbol{x} ^{\top} \boldsymbol{A} \boldsymbol{x}
  $$

  A quadratic form is can be written as a polynomial with terms all of second order

  $$
  \boldsymbol{x} ^{\top} \boldsymbol{A} \boldsymbol{x}  = \sum_{i, j=1}^n a_{ij} x_i x_j
  $$


Definition (Rayleigh quotient)
: - For a fixed symmetric matrix $\boldsymbol{A}$, the normalized quadratic form $\frac{\boldsymbol{x} ^{\top} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x} ^{\top} \boldsymbol{x} }$ is called a Rayleigh quotient.
  - In addition, given a positive definite matrix $\boldsymbol{B}$ of the same size, the quantity $\frac{\boldsymbol{x} ^{\top} \boldsymbol{A} \boldsymbol{x} }{\boldsymbol{x} ^{\top} \boldsymbol{B} \boldsymbol{x} }$ is called a generalized Rayleigh quotient.

Applications
: - PCA: $\max _{\boldsymbol{v} \neq 0} \frac{\boldsymbol{v}^{\top} \boldsymbol{\Sigma} \boldsymbol{v}}{\boldsymbol{v}^{\top} \boldsymbol{v}}$ where $\boldsymbol{\Sigma}$ is a covariance matrix
  - LDA: $\max _{\boldsymbol{v} \neq 0} \frac{\boldsymbol{v}^{\top} \boldsymbol{S}_{b} \boldsymbol{v}}{\boldsymbol{v}^{\top} \boldsymbol{S}_{w} \boldsymbol{v}}$ where $\boldsymbol{S} _b$ is a between-class scatter matrix, and $\boldsymbol{S} _w$ is a within-class scatter matrix
  - Spectral clustering (relaxed Ncut): $\max _{\boldsymbol{v} \neq \boldsymbol{0}} \frac{\boldsymbol{v}^{\top} \boldsymbol{L} \boldsymbol{v}}{\boldsymbol{v}^{\top} \boldsymbol{D} \boldsymbol{v}} \quad {s.t.} \boldsymbol{v} ^{\top} \boldsymbol{D} \boldsymbol{1}  = 0$ where $\boldsymbol{L}$ is graph Laplacian and $\boldsymbol{D}$ is degree matrix.

Theorem (Range of Rayleigh quotients)
: For any symmetric matrix $\boldsymbol{A} \in \mathbb{R} {n \times n}$,

  $$\begin{aligned}
  \max _{\boldsymbol{x} \in \mathbb{R}^{n}: \boldsymbol{x} \neq \boldsymbol{0}} \frac{\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{x}} &=\lambda_{\max } \\
  \min _{\boldsymbol{x} \in \mathbb{R}^{n}: \boldsymbol{x} \neq \boldsymbol{0}} \frac{\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{x}} &=\lambda_{\min }
  \end{aligned}$$

  That is, the largest and the smallest eigenvalues of $\boldsymbol{A}$ gives the range for the Rayleigh quotient. The maximum and the minimum is attainted when $\boldsymbol{x}$ is the corresponding eigenvector.

  In addition, if we add an orthogonal constraint that $\boldsymbol{x}$ is orthogonal to all the $j$ largest eigenvectors, then

  $$
  \max _{\boldsymbol{x} \in \mathbb{R}^{n}: \boldsymbol{x} \neq \boldsymbol{0}, \boldsymbol{x} \perp \boldsymbol{v} _1 \ldots, \boldsymbol{v} _j} \frac{\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{x}} =\lambda_{j+1}
  $$

  and the maximum is achieved when $\boldsymbol{x} = \boldsymbol{v} _{j+1}$.

  :::{admonition,dropdown,seealso} *Proof: Linear algebra approach*

  Consider EVD of $\boldsymbol{A}$:

  $$
  \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}=\boldsymbol{x}^{\top}\left(\boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^{\top}\right) \boldsymbol{x}=\left(\boldsymbol{x}^{\top} \boldsymbol{U}\right) \boldsymbol{\Lambda}\left(\boldsymbol{U}^{\top} \boldsymbol{x}\right)=\boldsymbol{y}^{\top} \boldsymbol{\Lambda} \boldsymbol{y}
  $$

  where $\boldsymbol{y} = \boldsymbol{U} ^{\top} \boldsymbol{x}$ is also a unit vector since $\left\| \boldsymbol{y}  \right\| ^2 = 1$. The original optimization problem becomes

  $$
  \max _{\boldsymbol{y} \in \mathbb{R}^{n}:\|\boldsymbol{y}\|=1} \quad \boldsymbol{y}^{\top} \underbrace{\boldsymbol{\Lambda}}_{\text {diagonal }} \boldsymbol{y}
  $$

  Note that the objective and constraint can be written as a weighted sum of eigenvalues

  $$
  \boldsymbol{y}^{\top} \boldsymbol{\Lambda} \boldsymbol{y}=\sum_{i=1}^{n} \underbrace{\lambda_{i}}_{\text {fixed }} y_{i}^{2} \quad \text { (subject to } y_{1}^{2}+y_{2}^{2}+\cdots+y_{n}^{2}=1)
  $$

  Let $\lambda_1 \ge \lambda_2 \ge \ldots \ge \lambda_n$, then when $y_1^2 = 1$ and $y_2^2 = \ldots = y_n ^2 = 0$, the objective function attains its maximum $\boldsymbol{y} ^{\top} \boldsymbol{\Lambda} \boldsymbol{y} = \lambda_1$. In terms of $\boldsymbol{x}$, the maximizer is

  $$
  \boldsymbol{x} ^* = \boldsymbol{U} \boldsymbol{y} ^* = \boldsymbol{U} (\pm \boldsymbol{e} _1) = \pm \boldsymbol{u}_1   
  $$

  In conclusion, when $\boldsymbol{x} = \pm \boldsymbol{u} _1$, i.e. the largest eigenvector, $\boldsymbol{x} ^{\top} \boldsymbol{A} \boldsymbol{x}$ attains its maximum value $\lambda_1$

  :::

  :::{admonition,dropdown,seealso} *Proof: Multivariable calculus approach*

  Alternatively, we can use the Method of Lagrange Multipliers to prove the theorem. First, we form the Lagrangian function

    $$
    L(\boldsymbol{x}, \lambda)=\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}-\lambda\left(\|\boldsymbol{x}\|^{2}-1\right)
    $$

    Differentiation gives

    $$
    \begin{aligned}
    \frac{\partial L}{\partial \boldsymbol{x}} &=2 \boldsymbol{A} \boldsymbol{x}-\lambda(2 \boldsymbol{x})=0 & \longrightarrow & \boldsymbol{A} \boldsymbol{x}=\lambda \boldsymbol{x} \\
    \frac{\partial L}{\partial \lambda} &=\|\boldsymbol{x}\|^{2}-1=0 & \longrightarrow &\|\boldsymbol{x}\|^{2}=1
    \end{aligned}
    $$

    This implies that $\boldsymbol{x}$ and $\lambda$ must be an eigenpair of $\boldsymbol{A}$. Moreover, for any solution $\lambda=\lambda_{i}, \boldsymbol{x}=\boldsymbol{v}_{i}$, the objective function takes the value


    $$
    \boldsymbol{v}_{i}^{\top} \boldsymbol{A} \boldsymbol{v}_{i}=\boldsymbol{v}_{i}^{\top}\left(\lambda_{i} \boldsymbol{v}_{i}\right)=\lambda_{i}\left\|\boldsymbol{v}_{i}\right\|^{2}=\lambda_{i}
    $$

    Therefore, the eigenvector $\boldsymbol{v} _1$ (corresponding to largest eigenvalue $\lambda_1$ of $\boldsymbol{A}$) is the global maximizer, and it yields the absolute maximum value $\lambda_1$.

  :::

Corollary (Generalized Rayleigh quotient problem)
: For the generalized Rayleigh quotient $\frac{\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{B} \boldsymbol{x}}$, the smallest and largest values $\lambda$ satisfy

  $$
  \boldsymbol{A v}=\lambda \boldsymbol{B v} \quad \Longleftrightarrow \quad \boldsymbol{B}^{-1} \boldsymbol{A v}=\lambda \boldsymbol{v}
  $$

  That is, the smallest/largest quotient value equals the smallest/largest eigenvalue of $(\boldsymbol{B} ^{-1} \boldsymbol{A})$. The left equation is called a generalized eigenvalue problem.

  :::{admonition,dropdown,seealso} *Proof*

  - Substitution approach

    Since $\boldsymbol{B}$ is p.d., we have $\boldsymbol{B} ^{1/2}$. Let $\boldsymbol{y} = \boldsymbol{B} ^{1/2}\boldsymbol{x}$, then the denominator can be written as

    $$
    \boldsymbol{x}^{\top} \boldsymbol{B} \boldsymbol{x}=\boldsymbol{x}^{\top} \boldsymbol{B}^{1 / 2} \boldsymbol{B}^{1 / 2} \boldsymbol{x}=\boldsymbol{y}^{\top} \boldsymbol{y}
    $$

    Substitute $\boldsymbol{x}=\left(\boldsymbol{B}^{1 / 2}\right)^{-1} \boldsymbol{y} \stackrel{\text { denote }}{=} \boldsymbol{B}^{-1 / 2} \boldsymbol{y}$ y into the numerator to rewrite it
    in terms of the new variable $\boldsymbol{y}$. This will convert the generalized Rayleigh
    quotient problem back to a regular Rayleigh quotient problem, which has
    been solved.

    $$
    \frac{\boldsymbol{y} \boldsymbol{B} ^{-1/2} \boldsymbol{A} \boldsymbol{B} ^{-1/2} \boldsymbol{y} }{\boldsymbol{y} ^{\top} \boldsymbol{y}}
    $$

  - Lagrange multipliers:

    $$
    \max _{\boldsymbol{x} \neq \boldsymbol{0}} \frac{\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}}{\boldsymbol{x}^{\top} \boldsymbol{B} \boldsymbol{x}}
    $$

    $$
    \max _{\boldsymbol{x} \in \mathbb{R}^{n}} \boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x} \quad \text { subject to } \boldsymbol{x}^{\top} \boldsymbol{B} \boldsymbol{x}=1
    $$


    $$
    L(\boldsymbol{x}, \lambda)=\boldsymbol{x}^{\top} \boldsymbol{A} \boldsymbol{x}-\lambda\left(\boldsymbol{x}^{\top} \boldsymbol{B} \boldsymbol{x}-1\right)
    $$

    Then

    $$
    \begin{aligned}
    \frac{\partial L}{\partial \boldsymbol{x}} &=2 \boldsymbol{A} \boldsymbol{x}-2\lambda \boldsymbol{B}  \boldsymbol{x}=0 & \longrightarrow & \boldsymbol{A} \boldsymbol{x}=\lambda \boldsymbol{B}\boldsymbol{x} \\
    \frac{\partial L}{\partial \lambda} &=0 & \longrightarrow & \boldsymbol{x} ^{\top} \boldsymbol{B} \boldsymbol{x} =1
    \end{aligned}
    $$

  :::


reference: [notes](https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec4RayleighQuotient.pdf)
