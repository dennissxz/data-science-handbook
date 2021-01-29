# Kernels

[Freda Shi]

## Definitions

Definition (Kernel)
: A kernel is a function $k: \mathcal{S} \times \mathcal{S} \rightarrow \mathbb{C}$ where $\mathbb{S}$ is an arbitrary set.

Definition (Positive semi-definite kernels)
: A kernel $k$ is a positive semi-definite kernel if $\forall x_{1}, \ldots, x_{n} \in \mathcal{S}$, the matrix $\boldsymbol{C}$ defined below is PSD,


$$
\mathbf{C}=\left(\begin{array}{l}
k\left(x_{1}, x_{1}\right), \ldots, k\left(x_{1}, x_{n}\right) \\
k\left(x_{2}, x_{1}\right), \ldots, k\left(x_{2}, x_{n}\right) \\
\vdots  \\
k\left(x_{n}, x_{1}\right), \ldots, k\left(x_{n}, x_{n}\right)
\end{array}\right) \in \mathbb{C}^{n \times n}
$$

Examples
: Dot product $k(\mathbf{x}, \mathbf{y})=\mathbf{x}^{*} \mathbf{y}$, where $\mathbf{x}, \mathbf{y} \in \mathbb{C}^{d}$, is a PSD kernel. For a matrix $\boldsymbol{X}$, we have $\boldsymbol{C} = \boldsymbol{X} ^* \boldsymbol{X}$, which is PSD.

: In general, $k(x, y) \triangleq \phi(x)^{T} \phi(y)$, where $\phi: \mathcal{S} \rightarrow \mathbb{R}^{d}$, is a PSD kernel,

: More generally, $k(x, y) \triangleq\langle\phi(x), \phi(y)\rangle_{\mathcal{H}}$, where $\phi: \mathcal{S} \rightarrow \mathcal{H}$, is a PSD kernel.

Why do we care about PSD kernels?

Theorem (Mercer 1909)
: Under faily general conditions, for any PSD kernel $k: \mathcal{S} \times \mathcal{S} \rightarrow \mathbb{C}$, there **exists** a map $\phi: \mathcal{S} \rightarrow \mathcal{H}$ such that $k(x, y)=\langle\phi(x), \phi(y)\rangle_{\mathcal{H}}$.


## Construction of Kernels
If $k,k_1, k_2$  are real PSD kernels on some arbitrary set $S$, the following ones are also real PSd kernels.

## Popular Kernels

## Representer Theorem

Representer Theorem (Simplified)

Consider the optimization problem on a data set $\mathcal{D} = \left\{ \boldsymbol{x}_i ,y_i \right\} _{i=1}^n, \boldsymbol{x}_i \in \mathcal{S}$

$$
\mathbf{w}^{*}=\underset{\mathbf{w}}{\arg \min } \sum_{i=1}^{n} \mathcal{L}\left(\left\langle\mathbf{w}, \phi\left(x_{i}\right)\right\rangle, y_{i}\right)+\|\mathbf{w}\|^{2}
$$

where

- $\phi: \mathcal{S} \rightarrow \mathbb{R} ^d$ (or $\mathcal{S} \rightarrow \mathcal{H}$ in general)

- $\boldsymbol{w} \in \mathbb{R} ^d$ is the coefficients in the model to be solved

- $\mathcal{L}(\cdot, \cdot): \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$ on some arbitrary loss function.

Then $\exists\left\{\alpha_{i} \in \mathbb{R}\right\}_{i=1}^{n}$ such that the solution $\boldsymbol{w} ^*$ is a linear combination of the $n$ transformed data vectors.

$$\mathbf{w}^{*}=\sum_{i=1}^{n} \alpha_{i} \phi\left(x_{i}\right)$$


We don't even need to know what $\phi$ is.

We can convert the problem of minimization over $w$ to that over $a$

Example
: In linear regression, we can write $\boldsymbol{\beta}$ as a linear combination of data vectors $\boldsymbol{x}_i$ (identical transformation $\phi$). see [here](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote14.html)


## Kernelized Models

Logic:

1. Feature transformation $\phi$ help improve model performance. But there are many choices of $\phi: S \rightarrow \mathbb{R} ^d$, hard to enumerate and handcraft an optimal one.

2. People find using kernels can improve model performance too, and more importantly kernels are related to feature transformation:

   - For every $\phi (\boldsymbol{x}_i)$ in $\mathcal{H}$, there is a kernel $k(\boldsymbol{x}_i , \boldsymbol{x}_j) = \phi(\boldsymbol{x}_i ) ^\top \phi(\boldsymbol{x}_j )$. In particular, if there is an optimal $\phi ^*$, then there is a kernel $k = \phi ^{* \top} \phi ^*$.
   - For every PSD kernel $k(\cdot, \cdot)$, there exists a $\phi(\cdot)$ (Mercer's theorem). Note that $\phi(\cdot)$ can be infinite dimensional for some $k(\cdot, \cdot)$.

3. Therefore, instead of handcrafting feature transformations $\phi$, people change to choosing (PSD) kernels.

4. The kernel methods only apply to models with loss function that involves dot product $\boldsymbol{x}_i ^\top \boldsymbol{x}_j$, for instance, SVM, PCA. If there is no dot product, there is no corresponding Kernel version of that model.
