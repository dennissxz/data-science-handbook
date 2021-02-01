# Kernels


## Definitions

Definition (Kernel)
: A kernel is a function $k: \mathcal{S} \times \mathcal{S} \rightarrow \mathbb{C}$ where $\mathbb{S}$ is an arbitrary set.

Definition (Positive semi-definite kernels)
: A kernel $k$ is a positive semi-definite kernel if $\forall x_{1}, \ldots, x_{n} \in \mathcal{S}$, the matrix $\boldsymbol{C}$ defined below is positive semi-definite,


$$
\boldsymbol{C}=\left(\begin{array}{l}
k\left(x_{1}, x_{1}\right), \ldots, k\left(x_{1}, x_{n}\right) \\
k\left(x_{2}, x_{1}\right), \ldots, k\left(x_{2}, x_{n}\right) \\
\vdots  \\
k\left(x_{n}, x_{1}\right), \ldots, k\left(x_{n}, x_{n}\right)
\end{array}\right) \in \mathbb{C}^{n \times n}
$$

Examples
: Dot product $k(\boldsymbol{x}, \boldsymbol{y})=\boldsymbol{x}^{*} \boldsymbol{y}$, where $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{C}^{d}$, is a PSD kernel, since the matrix $\boldsymbol{X}$ is PSD because $\boldsymbol{u} ^* \boldsymbol{C} \boldsymbol{u} =  \boldsymbol{u} ^* \boldsymbol{X} ^* \boldsymbol{X} \boldsymbol{u} \ge 0$.

: In general, $k(x, y) \triangleq \boldsymbol{\phi}(x) ^\top \boldsymbol{\phi}(y)$, with arbitrary map $\boldsymbol{\phi}: \mathcal{S} \rightarrow \mathbb{R}^{d}$, is a PSD kernel,

: More generally, $k(x, y) \triangleq\langle\boldsymbol{\phi}(x), \boldsymbol{\phi}(y)\rangle_{\mathcal{H}}$, with $\boldsymbol{\phi}: \mathcal{S} \rightarrow \mathcal{H}$, is a PSD kernel.

## Construction of PSD Kernels
If $k, k_1, k_2$  are real PSD kernels on some arbitrary set $\mathcal{S}$, the following ones are also real PSD kernels.

1. $\alpha k(x,y)$, where $\alpha \in \mathbb{R} _{\ge 0}$

1. $k_1(x,y) + k_2 (x,y)$
1. $k_1(x,y) k_2 (x,y)$
1. $p(k(x,y))$ where $p$ is a polinomial that has non-negative coefficients
2. $\exp (k(x,y ))$
3. $f(x)k(x,y)f(y), \forall \, f: \mathcal{S} \rightarrow \mathbb{R}$
4. $k\left(\xi\left(x^{\prime}\right), \xi\left(y^{\prime}\right)\right), \forall \xi: \mathcal{S}^{\prime} \rightarrow \mathcal{S}$

## Popular Kernels

- Linear kernel (dot product): $k(\boldsymbol{x}, \boldsymbol{y})=\boldsymbol{x}^{\top} \boldsymbol{y}$

- Polynomial kernel: $k(\boldsymbol{x}, \boldsymbol{y})=p\left(\boldsymbol{x}^{\top} \boldsymbol{y}\right)$ with non-negative coefficients

- Gaussian (RBF) kernel: $k\left(\boldsymbol{x}, \boldsymbol{y} ; \sigma^{2}\right)=\exp \left(-\frac{\|\boldsymbol{x}-\boldsymbol{y}\|^{2}}{2 \sigma^{2}}\right)$

All three are PSD kernels, and we can use them to construct PSD kernels.


## Why Using Kernels


Kernels are popular. Why? Let's first see two theorems.

### Mercer's Theorem

Theorem (Mercer's, 1909)
: Under fairly general conditions, for any PSD kernel $k: \mathcal{S} \times \mathcal{S} \rightarrow \mathbb{C}$, there **exists** a map $\boldsymbol{\phi}: \mathcal{S} \rightarrow \mathcal{H}$ such that $k(x, y)=\langle\boldsymbol{\phi}(x), \boldsymbol{\phi}(y)\rangle_{\mathcal{H}}$.


### Representer Theorem

Representer Theorem (Simplified)
: Consider the optimization problem on a data set $\mathcal{D} = \left\{ \boldsymbol{x}_i ,y_i \right\} _{i=1}^n, \boldsymbol{x}_i \in \mathcal{S}$

  $$
  \boldsymbol{w}^{*}=\underset{\boldsymbol{w}}{\arg \min } \sum_{i=1}^{n} \mathcal{L}\left(\left\langle\boldsymbol{w}, \boldsymbol{\phi} \left(\boldsymbol{x}_i \right)\right\rangle, y_{i}\right)+\|\boldsymbol{w}\|^{2}
  $$

  where

  - $\boldsymbol{\phi} : \mathcal{S} \rightarrow \mathbb{R} ^d$ (or $\mathcal{S} \rightarrow \mathcal{H}$ in general)

  - $\boldsymbol{w} \in \mathbb{R} ^d$ is the coefficients in the model to be solved

  - $\mathcal{L}(\cdot, \cdot): \mathbb{R} \times \mathbb{R} \rightarrow \mathbb{R}$ on some arbitrary loss function.

  Then there $\exists\left\{\alpha_{i} \in \mathbb{R}\right\}_{i=1}^{n}$ such that the solution $\boldsymbol{w} ^*$ is a linear combination of the $n$ transformed data vectors.

  $$\boldsymbol{w}^{*}=\sum_{i=1}^{n} \alpha_{i} \boldsymbol{\phi} \left(\boldsymbol{x}_i \right) = \boldsymbol{X} ^\top \boldsymbol{\alpha}$$


**Example (Linear regression OLS)**
: In linear regression, we can write $\boldsymbol{\beta}$ as a linear combination of data vectors $\boldsymbol{x}_i$ (identical transformation $\boldsymbol{\phi}$).


$$\begin{aligned}
&& \boldsymbol{X} ^\top \boldsymbol{\alpha} &= (\boldsymbol{X} ^\top \boldsymbol{X} )^{-1} \boldsymbol{X} ^\top \boldsymbol{y} \\
&\Rightarrow& (\boldsymbol{X} \boldsymbol{X}  ^\top )(\boldsymbol{X} \boldsymbol{X}  ^\top) \boldsymbol{\alpha} &=\boldsymbol{X} \left( \boldsymbol{X} ^\top \boldsymbol{X}  (\boldsymbol{X} ^\top \boldsymbol{X} )^{-1}  \right)\boldsymbol{X} ^\top \boldsymbol{y} \\
&\Rightarrow&  \boldsymbol{K}^2 \boldsymbol{\alpha} &= \boldsymbol{K} \boldsymbol{y}  \quad \text{let } \boldsymbol{K} = \boldsymbol{X} \boldsymbol{X} ^\top  \\
&\Rightarrow&  \boldsymbol{\alpha} &= \boldsymbol{K} ^{-1} \boldsymbol{y}  \\
\end{aligned}$$

**Implication**

Substituting $\boldsymbol{w} ^*$ back into the loss function, we have, for the loss of a data point $(\boldsymbol{x}, y)$,

$$\begin{aligned} \mathcal{L}\left(\left\langle\boldsymbol{w}^{*}, \boldsymbol{\phi}(\boldsymbol{x})\right\rangle, y\right) &=\mathcal{L}\left(\left\langle\sum_{i=1}^{n} \alpha_{i} \boldsymbol{\phi}\left(\boldsymbol{x}_{i}\right), \boldsymbol{\phi}(\boldsymbol{x})\right\rangle, y\right) \\ &=\mathcal{L}\left(\sum_{i=1}^{n} \alpha_{i}\left\langle\boldsymbol{\phi}\left(\boldsymbol{x}_{i}\right), \boldsymbol{\phi}(\boldsymbol{x})\right\rangle, y\right) \\ &=\mathcal{L}\left(\sum_{i=1}^{n} \alpha_{i} k\left(\boldsymbol{x}_{i}, \boldsymbol{x}\right), y\right) \end{aligned}$$

The last line implies that

- We don't even need to know what $\boldsymbol{\phi}$ is exactly. We just need to care about kernels and there is a corresponding $\boldsymbol{\phi}$ under the wood.

- We can convert the problem of minimization over $\boldsymbol{w}$ to that over $\boldsymbol{\alpha}$.

### Logic

Now we can summarize the logic of using kernels.

1. People find that feature transformation $\boldsymbol{\phi}$ help improve model performance. But there are many choices of $\boldsymbol{\phi}: S \rightarrow \mathbb{R} ^d$, hard to enumerate and handcraft an optimal one.

2. People find using kernels can improve model performance too, and more importantly kernels are related to feature transformation:

   - For every $\boldsymbol{\phi} (\boldsymbol{x}_i)$ in $\mathcal{H}$, there is a corresponding kernel $k(\boldsymbol{x}_i , \boldsymbol{x}_j) = \boldsymbol{\phi}(\boldsymbol{x}_i ) ^\top \boldsymbol{\phi}(\boldsymbol{x}_j )$. In particular, if there is an optimal $\boldsymbol{\phi} ^*$, then there is a kernel $k = \boldsymbol{\phi} ^{* \top} \boldsymbol{\phi} ^*$.
   - For every PSD kernel $k(\cdot, \cdot)$, there exists a feature transformatin $\boldsymbol{\phi}(\cdot)$ by Mercer's Theorem. Note that $\boldsymbol{\phi}(\cdot)$ can be infinite dimensional for some $k(\cdot, \cdot)$.

3. Therefore, instead of handcrafting feature transformations $\boldsymbol{\phi}$, people change to choosing (PSD) kernels to improve model performance.

:::{admonition,note} Dot product in loss function
Note that the kernel methods only apply to models with loss function that involves dot product $\boldsymbol{x}_i ^\top \boldsymbol{x}_j$, for instance, SVM, PCA. If there is no dot product, there is no corresponding kernelized version of that model, such as linear regression.
:::
