# Kernel PCA

Kernel PCA is a nonlinear extension of PCA where dot products are replaced with generalized dot products in feature space computed by a kernel function.

## Objective

Linear PCA only consider linear subspaces

$$
\hat{\boldsymbol{x}} = \boldsymbol{W} \boldsymbol{z}
$$

To solve this, we can apply feature transformation $\boldsymbol{\phi}: \mathbb{R} ^d \rightarrow \mathbb{R} ^p$ to include non-linear features, such as $\boldsymbol{\phi} ([x_1, x_2])= [x_1, x_1^2, x_1 x_2]$.

Then the data matrix changes from $\boldsymbol{X}_{n \times d}$ to $\boldsymbol{\Phi}_{n \times p}$. And the inner product matrix changes from $\boldsymbol{X} \boldsymbol{X} ^\top$ to $\boldsymbol{\Phi} \boldsymbol{\Phi} ^\top$.

, handcrafting feature transformation $\boldsymbol{\phi}$ is equivalent to choosing a kernel function $k(\boldsymbol{x}_1, \boldsymbol{x} _2) = \boldsymbol{\phi}(\boldsymbol{x} _1) ^\top \boldsymbol{\phi}(\boldsymbol{x} _2)$. We can compute the new inner product matrix by $\boldsymbol{K} = \boldsymbol{\Phi} \boldsymbol{\Phi} ^\top$.

Then we can run MDS on $\boldsymbol{K}$.



:::{admonition,note} Mean normalization

We've implicitly assumed zero-mean inputs. If the inputs $\boldsymbol{K}$ are not zero-mean, we center it by

$$
\boldsymbol{K} ^\prime = (\boldsymbol{I} - \boldsymbol{u} \boldsymbol{u} ^\top )\boldsymbol{K}(\boldsymbol{I} - \boldsymbol{u} \boldsymbol{u} ^\top)  
$$

where $\boldsymbol{u} = \frac{1}{\sqrt{n}}[1 \ldots 1]^{\top}$

:::

## Choice of Kernel

As [discussed](kernels-logic), in practice, we don't engineer $\phi(\cdot)$, but directly think about $k(\cdot, \cdot)$.

- One common choice of kernel is radial basis function (RBF), aka Gaussian kernel

    $$
    k\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right)=e^{\frac{-\left(\mathbf{x}_{1}-\mathbf{x}_{2}\right)^{2}}{2 \sigma^{2}}}
    $$

    where the standard deviation (radius) $\sigma$ is a tuning parameter.

    RBF corresponds to an implicit feature space of infinite dimensionality.

:::{figure} kernels-RBF
<img src="../imgs/kernels-RBF.png" width = "50%" alt=""/>

Gaussian kernels
:::


- Polynomial kernel

    $$
    k\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right)=\left(1+\mathbf{x}_{1}^{T} \mathbf{x}_{2}\right)^{p}
    $$

    where the polynomial degree $p$ is a tuning parameter. $p=2$ is common.

## Pros and Cos

**Pros**

Kernel PCA works well

- if the data is non-linear and fit the chosen kernel.

    :::{figure} kernel-pca-ep1
    <img src="../imgs/kernel-pca-ep1.png" width = "80%" alt=""/>

    Kernel PCA with a RBF kernel on points [Livescu 2021].

    :::

- if there is much noise in the data

    :::{figure}
    <img src="../imgs/kernel-pca-ep2.png" width = "60%" alt=""/>

    Kernel PCA on images [Livescu 2021].

    :::

**Cons**

- Kernel PCA works bad when the data lies a special manifold

    :::{figure}
    <img src="../imgs/kernel-pca-ep3.png" width = "50%" alt=""/>

    Kernel PCA with RBF kernel on a Swiss roll manifold [Livescu 2021]

    :::

- Computationally expensive to compute $n \times n$ pairwise kernel values when $n$ is large. Remedies include

  - use subset of the entire data set

  - use kernel approximation techniques

    - approximate $\boldsymbol{K} \approx \boldsymbol{F}^\top \boldsymbol{F}$ where $\boldsymbol{F} \in \mathbb{R} ^{m \times n}, k \ll m \ll n$. The value of $m$ should be as large as you can handle.
  
    - For RBF kernels, there is one remarkable good approximation (due to Fourier transform properties) called random Fourier features (Rahimi & Recht 2008), which replaces each data point $\boldsymbol{x}_i$ with

      $$
      \left[\cos \left(\boldsymbol{w}_{1}^{T} \boldsymbol{x}_{i}+b_{1}\right) \ldots \cos \left(\boldsymbol{w}_{m}^{T} \boldsymbol{x}_{i}+b_{m}\right)\right]^{T}=\boldsymbol{f} _{i}
      $$

      \boldsymbol{w}here

      $$
      \begin{aligned}
      b_{1}, \ldots, b_{m} & \sim \operatorname{Unif}[0,2 \pi] \\
      \boldsymbol{w}_{1}, \ldots, \boldsymbol{w}_{m} & \sim \mathcal{N}\left(0, \frac{2}{\sigma^{2}} \boldsymbol{I}_d \right)
      \end{aligned}
      $$

  - don't use kernel methods.

## Relation to Graph-based Spectral Methods

Both are motivates as extensions of MDS and involves a $n \times n$ matrix.

We can view kernel in kernel PCA as the edge weights in [graph-based spectral methods](23-graph-based-spectral-methods). But the main difference is that in kernel PCA we compute the kernel value of **every pair** of data points (computationally demanding), but in graph-based spectral methods we only compute weights between points that are neighbors.

## Projection of New Observations

To find a way to project observations, we first introduce an alternative formulation of the kernel PCA problem.

The covariance matrix in the feature space $\frac{1}{n} \boldsymbol{\Phi} ^\top \boldsymbol{\Phi}$ can be written as a sum of outer products of features over data points

$$
\boldsymbol{C} = \frac{1}{n} \sum_{i=1}^n \boldsymbol{\phi}(\boldsymbol{x}_i )  \boldsymbol{\phi}(\boldsymbol{x}_i ) ^\top
$$

To solve the kernel PCA, we find the eigenvectors of $\boldsymbol{C}$. The $j$-th PCA projection vector $\boldsymbol{w}$ is an eigenvector of $\boldsymbol{C}$.

$$
\boldsymbol{C} \boldsymbol{w}_j  = \lambda_j \boldsymbol{w}_j
$$

It can be shown that

$$
\boldsymbol{w}_j = \frac{1}{\lambda n} \sum_{i=1}^n \left( \boldsymbol{\phi}(\boldsymbol{x}_i ) ^\top  \boldsymbol{w}_j  \right) \boldsymbol{\phi}(\boldsymbol{x}_i ) = \sum_{i=1}^n \alpha_{ji} \boldsymbol{\phi}(\boldsymbol{x}_i)
$$

which implies that $\boldsymbol{w}$ lies in the column span of $\boldsymbol{\Phi} ^\top$.

Substituting the above expression back to $\boldsymbol{C} \boldsymbol{w}  = \lambda_j \boldsymbol{w}$ gives

$$\begin{aligned}
LHS
&= \boldsymbol{C} \boldsymbol{w} \\
&= \frac{1}{n} \sum_{i=1}^n \boldsymbol{\phi}(\boldsymbol{x}_i )  \boldsymbol{\phi}(\boldsymbol{x}_i ) ^\top \boldsymbol{w} \\
&= \frac{1}{n} \sum_{i=1}^n \boldsymbol{\phi}(\boldsymbol{x}_i )  \boldsymbol{\phi}(\boldsymbol{x}_i ) ^\top \left( \sum_{\ell=1}^n \alpha_{j\ell} \boldsymbol{\phi}(\boldsymbol{x}_i)  \right) \\
&= \frac{1}{n} \sum_{i=1}^n \boldsymbol{\phi}(\boldsymbol{x}_i ) \left( \sum_{\ell=1}^n \alpha_{j\ell} k(\boldsymbol{x}_{i} , \boldsymbol{x} _\ell)  \right) \\
RHS
&= \lambda_j \sum_{i=1}^n \alpha_{ji} \boldsymbol{\phi}(\boldsymbol{x}_i)
\end{aligned}$$


```{margin}
For kernelized problems, the original problem formulation (primal) is in the original feature space, and the kernelized problem formulation (dual) is in the transformed space.
```

and finally

$$
\boldsymbol{K} \boldsymbol{\alpha} _j= n \lambda_j \boldsymbol{\alpha} _j
$$

where $\boldsymbol{K} = \boldsymbol{\Phi}  \boldsymbol{\Phi}  ^\top$. So now our task changes to solve an eigenproblem of the above equation. The normalization constraint $\boldsymbol{w} _j ^\top \boldsymbol{w} _j = 1$ in the original problem $\boldsymbol{C} \boldsymbol{w}_j = \lambda_j \boldsymbol{w}$ becomes

$$\begin{aligned}
&& \boldsymbol{w} _j ^\top \boldsymbol{w} _j &= 1 \\
&\Rightarrow& \sum_{k=1}^n \sum_{\ell=1}^n \alpha_{j\ell} \alpha_{jk} \boldsymbol{\phi}(\boldsymbol{x} _{\ell})^\top \boldsymbol{\phi}(\boldsymbol{x} _k)  &=1 \\
&\Rightarrow& \boldsymbol{\alpha} _j ^\top \boldsymbol{K} \boldsymbol{\alpha} _j &= 1 \\
\end{aligned}$$


To project a new data point $\boldsymbol{x}$, we project its feature vector onto each eigenvector (like $\boldsymbol{z} = \boldsymbol{U} ^\top \boldsymbol{x}$ in standard PCA) using the kernel value with each of the $n$ data points

$$
z_j = \boldsymbol{w}_j ^\top  \boldsymbol{\phi}(\boldsymbol{x} ) = \sum_{i=1}^n \alpha_{ji} \boldsymbol{\phi}(\boldsymbol{x}_i) ^\top \boldsymbol{\phi} (\boldsymbol{x} ) = \sum_{i=1}^n \alpha_{ji} k(\boldsymbol{x}_i ,\boldsymbol{x})
$$

which allows the kernel version of the formulation (dual) and then project a new observation $\boldsymbol{x}$ not in the training set, without writing out the feature vector $\boldsymbol{\phi}(\boldsymbol{x})$.
