# Kernel CCA

Like Kernel PCA, we can generalize CCA with kernels.

## Kernelization

Recall the original optimization problem of CCA

$$\begin{align}
\max _{\boldsymbol{v}, \boldsymbol{w} } \operatorname{Corr}\left(\boldsymbol{v}^{\top} \boldsymbol{x} , \boldsymbol{w}^{\top} \boldsymbol{y} \right)
= \max &  _{\boldsymbol{v}, \boldsymbol{w}}  \frac{\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xy} \boldsymbol{w}}{\sqrt{\boldsymbol{v}^{\top} \boldsymbol{\Sigma}_{xx} \boldsymbol{v} \boldsymbol{w}^{\top} \boldsymbol{\Sigma}_{yy} \boldsymbol{w}}} \\
\text{s.t.}  &  \ \quad \boldsymbol{v} ^\top \boldsymbol{\Sigma} _{xx} \boldsymbol{v} =1 \\
  &  \ \quad \boldsymbol{w} ^\top \boldsymbol{\Sigma} _{yy} \boldsymbol{w} =1 \\
\end{align}$$

Consider two feature transformations, $\boldsymbol{\phi} _x: \mathbb{R} ^{d_x} \rightarrow \mathbb{R} ^{p_x}$ and $\boldsymbol{\phi} _x: \mathbb{R} ^{d_y} \rightarrow \mathbb{R} ^{p_y}$. Let

- $\boldsymbol{\Phi} _x$ and $\boldsymbol{\Phi} _y$ be the transformed data matrix.
- $\boldsymbol{C}_x = \frac{1}{n} \boldsymbol{\Phi} _x ^\top \boldsymbol{\Phi} _x, \boldsymbol{C}_y = \frac{1}{n} \boldsymbol{\Phi} _y ^\top \boldsymbol{\Phi} _y, \boldsymbol{C}_{xy} = \frac{1}{n} \boldsymbol{\Phi} _x ^\top \boldsymbol{\Phi} _y, \boldsymbol{C}_{yx} = \frac{1}{n} \boldsymbol{\Phi} _y ^\top \boldsymbol{\Phi} _x$ be the corresponding variance-covariance matrices.

Then the problem is to solve $\boldsymbol{v}, \boldsymbol{w}$ in

$$\begin{align}
\max _{\boldsymbol{v} , \boldsymbol{w}} \operatorname{Corr}\left(\boldsymbol{v}^{\top} \boldsymbol{\phi}_x ( \boldsymbol{x}), \boldsymbol{w}^{\top} \boldsymbol{\phi}_y (\boldsymbol{y})\right)= \max &  _{\boldsymbol{v}, \boldsymbol{w}}  \frac{\boldsymbol{v}^{\top} \boldsymbol{C}_{xy} \boldsymbol{w}}{\sqrt{\boldsymbol{v}^{\top} \boldsymbol{C}_{xx} \boldsymbol{v} \boldsymbol{w}^{\top} \boldsymbol{C}_{yy} \boldsymbol{w}}} \\
\text{s.t.}  &  \ \quad \boldsymbol{v} ^\top \boldsymbol{C} _{xx} \boldsymbol{v} =1 \\
  &  \ \quad \boldsymbol{w} ^\top \boldsymbol{C} _{yy} \boldsymbol{w} =1 \\
\end{align}$$


```{margin} How?

$$
\boldsymbol{C} ^{-1}  _x \boldsymbol{C} _{xy} \boldsymbol{C} _y ^{-1} \boldsymbol{C} _{yx} \boldsymbol{v}  = \lambda \boldsymbol{v}  \\
\frac{1}{n} \boldsymbol{\Phi} _x ^\top \boldsymbol{\Phi} _y \boldsymbol{C} _y ^{-1}
\frac{1}{n} \boldsymbol{\Phi} _y ^\top \boldsymbol{\Phi} _x  \boldsymbol{v}  = \lambda \boldsymbol{C} _x \boldsymbol{v}  \\
\boldsymbol{\Phi} _x  ^\top \boldsymbol{u}  = \lambda \boldsymbol{C} _x\boldsymbol{v}
$$

```

It can be shown that, the solution vectors $\boldsymbol{v}$ and $\boldsymbol{w}$ are linear combinations of the data vectors, i.e.

$$\begin{aligned}
\boldsymbol{v} &= \boldsymbol{\Phi} _x ^\top \boldsymbol{\alpha}\\
\boldsymbol{w} &= \boldsymbol{\Phi} _y ^\top \boldsymbol{\beta} \\
\end{aligned}$$

Substituting them back to the objective function, we have


$$\begin{aligned}
\rho&=\max _{\boldsymbol{\boldsymbol{\alpha}} , \boldsymbol{\boldsymbol{\beta}} } \frac{\boldsymbol{\alpha}^{\top} \boldsymbol{\Phi}_x \boldsymbol{\Phi}_x^{\top} \boldsymbol{\Phi}_y \boldsymbol{\Phi}_y^{\top} \boldsymbol{\beta}}{\sqrt{\boldsymbol{\alpha}^{\top} \boldsymbol{\Phi}_x \boldsymbol{\Phi}_x^{\top} \boldsymbol{\Phi}_x \boldsymbol{\Phi}_x^{\top} \boldsymbol{\alpha} \cdot \boldsymbol{\beta}^{\top} \boldsymbol{\Phi}_y \boldsymbol{\Phi}_y^{\top} \boldsymbol{\Phi}_y \boldsymbol{\Phi}_y^{\top} \boldsymbol{\beta}}} \\
&=\max _{\boldsymbol{\boldsymbol{\alpha}} , \boldsymbol{\boldsymbol{\beta}} } \frac{\boldsymbol{\alpha}^{\top} \boldsymbol{K}_x \boldsymbol{K}_y   \boldsymbol{\beta}}{\sqrt{\boldsymbol{\alpha}^\top \boldsymbol{K}_x^2  \boldsymbol{\alpha} \cdot \boldsymbol{\beta}^{\top} \boldsymbol{K}_y^2 \boldsymbol{\beta}}} \\
\end{aligned}$$

which is the **dual form**.

Note that the value if not affected by re-scaling of $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$ either together or independently. Hence the KCCA optimization problem is equivalent to

$$\begin{align}
\max  _{\boldsymbol{\alpha}, \boldsymbol{\beta}} &&  \boldsymbol{\alpha} ^\top \boldsymbol{K}_x \boldsymbol{K} _y \boldsymbol{\beta} \\
\text{s.t.} &&  \boldsymbol{\alpha} ^\top \boldsymbol{K} _x ^2 \boldsymbol{\alpha} = 1\\
&& \boldsymbol{\beta} ^\top \boldsymbol{K} _y ^2 \boldsymbol{\beta} =1\\
\end{align}$$


The corresponding Lagrangian is


$$
\mathcal{L}(\lambda, \boldsymbol{\alpha}, \boldsymbol{\beta})=\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta}-\frac{\lambda_{\boldsymbol{\alpha}}}{2}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}-1\right)-\frac{\lambda_{\boldsymbol{\beta}}}{2}\left(\boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta}-1\right)
$$

Taking derivatives w.r.t. $\boldsymbol{\boldsymbol{\alpha}}$ and $\boldsymbol{\boldsymbol{\beta}}$ gives

$$
\begin{array}{l}
\frac{\partial f}{\partial \boldsymbol{\alpha}}=\boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta}-\lambda_{\boldsymbol{\alpha}} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}=\mathbf{0} \\
\frac{\partial f}{\partial \boldsymbol{\beta}}=\boldsymbol{K}_{y} \boldsymbol{K}_{x} \boldsymbol{\alpha}-\lambda_{\boldsymbol{\beta}} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta}=\mathbf{0}
\end{array}
$$

Subtracting $\boldsymbol{\boldsymbol{\beta}} ^\top$ times the second equation from $\boldsymbol{\boldsymbol{\alpha}} ^\top$ times the first we have

$$
\begin{aligned}
0 &=\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta}-\boldsymbol{\alpha}^{\prime} \lambda_{\boldsymbol{\alpha}} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}-\boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y} \boldsymbol{K}_{x} \boldsymbol{\alpha}+\boldsymbol{\beta}^{\prime} \lambda_{\boldsymbol{\beta}} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta} \\
&=\lambda_{\boldsymbol{\beta}} \boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta}-\lambda_{\boldsymbol{\alpha}} \boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}
\end{aligned}
$$

which together with the constraints implies $\lambda_\alpha - \lambda_\beta =0$. Let them be $\lambda$. Suppose $\boldsymbol{\boldsymbol{K}} _x$ and $\boldsymbol{\boldsymbol{K}} _y$ are invertible (usually the case), then the system of equations given by the derivatives yields

$$
\begin{aligned}
\boldsymbol{\beta} &=\frac{\boldsymbol{K}_{y}^{-1} \boldsymbol{K}_{y}^{-1} \boldsymbol{K}_{y} \boldsymbol{K}_{x} \boldsymbol{\alpha}}{\lambda} \\
&=\frac{\boldsymbol{K}_{y}^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha}}{\lambda}
\end{aligned}
$$

and hence

$$
\boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{K}_{v}^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha}-\lambda^{2} \boldsymbol{K}_{x} \boldsymbol{K}_{x} \boldsymbol{\alpha}=0
$$

or

$$
I \boldsymbol{\alpha}=\lambda^{2} \boldsymbol{\alpha}
$$

Therefore, $\boldsymbol{\alpha}$ can be any unit vector $\boldsymbol{e} _j$, and we can find the corresponding $\boldsymbol{\beta}$ be the $j$-th column of $\boldsymbol{K}_{y}^{-1} \boldsymbol{K}_{x}$. Substituting back to the objective function, we found $\rho = 1$. The corresponding weights are $\boldsymbol{v} = \boldsymbol{\Phi} _x ^\top \boldsymbol{\alpha} = \boldsymbol{\phi} (\boldsymbol{x} _j)$, and $\boldsymbol{w} = \boldsymbol{\Phi} _y ^\top \boldsymbol{\beta}$.

This is a trivial solution. It is therefore clear that a naive application of CCA in kernel defined feature space will not provide useful results. Regularization is necessary.

## Regularization

To obtain non-trivial solution, we add regularization term, which is typically the norm of the weights $\boldsymbol{v}$ and $\boldsymbol{w}$.

$$
\begin{aligned}
\rho &=\max _{\boldsymbol{\alpha}, \boldsymbol{\beta}} \frac{\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta}}{\sqrt{\left.\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}+r\left\|\boldsymbol{v} \right\|^{2}\right) \cdot\left(\boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta}+r\left\|\boldsymbol{w} \right\|^{2}\right)\right)}} \\
&=\max _{\boldsymbol{\alpha}, \boldsymbol{\beta}} \frac{\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta}}{\sqrt{\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}+r \boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{\alpha}\right) \cdot\left(\boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta}+r \boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y} \boldsymbol{\beta}\right)}}
\end{aligned}
$$

Likewise, we observe that the new regularized equation is not affected by re-scaling of $\boldsymbol{\alpha}$ or $\boldsymbol{\beta}$, hence the optimization problem is subject to

$$
\begin{aligned}
\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}+r \boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{\alpha}\right) &=1 \\
\left(\boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta}+r \boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y} \boldsymbol{\beta}\right) &=1
\end{aligned}
$$

The resulting Lagrangian is

$$
\begin{aligned}
\mathcal{L}\left(\lambda_{\boldsymbol{\alpha}}, \lambda_{\boldsymbol{\beta}}, \boldsymbol{\alpha}, \boldsymbol{\beta}\right)=& \boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta} \\
&-\frac{\lambda_{\boldsymbol{\alpha}}}{2}\left(\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}+r \boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{\alpha}-1\right) \\
&-\frac{\lambda_{\boldsymbol{\beta}}}{2}\left(\boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y}^{2} \boldsymbol{\beta}+r \boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y} \boldsymbol{\beta}-1\right)
\end{aligned}
$$

The derivatives are

$$
\begin{aligned}
&\frac{\partial f}{\partial \boldsymbol{\alpha}}=\boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta}-\lambda_{\boldsymbol{\alpha}}\left(\boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}+r \boldsymbol{K}_{x} \boldsymbol{\alpha}\right)\\
&\frac{\partial f}{\partial \boldsymbol{\beta}}=\boldsymbol{K}_{y} \boldsymbol{K}_{x} \boldsymbol{\alpha}-\lambda_{\boldsymbol{\beta}}\left(\boldsymbol{K}_{y}^{2} \boldsymbol{\beta}+r \boldsymbol{K}_{y} \boldsymbol{\beta}\right)
\end{aligned}
$$

By the same trick, we have

$$
\begin{aligned}
0 &=\boldsymbol{\alpha}^{\prime} \boldsymbol{K}_{x} \boldsymbol{K}_{y} \boldsymbol{\beta}-\lambda_{\boldsymbol{\alpha}} \boldsymbol{\alpha}^{\prime}\left(\boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}+r \boldsymbol{K}_{x} \boldsymbol{\alpha}\right)-\boldsymbol{\beta}^{\prime} \boldsymbol{K}_{y} \boldsymbol{K}_{x} \boldsymbol{\alpha}+\lambda_{\boldsymbol{\beta}} \boldsymbol{\beta}^{\prime}\left(\boldsymbol{K}_{y}^{2} \boldsymbol{\beta}+r \boldsymbol{K}_{y} \boldsymbol{\beta}\right) \\
&=\lambda_{\boldsymbol{\beta}} \boldsymbol{\beta}^{\prime}\left(\boldsymbol{K}_{y}^{2} \boldsymbol{\beta}+r \boldsymbol{K}_{y} \boldsymbol{\beta}\right)-\lambda_{\boldsymbol{\alpha}} \boldsymbol{\alpha}^{\prime}\left(\boldsymbol{K}_{x}^{2} \boldsymbol{\alpha}+r \boldsymbol{K}_{x} \boldsymbol{\alpha}\right)
\end{aligned}
$$

which gives $\lambda_\alpha = \lambda_\beta =0$. Let them be $\lambda$, and suppose $\boldsymbol{K}_x$ and $\boldsymbol{K}_y$ are invertible, we have

$$
\begin{aligned}
\boldsymbol{\beta} &=\frac{\left(\boldsymbol{K}_{y}+r I\right)^{-1} \boldsymbol{K}_{y}^{-1} \boldsymbol{K}_{y} \boldsymbol{K}_{x} \boldsymbol{\alpha}}{\lambda} \\
&=\frac{\left(\boldsymbol{K}_{y}+r I\right)^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha}}{\lambda}
\end{aligned}
$$

and hence

$$
\begin{aligned}
\boldsymbol{K}_{x} \boldsymbol{K}_{y}\left(\boldsymbol{K}_{y}+r I\right)^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha} &=\lambda^{2} \boldsymbol{K}_{x}\left(\boldsymbol{K}_{x}+r I\right) \boldsymbol{\alpha} \\
\boldsymbol{K}_{y}\left(\boldsymbol{K}_{y}+r I\right)^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha} &=\lambda^{2}\left(\boldsymbol{K}_{x}+r I\right) \boldsymbol{\alpha} \\
\left(\boldsymbol{K}_{x}+r I\right)^{-1} \boldsymbol{K}_{y}\left(\boldsymbol{K}_{y}+r I\right)^{-1} \boldsymbol{K}_{x} \boldsymbol{\alpha} &=\lambda^{2} \boldsymbol{\alpha}
\end{aligned}
$$

Therefore, we just solve the above eigenproblem to get meaningful $\boldsymbol{\alpha}$, and then compute $\boldsymbol{\beta}$.

## Learning

From the analysis above, the steps to train a kernel PCA are

1. Choose a kernel function $k(\cdot, \cdot)$.
2. Compute the centered kernel matrix $\boldsymbol{K} ^\prime _x = (\boldsymbol{I} - \boldsymbol{u} \boldsymbol{u} ^\top )\boldsymbol{K}_x(\boldsymbol{I} - \boldsymbol{u} \boldsymbol{u} ^\top)$ and $\boldsymbol{K} _y ^\prime$
3. Find the first $k$ eigenvectors of $\left(\boldsymbol{K} ^\prime_{x}+r I\right)^{-1} \boldsymbol{K} ^\prime_{y}\left(\boldsymbol{K} ^\prime_{y}+r I\right)^{-1} \boldsymbol{K} ^\prime_{x}$, store in $\boldsymbol{A}$.

Then

- to project a new data vector $\boldsymbol{x}$, note that

    $$
    \boldsymbol{w} _ {x,j} = \boldsymbol{\Phi} _x ^\top  \boldsymbol{\alpha}_{j}
    $$

    Hence, the embeddings $\boldsymbol{z}$ of a vector $\boldsymbol{x}$ is

    $$
    \boldsymbol{z} _x
    = \left[\begin{array}{c}
    \boldsymbol{w} _{x, 1} ^\top \boldsymbol{\phi}(\boldsymbol{x}) \\
    \boldsymbol{w} _{x, 2} ^\top \boldsymbol{\phi}(\boldsymbol{x}) \\
    \vdots \\
    \boldsymbol{w} _{x, k} ^\top \boldsymbol{\phi}(\boldsymbol{x}) \\
    \end{array}\right]
    = \left[\begin{array}{c}
    \boldsymbol{\alpha} _1 ^\top  \boldsymbol{\Phi}_x  \\
    \boldsymbol{\alpha} _2 ^\top  \boldsymbol{\Phi}_x  \\
    \vdots \\
    \boldsymbol{\alpha} _k ^\top  \boldsymbol{\Phi}_x  \\
    \end{array}\right] \boldsymbol{\phi}(\boldsymbol{x})
    = \boldsymbol{A} ^\top \boldsymbol{\Phi}_x \boldsymbol{\phi}(\boldsymbol{x})  
    = \boldsymbol{A} ^\top \left[\begin{array}{c}
    k(\boldsymbol{x} _1, \boldsymbol{x}) \\
    k(\boldsymbol{x} _2, \boldsymbol{x}) \\
    \vdots \\
    k(\boldsymbol{x} _n, \boldsymbol{x}) \\
    \end{array}\right]\\
    $$

    where $\boldsymbol{A} _{n\times k}$ are the first $k$ eigenvectors $\boldsymbol{\alpha}$.


- to project the training data $\boldsymbol{X}$,

    $$\boldsymbol{Z} ^\top  = \boldsymbol{A} ^\top \boldsymbol{K}_x \text{ or } \boldsymbol{Z} = \boldsymbol{K}_x \boldsymbol{A} $$


- to project a new data matrix $\boldsymbol{X} ^\prime _{m \times d}$,

    $$
    \boldsymbol{Z} _{x ^\prime} ^\top = \boldsymbol{A} ^\top \left[\begin{array}{ccc}
    k(\boldsymbol{x} _1, \boldsymbol{x ^\prime_1}) & \ldots  &k(\boldsymbol{x} _1, \boldsymbol{x ^\prime_m}) \\
    k(\boldsymbol{x} _2, \boldsymbol{x ^\prime_1}) & \ldots  &k(\boldsymbol{x} _2, \boldsymbol{x ^\prime_m}) \\
    \vdots &&\vdots \\
    k(\boldsymbol{x} _n, \boldsymbol{x ^\prime_1}) & \ldots  &k(\boldsymbol{x} _n, \boldsymbol{x ^\prime_m}) \\
    \end{array}\right]\\
    $$

## Model Selection

Hyperparameters/settings include number of components $k$, choice of kernel, and the regularization coefficient $r$.
