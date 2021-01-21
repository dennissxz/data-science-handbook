# Support Vector Machine

We first introduce the basic linear separable case (aka. hard margin), and introduce the linear non-separable case later (aka. soft margin).

## Prerequisite

### Distance from a Point to a Hyperplane

Suppose there is a hyperplane in $p$-dimensional space characterized by

$$
w_{1} x_{1}+w_{2} x_{2}+\cdots+w_{p} x_{p} + b = 0
$$

or in vector form,

$$
\boldsymbol{w} ^\top  \boldsymbol{x}  + b = 0
$$

then the distance from a point $X$ with coordinates $\boldsymbol{x} = (x_1, x_2, \ldots, x_p)$ to this hyperplane is

$$
\frac{1}{\left\Vert \boldsymbol{w}  \right\Vert } \left\vert \boldsymbol{w} ^\top  \boldsymbol{x}  + b  \right\vert
$$

Note that the distance is always positive.

```{dropdown} Derivation
For any two points $Y,Z$ with coordinates $\boldsymbol{y}, \boldsymbol{z}$ on the hyperplane we have


$$\begin{align}
\boldsymbol{w} ^\top \boldsymbol{y} + b &= 0 \\
\boldsymbol{w} ^\top \boldsymbol{z} + b &= 0
\end{align}$$

Hence,

$$
\boldsymbol{w} ^\top (\boldsymbol{y} - \boldsymbol{z})= 0
$$

which implies that the vector $\boldsymbol{w}$ is orthogonal to the hyperplane.

The distance from point $X$ to the hyperplane can be formulated as

$$
d = \left\vert \left( \frac{\boldsymbol{w}}{\left\Vert \boldsymbol{w}  \right\Vert } \right)  ^\top (\boldsymbol{x} - \boldsymbol{y}) \right\vert
$$

where $\frac{\boldsymbol{w}}{\left\Vert \boldsymbol{w}  \right\Vert }$ is a unit vector orthogonal to the hyperplane and $\boldsymbol{x} - \boldsymbol{y}$ is a vector pointing from point $Y$ (on the hyperplane) to point $X$. The absolute value of the cross product is the of the projection of vector $\boldsymbol{x} - \boldsymbol{y}$ onto the direction of $\boldsymbol{w}$, i.e., $d$.

Substituting $\boldsymbol{w} ^\top \boldsymbol{y} + b = 0$ gives

$$
d = \frac{1}{\left\Vert \boldsymbol{w}  \right\Vert}\left\vert \boldsymbol{w} ^\top \boldsymbol{x} + \boldsymbol{b} \right\vert
$$
```

Note that the points on the same side of the hyperplane have the same sign of $\boldsymbol{w} ^\top  \boldsymbol{x}_i  + b$. If we label the points with positive values of $\boldsymbol{w} ^\top  \boldsymbol{x}_i  + b$ by $y_i = 1$ and those with negative values by $y_i = -1$, then the distance can be written as


$$
\frac{1}{\left\Vert \boldsymbol{w}  \right\Vert } y_i (\boldsymbol{w} ^\top  \boldsymbol{x}  + b )
$$


## Objective


:::{figure,myclass} markdown-fig
<img src="../imgs/svm-hard-margin.png" width = "50%" alt=""/>

caption
:::

Definition (Margin)
: The margin is defined as the shortest distance from a point to the hyperplane.

$$
\min _{i} \frac{1}{|\mathbf{w}|} y_{i}\left(\mathbf{w}^{T} \mathbf{x}_{i}+w_{0}\right)
$$

The objective of SVM is to find a hyperplane $\mathbf{w}^{T} \mathbf{x}_{i}+w_{0} = 0$ that separates two types of points and maximizes the margin.

$$
\arg \max _{\mathbf{w}, w_{0}}\left\{\min _i \frac{1}{\|\mathbf{w}\|} y_{i}\left(\mathbf{w}^{T} \mathbf{x}_{i}+w_{0}\right)\right\}
$$

i.e.,


$$
\arg \max _{\mathbf{w}, w_{0}} \frac{1}{\|\mathbf{w}\|} \left\{\min _i  y_{i}\left(\mathbf{w}^{T} \mathbf{x}_{i}+w_{0}\right)\right\}
$$

## Learning

We transform the optimization problem step by step such that it becomes easier to solve.

Note that distance is invariant to scaling of $\boldsymbol{w}$ and $b$ (or note that $\boldsymbol{w} ^\top \boldsymbol{x} +b = 0$ and $(k\boldsymbol{w}) ^\top \boldsymbol{x} + (kb) = 0$ characterize the same hyperplane), thus we can assume

$$
\min _{i} y_{i}\left(\mathbf{w}^{T} \mathbf{x}+w_{0}\right)=1
$$

Then the maximization-minimization problem becomes a constrained maximization problem


$$
\begin{equation}
\underset{\mathbf{w}, w_{0}}{\arg \max } \frac{1}{\|\mathbf{w}\|} \text { s.t. } \min _{i} y_{i}\left(\mathbf{w}^{T} \mathbf{x}+w_{0}\right)=1
\end{equation}
$$

Or equivalently,

$$
\begin{equation}
\underset{\mathbf{w}, w_{0}}{\arg \max } \|\mathbf{w}\| \quad \text { s.t. } y_{i}\left(\mathbf{w}^{T} \mathbf{x}_{i}+w_{0}\right) \geq 1
\end{equation}
$$

Why support vector?

Any $\boldsymbol{x}_i$ that satisfies $\hat{\boldsymbol{w}} ^\top \boldsymbol{x} + \hat{b} = 0$ is a support vector.
