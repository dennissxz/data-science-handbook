# Support Vector Machine

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

## Objective




Binary classification

Find a hyper-plane to separate data points.

Maximize the distance from the plane to the closest data.

##
