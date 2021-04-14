# Geometry

## Half-spaces

A half-space is either of the two parts into which a hyperplane $(\boldsymbol{a} ^\top \boldsymbol{x} = b)$ divides an affine space.

A strict linear inequality specifies an **open** half-space:

$$\boldsymbol{a} ^\top \boldsymbol{x} >b$$

A non-strict one specifies a **closed** half-space:

$$\boldsymbol{a} ^\top \boldsymbol{x}  \ge b$$

A half-space is a convex set.

## Polytopes

Let's first review some basics.

- A **polygon** (多边形) is a 2-dimensional polytope

  :::{figure} geometry-polygon
  <img src="../imgs/geometry-polygon.png" width = "40%" alt=""/>

  Polygons [[Wikipedia](https://en.wikipedia.org/wiki/Polytope)]
  :::

- A **polyhedron** (多面体) is a 3-dimensional polytope.


  :::{figure} geometry-polyhedron
  <img src="../imgs/geometry-polyhedron.png" width = "70%" alt=""/>

  Polyhedrons [[Wikipedia](https://en.wikipedia.org/wiki/Polytope)]
  :::

Then, a **polytope** (多胞体) is a generalization in any number of dimensions. It a geometric object with flat sides. Flat sides mean that the sides of a (k+1)-polytope consist of k-polytopes that may have (k−1)-polytopes in common. For instance, the sides of a polyhedron consist of polygons that may have 1-polytope (dion) in common.

Properties

- **Convex**: A polytope can be convex. For instance, intersection of a set of half-spaces.

- **Bounded**: A polytope is bounded if there is a ball of finite radius that contains it.

Definition (full-dimensional)
: A polyhedron $P \in \mathbb{R} ^n$ is full-dimensional if it has positive volume. Equivalently,
  - if it has $n+1$ affinely independent points
  - if it contains a ball of positive radius

(ellipsoid)=
## Ellipsoids

An ellipsoid is a generalization of an ellipse into any number of dimensions.

In 3-d case, the implicit equation of an ellipsoid has the standard form

$$
\frac{x^{2}}{a^{2}}+\frac{y^{2}}{b^{2}}+\frac{z^{2}}{c^{2}}=1
$$

:::{figure} geometry-ellipsoid
<img src="../imgs/geometry-ellipsoid.png" width = "50%" alt=""/>

Sphere ($a = b = c$, top), Spheroid ($a = b \ne c$, bottom left), Tri-axial ellipsoid ($a,b,c$ all different, bottom right)
:::

### Characterization

Formally, a set $E$ of vectors in $\mathbb{R} ^n$ of the form

$$
E=E(\boldsymbol{z} , \boldsymbol{D})=\left\{\boldsymbol{x}  \in \mathbb{R}^{n} \mid(\boldsymbol{x} -\boldsymbol{z} )^{\top} \boldsymbol{D} ^{-1}(\boldsymbol{x} -\boldsymbol{z} ) \leq 1\right\}
$$

where $\boldsymbol{D}$ is an $n\times n$ positive definite matrix, is called an ellipsoid with center $\boldsymbol{z} \in \mathbb{R} ^n$.

Other formulation: $E(\boldsymbol{b} , \boldsymbol{A})=\left\{\boldsymbol{x} \in \mathbb{R} ^2:\|\boldsymbol{A} \boldsymbol{x} -\boldsymbol{b} \|_{2} \leq 1\right\}$ where $\boldsymbol{A}$ is p.d. The volume of $E$ is [proportional](https://math.stackexchange.com/questions/3252200/volume-of-ellipsoid-via-determinant#:~:text=be%20the%20equation%20of%20a,det(A%E2%88%921)) to $\operatorname{det} (\boldsymbol{A} ^{-1} )$.

Relation between $(\boldsymbol{z} ,\boldsymbol{D})$ and $(\boldsymbol{b}, \boldsymbol{A})$:


$$\begin{aligned}
\|\boldsymbol{A} \boldsymbol{x} -\boldsymbol{b} \|_{2} &\leq 1 \\
\left( \boldsymbol{A} \boldsymbol{x}  - \boldsymbol{b}  \right) ^{\top} (\boldsymbol{A} \boldsymbol{x} - \boldsymbol{b} )&\le 1 \\
[\boldsymbol{A} \left( \boldsymbol{x}  - \boldsymbol{A} ^{-1}  \boldsymbol{b}  \right)] ^{\top} [\boldsymbol{A} \left( \boldsymbol{x}  - \boldsymbol{A} ^{-1}  \boldsymbol{b}  \right)]&\le 1 \\
(\boldsymbol{x} - \boldsymbol{A} ^{-1} \boldsymbol{b} ) ^{\top} (\boldsymbol{A} ^{\top} \boldsymbol{A} ) (\boldsymbol{x} - \boldsymbol{A} ^{-1} \boldsymbol{b} )&\le 1\\
\end{aligned}$$

Hence, the relation is $\boldsymbol{z} = \boldsymbol{A} ^{-1} \boldsymbol{b}$ and $\boldsymbol{D} = (\boldsymbol{A} ^{\top} \boldsymbol{A} ) ^{-1}$, or $\boldsymbol{b} = \boldsymbol{A} \boldsymbol{z}$ and $\boldsymbol{A} = \boldsymbol{D} ^{-1/2}$. Hence $\operatorname{vol} (E) \propto \operatorname{det} (\boldsymbol{D} ^{1/2})$, or $\operatorname{det} (\boldsymbol{D} ) \propto \operatorname{vol}(E)^2$. We use many properties of positive definite matrices, see the relevant [section](pd-matrix).

### Ball

[Wikipedia](https://en.wikipedia.org/wiki/Ball_(mathematics))

For any $r >0$, the ellipsoid

$$
\begin{aligned}
E\left(\boldsymbol{z} , r^{2} I\right) &=\left\{\boldsymbol{x}  \in \mathbb{R}^{n} \mid(\boldsymbol{x} -\boldsymbol{z} )^{\top}(\boldsymbol{x} -\boldsymbol{z} ) \leq r^{2}\right\} \\
&=\left\{\boldsymbol{x}  \in \mathbb{R}^{n} \mid\|\boldsymbol{x} -\boldsymbol{z} \|_2 \leq r\right\}
\end{aligned}
$$

is called a **ball** centered at $\boldsymbol{z}$ of radius $r$.
