# Fields and Vector Spaces


## Fields

Definition (Fields)
: A field is a set $F$ with *addition* and *multiplication* (mappings $F\times F \rightarrow F$), which satisfies the following properties

: - Commutativity: $\forall a, b \in F, a+b=b+a, a \cdot b=b \cdot a$

  - Association: $\forall a, b, c \in F, (a+b)+c=a+(b+c),(a \cdot b) \cdot c=a \cdot(b \cdot c)$

  - Additive and multiplicative identity: $\exists 0,1 \in F \text { s.t. } \forall a \in F, a+0=a \text { and } a \cdot 1=a$


  - Additive inverse: $\forall a \in F, \exists(-a) \in F, \text { s.t. } a+(-a)=0$

  - Multiplicative inverse: $\forall a \in F(a \neq 0), \exists a^{-1} \in F, \text { s.t. } a \cdot a^{-1}=1$

  - Distributivity of multiplication over addition: $\forall a, b, c \in F, a \cdot(b+c)=a \cdot b+a \cdot c$

**Examples**

- $\mathbb{Q}, \mathbb{R}, \mathbb{C}, \mathbb{F}_{2}, \mathbb{F}_{p}$ ($p$ is a prime).


## Vector Spaces

```{margin}
A vector space is defined per field.
```

Definition (Vector spaces)
: A vector space over field $F$ is a set $V$ associated with operations *addition* and *scalar multiplication* that satisfies the following properties

- Association of addition: $\forall \mathbf{u}, \mathbf{v}, \mathbf{w} \in V,(\mathbf{u}+\mathbf{v})+\mathbf{w}=\mathbf{u}+(\mathbf{v}+\mathbf{w})$

- Commutativity of addition: $\forall \mathbf{u}, \mathbf{v} \in V, \mathbf{u}+\mathbf{v}=\mathbf{v}+\mathbf{u}$

- Identity element of addition: $\exists \mathbf{0} \in V$ s.t. $\forall \mathbf{v} \in V, \mathbf{v}+\mathbf{0}=\mathbf{v}$

- Compatibility of scalar multiplication with field multiplication: $\forall a, b \in F, \mathbf{v} \in V, a(b \mathbf{v})=(a b) \mathbf{v}$

- Identity element of scalar multiplication: $1 \mathbf{v}=\mathbf{v}$ where $1$ is the multiplicative identity in $F$

- Distributivity of scalar multiplication with respect to vector addition: $\forall a \in F, \mathbf{u}, \mathbf{v} \in V, a(\mathbf{u}+\mathbf{v})=a \mathbf{u}+a \mathbf{v}$

- Distributivity of scalar multiplication with respect to field addition: $\forall a, b \in F, \mathbf{v} \in V,(a+b) \mathbf{v}=a \mathbf{v}+b \mathbf{v}$

**Examples**

- $\mathbb{R}^d$ is a vector space over field $\mathbb{R}$

- $\mathbb{R}$ is a vector space over $\mathbb{Q}$

- All Fibonacci sequences form a vector space over $\mathbb{R}$

    $$
    \left\{\left(x_{1}, x_{2}, x_{3} \ldots\right) \in \mathbb{R}^{\infty} \mid \forall i \geq 1: x_{i}+x_{i+1}=x_{i+2}\right\}
    $$


## Inner Product Spaces

Definition (Inner product spaces)
: An inner product space is a vector space $V$ over a field $\mathbb{F}$ **together** with a map $\langle\cdot, \cdot\rangle \rightarrow \mathbb{F}$ that satisfies

1. Linearity in the first argument

$$
\langle a \mathbf{x}, \mathbf{y}\rangle=a\langle\mathbf{x}, \mathbf{y}\rangle \\
\left\langle\mathbf{x}_{1}, \mathbf{y}\right\rangle+\left\langle\mathbf{x}_{2}, \mathbf{y}\right\rangle=\left\langle\mathbf{x}_{1}+\mathbf{x}_{2}, \mathbf{y}\right\rangle
$$

1. Conjugate symmetry (Hermitian symmetry):

$$\langle\mathbf{x}, \mathbf{y}\rangle=\overline{\langle\mathbf{y}, \mathbf{x}\rangle}$$

1. Positive definite

$$\langle\mathbf{x}, \mathbf{x}\rangle>0 \quad$ if $\mathbf{x} \neq \mathbf{0} \quad$ o.w. $\langle\mathbf{x}, \mathbf{x}\rangle=0$$

Note that $2$ and $3$ imply that $\langle\mathbf{x}, \mathbf{x}\rangle \in \mathbb{R}_{\geq 0}$.

Inner product spaces are (Hausdorff) Pre-Hilbert Spaces.

## Hilbert Spaces

Hilbert spaces are **complete** pre-Hilbert spaces. A Hilbert space has inner product, and is complete.

**Examples**

- $\mathbb{R} ^d$
