# Multidimensional Scaling

Does not project new data points. Only work for a given data set.

Input: a matrix of Euclidean distances between points, instead of a $n \times p$ data matrix.

## Objective

MDS seeks a $k$-dimensional representation $\boldsymbol{z} \in \mathbb{R} ^k$ of a data set that preserves inner products between pairs of data points $(\boldsymbol{x_i}, \boldsymbol{x}_j)$


$$
\begin{equation}
\min \sum_{i j}\left(\mathbf{x}_{i} \cdot \mathbf{x}_{j}-\mathbf{z}_{i} \cdot \mathbf{z}_{j}\right)^{2}
\end{equation}
$$

## Leraning

The solution can be obtained from the $N\times N$ Gram matrix of inner products


$$
\begin{equation}
\mathbf{G}=\mathbf{X} \mathbf{X}^{T}
\end{equation}
$$

where

$$
\begin{equation}
\mathbf{G}_{i j}=\mathbf{x}_{i} \cdot \mathbf{x}_{j}
\end{equation}
$$

The output projections are given by

$$
\begin{equation}
\mathbf{z}_{i \alpha}=\sqrt{\lambda_{\alpha}} \mathbf{v}_{\alpha i}, 1 \leq \alpha \leq k
\end{equation}
$$

where $\boldsymbol{v} _\alpha$ is the $\alpha^{\text{th}}$ eigenvector of $\boldsymbol{G}$

Define the Euclidean distances matrix

$$
\begin{equation}
\mathbf{F}_{i j}=\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|^{2}=\left\|\mathbf{x}_{i}\right\|^{2}-2 \mathbf{x}_{i}^{T} \mathbf{x}_{j}+\left\|\mathbf{x}_{j}\right\|^{2}
\end{equation}
$$

If $\boldsymbol{x} _i$ are centered (zero-mean), we can convert the Euclidean distance matrix to a Gram matirx by left- and right-multiplying by the centering matrix

$$
\begin{equation}
\mathbf{G}=-\frac{1}{2}\left(\mathbf{I}-\mathbf{u} \mathbf{u}^{T}\right) \mathbf{F}\left(\mathbf{I}-\mathbf{u} \mathbf{u}^{T}\right)
\end{equation}
$$

where

$$
\begin{equation}
\mathbf{u}=\frac{1}{\sqrt{N}}(1,1, \ldots, 1)^{T}
\end{equation}
$$

## Properties

MDS projections $\boldsymbol{z} _i$ are the same as those of PCA.
- For each eigenvector $\boldsymbol{u}_i$ of $\boldsymbol{S}$, there is a corresponding eigenvector $\boldsymbol{v} _j = \boldsymbol{X} ^\top \boldsymbol{w} _i$ ?? of $\boldsymbol{G} = \boldsymbol{X} ^\top \boldsymbol{X}$.
- The first $k$ vectors $\boldsymbol{v} _i$ gives the projected  data in both PCA and MDS.
