# Multidimensional Scaling

Does not project new data points. Only work for a given data set.

Input: a matrix of Euclidean distances between points, instead of a $n \times p$ data matrix.

## Objective

MDS seeks a $k$-dimensional representation $\boldsymbol{z} \in \mathbb{R} ^k$ of a data set that preserves inner products between pairs of data points $(\boldsymbol{x_i}, \boldsymbol{x}_j)$


$$
\begin{equation}
\min \sum_{i j}\left(\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}-\boldsymbol{z}_{i} \cdot \boldsymbol{z}_{j}\right)^{2}
\end{equation}
$$

## Leraning

The solution can be obtained from the $N\times N$ Gram matrix of inner products


$$
\begin{equation}
\boldsymbol{G}=\boldsymbol{X} \boldsymbol{X}^{T}
\end{equation}
$$

where

$$
\begin{equation}
g_{i j}=\boldsymbol{x}_{i} \cdot \boldsymbol{x}_{j}
\end{equation}
$$

The output projections are given by

$$
\begin{equation}
\boldsymbol{z}_{i \alpha}=\sqrt{\lambda_{\alpha}} \boldsymbol{v}_{\alpha i}, 1 \leq \alpha \leq k
\end{equation}
$$

where $\boldsymbol{v} _\alpha$ is the $\alpha^{\text{th}}$ eigenvector of $\boldsymbol{G}$

#### Special Case

Define the Euclidean distances matrix $\boldsymbol{F}$

$$
\begin{equation}
f_{i j}=\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}=\left\|\boldsymbol{x}_{i}\right\|^{2}-2 \boldsymbol{x}_{i}^{T} \boldsymbol{x}_{j}+\left\|\boldsymbol{x}_{j}\right\|^{2}
\end{equation}
$$

If $\boldsymbol{x} _i$ are centered (zero-mean), we can convert the Euclidean distance matrix to a Gram matirx by left- and right-multiplying by the centering matrix

$$
\begin{equation}
\boldsymbol{G}=-\frac{1}{2}\left(\boldsymbol{I}-\boldsymbol{u} \boldsymbol{u}^{T}\right) \boldsymbol{F}\left(\boldsymbol{I}-\boldsymbol{u} \boldsymbol{u}^{T}\right)
\end{equation}
$$

where

$$
\begin{equation}
\boldsymbol{u}=\frac{1}{\sqrt{N}}(1,1, \ldots, 1)^{T}
\end{equation}
$$

## Properties

MDS projections $\boldsymbol{z} _i$ are the same as those of PCA.
- For each eigenvector $\boldsymbol{u}_i$ of $\boldsymbol{S}$, there is a corresponding eigenvector $\boldsymbol{v} _j = \boldsymbol{X} ^\top \boldsymbol{w} _i$ ?? of $\boldsymbol{G} = \boldsymbol{X} ^\top \boldsymbol{X}$.
- The first $k$ vectors $\boldsymbol{v} _i$ gives the projected  data in both PCA and MDS.
- Many non-linear dimensionality reduction methods are extension to MDS.
- Unlike PCA, MDS only gives projections for the training set; it does not give us a way to project a new data point.
