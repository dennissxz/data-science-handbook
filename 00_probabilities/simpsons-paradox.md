# Simpson's Paradox

*Verbally describe and mathematically formulate Simpson's paradox*


## Verbal Description

From Wikipedia:

> Simpson's paradox is a phenomenon in probability and statistics, in which a trend appears in several different groups of data but disappears or reverses when these groups are combined.

The cause of Simpson's paradox is the existence of a confounding variable.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/f/fb/Simpsons_paradox_-_animation.gif" width="50%" align="center"/>
</p>


## Math Formulation


Mathematically, if the trends are correlations, it can be formulated as

$$\mathrm{Cov}(X,Y\vert Z) > 0 \not \Rightarrow \mathrm{Cov}(X,Y) > 0$$

where $Z$ is a confounding variable.

If the trends are proportions, we can illustrate them with vectors. In the below case, for a vector $v$, suppose the angle between it and the $x$-axis is $\theta$. The horizontal projection $\vert \overrightarrow{v}\cos\theta\vert$ is the number of applicants and the vertical projection $\vert \overrightarrow{v}\sin\theta\vert$ is the number of accepted candidates. A steeper vector then represents a larger acceptance rate.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/c/cd/Simpsons_paradox.jpg" width="50%" />
</p>



The Simpson's paradox says that, even if $\overrightarrow{L_{1}}$ has a smaller slope than $\overrightarrow{B_{1}}$ and $\overrightarrow{L_{2}}$ has a smaller slope than $\overrightarrow{B_{2}}$, the overall slop $\overrightarrow{L_{1}} + \overrightarrow{L_{2}}$ can be greater than $\overrightarrow{B_{1}}+\overrightarrow{B_{2}}$. For this to occur, one of the orange vectos must have a greater slope than one of the blue vectors (e.g. $\overrightarrow{L_{2}}$ vs. $\overrightarrow{B_{1}}$), and these will generally be **longer** than the alternatively subscripted vectors (i.e. imbalanced data) – thereby dominating the overall comparison.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Simpson_paradox_vectors.svg/1200px-Simpson_paradox_vectors.svg.png" width="50%">
</p>
