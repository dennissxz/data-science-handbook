# For Big Data

Problems
- Cannot even read the whole data. Sol: sub-linear time algorithms
- Cannot afford storing the data. Sol: streaming algorithms
- Single processor does not suffice. Sol: map-reduce algorithms

Tasks
- Searching through the data, e.g. nearest neighbor
- Efficient summary, e.g. summary statistics

Techniques
- Sampling, importance sampling
- Dimensionality reduction

Reference
- https://www.sketchingbigdata.org/fall17/lec/lec1.pdf
- CMPSCI 711: More Advanced Algorithms. [link](https://people.cs.umass.edu/~mcgregor/CS711S18/index.html)

Terms:

- **Sketching** is the compression $C(x)$ of some data set $x$ that allows us to query $f(x)$.  We can understand the sketch as a compression of $x$ which largely reduces the dimension of vector but it still has enough information to give a good estimation for the query we care about
  - For instance, $x$ is a sequence of observations, and we are interested in $\operatorname{mean} (x)$. But $x$ is too large, so we use $C(x)$ to (sometimes approximately) compute $\operatorname{mean} (x)$.
  - Linear sketch:
    - $C(x + y) = C(x) + C(y)$
    - $C(ax) = aC(x)$
  - In real life, if there are two data stream $x$ and $y$ in two different places, and the headquarter want to estimate $f(x + y)$, it’s not necessary to send $x$ and $y$ to the headquarter and do the calculation. Instead, we can send $C(x)$ and $C(y)$ to headquarter and output $f(C(x) + C(y))$.
  - It is possible to have binary $f$, i.e. compute $f(x, y)$ from $C(x)$ and $C(y)$.
