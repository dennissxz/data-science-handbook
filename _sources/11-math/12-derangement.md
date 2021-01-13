# Derangement

*At a party, $n$ gentlemen check their hats. They “have a good time”,
and are each handed a hat on the way out. In how many ways can
the hats be returned so that no one is returned his own hat?*

This is a derangement problem. An $n$-derangement is an $n$-permutation $\pi=p_1p_2\ldots p_n$ such that $p_1\ne1, p_2\ne2, \ldots, p_n\ne n$. We let $D_n$ be the number of all $n$-derangements, sometimes denoted by $!n$.

$$
D_{n}=n! \left( \frac{1}{2!}-\frac{1}{3!}+\ldots+(-1)^{n} \frac{1}{n!}\right)
$$

## Sol.1 Recurrence Relation

There is a recurrence relation

$$D_n = (n-1)\left(D_{n-1} + D_{n-2}\right)$$

where $D(1)=0, D(2)=1$.

To understand it, let person $i$'s hat be $h_i$, and suppose he gets person $j$'s hat $h_j$. Consider two cases that can happen to person $j$.

- He gets $h_i$. In this case, the other $n-2$ persons and the other $n-2$ hats formulate a problem $D(n-2)$.
- He gets a hat other than $h_i$. In this case, he and the other $n-2$ persons and the $n-1$ hats (total excluding $h_j$) formulate a problem $D(n-1)$. Why? Check the constraints: for each of the $n-1$ person, there is a hat that he cannot get. In particular, person $j$ cannot get $h_i$, and person $k$ cannot get $h_k$ for $k\ne j$. The constraints are the same as a regular $D(n-1)$ problem.

Moreover, for a person $i$, there are $n-1$ ways to get a hat $h_j$, so the recurrence relation has the above form.

To derive the explicit form of $D_n$, let $D_n = n ! M_{n}$, where $M_1=0, M_2=\frac{1}{2}$.

By the recurrence relation we have

$$\begin{align}
n ! M_{n} & = (n-1)(n-1) ! M_{n-1}+(n-1)(n-2) ! M_{n-2} \\
& =n ! M_{n-1}-(n-1) ! M_{n-1}+(n-1) ! M_{n-2}
\end{align}$$

Dividing by $(n-1)!$ on both sides gives

$$nM_n = nM_{n-1} - M_{n-1} + (n-1)!M_{n-2}$$

Rearrangement gives

$$\begin{align}
M_{n}-M_{n-1} &= -\frac{1}{n}\left(M_{n-1}-M_{n-2}\right) \\
&=\ldots \\
&=\left(-\frac{1}{n}\right)\left(-\frac{1}{n-1}\right) \ldots \left(-\frac{1}{3}\right)\left(M_{2}-M_{1}\right) \\
&=(-1)^{n} \frac{1}{n !}
\end{align}$$

Then

$$\begin{align}
M_{n} &= (M_n - M_{n-1}) + (M_{n-1} - M_{n-2}) + \ldots + (M_2 - M_1) \\
& =(-1)^{2} \frac{1}{2 !}+(-1)^{3} \frac{1}{3 !} \ldots+(-1)^{n} \frac{1}{n !} \\
\end{align}$$

Finally, substituting the equation into $D_n=n! M_n$ gives

$$D_{n}=n!\left(\frac{1}{2 !}-\frac{1}{3 !}+\ldots+(-1)^{n} \frac{1}{n !}\right)$$

## Sol.2 Inclusion-Exclusion Principle

Let $E_i$ be the event that the $i$-th person gets his hat. Then the event that at least one person get his hat is

$$E_{1} \cup E_{2} \cup \cdots \cup E_{n}$$

And we want to find the probability that no one get his hat, which is

$$\overline{E_{1}} \cap \overline{E_{2}} \cap \ldots \cap \overline{E_{n}} = \overline{E_{1} \cup E_{2} \cup \ldots \cup E_{n}}$$

By the inclusion-exclusion principle,


$$\begin{aligned}
\left\vert E_{1} \cup E_{2} \cup \cdots \cup E_{n}\right\vert &=\sum_{j=1}^{n}(-1)^{j+1} \sum_{i_{1}<i_{2}<\cdots<i_{j}} \left\vert E_{i_{1}} \cap E_{i_{2}} \cap \cdots \cap E_{i _{j}}\right\vert \\
&=\sum_{i=1}^{n} \left\vert E_{i}\right\vert-\sum_{i_{1}<i_{2}} \left\vert E_{i_{1}} \cap E_{i_{2}}\right\vert+\cdots \\
&\quad +(-1)^{j+1} \sum_{i_{1}<i_{2}<\cdots<i_{j}} \left\vert E_{i_{1}} \cap E_{i_{2}} \cap \cdots \cap E_{i_{j}}\right\vert \\
&\quad +\cdots \\
&\quad +(-1)^{n+1} \sum_{i_{1}<i_{2}<\cdots<i_{n}} \left\vert E_{i_{1}} \cap E_{i_{2}} \cap \cdots \cap E_{i_{n}}\right\vert
\end{aligned}$$

Note that the number of ways that exactly $j$ people get their hats is

$$\sum_{i_{1}<i_{2}<\cdots<i_{j}}\left\vert E_{i_{1}} \cap E_{i_{2}} \cap \cdots \cap E_{i _{j}}\right\vert=\frac{C_n^j (n-j)!}{n!} = \frac{n!}{j!} $$

Therefore,

$$\begin{align}
\left\vert E_{1} \cup E_{2} \cup \cdots \cup E_{n} \right\vert  &= \sum_{j=1}^{n}(-1)^{j+1} \sum_{i_{1}<i_{2}<\cdots<i_{j}} \left\vert E_{i_{1}} \cap E_{i_{2}} \cap \cdots \cap E_{i _{j}}\right\vert \\
&= n! \sum_{j=1}^{n}(-1)^{j+1} \frac{1}{j!} \\
\end{align}$$

and the detergent number is

$$\begin{align}
D_n &=\left\vert \overline{E_{1}} \cap \overline{E_{2}} \cap \ldots \cap \overline{E_{n}}\right\vert  \\
 &= n! - \left\vert E_{1} \cup E_{2} \cup \ldots \cup E_{n}\right\vert \\
&= n!\left(\frac{1}{2 !}-\frac{1}{3 !}+\ldots+(-1)^{n} \frac{1}{n !}\right) \\
\end{align}$$
