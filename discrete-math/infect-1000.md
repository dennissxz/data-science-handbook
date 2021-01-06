
# Infect 1000

*There are 1,000 people in one room. One of them carries a disease which infects 100% if one shakes hands with an infected person. In each minute all the people in the room are randomly paired to shake hands with each other. What is your estimate of the expected number of people infected after 10 minutes? Can you use only pen and paper to solve this?*


## Solution 1: Markov Chain

Let $I_t$ be the number of infected individuals after $t$ minutes. Define a transition matrix $P$, where

$$p_{i,j} = \operatorname{P}(I_{t+1}=j\vert I_t=i)$$

It's easy to see that

- $p_{1,2}=1$
- $p_{2,2}=\frac{1}{999}, p_{2,4}=\frac{998}{999}$

In general, we have, for $l=0,1,\ldots, \min(k, \frac{n}{2}-k)$,

$$p_{2k, 2(k+l)}= \frac{C_{2k}^{2l} C_{n-2k}^{2l} P_{2l}^{2l} R_{2k-2l} R_{n-2k-2l}}{R_{n}}$$

where
- $C_{2k}^{2l}$ is the number of ways to select $2l$ infected individuals from the total $2k$ infected individuals to be paired with $2l$ healthy individuals
- $C_{n-2k}^{2l}$ is the number of ways to select $2l$ healthy individuals from the total $n-n-2k$ healthy individuals to be paired with $2l$ infected individuals
- $P_{2l}^{2l}$ is the number of ways to pair $2l$ infected individuals and $2l$ healthy individuals, such that additional $2l$ individuals are infected
- $R_m$ is the number of ways to arrange a even number $m$ of individuals into $\frac{m}{2}$ pairs. It's easy to find

$$R_m=\frac{C_m^2C_{m-2}^2\ldots C_2^2 }{\frac{m}{2}!} = \frac{m!}{(2!)^{\frac{m}{2}}(\frac{m}{2}!)}$$


- $R_{2k-2l}$ is the number of ways to arrange $2k-2l$ infected individuals into pairs, who do not shake hands with healthy individuals
- $R_{n-2k-2l}$ is the number of ways to arrange $n-2k-2l$ healthy individuals into pairs, who do not shake hands with infected individuals
- $R_n$ is the total number of ways to pair $n$ people in the room

Simplification gives

$$p_{2k, 2(k+l)} =\frac{C_{n/2}^{k+l}(k+l)!4^l}{C_n^{2k}(k-l)!(2l)!}$$

Then the distribution of $I_{10}$ is

$$\boldsymbol{p}_{I_{10}}^\top=\boldsymbol{e}_1^\top P^{10}$$

where $\boldsymbol{e}_1 = [1,0,\ldots,0]$ and $\boldsymbol{p}_{I_{10}} = [p_{1,1}^{(10)}, \ldots, p_{1,n}^{(10)}]$

The expected number is

$$\mathrm{E}(I_{10}) = [1,2,3,\ldots, n]^\top\boldsymbol{p}_{I_{10}} $$

## Solution 2: Conditional Expectation

For each infected individual,

$$\begin{align}
p_t :&= \operatorname{P}(\text{an infected individual is paried with any healthy individuals at time $t$}) \\
& = \frac{n-I_t} {n - 1} \\
\mathrm{E}(I_{t+1}|I_t) & = I_t + I_t \times p_t \\
& = I_t + I_t \times \frac{n-I_t}{n - 1} \\
\end{align}$$

How to solve it?

By the law of total expectation,

$$\begin{align}

\mathrm{E}(I_{t+1}) & = E\left[\mathrm{E}(I_{t+1}|I_t)\right] \\
& = E\left(I_t + I_t \times \frac{n-I_t}{n - 1} \right)\\
\end{align}$$
In particular, since $\mathrm{P}(I_1=2\vert I_0=1)=1$,

$$\begin{align}

\mathrm{E}(I_2) &= E\left[\mathrm{E}(I_2\vert I_1)\right] \\
&= E\left(I_1 + I_1 \times \frac{n-I_1}{n - 1}\right)\\
& = 2 + 2 \times 998 / 999 \\
& = 3994/999
\end{align}$$

But when $t\ge 3$, it's hard to compute all $\mathrm{P}(I_{t+1} \vert I_t)$ by hand, since $I_t$ can take $2^{t-1}$ different values.

One may try to substitute $I_t$ by $\mathrm{E}(I_t)$ such that

$$\begin{align}

\mathrm{E}(I_{t+1}) &= E\left(I_1 + I_1 \times \frac{n-I_1}{n - 1}\right)\\
&= \mathrm{E}(I_t) + \mathrm{E}(I_t) \times \frac{n-\mathrm{E}(I_t)}{n - 1}

\end{align}$$

and use this recurrence relation to find $\mathrm{E}(I_{10})$. But clearly this is incorrect since expectation is a linear operator and $\mathrm{E}(I_t^2)\ne [\mathrm{E}(I_t)]^2$, unless $\mathrm{Var}(I_t)=0$, but it's clearly not.
