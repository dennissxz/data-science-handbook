# Boy or Girl Paradox

*Q1: Mr. Smith has two children. At least one of them is a boy. What is the probability that both children are boys?*

*Q2: Mr. Smith has two children at home. You give a call to Mr. Smith's home and one of the two children will answer your call equally likely. It turns out that the child who answers your call is a boy. What is the probability that both children are boys?*


reference:
  - https://en.wikipedia.org/wiki/Boy_or_Girl_paradox

## Solution to Q1

If we formulate the question in the other way: You tossed two fair coins. It's known that at least one of them came up a head. What is the probability that both coins came up heads?

Let $X_H$ be the number of heads. The intuitive solution is a conditional probability

$$\mathrm{P}(X_H=2  \,\vert\, X_H\ge 1) = \frac{\mathrm{P}(X_H=2\cap X_H\ge 1)}{\mathrm{P}(X_H\ge 1)} = \frac{\mathrm{P}(X_H=2)}{\mathrm{P}(X_H\ge 1)} = \frac{1/4}{3/4}= \frac{1}{3}$$


But when it comes to boy or girl, it simulated a great deal of controversy. The paradox occurs when it is not known how the statement "at least one is a boy" was generated. One may refer to the Wikipedia link for more details of the other answer $1/2$.



## Solution to Q2

This is also a controversial question. Let $X_B \in \{0,1,2\}$ be the number of boys in the family. Let $A \in \{B,G\}$ be the outcome of who answer the call. And let $O\in\{BB,BG,GB,GG\}$ be the birth order of the wo children. It's known that

$$\begin{align}
\mathrm{P}(A=B\,\vert\,O=BB) &= 1 \\
\mathrm{P}(A=B\,\vert\,O=BG) &= 1/2 \\
\mathrm{P}(A=B\,\vert\,O=GB) &= 1/2 \\
\mathrm{P}(A=B\,\vert\,O=GG) &= 0 \\
\mathrm{P}(O_i) &= 1/4 \ \forall \ i \\
\end{align}$$

So by the Bayeisan formula,


$$\begin{align}
\mathrm{P}(X_B=2\,\vert\,A=B) &= \frac{\mathrm{P}(X_B=2 \cap A=B)}{\mathrm{P}(A=B)} \\
 &= \frac{\mathrm{P}(X_B=2)}{\mathrm{P}(A=B)} \ \text{since}\  \mathrm{P}(X_B=2 \cap A=G)=0\\
&=  \frac{\mathrm{P}(X_B=2)}{\sum_i \mathrm{P}(A=B\,\vert\, O_i)\mathrm{P}(O_i)} \\
&= \frac{1/4}{2/4} \\
&= \frac{1}{2} \\
\end{align}$$

One may argue that, if we know a boy answered the call, can we infer "at least one boy"? Yes, of course. This reasoning is correct, since the two events has the subset relation

$$\{A=B\} \subset \{X_B\ge 1\}$$

But keep in mind that they are not equivalent

$$\{A=B\} \ne \{X_B\ge 1\}$$

So what is the set difference?

$$\{X_B\ge1\} \backslash \{A=B\}$$

Image an experiment on $n$ families with two children. For each family $i$, you make a call and ask a question "how many boys are there in your family?" and record the receiver's gender $A_i$ and the answer $X_{B,i}$. In this sense, it is easy to find there are four possible pairs

$$(A_i, X_{B,i}) \in \{(B,1), (B,2), (G,0), (G,1)\}$$

As a results, the families with the first two pairs of answers corresponds to $\{A=B\}$, but they are not $\{X_B \ge 1\}$. Yes, the last kind of family $(G, 1)$ is the set difference $\{X_B\ge1\} \backslash \{A=B\}$, since they have $X_{B,i}=1$ but $A_i=G$.

In particular, the four kinds of pairs should be equally likely to be observed.

Therefore, the inference is correct but one should replace $\{A=B\}$ by $\{X_B\ge 1\}$ in calculation. The latter corresponds to larger outcome space, then leads to a larger value of the denominator ($\frac{2}{4}$ vs. $\frac{3}{4}$), and hence a smaller value of the probability ($\frac{1}{2}$ vs. $\frac{1}{3}$).
