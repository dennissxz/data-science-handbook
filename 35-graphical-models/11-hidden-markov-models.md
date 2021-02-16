# Hidden Markov Models

A hidden Markov model has a two sequences: state $S_t$ and observation $O_t$. The hidden, unobservable states $S_t$ follow a Markov chain with transition probability $P(S_t \vert S_{t-1})$ and the observation follows a conditional distribution $P(O_t \vert S_t=s_t)$ parameterized by $s_t$.


$$\begin{aligned}
& S_1 \longrightarrow && S_2  \longrightarrow &&\ldots \longrightarrow && S_t  \\
&\downarrow &&\downarrow && &&\downarrow \\
& O_1 && O_2 && \ldots && O_t \\
\end{aligned}$$

Given a sequence of observations $o_1, \ldots, o_t$, we want to solve

- What is the probability of this sequence? (the scoring problem)
- What is your best guess of the sequence of hidden states $s_1, \ldots, s_t$? (the decoding problem)
- Given a large number of observations, how could you learn a
model of the conditional distribution $P(O_t\vert S_t)$? (the training problem)
