# Hidden Markov Models





A hidden Markov model has a two sequences: state $S_t$ and observation $O_t$. The hidden, unobservable states $S_t$ follow a Markov chain with transition probability $P(S_t \vert S_{t-1})$ and the observation follows a conditional distribution $P(O_t \vert S_t=s_t)$ parameterized by $s_t$.


$$\begin{aligned}
& S_1 &&\longrightarrow && S_2  &&\longrightarrow &&\ldots&& \longrightarrow && S_t  \\
&\downarrow&& &&\downarrow&& &&&& &&\downarrow \\
& O_1 &&&& O_2&& && \ldots && &&O_t \\
\end{aligned}$$

Given a sequence of observations $o_1, \ldots, o_t$, we want to solve

- What is the probability of this sequence $o_1, \ldots, o_t$? (the **scoring** problem)
- What is your best guess of the sequence of hidden states $s_1, \ldots, s_t$? (the **decoding** problem)
- Given a large number of observations, how could you learn a
model of the conditional distribution $P(O_t\vert S_t)$? (the **training** problem)

## Problem

Notations:

- $\boldsymbol{A}$: $N\times N$ state transition probability matrix where $a_{ij} = P(q_{t+1} = j\vert q_t = i)$
- $\boldsymbol{\pi}  = \left\{ \pi_i \right\}$: initial state distribution, where $\pi_i = P(q_1 = i)$
- $V= \left\{ v_1, \ldots, v_M \right\}$: set of $M$ possible observation labels (or vectors, in general). Obervation at time $t$ is $o_t \in V$
- $\boldsymbol{B}$: $N\times M$ observation (emission) distribution in state $i$ where $b_{i}(k) = P(o_{t} = v_k\vert q_t = i)$

The entire model parameters is $\lambda = \left\{ \boldsymbol{A} , \boldsymbol{B} ,\pi \right\}$

The problems can be formulated as

- Scoring problem
  Given $\boldsymbol{O_T} =$ and $\lambda$, compute $P(\boldsymbol{O} \vert \lambda )$
  forward algorithm

- Decoding
  Given

- Training (learning)
  Given $\boldsymbol{o}$, find $\lambda$ to maximize  


### Diagrams

transition diagrams, trellis

### Scoring

#### Forward Algorithm

Enumeration over $\boldsymbol{q}$ is computational expensive.

Use DP.

Define $a_t{i}$ the probability of observing $\boldsymbol{O}_T$ at time $t$ in state $i$,

$$
P\left(\mathbf{o}_{1} \mathbf{o}_{2} \ldots \mathbf{o}_{t}, q_{t}=i \mid \lambda\right)
$$

Then

- For $t=1$
- For $t>1$, for $a_t{j}$, we consider the all

Marginalize The final state


#### Backward Algorithm

Define $b_t{i}$ the probability of observing $\boldsymbol{O}_T$ at time $t$ in state $i$,


### Decoding

Joint probability of state sequence and observation sequence.

Viterbi Algorithm

### Training

ML to find for $\lambda$.

If states is given, then ML is easy by counting. Similar like Gaussian.

But the states are not given, so we provide initial guess, and iteratively update $\lambda$.

Baum-Welch Algorithm: EM for HMMs






## Applications

### Speech

### Pose

### Speech Tags

## Hidden Topic Markov Models

Topic states.

## Model Selection

Perplexity












.
