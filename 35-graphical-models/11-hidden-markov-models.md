# Hidden Markov Models


A hidden Markov model has a two sequences: state $S_t$ and observation $O_t$. The hidden, unobservable states $S_t$ follow a Markov chain with transition probability $P(S_t \vert S_{t-1})$ and the observation follows a conditional distribution $P(X_t \vert S_t=s_t)$ parameterized by $s_t$.


$$\begin{aligned}
& S_1 &&\longrightarrow && S_2  &&\longrightarrow &&\ldots&& \longrightarrow && S_t  \\
&\downarrow&& &&\downarrow&& &&&& &&\downarrow \\
& X_1 &&&& X_2&& && \ldots && &&X_t \\
\end{aligned}$$

Given a sequence of observations $x_1, \ldots, x_t$, we want to solve

- What is the probability of this sequence $x_1, \ldots, x_t$? (the **scoring** problem)
- What is your best guess of the sequence of hidden states $s_1, \ldots, s_t$? (the **decoding** problem)
- Given a large number of observations, how could you learn a
model of the conditional distribution $P(X_t\vert S_t)$? (the **training** problem)

## Problems

Notations:

- $\boldsymbol{P}$: $n\times n$ state transition probability matrix, where $p_{ij} = P(S_{t+1} = j\vert S_t = i)$
- $\boldsymbol{\pi}  = (\pi_1, \ldots, \pi_n)$: initial state distribution, where $\pi_s = P(S_1 = s)$
- $e _s(x)$: emission distribution in state $i$. $e_{i}(x) = P(X_{t} = x\vert S_t = s)$. Note that $x$ can be a vector. For simplicity, we use unbold symbol.

The entire model parameters is $\boldsymbol{\theta} = \left\{ \boldsymbol{P} , \boldsymbol{e} ,\pi \right\}$

Given an observed sequence $\boldsymbol{x_T} = (x_1, x_2, \ldots, x_T)$, the problems are

- Scoring problem: Suppose we know $\boldsymbol{\theta}$, then how to compute the probability of this sequence of observation $P(\boldsymbol{x}_T \vert \boldsymbol{\theta} )$? This can be solved by forward or backward algorithm

- Decoding: What is the most probable underlying state sequence $\boldsymbol{s} = (s_1, \ldots, s_T)$? This can be solved by Viterbi algorithm

- Training (learning): How to estimate the parameters $\boldsymbol{\theta} = \left\{ \boldsymbol{P} , \boldsymbol{e} ,\pi \right\}$? Baum-Welch algorithm (EM applied to HMMs)


To better understand the algorithms above, we introduce two diagrams of state transition. The left graph encodes the transition matrix $\boldsymbol{P}$, and the right trellis shows state transition along time.

:::{figure} markov-transition
<img src="../imgs/markov-transition.png" width = "80%" alt=""/>

Markov transition illustration
:::


### Scoring

Given $\boldsymbol{x}$, to compute $p(\boldsymbol{x} \vert \boldsymbol{\theta} )$, on may attempt to enumerate all possible transitions $\boldsymbol{s}$


$$\begin{aligned}
p(\boldsymbol{x} \vert \boldsymbol{\theta})
&= \sum _ \boldsymbol{s} p( \boldsymbol{x} ,\boldsymbol{s} \vert \boldsymbol{\theta} ) \\
&= \sum _ \boldsymbol{s} p( \boldsymbol{x} \vert \boldsymbol{s} , \boldsymbol{\theta} )p(\boldsymbol{s} \vert  \boldsymbol{\theta} ) \\
&= \sum _ \boldsymbol{s} \left[ e_{s_1}(x_1)e_{s_2}(x_2)\ldots e_{s_T}(x_T) \right] \left[ \pi_{s_1} p_{s_1 s_2} \ldots p_{s_{T-1} s_T} \right]\\
\end{aligned}$$

Clearly, the computation is intractable, since the number of possible state sequences $\boldsymbol{s}$ is $n^T$.


#### Forward Algorithm

Forward algorithm is a dynamic programming algorithm to compute $p(\boldsymbol{x} \vert \boldsymbol{\theta})$.

Define a forward probability, for time $1\le t \le T$, state $1\le s \le n$,

$$
f_t(s) = \operatorname{P}  \left\{ \boldsymbol{x} _{[:t]} = (x_1, \ldots , x_t), S_t = s \vert \boldsymbol{\theta} \right\}
$$

which is the probability of emitting sequence $x_1, \ldots, x_t$ and eventually reaching the state $s$ at time $t$.

We now figure out the iterative relation. At time $t-1$, the sequence is $(x_1, \ldots , x_t)$, and reaches at some state $k$. To arrive $s$ at time $t$, the transition probability is $p_{ks}$. To emit $x_t$, the emission probability is $e_s(x_t)$. Hence, the total probability is $p_{ks} e_s(x_t)$. There are $n$ number of possible states $k$ at time $t-1$. Therefore, the iteration relation is

$$
f_t(s) = \sum_{k=1}^n f_{t-1}(k) p_{ks} e_s(x_t)
$$

Finally, to compute $p(\boldsymbol{x} \vert \boldsymbol{\theta} )$, we look at time $T$, and sum over all states,

$$
p(\boldsymbol{x} \vert \boldsymbol{\theta}) = \sum_ {s=1}^n f_T(s)
$$

In matrix form,

$$\begin{aligned}
\boldsymbol{f}_{t+1} &=  (\boldsymbol{P} ^\top \boldsymbol{f}_t) * \boldsymbol{e}(x_t) \\
p(\boldsymbol{x} \vert \boldsymbol{\theta} )&= \boldsymbol{1} _n ^\top \boldsymbol{f}_T \\
\end{aligned}$$

where $\boldsymbol{f}_t = [f_t(1), \ldots, f_t(n)]^\top , \boldsymbol{e} (x_t) = [e_1(x_t), \ldots, e_n(x_t)] ^\top$ and $*$ stands for element-wise dot product.

---
Forward Algorithm

---

Construct a DP table of size $n\times T$ to store $f_t(s)$. Fill the entries column by column from left to right.

- For $t=1$,
  - for $s = 1, \ldots, n$, compute $f_1(s) = \pi_s e_s (x_1)$

- For $t = 2,\ldots, T$,
  - for $s = 1, \ldots, n$, compute $f_t(s) = \sum_{k=1}^n f_{t-1}(k) p_{ks} e_s(x_t)$

- Return $p(\boldsymbol{x} \vert \boldsymbol{\theta}) = \sum_ {s=1}^n f_T(s)$

---

There are $n\times T$ entries, and each entry takes $O(n)$ to compute. So the total complexity is $O(n^2 T)$, much smaller than the brute force's $O(n^T)$.

#### Backward Algorithm

Backward algorithm is a dynamic programming algorithm to compute $p(\boldsymbol{x} \vert \boldsymbol{\theta})$.

Define a backward probability, for time $1\le t \le T-1$, state $1\le s \le n$,

$$
b_t(s) = \operatorname{P}  \left\{ \boldsymbol{x} _{[-(T-t):]} = (x_{t+1}, \ldots , x_T), S_t = s \vert \boldsymbol{\theta} \right\}
$$

which is the probability of emitting future sequence $x_{t+1}, \ldots, x_T$ and from current state $s$ at time $t$.

We now figure out the iterative relation. At time $t+1$, it reaches some state $k$, and emits $x_{t+1}$. To arrive $k$ at time $t+1$, the transition probability is $p_{sk}$. To emit $x_t$, the emission probability is $e_k(x_{t+1})$. Hence, the total probability is $p_{ks} e_k(x_{t+1})$. There are $n$ number of possible states $k$ at time $t+1$. Therefore, the iteration relation is

$$
b_t(s) = \sum_{k=1}^n p_{sk} e_k(x_{t+1}) b_{t+1}(k)
$$

Finally, to compute $p(\boldsymbol{x} \vert \boldsymbol{\theta} )$, we look at time $1$. The probability of starting from state $s$ is $\pi_s$, and the probability of emitting the first observation $x_1$ is $e_s(x_1)$. Then the probability of emitting the remaining observations $x_2, \ldots, x_T$ given we start from state $s$ is $b_1(s)$. Finally, we sum over all possible starting states $s$.

$$
p(\boldsymbol{x} \vert \boldsymbol{\theta}) = \sum_ {s=1}^n \pi_s e_{s}(x_1) b_1(s)
$$


In matrix form,

$$\begin{aligned}
\boldsymbol{b}_{t} &=  \boldsymbol{P} ( \boldsymbol{e}(x_t) * \boldsymbol{b}_{t+1}) \\
p(\boldsymbol{x} \vert \boldsymbol{\theta} )&=
\boldsymbol{\pi} ^\top  (\boldsymbol{e} (x_1) * \boldsymbol{b} _1) \\
\end{aligned}$$

where $\boldsymbol{b}_t = [b_t(1), \ldots, b_t(n)]^\top , \boldsymbol{e} (x_t) = [e_1(x_t), \ldots, e_n(x_t)] ^\top$ and $*$ stands for element-wise dot product.

---
Backward Algorithm

---

Construct a DP table of size $n \times (T-1)$ to store $b_t(s)$. Fill the entries column by column **from right to left**.

- For $t=T$,
  - for $s = 1, \ldots, n$, initialize $b_t(s) = 1$

- For $t = T-1, T-2, \ldots, 1$,
  - for $s = 1, \ldots, n$, compute $b_t(s) = \sum_{k=1}^n p_{sk} e_k(x_{t+1}) b_{t+1}(k)$

- Return $p(\boldsymbol{x} \vert \boldsymbol{\theta}) = \sum_ {s=1}^n \pi_s e_{s}(x_1) b_1(s)$

---

As in forward algorithm, the complexity is $O(n^2 T)$.


:::{admonition,note} Forward and Backward algorithms

- Either algorithm alone can be used to compute $p(\boldsymbol{x} \vert \boldsymbol{\theta})$, but both $f$ and $b$ will be necessary for solving the training problem.

- The name "forward" and "backward" refer to the order of filling the entries in the DP table. In forward algorithm, we fill the entries by increasing order of $t$, so we call it "forward". In backward algorithm, we fill the entries by decreasing order of $t$, so we call it "backward".

:::




### Decoding

Recall the problem: given an observation sequence $\boldsymbol{x}$, what is the most probable underlying state sequence $\boldsymbol{s} = (s_1, \ldots, s_T)$?

One attempt is to choose the individually most probable state by

$$s^*_t = \arg\max_s P(S_t = s \vert \boldsymbol{x} , \boldsymbol{\theta} )$$

where $P(S_t = s \vert \boldsymbol{x} , \boldsymbol{\theta} ) = \frac{f_t(s)b_t(s)}{p(\boldsymbol{x} \vert \boldsymbol{\theta} )}$, and return $\boldsymbol{s} ^* = (s^*_1, \ldots, s^*_T)$. This “individually most likely” criterion maximizes the expected number of correct states. But clearly it does not consider the dependence among sequence from time to time, and may give a sequence that is totally impossible.

We should consider the sequence jointly. The correct criterion should be

$$
\boldsymbol{s} ^* = \underset{\boldsymbol{s} }{\operatorname{argmax}}\, p(\boldsymbol{s} \vert \boldsymbol{x} , \boldsymbol{\theta} )
$$

Viterbi Algorithm can solve this problem.

#### Viterbi Algorithm

Define probability $v_t(s)$ as the probability of the most probable state path for the observation sequence $(x_1, x_2, \ldots, x_t)$ ending in state $s$ at time $t$.

$$
v_t(s) = \max _{s_1, \ldots, s_{t-1}} \operatorname{P} \left\{ \boldsymbol{x} _{[:t]} =  (x_1, x_2, \ldots, x_t), S_t = s \vert \boldsymbol{\theta} \right\}
$$

We now figure out the iterative relation. 

$$
v_t(s) = e_s(x_t) \max _{1 \le k \le n } v_{t-1}(k) p_{ks}
$$


### Training

ML to find for $\lambda$.

If states is given, then ML is easy by counting. Similar like Gaussian.

But the states are not given, so we provide initial guess, and iteratively update $\lambda$.

Baum-Welch Algorithm: EM for HMMs






## Applications

HMM can be applied to anything that has "state" and "sequence" attributes.

**Automatic speech recognition**

- One HMM per word or phoneme
- Time index corresponds to a 10ms “frame”
- Observation is a vector of spectral (frequency) measurements
- Can think of HMM state as corresponding to a state of the speaker’s vocal tract

**Unsupervised speech unit (Word/Phoneme) Discovery**

- Learn an HMM to model a collection of unlabelled speech
- Group together frequently occurring sequences of states to define units

**Activity recognition in video or biometrics**

- States corresponds to pose
- As in speech recognition, the “activity” can be labeled or unlabeled

**Speech Tagging**

:::{figure} hmm-speech-tagging
<img src="../imgs/hmm-speech-tagging.png" width = "70%" alt=""/>

HMM in part-of-speech tagging [Julia Hockenmeyer]
:::


## Hidden Topic Markov Models

Topic states.

## Model Selection

Perplexity












.
