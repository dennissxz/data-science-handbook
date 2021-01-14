# Huffman Codes

## Problem

Input
- Text $T$, containing $n$ characters, defined over alphabet $\Sigma$.

Definition:

an encoding $c$ of alphabet $\Sigma$ is a mapping


$$
c: \Sigma \rightarrow \cup_{i\ge 1} \left\{ ?? \right\}
$$

if $T = (c_1, c_2, \ldots, c_n)$, encoding of $T$ is
$$
c(T) = c()
$$

We say that code $c$ has **unique decoding property** iff for only two distinct finite length strings $T_1 \ne T_2$ then $c(T_1) \ne c(T_2)$ .

Ideally: want to decode correctly and efficiently.

For instance, to encode English, $\left\vert \Sigma \right\vert$ we need string length of 5. And the frequency differs. (ASCII)

We want high-frequency letter with shorter string and low-frequency letter with longer string
