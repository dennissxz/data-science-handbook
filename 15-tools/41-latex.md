# LaTeX

## Symbols

Note the difference between

- $v\quad$ v
- $\nu\quad$ \nu
- $\upsilon\quad$ \upsilon


Greeks, arrows, symbols, see [here](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols).



## Alignment

In short, odd-indexed `&` are used before `=` to align them, and even-indexed `&` are necessary to separate the columns.

- 1 column, with 1 `&` in each line before `=`.

  ```latex
  \begin{align}
    x^2  + y^2 & = 1                       \\
    x          & = \sqrt{1-y^2}
  \end{align}
  ```

  $$
  \begin{align}
    x^2  + y^2 & = 1                       \\
    x          & = \sqrt{1-y^2}
  \end{align}
  $$

- 2 columns of equations, with 3 `&` in each line: 2 before `=` and 1 in-between columns.

  ```latex
  \begin{align}    \text{Compare}
    x^2 + y^2 &= 1              &    x^3 + y^3 &= 1               \\
    x         &= \sqrt{1-y^2}   &            x &= \sqrt[3]{1-y^3}
  \end{align}
  ```

  $$
  \begin{align}    \text{Compare}
    x^2 + y^2 &= 1              &    x^3 + y^3 &= 1               \\
    x         &= \sqrt{1-y^2}   &            x &= \sqrt[3]{1-y^3}
  \end{align}
  $$

- 1 column of equation and 1 column of text, use `&&`.


  ```latex
  \begin{align}
    x      &= y      && \text{by hypothesis} \\
        x' &= y'     && \text{by definition} \\
    x + x' &= y + y' && \text{by Axiom 1}
  \end{align}
  ```


  $$
  \begin{align}
    x      &= y      && \text{by hypothesis} \\
        x' &= y'     && \text{by definition} \\
    x + x' &= y + y' && \text{by Axiom 1}
  \end{align}
  $$

- 3 columns of equations, 5 `&` in each line: 3  before `=` and 2 in-between columns.

  ```latex
  \begin{align}
      x    &= y      &     X  &= Y      &   a  &= b+c      \\
      x'   &= y'     &     X' &= Y'     &   a' &= b        \\
    x + x' &= y + y' & X + X' &= Y + Y' &  a'b &= c'b
  \end{align}
  ```

  $$
  \begin{align}
      x    &= y      & X  &= Y  &     a  &= b+c                 \\
      x'   &= y'     & X' &= Y' &     a' &= b                   \\
    x + x' &= y + y'            & X + X' &= Y + Y' & a'b &= c'b
  \end{align}
  $$

- 2 columns, but different number of rows


  ```latex
  \begin{equation}
    \begin{aligned}
      x^2 + y^2  &= 1                 \\
      x          &= \sqrt{1-y^2}      \\
     \text{and also }y &= \sqrt{1-x^2}
    \end{aligned}               
  \qquad
    \begin{aligned}
     (a + b)^2 &= a^2 + 2ab + b^2       \\
     (a + b) \cdot (a - b) &= a^2 - b^2
    \end{aligned}      
  \end{equation}
  ```

  $$
  \begin{equation}
    \begin{aligned}
      x^2 + y^2  &= 1                   \\
      x          &= \sqrt{1-y^2}        \\
     \text{and also }y &= \sqrt{1-x^2}
    \end{aligned}               
  \qquad
    \begin{aligned}
     (a + b)^2 &= a^2 + 2ab + b^2       \\
     (a + b) \cdot (a - b) &= a^2 - b^2
    \end{aligned}      
  \end{equation}
  $$

- Similar example as above, with `[b]` (move upward) and `[t]` (move downward)


  ```latex
  \begin{equation}
  \begin{aligned}[b]
    x^2 + y^2  &= 1                 \\
    x          &= \sqrt{1-y^2}      \\
   \text{and also }y &= \sqrt{1-x^2}
  \end{aligned}               \qquad
  \begin{gathered}[t]
   (a + b)^2 = a^2 + 2ab + b^2      \\
   (a + b) \cdot (a - b) = a^2 - b^2
  \end{gathered}
  \end{equation}
  ```

  $$
  \begin{equation}
  \begin{aligned}[b]
    x^2 + y^2  &= 1                 \\
    x          &= \sqrt{1-y^2}      \\
   \text{and also }y &= \sqrt{1-x^2}
  \end{aligned}               \qquad
  \begin{gathered}[t]
   (a + b)^2 = a^2 + 2ab + b^2      \\
   (a + b) \cdot (a - b) = a^2 - b^2
  \end{gathered}
  \end{equation}
  $$
