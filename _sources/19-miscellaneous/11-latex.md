
# LaTeX


## Alignment


- One column, with one `&` in each line

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

- Two column-pairs, with three `&` in each line

  ```latex
  \begin{align}    \text{Compare}
    x^2 + y^2 &= 1               &
    x^3 + y^3 &= 1               \\
    x         &= \sqrt   {1-y^2} &
    x         &= \sqrt[3]{1-y^3}
  \end{align}
  ```

  $$
  \begin{align}    \text{Compare }
    x^2 + y^2 &= 1               &
    x^3 + y^3 &= 1               \\
    x         &= \sqrt   {1-y^2} &
    x         &= \sqrt[3]{1-y^3}
  \end{align}
  $$

- Another example of two column-pairs


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

- Three column-pairs, with four `&` in each line


  ```latex
  \begin{align}
      x    &= y      & X  &= Y  &     a  &= b+c                 \\
      x'   &= y'     & X' &= Y' &     a' &= b                   \\
    x + x' &= y + y'            & X + X' &= Y + Y' & a'b &= c'b
  \end{align}
  ```

  $$
  \begin{align}
      x    &= y      & X  &= Y  &     a  &= b+c                 \\
      x'   &= y'     & X' &= Y' &     a' &= b                   \\
    x + x' &= y + y'            & X + X' &= Y + Y' & a'b &= c'b
  \end{align}
  $$

- This example has two column-pairs, different number of rows


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

- Two columns, with `[b]` (move upward) and `[t]` (move downward)


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
