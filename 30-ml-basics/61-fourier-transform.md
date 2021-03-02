# Fourier Transform-based Representations

Why study Fourier transform in ML?

- Staple of feature representation for many types of data (audio, images, others) for many decades, still useful in many settings
- Also useful for deriving new algorithms in machine learning

Before we do any learning, we start with some data representation. Typically, this involves converting raw data into feature vectors. These days, much of feature engineering is replaced by neural network-based feature learning. But not all... choice of initial representation can still be important
- Visualization of raw data
- Qualitative understanding of the data
- Better input to a learning algorithm
- When number of observations is small, handcraft features are also useful

## Basics

Main idea
: Decompose a signal $x(t)$ as a sum of multiple sinusoids at different frequencies

  $$x(t)=\sum_{k} a_{k} f_{k}(t)$$

  where

  - Signal = function of a discrete “time” variable $n$
  - $[]$ notation reminds us that the time variable is discrete
  - $f_k[n]$ sinusoid (sine/cosine function) at frequency indexed by $k$
  - $a_k =$ “amount” of frequency $k$ present in $x[n]$, aka "**spectrum**" of the signal (but this word has other meanings).
  - Fourier transform algorithms: ways of finding $a_k$ given $x[n]$

Demo: http://www.falstad.com/dfilter/

:::{figure} spectral-clustering-speech-sep.png
<img src="../imgs/spectral-clustering-speech-sep.png" width = "50%" alt=""/>


:::




It is possible to automatically learn features from large training sets of raw speech signals, but they are not much better than starting from spectral features (for now!)

Other examples:
Financial market data
Weather data
Medical imaging, other scientific imaging Image compression (e.g. JPEG)

ear

Fourier methods include:
- Discrete-time and continuous-time Fourier series
- Discrete-time and continuous-time Fourier transforms
- Discrete Fourier transform (most common for digital signals)

Applications to machine learning
- Feature extraction, compression and de-noising of speech/images: Can be an important precursor to unsupervised (or supervised) learning
- Approximating kernels [Rahimi & Recht 2007]
- Speeding up convolutional neural networks [Mathieu et al. 2013]
- Analyzing and regularizating neural networks [Aghazadeh et al. 2020]

## Discrete Fourier Transform (DFT)

We often start with a very long signal and compute its spectrum over sub-sequences (“windows”) of fixed length

$$
x[t], x[t+1], \ldots x[t+N-1]
$$

For a signal of length $N$ samples starting at sample $t$, the discrete Fourier transform (DFT) is given by:

$$
X[k]=\sum_{n=t}^{t+N-1} x[n] e^{-j 2 \pi k n / N}, \quad k=0, \ldots, N-1
$$

- $X[k]$ is the value of the spectrum at the $k$-th frequency
- Euler's relation $e^{j a}=\cos (a)+j \sin (a)$
- Equivalently, we can consider $k = −N/2,...,0,...,N/2$
- $X[k]$ is in general complex-valued; often only the real part of it is used. (Why complex numbers? Real-world signals are real... but complex signals are often much easier to analyze)

### Complex Sinusoids

$$
x(t)=e^{j \omega_{0} t}=\cos \left(\omega_{0} t\right)+j \sin \left(\omega_{0} t\right)
$$

Can think of this as a complex number vector spinning around the origin in the complex plane
- Fundamental frequency ω0 radians/sec, period $T_0 = 2π/|ω_0|$
seconds
• (since $ejω_0(t+2π/|ω_0|) = ejω_0te±j2π = ejω_0t$)
- Fundamental frequency in Hertz $f_0 = ω_0/2π = 1/T_0$

### Discrete-time complex sinusoids

If we sample the continuous-time sinusoid x(t) = ejω0t at sampling rate f = 1/T, we get

$$
x[n]=e^{j \omega_{0} n T}=\cos \left(\omega_{0} n T\right)+j \sin \left(\omega_{0} n T\right)
$$

Note: Given a sampled signal, e.g. x[n] = cos( π2 n), we can’t talk about frequency in Hertz since discrete time has no units — the same discrete-time signal can correspond to many continuous-time signals!

If we know the sampling rate is f = 1/T = 8000 Hertz, then we can say that this discrete-time signal is the sampled version of
x(t) = cos(8000 π2 t) with fundamental frequency of ω0 = 4000π radians/sec or f0 = ω0/2π = 2000 Hertz

### Back

Start with a windowed signal x[n] over the window $t,\ldots,t+N −1$
We can think of it as one period of a periodic signal with fundamental period N (fundamental frequency $ω0 = 2π/N$), i.e. $x[n + N] = x[n] ∀n$

We would like to find coefficients $a_k$ such that

$$
x[n]=\sum_{k} a_{k} e^{j k \omega_{0} n}
$$

which is called the synthesis equation.

For historical reasons we will then define the DFT as $X[k] = Na_k$. How do we compute the $a_k$? (next)

Note: There are only N distinct complex sinusoids with fundamental period $N$:

$$
e^{j(k+N) \omega_{0} n}=e^{j k \omega_{0} n} e^{j \overbrace{N \omega_{0} n}^{2 \pi n}}=e^{j k \omega_{0} n}
$$

So we need only sum over any N consecutive terms

$$
\begin{aligned}
x[n] &=\sum_{k=<N>} a_{k} e^{j k \omega_{0} n} \\
&=\sum_{k=0}^{N-1} a_{k} e^{j k \omega_{0} n} \\
&=\sum_{k=-3}^{N-4} a_{k} e^{j k \omega_{0} n}
\end{aligned}
$$

Hence

$$
\begin{aligned}
x[n] &=\sum_{k=<N>} a_{k} e^{j k \omega_{0} n} \text { (synthesis equation) } \\
\Rightarrow x[0] &=\sum_{k=<N>} a_{k} \\
x[1] &=\sum_{k=<N>} a_{k} e^{j k \omega_{0}} \\
x[2] &=\sum_{k=<N>} a_{k} e^{j 2 k \omega_{0}} \\
& \vdots \\
x[N-1] &=\sum_{k=<N>} a_{k} e^{j(N-1) k \omega_{0}}
\end{aligned}
$$

N equations with N unknowns a0, a1, . . . , aN−1
In principle, we could solve the N linear equations for ak

A more direct approach: to compute am, multiply both sides of
synthesis equation by e−jmω0n and then sum over n =< N >:


$$
\begin{aligned}
x[n] &=\sum_{k=<N>} a_{k} e^{j k \omega_{0} n} \text { (synthesis equation) } \\
\Longrightarrow \sum_{n=<N>} x[n] e^{-j m \omega_{0} n} &=\sum_{n=<N>}\left(\sum_{k=<N>} a_{k} e^{j k \omega_{0} n}\right) e^{-j m \omega_{0} n} \\
&=\sum_{k=<N>} a_{k} \underbrace{\sum_{n=<N>}}_{=N \delta[k-m] \text { (orthogonality) }} \\
&=N a_{m}
\end{aligned}
$$

(Orthogonality can be shown via basic properties of summations) Can think of DFT as the projection of the signal onto the orthogonal basis of complex sinusoids. (cf. linear regression)


$$
\begin{array}{c}
x[n]=\sum_{k=<N>} a_{k} e^{j k \omega_{0} n} \quad \text { (Synthesis equation) } \\
a_{k}=\frac{1}{N} \sum_{n=<N>} x[n] e^{-j k \omega_{0} n} \quad \text { (Analysis equaiton) }
\end{array}
$$

- ω0 = 2π/N
- Notation: Sometimes written x[n] ↔ ak
- Convenient to think of ak as being defined for all k, although we only need a subset of N of them: ak+N = ak
- Note: Since x[n] is periodic, it is specified uniquely by only N numbers, either in time or in frequency domain

The DFT is defined as X[k] = Nak. We now return to considering a specific frame (window) of N samples t,...,t+N −1:


$$
X[k]=\sum_{n=t}^{t+N-1} x[n] e^{-j 2 \pi k n / N}, \quad k=0, \ldots, N-1
$$

The fast Fourier transform (FFT) is an algorithm used to compute
The DFT is also called the spectrum
DFT for window length M = 2m for some m
After doing this for all frames (windows) of a signal, the result is a
spectrogram
X[k] is in general complex-valued, but we often use only its magnitude or phase

Note on units:
- Each time sample n corresponds to a time in seconds
- Each frequency sample k corresponds to a frequency $f(k) = \frac{k}{N} R$ in Hz, where R is the sampling rate.

Important property: Spectra of real signals are conjugate-symmetric
- Magnitude is symmetric about k = 0 (equivalently about N/2)
- Phase is anti-symmetric about k = 0 (equivalently about N/2)
- So we need only think about positive frequencies

### Examples


$$
\begin{aligned}
x[n] &=\cos \left(\frac{\pi}{4} n\right) \\
&=\frac{1}{2}\left(e^{j \pi n / 4}+e^{-j \pi n / 4}\right) \\
\Longrightarrow \omega_{0} &=\pi / 4, N=8, a_{1}=a_{-1}=1 / 2, X[1]=X[-1]=8 \frac{1}{2}=4
\end{aligned}
$$


$$
\begin{aligned}
x[n] &=\sin \left(\frac{\pi}{4} n\right)^{\star} \\
&=\frac{1}{2 j}\left(e^{j \pi n / 4}-e^{-j \pi n / 4}\right) \\
\Longrightarrow \omega_{0} &=\pi / 4, N=8, a_{1}=\frac{1}{2 j}, a_{-1}=-\frac{1}{2 j}
\end{aligned}
$$


## Others

### Continuous-time Fourier Transform

$$
\begin{array}{l}
x(t)=\frac{1}{2 \pi} \int_{-\infty}^{\infty} X(j \omega) e^{j \omega t} d \omega \quad \text { Synthesis equation } \\
X(j \omega)=\int_{-\infty}^{\infty} x(t) e^{-j \omega t} d t \quad \text { Analysis equation }
\end{array}
$$


### 2-D discrete-“time” Fourier series/transforms


$$
\begin{aligned}
X_{k l}=& \frac{1}{M N} \sum_{<N>} \sum_{<M>} x[m, n] e^{-j 2 \pi(n k / N+m l / M)} \\
& \text{where }  0 \leq k \leq N-1,0 \leq l \leq M-1 \\
x[m, n]=& \sum_{k=0}^{N-1} \sum_{l=0}^{M-1} X_{k l} e^{j 2 \pi(n k / N+m l / M)}
\end{aligned}
$$

Equivalent to 1-D transforms when one frequency dim is **fixed** 2-D fast Fourier transform requires $M N (\log _{2} M) (\log _{2} N)$ operations.

### 2-D Discrete-"time" convolution

$$
y\left[n_{1}, n_{2}\right]=x\left[n_{1}, n_{2}\right] * h\left[n_{1}, n_{2}\right]=\sum_{k_{1}=-\infty}^{\infty} \sum_{k_{2}=-\infty}^{\infty} x\left[k_{1}, k_{2}\right] h\left[n_{1}-k_{1}, n_{2}-k_{2}\right]
$$

This is the operation being done in convolutional neural networks, on the image $x$ and the filter $h$.


- But we typically don’t bother with flipping the filter and state it as a dot product
- The properties of convolution tell us




.









































.
