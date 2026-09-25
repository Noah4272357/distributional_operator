Suppose your historical data are

$$
\{(P_X^{(k)},\,P_Y^{(k)})\}_{k=1}^K,
$$

where each target is Gaussian,

$$
P_Y^{(k)}=\mathcal N(\mu_k,\Sigma_k),
$$

and for a new input distribution \(P_X^\ast\), you want to infer \(P_Y^\ast\) by interpolating nearby historical examples rather than training a large neural network.

First define a distance between input distributions:

$$
d(P_X^\ast,P_X^{(k)}).
$$

use $W_2$ as $d$ and Kernel regression in distribution space

For each historical input distribution, compute

$$
d_k=d(P_X^\ast,P_X^{(k)}).
$$

Define weights such as

$$
w_k=
\frac{
\exp(-d_k^2/2h^2)
}{
\sum_j\exp(-d_j^2/2h^2)
}.
$$

Then

$$
\sum_k w_k=1.
$$

This is essentially a **Nadaraya-Watson kernel regression**, except that the input variable is a probability distribution rather than an ordinary vector.

Conceptually:

$$
\boxed{
P_X^\ast
\xrightarrow{\text{distance to history}}
\{d_k\}
\xrightarrow{\text{kernel}}
\{w_k\}
\xrightarrow{\text{interpolation}}
P_Y^\ast.
}
$$
