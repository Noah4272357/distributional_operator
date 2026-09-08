## heatmap: aggregate error over all test cases

Create covariance heatmap 

Suppose you have \(K\) test covariance distributions:

$$
\Sigma^{(1)},\ldots,\Sigma^{(K)}
$$

and predictions

$$
\hat\Sigma^{(1)},\ldots,\hat\Sigma^{(K)}.
$$

Calculate the elementwise mean absolute error:

$$
E_{ij}
=
\frac1K
\sum_{k=1}^K
\left|
\hat\Sigma^{(k)}_{ij}
-
\Sigma^{(k)}_{ij}
\right|.
$$

Then plot the \(4\times 4\) matrix

$$
E.
$$

Now the heatmap answers:

> **Which covariance components are systematically difficult for the model?**

