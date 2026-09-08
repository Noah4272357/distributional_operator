

###  Heatmap across many test examples

Let rows be test samples and columns be the 48 time bins:

Randomly choose 50 samples from test dataset, i.e. N=50.
$$
P=
\begin{bmatrix}
p_{11}&\cdots&p_{1,48}\\
\vdots&&\vdots\\
p_{N1}&\cdots&p_{N,48}
\end{bmatrix}.
$$

Make two heatmaps side-by-side:

```text
        Ground truth                    Prediction

time →                              time →
      ████                                ████
         █████                               █████
              ███                                ███
   █████                              █████
           █████                              █████
sample ↓                            sample ↓
```

Each pixel represents the probability of a spike occurring in that bin.



For the 49th no-spike probability, add a narrow separate column on the right:

```text
| bin 1 ... bin 48 | No spike |
```

