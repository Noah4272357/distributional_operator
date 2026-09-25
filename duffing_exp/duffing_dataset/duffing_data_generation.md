## 1. Fixed Duffing system

Use

$$
\ddot Y(t)+c\dot Y(t)+aY(t)+bY(t)^3=X(t),
\qquad
Y(0)=\dot Y(0)=0.
$$

I recommend initially using the classical hardening Duffing regime

$$
\boxed{a=1,\qquad b=1,\qquad c=0.2.}
$$

Thus,

$$
\boxed{
\ddot Y+0.2\dot Y+Y+Y^3=X(t).
}
$$

These values give clearly nonlinear behavior without making the numerical problem unnecessarily difficult.

The coefficient \(b=1\) is important: you want the output law to demonstrate a genuinely nonlinear pushforward. If \(b\) is too small, the experiment becomes close to a linear stochastic oscillator, which is less compelling.

I would **not vary \(a,b,c\) anywhere in the primary dataset**.

---

# 2. Forcing stochastic process

Use

$$
X(t)=A\sin(2\pi ft+\Phi)+\eta(t),
$$

which is the same basic forcing family as in your original design.

The critical distinction is:

* \(A,f,\sigma_\eta,\ell_\eta\) characterize a probability measure \(\mu\);
* \(\Phi\) and the realization of \(\eta\) are sampled independently for each trajectory.

In particular,

$$
\boxed{\Phi\sim U(0,2\pi)}
$$

should **not** be regarded as a dataset-level parameter.

For a fixed \(\mu\), every realization gets a new random phase.

---

# 3. Recommended parameter ranges

Define a stochastic-process distribution by

$$
\theta_\mu=(A,f,\sigma_\eta,\ell_\eta).
$$

I recommend starting with

$$
\boxed{
A\in[0.5,2.0],
\qquad
f\in[0.10,0.30],
}
$$

and

$$
\boxed{
\sigma_\eta\in[0.05,0.40],
\qquad
\ell_\eta\in[0.10,1.00].
}
$$

For every realization,

$$
\boxed{\Phi\sim U(0,2\pi).}
$$

A useful default parameter table is:

| Parameter       | Role                       | Recommended setting |
| --------------- | -------------------------- | ------------------: |
| \(a\)           | linear restoring force     |      \(1.0\), fixed |
| \(b\)           | nonlinear restoring force  |      \(1.0\), fixed |
| \(c\)           | damping                    |      \(0.2\), fixed |
| \(A\)           | periodic forcing amplitude |      \(U(0.5,2.0)\) |
| \(f\)           | forcing frequency          |    \(U(0.10,0.30)\) |
| \(\Phi\)        | realization phase          |       \(U(0,2\pi)\) |
| \(\sigma_\eta\) | GP noise magnitude         |    \(U(0.05,0.40)\) |
| \(\ell_\eta\)   | GP correlation length      |    \(U(0.10,1.00)\) |



---

# 4. Gaussian-process component

A convenient choice is a squared-exponential covariance:

$$
\eta\sim GP(0,K),
$$

with

$$
K(s,t)
=
\sigma_\eta^2
\exp\left(
-\frac{(s-t)^2}{2\ell_\eta^2}
\right).
$$

Thus

$$
X(t)
=
A\sin(2\pi ft+\Phi)+\eta(t).
$$

This gives you two distinct forms of uncertainty:

$$
\underbrace{\Phi}_{\text{global phase uncertainty}}
\qquad+\qquad
\underbrace{\eta(t)}_{\text{local correlated stochastic variation}}.
$$

That is more interesting than either component alone.


---

# 5. Time domain and grid

Use

$$
\boxed{t\in[0,20].}
$$

For model input/output, store

$$
\boxed{n_t=256}
$$

equally spaced values:

$$
t_k=\frac{20k}{255},
\qquad k=0,\ldots,255.
$$

So every trajectory is represented as a vector

$$
X=(X(t_0),\ldots,X(t_{255}))
\in\mathbb R^{256}.
$$

Likewise,

$$
Y\in\mathbb R^{256}.
$$

Therefore an empirical input measure with \(N\) samples has shape

$$
(N,256).
$$

For a batch of dataset instances:

$$
\boxed{
(\text{batch},N,256).
}
$$

This aligns nicely with a set encoder or distribution encoder.

---

# 6. Do not solve the ODE directly on only 256 points

This distinction matters.

Use a **much finer internal grid for simulation**, for example

$$
\Delta t_{\rm solve}=0.005
$$

or

$$
0.01.
$$

For

$$
T=20,
$$

that gives approximately \(2001\)–\(4001\) internal time points.

Then downsample the solution to the 256 model grid points.

So use:

$$
\boxed{
\text{fine simulation grid}
\rightarrow
\text{accurate solution}
\rightarrow
\text{256-point observation grid}.
}
$$

Otherwise discretization error can contaminate the target distribution.

---

# 7. Solver

Rewrite the second-order equation as

$$
\begin{aligned}
\dot y_1 &=y_2,\\
\dot y_2 &=X(t)-cy_2-ay_1-by_1^3.
\end{aligned}
$$

Then solve

$$
\frac{d}{dt}
\begin{pmatrix}
y_1\\y_2
\end{pmatrix}
=
\begin{pmatrix}
y_2\\
X(t)-cy_2-ay_1-by_1^3
\end{pmatrix}.
$$

For dataset generation I recommend an adaptive high-order method such as

$$
\boxed{\text{Dormand--Prince RK45}}
$$

with approximately

$$
\texttt{rtol}=10^{-8},
\qquad
\texttt{atol}=10^{-10}.
$$

In SciPy this corresponds naturally to `scipy.integrate.solve_ivp`.

Because \(X(t)\) is available only through a generated realization, create an interpolant

$$
\widetilde X(t)
$$

on the fine forcing grid and evaluate that inside the ODE right-hand side.



---

# 8. What constitutes one dataset example?

This is the most important organizational point.

One training example should correspond to **one probability measure** \(\mu_j\), not one trajectory.

First draw

$$
\theta_j
=
(A_j,f_j,\sigma_j,\ell_j).
$$

This defines

$$
\mu_j
=
P_X^{\theta_j}.
$$

Then generate the empirical input ensemble

$$
X_{j,1},\ldots,X_{j,N_{\rm in}}
\overset{iid}{\sim}\mu_j.
$$

Independently generate

$$
X'_{j,1},\ldots,X'_{j,N_{\rm out}}
\overset{iid}{\sim}\mu_j
$$

and propagate them:

$$
Y_{j,i}
=
\mathcal S(X'_{j,i}).
$$

Thus one example is

$$
\boxed{
\left(
\{X_{j,i}\}_{i=1}^{N_{\rm in}},
\{Y_{j,i}\}_{i=1}^{N_{\rm out}}
\right).
}
$$

There should be **no trajectory-level correspondence** between the two sets. This preserves the distribution-level nature of your original task.

---

# 9. Recommended ensemble size

I would initially use

$$
\boxed{
N_{\rm in}=N_{\rm out}=128.
}
$$

Therefore one training example has

$$
X_j\in\mathbb R^{128\times256},
$$

and

$$
Y_j\in\mathbb R^{128\times256}.
$$

This is large enough to give the model meaningful information about \(\mu_j\), while remaining manageable computationally.


This directly tests a theoretically meaningful question:

> How accurately can the distribution-to-distribution operator be approximated from a finite empirical representation of the input measure?

That experiment fits your theoretical motivation particularly well.

---

# 10. Number of probability measures

A reasonable first serious dataset is

$$
\boxed{
N_{\rm measure}=1400.
}
$$

Use:

$$
\begin{aligned}
N_{\rm train}&=1000,\\
N_{\rm val}&=200,\\
N_{\rm test}&=200.
\end{aligned}
$$

Each measure has 128 target ODE solves, giving approximately

$$
1400\times128=89600
$$

Duffing simulations.

That is quite feasible and sufficiently large for a controlled neural-network experiment.


---

# 11. Do not randomly split trajectories

Split at the **measure level**.

For example:

$$
\theta_1,\ldots,\theta_{1000}
\rightarrow\text{train},
$$

$$
\theta_{1001},\ldots,\theta_{1200}
\rightarrow\text{validation},
$$

$$
\theta_{1201},\ldots,\theta_{1400}
\rightarrow\text{test}.
$$

Never put samples from the same \(\mu_j\) into both training and test data.

Otherwise you would partly test interpolation between empirical samples from an already-seen distribution rather than generalization to an unseen stochastic process.

---

# 12. How to sample the parameter space

I would **not use naive IID uniform sampling** for the 1400 parameter configurations.

Use a space-filling design such as

$$
\boxed{\text{Latin Hypercube Sampling}}
$$

over

$$
(A,f,\sigma_\eta,\ell_\eta).
$$

It gives much better coverage of the four-dimensional parameter domain.

For example:

$$
\Theta
=
[0.5,2]
\times
[0.10,0.30]
\times
[0.05,0.40]
\times
[0.10,1].
$$

Generate 1400 Latin-hypercube points in \(\Theta\).

This is particularly useful because the physical response can change sharply near resonance. Random uniform sampling may leave important regions sparsely represented.

---

# 13. Stronger train/test design

For a paper, I would actually construct **two test sets**.

### Test set A: interpolation

Test parameters lie inside the training domain:

$$
\theta_{\rm test}\in\Theta.
$$

But none of the exact parameter tuples have appeared in training.

This measures ordinary interpolation/generalization.

### Test set B: controlled extrapolation

Train on, for example,

$$
A\in[0.6,1.8],
\qquad
f\in[0.12,0.27],
$$

then test near the boundaries:

$$
A\in[0.5,0.6]\cup[1.8,2.0]
$$

and/or

$$
f\in[0.10,0.12]\cup[0.27,0.30].
$$

Then you can distinguish

$$
\text{interpolation across probability laws}
$$

from

$$
\text{extrapolation across probability laws}.
$$

For a theory-driven paper, that distinction is valuable.

---

# 14. Exact realization-generation algorithm

For each measure \(j\):

### Step 1 — sample measure-level parameters

Generate

$$
(A_j,f_j,\sigma_j,\ell_j).
$$

These remain fixed for every realization belonging to \(\mu_j\).

---

### Step 2 — generate one forcing realization

For each realization \(i\), independently sample

$$
\Phi_{j,i}\sim U(0,2\pi).
$$

Generate

$$
\eta_{j,i}
\sim
GP\left(
0,
K_{\sigma_j,\ell_j}
\right).
$$

Then

$$
X_{j,i}(t)
=
A_j\sin(2\pi f_jt+\Phi_{j,i})
+
\eta_{j,i}(t).
$$

Repeat independently \(N_{\rm in}\) times.

---

### Step 3 — generate a separate target forcing ensemble

Do **not reuse** those \(X_{j,i}\).

Instead generate another IID ensemble

$$
X'_{j,i}\sim\mu_j.
$$

Again sample completely new

$$
\Phi'_{j,i}
$$

and

$$
\eta'_{j,i}.
$$

---

### Step 4 — solve Duffing

For every \(X'_{j,i}\), solve

$$
\ddot Y_{j,i}
+
0.2\dot Y_{j,i}
+
Y_{j,i}
+
Y_{j,i}^3
=
X'_{j,i}(t)
$$

with

$$
Y_{j,i}(0)=0,\qquad
\dot Y_{j,i}(0)=0.
$$

Save

$$
Y_{j,i}(t_k),
\qquad k=1,\ldots,256.
$$

---

# 15. Dataset tensor structure

I recommend storing:

$$
\boxed{
X:
(N_{\rm measure},N_{\rm in},N_t)
}
$$

and

$$
\boxed{
Y:
(N_{\rm measure},N_{\rm out},N_t).
}
$$

With the proposed settings:

$$
X:
(1400,128,256),
$$

$$
Y:
(1400,128,256).
$$

Also save

$$
\theta:
(1400,4)
$$

containing

$$
(A,f,\sigma_\eta,\ell_\eta).
$$

Importantly, **do not necessarily give \(\theta\) to your neural network**.

Keep it primarily for:

* diagnosing model performance;
* plotting errors versus \(A,f,\sigma,\ell\);
* constructing test subsets;
* verifying coverage of parameter space.

The actual model can still receive only

$$
\{X_i\}.
$$

That keeps the problem genuinely distribution based.

---

# 16. Normalization

Be careful here because you are predicting probability laws.

I recommend calculating global training-set statistics

$$
m_X,\quad s_X,\quad
m_Y,\quad s_Y
$$

and using

$$
\widetilde X
=
\frac{X-m_X}{s_X},
\qquad
\widetilde Y
=
\frac{Y-m_Y}{s_Y}.
$$

Do **not normalize every stochastic-process ensemble independently** to zero mean/unit variance.

Doing that could remove precisely the distributional information the model needs to distinguish different \(\mu\)'s.

---

# 17. Numerical validation before generating the full dataset

Before producing 896,00 trajectories, perform a convergence test.

Select perhaps 20 parameter configurations and solve using

$$
\Delta t=
0.02,\quad0.01,\quad0.005,\quad0.0025
$$

or compare RK4 against high-accuracy RK45.

Check

$$
\frac{
\|Y_{\Delta t}-Y_{\rm reference}\|_2
}{
\|Y_{\rm reference}\|_2
}.
$$

I would require approximately

$$
\boxed{
\text{relative numerical error}<10^{-4}
}
$$

for the final solver.

Then your ML error should dominate numerical discretization error.

---

# 18. Sanity checks on the generated dataset

Before training, plot several examples of:

$$
\{X_i(t)\}_{i=1}^{128}
$$

and

$$
\{Y_i(t)\}_{i=1}^{128}.
$$

For several parameter configurations also calculate

$$
\mathbb E[X(t)],\qquad
\operatorname{Std}[X(t)]
$$

and

$$
\mathbb E[Y(t)],\qquad
\operatorname{Std}[Y(t)].
$$

Check that changing \(A,f,\sigma,\ell\) produces noticeably different output distributions.

In particular, inspect cases:

$$
f<0.159,\qquad
f\approx0.159,\qquad
f>0.159.
$$

You should see qualitatively different oscillator behavior around the resonance region.

---

# 19. A compact recommended configuration

If you want one concrete configuration to implement immediately, I would use:

$$
\boxed{
\begin{aligned}
&a=1,\quad b=1,\quad c=0.2,\\
&T=20,\quad N_t=256,\\
&A\sim U(0.5,2.0),\\
&f\sim U(0.10,0.30),\\
&\sigma_\eta\sim U(0.05,0.40),\\
&\ell_\eta\sim U(0.10,1.0),\\
&\Phi\sim U(0,2\pi),\\
&\eta\sim GP(0,K_{\rm RBF}),\\
&N_{\rm in}=128,\quad N_{\rm out}=128,\\
&N_{\rm train}=1000,\\
&N_{\rm val}=200,\\
&N_{\rm test}=200.
\end{aligned}
}
$$

Use Latin hypercube sampling for the 1400 tuples

$$
(A,f,\sigma_\eta,\ell_\eta),
$$

RK45 with stringent tolerance for generating \(Y\), and independent forcing realizations for input and target ensembles.

The resulting learning problem is precisely

$$
\boxed{
\underbrace{
\{X_i\}_{i=1}^{128}
}_{\text{empirical representation of }\mu}
\quad
\longrightarrow
\quad
\underbrace{
\{Y_j\}_{j=1}^{128}
}_{\text{samples from }\mathcal S_\#\mu}.
}
$$

This gives you a controlled experiment in which the model must infer the input probability law from finitely many stochastic-process realizations and approximate its nonlinear pushforward through a fixed dynamical system—rather than learning paired trajectory-to-trajectory dynamics.
