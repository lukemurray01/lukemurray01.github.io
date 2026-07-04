---
layout: post
title: "Simulating the unsimulable: stochastic numerics for CIR and CEV"
subtitle: "Why naïve Euler breaks on square-root and CEV diffusions — and the schemes that fix it."
date: 2026-07-04
tags: [Quant, Stochastic Numerics, Monte Carlo, Python]
mathjax: true
description: "An exposition of the numerical challenges in simulating the CIR and CEV processes — positivity, convergence, and the schemes (full-truncation Euler, exact simulation) that solve them."
reading_time: "8 min read"
---

Two diffusions show up everywhere in quantitative finance, and both are quietly
hostile to a naïve simulator. The **CIR** process models short interest rates and
the variance in the Heston model; the **CEV** process models equity prices with a
leverage effect. They are the subject of my thesis, and this post is the short
version of why they are interesting to *simulate*.

## The two processes

The **Cox–Ingersoll–Ross (CIR)** process is the mean-reverting square-root diffusion

$$ dX_t = \kappa(\theta - X_t)\,dt + \sigma\sqrt{X_t}\,dW_t, $$

with mean-reversion speed \(\kappa>0\), long-run level \(\theta>0\) and volatility
\(\sigma>0\). Its defining feature is the **Feller condition**

$$ 2\kappa\theta \ge \sigma^2, $$

which guarantees the process stays strictly positive; violate it and the boundary
at zero becomes attainable.

The **constant-elasticity-of-variance (CEV)** process is

$$ dS_t = \mu S_t\,dt + \sigma S_t^{\beta}\,dW_t, \qquad 0 \le \beta < 1. $$

Here \(\beta=1\) recovers geometric Brownian motion, while \(\beta<1\) produces the
*leverage effect* — volatility rises as the price falls — and makes the origin an
attainable, typically absorbing, boundary.

## Why the obvious scheme fails

The textbook simulator is **Euler–Maruyama**. For CIR:

$$ \hat X_{n+1} = \hat X_n + \kappa(\theta - \hat X_n)\,\Delta t + \sigma\sqrt{\hat X_n}\,\sqrt{\Delta t}\,Z_n, \qquad Z_n \sim \mathcal N(0,1). $$

The problem is immediate: even when the Feller condition holds for the *continuous*
process, a single unlucky Gaussian draw can push \(\hat X_{n+1}\) **below zero**, and
then \(\sqrt{\hat X_{n+1}}\) is not real. The next step produces `nan` and the path
is dead.

This is not a coding bug — it is mathematical. The diffusion coefficient
\(\sigma\sqrt{x}\) is **not Lipschitz** at the origin (its derivative blows up), so
the classical convergence theory for Euler–Maruyama, which assumes globally
Lipschitz coefficients, simply does not apply. The same non-Lipschitz pathology hits
CEV through \(S^{\beta}\) for \(\beta<1\).

## Fixing the negatives

The pragmatic family of fixes keeps Euler but decides what to do when the state goes
negative. The best-behaved is **Full Truncation Euler** (Lord, Koekkoek & van Dijk,
2010): carry the possibly-negative state, but feed its positive part into the
coefficients,

$$ \hat X_{n+1} = \hat X_n + \kappa\bigl(\theta - \hat X_n^{+}\bigr)\Delta t + \sigma\sqrt{\hat X_n^{+}}\,\sqrt{\Delta t}\,Z_n, \qquad x^{+}=\max(x,0), $$

and report \(\hat X^{+}\). Among the simple fixes (absorption, reflection,
truncation) it has consistently the lowest bias.

```python
import numpy as np

def cir_full_truncation(x0, kappa, theta, sigma, T, n_steps, n_paths, rng):
    """Full-truncation Euler for dX = kappa(theta - X)dt + sigma*sqrt(X)dW."""
    dt = T / n_steps
    x = np.full(n_paths, float(x0))
    for _ in range(n_steps):
        xp = np.maximum(x, 0.0)                      # positive part in the coefficients
        z = rng.standard_normal(n_paths)
        x = x + kappa * (theta - xp) * dt + sigma * np.sqrt(xp * dt) * z
    return np.maximum(x, 0.0)
```

Drift-implicit and Milstein variants (Alfonsi; Dereich, Neuenkirch & Szpruch, 2012)
do better still: the implicit Milstein scheme recovers **strong order one** when
\(2\kappa\theta > \sigma^2\), instead of the degraded rate the explicit scheme
suffers near the boundary.

## The exact route

CIR is one of the rare SDEs whose transition law is known in closed form, so we can
sidestep discretisation entirely. Given \(X_t\), the next value is a scaled
**non-central chi-square** variate,

$$ X_{t+\Delta}\,\big|\,X_t \;=\; \frac{\sigma^{2}\bigl(1-e^{-\kappa\Delta}\bigr)}{4\kappa}\; \chi'^{2}_{d}(\lambda), $$

with degrees of freedom and non-centrality

$$ d = \frac{4\kappa\theta}{\sigma^{2}}, \qquad \lambda = X_t\,\frac{4\kappa e^{-\kappa\Delta}}{\sigma^{2}\bigl(1-e^{-\kappa\Delta}\bigr)}. $$

```python
from scipy.stats import ncx2

def cir_exact(x0, kappa, theta, sigma, T, n_paths, rng):
    """Exact one-step CIR sampling via the non-central chi-square law."""
    d   = 4 * kappa * theta / sigma**2
    c   = sigma**2 * (1 - np.exp(-kappa * T)) / (4 * kappa)
    lam = x0 * 4 * kappa * np.exp(-kappa * T) / (sigma**2 * (1 - np.exp(-kappa * T)))
    return c * ncx2.rvs(df=d, nc=lam, size=n_paths, random_state=rng)
```

This draws positive values by construction and carries **no time-discretisation
bias**. Andersen's Quadratic-Exponential (QE) scheme is the fast approximate cousin
used in production Heston engines. Pleasingly, CEV admits the same treatment: it is a
time-changed Bessel process, and Schroder's (1989) pricing formula is likewise built
on the non-central chi-square — so *both* of these awkward diffusions are, in the
end, chi-square in disguise.

## A sanity check worth keeping

Whatever the scheme, the CIR mean has a clean closed form,

$$ \mathbb{E}[X_T] = \theta + (X_0 - \theta)\,e^{-\kappa T}, $$

which makes a cheap, decisive test: simulate, average, compare.

```python
rng = np.random.default_rng(0)
p = dict(x0=0.04, kappa=0.5, theta=0.04, sigma=0.28, T=1.0)  # 2*k*th = 0.04 < sigma^2

approx = cir_full_truncation(**p, n_steps=250, n_paths=200_000, rng=rng).mean()
exact  = cir_exact(**p, n_paths=200_000, rng=rng).mean()
theory = p["theta"] + (p["x0"] - p["theta"]) * np.exp(-p["kappa"] * p["T"])
print(approx, exact, theory)   # all ≈ 0.04, and every path stayed non-negative
```

These parameters deliberately **break** the Feller condition
(\(2\kappa\theta = 0.04 < \sigma^2 = 0.0784\)), which is exactly the regime where
plain Euler falls over — and where the choice of scheme starts to matter for real
prices.

## Takeaways

- **Positivity is the whole game.** Square-root and CEV coefficients are
  non-Lipschitz at zero, so off-the-shelf convergence guarantees evaporate near the
  boundary.
- **Full-truncation Euler** is the right cheap default; **drift-implicit Milstein**
  buys back convergence order when you need it.
- **Exact simulation** exists for both processes via the non-central chi-square — no
  bias, at the cost of a special-function sampler.
- The closer you sit to violating Feller, the more the scheme choice shows up in your
  Monte Carlo prices.

*This is a condensed version of work from my thesis on stochastic numerics for CIR
and CEV processes. Corrections and questions are welcome by
[email](mailto:lukemurray220@gmail.com).*
