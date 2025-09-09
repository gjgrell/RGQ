# RGQ
We model the effect of increased lithium-like presence on the line ratios $\mathcal{R}$ = $z$ / ($x$ + $y$), $\mathcal{G}$ = ($x$ + $y$ + $z$) / $w$, and $\mathcal{Q}$ = ($q$ + $r$) / $w$ for astrophysically important ions in a photoionized plasma, by solving the steady-state rate equations as a function of Li-like column density. We calculate both local and observed values for these quantities, the observed values factoring in the geometry of the plasma (we assume a conical geometry) and the position of the observer.

## 1. Download
Download the latest source packages from
[here](https://github.com/gjgrell/RGQ),
gunzip, and extract it.

## 2. Usage
The input parameters for the ratio calculations are as follows:

- Z - atomic symbol (string)
- $N_{Li}$ - lithium-like column density ($cm^{-2}$)
- $N_{He}$ - helium-like column density ($cm^{-2}$)
- v - turbulent velocity ($cm$ $s^{-1}$)
- $\phi_{UV}$ - UV photoexcitation rate ($s^{-1}$)
- $n_{e}$ - electron density ($cm^{-3}$)
- mixing factor
- $\gamma$ - energy power law spectral index
- $\alpha$ - opening angle (arcsec) $[Observed ratios only]$
- R - maximum cone radius (cm) (Observed ratios only)

The following commands calculate the $\mathcal{R}$, $\mathcal{G}$, and $\mathcal{Q}$ ratios:

- $\mathcal{R}$ = R_analytic_local(Z, $N_{Li}$, $N_{He}$, v, $\phi_{UV}$, $n_{e}$, mixing, gamma)
- $\bar{\mathcal{R}}$ = R_analytic_obs(Z, $N_{Li}$, $N_{He}$, v, $\phi_{UV}$, $n_{e}$, mixing, gamma, alpha, R)
- $\mathcal{G}$ = G_analytic_local(Z, $N_{Li}$, $N_{He}$, v, $\phi_{UV}$, $n_{e}$, mixing, gamma)
- $\bar{\mathcal{G}}$ = G_analytic_obs(Z, $N_{Li}$, $N_{He}$, v, $\phi_{UV}$, $n_{e}$, mixing, gamma, alpha, R)
- $\mathcal{Q}$ = Q_analytic(Z, $N_{Li}$, $N_{He}$, v, $\phi_{UV}$, $n_{e}$, gamma)
