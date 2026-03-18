# RGQ
We model the effect of increased lithium-like presence on the line ratios $\mathcal{R}$ = $z$ / ($x$ + $y$), $\mathcal{G}$ = ($x$ + $y$ + $z$) / $w$, and $\mathcal{Q}$ = ($q$ + $r$) / $w$ for astrophysically important ions in a photoionized plasma, by solving the steady-state rate equations as a function of Li-like column density. We calculate both local and observed values for these quantities, the observed values factoring in the geometry of the plasma (we assume a conical geometry) and the position of the observer.

## 1. Download
Download the latest source packages from
[here](https://github.com/gjgrell/RGQ),
gunzip, and extract it.

## 2. Usage
The following commands calculate the local (unobservable) $\mathcal{R}$, $\mathcal{G}$, and $\mathcal{Q}$ ratios:

- $\mathcal{R}$ = R_analytic_local(Z, $N_{Li}$, $N_{He}$, $\sigma_{v}$, $\phi_{UV}$, $n_{e}$, mixing, gamma)
- $\mathcal{G}$ = G_analytic_local(Z, $N_{Li}$, $N_{He}$, $\sigma_{v}$, $\phi_{UV}$, $n_{e}$, mixing, gamma)
- $\mathcal{Q}$ = Q_analytic(Z, $N_{Li}$, $N_{He}$, $\sigma_{v}$ $\phi_{UV}$, $n_{e}$, $\gamma$)

The input parameters for the ratio calculations are as follows:

- Z - atomic symbol (string)
- $N_{Li}$ - lithium-like column density ($cm^{-2}$)
- $N_{He}$ - helium-like column density ($cm^{-2}$)
- $\sigma_{v}$ - velocity dispersion ($km$ $s^{-1}$)
- $\phi_{UV}$ - UV photoexcitation rate ($s^{-1}$)
- $n_{e}$ - electron density ($cm^{-3}$)
- mixing factor
- $\gamma$ - energy power law spectral index

The following commands calculate the observed (global) $\mathcal{R}$ and $\mathcal{G}$ ratios averaged over entire medium factoring in the plasma geometry and observer position. Here we assume a simple, truncated cone for geometry, and a cold photoionized plasma (ne << nc, phi_UV << phi_c):
- $\bar{\mathcal{R}}$ = R_analytic_obs(Z, $N_{Li}$, $N_{He}$, $\sigma_{v}$, mixing, $\gamma$, $\alpha$, $R_0$, R, L, $\beta$, $\xi_{max}$)
- $\bar{\mathcal{G}}$ = G_analytic_obs(Z, $N_{Li}$, $N_{He}$, $\sigma_{v}$, mixing, $\gamma$, $\alpha$, $R_0$, R, L, $\beta$, $\xi_{max}$)

The following commands also calculate the observed (global) $\mathcal{R}$ ratio averaged over entire medium factoring in the plasma geometry and observer position, but with the column densities calculated as a function of radial distribution. We again assume a simple, truncated cone for geometry, and a cold photoionized plasma (ne << nc, phi_UV << phi_c):
- $\bar{\mathcal{R}}^\prime$ = R_analytic_global(Z, $\sigma_{v}$, mixing, $\gamma$, $\alpha$, $R_0$, R, L, $\beta$, $\xi_{max}$)

For the observed ratio calculations, the additional parameters are:
- $\alpha$ - opening angle (arcsec)
- R0 - cone inner radius (parsecs)
- R - maximum cone radius (parsecs)
- L - luminosity ($erg$ $s^{-1}$) 
- $\beta$ - wind velocity coefficient 
- $\xi_{max}$ - maximum ionization parameter 
