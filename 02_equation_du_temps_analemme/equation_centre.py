#!/usr/bin/env python3
"""
Equation du centre f - M en fonction de M, parametree par l'excentricite
=======================================================================
Reprend la figure 2 de l'article "Saisons et equation du temps sur Mars"
(BUP n. 894, 2007) : la Terre (e = 0,0167), Mars (e = 0,0934) et des
excentricites croissantes par pas de 0,03, l'equation de Kepler etant
resolue exactement par la methode de Newton.

Sortie : figures/equation_centre.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


def kepler(M, e, n_iter=8):
    x = M.copy()
    for _ in range(n_iter):
        x = x - (x - e*np.sin(x) - M)/(1 - e*np.cos(x))
    return x


def equation_du_centre(M, e):
    E = kepler(M, e)
    f = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    return np.degrees(np.angle(np.exp(1j*(f - M))))


def main():
    M = np.radians(np.linspace(0, 360, 721))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for e in (0.0167 + 0.03*k for k in range(1, 7)):
        C = equation_du_centre(M, e)
        ax.plot(np.degrees(M), C, "-", color="0.75", lw=1)
        if e < 0.09:      # sous la courbe de Mars : etiquette sous la courbe vers M = 225
            i = np.argmin(np.abs(np.degrees(M) - 225)); y, va = C[i] - 0.3, "top"
        else:
            i = np.argmax(C); y, va = C[i] + 0.4, "bottom"
        ax.text(np.degrees(M[i]), y, f"$e = {e:.4f}$".replace(".", ","),
                fontsize=8, color="k", ha="center", va=va,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.plot(np.degrees(M), equation_du_centre(M, 0.0167), "b-", lw=2.2, label="Terre : $e = 0{,}0167$")
    ax.plot(np.degrees(M), equation_du_centre(M, 0.0934), "r-", lw=2.2, label="Mars : $e = 0{,}0934$")
    # approximation au premier ordre pour Mars
    ax.plot(np.degrees(M), np.degrees(2*0.0934*np.sin(M)), "r--", lw=1, label="Mars, ordre 1 : $2e\\,\\sin M$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("anomalie moyenne $M$ (degrés)")
    ax.set_ylabel("équation du centre $f - M$ (degrés)")
    ax.set_title("Équation du centre en fonction de l'anomalie moyenne, pour diverses excentricités")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.savefig(os.path.join(OUT, "equation_centre.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figure ecrite :", os.path.join(OUT, "equation_centre.png"))


if __name__ == "__main__":
    main()
