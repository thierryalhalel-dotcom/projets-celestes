#!/usr/bin/env python3
"""
Figure : conservation de l'energie selon l'integrateur (chapitre Terre - Lune - Soleil).

Integre le systeme Soleil - Terre - Lune (equations (I), memes conditions
initiales que simulation_TLS_Jupiter.py) avec trois methodes et le meme pas
dt = 500 s, et trace l'erreur relative sur l'energie totale du systeme :
  - Euler explicite (non symplectique, ordre 1)
  - Runge-Kutta 4   (non symplectique, ordre 4, 4 evaluations de force par pas)
  - Euler symplectique (ordre 1, methode du livre)
Une quatrieme courbe montre Runge-Kutta 4 a cout de calcul egal (dt = 2000 s).

Le noyau de calcul energie_core.c est compile automatiquement (gcc).
Sortie : resultats/fig_energie_integrateurs.png
"""
import os, sys, subprocess, ctypes as ct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from simulation_TLS_Jupiter import (G, M_S, M_T, M_L, JOUR,
    xT0, yT0, zT0, vxT0, vyT0, vzT0, xL0, yL0, zL0, vxL0, vyL0, vzL0)

AN = 365.25 * JOUR


def bibliotheque():
    src = os.path.join(HERE, "energie_core.c"); lib = os.path.join(HERE, "libenergie.so")
    if not os.path.exists(lib) or os.path.getmtime(src) > os.path.getmtime(lib):
        r = subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", lib, src, "-lm"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr); sys.exit(1)
    L = ct.CDLL(lib)
    D = ct.c_double; P = ct.POINTER(D)
    L.run.argtypes = [ct.c_int, D, ct.c_long, ct.c_int, P, D, D, D, D, P]
    L.run.restype = ct.c_long
    return L


def integre(L, methode, dt, annees, points=4000):
    n = int(annees*AN/dt); se = max(1, n//points)
    etat = np.array([xT0, yT0, zT0, xL0, yL0, zL0, vxT0, vyT0, vzT0, vxL0, vyL0, vzL0])
    E = np.zeros(n//se + 2)
    P = ct.POINTER(ct.c_double)
    ns = L.run(methode, dt, n, se, etat.ctypes.data_as(P), G, M_S, M_T, M_L, E.ctypes.data_as(P))
    t = np.arange(ns)*se*dt/AN
    return t, E[:ns]          # enveloppe de |E/E0 - 1| (maximum par intervalle)


def main():
    L = bibliotheque()
    out = os.path.join(HERE, "resultats"); os.makedirs(out, exist_ok=True)
    ANNEES = 4000.0

    courbes = [
        ("Euler explicite ($\\Delta t$ = 500 s)", 0, 500.0, 2.0, "#888888", "-"),
        ("Runge-Kutta 4 ($\\Delta t$ = 500 s)", 1, 500.0, ANNEES, "#b03a2e", "-"),
        ("Runge-Kutta 4 ($\\Delta t$ = 2000 s, même coût)", 1, 2000.0, ANNEES, "#b03a2e", "--"),
        ("Euler symplectique ($\\Delta t$ = 500 s)", 2, 500.0, ANNEES, "#1f4e79", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for nom, m, dt, an, c, st in courbes:
        panneaux = [(axes[0], min(an, 20.0))] + ([(axes[1], an)] if an >= ANNEES else [])
        for ax, duree in panneaux:
            t, err = integre(L, m, dt, duree, points=2000)
            ax.semilogy(t[1:], np.maximum(err[1:], 1e-16), st, color=c, lw=1.4, label=nom)
        print(f"{nom:45s} erreur max sur {an:.0f} ans : {err.max():.2e}, "
              f"sur le dernier dixieme : {err[-len(err)//10:].max():.2e}")
    axes[0].set_title("Sur 20 ans", fontsize=15)
    axes[1].set_title(f"Sur {ANNEES:.0f} ans", fontsize=15)
    for ax in axes:
        ax.set_xlabel("temps (années)", fontsize=13)
        ax.set_ylabel("erreur relative sur l'énergie $|E/E_0 - 1|$", fontsize=12)
        ax.grid(alpha=0.3, which="both"); ax.tick_params(labelsize=11)
        ax.set_ylim(1e-14, 1e-1)
    axes[0].legend(fontsize=10.5, loc="center", bbox_to_anchor=(0.55, 0.36), framealpha=0.95)
    fig.tight_layout()
    f = os.path.join(out, "fig_energie_integrateurs.png")
    fig.savefig(f, dpi=200); plt.close(fig); print("->", f)


if __name__ == "__main__":
    main()
