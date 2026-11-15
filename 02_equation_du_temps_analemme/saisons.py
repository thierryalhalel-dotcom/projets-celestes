#!/usr/bin/env python3
"""
Saisons et mouvement relatif du Soleil (chapitre "Mars", exercice sur les saisons)
=================================================================================
Redessine la figure "saisons1.png" : orbite apparente du Soleil autour de la
Terre (placee au foyer), avec une excentricite tres exageree, les quatre
directions lambda = 0, 90, 180, 270 degres qui bornent les saisons, le grand
axe perihelie / aphelie et le sens du mouvement du Soleil.

Sortie : figures/saisons_calcule.png
Usage  : python3 saisons.py     (depuis le dossier python/)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

E_EXAGERE = 0.45                 # excentricite volontairement exageree (reelle : 0,0167)
LAMBDA_P  = np.radians(283.0)    # longitude du perihelie de l'orbite apparente du Soleil
P = 1.0                          # parametre de l'ellipse (unite arbitraire)
BLEU, ROUGE, GRIS = "#00288C", "#C0392B", "#666666"


def direction(lam):
    """Vecteur unitaire du plan de la figure : lambda = 0 vers le bas (point vernal),
    lambda = 90 vers la droite, le Soleil tournant donc dans le sens direct."""
    return np.array([np.sin(lam), -np.cos(lam)])


def position(lam):
    f = lam - LAMBDA_P                       # anomalie vraie
    r = P/(1 + E_EXAGERE*np.cos(f))
    return r*direction(lam)


def main():
    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.set_aspect("equal"); ax.axis("off")

    # orbite apparente du Soleil
    lam = np.linspace(0, 2*np.pi, 720)
    xy = np.array([position(l) for l in lam])
    ax.plot(xy[:, 0], xy[:, 1], "-", color=ROUGE, lw=2.2, zorder=3)

    # axes lambda = 0 / 180 (vertical) et 90 / 270 (horizontal), depuis la Terre
    L = 2.05
    ax.annotate("", xy=(0, -L), xytext=(0, L),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6, mutation_scale=18))
    ax.plot([-L, L], [0, 0], "k-", lw=1.2)
    ax.text(0.14, -L + 0.12, "$\\lambda = 0$ : point vernal $\\gamma$", ha="left", va="bottom", fontsize=12)
    ax.text(L + 0.05, 0, "$\\lambda = 90°$", ha="left", va="center", fontsize=12)
    ax.text(0.10, L - 0.02, "$\\lambda = 180°$", ha="left", va="top", fontsize=12)
    ax.text(-L - 0.05, 0, "$\\lambda = 270°$", ha="right", va="center", fontsize=12)

    # Terre au foyer
    ax.plot(0, 0, "o", color=BLEU, ms=11, zorder=5)
    ax.text(0.08, 0.10, "Terre", color=BLEU, fontsize=12, fontweight="bold")

    # grand axe : perihelie et aphelie
    per, aph = position(LAMBDA_P), position(LAMBDA_P + np.pi)
    ax.plot([per[0], aph[0]], [per[1], aph[1]], "-", color=GRIS, lw=1.0, zorder=2)
    ax.plot(*per, "s", color=ROUGE, ms=7, zorder=5)
    ax.plot(*aph, "s", color=ROUGE, ms=7, zorder=5)
    ax.text(per[0] - 0.10, per[1] - 0.04, "périhélie\n($\\lambda \\simeq 283°$)", ha="right", va="top",
            fontsize=11, color=ROUGE)
    ax.text(aph[0] + 0.06, aph[1] + 0.06, "aphélie\n($\\lambda \\simeq 103°$)", ha="left", va="bottom",
            fontsize=11, color=ROUGE)

    # les quatre saisons, dans les quadrants qu'elles occupent
    for nom, l0 in (("printemps", 45), ("été", 135), ("automne", 225), ("hiver", 315)):
        d = direction(np.radians(l0))
        ax.text(*(0.72*d), nom, ha="center", va="center", fontsize=14, style="italic")

    # sens du mouvement du Soleil : fleches tangentes en quatre points de l'orbite
    for l0 in (30, 120, 210, 300):
        a, b = position(np.radians(l0)), position(np.radians(l0 + 8))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", color=ROUGE, mutation_scale=22, lw=0, zorder=4))
    s = position(np.radians(60))
    ax.plot(*s, "o", color="#F5A623", ms=14, mec="#B8860B", zorder=6)
    ax.text(s[0] + 0.12, s[1] - 0.02, "Soleil", ha="left", va="center", fontsize=12, color="#8B6508")

    # encadre explicatif
    ax.text(-2.95, -1.30,
            "Orbite apparente du Soleil autour\n"
            "de la Terre, d'excentricité très\n"
            "exagérée ($e = 0{,}45$ au lieu de $0{,}0167$).\n"
            "Plus rapide près du périhélie\n"
            "(2$^{\\rm e}$ loi de Kepler), le Soleil ne\n"
            "passe pas le même temps dans\n"
            "chaque quadrant : les saisons ont\n"
            "des durées inégales.",
            ha="left", va="top", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="k", lw=0.8))

    ax.set_xlim(-3.0, 2.9); ax.set_ylim(-2.35, 2.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "saisons_calcule.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figure ecrite :", os.path.join(OUT, "saisons_calcule.png"))


if __name__ == "__main__":
    main()
