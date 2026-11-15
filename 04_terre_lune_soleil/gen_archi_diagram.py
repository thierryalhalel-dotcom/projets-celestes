#!/usr/bin/env python3
"""
Genere archi_diagram.png : architecture du programme Terre - Lune - Soleil
(Python / bibliotheque partagee .so / source C).

Dessine a l'echelle de la page (11 x 6 pouces, polices 12-15 pt) pour rester
lisible une fois insere sur toute la largeur du texte.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

C_PY, C_SO, C_C = "#1f4e79", "#a93226", "#2e6b30"
C_DATA, C_GCC = "#c0590a", "#6a1b9a"
FT, FI, FA = 13.5, 12, 11.5          # titres, lignes, etiquettes de fleches

fig, ax = plt.subplots(figsize=(12, 5.8))
ax.set_xlim(0, 112); ax.set_ylim(0, 54); ax.axis("off")


def bloc(x, y, w, h, couleur, titre, sous_titre, lignes):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.8",
                                facecolor=couleur, edgecolor="none", zorder=2))
    ax.text(x + w/2, y + h - 2.6, titre, ha="center", va="center", fontsize=FT,
            fontweight="bold", color="white", family="monospace", zorder=3)
    ax.text(x + w/2, y + h - 6.0, sous_titre, ha="center", va="center", fontsize=FI,
            style="italic", color="white", alpha=0.9, zorder=3)
    yy = y + h - 10.2
    for l in lignes:
        ax.text(x + 1.6, yy, l, ha="left", va="center", fontsize=FI, color="white", zorder=3)
        yy -= 3.7


def fleche(x1, y1, x2, y2, couleur, texte, dx=0.0, dy=0.0, rad=0.0, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=4,
                arrowprops=dict(arrowstyle=style, color=couleur, lw=2.4, linestyle=ls,
                                mutation_scale=20, connectionstyle=f"arc3,rad={rad}"))
    t = ax.text((x1 + x2)/2 + dx, (y1 + y2)/2 + dy, texte, ha="center", va="center",
                fontsize=FA, color=couleur, fontweight="bold", zorder=5)
    t.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])


# Blocs
bloc(1, 10, 38, 35, C_PY, "simulation_TLS_Jupiter.py", "Python : le chef d'orchestre", [
    "1. lit les options",
    "2. compile le C si besoin (gcc)",
    "3. conditions initiales",
    "4. réserve les tableaux NumPy",
    "5. appelle simulate() (ctypes)",
    "6. FFT, éphémérides, figures",
    "7. sauvegarde le fichier .npz",
])
bloc(50, 13, 30, 29, C_SO, "libsimulation.so", "code machine compilé", [
    "simulate(...)",
    "Euler symplectique :",
    "  ≈ 250 millions de pas",
    "périgées, nœuds, années",
])
bloc(87, 19, 24, 23, C_C, "simulation_core.c", "source C (lisible)", [
    "équations (I)",
    "intégrateur",
    "détection des",
    "  événements",
])

# Fleches (etiquettes placees hors des blocs)
fleche(40, 34, 49, 34, C_DATA, "", )
ax.text(44.5, 49.5, "appel ctypes : conditions initiales\n+ adresses des tableaux NumPy",
        ha="center", va="center", fontsize=FA, color=C_DATA, fontweight="bold")
ax.annotate("", xy=(44.5, 35.2), xytext=(44.5, 46.6),
            arrowprops=dict(arrowstyle="-", color=C_DATA, lw=1.2, linestyle=":"))
fleche(49, 20, 40, 20, C_DATA, "")
ax.text(44.5, 5.0, "le C remplit directement\nles tableaux NumPy (sans copie)",
        ha="center", va="center", fontsize=FA, color=C_DATA, fontweight="bold")
ax.annotate("", xy=(44.5, 18.8), xytext=(44.5, 8.0),
            arrowprops=dict(arrowstyle="-", color=C_DATA, lw=1.2, linestyle=":"))
fleche(99, 18.4, 81, 22, C_GCC, "", rad=-0.35)
ax.text(99, 11.5, "gcc compile le .c\nen .so (1er lancement)",
        ha="center", va="center", fontsize=FA, color=C_GCC, fontweight="bold")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archi_diagram.png")
plt.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print("->", out)
