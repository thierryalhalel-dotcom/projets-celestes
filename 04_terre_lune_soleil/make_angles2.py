#!/usr/bin/env python3
"""
Genere angles2.png : coordonnees ecliptiques (lambda, beta) de la Lune
sur la sphere celeste geocentrique (figure "coordonnees lunaires").

La figure est construite par une vraie projection orthographique 3D :
tous les points (Lune, projection sur l'ecliptique, extremites des arcs)
sont calcules a partir de leurs coordonnees spheriques, ce qui garantit
que les vecteurs partent du centre et aboutissent exactement aux points.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------
LAM = np.radians(70)     # longitude ecliptique de la Lune (exageree pour la lisibilite)
BET = np.radians(32)      # latitude ecliptique (exageree : la vraie valeur est < 5,3 deg)
AZ = np.radians(-118)     # azimut de l'observateur
EL = np.radians(18)       # elevation de l'observateur au-dessus du plan de l'ecliptique

# Palette sobre : gris pour la structure, deux couleurs pour les deux angles
C_STRUCT = "#555555"
C_ECL = "#1f4e79"         # ecliptique et longitude (bleu sombre)
C_BET = "#b03a2e"         # latitude (rouge brique)
C_MOON = "#222222"
LW = 2.2


def P(v):
    """Projection orthographique d'un point 3D (x vers le point vernal,
    z vers le pole nord ecliptique). Renvoie (X, Y, profondeur)."""
    x, y, z = v
    # rotation autour de z (azimut)
    x1 = x*np.cos(AZ) - y*np.sin(AZ)
    y1 = x*np.sin(AZ) + y*np.cos(AZ)
    # rotation autour de l'axe horizontal de l'ecran (elevation)
    y2 = y1*np.cos(EL) - z*np.sin(EL)
    z2 = y1*np.sin(EL) + z*np.cos(EL)
    return np.array([x1, z2]), y2      # y2 > 0 : point situe derriere


def sph(lam, bet, r=1.0):
    return np.array([r*np.cos(bet)*np.cos(lam), r*np.cos(bet)*np.sin(lam), r*np.sin(bet)])


def courbe(points, ax, couleur, lw=LW, avant_seul=False, **kw):
    """Trace une courbe 3D : trait plein devant, tirets derriere."""
    pts = [P(p) for p in points]
    xy = np.array([q[0] for q in pts]); dep = np.array([q[1] for q in pts])
    devant = dep <= 1e-9
    for masque, style in ((devant, "-"), (~devant, "--")):
        if avant_seul and style == "--":
            continue
        seg = np.where(masque, 0, 1)
        # decoupe en morceaux continus
        idx = np.where(masque)[0]
        if len(idx) == 0:
            continue
        coupures = np.where(np.diff(idx) > 1)[0]
        for morceau in np.split(idx, coupures + 1):
            if len(morceau) > 1:
                ax.plot(xy[morceau, 0], xy[morceau, 1], style, color=couleur, lw=lw,
                        dashes=(5, 4) if style == "--" else (None, None), **kw)


def fleche(ax, a3, b3, couleur, lw=LW, tete=14):
    a, _ = P(a3); b, _ = P(b3)
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle="-|>", color=couleur, lw=lw,
                                mutation_scale=tete, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.2, 6.6))
ax.set_aspect("equal"); ax.axis("off")
O = np.zeros(3)
t = np.linspace(0, 2*np.pi, 721)

# Sphere celeste : contour apparent (cercle de rayon 1)
ax.plot(np.cos(t), np.sin(t), color=C_STRUCT, lw=1.4)

# Ecliptique (grand cercle z = 0)
courbe([sph(a, 0) for a in t], ax, C_ECL, lw=1.8)

# Axe du pole nord ecliptique (partie interne en pointille, partie externe fleche)
courbe([np.array([0, 0, s]) for s in np.linspace(-1.0, 1.0, 50)], ax, C_STRUCT, lw=1.2)
fleche(ax, np.array([0, 0, 1.0]), np.array([0, 0, 1.30]), C_STRUCT, lw=1.6)
pn, _ = P(np.array([0, 0, 1.33]))
ax.text(pn[0], pn[1] + 0.02, "Pôle nord\nécliptique", ha="center", va="bottom",
        fontsize=13, color=C_STRUCT)

# Direction du point vernal (axe x)
fleche(ax, O, np.array([1.45, 0, 0]), C_STRUCT, lw=1.8)
pv, _ = P(np.array([1.47, 0, 0]))
ax.text(pv[0] + 0.02, pv[1] - 0.08, "Point vernal", ha="center", va="top",
        fontsize=13, color=C_STRUCT)

# Lune M, sa projection H sur l'ecliptique
M = sph(LAM, BET); H = sph(LAM, 0)

# Rayon O -> H (dans le plan de l'ecliptique) et meridien H -> M
courbe([s*H for s in np.linspace(0, 1, 30)], ax, C_ECL, lw=1.3, alpha=0.9)
courbe([sph(LAM, b) for b in np.linspace(0, BET, 60)], ax, C_STRUCT, lw=1.3)

# Vecteur Terre -> Lune, du centre exactement jusqu'a la Lune
fleche(ax, O, M*0.985, C_MOON, lw=2.4, tete=16)
pm, _ = P(M)
ax.plot(*pm, "o", ms=13, color=C_MOON, zorder=5)
ax.text(pm[0] + 0.06, pm[1] + 0.05, "Lune", ha="left", va="bottom", fontsize=14,
        color=C_MOON, fontweight="bold")
ph, _ = P(H)
ax.plot(*ph, "o", ms=5, color=C_ECL, zorder=5)

# Terre au centre
p0, _ = P(O)
ax.plot(*p0, "o", ms=9, color="#2e75b6", zorder=6)
ax.text(p0[0] - 0.04, p0[1] + 0.03, "Terre", ha="right", va="bottom", fontsize=13, color=C_STRUCT)

# Arc lambda : dans le plan de l'ecliptique, du point vernal a la direction OH
r_l = 0.55
arc_l = [sph(a, 0, r_l) for a in np.linspace(0, LAM, 80)]
courbe(arc_l, ax, C_ECL, lw=2.6, avant_seul=False)
fleche(ax, arc_l[-4], arc_l[-1], C_ECL, lw=2.6, tete=13)
pl, _ = P(sph(LAM*0.62, 0, r_l + 0.16))
ax.text(pl[0], pl[1], "$\\lambda$", ha="center", va="center", fontsize=20, color=C_ECL)

# Arc beta : dans le plan meridien, de OH a OM
r_b = 0.62
arc_b = [sph(LAM, b, r_b) for b in np.linspace(0, BET, 60)]
courbe(arc_b, ax, C_BET, lw=2.6)
fleche(ax, arc_b[-4], arc_b[-1], C_BET, lw=2.6, tete=13)
pb, _ = P(sph(LAM, BET/2, r_b + 0.12))
ax.text(pb[0], pb[1], "$\\beta$", ha="center", va="center", fontsize=20, color=C_BET)

# Legende de l'ecliptique
pe, _ = P(sph(np.radians(100), 0, 1.0))
ax.text(pe[0] + 0.06, pe[1] - 0.06, "Écliptique", ha="left", va="top", fontsize=13,
        color=C_ECL)

ax.set_xlim(-1.8, 1.6); ax.set_ylim(-1.15, 1.6)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "angles2.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print("->", out)
