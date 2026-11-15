#!/usr/bin/env python3
"""
Genere angles3.png : parametres de l'orbite lunaire calcules par le programme
(figure "parametres de l'orbite lunaire" du chapitre Terre - Lune - Soleil).

Elements representes : plan de l'ecliptique, orbite lunaire inclinee de i,
noeuds ascendant N1 et descendant N2, ligne des noeuds, longitude du noeud
ascendant Omega, perigee P et apogee A, distance r_p au perigee, argument du
perigee omega (compte dans le plan de l'orbite depuis N1), et angle Xi :
angle, compte depuis Ox, de la projection du perigee sur l'ecliptique
(c'est la grandeur calculee par le programme, via cos Xi et sin Xi).

Construction par vraie projection orthographique 3D : tous les points sont
calcules a partir des elements orbitaux, donc geometriquement coherents.
Inclinaison et excentricite sont exagerees pour la lisibilite.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Elements orbitaux (exageres pour la lisibilite)
# ---------------------------------------------------------------------------
I = np.radians(55)        # inclinaison (vraie valeur : 5,15 deg)
OM = np.radians(95)       # longitude du noeud ascendant Omega
W = np.radians(75)        # argument du perigee omega
E = 0.20                  # excentricite (vraie valeur : 0,055 en moyenne)
A = 1.0                   # demi-grand axe (unite arbitraire)
R_ECL = 1.0               # (recalcule plus bas : rayon du cercle passant par N1 et N2)

AZ = np.radians(-180)     # orientation de la vue (comme la figure des coordonnees)
EL = np.radians(27)

C_STRUCT = "#555555"
C_ECL = "#1f4e79"
C_ORB = "#222222"
C_ANG = "#b03a2e"
C_ANG2 = "#7d5a00"
LW = 2.0


def P(v):
    x, y, z = v
    x1 = x*np.cos(AZ) - y*np.sin(AZ)
    y1 = x*np.sin(AZ) + y*np.cos(AZ)
    y2 = y1*np.cos(EL) - z*np.sin(EL)
    z2 = y1*np.sin(EL) + z*np.cos(EL)
    return np.array([x1, z2])


def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def Rx(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


ROT = Rz(OM) @ Rx(I)                     # repere du plan de l'orbite (x' vers N1)


def orbite(nu):
    """Position sur l'orbite pour l'anomalie vraie nu (Terre au foyer)."""
    r = A*(1 - E**2)/(1 + E*np.cos(nu))
    u = W + nu                            # argument de latitude
    return ROT @ np.array([r*np.cos(u), r*np.sin(u), 0.0])


def dans_orbite(u, r):
    """Point du plan de l'orbite a l'angle u compte depuis N1."""
    return ROT @ np.array([r*np.cos(u), r*np.sin(u), 0.0])


def dans_ecl(lam, r):
    return np.array([r*np.cos(lam), r*np.sin(lam), 0.0])


def trace(ax, pts, couleur, lw=LW, style="-", **kw):
    xy = np.array([P(p) for p in pts])
    ax.plot(xy[:, 0], xy[:, 1], style, color=couleur, lw=lw, **kw)


def cache_par_disque(p):
    """Vrai si le point p (sous le plan de l'ecliptique) est cache, pour
    l'observateur, par le disque de l'ecliptique (rayon R_ECL, centre C_ECL_CENTRE)."""
    if p[2] >= 0:
        return False
    v = np.array([-np.cos(EL)*np.sin(AZ), -np.cos(EL)*np.cos(AZ), np.sin(EL)])  # vers l'observateur
    q = p + (-p[2]/v[2])*v                       # intersection du rayon visuel avec le plan z = 0
    return np.hypot(q[0] - C_ECL_CENTRE[0], q[1] - C_ECL_CENTRE[1]) < R_ECL


def trace_orbite(ax, pts):
    """Trait plein pour les parties visibles, tirets pour les parties cachees par le disque."""
    cache = np.array([cache_par_disque(p) for p in pts]); xy = np.array([P(p) for p in pts])
    for masque, style in ((~cache, "-"), (cache, "--")):
        idx = np.where(masque)[0]
        if len(idx) < 2:
            continue
        for m in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if len(m) > 1:
                ax.plot(xy[m, 0], xy[m, 1], style, color=C_ORB, lw=2.4,
                        dashes=(6, 4) if style == "--" else (None, None))


def fleche(ax, a3, b3, couleur, lw=LW, tete=13):
    ax.annotate("", xy=P(b3), xytext=P(a3),
                arrowprops=dict(arrowstyle="-|>", color=couleur, lw=lw,
                                mutation_scale=tete, shrinkA=0, shrinkB=0))


def arc(ax, pts, couleur, lw=2.4, tete=12):
    trace(ax, pts, couleur, lw=lw)
    fleche(ax, pts[-3], pts[-1], couleur, lw=lw, tete=tete)


def texte(ax, p3, s, dx=0.0, dy=0.0, **kw):
    q = P(p3)
    ax.text(q[0] + dx, q[1] + dy, s, **kw)


fig, ax = plt.subplots(figsize=(14.0, 11.5))
ax.set_aspect("equal"); ax.axis("off")
O = np.zeros(3)
t = np.linspace(0, 2*np.pi, 721)

# Plan de l'ecliptique : disque legerement teinte, dont le bord est le cercle
# du plan de l'ecliptique passant par les deux noeuds N1 et N2
N1_ = orbite(-W); N2_ = orbite(np.pi - W)
C_ECL_CENTRE = (N1_ + N2_)/2
R_ECL = np.linalg.norm(N1_ - N2_)/2
bord = np.array([P(C_ECL_CENTRE + dans_ecl(a, R_ECL)) for a in t])
ax.fill(bord[:, 0], bord[:, 1], color=C_ECL, alpha=0.07, lw=0)
ax.plot(bord[:, 0], bord[:, 1], color=C_ECL, lw=1.6)
texte(ax, C_ECL_CENTRE + dans_ecl(np.radians(180), R_ECL), "Plan de l'écliptique", dx=0.06, dy=0.0,
      fontsize=16, color=C_ECL, ha="left", va="bottom")

# Axe Ox (point vernal)
fleche(ax, O, dans_ecl(0, R_ECL + 0.30), C_STRUCT, lw=1.8, tete=15)
texte(ax, dans_ecl(0, R_ECL + 0.34), "$Ox$ (point vernal)", dx=-0.04, dy=0.0,
      fontsize=16, color=C_STRUCT, ha="right", va="center")

# Orbite lunaire
pts = [orbite(n) for n in t]
trace_orbite(ax, pts)
fleche(ax, orbite(np.radians(-62)), orbite(np.radians(-55)), C_ORB, lw=2.4, tete=16)
texte(ax, orbite(np.radians(215)), "Orbite\nlunaire", dx=-0.10, dy=0.0, fontsize=16,
      color=C_ORB, ha="right", va="center")

# Noeuds et ligne des noeuds (N1 : u = 0, N2 : u = 180 deg)
N1 = orbite(-W); N2 = orbite(np.pi - W)
trace(ax, [N2*1.15, N1*1.30], C_STRUCT, lw=1.3, style="-")
for N, nom, dx, dy, ha in ((N1, "$N_1$", -0.12, -0.05, "right"), (N2, "$N_2$", -0.07, 0.07, "right")):
    q = P(N); ax.plot(*q, "o", ms=8, color=C_ECL, zorder=6)
    ax.text(q[0] + dx, q[1] + dy, nom, fontsize=19, color=C_ECL, ha=ha, va="center")
texte(ax, N1*1.30, "Ligne des nœuds", dx=0.05, dy=-0.02, fontsize=15,
      color=C_STRUCT, ha="left", va="top")

# Perigee P, apogee A, ligne des apsides
Pe = orbite(0.0); Ap = orbite(np.pi)
trace(ax, [Ap, Pe], C_STRUCT, lw=1.1, style=":")
fleche(ax, O, Pe*0.975, C_ORB, lw=2.2, tete=15)
q = P(Pe); ax.plot(*q, "o", ms=7, color=C_ORB, zorder=6)
ax.text(q[0] + 0.05, q[1] + 0.03, "Périgée", fontsize=16, color=C_ORB, ha="left", va="bottom")
qm = (P(O) + P(Pe))/2
ax.text(qm[0] - 0.03, qm[1] + 0.05, "$r_p$", fontsize=20, color=C_ORB, ha="right", va="bottom")
q = P(Ap); ax.plot(*q, "o", ms=6, color=C_ORB, zorder=6)
ax.text(q[0] - 0.05, q[1] + 0.02, "Apogée", fontsize=15, color=C_ORB, ha="right", va="bottom")

# Projection du perigee sur l'ecliptique et angle Xi
Pp = np.array([Pe[0], Pe[1], 0.0])
trace(ax, [Pe, Pp], C_ANG2, lw=1.2, style="--")
trace(ax, [O, Pp], C_ANG2, lw=1.2, style="--")
ax.plot(*P(Pp), "o", ms=5, color=C_ANG2, zorder=6)
lam_p = np.arctan2(Pp[1], Pp[0])
arc(ax, [dans_ecl(a, 0.50) for a in np.linspace(0, lam_p, 90)], C_ANG2, lw=2.0)
texte(ax, dans_ecl(lam_p*0.30, 0.62), "$\\Xi$", dy=0.0, fontsize=22, color=C_ANG2, ha="center", va="center")

# Longitude du noeud ascendant Omega (dans l'ecliptique, de Ox a N1)
arc(ax, [dans_ecl(a, 0.38) for a in np.linspace(0, OM, 40)], C_ANG, lw=2.4)
texte(ax, dans_ecl(OM*0.55, 0.25), "$\\Omega$", fontsize=22, color=C_ANG, ha="center", va="center")

# Argument du perigee omega (dans le plan de l'orbite, de N1 au perigee)
arc(ax, [dans_orbite(u, 0.24) for u in np.linspace(0, W, 60)], C_ANG, lw=2.4)
texte(ax, dans_orbite(W*0.55, 0.44), "$\\omega$", fontsize=22, color=C_ANG, ha="center", va="center")

# Inclinaison i au noeud N1 : angle entre l'ecliptique et l'orbite,
# dans le plan perpendiculaire a la ligne des noeuds
n1 = N1/np.linalg.norm(N1)
e_ecl = np.cross([0, 0, 1], n1)              # direction dans l'ecliptique, perpendiculaire aux noeuds
e_orb = ROT @ np.array([0, 1.0, 0])          # direction dans l'orbite, perpendiculaire aux noeuds
r_i = 0.22
arc_i = [N1 + r_i*(np.cos(s)*e_ecl + np.sin(s)*np.cross(n1, e_ecl)) for s in np.linspace(0, I, 30)]
trace(ax, [N1, N1 + 0.46*e_ecl], C_ECL, lw=1.6)
trace(ax, [N1, N1 + 0.46*e_orb], C_ORB, lw=1.6)
arc(ax, arc_i, C_ANG, lw=3.0, tete=15)
texte(ax, N1 + 0.30*e_ecl, "$i$",
      dx=0.0, dy=-0.08, fontsize=22, color=C_ANG, ha="left", va="center")

# Terre au foyer
q = P(O); ax.plot(*q, "o", ms=13, color="#2e75b6", zorder=7)
ax.text(q[0] - 0.03, q[1] + 0.13, "Terre", fontsize=16, color=C_STRUCT, ha="right", va="bottom")

ax.set_xlim(-2.2, 1.95); ax.set_ylim(-1.35, 1.45)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "angles3.png")
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print("->", out)
