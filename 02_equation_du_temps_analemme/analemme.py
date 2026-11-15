#!/usr/bin/env python3
"""
Analemme calcule (chapitre "Mars", figure 2.6)
==============================================
Reprend le modele du listing du chapitre (elements moyens du Soleil d'apres
Meeus, equation du temps sous la forme 1 a l'ordre 2) et trace :

  * figures/analemme_calcule.png       : l'analemme de 2025, avec les dates
                                      remarquables (textes en bleu sombre)
  * figures/analemme_superpose.png     : superposition 2025 / 2035 / 2055 / 2105,
                                      (2025 / 2105 / 2505 / 3005 par defaut) :
                                      la derive (avance du perihelie, baisse
                                      de l'obliquite) devient visible a
                                      l'echelle du siecle... et du millenaire.

Usage :  python3 analemme.py     (a lancer depuis le dossier python/)
"""
import datetime, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

JD2000 = 2451545.0          # epoque J2000.0
MIN    = 1440.0/(2*np.pi)   # 1 radian en minutes de temps
BLEU   = "#00288C"          # bleu sombre pour les annotations


def elements(jd):
    """Elements moyens du Soleil (Meeus, chap. 25)."""
    T = (jd - JD2000)/36525.0
    L = np.radians(280.46646 + 36000.76983*T + 0.0003032*T**2)   # longitude moyenne
    M = np.radians(357.52911 + 35999.05029*T - 0.0001537*T**2)   # anomalie moyenne
    e = 0.016708634 - 0.000042037*T - 0.0000001267*T**2          # excentricite
    eps = np.radians(23.4392911 - 0.0130042*T)                   # obliquite
    return L, M, e, eps


def analemme(jd):
    """Retourne (E en minutes, declinaison en degres) pour les dates juliennes jd."""
    L, M, e, eps = elements(jd)
    U = np.tan(eps/2.0)**2
    E  = 2*e*np.sin(M) + 1.25*e**2*np.sin(2*M)          # equation du centre
    E += -U*np.sin(2*L) - 4*e*U*np.sin(M)*np.cos(2*L)   # reduction a l'equateur
    E += 0.5*U**2*np.sin(4*L)
    lam = L + 2*e*np.sin(M) + 1.25*e**2*np.sin(2*M)     # longitude vraie
    dec = np.degrees(np.arcsin(np.sin(eps)*np.sin(lam)))  # sin(delta) = sin(eps) sin(lambda)
    return E*MIN, dec


def jd_debut(annee):
    """Date julienne du 1er janvier de l'annee a 0 h UT."""
    return 2451544.5 + (datetime.date(annee, 1, 1) - datetime.date(2000, 1, 1)).days


def jour_de(annee, mois, jour):
    return (datetime.date(annee, mois, jour) - datetime.date(annee, 1, 1)).days


DATES = [(6, 21, "21 juin", (0.4, 0.8)), (3, 21, "21 mars", (0.7, 1.1)),
         (9, 23, "23 sept.", (-2.8, 1.1)), (2, 11, "11 févr.", (0.4, 0.8)),
         (11, 3, "3 nov.", (-2.4, 1.0)), (12, 21, "21 déc.", (-1.5, 0.8))]


def dates_remarquables(ax, annee, ms=9, fontsize=13, decalages=None):
    """Points et etiquettes (bleu sombre) aux solstices, equinoxes et extrema de E.
    `decalages` : dict nom -> (dx, dy) pour remplacer les decalages par defaut."""
    for m, d, nom, (dx, dy) in DATES:
        if decalages and nom in decalages:
            dx, dy = decalages[nom]
        Ej, dj = analemme(jd_debut(annee) + jour_de(annee, m, d))
        ax.plot(-Ej, dj, "o", color=BLEU, ms=ms, zorder=5)
        ax.text(-Ej + dx, dj + dy, nom, color=BLEU, fontsize=fontsize)


def fig_2025():
    annee = 2025
    t = np.arange(0, 365, 0.25)
    E, dec = analemme(jd_debut(annee) + t)

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.plot(-E, dec, "k-", lw=2.2)
    dates_remarquables(ax, annee)
    ax.set_title(f"L'analemme calculé, année {annee}", fontsize=16)
    ax.set_xlabel("écart Est-Ouest : $-E(t)$, en minutes de temps\n"
                  "(l'Est est à gauche, comme dans le ciel vu vers le Sud)", fontsize=14)
    ax.set_ylabel("déclinaison du Soleil $\\delta_\\odot$ (degrés)", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(-16, 18); ax.set_ylim(-26, 26)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "analemme_calcule.png"), dpi=150)
    plt.close(fig)


def fig_superpose(annees=(2025, 2105, 2505, 3005)):
    t = np.arange(0, 365, 0.25)
    couleurs = ["k", "#d62728", "#1f77b4", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8, 9))
    for annee, c in zip(annees, couleurs):
        E, dec = analemme(jd_debut(annee) + t)
        ax.plot(-E, dec, "-", color=c, lw=1.6, label=str(annee))
    dates_remarquables(ax, annees[0], ms=8, fontsize=12)   # dates sur la 1re annee seule
    ax.set_title("L'analemme calculé : " + ", ".join(str(a) for a in annees[:-1]) + f" et {annees[-1]}", fontsize=16)
    ax.set_xlabel("écart Est-Ouest : $-E(t)$, en minutes de temps\n"
                  "(l'Est est à gauche, comme dans le ciel vu vers le Sud)", fontsize=14)
    ax.set_ylabel("déclinaison du Soleil $\\delta_\\odot$ (degrés)", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(-16, 18); ax.set_ylim(-26, 26)
    ax.legend(fontsize=13, loc="upper left", title="année", title_fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "analemme_superpose.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_superpose()
    # fig_2025()   # decommenter pour regenerer la figure 2025 seule
    print("figures ecrites dans", OUT)
