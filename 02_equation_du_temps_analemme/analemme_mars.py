#!/usr/bin/env python3
"""
Equation du temps et analemme sur Mars (chapitre "Mars")
========================================================
Complement de la figure "eqdMars.png" (E en fonction de la longitude solaire
Ls) : ici E est calculee en fonction de la DATE, ce qui impose de resoudre
l'equation de Kepler puisque, avec e = 0,093, Ls n'est plus une fonction
lineaire du temps.

  * figures/eqdMars_temps.png  : E(t) sur Mars et sur Terre en fonction du temps,
                              compte en fraction d'annee depuis l'equinoxe de
                              printemps (Ls = 0)
  * figures/analemme_mars.png  : analemmes de la Terre et de Mars a la meme echelle

Usage :  python3 analemme_mars.py     (a lancer depuis le dossier python/)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import analemme as TERRE          # modele terrestre du chapitre (Meeus)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
BLEU = "#00288C"

# ---- parametres martiens (tableau "marsParams" du chapitre) -------------
e_M    = 0.0934                         # excentricite
eps_M  = np.radians(25.183)             # obliquite
varpi  = np.radians(250.98)             # Ls du perihelie
SOL    = 1.027491                       # 1 sol en jours terrestres
AN_ANO = 686.9951/SOL                   # annee anomalistique en sols (668,61)
AN_TRO = 686.9725/SOL                   # annee tropique en sols (668,59)
MIN_M  = 24.6598*60/(2*np.pi)           # 1 radian en minutes martiennes (1479,6 min/sol)


def kepler(M, e, n_iter=30):
    """Anomalie excentrique E telle que E - e sin E = M (iteration de point fixe)."""
    E = M.copy()
    for _ in range(n_iter):
        E = M + e*np.sin(E)
    return E


def mars(t_sols):
    """Pour des dates en sols depuis le perihelie : Ls, E (min), declinaison (deg)."""
    M  = 2*np.pi*t_sols/AN_ANO                      # anomalie moyenne
    Ek = kepler(M, e_M)
    f  = 2*np.arctan2(np.sqrt(1+e_M)*np.sin(Ek/2), np.sqrt(1-e_M)*np.cos(Ek/2))
    Ls = f + varpi                                   # longitude solaire (vraie)
    C  = np.angle(np.exp(1j*(f - M)))                # equation du centre f - M
    alpha = np.arctan2(np.cos(eps_M)*np.sin(Ls), np.cos(Ls))
    R  = np.angle(np.exp(1j*(alpha - Ls)))           # reduction a l'equateur alpha - Ls
    E  = (C + R)*MIN_M                               # E = alpha_S - alpha_M, en minutes
    dec = np.degrees(np.arcsin(np.sin(eps_M)*np.sin(Ls)))
    return np.mod(Ls, 2*np.pi), E, dec


def depuis_equinoxe():
    """Une annee martienne echantillonnee depuis Ls = 0 : (sols, Ls, E, dec)."""
    t = np.arange(0, AN_ANO, 0.25)
    Ls, E, dec = mars(t)
    i0 = np.argmin(np.abs(np.angle(np.exp(1j*Ls))))  # premier passage a Ls = 0
    t = np.roll(t, -i0); Ls = np.roll(Ls, -i0); E = np.roll(E, -i0); dec = np.roll(dec, -i0)
    return t - t[0] + (t < t[0])*AN_ANO if False else np.arange(len(t))*0.25, Ls, E, dec


def terre_depuis_equinoxe(annee=2025):
    """Terre : jours depuis l'equinoxe de mars 2025 (20 mars), E et declinaison."""
    jd0 = TERRE.jd_debut(annee) + TERRE.jour_de(annee, 3, 20) + 0.38
    t = np.arange(0, 365.25, 0.25)
    E, dec = TERRE.analemme(jd0 + t)
    return t, E, dec


def fig_eqdt_temps():
    tm, Ls, Em, _ = depuis_equinoxe()
    tt, Et, _ = terre_depuis_equinoxe()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tm/AN_TRO, Em, "r-", lw=2, label="Mars (annee de 668,6 sols)")
    ax.plot(tt/365.2422, Et, "b-", lw=2, label="Terre (annee de 365,2 jours)")
    ax.axhline(0, color="gray", lw=0.5)
    # saisons martiennes (Ls = 90, 180, 270) et perihelie (Ls = 251)
    for L, nom in ((90, "Ls = 90°"), (180, "Ls = 180°"), (270, "Ls = 270°")):
        i = np.argmin(np.abs(np.angle(np.exp(1j*(Ls - np.radians(L))))))
        ax.axvline(tm[i]/AN_TRO, color="r", lw=0.7, ls=":")
        ax.text(tm[i]/AN_TRO + 0.005, 56, nom, color="r", fontsize=9, va="top")
    i = np.argmin(np.abs(np.angle(np.exp(1j*(Ls - varpi)))))
    ax.plot(tm[i]/AN_TRO, Em[i], "o", color="r", ms=7)
    ax.annotate("perihelie de Mars\n(Ls = 251°)", (tm[i]/AN_TRO, Em[i]),
                xytext=(0.80, -34), fontsize=9, color="r",
                arrowprops=dict(arrowstyle="->", color="r", lw=0.8))
    # extrema martiens
    for i in range(1, len(Em)-1):
        if (Em[i]-Em[i-1])*(Em[i+1]-Em[i]) < 0:
            ax.annotate(f"{Em[i]:+.1f} min\n(sol {tm[i]:.0f}, Ls = {np.degrees(Ls[i]):.0f}°)",
                        (tm[i]/AN_TRO, Em[i]), xytext=(0.93, 30) if Em[i] > 0 else (0.68, -46),
                        ha="center", va="top", fontsize=9, color="r",
                        arrowprops=dict(arrowstyle="->", color="r", lw=0.8))
    ax.set_xlim(0, 1); ax.set_ylim(-60, 60)
    ax.set_xlabel("fraction de l'année écoulée depuis l'équinoxe de printemps (Ls = 0)")
    ax.set_ylabel("E(t)  (minutes)   [horloge − cadran solaire]")
    ax.set_title("Équation du temps en fonction de la date : Terre et Mars")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.savefig(os.path.join(OUT, "eqdMars_temps.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_analemmes():
    tm, Ls, Em, dm = depuis_equinoxe()
    tt, Et, dt = terre_depuis_equinoxe()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    for ax in (ax1, ax2):
        ax.set_xlim(-62, 50); ax.set_ylim(-30, 30); ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("−E(t)  (minutes)  —  l'Est à gauche")
    ax1.set_ylabel("déclinaison du Soleil (degrés)")
    ax1.plot(-Et, dt, "b-", lw=2); ax1.set_title("Terre (2025)")
    TERRE.dates_remarquables(ax1, 2025, ms=6, fontsize=9, decalages={
        "21 juin": (2, 1), "21 mars": (-17, 1), "23 sept.": (2, -3),
        "11 févr.": (-17, 0.5), "3 nov.": (2, -1), "21 déc.": (2, -2)})
    ax2.plot(-Em, dm, "r-", lw=2); ax2.set_title("Mars")
    reperes = [(0, "Ls = 0°\n(printemps N)", (2, 1.5)), (90, "Ls = 90°\n(été N)", (2, 1)),
               (180, "Ls = 180°\n(automne N)", (2, -4.5)), (270, "Ls = 270°\n(hiver N)", (-24, -4)),
               (251, "périhélie", (-16, 1.5)), (329, "E max\n(Ls = 329°)", (1.5, 1))]
    for L, nom, (dx, dy) in reperes:
        i = np.argmin(np.abs(np.angle(np.exp(1j*(Ls - np.radians(L))))))
        ax2.plot(-Em[i], dm[i], "o", color=BLEU, ms=6, zorder=5)
        ax2.text(-Em[i] + dx, dm[i] + dy, nom, color=BLEU, fontsize=9)
    fig.suptitle("Analemmes de la Terre et de Mars à la même échelle", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "analemme_mars.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_eqdt_temps()
    fig_analemmes()
    tm, Ls, Em, dm = depuis_equinoxe()
    print("Mars : E min %.1f, max %.1f min ; declinaison ±%.1f°" % (Em.min(), Em.max(), dm.max()))
    print("figures ecrites dans", OUT)
