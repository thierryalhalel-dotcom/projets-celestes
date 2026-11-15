#!/usr/bin/env python3
"""
Figures comparatives avec / sans Jupiter du chapitre "Mouvement Terre/Lune/Soleil"
=================================================================================
Produit dans resultats/ :
    fig5b_fft_params_lunaires.png   parametres lunaires + FFT, avec vs sans Jupiter
    fig1c_excentricite_fft.png      excentricite terrestre + FFT sur 2000 ans
    fig1d_jupiter_terre.png         effets de Jupiter sur l'orbite terrestre
    fig1e_jupiter_lune.png          absence d'effet de Jupiter sur la Lune

Usage :
    python3 figures_jupiter.py [--years 4000] [--dt 500]

Les deux simulations (avec et sans Jupiter) sont relancees a chaque appel ;
elles prennent une dizaine de secondes chacune.
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulation_TLS_Jupiter as S
from simulation_TLS_Jupiter import (JOUR, fft_analyse, _pics_locaux, _style_axes,
                                    FS2_SUPTITLE, FS2_TITLE, FS2_LABEL, FS2_TICK,
                                    FS2_ANNOT, FS3_SUPTITLE, FS3_TITLE, FS3_LABEL,
                                    FS3_TICK, FS3_ANNOT)

YEAR_S = 365.25 * JOUR
T_JUP = 11.862          # periode orbitale de Jupiter, en annees
ROUGE, BLEU, VERT = "red", "blue", "green"


def _style(ax, titre=None, xlabel=None, ylabel=None, dense=False):
    """dense=True : figure a 16 panneaux (fig5b), polices un cran plus petites."""
    if dense:
        _style_axes(ax, title=titre, xlabel=xlabel, ylabel=ylabel,
                    fs_title=FS3_TITLE, fs_label=FS3_LABEL, fs_tick=FS3_TICK)
    else:
        _style_axes(ax, title=titre, xlabel=xlabel, ylabel=ylabel,
                    fs_title=FS2_TITLE, fs_label=FS2_LABEL, fs_tick=FS2_TICK)


def _serie(t_s, y, span_ans):
    """Restreint une serie d'evenements aux `span_ans` premieres annees."""
    t = t_s / YEAR_S
    m = t <= span_ans
    return t[m], y[m]


def _fft_evenements(t_s, y, unite="an"):
    """FFT d'une grandeur echantillonnee aux instants d'evenements (perigees,
    passages annuels...) : on utilise l'intervalle moyen comme pas."""
    if len(y) < 8:
        return np.array([0.0]), np.array([0.0])
    dt = np.mean(np.diff(t_s))
    dt = dt / YEAR_S if unite == "an" else dt / JOUR
    return fft_analyse(y, dt)


def _log(sp):
    return np.where(sp > 0, sp, 1e-14)


def _marque_jupiter(ax, harmonique=True):
    ax.axvline(1.0/T_JUP, color=VERT, ls="--", lw=1.8)
    if harmonique:
        ax.axvline(2.0/T_JUP, color=VERT, ls=":", lw=1.8)


# ===================================================================
# Fig 1d : effets de Jupiter sur l'orbite terrestre
# ===================================================================
def fig_jupiter_terre(avec, sans, out_dir, span_ans=60.0, span_rayon=500.0):
    """Colonne de gauche : signaux temporels ; colonne de droite : FFT sur la
    totalite de la simulation.

    Les deux premiers panneaux sont traces sur `span_ans` annees (60 : cinq
    periodes de Jupiter, de quoi voir le battement sans tout confondre). Le
    rayon au perihelie garde `span_rayon` annees (500) car son effet le plus
    net est une derive seculaire, invisible sur quelques dizaines d'annees.
    """
    lignes = [
        ("Excentricite terrestre", "$e_T$", span_ans,
         lambda r: (r["perihelion"][:,0], r["perihelion"][:,2])),
        ("Duree de l'annee siderale", "jours $-365$", span_ans,
         lambda r: (r["year"][:,0], r["year"][:,1] - 365.0)),
        ("Rayon au perihelie", "M km", span_rayon,
         lambda r: (r["perihelion"][:,0], r["perihelion"][:,1]/1e9)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Effets de Jupiter sur l'orbite terrestre : analyse FFT comparee",
                 fontsize=FS2_SUPTITLE, fontweight="bold")

    for k, (titre, ylab, span, extrait) in enumerate(lignes):
        ax_g, ax_d = axes[k, 0], axes[k, 1]
        lw_g = 1.4 if span <= 100 else 0.8
        for res, coul, lbl in ((avec, ROUGE, "avec Jupiter"), (sans, BLEU, "sans Jupiter")):
            t_s, y = extrait(res)
            t, ys = _serie(t_s, y, span)
            ax_g.plot(t, ys, color=coul, lw=lw_g, alpha=0.85, label=lbl)
            f, sp = _fft_evenements(t_s, y, "an")
            ax_d.semilogy(f, _log(sp), color=coul, lw=0.7, alpha=0.8, label=lbl)
        _style(ax_g, f"{titre} ({span:.0f} ans)", "annees", ylab)
        ax_g.set_xlim(0, span); ax_g.legend(fontsize=FS2_ANNOT, loc="upper right")
        _style(ax_d, f"FFT {titre.lower()}", "frequence (an$^{-1}$)")
        ax_d.set_xlim(0, 0.25); ax_d.legend(fontsize=FS2_ANNOT, loc="upper right")
        _marque_jupiter(ax_d)
        y0, y1 = ax_d.get_ylim()
        ax_d.set_ylim(y0, y1*30)          # place pour l'annotation
        ax_d.annotate(f"Jupiter\n{T_JUP:.2f} a", xy=(1.0/T_JUP, np.sqrt(y0*y1)),
                      xytext=(1.0/T_JUP + 0.03, y1), color=VERT,
                      fontsize=FS2_ANNOT, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color=VERT, lw=1.5))

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(out_dir, "fig1d_jupiter_terre.png"), dpi=150)
    plt.close(fig)
    print("  -> fig1d_jupiter_terre.png")


# ===================================================================
# Fig 1e : absence d'effet de Jupiter sur la Lune
# ===================================================================
def fig_jupiter_lune(avec, sans, out_dir, span_ans=20.0):
    """Colonne de gauche : signaux sur `span_ans` annees (20 par defaut, pour
    que les oscillations restent visibles) ; colonne de droite : FFT sur la
    totalite de la simulation."""
    lignes = [
        ("Excentricite lunaire", "$e_L$", 5.0,
         lambda r: (r["perigee"][:,0], r["perigee"][:,2])),
        ("Inclinaison lunaire", "deg", 3.0,
         lambda r: (r["node"][:,0], r["node"][:,1])),
        ("Periode siderale lunaire", "jours", 5.0,
         lambda r: (r["sidereal"][:,0], r["sidereal"][:,1])),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Absence d'effet de Jupiter sur la Lune : analyse FFT comparee",
                 fontsize=FS2_SUPTITLE, fontweight="bold")

    for k, (titre, ylab, fmax, extrait) in enumerate(lignes):
        ax_g, ax_d = axes[k, 0], axes[k, 1]
        for res, coul, lbl in ((avec, ROUGE, "avec Jupiter"), (sans, BLEU, "sans Jupiter")):
            t_s, y = extrait(res)
            t, ys = _serie(t_s, y, span_ans)
            ax_g.plot(t, ys, color=coul, lw=1.1, alpha=0.8, label=lbl)
            f, sp = _fft_evenements(t_s, y, "an")
            ax_d.semilogy(f, _log(sp), color=coul, lw=0.6, alpha=0.7, label=lbl)
        _style(ax_g, f"{titre} ({span_ans:.0f} ans)", "annees", ylab)
        ax_g.set_xlim(0, span_ans); ax_g.legend(fontsize=FS2_ANNOT, loc="upper right")
        _style(ax_d, f"FFT {titre.lower()}", "frequence (an$^{-1}$)")
        ax_d.set_xlim(0, fmax); ax_d.legend(fontsize=FS2_ANNOT, loc="upper right")
        _marque_jupiter(ax_d)
        y0, y1 = ax_d.get_ylim()
        ax_d.set_ylim(y0, y1*10)
        ax_d.annotate("Jupiter ?\n(absent)", xy=(1.0/T_JUP, np.sqrt(y0*y1)),
                      xytext=(0.10*fmax, y1/50.0), color=VERT, fontsize=FS2_ANNOT,
                      arrowprops=dict(arrowstyle="->", color=VERT, ls="--", lw=1.5))

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(out_dir, "fig1e_jupiter_lune.png"), dpi=150)
    plt.close(fig)
    print("  -> fig1e_jupiter_lune.png")


# ===================================================================
# Fig 1c : excentricite terrestre + FFT sur 2000 ans
# ===================================================================
def fig_excentricite_fft(avec, sans, out_dir, span_ans=2000.0):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    fig.suptitle("Excentricite terrestre + FFT : effet de Jupiter",
                 fontsize=FS2_SUPTITLE, fontweight="bold")

    spectres = {}
    for k, (res, coul, lbl) in enumerate(((avec, ROUGE, "Avec Jupiter"),
                                          (sans, BLEU, "Sans Jupiter"))):
        ph = res["perihelion"]
        t, e = _serie(ph[:,0], ph[:,2], span_ans)
        ax = fig.add_subplot(gs[k, 0])
        ax.plot(t, e, color=coul, lw=0.5)
        _style(ax, f"{lbl} ({span_ans:.0f} ans)", "annees", "excentricite")
        ax.set_xlim(0, span_ans)

        m = ph[:,0] <= span_ans * YEAR_S
        f, sp = _fft_evenements(ph[m,0], ph[m,2], "an")
        spectres[lbl] = (f, sp)
        ax = fig.add_subplot(gs[k, 1])
        ax.semilogy(f, _log(sp), color=coul, lw=0.8)
        ax.set_xlim(0, 0.5); ax.set_ylim(1e-7, None)
        _style(ax, f"FFT ({lbl.lower()}, echelle log)", "an$^{-1}$")
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1*30)
        # deux pics dominants annotes (on ecarte la derive tres basse frequence)
        idx = [j for j in _pics_locaux(f, sp) if f[j] > 0.05]
        for j in sorted(idx, key=lambda j: -sp[j])[:2]:
            ax.annotate(f"T={1/f[j]:.2f} a", xy=(f[j], sp[j]), fontsize=FS2_ANNOT,
                        color=coul, xytext=(f[j]-0.13, sp[j]*4),
                        arrowprops=dict(arrowstyle="->", color=coul, lw=1.0))

    ax = fig.add_subplot(gs[2, :])
    for lbl, coul in (("Avec Jupiter", ROUGE), ("Sans Jupiter", BLEU)):
        f, sp = spectres[lbl]
        ax.semilogy(f, _log(sp), color=coul, lw=1.0,
                    label="avec Jupiter" if coul == ROUGE else "sans Jupiter")
    ax.set_xlim(0, 0.20); ax.set_ylim(5e-7, None)
    y0b, y1b = ax.get_ylim(); ax.set_ylim(y0b, y1b*6)
    _style(ax, "Zoom basses frequences : superposition avec (rouge) et sans (bleu) Jupiter",
           "frequence (an$^{-1}$)", "amplitude")
    ax.legend(fontsize=FS2_ANNOT, loc="upper right")
    _marque_jupiter(ax)
    y0, y1 = ax.get_ylim()
    ax.annotate(f"Jupiter\nT = {T_JUP:.2f} ans", xy=(1.0/T_JUP, np.sqrt(y0*y1)),
                xytext=(1.0/T_JUP + 0.015, y1/8.0), color=VERT,
                fontsize=FS2_ANNOT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VERT, lw=1.5))
    ax.annotate(f"Jupiter/2\nT = {T_JUP/2:.2f} ans", xy=(2.0/T_JUP, np.sqrt(y0*y1)),
                xytext=(2.0/T_JUP - 0.045, y1/8.0), color=VERT, fontsize=FS2_ANNOT,
                arrowprops=dict(arrowstyle="->", color=VERT, lw=1.5))

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(out_dir, "fig1c_excentricite_fft.png"), dpi=150)
    plt.close(fig)
    print("  -> fig1c_excentricite_fft.png")


# ===================================================================
# Fig 1f : derive seculaire de l'excentricite terrestre sur 4000 ans
# ===================================================================
def fit_derive_eT(res, T_period=T_JUP):
    """Regression e_T(t) = a + b.t + c.cos(2 pi t/T) + d.sin(2 pi t/T) sur les
    excentricites au perihelie. Renvoie a, b (pente en /an), son ecart-type,
    et la fonction tendance a+b.t evaluee aux memes instants t (en annees)."""
    ph = res["perihelion"]
    t = ph[:,0] / YEAR_S
    e = ph[:,2]
    omega = 2 * np.pi / T_period
    A = np.column_stack([np.ones_like(t), t, np.cos(omega*t), np.sin(omega*t)])
    coef, _, _, _ = np.linalg.lstsq(A, e, rcond=None)
    a, b, c, d = coef
    resid = e - A @ coef
    b_err = np.sqrt((resid.std()**2) * np.linalg.inv(A.T @ A)[1, 1])
    return dict(t=t, e=e, a=a, b=b, b_err=b_err, tendance=a + b*t)


def fig_derive_eT_jupiter(avec, sans, out_dir):
    """Recherche d'une derive seculaire de e_T sur la totalite de la simulation
    (4000 ans), avec et sans Jupiter : regression lineaire + sinusoide a la
    periode de Jupiter, superposee au signal brut (excentricite au perihelie).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Dérive séculaire de l'excentricité terrestre $e_T$ (4000 ans)",
                 fontsize=FS2_SUPTITLE - 3, fontweight="bold")

    for ax, (res, coul, lbl) in zip(axes, ((avec, ROUGE, "Avec Jupiter"),
                                            (sans, BLEU, "Sans Jupiter"))):
        r = fit_derive_eT(res)
        ax.plot(r["t"], r["e"], color=coul, lw=0.4, alpha=0.8)
        ax.plot(r["t"], r["tendance"], color="black", lw=2.2, ls="--",
                label=f"tendance : {r['b']*1e6:+.2f} $\\times 10^{{-6}}$/an "
                      f"({abs(r['b'])/r['b_err']:.0f}$\\sigma$)")
        _style(ax, f"{lbl}", "annees" if ax is axes[-1] else None, "$e_T$")
        ax.set_xlim(r["t"].min(), r["t"].max())
        ax.legend(fontsize=FS2_ANNOT, loc="upper right", framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig1f_derive_eT_jupiter.png"), dpi=150)
    plt.close(fig)
    print("  -> fig1f_derive_eT_jupiter.png")


# ===================================================================
# Fig 5b : parametres lunaires + FFT, avec vs sans Jupiter
# ===================================================================
def _etiquette_periode(T_j):
    return f"{T_j/365.25:.1f} a" if T_j > 365.0 else f"{T_j:.0f} j"


def fig_fft_params_lunaires_3corps(sans, out_dir, span_ans=20.0, n_pics=4):
    """Version a trois corps (sans Jupiter) de la figure precedente, utilisee
    dans la section sur les inegalites lunaires : 4 lignes (e_L, i, T_L, v_p)
    x 2 colonnes (signal sur `span_ans` annees, FFT sur toute la simulation).
    Moins dense que la version comparative, donc polices plus grandes ; la
    comparaison avec / sans Jupiter est reservee a l'exercice (fig1e)."""
    lignes = [
        ("Excentricité $e_L$",       "Spectre de $e_L$",  "$e_L$", lambda r: (r["perigee"][:,0],  r["perigee"][:,2])),
        ("Inclinaison $i$",          "Spectre de $i$",    "deg",   lambda r: (r["node"][:,0],     r["node"][:,1])),
        ("Période sidérale $T_L$",   "Spectre de $T_L$",  "jours", lambda r: (r["sidereal"][:,0], r["sidereal"][:,1])),
        ("Vitesse au périgée $v_p$", "Spectre de $v_p$",  "km/s",  lambda r: (r["perigee"][:,0],  r["perigee"][:,3])),
    ]
    up = 3   # figure imprimee reduite : polices grossies (cf. plot_fig5_ensemble)
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    fig.suptitle("Paramètres orbitaux lunaires (sans Jupiter)",
                 fontsize=FS2_SUPTITLE + up, fontweight="bold")

    for k, (titre, titre_fft, ylab, extrait) in enumerate(lignes):
        t_s, y = extrait(sans)
        t, ys = _serie(t_s, y, span_ans)
        ax = axes[k, 0]
        ax.plot(t, ys, color=BLEU, lw=1.2)
        _style_axes(ax, title=titre, ylabel=ylab,
                    xlabel="années" if k == 3 else None,
                    fs_title=FS2_TITLE + up, fs_label=FS2_LABEL + up, fs_tick=FS2_TICK + up)
        ax.set_xlim(0, span_ans)

        f, sp = _fft_evenements(t_s, y, "jour")
        ax = axes[k, 1]
        ax.semilogy(f, _log(sp), color=BLEU, lw=0.6)
        ax.set_xlim(0, 0.02)
        y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1*3000)
        _style_axes(ax, title=titre_fft,
                    xlabel="fréquence (jour$^{-1}$)" if k == 3 else None,
                    fs_title=FS2_TITLE + up, fs_label=FS2_LABEL + up, fs_tick=FS2_TICK + up)
        idx = _pics_locaux(f, sp)
        pics = sorted(sorted(idx, key=lambda j: -sp[j])[:n_pics], key=lambda j: f[j])
        for m, j in enumerate(pics):
            # etiquettes sur deux rangees au-dessus des pics, en quinconce,
            # pour eviter les chevauchements
            ax.annotate(_etiquette_periode(1.0/f[j]), xy=(f[j], sp[j]),
                        fontsize=FS2_ANNOT + up, color=BLEU, ha="center",
                        xytext=(f[j], y1*(20 if m % 2 == 0 else 400)),
                        arrowprops=dict(arrowstyle="->", color=BLEU, lw=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(out_dir, "fig5b_fft_params_lunaires.png"), dpi=150)
    plt.close(fig)
    print("  -> fig5b_fft_params_lunaires.png (3 corps)")


def fig_fft_params_lunaires(avec, sans, out_dir, span_ans=20.0, n_pics=5):
    """Signaux temporels sur `span_ans` annees (20 par defaut : sur 500 ans les
    oscillations se confondent) ; les FFT, elles, portent toujours sur la
    totalite de la simulation."""
    lignes = [
        ("Excentricite $e_L$", "$e_L$",
         lambda r: (r["perigee"][:,0], r["perigee"][:,2])),
        ("Inclinaison $i$", "deg",
         lambda r: (r["node"][:,0], r["node"][:,1])),
        ("Periode siderale $T_L$", "jours",
         lambda r: (r["sidereal"][:,0], r["sidereal"][:,1])),
        ("Vitesse perigee $v_p$", "km/s",
         lambda r: (r["perigee"][:,0], r["perigee"][:,3])),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    fig.suptitle("Parametres orbitaux lunaires : avec Jupiter (rouge) vs sans Jupiter (bleu)",
                 fontsize=FS3_SUPTITLE, fontweight="bold")

    for k, (titre, ylab, extrait) in enumerate(lignes):
        for c, (res, coul, lbl) in enumerate(((avec, ROUGE, "Avec Jupiter"),
                                              (sans, BLEU, "Sans Jupiter"))):
            t_s, y = extrait(res)
            t, ys = _serie(t_s, y, span_ans)
            ax = axes[k, c]
            ax.plot(t, ys, color=coul, lw=1.2)
            _style(ax, f"{lbl} ({span_ans:.0f} ans)" if k == 0 else None,
                   "annees" if k == 3 else None, ylab if c == 0 else None, dense=True)
            ax.set_xlim(0, span_ans)
            if c == 0:
                ax.text(0.02, 0.97, titre, transform=ax.transAxes, fontsize=FS3_ANNOT,
                        va="top", fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

            f, sp = _fft_evenements(t_s, y, "jour")
            ax = axes[k, 2+c]
            ax.semilogy(f, _log(sp), color=coul, lw=0.5)
            ax.set_xlim(0, 0.025)
            y0, y1 = ax.get_ylim(); ax.set_ylim(y0, y1*300)
            _style(ax, f"FFT {lbl.lower()}" if k == 0 else None,
                   "frequence (jour$^{-1}$)" if k == 3 else None, dense=True)
            idx = _pics_locaux(f, sp)
            for j in sorted(idx, key=lambda j: -sp[j])[:n_pics]:
                ax.annotate(_etiquette_periode(1.0/f[j]), xy=(f[j], sp[j]),
                            fontsize=FS3_ANNOT, color=coul,
                            xytext=(f[j]-0.0015, sp[j]*8),
                            arrowprops=dict(arrowstyle="->", color=coul, lw=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(out_dir, "fig5b_fft_params_lunaires_avec_sans.png"), dpi=150)
    plt.close(fig)
    print("  -> fig5b_fft_params_lunaires_avec_sans.png")


# ===================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Figures comparatives avec/sans Jupiter")
    p.add_argument("--years", type=float, default=4000.0)
    p.add_argument("--dt", type=float, default=500.0)
    p.add_argument("--store-every", type=int, default=400)
    a = p.parse_args()

    out_dir = os.path.join(HERE, "resultats")
    os.makedirs(out_dir, exist_ok=True)

    base = dict(dt=a.dt, years=a.years, store_every=a.store_every, tag=None)
    print("Simulation AVEC Jupiter...")
    avec = S.run_simulation(argparse.Namespace(no_jupiter=False, **base))
    print("Simulation SANS Jupiter...")
    sans = S.run_simulation(argparse.Namespace(no_jupiter=True, **base))

    print("Figures...")
    fig_jupiter_terre(avec, sans, out_dir)
    fig_jupiter_lune(avec, sans, out_dir)
    fig_excentricite_fft(avec, sans, out_dir)
    fig_fft_params_lunaires(avec, sans, out_dir)
    print("Termine.")
