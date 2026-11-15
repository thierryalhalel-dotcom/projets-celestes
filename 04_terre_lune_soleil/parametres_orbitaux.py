#!/usr/bin/env python3
"""
Visualisation des parametres orbitaux de la Terre et de la Lune
===============================================================

Charge un fichier .npz produit par simulation_TLS_Jupiter.py et trace
deux figures (Terre et Lune) sur une plage temporelle reglable.

Usage :
    python3 parametres_orbitaux.py                          # defaut : 500 ans, avec Jupiter
    python3 parametres_orbitaux.py --tmin 0 --tmax 200      # 200 premieres annees
    python3 parametres_orbitaux.py --tmax 50                 # zoom 50 ans
    python3 parametres_orbitaux.py --sans-jupiter            # donnees sans Jupiter
    python3 parametres_orbitaux.py --both                    # superpose avec/sans Jupiter
    python3 parametres_orbitaux.py --save                    # sauvegarde PNG au lieu d'afficher
"""

import numpy as np
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
JOUR = 86400.0
YEAR_S = 365.25 * JOUR


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d


def parse_args():
    p = argparse.ArgumentParser(
        description="Parametres orbitaux Terre/Lune - visualisation interactive")
    p.add_argument("--tmin", type=float, default=0,
                   help="Debut de la plage (annees, defaut 0)")
    p.add_argument("--tmax", type=float, default=500,
                   help="Fin de la plage (annees, defaut 500)")
    p.add_argument("--sans-jupiter", action="store_true",
                   help="Utiliser les donnees sans Jupiter")
    p.add_argument("--both", action="store_true",
                   help="Superposer avec et sans Jupiter")
    p.add_argument("--npz", type=str, default=None,
                   help="Fichier .npz specifique")
    p.add_argument("--save", action="store_true",
                   help="Sauvegarder en PNG au lieu d'afficher")
    p.add_argument("--dpi", type=int, default=150,
                   help="Resolution des figures (defaut 150)")
    return p.parse_args()


def select_range(t_seconds, tmin_yr, tmax_yr):
    t_yr = t_seconds / YEAR_S
    mask = (t_yr >= tmin_yr) & (t_yr <= tmax_yr)
    return mask, t_yr


def plot_terre(datasets, tmin, tmax, save=False, dpi=150):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_data = len(datasets)
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle(f"Parametres orbitaux de la Terre ({tmin:.0f} - {tmax:.0f} ans)",
                 fontsize=14, fontweight="bold")

    for label, color, alpha, d in datasets:
        yr = d["year"]
        ph = d["perihelion"]
        t_samp = d["t_samp"]
        rST = d["rST_samp"]
        dt_s = float(d["dt"]) * int(d["store_every"])

        # Annee siderale
        t_yr_y = yr[:, 0] / YEAR_S
        mask_y = (t_yr_y >= tmin) & (t_yr_y <= tmax)
        if np.any(mask_y):
            axes[0].plot(t_yr_y[mask_y], yr[mask_y, 1], "-",
                         color=color, lw=1.0, alpha=alpha, label=label)
        axes[0].set_ylabel("jours")
        axes[0].set_title("Annee siderale")

        # Rayon orbital
        t_yr_s = t_samp / YEAR_S
        mask_s = (t_yr_s >= tmin) & (t_yr_s <= tmax)
        if np.any(mask_s):
            sub = max(1, np.sum(mask_s) // 5000)
            idx = np.where(mask_s)[0][::sub]
            axes[1].plot(t_yr_s[idx], rST[idx] / 1e9, "-",
                         color=color, lw=0.6, alpha=alpha, label=label)
        axes[1].set_ylabel("Mkm")
        axes[1].set_title("Distance Soleil-Terre")

        # Excentricite au perihelie
        t_yr_p = ph[:, 0] / YEAR_S
        mask_p = (t_yr_p >= tmin) & (t_yr_p <= tmax)
        if np.any(mask_p):
            axes[2].plot(t_yr_p[mask_p], ph[mask_p, 2], "o-",
                         color=color, lw=1.0, markersize=2.5, alpha=alpha, label=label)
        axes[2].set_ylabel("$e_T$")
        axes[2].set_title("Excentricite terrestre (au perihelie)")

        # Rayon au perihelie
        if np.any(mask_p):
            axes[3].plot(t_yr_p[mask_p], ph[mask_p, 1] / 1e9, "o-",
                         color=color, lw=1.0, markersize=2.5, alpha=alpha, label=label)
        axes[3].set_ylabel("Mkm")
        axes[3].set_title("Rayon au perihelie")
        axes[3].set_xlabel("annees")

    for ax in axes:
        ax.set_xlim(tmin, tmax)
        ax.grid(True, alpha=0.3)
        if n_data > 1:
            ax.legend(fontsize=8)

    # Lignes de reference
    axes[0].axhline(365.256, color="gray", ls="--", lw=0.8, label="ref 365.256 j")

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, "resultats",
                            f"params_terre_{tmin:.0f}_{tmax:.0f}ans.png")
        fig.savefig(path, dpi=dpi)
        print(f"Sauvegarde : {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_lune(datasets, tmin, tmax, save=False, dpi=150):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_data = len(datasets)
    fig, axes = plt.subplots(5, 1, figsize=(15, 20))
    fig.suptitle(f"Parametres orbitaux de la Lune ({tmin:.0f} - {tmax:.0f} ans)",
                 fontsize=14, fontweight="bold")

    for label, color, alpha, d in datasets:
        pr = d["perigee"]
        nd = d["node"]
        sd = d["sidereal"]

        # Excentricite
        t_yr = pr[:, 0] / YEAR_S
        mask = (t_yr >= tmin) & (t_yr <= tmax)
        if np.any(mask):
            axes[0].plot(t_yr[mask], pr[mask, 2], "-",
                         color=color, lw=0.5, alpha=alpha, label=label)
        axes[0].set_ylabel("$e_L$")
        axes[0].set_title("Excentricite lunaire (au perigee)")

        # Inclinaison
        t_yr_n = nd[:, 0] / YEAR_S
        mask_n = (t_yr_n >= tmin) & (t_yr_n <= tmax)
        if np.any(mask_n):
            axes[1].plot(t_yr_n[mask_n], nd[mask_n, 1], "-",
                         color=color, lw=0.5, alpha=alpha, label=label)
        axes[1].set_ylabel("deg")
        axes[1].set_title("Inclinaison (au noeud ascendant)")

        # Periode siderale
        t_yr_s = sd[:, 0] / YEAR_S
        mask_s = (t_yr_s >= tmin) & (t_yr_s <= tmax)
        if np.any(mask_s):
            axes[2].plot(t_yr_s[mask_s], sd[mask_s, 1], "-",
                         color=color, lw=0.5, alpha=alpha, label=label)
        axes[2].set_ylabel("jours")
        axes[2].set_title("Periode siderale")
        axes[2].axhline(27.322, color="gray", ls="--", lw=0.8)

        # Vitesse au perigee
        if np.any(mask):
            axes[3].plot(t_yr[mask], pr[mask, 3], "-",
                         color=color, lw=0.5, alpha=alpha, label=label)
        axes[3].set_ylabel("km/s")
        axes[3].set_title("Vitesse au perigee")

        # Longitude du noeud (sin Omega)
        if np.any(mask_n):
            axes[4].plot(t_yr_n[mask_n], nd[mask_n, 2], "-",
                         color=color, lw=0.6, alpha=alpha, label=label)
        axes[4].set_ylabel("sin($\\Omega$)")
        axes[4].set_title("Longitude du noeud ascendant (regression, T = 18.6 ans)")
        axes[4].set_xlabel("annees")

    for ax in axes:
        ax.set_xlim(tmin, tmax)
        ax.grid(True, alpha=0.3)
        if n_data > 1:
            ax.legend(fontsize=8)

    fig.tight_layout()
    if save:
        path = os.path.join(HERE, "resultats",
                            f"params_lune_{tmin:.0f}_{tmax:.0f}ans.png")
        fig.savefig(path, dpi=dpi)
        print(f"Sauvegarde : {path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()

    # Charger les donnees
    datasets = []

    if args.npz:
        d = load_data(args.npz)
        jup = "avec Jupiter" if bool(d["jupiter"]) else "sans Jupiter"
        datasets.append((jup, "red", 0.8, d))
    elif args.both:
        path_a = os.path.join(HERE, "resultats", "simulation_data_avecJupiter.npz")
        path_s = os.path.join(HERE, "resultats", "simulation_data_sansJupiter.npz")
        if not os.path.exists(path_a) or not os.path.exists(path_s):
            print("Erreur : fichiers .npz manquants dans resultats/")
            sys.exit(1)
        datasets.append(("avec Jupiter", "red", 0.7, load_data(path_a)))
        datasets.append(("sans Jupiter", "blue", 0.5, load_data(path_s)))
    elif args.sans_jupiter:
        path = os.path.join(HERE, "resultats", "simulation_data_sansJupiter.npz")
        datasets.append(("sans Jupiter", "blue", 0.8, load_data(path)))
    else:
        path = os.path.join(HERE, "resultats", "simulation_data_avecJupiter.npz")
        datasets.append(("avec Jupiter", "red", 0.8, load_data(path)))

    tmin, tmax = args.tmin, args.tmax
    print(f"Plage : {tmin:.0f} - {tmax:.0f} ans")
    print(f"Jeux de donnees : {', '.join(lbl for lbl, _, _, _ in datasets)}")

    plot_terre(datasets, tmin, tmax, save=args.save, dpi=args.dpi)
    plot_lune(datasets, tmin, tmax, save=args.save, dpi=args.dpi)


if __name__ == "__main__":
    main()
