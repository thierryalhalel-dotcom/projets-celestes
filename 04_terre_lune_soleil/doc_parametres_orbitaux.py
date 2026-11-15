#!/usr/bin/env python3
"""
Documentation PDF du programme parametres_orbitaux.py
=====================================================

Genere un PDF expliquant :
  - Les options en ligne de commande
  - Le fonctionnement interne (chargement, selection, trace)
  - Comment modifier les couleurs et epaisseurs de trait

Produit : resultats/doc_parametres_orbitaux.pdf

Usage :
    python3 doc_parametres_orbitaux.py
"""

import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
JOUR = 86400.0
YEAR_S = 365.25 * JOUR


def page_titre(pdf):
    """Page 1 : titre et presentation generale."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.93,
            "parametres_orbitaux.py",
            transform=ax.transAxes, fontsize=22, ha="center",
            fontweight="bold", fontfamily="monospace")
    ax.text(0.5, 0.87,
            "Visualisation des parametres orbitaux de la Terre et de la Lune",
            transform=ax.transAxes, fontsize=14, ha="center",
            style="italic", color="gray")

    txt = """
PRESENTATION
============

Ce programme charge les donnees de simulation produites par
simulation_TLS_Jupiter.py (fichiers .npz) et trace deux figures
multi-panneaux montrant l'evolution des parametres orbitaux :

  FIGURE TERRE (4 panneaux) :
    1. Annee siderale (jours)        — duree de chaque revolution
    2. Distance Soleil-Terre (Mkm)   — rayon orbital instantane
    3. Excentricite terrestre (e_T)   — mesure au perihelie
    4. Rayon au perihelie (Mkm)      — distance minimale annuelle

  FIGURE LUNE (5 panneaux) :
    1. Excentricite lunaire (e_L)    — mesure au perigee
    2. Inclinaison (deg)             — au noeud ascendant
    3. Periode siderale (jours)      — duree du mois sideral
    4. Vitesse au perigee (km/s)     — vitesse maximale orbitale
    5. sin(Omega)                    — longitude du noeud ascendant
                                       (regression, T = 18.6 ans)

MODES DE FONCTIONNEMENT :
  - Affichage interactif (defaut) : fenetre matplotlib zoomable
  - Sauvegarde PNG (--save)       : ecriture dans resultats/
  - Jeu unique ou superposition   : avec/sans Jupiter ou les deux

DONNEES REQUISES :
  resultats/simulation_data_avecJupiter.npz   (par defaut)
  resultats/simulation_data_sansJupiter.npz   (option --sans-jupiter)
"""
    ax.text(0.05, 0.78, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_options(pdf):
    """Page 2 : tableau des options CLI avec exemples."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.96, "Options en ligne de commande",
            transform=ax.transAxes, fontsize=16, ha="center",
            fontweight="bold")

    header = ["Option", "Defaut", "Type", "Description"]
    rows = [
        ["--tmin FLOAT",   "0",    "float", "Debut de la plage temporelle (annees)"],
        ["--tmax FLOAT",   "500",  "float", "Fin de la plage temporelle (annees)"],
        ["--sans-jupiter", "(off)", "flag",  "Utiliser simulation_data_sansJupiter.npz"],
        ["--both",         "(off)", "flag",  "Superposer avec et sans Jupiter"],
        ["--npz FICHIER",  "auto",  "str",  "Chemin vers un fichier .npz specifique"],
        ["--save",         "(off)", "flag",  "Sauvegarder PNG au lieu d'afficher"],
        ["--dpi ENTIER",   "150",   "int",  "Resolution des PNG (points par pouce)"],
    ]

    colors_table = [["#e8f4fd"] * 4] * len(rows)
    table = ax.table(cellText=rows, colLabels=header,
                     cellColours=colors_table,
                     colColours=["#4a86c8"] * 4,
                     loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(color="white", fontweight="bold")
        if j == 0:
            cell.set_text_props(fontfamily="monospace")

    y_ex = 0.42
    ax.text(0.05, y_ex, "EXEMPLES D'UTILISATION",
            transform=ax.transAxes, fontsize=13, fontweight="bold")

    examples = """
# Lancement par defaut : avec Jupiter, 0-500 ans, affichage interactif
python3 parametres_orbitaux.py

# Zoom sur les 50 premieres annees
python3 parametres_orbitaux.py --tmin 0 --tmax 50

# Plage 100-300 ans, sauvegarde PNG haute resolution
python3 parametres_orbitaux.py --tmin 100 --tmax 300 --save --dpi 300

# Donnees sans Jupiter
python3 parametres_orbitaux.py --sans-jupiter

# Superposition avec/sans Jupiter pour comparer
python3 parametres_orbitaux.py --both --tmax 200

# Fichier .npz personnalise
python3 parametres_orbitaux.py --npz resultats/simulation_data_avecJupiter_fin.npz

# Combinaison : zoom + superposition + sauvegarde
python3 parametres_orbitaux.py --both --tmin 0 --tmax 100 --save --dpi 200
"""
    ax.text(0.05, y_ex - 0.04, examples, transform=ax.transAxes,
            fontsize=8.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", alpha=0.9))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_fonctionnement(pdf):
    """Page 3 : schema de fonctionnement interne."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.96, "Fonctionnement interne",
            transform=ax.transAxes, fontsize=16, ha="center",
            fontweight="bold")

    schema = """
FLUX D'EXECUTION
================

  main()
    |
    +-- parse_args()           Analyse des arguments CLI
    |                          (argparse)
    |
    +-- Chargement des donnees
    |   |
    |   +-- --npz fourni ?      --> load_data(args.npz)
    |   +-- --both ?            --> load_data(avecJupiter) + load_data(sansJupiter)
    |   +-- --sans-jupiter ?    --> load_data(sansJupiter)
    |   +-- (defaut)            --> load_data(avecJupiter)
    |
    |   Chaque jeu = (label, couleur, alpha, donnees_npz)
    |                 "avec Jupiter"  "red"   0.8   {...}
    |                 "sans Jupiter"  "blue"  0.5   {...}
    |
    +-- plot_terre(datasets, tmin, tmax)
    |   |
    |   +-- Pour CHAQUE dataset dans datasets :
    |   |     Panneau 0 : year[:, 1]       --> annee siderale
    |   |     Panneau 1 : rST_samp / 1e9   --> distance ST (sous-echantillonne)
    |   |     Panneau 2 : perihelion[:, 2]  --> excentricite e_T
    |   |     Panneau 3 : perihelion[:, 1]  --> rayon perihelie
    |   |
    |   +-- Masque temporel : tmin <= t_annees <= tmax
    |   +-- Sous-echantillonnage : max 5000 points par courbe (panneau 1)
    |   +-- Ligne de reference : 365.256 j (panneau 0)
    |   +-- --save ? --> fig.savefig(PNG)    sinon --> plt.show()
    |
    +-- plot_lune(datasets, tmin, tmax)
        |
        +-- Pour CHAQUE dataset :
        |     Panneau 0 : perigee[:, 2]   --> excentricite e_L
        |     Panneau 1 : node[:, 1]      --> inclinaison i
        |     Panneau 2 : sidereal[:, 1]  --> periode siderale
        |     Panneau 3 : perigee[:, 3]   --> vitesse perigee
        |     Panneau 4 : node[:, 2]      --> sin(Omega)
        |
        +-- Ligne de reference : 27.322 j (panneau 2)
        +-- --save ? --> fig.savefig(PNG)    sinon --> plt.show()

CLES NPZ UTILISEES
===================
  t_samp, rST_samp         --> donnees echantillonnees (N ~ 2.5M points)
  year[:, 0:2]              --> (t, duree_annee)          ~4000 evenements
  perihelion[:, 0:3]        --> (t, r_ST, e_T)            ~4000 evenements
  perigee[:, 0:4]           --> (t, r_TL, e_L, v_perig)   ~53000 evenements
  node[:, 0:3]              --> (t, incl, sin_Omega)       ~53000 evenements
  sidereal[:, 0:2]          --> (t, duree_mois)            ~53000 evenements
"""
    ax.text(0.03, 0.88, schema, transform=ax.transAxes,
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_structure_datasets(pdf):
    """Page 4 : structure datasets et logique de superposition."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.96, "Structure des datasets et logique de trace",
            transform=ax.transAxes, fontsize=16, ha="center",
            fontweight="bold")

    txt = """
STRUCTURE D'UN DATASET
======================

Chaque jeu de donnees est un tuple de 4 elements :

    (label,            color,    alpha,   data)
     str               str       float    NpzFile
     "avec Jupiter"    "red"     0.8      d = np.load(...)

Ces 4 valeurs controlent TOUT l'aspect visuel du trace :

  label  -->  texte affiche dans la legende (ax.legend)
  color  -->  couleur du trait et des marqueurs
  alpha  -->  transparence (0 = invisible, 1 = opaque)
  data   -->  dictionnaire numpy avec les tableaux de la simulation


LOGIQUE DE SUPERPOSITION (mode --both)
=======================================

datasets = [
    ("avec Jupiter",  "red",   0.7,  load_data(avecJupiter.npz)),
    ("sans Jupiter",  "blue",  0.5,  load_data(sansJupiter.npz)),
]

La boucle  for label, color, alpha, d in datasets:  trace chaque
jeu sur les MEMES axes. Le premier jeu (rouge, alpha=0.7) est trace
en premier, le second (bleu, alpha=0.5) par-dessus. L'alpha plus
faible du second evite de masquer le premier.


PARAMETRES DE TRACE DANS LE CODE (valeurs par defaut)
=====================================================

FIGURE TERRE :
  Panneau 0 (annee sid.)  :  "-"  lw=1.0                 (ligne continue)
  Panneau 1 (distance ST) :  "-"  lw=0.6                 (ligne fine)
  Panneau 2 (excentricite):  "o-" lw=1.0  markersize=2.5 (ligne + points)
  Panneau 3 (r perihelie) :  "o-" lw=1.0  markersize=2.5 (ligne + points)
  Grille                  :  grid(True, alpha=0.3)
  Ref. annee siderale     :  axhline  color="gray" ls="--" lw=0.8

FIGURE LUNE :
  Panneau 0 (excentricite):  "-"  lw=0.5  (ligne tres fine)
  Panneau 1 (inclinaison) :  "-"  lw=0.5
  Panneau 2 (per. sider.) :  "-"  lw=0.5
  Panneau 3 (vitesse)     :  "-"  lw=0.5
  Panneau 4 (sin Omega)   :  "-"  lw=0.6
  Ref. mois sideral        :  axhline  27.322 j, color="gray" ls="--" lw=0.8
"""
    ax.text(0.03, 0.88, txt, transform=ax.transAxes,
            fontsize=8.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", alpha=0.9))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_couleurs_catalogue(pdf):
    """Page 5 : catalogue visuel des couleurs matplotlib."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax_title = fig.add_axes([0.0, 0.92, 1.0, 0.08])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "Catalogue des couleurs matplotlib",
                  fontsize=16, ha="center", fontweight="bold",
                  transform=ax_title.transAxes)

    named = [
        ("red", "red"), ("blue", "blue"), ("green", "green"),
        ("orange", "orange"), ("purple", "purple"), ("brown", "brown"),
        ("pink", "pink"), ("gray", "gray"), ("cyan", "cyan"),
        ("magenta", "magenta"), ("olive", "olive"), ("navy", "navy"),
        ("darkred", "darkred"), ("steelblue", "steelblue"),
        ("forestgreen", "forestgreen"), ("darkorange", "darkorange"),
        ("darkviolet", "darkviolet"), ("sienna", "sienna"),
        ("crimson", "crimson"), ("teal", "teal"),
        ("royalblue", "royalblue"), ("tomato", "tomato"),
        ("goldenrod", "goldenrod"), ("seagreen", "seagreen"),
    ]

    tab10 = [
        ("C0 (bleu)", "C0"), ("C1 (orange)", "C1"), ("C2 (vert)", "C2"),
        ("C3 (rouge)", "C3"), ("C4 (violet)", "C4"), ("C5 (brun)", "C5"),
        ("C6 (rose)", "C6"), ("C7 (gris)", "C7"), ("C8 (olive)", "C8"),
        ("C9 (cyan)", "C9"),
    ]

    hex_ex = [
        ("#1f77b4", "#1f77b4"), ("#ff7f0e", "#ff7f0e"),
        ("#2ca02c", "#2ca02c"), ("#d62728", "#d62728"),
        ("#9467bd", "#9467bd"), ("#8c564b", "#8c564b"),
        ("#e377c2", "#e377c2"), ("#17becf", "#17becf"),
    ]

    ax1 = fig.add_axes([0.03, 0.52, 0.45, 0.38])
    ax1.set_title("Couleurs nommees", fontsize=11, fontweight="bold")
    for i, (label, col) in enumerate(named):
        row = i // 4
        ccol = i % 4
        x = ccol * 0.25 + 0.02
        y = 1.0 - row * 0.16 - 0.08
        ax1.add_patch(plt.Rectangle((x, y), 0.06, 0.10,
                                    facecolor=col, edgecolor="black",
                                    linewidth=0.5,
                                    transform=ax1.transAxes))
        ax1.text(x + 0.075, y + 0.05, label, transform=ax1.transAxes,
                 fontsize=7, va="center", fontfamily="monospace")
    ax1.axis("off")

    ax2 = fig.add_axes([0.52, 0.52, 0.45, 0.38])
    ax2.set_title("Cycle automatique (C0-C9) et hex", fontsize=11,
                  fontweight="bold")
    for i, (label, col) in enumerate(tab10):
        y = 1.0 - i * 0.09 - 0.05
        ax2.add_patch(plt.Rectangle((0.02, y), 0.06, 0.065,
                                    facecolor=col, edgecolor="black",
                                    linewidth=0.5,
                                    transform=ax2.transAxes))
        ax2.text(0.10, y + 0.032, label, transform=ax2.transAxes,
                 fontsize=7.5, va="center", fontfamily="monospace")

    for i, (label, col) in enumerate(hex_ex):
        y = 1.0 - i * 0.09 - 0.05
        ax2.add_patch(plt.Rectangle((0.52, y), 0.06, 0.065,
                                    facecolor=col, edgecolor="black",
                                    linewidth=0.5,
                                    transform=ax2.transAxes))
        ax2.text(0.60, y + 0.032, label, transform=ax2.transAxes,
                 fontsize=7.5, va="center", fontfamily="monospace")
    ax2.axis("off")

    ax3 = fig.add_axes([0.03, 0.05, 0.94, 0.44])
    ax3.axis("off")
    ax3.set_title("Comment specifier une couleur dans matplotlib",
                  fontsize=11, fontweight="bold")

    txt = """
TROIS FACONS DE SPECIFIER UNE COULEUR
======================================

1. PAR NOM (chaine de caracteres) :
   color="red"           color="steelblue"        color="darkviolet"
   color="forestgreen"   color="crimson"          color="navy"

2. PAR CODE HEXADECIMAL (#RRGGBB) :
   color="#d62728"        (rouge matplotlib)
   color="#1f77b4"        (bleu matplotlib)
   color="#2ca02c"        (vert matplotlib)
   color="#ff6600"        (orange personnalise)

3. PAR CYCLE AUTOMATIQUE (C0 a C9) :
   color="C0"  -->  bleu par defaut        color="C3"  -->  rouge
   color="C1"  -->  orange par defaut      color="C4"  -->  violet

4. PAR TUPLE RGB ou RGBA (0.0 a 1.0) :
   color=(0.8, 0.2, 0.1)          rouge sombre
   color=(0.2, 0.5, 0.8, 0.7)     bleu semi-transparent (alpha=0.7)

OU MODIFIER DANS parametres_orbitaux.py :
  Ligne 219 :  datasets.append((jup, "red", 0.8, d))
                                      ^^^^^  ^^^
                                    couleur  alpha
  --> Changer "red" par n'importe quelle specification ci-dessus.
  --> Changer 0.8 par une valeur entre 0.0 et 1.0 (transparence).

  Mode --both, lignes 226-227 :
    datasets.append(("avec Jupiter", "red",  0.7, ...))
    datasets.append(("sans Jupiter", "blue", 0.5, ...))
    --> Personnaliser independamment chaque jeu.
"""
    ax3.text(0.02, 0.97, txt, transform=ax3.transAxes,
             fontsize=8.5, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_epaisseur_trait(pdf):
    """Page 6 : demonstration visuelle des epaisseurs et styles de trait."""
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("Epaisseur et style de trait — Demonstration visuelle",
                 fontsize=16, fontweight="bold")

    t = np.linspace(0, 4 * np.pi, 300)

    # (0,0) Epaisseurs de trait (lw)
    ax = axes[0, 0]
    widths = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
    for i, w in enumerate(widths):
        y = np.sin(t) + i * 2.5
        ax.plot(t, y, "-", color="steelblue", lw=w)
        ax.text(-0.5, y[0], f"lw={w}", fontsize=8, va="center",
                fontfamily="monospace")
    ax.set_title("Epaisseur de trait (linewidth = lw)", fontsize=11)
    ax.set_xlim(-1.5, 4 * np.pi)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    txt_lw = ("Valeurs utilisees dans le programme :\n"
              "  Terre panneau 0 : lw=1.0   (annee siderale)\n"
              "  Terre panneau 1 : lw=0.6   (distance ST)\n"
              "  Lune panneaux   : lw=0.5   (excentricite, etc.)\n"
              "  Lune panneau 4  : lw=0.6   (sin Omega)\n"
              "  References      : lw=0.8   (lignes tiretees)")
    ax.text(0.35, 0.02, txt_lw, transform=ax.transAxes, fontsize=7.5,
            va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # (0,1) Styles de trait (linestyle)
    ax = axes[0, 1]
    styles = [
        ("-",   "'-'   solide (defaut)"),
        ("--",  "'--'  tirete"),
        ("-.",  "'-.'  tirete-point"),
        (":",   "':'   pointille"),
    ]
    for i, (ls, label) in enumerate(styles):
        y = np.sin(t) + i * 2.5
        ax.plot(t, y, ls=ls, color="darkred", lw=1.5)
        ax.text(-0.5, y[0], label, fontsize=8, va="center",
                fontfamily="monospace")
    ax.set_title("Style de trait (linestyle = ls)", fontsize=11)
    ax.set_xlim(-3.5, 4 * np.pi)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    # (1,0) Marqueurs
    ax = axes[1, 0]
    markers = [
        ("o",  "cercle"),
        ("s",  "carre"),
        ("^",  "triangle"),
        ("D",  "diamant"),
        ("*",  "etoile"),
        ("x",  "croix"),
        ("+",  "plus"),
        (".",  "point"),
    ]
    t_sparse = np.linspace(0, 4 * np.pi, 20)
    for i, (mk, label) in enumerate(markers):
        y = np.sin(t_sparse) + i * 2.5
        ax.plot(t_sparse, y, ls="-", marker=mk, color="forestgreen",
                lw=0.8, markersize=5)
        ax.text(-0.5, y[0], f"'{mk}'  {label}", fontsize=8,
                va="center", fontfamily="monospace")
    ax.set_title("Marqueurs (marker)", fontsize=11)
    ax.set_xlim(-2.0, 4 * np.pi)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    txt_mk = ("Dans le programme :\n"
              "  Terre panneaux 2-3 : marker='o'  markersize=2.5\n"
              "  Pour ajouter des marqueurs : changer '-' en 'o-'\n"
              "  Pour enlever les marqueurs : changer 'o-' en '-'")
    ax.text(0.35, 0.02, txt_mk, transform=ax.transAxes, fontsize=7.5,
            va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # (1,1) Transparence (alpha)
    ax = axes[1, 1]
    alphas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    for i, a in enumerate(alphas):
        y = np.sin(t) + i * 2.5
        ax.plot(t, y, "-", color="red", lw=2.0, alpha=a)
        ax.text(-0.5, y[0], f"alpha={a}", fontsize=8, va="center",
                fontfamily="monospace")
    ax.set_title("Transparence (alpha)", fontsize=11)
    ax.set_xlim(-1.5, 4 * np.pi)
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    txt_al = ("Valeurs utilisees :\n"
              "  Mode simple  :  alpha=0.8  (un seul jeu)\n"
              "  Mode --both  :  alpha=0.7 (Jupiter)\n"
              "                   alpha=0.5 (sans Jupiter)\n"
              "  Grille       :  alpha=0.3")
    ax.text(0.35, 0.02, txt_al, transform=ax.transAxes, fontsize=7.5,
            va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_modifications(pdf):
    """Page 7 : guide pratique des modifications avec numeros de lignes."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.96, "Guide de modification — Couleurs et epaisseurs",
            transform=ax.transAxes, fontsize=16, ha="center",
            fontweight="bold")

    txt = """
MODIFICATIONS RAPIDES (sans changer la structure du code)
==========================================================

1. CHANGER LA COULEUR D'UN JEU DE DONNEES
   ----------------------------------------
   Ligne 219 (mode par defaut, avec Jupiter) :
     datasets.append((jup, "red", 0.8, d))
                            ^^^^
     Remplacer "red" par :  "steelblue"  "darkgreen"  "#ff6600"  etc.

   Ligne 230 (mode --sans-jupiter) :
     datasets.append(("sans Jupiter", "blue", 0.8, load_data(path)))
                                       ^^^^^

   Lignes 226-227 (mode --both) :
     datasets.append(("avec Jupiter", "red",  0.7, load_data(path_a)))
     datasets.append(("sans Jupiter", "blue", 0.5, load_data(path_s)))
                                       ^^^^^  ^^^
                                      couleur alpha


2. CHANGER L'EPAISSEUR DES TRAITS
   --------------------------------
   Figure Terre — plot_terre() :
     Ligne  82 :  lw=1.0   --> annee siderale
     Ligne  93 :  lw=0.6   --> distance Soleil-Terre
     Ligne 101 :  lw=1.0   --> excentricite (+ markersize=2.5)
     Ligne 108 :  lw=1.0   --> rayon perihelie (+ markersize=2.5)

   Figure Lune — plot_lune() :
     Ligne 154 :  lw=0.5   --> excentricite lunaire
     Ligne 163 :  lw=0.5   --> inclinaison
     Ligne 172 :  lw=0.5   --> periode siderale
     Ligne 181 :  lw=0.5   --> vitesse perigee
     Ligne 187 :  lw=0.6   --> sin(Omega)


3. AJOUTER / RETIRER DES MARQUEURS
   ---------------------------------
   Changer le format string dans ax.plot() :
     "-"     trait continu seul
     "o-"    trait + cercles        (panneaux 2-3 de Terre)
     ".-"    trait + petits points
     "s-"    trait + carres
   Le parametre markersize controle la taille :
     markersize=2.5  (actuel)  -->  markersize=4  (plus gros)


4. CHANGER LA GRILLE
   -------------------
   Ligne 116 (Terre) et 194 (Lune) :
     ax.grid(True, alpha=0.3)
     --> alpha=0.5 pour grille plus visible
     --> ax.grid(False)  pour supprimer la grille


5. CHANGER LES LIGNES DE REFERENCE
   ----------------------------------
   Ligne 121 (ref annee siderale) :
     axes[0].axhline(365.256, color="gray", ls="--", lw=0.8)
     --> color="red", ls=":"  pour pointille rouge

   Ligne 176 (ref mois sideral) :
     axes[2].axhline(27.322, color="gray", ls="--", lw=0.8)


6. CHANGER LA TAILLE DE LA FIGURE
   --------------------------------
   Ligne  66 (Terre) :  figsize=(15, 16)   largeur x hauteur en pouces
   Ligne 141 (Lune)  :  figsize=(15, 20)
"""
    ax.text(0.02, 0.90, txt, transform=ax.transAxes,
            fontsize=7.8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_exemples_visuels(pdf):
    """Page 8 : exemples visuels de combinaisons couleur/epaisseur."""
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("Exemples de personnalisation visuelle",
                 fontsize=16, fontweight="bold")

    t = np.linspace(0, 6 * np.pi, 500)
    sig1 = np.sin(t) * 0.03 + 0.055
    sig2 = np.sin(t * 1.1 + 0.5) * 0.03 + 0.056

    # (0,0) Style par defaut
    ax = axes[0, 0]
    ax.plot(t / (2 * np.pi), sig1, "-", color="red", lw=0.5, alpha=0.8,
            label="avec Jupiter")
    ax.set_title("Style actuel (defaut)", fontsize=11, fontweight="bold")
    ax.set_ylabel("$e_L$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, 'color="red", lw=0.5, alpha=0.8',
            transform=ax.transAxes, fontsize=7, fontfamily="monospace",
            bbox=dict(fc="white", alpha=0.8))

    # (0,1) Trait plus epais, couleur differente
    ax = axes[0, 1]
    ax.plot(t / (2 * np.pi), sig1, "-", color="darkviolet", lw=1.2,
            alpha=0.9, label="avec Jupiter")
    ax.set_title("Trait epais + violet", fontsize=11, fontweight="bold")
    ax.set_ylabel("$e_L$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, 'color="darkviolet", lw=1.2, alpha=0.9',
            transform=ax.transAxes, fontsize=7, fontfamily="monospace",
            bbox=dict(fc="white", alpha=0.8))

    # (1,0) Superposition avec/sans Jupiter
    ax = axes[1, 0]
    ax.plot(t / (2 * np.pi), sig1, "-", color="red", lw=0.5, alpha=0.7,
            label="avec Jupiter")
    ax.plot(t / (2 * np.pi), sig2, "-", color="blue", lw=0.5, alpha=0.5,
            label="sans Jupiter")
    ax.set_title("Superposition (--both) actuelle", fontsize=11,
                 fontweight="bold")
    ax.set_ylabel("$e_L$")
    ax.set_xlabel("annees")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02,
            '"red" alpha=0.7  +  "blue" alpha=0.5',
            transform=ax.transAxes, fontsize=7, fontfamily="monospace",
            bbox=dict(fc="white", alpha=0.8))

    # (1,1) Superposition alternative
    ax = axes[1, 1]
    ax.plot(t / (2 * np.pi), sig1, "-", color="#d62728", lw=1.0,
            alpha=0.9, label="avec Jupiter")
    ax.plot(t / (2 * np.pi), sig2, "--", color="forestgreen", lw=1.0,
            alpha=0.8, label="sans Jupiter")
    ax.set_title("Alternative : tirete vert + rouge epais", fontsize=11,
                 fontweight="bold")
    ax.set_ylabel("$e_L$")
    ax.set_xlabel("annees")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.text(0.02, 0.02,
            '"#d62728" lw=1.0  +  "forestgreen" "--" lw=1.0',
            transform=ax.transAxes, fontsize=7, fontfamily="monospace",
            bbox=dict(fc="white", alpha=0.8))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def page_recapitulatif(pdf):
    """Page 9 : recapitulatif des parametres modifiables."""
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.96, "Recapitulatif — Parametres modifiables",
            transform=ax.transAxes, fontsize=16, ha="center",
            fontweight="bold")

    header = ["Parametre", "Mot-cle", "Valeurs", "Exemple"]
    rows = [
        ["Couleur",      "color",      "nom, hex, C0-C9, (r,g,b)",
         'color="steelblue"'],
        ["Epaisseur",    "lw",         "float (0.1 a 5+)",
         "lw=1.5"],
        ["Transparence", "alpha",      "0.0 (invisible) a 1.0",
         "alpha=0.7"],
        ["Style trait",  "ls",         "'- -- -. :'",
         'ls="--"'],
        ["Marqueur",     "marker",     "'o s ^ D * x + .'",
         'marker="o"'],
        ["Taille marq.", "markersize", "float (1 a 15+)",
         "markersize=4"],
        ["Taille fig.",  "figsize",    "(largeur, hauteur) pouces",
         "figsize=(15, 16)"],
        ["Resolution",   "--dpi",      "entier (72 a 600)",
         "--dpi 300"],
        ["Grille",       "grid",       "True/False + alpha",
         "grid(True, alpha=0.5)"],
    ]

    colors_t = [["#e8f4fd"] * 4] * len(rows)
    table = ax.table(cellText=rows, colLabels=header,
                     cellColours=colors_t,
                     colColours=["#4a86c8"] * 4,
                     loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(color="white", fontweight="bold")
        if j == 3:
            cell.set_text_props(fontfamily="monospace", fontsize=8)

    ax.text(0.05, 0.30, "RESUME DES FICHIERS PRODUITS",
            transform=ax.transAxes, fontsize=13, fontweight="bold")

    files_txt = """
  --save active :
    resultats/params_terre_{tmin}_{tmax}ans.png    Figure Terre (4 panneaux)
    resultats/params_lune_{tmin}_{tmax}ans.png     Figure Lune (5 panneaux)

  Exemples de noms :
    params_terre_0_500ans.png          (plage par defaut)
    params_terre_0_50ans.png           (zoom 50 ans)
    params_terre_100_300ans.png        (plage personnalisee)
    params_lune_0_200ans.png

  Sans --save : affichage interactif (fenetre matplotlib zoomable,
                sauvegarde manuelle via l'icone disquette).
"""
    ax.text(0.05, 0.26, files_txt, transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", alpha=0.9))

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main():
    out_dir = os.path.join(HERE, "resultats")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "doc_parametres_orbitaux.pdf")

    print("Generation du PDF de documentation...")

    with PdfPages(out_path) as pdf:
        page_titre(pdf)
        page_options(pdf)
        page_fonctionnement(pdf)
        page_structure_datasets(pdf)
        page_couleurs_catalogue(pdf)
        page_epaisseur_trait(pdf)
        page_modifications(pdf)
        page_exemples_visuels(pdf)
        page_recapitulatif(pdf)

    print(f"PDF genere : {out_path}  (9 pages)")


if __name__ == "__main__":
    main()
