#!/usr/bin/env python3
"""
Prediction d'eclipses et cycle de Saros
========================================
Post-traitement des donnees de simulation_TLS_Jupiter.py

Usage :
    python3 eclipses_saros.py                                    # donnees par defaut
    python3 eclipses_saros.py resultats/simulation_data_avecJupiter.npz
    python3 eclipses_saros.py --years-max 500                    # limiter a 500 ans

Principe :
    1. Calcul de l'elongation geocentrique Lune-Soleil
    2. Detection des Nouvelles Lunes (elongation = 0) et Pleines Lunes (elongation = pi)
    3. Verification des conditions geometriques d'eclipse (latitude ecliptique)
    4. Classification : totale / annulaire / partielle (solaire)
                        totale / partielle / penombrale (lunaire)
    5. Identification du cycle de Saros (~18,03 ans = 6585,32 jours)

Fichiers produits dans resultats/ :
    fig9_eclipses_timeline_<tag>.png    chronologie des eclipses
    fig10_saros_<tag>.png               mise en evidence du cycle de Saros
    fig11_stats_eclipses_<tag>.png      statistiques et resume
    fig10b_saros_series_4000ans.png     series de Saros montantes/descendantes
    fig13_cycles_eclipses_<tag>.png     Tritos, Inex, Exeligmos
    fig12_divergence_<tag>.png          divergence vs ephemerides astropy
"""

import numpy as np
import os, sys, argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Le simulateur et ses resultats se trouvent dans le dossier voisin du chapitre 4
TLS_DIR = os.path.join(HERE, "..", "04_terre_lune_soleil")
import sys as _sys
_sys.path.insert(0, TLS_DIR)

# ===================================================================
# LISIBILITE DES FIGURES DANS LE LIVRE
# ===================================================================
# Les figures sont inserees en pleine largeur (\includegraphics[width=\textwidth])
# et donc reduites d'un facteur ~0,38 : sans ces tailles, les graduations
# tombent a 3 pt sur la page imprimee.
FS_SUPTITLE = 24
FS_TITLE    = 20
FS_LABEL    = 18
FS_TICK     = 16
FS_LEGEND   = 15
FS_TABLE    = 18     # encadres de resume (police a chasse fixe)


def encadre_resume(ax, info, titre, hauteur_pouces=4.8):
    """Encadre jaune de resume : la police s'ajuste pour que tout tienne."""
    n_lignes = info.count("\n") + 1
    fs = min(FS_TABLE, max(10, int(hauteur_pouces * 72 * 0.75 / (n_lignes * 1.05))))
    ax.set_facecolor("lightyellow")
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=fs,
            va="top", fontfamily="monospace", linespacing=1.05)
    ax.set_title(titre)
    ax.set_xticks([]); ax.set_yticks([])


def appliquer_polices():
    """Tailles de police communes a toutes les figures du chapitre."""
    plt.rcParams.update({
        "font.size":        FS_TICK,
        "axes.titlesize":   FS_TITLE,
        "axes.labelsize":   FS_LABEL,
        "xtick.labelsize":  FS_TICK,
        "ytick.labelsize":  FS_TICK,
        "legend.fontsize":  FS_LEGEND,
    })

# ===================================================================
# CONSTANTES PHYSIQUES (SI)
# ===================================================================
R_SUN   = 6.96e8       # rayon du Soleil (m)
R_EARTH = 6.371e6      # rayon de la Terre (m)
R_MOON  = 1.7374e6     # rayon de la Lune (m)
JOUR    = 86400.0
YEAR_S  = 365.25 * JOUR
SAROS_DAYS = 6585.3211  # 223 mois synodiques (jours)
SAROS_YEARS = SAROS_DAYS / 365.25  # ~18.03 ans

# Epoque de la simulation : 2 janvier 2005, 01h00 UTC
EPOCH_JD = 2453372.5 + 1.0 / 24.0


# ===================================================================
# CHARGEMENT DES DONNEES
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Prediction d'eclipses et cycle de Saros")
    p.add_argument("npz", nargs="?",
                   default=os.path.join(TLS_DIR, "resultats", "simulation_data_avecJupiter.npz"),
                   help="Fichier .npz de la simulation")
    p.add_argument("--years-max", type=float, default=None,
                   help="Limiter l'analyse aux N premieres annees")
    return p.parse_args()


def load_data(path, years_max=None):
    d = np.load(path, allow_pickle=True)
    t = np.array(d["t_samp"])
    mask = np.ones(len(t), dtype=bool)
    if years_max is not None:
        mask = t < years_max * YEAR_S

    keys = ["t_samp", "rST_samp", "rTL_samp", "xT", "yT", "zT",
            "xTL", "yTL", "zTL"]
    data = {k: np.array(d[k])[mask] for k in keys}
    data["dt"] = float(d["dt"])
    data["store_every"] = int(d["store_every"])
    data["jupiter"] = bool(d["jupiter"])
    return data


# ===================================================================
# DETECTION DES SYZYGIES (NOUVELLES ET PLEINES LUNES)
# ===================================================================
def detect_syzygies(data):
    """
    Detecte les Nouvelles Lunes et Pleines Lunes par zero-crossing
    de sin(D) ou D = lambda_Lune - lambda_Soleil est l'elongation.

    Nouvelle Lune : sin(D) = 0 et cos(D) > 0  (D = 0 mod 2pi)
    Pleine Lune   : sin(D) = 0 et cos(D) < 0  (D = pi mod 2pi)
    """
    lambda_L = np.arctan2(data["yTL"], data["xTL"])
    lambda_S = np.arctan2(-data["yT"], -data["xT"])

    D = lambda_L - lambda_S
    sinD = np.sin(D)
    cosD = np.cos(D)

    t   = data["t_samp"]
    rST = data["rST_samp"]
    rTL = data["rTL_samp"]
    zTL = data["zTL"]

    new_moons  = []   # (t, rST, rTL, beta)
    full_moons = []

    for i in range(1, len(t)):
        if sinD[i-1] * sinD[i] >= 0:
            continue
        frac = -sinD[i-1] / (sinD[i] - sinD[i-1])
        t_ev  = t[i-1]   + frac * (t[i]   - t[i-1])
        r_ST  = rST[i-1] + frac * (rST[i] - rST[i-1])
        r_TL  = rTL[i-1] + frac * (rTL[i] - rTL[i-1])
        z_TL  = zTL[i-1] + frac * (zTL[i] - zTL[i-1])
        beta  = np.arcsin(np.clip(z_TL / r_TL, -1, 1))
        cosD_ev = cosD[i-1] + frac * (cosD[i] - cosD[i-1])

        entry = (t_ev, r_ST, r_TL, beta)
        if cosD_ev > 0:
            new_moons.append(entry)
        else:
            full_moons.append(entry)

    return new_moons, full_moons


# ===================================================================
# CLASSIFICATION DES ECLIPSES
# ===================================================================
def classify_solar(gamma, r_ST, r_TL):
    """
    Eclipse solaire : la Lune masque le Soleil.
      alpha_S = R_sun / r_ST   (demi-diametre apparent du Soleil)
      alpha_L = R_moon / r_TL  (demi-diametre apparent de la Lune)
      pi_L    = R_earth / r_TL (parallaxe horizontale de la Lune)
      gamma   = separation angulaire geocentrique Lune-Soleil

    La parallaxe lunaire (~0.95 deg) permet a un observateur
    de surface d'etre decale de pi_L par rapport au centre de
    la Terre. On utilise gamma_min = max(0, gamma - pi_L)
    pour classer l'eclipse du point de vue du meilleur observateur.

    Totale    : gamma_min < |alpha_S - alpha_L| et alpha_L >= alpha_S
    Annulaire : gamma_min < |alpha_S - alpha_L| et alpha_L <  alpha_S
    Partielle : gamma_min < alpha_S + alpha_L
    """
    alpha_S = R_SUN / r_ST
    alpha_L = R_MOON / r_TL
    pi_L    = R_EARTH / r_TL

    gamma_min = max(0.0, gamma - pi_L)

    if gamma_min >= alpha_S + alpha_L:
        return None
    if gamma_min < abs(alpha_S - alpha_L):
        return "totale" if alpha_L >= alpha_S else "annulaire"
    return "partielle"


def classify_lunar(gamma, r_ST, r_TL):
    """
    Eclipse lunaire : la Lune entre dans l'ombre de la Terre.
      alpha_U = R_earth/r_TL - R_sun/r_ST  (demi-angle de l'ombre)
      alpha_P = R_earth/r_TL + R_sun/r_ST  (demi-angle de la penombre)
      alpha_L = R_moon / r_TL

    Totale     : gamma < alpha_U - alpha_L  (Lune entierement dans l'ombre)
    Partielle  : gamma < alpha_U + alpha_L  (Lune partiellement dans l'ombre)
    Penombrale : gamma < alpha_P + alpha_L  (Lune dans la penombre)
    """
    alpha_U = R_EARTH / r_TL - R_SUN / r_ST
    alpha_P = R_EARTH / r_TL + R_SUN / r_ST
    alpha_L = R_MOON / r_TL

    if alpha_U < 0:
        alpha_U = 0
    if gamma >= alpha_P + alpha_L:
        return None
    if alpha_U > alpha_L and gamma < alpha_U - alpha_L:
        return "totale"
    if gamma < alpha_U + alpha_L:
        return "partielle"
    return "penombrale"


def detect_eclipses(new_moons, full_moons):
    """
    Pour chaque syzygie, verifie si les conditions geometriques
    d'une eclipse sont remplies. A la syzygie exacte,
    la separation angulaire gamma ≈ |beta_Lune|.
    """
    solar = []
    for t_ev, r_ST, r_TL, beta in new_moons:
        gamma = abs(beta)
        typ = classify_solar(gamma, r_ST, r_TL)
        if typ is not None:
            solar.append({"t": t_ev, "type": typ,
                          "gamma_deg": np.degrees(gamma),
                          "beta_deg": np.degrees(beta),
                          "rST": r_ST, "rTL": r_TL})

    lunar = []
    for t_ev, r_ST, r_TL, beta in full_moons:
        gamma = abs(beta)
        typ = classify_lunar(gamma, r_ST, r_TL)
        if typ is not None:
            lunar.append({"t": t_ev, "type": typ,
                          "gamma_deg": np.degrees(gamma),
                          "beta_deg": np.degrees(beta),
                          "rST": r_ST, "rTL": r_TL})

    return solar, lunar


# ===================================================================
# CYCLE DE SAROS
# ===================================================================
def detect_saros(eclipses, label=""):
    """
    Identifie le cycle de Saros : deux eclipses de meme nature
    separees de 223 mois synodiques = 6585.32 jours.

    Pour chaque eclipse, on cherche (par recherche dichotomique)
    l'eclipse la plus proche de t + Saros. Si l'ecart est < 15 j,
    on les relie dans la meme serie.
    """
    if not eclipses:
        return {}, np.array([])

    times_days = np.array([e["t"] / JOUR for e in eclipses])
    n = len(times_days)

    saros_next = [-1] * n
    saros_prev = [-1] * n
    intervals = []

    for i in range(n):
        t_target = times_days[i] + SAROS_DAYS
        j = np.searchsorted(times_days, t_target)
        best_j, best_diff = -1, 15.0
        for jj in [j - 1, j]:
            if 0 <= jj < n and jj > i:
                diff = abs(times_days[jj] - t_target)
                if diff < best_diff:
                    best_diff = diff
                    best_j = jj
        if best_j >= 0:
            saros_next[i] = best_j
            if saros_prev[best_j] == -1:
                saros_prev[best_j] = i
            intervals.append(times_days[best_j] - times_days[i])

    # Construction des series de Saros
    saros_series = {}
    series_id = 0
    for i in range(n):
        if saros_prev[i] == -1 and saros_next[i] != -1:
            chain = [i]
            j = saros_next[i]
            while j != -1:
                chain.append(j)
                j = saros_next[j]
            saros_series[series_id] = chain
            series_id += 1

    intervals = np.array(intervals) if intervals else np.array([])

    if len(intervals) > 0:
        print(f"\n  {label}")
        print(f"  Paires Saros trouvees   : {len(intervals)}")
        print(f"  Periode mesuree         : {np.mean(intervals):.2f}"
              f" +/- {np.std(intervals):.2f} jours")
        print(f"  Reference               : {SAROS_DAYS:.2f} jours"
              f"  ({SAROS_YEARS:.3f} ans)")
        print(f"  Series identifiees      : {len(saros_series)}")
        if saros_series:
            lens = [len(v) for v in saros_series.values()]
            print(f"  Eclipses / serie        : {min(lens)}-{max(lens)}"
                  f"  (moy {np.mean(lens):.1f})")

    return saros_series, intervals


# ===================================================================
# GRAPHIQUES
# ===================================================================
# ===================================================================
# CYCLES D'ECLIPSES CONNUS  (en mois synodiques)
# ===================================================================
MOIS_SYNODIQUE = 29.530589        # jours
CYCLES = [
    # nom,        mois syn, couleur,   style
    ("Semester",     6,  "gray",   "--"),
    ("Tritos",     135,  "orange", "--"),
    ("Saros",      223,  "red",    "--"),
    ("Inex",       358,  "blue",   "--"),
    ("Exeligmos",  669,  "green",  "--"),
    ("2xInex",     716,  "purple", "--"),
]


def cycle_annees(mois_syn):
    return mois_syn * MOIS_SYNODIQUE / 365.25


def _lisser(y, n=5):
    """Moyenne glissante pour l'affichage du spectre de recurrence uniquement :
    le comptage par fenetres glissantes tres proches (T_probe.step ~ dT) cree
    un peigne de dents de scie qui masque la forme du spectre. Le lissage ne
    change pas la position des pics (bien plus larges que la fenetre), il
    retire juste le bruit haute frequence entre echantillons voisins. Les
    comptages exacts par cycle (compte_paires_cycle) restent calcules sur le
    signal brut, non lisse.
    """
    if n <= 1 or len(y) < n:
        return y
    noyau = np.ones(n) / n
    ypad = np.pad(y, (n // 2, n // 2), mode="edge")
    return np.convolve(ypad, noyau, mode="valid")[:len(y)]


def spectre_recurrence(t_years, T_probe, dT):
    """Pour chaque delai T, nombre de couples d'eclipses separees de T +/- dT."""
    t = np.sort(t_years)
    counts = np.zeros(len(T_probe))
    for k, T in enumerate(T_probe):
        gauche = np.searchsorted(t, t + (T - dT), side="left")
        droite = np.searchsorted(t, t + (T + dT), side="right")
        counts[k] = np.sum(np.maximum(droite - gauche, 0))
    return counts


def compte_paires_cycle(t_years, T_cycle, dT):
    """Nombre de couples d'eclipses separees exactement de T_cycle +/- dT."""
    return int(spectre_recurrence(t_years, np.array([T_cycle]), dT)[0])


# ===================================================================
# Fig 10b : series de Saros sur toute la simulation
# ===================================================================
def plot_saros_series(solar, saros_sol, out_dir, n_exemples=3):
    """Figure 5.4 du livre : familles de series de Saros montantes et
    descendantes, exemples individuels, puis zoom avec le type d'eclipse.

    Une serie est *montante* si la latitude ecliptique beta croit d'une
    eclipse a la suivante (noeud ascendant), *descendante* sinon.
    """
    appliquer_polices()
    if not saros_sol:
        return

    series = []
    for sid, chain in saros_sol.items():
        ts = np.array([solar[i]["t"] / YEAR_S for i in chain])
        bs = np.array([solar[i]["beta_deg"] for i in chain])
        montante = bs[-1] > bs[0]
        series.append((sid, chain, ts, bs, montante))

    n_mont = sum(1 for x in series if x[4])
    n_desc = len(series) - n_mont

    fig, axes = plt.subplots(3, 1, figsize=(13, 13))
    fig.suptitle("Series de Saros solaires sur toute la simulation",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    # --- (0) toutes les series ---
    for sid, chain, ts, bs, mont in series:
        axes[0].plot(ts, bs, "-", color="red" if mont else "blue",
                     lw=0.8, alpha=0.45)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("annees"); axes[0].set_ylabel("latitude beta (deg)")
    axes[0].set_title(f"Toutes les series ({n_mont} montantes en rouge, "
                      f"{n_desc} descendantes en bleu)")
    axes[0].grid(True, alpha=0.3)

    # --- (1) quelques series completes ---
    longues_m = sorted([x for x in series if x[4]], key=lambda x: -len(x[1]))[:n_exemples]
    longues_d = sorted([x for x in series if not x[4]], key=lambda x: -len(x[1]))[:n_exemples]
    for k, (sid, chain, ts, bs, mont) in enumerate(longues_m + longues_d):
        coul = plt.cm.autumn(0.15 + 0.25*k) if mont else plt.cm.winter(0.15 + 0.25*(k - n_exemples))
        axes[1].plot(ts, bs, "o-", color="red" if mont else "blue",
                     alpha=0.45 + 0.18*(k % n_exemples), ms=5, lw=1.4,
                     label=f"Serie {sid} ({len(chain)} ecl., "
                           f"{'montante' if mont else 'descendante'})")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("annees"); axes[1].set_ylabel("latitude beta (deg)")
    axes[1].set_title(f"Exemples de series individuelles "
                      f"({n_exemples} montantes, {n_exemples} descendantes)")
    y0, y1 = axes[1].get_ylim()
    axes[1].set_ylim(y0, y1 + 0.45*(y1 - y0))
    axes[1].legend(fontsize=FS_LEGEND - 2, ncol=2, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # --- (2) zoom : une montante et une descendante, colorees par type ---
    cols = {"totale": "red", "annulaire": "orange", "partielle": "steelblue"}
    vus = set()
    for (sid, chain, ts, bs, mont), marque in ((longues_m[0], "o"), (longues_d[0], "s")):
        axes[2].plot(ts, bs, "-", color="gray", lw=1.0, alpha=0.6, zorder=2)
        for typ, coul in cols.items():
            sel = [k for k, i in enumerate(chain) if solar[i]["type"] == typ]
            if not sel:
                continue
            lbl = f"{typ} ({'montante' if mont else 'descendante'})"
            axes[2].scatter(ts[sel], bs[sel], c=coul, marker=marque, s=55,
                            zorder=3, label=lbl if lbl not in vus else None)
            vus.add(lbl)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xlabel("annees"); axes[2].set_ylabel("latitude beta (deg)")
    axes[2].set_title(f"Zoom : serie montante ({len(longues_m[0][1])} ecl., ronds) "
                      f"et descendante ({len(longues_d[0][1])} ecl., carres)")
    y0, y1 = axes[2].get_ylim()
    axes[2].set_ylim(y0, y1 + 0.35*(y1 - y0))
    axes[2].legend(fontsize=FS_LEGEND - 1, ncol=3, loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig10b_saros_series_4000ans.png"), dpi=150)
    plt.close(fig)
    print("  -> fig10b_saros_series_4000ans.png")


# ===================================================================
# Fig 13 : les autres cycles d'eclipses
# ===================================================================
def plot_cycles_eclipses(solar, out_dir, tag, jupiter):
    """Figure 5.5 du livre : spectre de recurrence etendu, ou l'on retrouve
    le Saros mais aussi le Tritos, l'Inex et l'Exeligmos."""
    appliquer_polices()
    if len(solar) < 100:
        return
    jup_str = "avec Jupiter" if jupiter else "sans Jupiter"
    t_years = np.array(sorted([e["t"] / YEAR_S for e in solar]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    fig.suptitle(f"Cycles d'eclipses au-dela du Saros  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    # --- (0,0) spectre large 0-80 ans ---
    dT1 = 0.2
    T1 = np.arange(0.2, 80.0, 0.1)
    c1 = spectre_recurrence(t_years, T1, dT1)
    c1_disp = _lisser(c1, 5)
    axes[0, 0].fill_between(T1, c1_disp, color="red", alpha=0.3)
    axes[0, 0].plot(T1, c1_disp, "r-", lw=1.0)
    y1 = c1_disp.max()
    for k, (nom, mois, coul, ls) in enumerate(CYCLES):
        T = cycle_annees(mois)
        if T > 80:
            continue
        axes[0, 0].axvline(T, color=coul, ls=ls, lw=1.8)
        axes[0, 0].annotate(nom, xy=(T, y1),
                            xytext=(T + 2.0, y1 * (1.03 + 0.055*(k % 3))),
                            color=coul, fontsize=FS_LEGEND, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=coul, lw=1.2))
    axes[0, 0].set_ylim(0, y1 * 1.32)
    axes[0, 0].set_xlabel("delai T entre deux eclipses solaires (annees)")
    axes[0, 0].set_ylabel(f"paires a T +/- {dT1} an")
    axes[0, 0].set_title("Spectre de recurrence (0-80 ans)")
    axes[0, 0].grid(True, alpha=0.3)

    # --- (0,1) zoom Tritos / Saros / Inex ---
    dT2 = 0.1
    T2 = np.arange(8.0, 32.0, 0.05)
    c2 = spectre_recurrence(t_years, T2, dT2)
    c2_disp = _lisser(c2, 5)
    axes[0, 1].fill_between(T2, c2_disp, color="red", alpha=0.3)
    axes[0, 1].plot(T2, c2_disp, "r-", lw=1.2)
    y2 = c2_disp.max()
    for nom, mois, coul, ls in CYCLES:
        T = cycle_annees(mois)
        if not (8.0 <= T <= 32.0):
            continue
        axes[0, 1].axvline(T, color=coul, ls=ls, lw=2.0)
        axes[0, 1].annotate(f"{nom}\n({T:.1f} ans)", xy=(T, y2),
                            xytext=(T - 3.0, y2 * 1.08), color=coul,
                            fontsize=FS_LEGEND, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=coul, lw=1.2))
    axes[0, 1].set_ylim(0, y2 * 1.25)
    axes[0, 1].set_xlabel("delai (annees)")
    axes[0, 1].set_ylabel(f"paires a T +/- {dT2} an")
    axes[0, 1].set_title("Zoom : Tritos, Saros, Inex")
    axes[0, 1].grid(True, alpha=0.3)

    # --- (1,0) nombre de paires par cycle ---
    principaux = [("Saros", 223, "red"), ("Inex", 358, "blue"),
                  ("Tritos", 135, "orange"), ("Exeligmos", 669, "green")]
    noms, valeurs, couleurs = [], [], []
    for nom, mois, coul in principaux:
        T = cycle_annees(mois)
        noms.append(f"{nom}\n{T:.2f} ans")
        valeurs.append(compte_paires_cycle(t_years, T, dT1))
        couleurs.append(coul)
    barres = axes[1, 0].bar(noms, valeurs, color=couleurs, alpha=0.75,
                            edgecolor="black")
    for b, v in zip(barres, valeurs):
        axes[1, 0].text(b.get_x() + b.get_width()/2, v * 1.01, f"{v}",
                        ha="center", va="bottom", fontsize=FS_LEGEND,
                        fontweight="bold")
    axes[1, 0].set_ylim(0, max(valeurs) * 1.12)
    axes[1, 0].tick_params(axis="x", labelsize=FS_TICK - 2)
    axes[1, 0].set_ylabel("paires d'eclipses reliees")
    axes[1, 0].set_title("Nombre de paires detectees par cycle")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # --- (1,1) resume ---
    info = "CYCLES D'ECLIPSES\n" + "=" * 38 + "\n\n"
    info += f"{'Cycle':<11}{'Mois syn':>9}{'Jours':>10}{'Annees':>9}\n"
    info += "-" * 39 + "\n"
    for nom, mois, _, _ in CYCLES[:-1]:
        info += (f"{nom:<11}{mois:>9d}{mois*MOIS_SYNODIQUE:>10.1f}"
                 f"{cycle_annees(mois):>9.2f}\n")
    info += "\nSaros = cycle principal\n"
    info += "  (meme noeud, meme phase,\n   meme distance TL)\n\n"
    info += "Inex = noeud OPPOSE\n"
    info += "  (la Lune revient a l'autre\n   noeud : eclipse similaire\n"
    info += "   mais hemisphere inverse)\n\n"
    info += "Tritos = Inex - Saros\n\n"
    info += "Exeligmos = 3 x Saros\n  (retour au meme lieu\n   sur Terre)\n"
    encadre_resume(axes[1, 1], info, "Resume des cycles", hauteur_pouces=4.8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, f"fig13_cycles_eclipses_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"  -> fig13_cycles_eclipses_{tag}.png")


def generate_plots(solar, lunar, saros_sol, saros_lun,
                   int_sol, int_lun, jupiter, out_dir, span_fig9=None):
    """span_fig9 : duree tracee sur la chronologie (fig. 9), None = tout.

    Le motif en losange est un effet de densite : il ne se lit qu'avec les
    ~215 cycles nodaux des 4000 ans. On trace donc tout par defaut, avec des
    marqueurs fins."""

    appliquer_polices()
    jup_str = "avec Jupiter" if jupiter else "sans Jupiter"
    tag = "avecJupiter" if jupiter else "sansJupiter"

    # ---------------------------------------------------------------
    # Fig 9 : Chronologie des eclipses
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    fig.suptitle(f"Eclipses detectees  ({jup_str})", fontsize=FS_SUPTITLE, fontweight="bold")

    colors_s = {"totale": "red", "annulaire": "orange", "partielle": "steelblue"}
    t_max9 = (span_fig9 * YEAR_S) if span_fig9 else np.inf
    lbl9 = f"{span_fig9:.0f} premieres annees" if span_fig9 else "4000 ans"
    for typ, col in colors_s.items():
        sel = [e for e in solar if e["type"] == typ and e["t"] <= t_max9]
        if sel:
            axes[0].scatter([e["t"] / YEAR_S for e in sel],
                            [e["beta_deg"] for e in sel],
                            c=col, s=5, label=f"{typ} ({len(sel)})", alpha=0.55)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("latitude beta (deg)")
    axes[0].set_title(f"Eclipses solaires sur {lbl9}  (total : {len(solar)})")
    axes[0].legend(fontsize=FS_LEGEND, markerscale=4)
    axes[0].grid(True, alpha=0.3)

    colors_l = {"totale": "darkred", "partielle": "purple", "penombrale": "gray"}
    for typ, col in colors_l.items():
        sel = [e for e in lunar if e["type"] == typ and e["t"] <= t_max9]
        if sel:
            axes[1].scatter([e["t"] / YEAR_S for e in sel],
                            [e["beta_deg"] for e in sel],
                            c=col, s=5, label=f"{typ} ({len(sel)})", alpha=0.55)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("annees depuis 2 jan 2005")
    axes[1].set_ylabel("latitude beta (deg)")
    axes[1].set_title(f"Eclipses lunaires sur {lbl9}  (total : {len(lunar)})")
    axes[1].legend(fontsize=FS_LEGEND, markerscale=4)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig9_eclipses_timeline_{tag}.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 10 : Cycle de Saros (4 panneaux convaincants)
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    fig.suptitle(f"Mise en evidence du cycle de Saros  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    # --- (0,0) Autocorrelation (bins larges = 0.5 an) ---
    # Avec des bins de 0.5 an, les oscillations rapides (mois)
    # sont moyennees et seuls les pics a 18, 36, 54 ans ressortent.
    all_ecl = solar + lunar
    if len(all_ecl) > 100:
        t_ecl_years = np.array(sorted([e["t"] / YEAR_S for e in all_ecl]))
        dt_bin = 0.5
        t_max = t_ecl_years[-1]
        n_bins = int(t_max / dt_bin) + 1
        signal = np.zeros(n_bins)
        for te in t_ecl_years:
            idx = int(te / dt_bin)
            if idx < n_bins:
                signal[idx] += 1
        signal -= np.mean(signal)
        corr = np.correlate(signal, signal, mode="full")
        corr = corr[len(corr)//2:]
        corr /= corr[0]
        lags = np.arange(len(corr)) * dt_bin
        n_show = int(80 / dt_bin)
        axes[0, 0].plot(lags[:n_show], corr[:n_show], "r-", lw=1.8)
        for k, (mult, ls, alpha) in enumerate(
                [(1, "--", 1.0), (2, ":", 0.7), (3, ":", 0.5)]):
            lbl = f"{mult} x Saros ({mult*SAROS_YEARS:.1f} ans)" if k < 2 else None
            axes[0, 0].axvline(mult * SAROS_YEARS, color="blue", ls=ls,
                               lw=1.5, alpha=alpha, label=lbl)
        axes[0, 0].set_xlabel("decalage (annees)")
        axes[0, 0].set_ylabel("autocorrelation")
        axes[0, 0].set_title("Autocorrelation (bins de 0.5 an)")
        axes[0, 0].legend(fontsize=FS_LEGEND)
        axes[0, 0].grid(True, alpha=0.3)

    # --- (0,1) Spectre de recurrence ---
    # Pour chaque delai T, on compte combien d'eclipses sont
    # suivies d'une autre eclipse a T ± dT pres.
    # C'est comme un histogramme d'intervalles, mais lisse.
    if len(solar) > 50:
        t_sol = np.array(sorted([e["t"] / YEAR_S for e in solar]))
        T_probe = np.arange(0.5, 55, 0.25)
        dT = 0.25
        counts = np.zeros(len(T_probe))
        for k, T in enumerate(T_probe):
            for i in range(len(t_sol)):
                target = t_sol[i] + T
                j = np.searchsorted(t_sol, target - dT)
                while j < len(t_sol) and t_sol[j] < target + dT:
                    if j > i:
                        counts[k] += 1
                    j += 1
        counts_disp = _lisser(counts, 5)
        axes[0, 1].fill_between(T_probe, counts_disp, color="red", alpha=0.4)
        axes[0, 1].plot(T_probe, counts_disp, "r-", lw=1.6)
        for k, (mult, ls, alpha) in enumerate(
                [(1, "--", 1.0), (2, ":", 0.7), (3, ":", 0.5)]):
            lbl = f"Saros ({mult*SAROS_YEARS:.1f} ans)" if k == 0 else None
            axes[0, 1].axvline(mult * SAROS_YEARS, color="blue", ls=ls,
                               lw=1.5, alpha=alpha, label=lbl)
        axes[0, 1].set_xlabel("delai T entre deux eclipses solaires (annees)")
        axes[0, 1].set_ylabel("paires a T +/- 0.25 an")
        axes[0, 1].set_title("Spectre de recurrence des eclipses")
        axes[0, 1].legend(fontsize=FS_LEGEND)
        axes[0, 1].grid(True, alpha=0.3)

    # --- (1,0) Exemple d'une serie Saros : evolution de beta ---
    best_series = None
    best_ecl = None
    for series, ecl, lbl in [(saros_sol, solar, "solaire"),
                              (saros_lun, lunar, "lunaire")]:
        if series:
            longest = max(series.items(), key=lambda kv: len(kv[1]))
            if best_series is None or len(longest[1]) > len(best_series[1]):
                best_series = longest
                best_ecl = ecl
                best_lbl = lbl

    if best_series is not None:
        sid, chain = best_series
        ts = [best_ecl[idx]["t"] / YEAR_S for idx in chain]
        betas = [best_ecl[idx]["beta_deg"] for idx in chain]
        types = [best_ecl[idx]["type"] for idx in chain]
        cols = {"totale": "red", "annulaire": "orange", "partielle": "steelblue",
                "penombrale": "gray"}
        for typ in set(types):
            mask = [t for t, ty in zip(range(len(chain)), types) if ty == typ]
            axes[1, 0].scatter([ts[m] for m in mask], [betas[m] for m in mask],
                               c=cols.get(typ, "black"), s=45, label=typ, zorder=3)
        axes[1, 0].plot(ts, betas, "k-", lw=1.4, alpha=0.6, zorder=2)
        axes[1, 0].axhline(0, color="k", lw=0.5)
        axes[1, 0].set_xlabel("annees")
        axes[1, 0].set_ylabel("latitude beta (deg)")
        axes[1, 0].set_title(f"Serie Saros {best_lbl} ({len(chain)} eclipses, "
                             f"duree {ts[-1]-ts[0]:.0f} ans)")
        axes[1, 0].legend(fontsize=FS_LEGEND)
        axes[1, 0].grid(True, alpha=0.3)

        # Annotations des intervalles
        if len(ts) >= 2:
            dt_series = np.diff(ts)
            axes[1, 0].text(0.02, 0.02,
                            f"Intervalle moyen : {np.mean(dt_series):.2f} ans"
                            f" ({np.mean(dt_series)*365.25:.1f} j)",
                            transform=axes[1, 0].transAxes, fontsize=FS_LEGEND,
                            bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # --- (1,1) Resume chiffre + mesure du Saros ---
    all_intv = np.concatenate([int_sol, int_lun]) if len(int_sol) > 0 and len(int_lun) > 0 else (
        int_sol if len(int_sol) > 0 else int_lun)
    info = "CYCLE DE SAROS\n" + "=" * 36 + "\n\n"
    info += "Principe : si une eclipse a lieu\n"
    info += "a la date t, une eclipse similaire\n"
    info += "a lieu a t + Saros.\n\n"
    info += f"Saros theorique :\n"
    info += f"  223 mois synodiques\n"
    info += f"  = {SAROS_DAYS:.2f} jours\n"
    info += f"  = {SAROS_YEARS:.3f} annees\n\n"
    if len(all_intv) > 0:
        info += f"Saros mesure (simulation) :\n"
        info += f"  {np.mean(all_intv):.2f} +/- {np.std(all_intv):.2f} j\n"
        info += f"  = {np.mean(all_intv)/365.25:.3f} annees\n\n"
        info += f"Ecart : {abs(np.mean(all_intv)-SAROS_DAYS):.2f} j\n"
        info += f"  (du a l'erreur de {0.002:.3f} j/mois\n"
        info += f"   sur le mois synodique)\n\n"
    info += f"Series Saros identifiees :\n"
    n_s = len(saros_sol) + len(saros_lun)
    info += f"  {len(saros_sol)} solaires + {len(saros_lun)} lunaires\n"
    if saros_sol:
        lens = [len(v) for v in saros_sol.values()]
        info += f"  Longueur sol. : {min(lens)}-{max(lens)} ecl.\n"

    encadre_resume(axes[1, 1], info, "Resume du cycle de Saros")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig10_saros_{tag}.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------
    # Fig 11 : Statistiques et resume
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    fig.suptitle(f"Statistiques des eclipses  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    if solar:
        axes[0, 0].hist([e["gamma_deg"] for e in solar], bins=50,
                        color="red", alpha=0.7, edgecolor="black", lw=0.3)
        axes[0, 0].set_xlabel("separation angulaire gamma (deg)")
        axes[0, 0].set_title("Distribution gamma - ecl. solaires")
        axes[0, 0].grid(True, alpha=0.3)

    if lunar:
        axes[0, 1].hist([e["gamma_deg"] for e in lunar], bins=50,
                        color="purple", alpha=0.7, edgecolor="black", lw=0.3)
        axes[0, 1].set_xlabel("separation angulaire gamma (deg)")
        axes[0, 1].set_title("Distribution gamma - ecl. lunaires")
        axes[0, 1].grid(True, alpha=0.3)

    # Frequence par siecle (fenetre glissante)
    all_t = sorted([e["t"] / YEAR_S for e in solar + lunar])
    if len(all_t) > 100:
        t_arr = np.array(all_t)
        win = 100
        tc = np.arange(win / 2, t_arr[-1] - win / 2, 10)
        counts = np.array([np.sum((t_arr > c - win / 2) & (t_arr < c + win / 2))
                           for c in tc])
        axes[1, 0].plot(tc, counts, "r-", lw=1.6)
        axes[1, 0].set_xlabel("annees")
        axes[1, 0].set_ylabel(f"eclipses / {win} ans")
        axes[1, 0].set_title("Frequence des eclipses (fenetre glissante)")
        axes[1, 0].grid(True, alpha=0.3)

    # Resume textuel
    dur = max(all_t) if all_t else 0
    n_sol, n_lun = len(solar), len(lunar)
    info = "RESUME\n" + "=" * 40 + "\n\n"
    info += f"Duree analysee : {dur:.0f} ans\n\n"

    info += f"Eclipses solaires : {n_sol}\n"
    for typ in ["totale", "annulaire", "partielle"]:
        c = sum(1 for e in solar if e["type"] == typ)
        info += f"  {typ:12s}: {c:5d} ({100*c/max(n_sol,1):.1f}%)\n"
    info += f"  par an     : {n_sol/max(dur,1):.2f}\n\n"

    info += f"Eclipses lunaires : {n_lun}\n"
    for typ in ["totale", "partielle", "penombrale"]:
        c = sum(1 for e in lunar if e["type"] == typ)
        info += f"  {typ:12s}: {c:5d} ({100*c/max(n_lun,1):.1f}%)\n"
    info += f"  par an     : {n_lun/max(dur,1):.2f}\n\n"

    if len(int_sol) > 0:
        info += f"Saros solaire : {np.mean(int_sol):.2f} j\n"
    if len(int_lun) > 0:
        info += f"Saros lunaire : {np.mean(int_lun):.2f} j\n"
    info += f"Reference     : {SAROS_DAYS:.2f} j\n"
    info += f"              = {SAROS_YEARS:.3f} ans\n\n"

    # Mois synodique verifie
    info += "Mois synodique ref : 29.5306 j\n"
    info += f"Total eclipses/an  : {(n_sol+n_lun)/max(dur,1):.2f}\n"
    info += f"Ref NASA (5 mill.) : 4.79 / an\n"

    encadre_resume(axes[1, 1], info, "Resume")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig11_stats_eclipses_{tag}.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures sauvegardees dans {out_dir}/")


# ===================================================================
# ETUDE DE DIVERGENCE : simulation vs ephemerides astropy
# ===================================================================
def study_divergence(solar_eclipses, out_dir, tag, max_years=500):
    """
    Compare les dates d'eclipses simulees avec les ephemerides astropy.
    A chaque eclipse simulee (temps t_sim), on calcule l'elongation
    D_astropy(t_sim). Si D != 0, la Nouvelle Lune reelle est decalee
    de delta_t ≈ D / (dD/dt) ou dD/dt ≈ 12.19 deg/jour.
    On etudie la croissance de |delta_t| avec le temps.
    """
    try:
        from simulation_TLS_Jupiter import _patch_astropy_numpy_compat
        _patch_astropy_numpy_compat()   # astropy 5.1 + numpy >= 1.24
        from astropy.time import Time
        from astropy.coordinates import get_body, GeocentricTrueEcliptic
        import astropy.units as u
    except ImportError:
        print("  Astropy non disponible, etude de divergence ignoree.")
        return

    ecl = [e for e in solar_eclipses if e["t"] / YEAR_S < max_years]
    if len(ecl) < 10:
        print("  Pas assez d'eclipses pour l'etude de divergence.")
        return

    n = len(ecl)
    print(f"  Calcul astropy pour {n} eclipses solaires"
          f" (0-{max_years} ans)...")

    t_days = np.array([e["t"] / JOUR for e in ecl])
    t_years = t_days / 365.25
    jd = EPOCH_JD + t_days
    times = Time(jd, format="jd", scale="utc")

    sun  = get_body("sun",  times).transform_to(GeocentricTrueEcliptic())
    moon = get_body("moon", times).transform_to(GeocentricTrueEcliptic())

    D_deg = (moon.lon - sun.lon).wrap_at(180 * u.deg).deg
    beta_astropy = moon.lat.deg

    dD_dt = 12.19
    delta_t_hours = np.abs(D_deg) / dD_dt * 24.0

    beta_sim = np.array([e["beta_deg"] for e in ecl])
    delta_beta = np.abs(beta_sim - beta_astropy)

    # Classification avec les valeurs astropy
    type_match = np.zeros(n, dtype=bool)
    for i, e in enumerate(ecl):
        gamma_astr = np.abs(np.radians(beta_astropy[i]))
        typ_astr = classify_solar(gamma_astr, e["rST"], e["rTL"])
        type_match[i] = (typ_astr == e["type"])

    # Ajustements : exponentiel et puissance + R²
    mask_fit = (t_years > 1) & (delta_t_hours > 0.01)
    t_f = t_years[mask_fit]
    dt_f = delta_t_hours[mask_fit]
    log_dt_f = np.log(dt_f)

    fit_exp, fit_pow = None, None
    r2_exp, r2_pow = 0, 0
    ss_tot = np.sum((log_dt_f - np.mean(log_dt_f))**2) if len(t_f) > 20 else 1

    if len(t_f) > 20:
        try:
            fit_exp = np.polyfit(t_f, log_dt_f, 1)
            resid_exp = log_dt_f - (fit_exp[1] + fit_exp[0] * t_f)
            r2_exp = 1 - np.sum(resid_exp**2) / ss_tot
        except Exception:
            pass
        try:
            fit_pow = np.polyfit(np.log(t_f), log_dt_f, 1)
            resid_pow = log_dt_f - (fit_pow[1] + fit_pow[0] * np.log(t_f))
            r2_pow = 1 - np.sum(resid_pow**2) / ss_tot
        except Exception:
            pass

    # Valeurs a des epoques cles (mediane par tranche)
    checkpoints = {}
    for yr_check in [1, 5, 10, 20, 50, 100, 200, 500]:
        m = (t_years > yr_check * 0.8) & (t_years < yr_check * 1.2)
        if np.sum(m) > 3:
            checkpoints[yr_check] = np.median(delta_t_hours[m])

    # --- Fig 12 : divergence ---
    appliquer_polices()
    jup_str = "avec Jupiter" if "avecJupiter" in tag else "sans Jupiter"
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    fig.suptitle(f"Divergence des predictions d'eclipses  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    # (0,0) Erreur temporelle — echelle LOG-LOG
    # En log-log, la loi de puissance est une droite,
    # l'exponentielle courbe vers le haut.
    axes[0, 0].scatter(t_years, delta_t_hours, s=9, c="red", alpha=0.35)
    t_plot = np.logspace(np.log10(1), np.log10(max_years), 300)
    if fit_pow is not None:
        axes[0, 0].plot(t_plot, np.exp(fit_pow[1]) * t_plot**fit_pow[0],
                        "g-", lw=2.5,
                        label=f"loi de puissance : "
                              f"|dt| ~ t^{{{fit_pow[0]:.2f}}}  "
                              f"(R² = {r2_pow:.4f})")
    if fit_exp is not None:
        axes[0, 0].plot(t_plot, np.exp(fit_exp[1] + fit_exp[0] * t_plot),
                        "b--", lw=1.5, alpha=0.7,
                        label=f"exponentielle : "
                              f"tau = {1/fit_exp[0]:.0f} ans  "
                              f"(R² = {r2_exp:.4f})")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("annees depuis 2 jan 2005")
    axes[0, 0].set_ylabel("|delta t| (heures)")
    axes[0, 0].set_title("Erreur de datation (log-log)")
    axes[0, 0].legend(fontsize=FS_LEGEND)
    axes[0, 0].grid(True, alpha=0.3, which="both")

    # (0,1) Erreur de latitude (log-log aussi)
    axes[0, 1].scatter(t_years, delta_beta, s=9, c="purple", alpha=0.35)
    axes[0, 1].set_xlabel("annees")
    axes[0, 1].set_ylabel("|delta beta| (deg)")
    axes[0, 1].set_title("Erreur de latitude ecliptique (log-log)")
    axes[0, 1].grid(True, alpha=0.3, which="both")
    if fit_pow is not None:
        mask_b = (t_years > 1) & (delta_beta > 0.001)
        if np.sum(mask_b) > 20:
            try:
                log_db = np.log(delta_beta[mask_b])
                bp = np.polyfit(np.log(t_years[mask_b]), log_db, 1)
                ss_tot_b = np.sum((log_db - np.mean(log_db))**2)
                resid_b = log_db - (bp[1] + bp[0] * np.log(t_years[mask_b]))
                r2_beta = 1 - np.sum(resid_b**2) / ss_tot_b
                axes[0, 1].plot(t_plot, np.exp(bp[1]) * t_plot**bp[0],
                                "g-", lw=2.5,
                                label=f"puissance : alpha = {bp[0]:.2f}"
                                      f"  (R² = {r2_beta:.4f})")
                axes[0, 1].set_xscale("log")
                axes[0, 1].set_yscale("log")
                axes[0, 1].legend(fontsize=FS_LEGEND)
            except Exception:
                pass

    # (1,0) Pourcentage de types divergents (cumulatif glissant)
    window_ecl = 50
    if n > window_ecl:
        pct = np.array([np.mean(~type_match[max(0,i-window_ecl):i+1]) * 100
                        for i in range(n)])
        axes[1, 0].plot(t_years, pct, "r-", lw=1.6)
        axes[1, 0].set_xlabel("annees")
        axes[1, 0].set_ylabel("eclipses mal classees (%)")
        axes[1, 0].set_title(f"Types divergents (fenetre de {window_ecl} ecl.)")
        axes[1, 0].grid(True, alpha=0.3)

    # (1,1) Resume
    info = "DIVERGENCE DES PREDICTIONS\n"
    info += "=" * 40 + "\n\n"
    info += f"Eclipses analysees : {n}\n"
    info += f"Duree : 0-{max_years} ans\n\n"
    info += "Erreur temporelle mediane :\n"
    for yr, val in sorted(checkpoints.items()):
        unit = "h" if val < 48 else "j"
        v = val if val < 48 else val / 24
        info += f"  a {yr:4d} ans : {v:8.1f} {unit}\n"
    info += "\n"
    if fit_pow is not None:
        info += f"Modele puissance (meilleur) :\n"
        info += f"  alpha = {fit_pow[0]:.2f},  R2 = {r2_pow:.4f}\n"
        info += f"  |dt| ~ t^{fit_pow[0]:.2f}  (lineaire)\n\n"
    if fit_exp is not None:
        info += f"Modele exponentiel (rejete) :\n"
        info += f"  tau = {1/fit_exp[0]:.0f} ans,  R2 = {r2_exp:.4f}\n\n"
    pct_bad_total = np.mean(~type_match) * 100
    info += f"Types divergents : {pct_bad_total:.1f}%\n"
    info += f"  (a 10 ans : ~{np.mean(~type_match[t_years<10])*100:.0f}%)\n"
    info += f"  (a 100 ans : ~{np.mean(~type_match[(t_years>80)&(t_years<120)])*100:.0f}%)\n"

    encadre_resume(axes[1, 1], info, "Resume")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig12_divergence_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"  -> fig12_divergence_{tag}.png")

    return {
        "checkpoints": checkpoints,
        "fit_exp_tau": 1/fit_exp[0] if fit_exp is not None else None,
        "fit_pow_alpha": fit_pow[0] if fit_pow is not None else None,
        "pct_bad_type": pct_bad_total,
    }


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.npz):
        print(f"Erreur : fichier {args.npz} introuvable.")
        print("Executez d'abord, dans ../04_terre_lune_soleil : python3 simulation_TLS_Jupiter.py")
        sys.exit(1)

    print(f"Chargement : {args.npz}")
    data = load_data(args.npz, args.years_max)

    dt_s = data["dt"] * data["store_every"]
    n_pts = len(data["t_samp"])
    dur_years = data["t_samp"][-1] / YEAR_S
    print(f"Echantillonnage : dt = {dt_s:.0f} s = {dt_s/JOUR:.2f} j")
    print(f"Points : {n_pts:,}   Duree : {dur_years:.0f} ans")
    print(f"Jupiter : {'OUI' if data['jupiter'] else 'NON'}")

    print("\n--- Detection des syzygies ---")
    new_moons, full_moons = detect_syzygies(data)
    print(f"Nouvelles Lunes : {len(new_moons):,}")
    print(f"Pleines Lunes   : {len(full_moons):,}")

    if len(new_moons) > 1:
        syn = np.diff([nm[0] for nm in new_moons]) / JOUR
        print(f"Mois synodique  : {np.mean(syn):.4f}"
              f" +/- {np.std(syn):.4f} j  (ref : 29.5306 j)")

    print("\n--- Classification des eclipses ---")
    solar, lunar = detect_eclipses(new_moons, full_moons)

    print(f"Eclipses solaires : {len(solar)}")
    for typ in ["totale", "annulaire", "partielle"]:
        c = sum(1 for e in solar if e["type"] == typ)
        if c: print(f"  {typ:12s}: {c}")

    print(f"Eclipses lunaires : {len(lunar)}")
    for typ in ["totale", "partielle", "penombrale"]:
        c = sum(1 for e in lunar if e["type"] == typ)
        if c: print(f"  {typ:12s}: {c}")

    # Premieres eclipses (pour verification)
    print("\n--- Premieres eclipses solaires ---")
    for e in solar[:8]:
        jours = e["t"] / JOUR
        ans = e["t"] / YEAR_S
        print(f"  t = {jours:8.1f} j  ({ans:6.2f} ans)  "
              f"type = {e['type']:10s}  beta = {e['beta_deg']:+6.3f} deg")

    print("\n--- Premieres eclipses lunaires ---")
    for e in lunar[:8]:
        jours = e["t"] / JOUR
        ans = e["t"] / YEAR_S
        print(f"  t = {jours:8.1f} j  ({ans:6.2f} ans)  "
              f"type = {e['type']:10s}  beta = {e['beta_deg']:+6.3f} deg")

    print("\n--- Recherche du cycle de Saros ---")
    saros_sol, int_sol = detect_saros(solar, "Eclipses solaires")
    saros_lun, int_lun = detect_saros(lunar, "Eclipses lunaires")

    out_dir = os.path.join(HERE, "resultats")
    os.makedirs(out_dir, exist_ok=True)

    print("\n--- Generation des graphiques ---")
    tag = "avecJupiter" if data["jupiter"] else "sansJupiter"
    generate_plots(solar, lunar, saros_sol, saros_lun,
                   int_sol, int_lun, data["jupiter"], out_dir)
    plot_saros_series(solar, saros_sol, out_dir)
    plot_cycles_eclipses(solar, out_dir, tag, data["jupiter"])

    print("\n--- Etude de divergence (simulation vs astropy) ---")
    study_divergence(solar, out_dir, tag, max_years=500)

    print("\nTermine.")
