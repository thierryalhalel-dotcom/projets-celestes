#!/usr/bin/env python3
"""
Eclipses du premier Saros — Simulation Chapront vs JPL/NASA
=============================================================

Detecte les eclipses (solaires et lunaires) sur le premier cycle de Saros
(~18.03 ans, 2 jan 2005 → ~20 jan 2023) a partir de la simulation
TLS-Jupiter (conditions initiales Chapront), puis compare avec le
catalogue NASA/JPL.

Produit un PDF multi-pages dans resultats/ avec :
  - Explication de la methode et des parametres
  - Chronologie comparee (simulation vs JPL)
  - Ecart de dates et concordance des types
  - Comparaison des durees estimees vs reelles
  - Resume statistique

Usage :
    python3 comparaison_eclipses_premier_saros.py
    python3 comparaison_eclipses_premier_saros.py --npz resultats/simulation_data_sansJupiter.npz
"""

import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
# Le simulateur et ses resultats se trouvent dans le dossier voisin du chapitre 4
TLS_DIR = os.path.join(HERE, "..", "04_terre_lune_soleil")
import sys as _sys
_sys.path.insert(0, TLS_DIR)

# ============================================================
# CONSTANTES PHYSIQUES (SI)
# ============================================================
JOUR = 86400.0
YEAR_S = 365.25 * JOUR
R_SUN = 6.96e8
R_EARTH = 6.371e6
R_MOON = 1.7374e6
SAROS_DAYS = 6585.32
T_SYNODIC_S = 29.53059 * JOUR
OMEGA_SYN = 2 * np.pi / T_SYNODIC_S
OMEGA_EARTH = 2 * np.pi / JOUR

EPOCH = datetime(2005, 1, 2, 1, 0, 0)
SAROS_END = EPOCH + timedelta(days=SAROS_DAYS)

# ============================================================
# CATALOGUE JPL/NASA — eclipses solaires (2005-2023)
# (date, type, duree_phase_centrale_s)
# duree = max duration de totalite ou annularite ; 0 si partielle
# Source : eclipse.gsfc.nasa.gov
# ============================================================
ECLIPSES_JPL_SOLAR = [
    ("2005-04-08", "annulaire/totale",  42),
    ("2005-10-03", "annulaire",        272),
    ("2006-03-29", "totale",           247),
    ("2006-09-22", "annulaire",        429),
    ("2007-03-19", "partielle",          0),
    ("2007-09-11", "partielle",          0),
    ("2008-02-07", "annulaire",        132),
    ("2008-08-01", "totale",           147),
    ("2009-01-26", "annulaire",        474),
    ("2009-07-22", "totale",           399),
    ("2010-01-15", "annulaire",        668),
    ("2010-07-11", "totale",           320),
    ("2011-01-04", "partielle",          0),
    ("2011-06-01", "partielle",          0),
    ("2011-07-01", "partielle",          0),
    ("2011-11-25", "partielle",          0),
    ("2012-05-20", "annulaire",        346),
    ("2012-11-13", "totale",           242),
    ("2013-05-10", "annulaire",        363),
    ("2013-11-03", "annulaire/totale", 100),
    ("2014-04-29", "annulaire",          0),
    ("2014-10-23", "partielle",          0),
    ("2015-03-20", "totale",           167),
    ("2015-09-13", "partielle",          0),
    ("2016-03-09", "totale",           249),
    ("2016-09-01", "annulaire",        186),
    ("2017-02-26", "annulaire",         44),
    ("2017-08-21", "totale",           160),
    ("2018-02-15", "partielle",          0),
    ("2018-07-13", "partielle",          0),
    ("2018-08-11", "partielle",          0),
    ("2019-01-06", "partielle",          0),
    ("2019-07-02", "totale",           273),
    ("2019-12-26", "annulaire",        220),
    ("2020-06-21", "annulaire",         38),
    ("2020-12-14", "totale",           130),
    ("2021-06-10", "annulaire",        231),
    ("2021-12-04", "totale",           114),
    ("2022-04-30", "partielle",          0),
    ("2022-10-25", "partielle",          0),
    ("2023-04-20", "annulaire/totale",  76),
    ("2023-10-14", "annulaire",        317),
]

# ============================================================
# CATALOGUE JPL/NASA — eclipses lunaires (2005-2023)
# (date, type, duree_totalite_s)
# duree = totalite (U2-U3) pour eclipses totales ; 0 sinon
# ============================================================
ECLIPSES_JPL_LUNAR = [
    ("2005-04-24", "penombrale",    0),
    ("2005-10-17", "partielle",     0),
    ("2006-03-14", "penombrale",    0),
    ("2006-09-07", "partielle",     0),
    ("2007-03-03", "totale",     4440),
    ("2007-08-28", "totale",     5400),
    ("2008-02-21", "totale",     3060),
    ("2008-08-16", "partielle",     0),
    ("2009-02-09", "penombrale",    0),
    ("2009-07-07", "penombrale",    0),
    ("2009-08-06", "penombrale",    0),
    ("2009-12-31", "partielle",     0),
    ("2010-06-26", "partielle",     0),
    ("2010-12-21", "totale",     4320),
    ("2011-06-15", "totale",     6000),
    ("2011-12-10", "totale",     3060),
    ("2012-06-04", "partielle",     0),
    ("2012-11-28", "penombrale",    0),
    ("2013-04-25", "partielle",     0),
    ("2013-05-25", "penombrale",    0),
    ("2013-10-18", "penombrale",    0),
    ("2014-04-15", "totale",     4680),
    ("2014-10-08", "totale",     3480),
    ("2015-04-04", "totale",      720),
    ("2015-09-28", "totale",     4320),
    ("2016-03-23", "penombrale",    0),
    ("2016-09-16", "penombrale",    0),
    ("2017-02-11", "penombrale",    0),
    ("2017-08-07", "partielle",     0),
    ("2018-01-31", "totale",     4620),
    ("2018-07-27", "totale",     6180),
    ("2019-01-21", "totale",     3720),
    ("2019-07-16", "partielle",     0),
    ("2020-01-10", "penombrale",    0),
    ("2020-06-05", "penombrale",    0),
    ("2020-07-05", "penombrale",    0),
    ("2020-11-30", "penombrale",    0),
    ("2021-05-26", "totale",      900),
    ("2021-11-19", "partielle",     0),
    ("2022-05-16", "totale",     5100),
    ("2022-11-08", "totale",     5100),
    ("2023-05-05", "penombrale",    0),
    ("2023-10-28", "partielle",     0),
]


# ============================================================
# ESTIMATION DES DUREES
# ============================================================
def estimate_solar_central_duration(r_ST, r_TL, gamma_rad):
    """
    Estime la duree maximale de la phase centrale (totalite ou
    annularite) d'une eclipse solaire, en secondes.

    Formule : dt = 2 * sqrt(delta_alpha^2 - gamma_eff^2) / omega_syn
    avec correction pour la rotation terrestre (facteur ~1.5).
    gamma_eff = max(0, |beta| - pi_L) : la parallaxe lunaire permet
    a un observateur de surface d'etre plus proche de l'axe d'ombre.
    """
    alpha_S = R_SUN / r_ST
    alpha_L = R_MOON / r_TL
    pi_L = R_EARTH / r_TL
    gamma_eff = max(0.0, gamma_rad - pi_L)
    delta_alpha = abs(alpha_L - alpha_S)
    if gamma_eff >= delta_alpha:
        return 0.0
    half_chord = np.sqrt(delta_alpha**2 - gamma_eff**2)
    dt_geo = 2 * half_chord / OMEGA_SYN
    v_shadow = OMEGA_SYN * r_TL
    v_earth_avg = OMEGA_EARTH * R_EARTH * 0.7
    factor = v_shadow / (v_shadow - v_earth_avg)
    return dt_geo * factor


def estimate_lunar_totality_duration(r_ST, r_TL, gamma_rad):
    """
    Estime la duree de la totalite d'une eclipse lunaire totale,
    en secondes.

    Formule : dt = 2 * sqrt((alpha_U - alpha_L)^2 - gamma^2) / omega_syn
    Pas de correction pour la rotation terrestre (tous les observateurs
    voient la meme eclipse lunaire).
    """
    alpha_U = R_EARTH / r_TL - R_SUN / r_ST
    alpha_L = R_MOON / r_TL
    gamma = abs(gamma_rad)
    if alpha_U <= 0 or gamma >= alpha_U - alpha_L:
        return 0.0
    half_chord = np.sqrt((alpha_U - alpha_L)**2 - gamma**2)
    return 2 * half_chord / OMEGA_SYN


# ============================================================
# DETECTION DES ECLIPSES DANS LA SIMULATION
# ============================================================
def detect_eclipses(d, tmax_days):
    """Detecte et classifie toutes les eclipses sur [0, tmax_days]."""
    t = d['t_samp']
    xT, yT, zT = d['xT'], d['yT'], d['zT']
    xTL, yTL, zTL = d['xTL'], d['yTL'], d['zTL']
    rST, rTL = d['rST_samp'], d['rTL_samp']

    lam_L = np.arctan2(yTL, xTL)
    lam_S = np.arctan2(-yT, -xT)
    beta_L = np.arcsin(np.clip(zTL / rTL, -1, 1))
    D = lam_L - lam_S
    sinD = np.sin(D)
    # D est reconstruit a partir de deux arctan2 : il saute de 2*pi des que
    # l'une des longitudes franchit +/-pi. Interpoler l'angle brut D a travers
    # un tel saut donne un cos(D) de signe faux, et la syzygie est alors
    # etiquetee Pleine Lune au lieu de Nouvelle Lune (ou l'inverse). On
    # interpole donc cos(D), qui est continu. Meme convention que
    # detect_syzygies() dans eclipses_saros.py.
    cosD = np.cos(D)

    tmax_s = tmax_days * JOUR
    solar, lunar = [], []

    for i in range(1, len(sinD)):
        if t[i] > tmax_s:
            break
        if sinD[i - 1] * sinD[i] > 0 or sinD[i - 1] == sinD[i]:
            continue

        frac = -sinD[i - 1] / (sinD[i] - sinD[i - 1])
        t_ev = t[i - 1] + frac * (t[i] - t[i - 1])
        beta_ev = beta_L[i - 1] + frac * (beta_L[i] - beta_L[i - 1])
        rTL_ev = rTL[i - 1] + frac * (rTL[i] - rTL[i - 1])
        rST_ev = rST[i - 1] + frac * (rST[i] - rST[i - 1])
        cosD_ev = cosD[i - 1] + frac * (cosD[i] - cosD[i - 1])
        is_NL = cosD_ev > 0

        gamma = abs(beta_ev)
        alpha_S = R_SUN / rST_ev
        alpha_L = R_MOON / rTL_ev
        pi_L = R_EARTH / rTL_ev

        date = EPOCH + timedelta(seconds=t_ev)

        if is_NL:
            gamma_min = max(0, gamma - pi_L)
            if gamma_min >= alpha_S + alpha_L:
                continue
            if gamma_min < abs(alpha_S - alpha_L):
                typ = "totale" if alpha_L >= alpha_S else "annulaire"
            else:
                typ = "partielle"
            dur = estimate_solar_central_duration(rST_ev, rTL_ev, gamma)
            solar.append({
                "t_s": t_ev, "date": date, "type": typ,
                "beta_deg": np.degrees(beta_ev),
                "gamma_deg": np.degrees(gamma),
                "rTL_km": rTL_ev / 1e3,
                "rST_km": rST_ev / 1e3,
                "alpha_S_as": np.degrees(alpha_S) * 3600,
                "alpha_L_as": np.degrees(alpha_L) * 3600,
                "duration_s": dur,
            })
        else:
            alpha_U = R_EARTH / rTL_ev - R_SUN / rST_ev
            alpha_P = R_EARTH / rTL_ev + R_SUN / rST_ev
            if gamma >= alpha_P + alpha_L:
                continue
            if alpha_U > 0 and gamma < alpha_U - alpha_L:
                typ = "totale"
            elif alpha_U > 0 and gamma < alpha_U + alpha_L:
                typ = "partielle"
            else:
                typ = "penombrale"
            dur = estimate_lunar_totality_duration(rST_ev, rTL_ev, gamma)
            lunar.append({
                "t_s": t_ev, "date": date, "type": typ,
                "beta_deg": np.degrees(beta_ev),
                "gamma_deg": np.degrees(gamma),
                "rTL_km": rTL_ev / 1e3,
                "rST_km": rST_ev / 1e3,
                "duration_s": dur,
            })

    return solar, lunar


# ============================================================
# APPARIEMENT SIMULATION ↔ JPL
# ============================================================
def match_eclipses(sim_list, jpl_list, tolerance_days=15):
    """Apparie chaque eclipse simulee a l'eclipse JPL la plus proche.

    Appariement glouton par ordre croissant d'ecart -- et non par ordre
    chronologique des eclipses simulees. Avec un parcours simplement
    chronologique, une eclipse simulee eloignee mais traitee en premier peut
    "voler" la meilleure correspondance JPL d'une eclipse simulee plus proche
    traitee plus tard (ex. observe : la simulation du 14 mars 2006 s'appariait
    a tort a l'eclipse JPL du 29 mars 2006, laissant sans partenaire l'eclipse
    simulee du 29 mars 2006, elle-meme une correspondance quasi exacte). En
    traitant les paires candidates par ecart croissant, la meilleure
    correspondance est toujours affectee en premier.
    """
    candidats = []
    for i, ecl in enumerate(sim_list):
        for j, (date_str, typ_jpl, dur_jpl) in enumerate(jpl_list):
            date_jpl = datetime.strptime(date_str, "%Y-%m-%d")
            dt = abs((ecl["date"] - date_jpl).total_seconds()) / JOUR
            if dt < tolerance_days:
                candidats.append((dt, i, j))
    candidats.sort(key=lambda c: c[0])

    used_sim, used_jpl, pairs = set(), set(), {}
    for dt, i, j in candidats:
        if i in used_sim or j in used_jpl:
            continue
        used_sim.add(i)
        used_jpl.add(j)
        pairs[i] = (j, dt)

    matches = []
    for i, ecl in enumerate(sim_list):
        if i in pairs:
            j, dt = pairs[i]
            date_str, typ_jpl, dur_jpl = jpl_list[j]
            type_ok = ecl["type"] in typ_jpl or typ_jpl in ecl["type"]
            matches.append({
                "sim": ecl,
                "jpl_date": date_str,
                "jpl_type": typ_jpl,
                "jpl_dur_s": dur_jpl,
                "dt_days": dt,
                "type_match": type_ok,
            })
        else:
            matches.append({
                "sim": ecl,
                "jpl_date": None, "jpl_type": None,
                "jpl_dur_s": None, "dt_days": None,
                "type_match": None,
            })

    unmatched_jpl = []
    for j, (date_str, typ_jpl, dur_jpl) in enumerate(jpl_list):
        if j not in used_jpl:
            unmatched_jpl.append({"date": date_str, "type": typ_jpl,
                                  "dur_s": dur_jpl})
    return matches, unmatched_jpl


# ============================================================
# COULEURS ET STYLES PAR TYPE
# ============================================================
TYPE_COLORS = {
    "totale": "#d62728",
    "annulaire": "#ff7f0e",
    "annulaire/totale": "#e377c2",
    "partielle": "#1f77b4",
    "penombrale": "#7f7f7f",
}
TYPE_MARKERS = {
    "totale": "o",
    "annulaire": "D",
    "annulaire/totale": "p",
    "partielle": "^",
    "penombrale": "s",
}


def type_color(typ):
    return TYPE_COLORS.get(typ, "black")


def type_marker(typ):
    return TYPE_MARKERS.get(typ, "x")


# ============================================================
# GENERATION DU PDF
# ============================================================
FS_SUPTITLE = 22
FS_TITLE    = 18
FS_LABEL    = 16
FS_TICK     = 14
FS_LEGEND   = 13
FS_TABLE    = 18     # encadres de texte a chasse fixe


def texte_monospace(ax, txt, hauteur_pouces, fs_max=None):
    """Bloc de texte a chasse fixe dont la police s'ajuste pour tenir dans le
    cadre (les resumes font une vingtaine de lignes)."""
    fs_max = FS_TABLE if fs_max is None else fs_max
    n_lignes = txt.count("\n") + 1
    fs = min(fs_max, max(9, int(hauteur_pouces * 72 * 0.75 / (n_lignes * 1.05))))
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=fs,
            va="top", fontfamily="monospace", linespacing=1.05)


def appliquer_polices():
    """Tailles de police lisibles une fois les pages reduites dans le livre."""
    plt.rcParams.update({
        "font.size":       FS_TICK,
        "axes.titlesize":  FS_TITLE,
        "axes.labelsize":  FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
    })


def generate_pdf(solar_m, lunar_m, solar_un, lunar_un, jup_str, path):
    appliquer_polices()
    with PdfPages(path) as pdf:
        _page_cover(pdf, solar_m, lunar_m, jup_str)
        _page_timeline_solar(pdf, solar_m, solar_un, jup_str)
        _page_timeline_lunar(pdf, lunar_m, lunar_un, jup_str)
        _page_ecarts(pdf, solar_m, lunar_m, jup_str)
        _page_types(pdf, solar_m, lunar_m, jup_str)
        _page_durees(pdf, solar_m, lunar_m, jup_str)
        _page_resume(pdf, solar_m, lunar_m, solar_un, lunar_un, jup_str)


# ---------- Page 1 : couverture ----------
def _page_cover(pdf, solar_m, lunar_m, jup_str):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax_title = fig.add_axes([0.0, 0.84, 1.0, 0.16])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "Eclipses du premier Saros (2005 – 2023)\n"
                  "Comparaison  Simulation Chapront  vs  JPL / NASA",
                  transform=ax_title.transAxes, fontsize=FS_SUPTITLE + 4, ha="center",
                  va="center", fontweight="bold")

    n_sol_matched = sum(1 for m in solar_m if m["jpl_date"])
    n_lun_matched = sum(1 for m in lunar_m if m["jpl_date"])
    n_sol_typeok = sum(1 for m in solar_m if m["type_match"])
    n_lun_typeok = sum(1 for m in lunar_m if m["type_match"])

    info = (
        f"Simulation : {jup_str}\n"
        f"Epoque       : 2 janvier 2005, 01h00 UTC  (C.I. Chapront)\n"
        f"Saros        : {SAROS_DAYS:.2f} j = {SAROS_DAYS / 365.25:.3f} ans\n"
        f"Fin Saros    : {SAROS_END.strftime('%d %B %Y')}\n"
        f"\n"
        f"METHODE DE DETECTION\n"
        f"{'=' * 52}\n"
        f"1. Elongation geocentrique  D = lambda_L - lambda_S\n"
        f"2. Syzygie : sin(D) = 0\n"
        f"   Nouvelle Lune (cos D > 0)  ->  eclipse solaire\n"
        f"   Pleine Lune   (cos D < 0)  ->  eclipse lunaire\n"
        f"3. Classification par rayons angulaires apparents :\n"
        f"   alpha_S = R_Sun / r_ST      (Soleil)\n"
        f"   alpha_L = R_Moon / r_TL     (Lune)\n"
        f"   pi_L    = R_Earth / r_TL    (parallaxe)\n"
        f"   gamma   = |beta|            (separation)\n"
        f"4. Duree estimee de la phase centrale :\n"
        f"   dt = 2 * sqrt(Dalpha^2 - gamma_eff^2) / omega_syn\n"
        f"   x facteur rotation terrestre (~1.5)\n"
        f"\n"
        f"RESULTATS\n"
        f"{'=' * 52}\n"
        f"Solaires : {len(solar_m):2d} simulees   "
        f"({n_sol_matched} appariees, {n_sol_typeok} types OK)\n"
        f"           {len(ECLIPSES_JPL_SOLAR):2d} JPL/NASA\n"
        f"Lunaires : {len(lunar_m):2d} simulees   "
        f"({n_lun_matched} appariees, {n_lun_typeok} types OK)\n"
        f"           {len(ECLIPSES_JPL_LUNAR):2d} JPL/NASA\n"
    )
    ax_box = fig.add_axes([0.03, 0.02, 0.94, 0.80])
    ax_box.set_facecolor("lightyellow")
    ax_box.axis("off")
    texte_monospace(ax_box, info, 0.80 * 8.27)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ---------- Utilitaire timeline ----------
def _date_to_year(dt_obj):
    """Convertit un datetime en annee decimale."""
    start = datetime(dt_obj.year, 1, 1)
    end = datetime(dt_obj.year + 1, 1, 1)
    frac = (dt_obj - start).total_seconds() / (end - start).total_seconds()
    return dt_obj.year + frac


def _draw_timeline(ax, matches, jpl_list, unmatched_jpl, title):
    """Trace la chronologie comparee simulation / JPL."""
    y_sim = 1.0
    y_jpl = 0.0

    for m in matches:
        ecl = m["sim"]
        x_sim = _date_to_year(ecl["date"])
        ax.scatter(x_sim, y_sim, c=type_color(ecl["type"]),
                   marker=type_marker(ecl["type"]), s=90, zorder=5,
                   edgecolors="black", linewidths=0.5)
        if m["jpl_date"]:
            x_jpl = _date_to_year(
                datetime.strptime(m["jpl_date"], "%Y-%m-%d"))
            ax.plot([x_sim, x_jpl], [y_sim - 0.05, y_jpl + 0.05],
                    "-", color="gray", lw=1.0, alpha=0.6)

    for date_str, typ_jpl, _ in jpl_list:
        x_jpl = _date_to_year(datetime.strptime(date_str, "%Y-%m-%d"))
        ax.scatter(x_jpl, y_jpl, c=type_color(typ_jpl),
                   marker=type_marker(typ_jpl), s=90, zorder=5,
                   edgecolors="black", linewidths=0.5)

    for um in unmatched_jpl:
        x = _date_to_year(datetime.strptime(um["date"], "%Y-%m-%d"))
        ax.scatter(x, y_jpl, c=type_color(um["type"]),
                   marker=type_marker(um["type"]), s=140, zorder=6,
                   edgecolors="red", linewidths=1.5)

    ax.axhline(y_sim, color="steelblue", lw=1.6, alpha=0.35)
    ax.axhline(y_jpl, color="darkgreen", lw=1.6, alpha=0.35)
    ax.set_yticks([y_jpl, y_sim])
    ax.set_yticklabels(["JPL / NASA", "Simulation"], fontsize=FS_LABEL)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("Annee", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")

    for year in range(2005, 2024):
        ax.axvline(year, color="lightgray", lw=0.3, zorder=0)
    ax.set_xlim(2004.8, 2023.5)
    ax.grid(axis="x", alpha=0.2)


def _type_legend(ax):
    """Ajoute une legende des types."""
    handles = []
    for typ in ["totale", "annulaire", "annulaire/totale",
                "partielle", "penombrale"]:
        handles.append(Line2D([0], [0], marker=type_marker(typ),
                              color="w", markerfacecolor=type_color(typ),
                              markeredgecolor="black", markersize=11,
                              label=typ))
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="white", markeredgecolor="red",
                          markeredgewidth=1.5, markersize=11,
                          label="JPL non appariee"))
    ax.legend(handles=handles, loc="upper right", fontsize=FS_LEGEND,
              ncol=3, framealpha=0.9)


# ---------- Page 2 : chronologie solaire ----------
def _page_timeline_solar(pdf, solar_m, solar_un, jup_str):
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 7.2),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Eclipses solaires — premier Saros  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    _draw_timeline(axes[0], solar_m, ECLIPSES_JPL_SOLAR,
                   solar_un, "Chronologie comparee")
    _type_legend(axes[0])

    matched = [m for m in solar_m if m["jpl_date"]]
    if matched:
        dates = [_date_to_year(m["sim"]["date"]) for m in matched]
        ecarts = [m["dt_days"] for m in matched]
        colors = ["green" if m["type_match"] else "red" for m in matched]
        axes[1].bar(dates, ecarts, width=0.15, color=colors, alpha=0.7,
                    edgecolor="black", linewidth=0.3)
        axes[1].set_ylabel("Ecart (jours)", fontsize=FS_LABEL)
        axes[1].set_xlabel("Annee", fontsize=FS_LABEL)
        axes[1].set_title("Ecart de date |sim - JPL| (vert = type OK, "
                          "rouge = type different)", fontsize=FS_LABEL)
        axes[1].set_xlim(2004.8, 2023.5)
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ---------- Page 3 : chronologie lunaire ----------
def _page_timeline_lunar(pdf, lunar_m, lunar_un, jup_str):
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"Eclipses lunaires — premier Saros  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    _draw_timeline(axes[0], lunar_m, ECLIPSES_JPL_LUNAR,
                   lunar_un, "Chronologie comparee")
    _type_legend(axes[0])

    matched = [m for m in lunar_m if m["jpl_date"]]
    if matched:
        dates = [_date_to_year(m["sim"]["date"]) for m in matched]
        ecarts = [m["dt_days"] for m in matched]
        colors = ["green" if m["type_match"] else "red" for m in matched]
        axes[1].bar(dates, ecarts, width=0.15, color=colors, alpha=0.7,
                    edgecolor="black", linewidth=0.3)
        axes[1].set_ylabel("Ecart (jours)", fontsize=FS_LABEL)
        axes[1].set_xlabel("Annee", fontsize=FS_LABEL)
        axes[1].set_title("Ecart de date |sim - JPL| (vert = type OK, "
                          "rouge = type different)", fontsize=FS_LABEL)
        axes[1].set_xlim(2004.8, 2023.5)
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ---------- Page 4 : ecarts ----------
def _page_ecarts(pdf, solar_m, lunar_m, jup_str):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 7.2))
    fig.suptitle(f"Analyse des ecarts de date  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    all_matched = ([m for m in solar_m if m["jpl_date"]] +
                   [m for m in lunar_m if m["jpl_date"]])
    sol_matched = [m for m in solar_m if m["jpl_date"]]
    lun_matched = [m for m in lunar_m if m["jpl_date"]]

    # (0,0) Histogramme global
    if all_matched:
        dt_all = [m["dt_days"] for m in all_matched]
        axes[0, 0].hist(dt_all, bins=25, color="steelblue", alpha=0.7,
                        edgecolor="black", linewidth=0.5)
        axes[0, 0].axvline(np.mean(dt_all), color="red", ls="--", lw=1.5,
                           label=f"moyenne = {np.mean(dt_all):.1f} j")
        axes[0, 0].axvline(np.median(dt_all), color="green", ls=":", lw=1.5,
                           label=f"mediane = {np.median(dt_all):.1f} j")
        axes[0, 0].set_xlabel("Ecart |sim - JPL| (jours)")
        axes[0, 0].set_ylabel("Nombre d'eclipses")
        axes[0, 0].set_title("Distribution des ecarts (toutes eclipses)")
        axes[0, 0].legend(fontsize=FS_LEGEND)
        axes[0, 0].grid(True, alpha=0.3)

    # (0,1) Ecart vs temps
    if sol_matched:
        x = [_date_to_year(m["sim"]["date"]) for m in sol_matched]
        y = [m["dt_days"] for m in sol_matched]
        axes[0, 1].scatter(x, y, c="red", s=45, alpha=0.75,
                           label="solaires", edgecolors="black",
                           linewidths=0.3)
    if lun_matched:
        x = [_date_to_year(m["sim"]["date"]) for m in lun_matched]
        y = [m["dt_days"] for m in lun_matched]
        axes[0, 1].scatter(x, y, c="purple", s=45, alpha=0.75,
                           label="lunaires", edgecolors="black",
                           linewidths=0.3)
    axes[0, 1].set_xlabel("Annee")
    axes[0, 1].set_ylabel("Ecart (jours)")
    axes[0, 1].set_title("Ecart de date en fonction du temps")
    axes[0, 1].legend(fontsize=FS_LEGEND)
    axes[0, 1].grid(True, alpha=0.3)

    # (1,0) Histogramme solaire vs lunaire
    if sol_matched and lun_matched:
        dt_sol = [m["dt_days"] for m in sol_matched]
        dt_lun = [m["dt_days"] for m in lun_matched]
        bins = np.linspace(0, max(max(dt_sol), max(dt_lun)) + 0.5, 20)
        axes[1, 0].hist(dt_sol, bins=bins, color="red", alpha=0.5,
                        label=f"solaires (moy {np.mean(dt_sol):.1f} j)",
                        edgecolor="black", linewidth=0.3)
        axes[1, 0].hist(dt_lun, bins=bins, color="purple", alpha=0.5,
                        label=f"lunaires (moy {np.mean(dt_lun):.1f} j)",
                        edgecolor="black", linewidth=0.3)
        axes[1, 0].set_xlabel("Ecart (jours)")
        axes[1, 0].set_ylabel("Nombre")
        axes[1, 0].set_title("Comparaison solaires / lunaires")
        axes[1, 0].legend(fontsize=FS_LEGEND)
        axes[1, 0].grid(True, alpha=0.3)

    # (1,1) Resume texte
    axes[1, 1].axis("off")
    if all_matched:
        dt_all = [m["dt_days"] for m in all_matched]
        dt_sol = [m["dt_days"] for m in sol_matched]
        dt_lun = [m["dt_days"] for m in lun_matched]
        txt = (
            "ECARTS DE DATE  (simulation - JPL)\n"
            f"{'=' * 45}\n\n"
            f"Toutes eclipses :\n"
            f"  N appariees   : {len(all_matched)}\n"
            f"  Ecart moyen   : {np.mean(dt_all):.2f} j\n"
            f"  Ecart median  : {np.median(dt_all):.2f} j\n"
            f"  Ecart max     : {np.max(dt_all):.2f} j\n"
            f"  Ecart-type    : {np.std(dt_all):.2f} j\n\n"
            f"Solaires :\n"
            f"  N = {len(dt_sol)},  moy = {np.mean(dt_sol):.2f} j\n\n"
            f"Lunaires :\n"
            f"  N = {len(dt_lun)},  moy = {np.mean(dt_lun):.2f} j\n\n"
            f"L'ecart moyen de ~{np.mean(dt_all):.0f} jours\n"
            f"provient de l'erreur de ~0.002 j/mois\n"
            f"sur le mois synodique simule\n"
            f"(29.529 j vs 29.531 j ref.)\n"
        )
        axes[1, 1].set_facecolor("lightyellow")
        texte_monospace(axes[1, 1], txt, 4.0)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ---------- Page 5 : types ----------
def _page_types(pdf, solar_m, lunar_m, jup_str):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"Concordance des types d'eclipse  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    # --- Solaires ---
    sol_matched = [m for m in solar_m if m["jpl_date"]]
    types_sol_sim = ["totale", "annulaire", "partielle"]
    types_sol_jpl = ["totale", "annulaire", "annulaire/totale", "partielle"]

    # (0,0) Matrice de confusion solaire
    if sol_matched:
        mat = np.zeros((len(types_sol_sim), len(types_sol_jpl)))
        for m in sol_matched:
            i = types_sol_sim.index(m["sim"]["type"]) if m["sim"]["type"] in types_sol_sim else -1
            j = types_sol_jpl.index(m["jpl_type"]) if m["jpl_type"] in types_sol_jpl else -1
            if i >= 0 and j >= 0:
                mat[i, j] += 1
        im = axes[0, 0].imshow(mat, cmap="YlOrRd", aspect="auto")
        axes[0, 0].set_xticks(range(len(types_sol_jpl)))
        axes[0, 0].set_xticklabels(types_sol_jpl, fontsize=FS_TICK - 1, rotation=30)
        axes[0, 0].set_yticks(range(len(types_sol_sim)))
        axes[0, 0].set_yticklabels(types_sol_sim, fontsize=FS_LEGEND)
        axes[0, 0].set_xlabel("Type JPL")
        axes[0, 0].set_ylabel("Type Simulation")
        axes[0, 0].set_title("Matrice de confusion — solaires")
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                if mat[ii, jj] > 0:
                    axes[0, 0].text(jj, ii, f"{int(mat[ii, jj])}",
                                    ha="center", va="center", fontsize=FS_LEGEND,
                                    fontweight="bold")
        fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # --- Lunaires ---
    lun_matched = [m for m in lunar_m if m["jpl_date"]]
    types_lun = ["totale", "partielle", "penombrale"]

    # (0,1) Matrice de confusion lunaire
    if lun_matched:
        mat = np.zeros((len(types_lun), len(types_lun)))
        for m in lun_matched:
            i = types_lun.index(m["sim"]["type"]) if m["sim"]["type"] in types_lun else -1
            j = types_lun.index(m["jpl_type"]) if m["jpl_type"] in types_lun else -1
            if i >= 0 and j >= 0:
                mat[i, j] += 1
        im = axes[0, 1].imshow(mat, cmap="YlOrRd", aspect="auto")
        axes[0, 1].set_xticks(range(len(types_lun)))
        axes[0, 1].set_xticklabels(types_lun, fontsize=FS_TICK - 1, rotation=30)
        axes[0, 1].set_yticks(range(len(types_lun)))
        axes[0, 1].set_yticklabels(types_lun, fontsize=FS_LEGEND)
        axes[0, 1].set_xlabel("Type JPL")
        axes[0, 1].set_ylabel("Type Simulation")
        axes[0, 1].set_title("Matrice de confusion — lunaires")
        for ii in range(mat.shape[0]):
            for jj in range(mat.shape[1]):
                if mat[ii, jj] > 0:
                    axes[0, 1].text(jj, ii, f"{int(mat[ii, jj])}",
                                    ha="center", va="center", fontsize=FS_LEGEND,
                                    fontweight="bold")
        fig.colorbar(im, ax=axes[0, 1], shrink=0.8)

    # (1,0) Barres : nombre par type (sim vs JPL) — solaires
    if sol_matched:
        types_for_bar = ["totale", "annulaire", "partielle"]
        counts_sim = [sum(1 for m in sol_matched
                         if m["sim"]["type"] == t) for t in types_for_bar]
        counts_jpl = []
        for t in types_for_bar:
            c = sum(1 for _, tp, _ in ECLIPSES_JPL_SOLAR if t in tp)
            counts_jpl.append(c)
        x = np.arange(len(types_for_bar))
        w = 0.35
        axes[1, 0].bar(x - w / 2, counts_sim, w, color="steelblue",
                       label="Simulation", edgecolor="black", linewidth=0.5)
        axes[1, 0].bar(x + w / 2, counts_jpl, w, color="orange",
                       label="JPL/NASA", edgecolor="black", linewidth=0.5)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(types_for_bar, fontsize=FS_LABEL)
        axes[1, 0].set_ylabel("Nombre")
        axes[1, 0].set_title("Repartition par type — solaires")
        axes[1, 0].legend(fontsize=FS_LEGEND)
        axes[1, 0].grid(axis="y", alpha=0.3)

    # (1,1) Barres : nombre par type (sim vs JPL) — lunaires
    if lun_matched:
        types_for_bar = ["totale", "partielle", "penombrale"]
        counts_sim = [sum(1 for m in lun_matched
                         if m["sim"]["type"] == t) for t in types_for_bar]
        counts_jpl = [sum(1 for _, tp, _ in ECLIPSES_JPL_LUNAR
                         if tp == t) for t in types_for_bar]
        x = np.arange(len(types_for_bar))
        w = 0.35
        axes[1, 1].bar(x - w / 2, counts_sim, w, color="steelblue",
                       label="Simulation", edgecolor="black", linewidth=0.5)
        axes[1, 1].bar(x + w / 2, counts_jpl, w, color="orange",
                       label="JPL/NASA", edgecolor="black", linewidth=0.5)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(types_for_bar, fontsize=FS_LABEL)
        axes[1, 1].set_ylabel("Nombre")
        axes[1, 1].set_title("Repartition par type — lunaires")
        axes[1, 1].legend(fontsize=FS_LEGEND)
        axes[1, 1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ---------- Page 6 : durees ----------
def _page_durees(pdf, solar_m, lunar_m, jup_str):
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle(f"Comparaison des durees estimees  ({jup_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    # Solaires centrales (total ou annulaire)
    sol_central = [m for m in solar_m
                   if m["jpl_date"]
                   and m["jpl_dur_s"] is not None and m["jpl_dur_s"] > 0
                   and m["sim"]["duration_s"] > 0]

    if sol_central:
        d_sim = [m["sim"]["duration_s"] for m in sol_central]
        d_jpl = [m["jpl_dur_s"] for m in sol_central]
        colors = [type_color(m["sim"]["type"]) for m in sol_central]

        axes[0, 0].scatter(d_jpl, d_sim, c=colors, s=50, edgecolors="black",
                           linewidths=0.5, zorder=5)
        max_d = max(max(d_sim), max(d_jpl)) * 1.1
        axes[0, 0].plot([0, max_d], [0, max_d], "k--", lw=1, alpha=0.5,
                        label="y = x")
        if len(d_sim) > 1:
            coef = np.polyfit(d_jpl, d_sim, 1)
            x_fit = np.linspace(0, max_d, 100)
            axes[0, 0].plot(x_fit, np.polyval(coef, x_fit), "r-", lw=1.5,
                            label=f"regression : y = {coef[0]:.2f}x "
                                  f"+ {coef[1]:.0f}")
        axes[0, 0].set_xlabel("Duree JPL (s)")
        axes[0, 0].set_ylabel("Duree estimee simulation (s)")
        axes[0, 0].set_title("Eclipses solaires centrales")
        axes[0, 0].legend(fontsize=FS_LEGEND)
        axes[0, 0].grid(True, alpha=0.3)

        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=type_color(t),
                          markeredgecolor="black", markersize=11, label=t)
                   for t in ["totale", "annulaire"]]
        axes[0, 0].legend(handles=handles + axes[0, 0].get_legend_handles_labels()[0],
                          fontsize=FS_LEGEND - 2, loc="upper left")

    # Solaires : durees en barres
    sol_bar = [m for m in solar_m
               if m["jpl_date"]
               and m["sim"]["type"] in ("totale", "annulaire")
               and m["jpl_type"] in ("totale", "annulaire",
                                     "annulaire/totale")]
    if sol_bar:
        labels = [m["jpl_date"][:7] for m in sol_bar]
        d_sim = [m["sim"]["duration_s"] for m in sol_bar]
        d_jpl = [m["jpl_dur_s"] for m in sol_bar]
        x = np.arange(len(labels))
        w = 0.35
        axes[0, 1].barh(x - w / 2, d_sim, w, color="steelblue",
                        label="Simulation", edgecolor="black", linewidth=0.3)
        axes[0, 1].barh(x + w / 2, d_jpl, w, color="orange",
                        label="JPL/NASA", edgecolor="black", linewidth=0.3)
        axes[0, 1].set_yticks(x)
        axes[0, 1].set_yticklabels(labels, fontsize=FS_LEGEND - 2)
        axes[0, 1].set_xlabel("Duree phase centrale (s)")
        axes[0, 1].set_title("Solaires centrales — sim vs JPL")
        axes[0, 1].legend(fontsize=FS_LEGEND)
        axes[0, 1].grid(axis="x", alpha=0.3)
        axes[0, 1].invert_yaxis()

    # Lunaires totales
    lun_total = [m for m in lunar_m
                 if m["jpl_date"]
                 and m["jpl_dur_s"] is not None and m["jpl_dur_s"] > 0
                 and m["sim"]["duration_s"] > 0]

    if lun_total:
        d_sim = [m["sim"]["duration_s"] for m in lun_total]
        d_jpl = [m["jpl_dur_s"] for m in lun_total]

        axes[1, 0].scatter(d_jpl, d_sim, c="darkred", s=50,
                           edgecolors="black", linewidths=0.5, zorder=5)
        max_d = max(max(d_sim), max(d_jpl)) * 1.1
        axes[1, 0].plot([0, max_d], [0, max_d], "k--", lw=1, alpha=0.5,
                        label="y = x")
        if len(d_sim) > 1:
            coef = np.polyfit(d_jpl, d_sim, 1)
            x_fit = np.linspace(0, max_d, 100)
            axes[1, 0].plot(x_fit, np.polyval(coef, x_fit), "r-", lw=1.5,
                            label=f"regression : y = {coef[0]:.2f}x "
                                  f"+ {coef[1]:.0f}")
        axes[1, 0].set_xlabel("Duree totalite JPL (s)")
        axes[1, 0].set_ylabel("Duree estimee simulation (s)")
        axes[1, 0].set_title("Eclipses lunaires totales")
        axes[1, 0].legend(fontsize=FS_LEGEND)
        axes[1, 0].grid(True, alpha=0.3)

    # Lunaires : barres
    if lun_total:
        labels = [m["jpl_date"][:7] for m in lun_total]
        d_sim = [m["sim"]["duration_s"] / 60 for m in lun_total]
        d_jpl = [m["jpl_dur_s"] / 60 for m in lun_total]
        x = np.arange(len(labels))
        w = 0.35
        axes[1, 1].barh(x - w / 2, d_sim, w, color="steelblue",
                        label="Simulation", edgecolor="black", linewidth=0.3)
        axes[1, 1].barh(x + w / 2, d_jpl, w, color="orange",
                        label="JPL/NASA", edgecolor="black", linewidth=0.3)
        axes[1, 1].set_yticks(x)
        axes[1, 1].set_yticklabels(labels, fontsize=FS_LEGEND - 2)
        axes[1, 1].set_xlabel("Duree totalite (minutes)")
        axes[1, 1].set_title("Lunaires totales — sim vs JPL")
        axes[1, 1].legend(fontsize=FS_LEGEND)
        axes[1, 1].grid(axis="x", alpha=0.3)
        axes[1, 1].invert_yaxis()

    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ---------- Page 7 : resume ----------
def _page_resume(pdf, solar_m, lunar_m, solar_un, lunar_un, jup_str):
    fig = plt.figure(figsize=(11.69, 8.27))

    # Tableau solaire
    ax1 = fig.add_axes([0.03, 0.52, 0.94, 0.42])
    ax1.axis("off")
    ax1.set_title("LISTE DES ECLIPSES SOLAIRES — Simulation vs JPL",
                  fontsize=FS_TITLE, fontweight="bold", loc="left")

    header = ["#", "Date sim", "Type sim", "Duree sim", "Date JPL",
              "Type JPL", "Duree JPL", "Ecart(j)", "Type OK"]
    rows = []
    for i, m in enumerate(solar_m):
        e = m["sim"]
        d_sim = f"{e['duration_s']:.0f}s" if e["duration_s"] > 0 else "-"
        if m["jpl_date"]:
            d_jpl = (f"{m['jpl_dur_s']:.0f}s"
                     if m["jpl_dur_s"] and m["jpl_dur_s"] > 0 else "-")
            dt_str = f"{m['dt_days']:.1f}"
            tok = "OUI" if m["type_match"] else "non"
        else:
            d_jpl, dt_str, tok = "-", "-", "-"
        rows.append([
            str(i + 1),
            e["date"].strftime("%Y-%m-%d"),
            e["type"],
            d_sim,
            m["jpl_date"] or "-",
            m["jpl_type"] or "-",
            d_jpl,
            dt_str,
            tok,
        ])

    if rows:
        colors = []
        for m in solar_m:
            if m["type_match"] is True:
                colors.append(["white"] * 9)
            elif m["type_match"] is False:
                colors.append(["#ffcccc"] * 9)
            else:
                colors.append(["#ffffcc"] * 9)
        table = ax1.table(cellText=rows, colLabels=header,
                          cellColours=colors,
                          loc="upper left", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.05)

    # Tableau lunaire
    ax2 = fig.add_axes([0.03, 0.02, 0.94, 0.48])
    ax2.axis("off")
    ax2.set_title("LISTE DES ECLIPSES LUNAIRES — Simulation vs JPL",
                  fontsize=FS_TITLE, fontweight="bold", loc="left")

    rows2 = []
    for i, m in enumerate(lunar_m):
        e = m["sim"]
        d_sim = (f"{e['duration_s'] / 60:.0f}m"
                 if e["duration_s"] > 0 else "-")
        if m["jpl_date"]:
            d_jpl = (f"{m['jpl_dur_s'] / 60:.0f}m"
                     if m["jpl_dur_s"] and m["jpl_dur_s"] > 0 else "-")
            dt_str = f"{m['dt_days']:.1f}"
            tok = "OUI" if m["type_match"] else "non"
        else:
            d_jpl, dt_str, tok = "-", "-", "-"
        rows2.append([
            str(i + 1),
            e["date"].strftime("%Y-%m-%d"),
            e["type"],
            d_sim,
            m["jpl_date"] or "-",
            m["jpl_type"] or "-",
            d_jpl,
            dt_str,
            tok,
        ])

    if rows2:
        colors2 = []
        for m in lunar_m:
            if m["type_match"] is True:
                colors2.append(["white"] * 9)
            elif m["type_match"] is False:
                colors2.append(["#ffcccc"] * 9)
            else:
                colors2.append(["#ffffcc"] * 9)
        table2 = ax2.table(cellText=rows2, colLabels=header,
                           cellColours=colors2,
                           loc="upper left", cellLoc="center")
        table2.auto_set_font_size(False)
        table2.set_fontsize(7)
        table2.scale(1.0, 1.05)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ============================================================
# AFFICHAGE TERMINAL
# ============================================================
def print_summary(solar_m, lunar_m, solar_un, lunar_un):
    print("\n" + "=" * 80)
    print("ECLIPSES SOLAIRES — SIMULATION vs JPL")
    print("=" * 80)
    print(f"{'#':>3s}  {'Date sim':>12s}  {'Type sim':>11s}  {'Dur sim':>8s}"
          f"  {'Date JPL':>12s}  {'Type JPL':>16s}  {'Dur JPL':>8s}"
          f"  {'Ecart':>6s}  {'OK':>3s}")
    print("-" * 100)
    for i, m in enumerate(solar_m):
        e = m["sim"]
        ds = f"{e['duration_s']:.0f}s" if e["duration_s"] > 0 else "-"
        if m["jpl_date"]:
            dj = (f"{m['jpl_dur_s']:.0f}s"
                  if m["jpl_dur_s"] and m["jpl_dur_s"] > 0 else "-")
            dt = f"{m['dt_days']:.1f}"
            ok = "OUI" if m["type_match"] else "non"
        else:
            dj, dt, ok = "-", "-", "-"
        print(f"{i + 1:3d}  {e['date'].strftime('%Y-%m-%d'):>12s}"
              f"  {e['type']:>11s}  {ds:>8s}"
              f"  {m['jpl_date'] or '-':>12s}  {m['jpl_type'] or '-':>16s}"
              f"  {dj:>8s}  {dt:>6s}  {ok:>3s}")

    if solar_un:
        print(f"\nJPL non appariees ({len(solar_un)}) :")
        for u in solar_un:
            print(f"  {u['date']}  {u['type']}")

    print("\n" + "=" * 80)
    print("ECLIPSES LUNAIRES — SIMULATION vs JPL")
    print("=" * 80)
    print(f"{'#':>3s}  {'Date sim':>12s}  {'Type sim':>11s}  {'Dur sim':>8s}"
          f"  {'Date JPL':>12s}  {'Type JPL':>16s}  {'Dur JPL':>8s}"
          f"  {'Ecart':>6s}  {'OK':>3s}")
    print("-" * 100)
    for i, m in enumerate(lunar_m):
        e = m["sim"]
        ds = (f"{e['duration_s'] / 60:.0f}m"
              if e["duration_s"] > 0 else "-")
        if m["jpl_date"]:
            dj = (f"{m['jpl_dur_s'] / 60:.0f}m"
                  if m["jpl_dur_s"] and m["jpl_dur_s"] > 0 else "-")
            dt = f"{m['dt_days']:.1f}"
            ok = "OUI" if m["type_match"] else "non"
        else:
            dj, dt, ok = "-", "-", "-"
        print(f"{i + 1:3d}  {e['date'].strftime('%Y-%m-%d'):>12s}"
              f"  {e['type']:>11s}  {ds:>8s}"
              f"  {m['jpl_date'] or '-':>12s}  {m['jpl_type'] or '-':>16s}"
              f"  {dj:>8s}  {dt:>6s}  {ok:>3s}")

    if lunar_un:
        print(f"\nJPL non appariees ({len(lunar_un)}) :")
        for u in lunar_un:
            print(f"  {u['date']}  {u['type']}")

    # Statistiques globales
    all_matched = ([m for m in solar_m if m["jpl_date"]] +
                   [m for m in lunar_m if m["jpl_date"]])
    dt_all = [m["dt_days"] for m in all_matched]
    type_ok = sum(1 for m in all_matched if m["type_match"])
    print(f"\n{'=' * 60}")
    print("RESUME")
    print(f"{'=' * 60}")
    print(f"Eclipses appariees : {len(all_matched)}")
    print(f"Ecart moyen        : {np.mean(dt_all):.2f} j")
    print(f"Ecart max          : {np.max(dt_all):.2f} j")
    print(f"Types corrects     : {type_ok}/{len(all_matched)} "
          f"({100 * type_ok / max(1, len(all_matched)):.0f}%)")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Eclipses premier Saros : simulation vs JPL/NASA")
    parser.add_argument("--npz", default=os.path.join(
        TLS_DIR, "resultats", "simulation_data_avecJupiter.npz"))
    args = parser.parse_args()

    if not os.path.exists(args.npz):
        print(f"Erreur : fichier {args.npz} introuvable.")
        print("Executez d'abord, dans ../04_terre_lune_soleil : python3 simulation_TLS_Jupiter.py")
        sys.exit(1)

    print(f"Chargement : {args.npz}")
    d = np.load(args.npz, allow_pickle=True)
    jup_str = "avec Jupiter" if bool(d["jupiter"]) else "sans Jupiter"

    print(f"Simulation : {jup_str}")
    print(f"Premier Saros : {EPOCH.strftime('%Y-%m-%d')} -> "
          f"{SAROS_END.strftime('%Y-%m-%d')} ({SAROS_DAYS:.0f} j)")

    print("\nDetection des eclipses...")
    solar, lunar = detect_eclipses(d, SAROS_DAYS)
    print(f"  Solaires : {len(solar)}")
    print(f"  Lunaires : {len(lunar)}")

    print("\nAppariement avec le catalogue JPL/NASA...")
    solar_m, solar_un = match_eclipses(solar, ECLIPSES_JPL_SOLAR)
    lunar_m, lunar_un = match_eclipses(lunar, ECLIPSES_JPL_LUNAR)

    print_summary(solar_m, lunar_m, solar_un, lunar_un)

    out_dir = os.path.join(HERE, "resultats")
    os.makedirs(out_dir, exist_ok=True)
    tag = "avecJupiter" if "avec" in jup_str else "sansJupiter"
    out_path = os.path.join(out_dir, f"comparaison_eclipses_saros_{tag}.pdf")

    print(f"\nGeneration du PDF...")
    generate_pdf(solar_m, lunar_m, solar_un, lunar_un, jup_str, out_path)
    print(f"PDF genere : {out_path}")


if __name__ == "__main__":
    main()
