#!/usr/bin/env python3
"""
Simulation du systeme Soleil-Terre-Lune avec perturbation optionnelle de Jupiter
=================================================================================
Usage :
    python3 simulation_TLS_Jupiter.py                  # avec Jupiter (defaut)
    python3 simulation_TLS_Jupiter.py --no-jupiter     # sans Jupiter
    python3 simulation_TLS_Jupiter.py --dt 600 --years 3800

Fichiers produits dans le dossier resultats/ :
    simulation_data_<tag>.npz   donnees brutes
    fig1 a fig7                 graphiques PNG

Le noyau C (simulation_core.c) est compile automatiquement si necessaire.
"""

import argparse
import numpy as np
import ctypes as ct
import os, sys, time as time_mod
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ===================================================================
# CONSTANTES PHYSIQUES (SI)
# ===================================================================
G    = 6.67259e-11
M_S  = 1.98891e30
M_T  = 5.9736e24
M_L  = 7.3475e22
M_J  = 1.89813e27
UA   = 149597870691.0
JOUR = 86400.0

# ===================================================================
# CONDITIONS INITIALES  -  Chapront, 2 jan 2005 01h00
# ===================================================================
xT0  = -0.198811444 * UA;  yT0 =  0.962988397 * UA;  zT0 = -0.000012641 * UA
vxT0 = -0.017128381 * UA/JOUR; vyT0 = -0.003536105 * UA/JOUR; vzT0 = 0.000000553 * UA/JOUR
xL0  = -0.201425635 * UA;  yL0 =  0.963356840 * UA;  zL0 =  0.000126956 * UA
vxL0 = -0.017180654 * UA/JOUR; vyL0 = -0.004103967 * UA/JOUR; vzL0 = -0.000041144 * UA/JOUR

STORE_EVERY = 400

# Inegalites lunaires connues (Brown)
EPSILON = np.radians(23.439281)  # obliquite de l'ecliptique

INEQ_LONGITUDE = [
    ("Periode siderale",       27.3217),
    ("Evection",               31.812),
    ("Variation (T Brahe)",    14.765),
    ("Reduction ecliptique",   13.606),
    ("Inegalite des apsides", 205.9),
    ("Equation annuelle",     365.26),
    ("Mois anomalistique",     27.555),
    ("Parallaxe solaire",      29.530),
    ("Ineg. semi-annuelle",   182.6),
    ("Eq. du centre Lune",     27.555),
    ("Terme de Newcomb",       14.694),
    ("Ineg. mensuelle",        31.812),
]
INEQ_LATITUDE = [
    ("Grande inegalite lat.",  32.28),
    ("Periode siderale",       27.212),
    ("Demi-mois",             13.777),
    ("Ineg. lat. evection",    31.812),
    ("Lat. annuelle",         365.26),
]


def parse_args():
    p = argparse.ArgumentParser(description="Simulation Terre-Lune-Soleil (+Jupiter)")
    p.add_argument("--no-jupiter", action="store_true",
                   help="Desactive la perturbation de Jupiter")
    p.add_argument("--dt", type=float, default=500.0,
                   help="Pas de temps en secondes (defaut: 500)")
    p.add_argument("--years", type=float, default=4000.0,
                   help="Duree simulee en annees (defaut: 4000)")
    p.add_argument("--store-every", type=int, default=400,
                   help="Echantillonnage : 1 point tous les N pas (defaut: 400)")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffixe pour les fichiers de sortie")
    return p.parse_args()


def compute_jupiter_ic():
    JD = 2453372.0 + 1.0/24.0 + 0.5
    T  = (JD - 2451545.0) / 36525.0
    a_au = 5.20288700 + (-0.00011607)*T
    e    = 0.04838624 + (-0.00013253)*T
    I    = np.radians(1.30439695 + (-0.00183714)*T)
    L    = np.radians((34.39644051 + 3034.74612775*T) % 360)
    wbar = np.radians((14.72847983 + 0.21252668*T) % 360)
    Om   = np.radians((100.47390909 + 0.20469106*T) % 360)
    omega = wbar - Om;  M = L - wbar
    E = M + 0.0
    for _ in range(30):
        E = M + e*np.sin(E)
    nu = 2.0*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r_au = a_au*(1.0 - e*np.cos(E))
    x_o = r_au*UA*np.cos(nu);  y_o = r_au*UA*np.sin(nu)
    mu = G*(M_S+M_J);  a_m = a_au*UA;  h = np.sqrt(mu*a_m*(1-e**2))
    vx_o = -(mu/h)*np.sin(nu);  vy_o = (mu/h)*(e+np.cos(nu))
    co,so = np.cos(omega),np.sin(omega)
    cO,sO = np.cos(Om),np.sin(Om)
    ci,si = np.cos(I),np.sin(I)
    P = np.array([[cO*co-sO*so*ci, -cO*so-sO*co*ci, sO*si],
                  [sO*co+cO*so*ci, -sO*so+cO*co*ci,-cO*si],
                  [so*si,           co*si,           ci   ]])
    pos = P @ np.array([x_o, y_o, 0.0])
    vel = P @ np.array([vx_o, vy_o, 0.0])
    return pos, vel


def ensure_library():
    c_src  = os.path.join(HERE, "simulation_core.c")
    so_lib = os.path.join(HERE, "libsimulation.so")
    if not os.path.exists(so_lib) or os.path.getmtime(c_src) > os.path.getmtime(so_lib):
        print("Compilation du noyau C...")
        r = subprocess.run(
            ["gcc", "-O2", "-shared", "-fPIC", "-o", so_lib, c_src, "-lm"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr); sys.exit(1)
        print("  -> libsimulation.so")
    return ct.CDLL(so_lib)


def run_simulation(args):
    lib = ensure_library()

    D  = ct.c_double;  LP = ct.POINTER(ct.c_double);  IP = ct.POINTER(ct.c_int)
    lib.simulate.restype  = None
    lib.simulate.argtypes = [
        D,D,D,D,D,D, D,D,D,D,D,D, D,D,D,D,D,D,   # 18 CI
        D,D,D,D,D,                                   # G,ms,mt,ml,mj
        D, ct.c_long, ct.c_int,                      # dt, n_steps, store_every
        LP,LP,LP, LP,LP, LP,LP,LP,LP,LP,LP, IP, ct.c_int,  # sampled (11 arrays + count + max)
        LP, IP, ct.c_int,                             # year
        LP, IP, ct.c_int,                             # perigee
        LP, IP, ct.c_int,                             # node
        LP, IP, ct.c_int,                             # sidereal
        LP, IP, ct.c_int,                             # perihelion
    ]

    def alloc(n):
        a = np.zeros(n, dtype=np.float64)
        return a, a.ctypes.data_as(LP)
    def alloc_int():
        a = np.zeros(1, dtype=np.int32)
        return a, a.ctypes.data_as(IP)

    pos_J, vel_J = compute_jupiter_ic()
    mj_eff = 0.0 if args.no_jupiter else M_J
    jupiter_str = "DESACTIVE" if args.no_jupiter else "ACTIVE"

    dt = args.dt
    n_steps = int(args.years * 365.25 * JOUR / dt)
    store_every = args.store_every

    max_samp  = n_steps // store_every + 10
    max_years = int(args.years) + 500
    max_ev    = int(args.years * 365.25 / 27.0) + 1000

    t_s, pt = alloc(max_samp); rST_s, prST = alloc(max_samp)
    rTL_s, prTL = alloc(max_samp)
    cL_s, pcL = alloc(max_samp); sB_s, psB = alloc(max_samp)
    xTs, pxT = alloc(max_samp); yTs, pyT = alloc(max_samp); zTs, pzT = alloc(max_samp)
    xTLs,pxTL= alloc(max_samp); yTLs,pyTL= alloc(max_samp); zTLs,pzTL= alloc(max_samp)
    n_samp, pns = alloc_int()

    yr, pyr = alloc(max_years*4); n_yr, pny = alloc_int()
    pr, ppr = alloc(max_ev*6);    n_pr, pnp = alloc_int()
    nd, pnd = alloc(max_ev*4);    n_nd, pnn = alloc_int()
    sd, psd = alloc(max_ev*2);    n_sd, pnsd = alloc_int()
    ph, pph = alloc(max_years*3); n_ph, pnph = alloc_int()

    print(f"Jupiter : {jupiter_str}")
    print(f"Parametres : dt={dt:.0f} s, duree={args.years:.0f} ans, "
          f"store_every={store_every}")
    print(f"Nombre de pas : {n_steps:,}")
    t0 = time_mod.time()

    lib.simulate(
        D(xT0),D(yT0),D(zT0),D(vxT0),D(vyT0),D(vzT0),
        D(xL0),D(yL0),D(zL0),D(vxL0),D(vyL0),D(vzL0),
        D(pos_J[0]),D(pos_J[1]),D(pos_J[2]),
        D(vel_J[0]),D(vel_J[1]),D(vel_J[2]),
        D(G), D(M_S), D(M_T), D(M_L), D(mj_eff),
        D(dt), ct.c_long(n_steps), ct.c_int(store_every),
        pt, prST, prTL, pcL, psB, pxT, pyT, pzT, pxTL, pyTL, pzTL,
        pns, ct.c_int(max_samp),
        pyr, pny, ct.c_int(max_years),
        ppr, pnp, ct.c_int(max_ev),
        pnd, pnn, ct.c_int(max_ev),
        psd, pnsd, ct.c_int(max_ev),
        pph, pnph, ct.c_int(max_years),
    )
    elapsed = time_mod.time() - t0
    print(f"Duree calcul : {elapsed:.1f} s")

    ns=int(n_samp[0]); ny=int(n_yr[0]); np_=int(n_pr[0])
    nn=int(n_nd[0]); nsd=int(n_sd[0]); nph=int(n_ph[0])

    res = {
        "t_samp": t_s[:ns], "rST_samp": rST_s[:ns], "rTL_samp": rTL_s[:ns],
        "cos_lambda": cL_s[:ns], "sin_beta": sB_s[:ns],
        "xT": xTs[:ns], "yT": yTs[:ns], "zT": zTs[:ns],
        "xTL": xTLs[:ns], "yTL": yTLs[:ns], "zTL": zTLs[:ns],
        "year": yr[:ny*4].reshape(ny,4),
        "perigee": pr[:np_*6].reshape(np_,6),
        "node": nd[:nn*4].reshape(nn,4),
        "sidereal": sd[:nsd*2].reshape(nsd,2),
        "perihelion": ph[:nph*3].reshape(nph,3),
        "dt": dt, "store_every": store_every,
        "jupiter": not args.no_jupiter,
        "years": args.years,
    }

    if ny > 0:
        yd = res["year"][:,1]
        print(f"\nAnnee siderale Terre : {np.mean(yd):.4f} +/- {np.std(yd):.4f} j")
    if np_ > 0:
        print(f"Excentricite Lune    : {np.mean(res['perigee'][:,2]):.4f} "
              f"+/- {np.std(res['perigee'][:,2]):.4f}")
    if nn > 0:
        print(f"Inclinaison Lune     : {np.mean(res['node'][:,1]):.3f} "
              f"+/- {np.std(res['node'][:,1]):.3f} deg")
    if nsd > 0:
        print(f"Periode siderale Lune: {np.mean(res['sidereal'][:,1]):.4f} "
              f"+/- {np.std(res['sidereal'][:,1]):.4f} j")

    return res


# ===================================================================
# ANALYSE FFT
# ===================================================================
def fft_analyse(signal, dt_days, label=""):
    N = len(signal)
    sig = signal - np.mean(signal)
    window = np.hanning(N)
    spectrum = np.abs(np.fft.rfft(sig * window)) * 2.0 / np.sum(window)
    freqs = np.fft.rfftfreq(N, d=dt_days)
    return freqs, spectrum


def find_peaks(freqs, spectrum, threshold_ratio=0.05):
    threshold = threshold_ratio * np.max(spectrum[1:])
    peaks = []
    for i in range(2, len(spectrum)-1):
        if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1] and spectrum[i] > threshold:
            peaks.append((freqs[i], spectrum[i], 1.0/freqs[i] if freqs[i] > 0 else np.inf))
    peaks.sort(key=lambda x: -x[1])
    return peaks


def label_peaks(peaks, known_ineq, max_labels=8):
    labeled = []
    for f, amp, T in peaks[:max_labels]:
        best = min(known_ineq, key=lambda iq: abs(1.0/iq[1] - f))
        if abs(T - best[1]) / best[1] < 0.15:
            labeled.append((f, amp, T, best[0]))
        else:
            labeled.append((f, amp, T, f"T={T:.1f}j"))
    return labeled


# ===================================================================
# GRAPHIQUES
# ===================================================================
# Tailles de police des figures inserees en pleine largeur dans le livre :
# les figures (12 pouces de large) sont reduites d'un facteur ~0.4 par
# \includegraphics[width=\textwidth], d'ou des polices volontairement grandes.
FS_SUPTITLE = 27
FS_TITLE    = 23
FS_LABEL    = 20
FS_TICK     = 18
FS_LEGEND   = 17


# Figures denses (beaucoup de panneaux) : jeu de tailles reduit, sinon les
# titres et annotations se chevauchent.
FS2_SUPTITLE = 27
FS2_TITLE    = 23
FS2_LABEL    = 22
FS2_TICK     = 20
FS2_ANNOT    = 17
FS2_TABLE    = 11   # encadre fig. 4.12 (3x2 : panneau bas, 13 lignes)
FS2_TABLE_LAT = 15  # encadre fig. 4.13 (2x2 : panneau plus haut, 9 lignes)

# Figures tres denses (16 panneaux) : un cran en dessous.
FS3_SUPTITLE = 26
FS3_TITLE    = 21
FS3_LABEL    = 19
FS3_TICK     = 17
FS3_ANNOT    = 16


def _style_axes(ax, title=None, xlabel=None, ylabel=None,
                fs_title=None, fs_label=None, fs_tick=None):
    """Applique les tailles de police communes a un panneau."""
    fs_title = FS_TITLE if fs_title is None else fs_title
    fs_label = FS_LABEL if fs_label is None else fs_label
    fs_tick  = FS_TICK  if fs_tick  is None else fs_tick
    if title is not None:
        ax.set_title(title, fontsize=fs_title)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fs_label)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fs_label)
    ax.tick_params(axis="both", labelsize=fs_tick)
    ax.grid(True, alpha=0.3)


def plot_fig1_terre(res, tag, out_dir, span_years=500.0, span_radius_years=20.0):
    """Figure 1 : parametres orbitaux de la Terre.

    La simulation couvre typiquement 4000 ans, mais l'annee siderale et
    l'excentricite ne sont tracees que sur les `span_years` premieres annees
    (500 par defaut) : sur 4000 ans les courbes sont illisibles.
    Le rayon de l'orbite est trace sur `span_radius_years` annees avec un
    trait epais pour rester lisible.
    """
    YEAR_S = 365.25 * JOUR
    dt_s = res["dt"] * res["store_every"]
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5))
    fig.suptitle(f"Parametres orbitaux de la Terre  ({jupiter_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    # --- Annee siderale : zoom sur les span_years premieres annees ---
    yr = res["year"]
    t_yr = yr[:,0] / YEAR_S                 # date de chaque passage, en annees
    m_yr = t_yr <= span_years
    if not np.any(m_yr):                    # simulation plus courte que span_years
        m_yr = np.ones(len(yr), dtype=bool)
    yr_offset = yr[m_yr,1] - 365.0
    axes[0].plot(t_yr[m_yr], yr_offset, "r-", lw=0.7)
    axes[0].axhline(365.2564 - 365.0, color="b", ls="--", lw=1.2, label="ref 365.256 j")
    axes[0].axhline(np.mean(yr[:,1]) - 365.0, color="g", ls="--", lw=1.2,
                    label=f"moy {np.mean(yr[:,1]):.3f} j (sur {res['years']:.0f} ans)")
    axes[0].set_xlim(0, min(span_years, t_yr.max()))
    _style_axes(axes[0],
                title=f"Annee siderale terrestre ({span_years:.0f} premieres annees)",
                xlabel="annees", ylabel="jours $- 365$")
    # on degage un peu de place en bas pour que la legende ne masque pas la courbe
    y0, y1 = axes[0].get_ylim()
    axes[0].set_ylim(y0 - 0.38*(y1 - y0), y1)
    axes[0].legend(fontsize=FS_LEGEND, loc="lower right", framealpha=0.95)

    # --- Rayon de l'orbite : trait epais pour la lisibilite ---
    n20 = int(span_radius_years*YEAR_S/dt_s)
    axes[1].plot(res["t_samp"][:n20]/YEAR_S, res["rST_samp"][:n20]/1e9, "r-", lw=2.0)
    _style_axes(axes[1],
                title=f"Rayon orbite terrestre ({span_radius_years:.0f} ans)",
                xlabel="annees", ylabel="M km")

    # --- Excentricite : zoom sur les span_years premieres annees ---
    ph = res["perihelion"]
    t_ph = ph[:,0] / YEAR_S
    m_ph = t_ph <= span_years
    if not np.any(m_ph):
        m_ph = np.ones(len(ph), dtype=bool)
    axes[2].plot(t_ph[m_ph], ph[m_ph,2], "r-", lw=0.9)
    axes[2].set_xlim(0, min(span_years, t_ph.max()))
    _style_axes(axes[2],
                title=f"Excentricite terrestre au perihelie ({span_years:.0f} premieres annees)",
                xlabel="annees", ylabel="excentricite")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig1_terre_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_fig2_distance_TL(res, tag, out_dir, span_years=4.0, n_perigees=500):
    """Figure 2 : distance Terre-Lune.

    En haut : les `span_years` premieres annees, en trait epais (l'oscillation
    mensuelle reste ainsi lisible une fois la figure reduite dans le texte).
    En bas : les `n_perigees` premiers perigees seulement (la simulation en
    compte plus de 50 000 sur 4000 ans, indiscernables sur un seul graphe ;
    avec 500 perigees, soit une quarantaine d'annees, l'oscillation
    individuelle reste visible).
    """
    YEAR_S = 365.25 * JOUR
    dt_s = res["dt"] * res["store_every"]
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5))
    fig.suptitle(f"Distance Terre-Lune  ({jupiter_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    n4 = int(span_years*YEAR_S/dt_s)
    axes[0].plot(res["t_samp"][:n4]/YEAR_S, res["rTL_samp"][:n4]/1e3, "r-", lw=1.6)
    _style_axes(axes[0], title=f"{span_years:.0f} premieres annees",
                xlabel="annees", ylabel="km")

    pr = res["perigee"]
    npg = min(n_perigees, len(pr))
    axes[1].plot(np.arange(npg), pr[:npg,1]/1e3, "r-", lw=1.4)
    _style_axes(axes[1],
                title=f"Distance au perigee ({npg} premiers perigees)",
                xlabel="numero de perigee", ylabel="km")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig2_distanceTL_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_fig3_lune(res, tag, out_dir, n_court=250, n_long=1000):
    """Figure 3 : parametres orbitaux de la Lune (zoom sur le debut).

    Le nombre de points traces est inchange (250 perigees / noeuds, 1000 mois
    sideraux) ; seule l'epaisseur des traits et la taille des polices ont ete
    augmentees, la figure etant fortement reduite dans le texte.
    """
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"
    pr = res["perigee"]; nd = res["node"]; sd = res["sidereal"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    fig.suptitle(f"Parametres orbitaux de la Lune  ({jupiter_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    ns = min(n_court, len(pr))
    axes[0].plot(np.arange(ns), pr[:ns,2], "r-", lw=1.8)
    _style_axes(axes[0], title=f"Excentricite au perigee (premiers {ns})", ylabel="e")

    nn = min(n_court, len(nd))
    axes[1].plot(np.arange(nn), nd[:nn,1], "r-", lw=1.8)
    _style_axes(axes[1], title=f"Inclinaison au noeud N1 (premiers {nn})", ylabel="deg")

    nss = min(n_long, len(sd))
    axes[2].plot(np.arange(nss), sd[:nss,1], "r-", lw=1.0)
    axes[2].axhline(27.3217, color="b", ls="--", lw=1.6)
    _style_axes(axes[2], title=f"Periode siderale (premiers {nss} mois)", ylabel="jours")

    nv = min(n_long, len(pr))
    axes[3].plot(np.arange(nv), pr[:nv,3], "r-", lw=1.0)
    _style_axes(axes[3], title=f"Vitesse au perigee (premiers {nv})",
                xlabel="numero d'evenement", ylabel="km/s")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig3_lune_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_fig4_seculaires(res, tag, out_dir, span_ans=100.0):
    """Figure 4 : mouvements seculaires lunaires (noeuds, perigee) et leur FFT.

    Colonne de gauche : les `span_ans` premieres annees seulement. Sur 4000 ans
    le trace comptait 1000 points pour 215 cycles de 18,6 ans, illisibles ; sur
    100 ans on lit les cinq oscillations de la ligne des noeuds et les onze du
    perigee. Les FFT (colonne de droite) portent toujours sur la totalite.
    """
    YEAR_S = 365.25 * JOUR
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"
    nd = res["node"]; pr = res["perigee"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Mouvements seculaires lunaires + FFT  ({jupiter_str})",
                 fontsize=FS_SUPTITLE, fontweight="bold")

    t_nd = nd[:,0] / YEAR_S
    m_nd = t_nd <= span_ans
    axes[0,0].plot(t_nd[m_nd], nd[m_nd,2], "r-", lw=1.4)
    axes[0,0].set_xlim(0, span_ans)
    _style_axes(axes[0,0], title=f"Longitude noeud ascendant ({span_ans:.0f} ans)",
                xlabel="annees", ylabel="sin(Omega)")

    dt_node_yr = np.mean(np.diff(nd[:,0])) / YEAR_S if len(nd)>1 else 1.0
    f_om, sp_om = fft_analyse(nd[:,2], dt_node_yr)
    axes[0,1].plot(f_om, sp_om, "r-", lw=1.2)
    axes[0,1].set_xlim(0.03, 0.08)
    _style_axes(axes[0,1], title="FFT sin(Omega)", xlabel="annee$^{-1}$")
    pk = find_peaks(f_om, sp_om)
    if pk:
        fp, ap, Tp = pk[0][:3]
        axes[0,1].annotate(f"T = {Tp:.2f} ans", xy=(fp, ap), fontsize=FS_LEGEND, color="blue",
                           xytext=(fp+0.005, ap*0.8), arrowprops=dict(arrowstyle="->", color="blue"))

    t_pr = pr[:,0] / YEAR_S
    m_pr = t_pr <= span_ans
    axes[1,0].plot(t_pr[m_pr], pr[m_pr,4], "r-", lw=1.4)
    axes[1,0].set_xlim(0, span_ans)
    _style_axes(axes[1,0], title=f"Longitude perigee ({span_ans:.0f} ans)",
                xlabel="annees", ylabel="cos(Xi)")

    dt_peri_yr = np.mean(np.diff(pr[:,0])) / YEAR_S if len(pr)>1 else 1.0
    f_xi, sp_xi = fft_analyse(pr[:,4], dt_peri_yr)
    axes[1,1].plot(f_xi, sp_xi, "r-", lw=1.2)
    axes[1,1].set_xlim(0.08, 0.16)
    _style_axes(axes[1,1], title="FFT cos(Xi)", xlabel="annee$^{-1}$")
    pk2 = find_peaks(f_xi, sp_xi)
    if pk2:
        fp2, ap2, Tp2 = pk2[0][:3]
        axes[1,1].annotate(f"T = {Tp2:.2f} ans", xy=(fp2, ap2), fontsize=FS_LEGEND, color="blue",
                           xytext=(fp2+0.005, ap2*0.8), arrowprops=dict(arrowstyle="->", color="blue"))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig4_seculaires_{tag}.png"), dpi=150)
    plt.close(fig)


def _enveloppe_et_moyenne(t_yr, y, fenetre=50.0):
    """Min, max et moyenne de y par fenetres de `fenetre` annees."""
    bords = np.arange(t_yr.min(), t_yr.max() + fenetre, fenetre)
    idx = np.clip(np.digitize(t_yr, bords) - 1, 0, len(bords)-2)
    nb = len(bords) - 1
    tc = 0.5*(bords[:-1] + bords[1:])
    ymin = np.full(nb, np.nan); ymax = np.full(nb, np.nan); ymoy = np.full(nb, np.nan)
    for k in range(nb):
        sel = (idx == k)
        if np.any(sel):
            ymin[k] = y[sel].min(); ymax[k] = y[sel].max(); ymoy[k] = y[sel].mean()
    ok = ~np.isnan(ymoy)
    return tc[ok], ymin[ok], ymax[ok], ymoy[ok]


def plot_fig5_ensemble(res, tag, out_dir, fenetre_env=50.0, fenetre_moy=200.0):
    """Figure 5 : recherche d'une derive seculaire sur les parametres lunaires.

    Chaque panneau montre l'enveloppe min/max (amplitude physique de
    l'oscillation orbitale, contexte, bords nets) et la moyenne glissante
    (le seul signal pertinent pour la question posee : y a-t-il une derive
    sur 4000 ans ?). Comme la moyenne est ecrasee a l'echelle de l'enveloppe,
    un encart zoome sur sa propre plage de variation avec une ligne de
    reference : une derive, si elle existe, y saute aux yeux.
    """
    YEAR_S = 365.25 * JOUR
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"
    pr = res["perigee"]; nd = res["node"]; sd = res["sidereal"]

    panneaux = [
        (pr[:,0]/YEAR_S, pr[:,2], "Excentricité lunaire au périgée", "$e_L$"),
        (nd[:,0]/YEAR_S, nd[:,1], "Inclinaison au nœud ascendant",   "deg"),
        (sd[:,0]/YEAR_S, sd[:,1], "Période sidérale",                "jours"),
        (pr[:,0]/YEAR_S, pr[:,3], "Vitesse au périgée",              "km/s"),
    ]

    # Cette figure est imprimee reduite (0,9 textwidth pour 14 pouces de large) :
    # on grossit les polices de 5 pt par rapport aux autres figures.
    up = 5
    fig, axes = plt.subplots(4, 1, figsize=(14, 15))
    fig.suptitle(f"Dérive séculaire des paramètres lunaires  ({jupiter_str})",
                 fontsize=FS_SUPTITLE - 2 + up, fontweight="bold")

    for i, (ax, (t_yr, y, titre, ylab)) in enumerate(zip(axes, panneaux)):
        tc_e, ymin, ymax, _ = _enveloppe_et_moyenne(t_yr, y, fenetre_env)
        tc_m, _, _, ymoy = _enveloppe_et_moyenne(t_yr, y, fenetre_moy)

        ax.fill_between(tc_e, ymin, ymax, color="#c3c3ff", alpha=0.35, linewidth=0,
                        label="enveloppe min/max")
        ax.plot(tc_e, ymin, color="#7a7ae0", lw=0.9)
        ax.plot(tc_e, ymax, color="#7a7ae0", lw=0.9)
        ax.plot(tc_m, ymoy, color="#1f1f8f", lw=2.2,
                label=f"moyenne glissante ({fenetre_moy:.0f} ans)")
        _style_axes(ax, title=titre, ylabel=ylab, fs_title=FS_TITLE + up,
                    fs_label=FS_LABEL + up, fs_tick=FS_TICK + up)
        ax.set_xlim(t_yr.min(), t_yr.max())
        if i == 0:
            ax.legend(fontsize=FS_LEGEND + up, loc="upper left", framealpha=0.95)

        # --- encart : zoom sur l'ecart de la moyenne glissante a sa reference ---
        # On trace l'ecart (ymoy - ref), pas la valeur absolue : les valeurs
        # affichees restent de petits nombres lisibles, sans recourir a la
        # notation scientifique decalee de matplotlib (ex. "1e-5 + 1.084"),
        # illisible pour un lecteur non averti.
        axins = ax.inset_axes([0.62, 0.06, 0.36, 0.40])
        ref = ymoy.mean()
        ecart = ymoy - ref
        marge = max(3.0 * ecart.std(), 1e-12)
        axins.axhline(0.0, color="#999999", lw=1.0, ls="--")
        axins.plot(tc_m, ecart, color="#1f1f8f", lw=1.4)
        axins.set_xlim(t_yr.min(), t_yr.max())
        axins.set_ylim(-marge, marge)
        axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
        axins.set_title(f"écart à la moyenne ({fenetre_moy:.0f} ans)", fontsize=FS_LEGEND - 4 + up,
                        backgroundcolor="white")
        # memes abscisses que l'axe principal : on ne les repete pas dans l'encart
        axins.tick_params(labelsize=FS_LEGEND - 6 + up, labelbottom=False)
        axins.yaxis.get_offset_text().set_fontsize(FS_LEGEND - 6 + up)
        axins.grid(True, alpha=0.3)

    axes[-1].set_xlabel("années", fontsize=FS_LABEL + up)
    # rect : on reserve une bande en haut pour que le titre general ne
    # chevauche pas le titre du premier panneau
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(out_dir, f"fig5_ensemble_{tag}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inegalites lunaires : FFT de la longitude "residuelle" et de la latitude
# ---------------------------------------------------------------------------
# On n'analyse PAS cos(lambda) : comme lambda = n t + sum A_k sin(theta_k),
# cos(lambda) est une porteuse a la periode siderale modulee en phase, et ses
# pics tombent aux frequences n +/- theta_k (bandes laterales), pas aux
# theta_k. On analyse donc le residu  d_lambda = lambda - (lambda0 + n t),
# lambda etant deroule (np.unwrap), dont le spectre donne directement les
# arguments theta_k et les amplitudes A_k (en secondes d'arc).
# Pour la latitude, beta est petit : beta = sum B_k sin(theta_k) directement.

# Vitesses angulaires moyennes des arguments de Delaunay (deg/jour)
_V_D, _V_LP, _V_L, _V_F = 12.190749, 0.985600, 13.064993, 13.229350

# (nom affiche, (cD, cl', cl, cF), amplitude de reference en secondes d'arc)
# Amplitudes de reference : theorie ELP-2000/82 tronquee (Meeus, Astronomical
# Algorithms, tables 47.A et 47.B), en valeur absolue. Les termes l-2F et
# 2l-F, noyes dans la fuite spectrale des pics a 27,55 et 27,21 j, sont omis.
INEQ_LONGITUDE_ENRICHIE = [
    ("Eq. du centre",        (0, 0, 1, 0), 22640),
    ("Evection",             (2, 0,-1, 0),  4586),
    ("Variation",            (2, 0, 0, 0),  2370),
    ("Eq. centre 2l",        (0, 0, 2, 0),   769),
    ("Eq. annuelle",         (0, 1, 0, 0),   666),
    ("Red. ecliptique",      (0, 0, 0, 2),   412),
    ("2D-l'-l",              (2,-1,-1, 0),   205),
    ("2D-2l",                (2, 0,-2, 0),   212),
    ("2D+l",                 (2, 0, 1, 0),   192),
    ("2D-l'",                (2,-1, 0, 0),   165),
    ("l-l'",                 (0,-1, 1, 0),   147),
    ("Parallactique D",      (1, 0, 0, 0),   125),
    ("l+l'",                 (0, 1, 1, 0),   109),
    ("2D-2F",                (2, 0, 0,-2),    55),
    ("l+2F",                 (0, 0, 1, 2),    45),
    ("4D-l",                 (4, 0,-1, 0),    38),
    ("3l",                   (0, 0, 3, 0),    36),
    ("4D-2l",                (4, 0,-2, 0),    31),
    ("2D+l'",                (2, 1, 0, 0),    24),
]

INEQ_LATITUDE_ENRICHIE = [
    ("Terme principal F",    (0, 0, 0, 1), 18461),
    ("l+F",                  (0, 0, 1, 1),  1010),
    ("l-F",                  (0, 0, 1,-1),  1000),
    ("Grande ineg. 2D-F",    (2, 0, 0,-1),   624),
    ("2D-l+F",               (2, 0,-1, 1),   199),
    ("2D-l-F",               (2, 0,-1,-1),   167),
    ("2D+F",                 (2, 0, 0, 1),   117),
    ("2l+F",                 (0, 0, 2, 1),    62),
    ("2D+l-F",               (2, 0, 1,-1),    33),
    ("2D-l'-F",              (2,-1, 0,-1),    30),
]


def _freq_argument(coefs):
    """Frequence (jour^-1) d'un argument cD*D + cl'*l' + cl*l + cF*F."""
    cD, cLp, cL, cF = coefs
    v = cD*_V_D + cLp*_V_LP + cL*_V_L + cF*_V_F
    return abs(v) / 360.0


def _fft_fine(signal, dt_days, pad=8):
    """Spectre d'amplitude (fenetre de Hann, zero-padding x pad) : le
    zero-padding interpole le spectre et limite la sous-estimation des
    amplitudes quand un pic tombe entre deux raies."""
    N = len(signal)
    sig = signal - np.mean(signal)
    w = np.hanning(N)
    M = pad * N
    sp = np.abs(np.fft.rfft(sig * w, n=M)) * 2.0 / np.sum(w)
    f = np.fft.rfftfreq(M, d=dt_days)
    return f, sp


def mesure_inegalites(freqs, spectrum, liste, T_total_days, k_res=1.0):
    """Pour chaque argument theorique, cherche le maximum local du spectre
    a moins de k_res resolution(s) spectrale(s) (1/T_total) de la frequence
    theorique. Renvoie (nom, T_mesure, f_mesure, amplitude, T_theo, A_ref)
    trie par amplitude decroissante ; les arguments sans pic sont omis."""
    df = 1.0 / T_total_days
    res = []
    for nom, coefs, A_ref in liste:
        f_th = _freq_argument(coefs)
        m = np.where(np.abs(freqs - f_th) <= k_res * df)[0]
        if len(m) < 3:
            continue
        j = m[np.argmax(spectrum[m])]
        if j in (m[0], m[-1]):      # pas de maximum local dans la fenetre
            continue
        res.append((nom, 1.0/freqs[j], freqs[j], spectrum[j], 1.0/f_th, A_ref))
    res.sort(key=lambda r: -r[3])
    return res


def _tableau_inegalites(mesures, titre, nmax=12):
    """Texte monospace du resume des inegalites."""
    txt = titre + "\n"
    txt += f"{'Argument':<18}{'T th.':>8}{'T mes.':>8}{'A mes.':>8}{'A ELP':>7}\n"
    txt += "-" * 49 + "\n"
    for nom, T, f, a, T_th, A_ref in [m[:6] for m in mesures[:nmax]]:
        txt += f"{nom:<18}{T_th:>8.2f}{T:>8.2f}{a:>7.0f}\"{A_ref:>6}\"\n"
    return txt


def _longitude_residuelle(res, n):
    """lambda (deroulee) moins sa tendance lineaire, en secondes d'arc."""
    lam = np.unwrap(np.arctan2(res["yTL"][:n], res["xTL"][:n]))
    t = res["t_samp"][:n] / JOUR
    p = np.polyfit(t, lam, 1)
    return np.degrees(lam - np.polyval(p, t)) * 3600.0, np.degrees(p[0])


def _latitude_arcsec(res, n):
    x, y, z = res["xTL"][:n], res["yTL"][:n], res["zTL"][:n]
    return np.degrees(np.arcsin(z / np.sqrt(x*x + y*y + z*z))) * 3600.0


def _arg_court(nom, liste):
    """Argument sous forme courte ("2D-l", "l+F"...) a partir du nom."""
    for n, coefs, _ in liste:
        if n == nom:
            parts = []
            for c, sym in zip(coefs, ("D", "l'", "l", "F")):
                if c == 0:
                    continue
                sgn = "-" if c < 0 else ("+" if parts else "")
                parts.append(f"{sgn}{abs(c) if abs(c) > 1 else ''}{sym}")
            return "".join(parts)
    return nom


def _annoter_reparti(ax, pics, fmin, fmax, y_lab, labels, log=False):
    """Etiquettes reparties regulierement en abscisse au-dessus des pics,
    reliees a ceux-ci par une fleche : evite tout chevauchement."""
    pics = sorted(pics, key=lambda m: m[2])
    n = len(pics)
    for k, (m, lab) in enumerate(zip(pics, labels)):
        f, a = m[2], m[3]
        xt = fmin + (k + 0.5) * (fmax - fmin) / n
        yt = y_lab * (3.0 if (log and k % 2) else 1.0) if log else y_lab * (1.0 + 0.18*(k % 2))
        ax.annotate(lab, xy=(f, a), xytext=(xt, yt), fontsize=FS2_ANNOT-1,
                    color="blue", ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="->", color="blue", lw=0.9))


def _annoter_zoom(ax, mes, fmin, fmax, ymax, nmax=3):
    dans_bande = sorted([m for m in mes if fmin <= m[2] <= fmax], key=lambda m: -m[3])[:nmax]
    dans_bande = sorted(dans_bande, key=lambda m: m[2])
    labels = [f"{m[0]}\nT={m[1]:.2f} j" for m in dans_bande]
    _annoter_reparti(ax, dans_bande, fmin, fmax, 1.12*ymax, labels)


def plot_fig6_fft_longitude(res, tag, out_dir, duree_ans=19.0, duree_trace=2.0):
    """Figure 6 : inegalites en longitude ecliptique lunaire.

    FFT du residu d_lambda = lambda - n t (secondes d'arc) sur `duree_ans`
    annees ; spectre complet en echelle log puis trois zooms (22-40 j,
    12-17 j, 8-11 j). Le signal n'est trace que sur `duree_trace` annees.
    """
    YEAR_S = 365.25 * JOUR
    dt_s = res["dt"] * res["store_every"]
    dt_days = dt_s / JOUR
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"

    n19 = min(int(duree_ans * YEAR_S / dt_s), len(res["xTL"]))
    dlam, n_moy = _longitude_residuelle(res, n19)
    t19 = res["t_samp"][:n19]
    T_tot = n19 * dt_days

    f_lon, sp_lon = _fft_fine(dlam, dt_days)
    mes = mesure_inegalites(f_lon, sp_lon, INEQ_LONGITUDE_ENRICHIE, T_tot)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Inegalites en longitude ecliptique lunaire  ({jupiter_str})",
                 fontsize=FS2_SUPTITLE, fontweight="bold")

    n_tr = min(int(duree_trace * YEAR_S / dt_s), n19)
    axes[0,0].plot(t19[:n_tr]/JOUR, dlam[:n_tr]/3600.0, "r-", lw=1.4)
    _style_axes(axes[0,0], fs_title=FS2_TITLE, fs_label=FS2_LABEL, fs_tick=FS2_TICK,
                title=f"$\\lambda - \\bar n t$ : {duree_trace:.0f} premieres annees",
                xlabel="jours", ylabel="degres")

    sp_log = np.where(sp_lon > 0, sp_lon, 1e-10)
    axes[0,1].semilogy(f_lon, sp_log, "r-", lw=0.6)
    axes[0,1].set_xlim(0, 0.12); axes[0,1].set_ylim(5, 3e6)
    _style_axes(axes[0,1], fs_title=FS2_TITLE, fs_label=FS2_LABEL, fs_tick=FS2_TICK,
                title=f"FFT sur {T_tot/365.25:.0f} ans (echelle log)",
                xlabel="jour$^{-1}$", ylabel="amplitude ('')")
    gros = [m for m in mes[:6] if 0.002 < m[2] < 0.12]
    _annoter_reparti(axes[0,1], gros, 0.0, 0.12, 8e4,
                     [_arg_court(m[0], INEQ_LONGITUDE_ENRICHIE) for m in sorted(gros, key=lambda m: m[2])],
                     log=True)

    zooms = [
        (axes[1,0], 22.0, 40.0, "Zoom : 22-40 j"),
        (axes[1,1], 12.0, 17.0, "Zoom : 12-17 j"),
        (axes[2,0],  8.0, 11.0, "Zoom : 8-11 j"),
    ]
    for ax, Tmin, Tmax, titre in zooms:
        fmin, fmax = 1.0/Tmax, 1.0/Tmin
        m = (f_lon >= fmin) & (f_lon <= fmax)
        ax.plot(f_lon[m], sp_lon[m], "r-", lw=1.4)
        ax.set_xlim(fmin, fmax)
        ymax = np.max(sp_lon[m]) if np.any(m) else 1.0
        ax.set_ylim(0, ymax*1.85)
        _style_axes(ax, title=titre, xlabel="jour$^{-1}$", ylabel="('')",
                    fs_title=FS2_TITLE, fs_label=FS2_LABEL, fs_tick=FS2_TICK)
        _annoter_zoom(ax, mes, fmin, fmax, ymax)

    info = _tableau_inegalites(
        mes, f"N={n19}, dt={dt_days:.2f} j, Nyquist={1/(2*dt_days):.2f} j-1, "
             f"df={1/T_tot:.1e} j-1", nmax=11)
    axes[2,1].set_facecolor("lightyellow")
    axes[2,1].text(0.02, 0.98, info, transform=axes[2,1].transAxes, fontsize=FS2_TABLE-1,
                   linespacing=1.1, verticalalignment="top", fontfamily="monospace")
    axes[2,1].set_title("Principaux termes ('')", fontsize=FS2_TITLE)
    axes[2,1].set_xticks([]); axes[2,1].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(out_dir, f"fig6_fft_longitude_{tag}.png"), dpi=150)
    plt.close(fig)
    return mes, n_moy


def plot_fig7_fft_latitude(res, tag, out_dir, duree_ans=19.0, duree_trace=2.0):
    """Figure 7 : inegalites en latitude ecliptique lunaire (FFT de beta en '')."""
    YEAR_S = 365.25 * JOUR
    dt_s = res["dt"] * res["store_every"]
    dt_days = dt_s / JOUR
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"

    n19 = min(int(duree_ans * YEAR_S / dt_s), len(res["xTL"]))
    beta = _latitude_arcsec(res, n19)
    t19 = res["t_samp"][:n19]
    T_tot = n19 * dt_days

    f_lat, sp_lat = _fft_fine(beta, dt_days)
    mes = mesure_inegalites(f_lat, sp_lat, INEQ_LATITUDE_ENRICHIE, T_tot)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Inegalites en latitude ecliptique lunaire  ({jupiter_str})",
                 fontsize=FS2_SUPTITLE, fontweight="bold")

    n_tr = min(int(duree_trace * YEAR_S / dt_s), n19)
    axes[0,0].plot(t19[:n_tr]/JOUR, beta[:n_tr]/3600.0, "r-", lw=1.4)
    _style_axes(axes[0,0], fs_title=FS2_TITLE, fs_label=FS2_LABEL, fs_tick=FS2_TICK,
                title=f"$\\beta$ : {duree_trace:.0f} premieres annees",
                xlabel="jours", ylabel="degres")

    sp_log = np.where(sp_lat > 0, sp_lat, 1e-10)
    axes[0,1].semilogy(f_lat, sp_log, "r-", lw=0.6)
    axes[0,1].set_xlim(0, 0.12); axes[0,1].set_ylim(5, 3e6)
    _style_axes(axes[0,1], fs_title=FS2_TITLE, fs_label=FS2_LABEL, fs_tick=FS2_TICK,
                title=f"FFT sur {T_tot/365.25:.0f} ans (echelle log)",
                xlabel="jour$^{-1}$", ylabel="amplitude ('')")
    gros = [m for m in mes[:5] if 0.002 < m[2] < 0.12]
    _annoter_reparti(axes[0,1], gros, 0.0, 0.12, 8e4,
                     [_arg_court(m[0], INEQ_LATITUDE_ENRICHIE) for m in sorted(gros, key=lambda m: m[2])],
                     log=True)

    fmin, fmax = 1.0/45.0, 1.0/22.0
    m = (f_lat >= fmin) & (f_lat <= fmax)
    ymax = np.max(sp_lat[m]) if np.any(m) else 1.0
    axes[1,0].semilogy(f_lat[m], np.where(sp_lat[m] > 0, sp_lat[m], 1e-10), "r-", lw=1.4)
    axes[1,0].set_xlim(fmin, fmax); axes[1,0].set_ylim(5, ymax*300)
    _style_axes(axes[1,0], fs_title=FS2_TITLE, fs_label=FS2_LABEL, fs_tick=FS2_TICK,
                title="Zoom 22-45 j (echelle log)", xlabel="jour$^{-1}$", ylabel="('')")
    zb = sorted([mm for mm in mes if fmin <= mm[2] <= fmax][:3], key=lambda m: m[2])
    _annoter_reparti(axes[1,0], zb, fmin, fmax, ymax*3,
                     [f"{_arg_court(m[0], INEQ_LATITUDE_ENRICHIE)}\nT={m[1]:.2f} j" for m in zb], log=True)

    info = _tableau_inegalites(mes, "Pics identifies", nmax=11)
    axes[1,1].set_facecolor("lightyellow")
    axes[1,1].text(0.02, 0.98, info, transform=axes[1,1].transAxes, fontsize=FS2_TABLE_LAT-2,
                   linespacing=1.2, verticalalignment="top", fontfamily="monospace")
    axes[1,1].set_title("Inegalites en latitude ('')", fontsize=FS2_TITLE)
    axes[1,1].set_xticks([]); axes[1,1].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, f"fig7_fft_latitude_{tag}.png"), dpi=150)
    plt.close(fig)
    return mes


def generate_plots(res, tag, out_dir):
    YEAR_S = 365.25 * JOUR
    dt_s = res["dt"] * res["store_every"]
    dt_days = dt_s / JOUR
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"

    # --- Fig 1 : Terre ---
    plot_fig1_terre(res, tag, out_dir)

    # --- Fig 2 : Distance TL ---
    plot_fig2_distance_TL(res, tag, out_dir)
    pr = res["perigee"]

    # --- Fig 3 : Params lunaires (zoom) ---
    plot_fig3_lune(res, tag, out_dir)
    nd = res["node"]; sd = res["sidereal"]

    # --- Fig 4 : Mouvements seculaires + FFT ---
    plot_fig4_seculaires(res, tag, out_dir)

    # --- Fig 5 : Vue d'ensemble 4000 ans ---
    plot_fig5_ensemble(res, tag, out_dir)

    # --- Fig 6 : FFT longitude ecliptique lunaire (version enrichie) ---
    plot_fig6_fft_longitude(res, tag, out_dir)

    # --- Fig 7 : FFT latitude ecliptique lunaire ---
    plot_fig7_fft_latitude(res, tag, out_dir)

    print(f"Figures sauvegardees dans {out_dir}/")


# ===================================================================
# EPHEMERIDES : comparaison avec astropy
# ===================================================================
def compute_ephemeris_from_sim(res):
    """Calcule RA et dec du Soleil et de la Lune a partir des positions simulees."""
    ce = np.cos(EPSILON); se = np.sin(EPSILON)
    xT = res["xT"]; yT = res["yT"]; zT = res["zT"]
    xTL = res["xTL"]; yTL = res["yTL"]; zTL = res["zTL"]
    rST = res["rST_samp"]; rTL = res["rTL_samp"]

    # Soleil geocentrique equatorial
    xs = -xT;  ys = -ce*yT + se*zT;  zs = -se*yT - ce*zT
    sin_dec_S = zs / rST
    sin_ra_S  = ys / np.sqrt(xs**2 + ys**2)

    # Lune geocentrique equatorial
    xLe = xTL;  yLe = ce*yTL - se*zTL;  zLe = se*yTL + ce*zTL
    sin_dec_L = zLe / rTL
    sin_ra_L  = yLe / np.sqrt(xLe**2 + yLe**2)

    return sin_ra_S, sin_dec_S, sin_ra_L, sin_dec_L


def _patch_astropy_numpy_compat():
    """Compatibilite astropy 5.1 / numpy >= 1.24.

    Depuis numpy 1.24, np.stack transmet les arguments `dtype` et `casting` a
    np.concatenate ; le helper Quantity d'astropy 5.1 ne les connait pas et
    leve  TypeError: concatenate() got an unexpected keyword argument 'dtype'.
    On enveloppe le helper pour absorber ces deux arguments quand ils sont a
    leur valeur par defaut (cas de np.stack).
    """
    try:
        from astropy.units.quantity_helper import function_helpers as fh
    except Exception:
        return
    helper = fh.FUNCTION_HELPERS.get(np.concatenate)
    if helper is None or getattr(helper, "_patch_dtype", False):
        return

    def concatenate_helper(arrays, axis=0, out=None, dtype=None, casting="same_kind"):
        if dtype is not None:
            raise TypeError("concatenate : argument dtype non supporte")
        return helper(arrays, axis=axis, out=out)

    concatenate_helper._patch_dtype = True
    fh.FUNCTION_HELPERS[np.concatenate] = concatenate_helper


def get_astropy_reference(t_days, t0_jd=2453372.5417):
    """Ephemerides de reference via astropy pour les jours donnes (depuis t0)."""
    _patch_astropy_numpy_compat()
    from astropy.time import Time
    from astropy.coordinates import get_body, GeocentricTrueEcliptic
    import astropy.units as u

    jds = t0_jd + t_days
    times = Time(jds, format="jd", scale="utc")

    sun = get_body("sun", times)
    moon = get_body("moon", times)

    sin_ra_S  = np.sin(sun.ra.rad)
    sin_dec_S = np.sin(sun.dec.rad)
    sin_ra_L  = np.sin(moon.ra.rad)
    sin_dec_L = np.sin(moon.dec.rad)

    return sin_ra_S, sin_dec_S, sin_ra_L, sin_dec_L


def generate_ephemeris_plots(res, tag, out_dir):
    """Compare la simulation aux ephemerides astropy (2005-2006, 10 ans, 2025)."""
    YEAR_S = 365.25 * JOUR
    dt_s = res["dt"] * res["store_every"]
    t_samp = res["t_samp"]
    t_days = t_samp / JOUR
    jupiter_str = "avec Jupiter" if res["jupiter"] else "sans Jupiter"

    sin_ra_S, sin_dec_S, sin_ra_L, sin_dec_L = compute_ephemeris_from_sim(res)

    # --- Fenetre 1 : 2005-2006 (730 jours) ---
    mask1 = t_days < 730
    # --- Fenetre 2 : 2005-2015 (10 ans) ---
    mask10 = t_days < 10.0 * 365.25
    # --- Fenetre 3 : 2025 (20 ans apres, 365 jours) ---
    t_2025 = 20.0 * 365.25  # jours depuis 2 jan 2005
    mask2 = (t_days > t_2025) & (t_days < t_2025 + 365)

    try:
        print("  Calcul ephemerides astropy (2005-2006)...")
        ref1 = get_astropy_reference(t_days[mask1])
        print("  Calcul ephemerides astropy (10 ans)...")
        ref10 = get_astropy_reference(t_days[mask10])
        print("  Calcul ephemerides astropy (2025)...")
        ref2 = get_astropy_reference(t_days[mask2])
        has_ref = True
    except Exception as e:
        print(f"  Astropy non disponible : {e}")
        has_ref = False

    windows = [
        ("2005-2006", "2005-2006", mask1, ref1 if has_ref else None, 1.4),
        ("10ans", "2005-2015 (10 ans)", mask10, ref10 if has_ref else None, 1.0),
        ("2025", "2025", mask2, ref2 if has_ref else None, 1.4),
    ]

    for label, titre, mask, ref, lw in windows:
        td = t_days[mask]
        up = 5   # figure imprimee reduite : polices grossies (cf. plot_fig5_ensemble)
        fig, axes = plt.subplots(4, 2, figsize=(14, 15))
        fig.suptitle(f"Ephemerides {titre}  ({jupiter_str})",
                     fontsize=FS_SUPTITLE + up, fontweight="bold")

        signals = [
            (sin_ra_S[mask], "sin(RA Soleil)", "r"),
            (sin_dec_S[mask], "sin(dec Soleil)", "r"),
            (sin_ra_L[mask], "sin(RA Lune)", "r"),
            (sin_dec_L[mask], "sin(dec Lune)", "r"),
        ]
        ref_signals = ref if ref is not None else [None]*4

        for i, ((sig, title, col), ref_sig) in enumerate(zip(signals, ref_signals)):
            ax_left = axes[i, 0]
            ax_right = axes[i, 1]

            ax_left.plot(td, sig, "b-", lw=lw, label="simulation")
            if ref_sig is not None:
                ax_left.plot(td, ref_sig, "r--", lw=lw, label="astropy", alpha=0.7)
            _style_axes(ax_left, title=title, ylabel=title, fs_title=FS_TITLE + up,
                        fs_label=FS_LABEL + up, fs_tick=FS_TICK + up)
            if i == 0: ax_left.legend(fontsize=FS_LEGEND + up)

            if ref_sig is not None:
                diff = sig - ref_sig
                ax_right.plot(td, diff, "g-", lw=lw)
                _style_axes(ax_right,
                            title=f"Ecart (max = {np.max(np.abs(diff)):.2e})",
                            ylabel="difference", fs_title=FS_TITLE + up,
                            fs_label=FS_LABEL + up, fs_tick=FS_TICK + up)
                ax_right.yaxis.get_offset_text().set_fontsize(FS_TICK + up)
            else:
                ax_right.text(0.5, 0.5, "Pas de reference", transform=ax_right.transAxes,
                             ha="center", fontsize=FS_TITLE)
                ax_right.axis("off")

        axes[-1, 0].set_xlabel("jours depuis 2 jan 2005", fontsize=FS_LABEL + up)
        axes[-1, 1].set_xlabel("jours depuis 2 jan 2005", fontsize=FS_LABEL + up)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig8_ephemerides_{label}_{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  -> fig8_ephemerides_{label}_{tag}.png")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    args = parse_args()

    if args.tag:
        tag = args.tag
    else:
        tag = "sansJupiter" if args.no_jupiter else "avecJupiter"

    results = run_simulation(args)

    out_dir = os.path.join(HERE, "resultats")
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f"simulation_data_{tag}.npz")
    np.savez_compressed(npz_path, **{k: v for k, v in results.items()
                                     if isinstance(v, np.ndarray)},
                        jupiter=results["jupiter"], dt=results["dt"],
                        store_every=results["store_every"], years=results["years"])
    print(f"\nDonnees -> {npz_path}")

    print("Generation des graphiques...")
    generate_plots(results, tag, out_dir)

    print("\nComparaison ephemerides...")
    generate_ephemeris_plots(results, tag, out_dir)

    print("\nTermine.")
