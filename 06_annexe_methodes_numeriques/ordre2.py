#!/usr/bin/env python3
"""
Resolution numerique de l'equation differentielle y'' + 3y = 0
Conditions initiales : y(0) = 1, y'(0) = -1
Methodes : Euler explicite, Euler implicite, Euler symplectique,
           Leap-Frog (schema centre a deux pas), Runge-Kutta 4

Le systeme d'ordre 1 est  z1' = z2,  z2' = -omega2 * z1  (z1 = y, z2 = y').
Les schemas sont ecrits exactement comme dans l'annexe du livre.

Figures produites (dans figures/) :
    ordre2-0.05.png     solutions y(t), dt = 0.05
    energie-ordre2.png  energie E(t), dt = 0.05
    energieC.png        energie E(t) en echelle log + zoom, dt = 0.1
    portraitP.png       portrait de phase (z1, z2), dt = 0.1
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "figures")
os.makedirs(IMG, exist_ok=True)

y0 = 1.0            # y(0)
v0 = -1.0           # y'(0)
omega2 = 3.0        # y'' + omega2 * y = 0
omega = np.sqrt(omega2)
E0 = 0.5 * v0**2 + 0.5 * omega2 * y0**2      # = 2


def exact(t):
    return y0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)


def exact_v(t):
    return -y0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)


# ============================================================================
# SCHEMAS  (chaque fonction renvoie z1 = y et z2 = y')
# ============================================================================
def euler_explicite(t, dt):
    """
    z1(n+1) = z1(n) + dt * z2(n)
    z2(n+1) = z2(n) - omega2 * dt * z1(n)
    """
    n = len(t)
    z1, z2 = np.zeros(n), np.zeros(n)
    z1[0], z2[0] = y0, v0
    for i in range(n - 1):
        z1[i + 1] = z1[i] + dt * z2[i]
        z2[i + 1] = z2[i] - omega2 * dt * z1[i]
    return z1, z2


def euler_implicite(t, dt):
    """
    z1(n+1) = z1(n) + dt * z2(n+1)
    z2(n+1) = z2(n) - omega2 * dt * z1(n+1)
    Apres substitution :
    z2(n+1) = (z2(n) - omega2 * dt * z1(n)) / (1 + omega2 * dt^2)
    """
    n = len(t)
    z1, z2 = np.zeros(n), np.zeros(n)
    z1[0], z2[0] = y0, v0
    factor = 1.0 / (1.0 + omega2 * dt**2)
    for i in range(n - 1):
        z2[i + 1] = (z2[i] - omega2 * dt * z1[i]) * factor
        z1[i + 1] = z1[i] + dt * z2[i + 1]
    return z1, z2


def euler_symplectique(t, dt):
    """
    z1(n+1) = z1(n) + dt * z2(n)            (ancienne vitesse)
    z2(n+1) = z2(n) - omega2 * dt * z1(n+1) (nouvelle position)
    Meme cout qu'Euler explicite : seul l'ordre des operations change.
    C'est le schema du noyau C du chapitre Terre-Lune-Soleil.
    """
    n = len(t)
    z1, z2 = np.zeros(n), np.zeros(n)
    z1[0], z2[0] = y0, v0
    for i in range(n - 1):
        z1[i + 1] = z1[i] + dt * z2[i]
        z2[i + 1] = z2[i] - omega2 * dt * z1[i + 1]
    return z1, z2


def leap_frog(t, dt):
    """
    Schema centre a deux pas (leapfrog au sens strict), tel qu'ecrit dans l'annexe :
    z1(n+1) = z1(n-1) + 2 dt * z2(n)
    z2(n+1) = z2(n-1) - 2 omega2 dt * z1(n)
    Le premier pas est fait par Euler explicite.
    Stable pour omega * dt < 1, soit dt < 1/sqrt(3) ~ 0.577.
    """
    n = len(t)
    z1, z2 = np.zeros(n), np.zeros(n)
    z1[0], z2[0] = y0, v0
    z1[1] = z1[0] + dt * z2[0]
    z2[1] = z2[0] - omega2 * dt * z1[0]
    for i in range(1, n - 1):
        z1[i + 1] = z1[i - 1] + 2.0 * dt * z2[i]
        z2[i + 1] = z2[i - 1] - 2.0 * omega2 * dt * z1[i]
    return z1, z2


def runge_kutta_4(t, dt):
    n = len(t)
    z1, z2 = np.zeros(n), np.zeros(n)
    z1[0], z2[0] = y0, v0
    for i in range(n - 1):
        k1_y = z2[i];                    k1_v = -omega2 * z1[i]
        k2_y = z2[i] + 0.5 * dt * k1_v;  k2_v = -omega2 * (z1[i] + 0.5 * dt * k1_y)
        k3_y = z2[i] + 0.5 * dt * k2_v;  k3_v = -omega2 * (z1[i] + 0.5 * dt * k2_y)
        k4_y = z2[i] + dt * k3_v;        k4_v = -omega2 * (z1[i] + dt * k3_y)
        z1[i + 1] = z1[i] + (dt / 6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        z2[i + 1] = z2[i] + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return z1, z2


def energy(z1, z2):
    """E = 1/2 z2^2 + omega2/2 z1^2  (vaut 2 avec nos conditions initiales)"""
    return 0.5 * z2**2 + 0.5 * omega2 * z1**2


METHODES = [
    # nom, fonction, couleur, style
    ("Euler explicite",   euler_explicite,    "#e63946", "-"),
    ("Euler implicite",   euler_implicite,    "#3a7bd5", "-"),
    ("Euler symplectique", euler_symplectique, "#8e44ad", "-"),
    ("Leap-Frog",         leap_frog,          "#2a9d8f", "-"),
    ("RK4",               runge_kutta_4,      "#f4820b", "--"),
]


def solve_all(dt, t_max):
    t = np.arange(0, t_max + dt / 2, dt)
    res = {}
    for nom, f, _, _ in METHODES:
        z1, z2 = f(t, dt)
        res[nom] = (z1, z2, energy(z1, z2))
    return t, res


def col(nom):
    return next(c for n, _, c, _ in METHODES if n == nom)


plt.rcParams["font.size"] = 12
plt.rcParams["lines.linewidth"] = 2.5

# ============================================================================
# FIGURES dt = 0.05 : solutions et energie
# ============================================================================
dt = 0.05
t, R = solve_all(dt, 10.0)
y_ex = exact(t)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Resolution de l'equation y'' + 3y = 0 avec y(0) = 1, y'(0) = -1",
             fontsize=16, fontweight="bold")

axes[0].plot(t, y_ex, "k-", label="Solution exacte")
axes[0].plot(t, R["Euler explicite"][0], "b--", label="Euler explicite")
axes[0].plot(t, R["Euler implicite"][0], "r-.", label="Euler implicite")
axes[0].set_title("Methodes d'Euler", fontweight="bold")

axes[1].plot(t, y_ex, "k-", label="Solution exacte")
axes[1].plot(t, R["Euler symplectique"][0], "--", color=col("Euler symplectique"),
             label="Euler symplectique")
axes[1].plot(t, R["Leap-Frog"][0], "g:", label="Leap-Frog")
axes[1].set_title("Euler symplectique et Leap-Frog", fontweight="bold")

axes[2].plot(t, y_ex, "k-", label="Solution exacte")
axes[2].plot(t, R["RK4"][0], "m--", label="Runge-Kutta 4")
axes[2].set_title("Methode Runge-Kutta 4", fontweight="bold")

for ax in axes:
    ax.set_xlabel("Temps t", fontweight="bold")
    ax.set_ylabel("y(t)", fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 10.0])
plt.tight_layout()
fig.savefig(os.path.join(IMG, "ordre2-0.05.png"), dpi=250)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axhline(E0, color="k", label="Energie exacte")
for nom, _, c, ls in METHODES:
    ax.plot(t, R[nom][2], ls, color=c, label=nom)
ax.set_xlabel("Temps t", fontweight="bold")
ax.set_ylabel("Energie E", fontweight="bold")
ax.set_title("Conservation de l'energie pour y'' + 3y = 0  (dt = 0.05)",
             fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(IMG, "energie-ordre2.png"), dpi=250)
plt.close(fig)

# ============================================================================
# FIGURES dt = 0.1 : energie (log + zoom) et portrait de phase
# ============================================================================
dt = 0.1
T = 2 * np.pi / omega
t, R = solve_all(dt, 4 * T)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle(r"Oscillateur harmonique conservatif  $y'' + 3y = 0$  —  $dt = 0.1$",
             fontsize=20, fontweight="bold")

for nom, _, c, ls in METHODES:
    ax1.semilogy(t, R[nom][2], ls, color=c, label=nom)
ax1.axhline(E0, color="k", ls="--", label=r"Exact : $E_0 = 2$")
ax1.set_title("Énergie — toutes méthodes", fontsize=18, fontweight="bold")
ax1.set_xlabel(r"$t$", fontsize=18)
ax1.set_ylabel(r"$E(t)$  (échelle log)", fontsize=18)
ax1.set_ylim(5e-4, 3e2)
ax1.annotate("diverge (+énergie)", xy=(10.2, 32), xytext=(5.5, 40),
             color=col("Euler explicite"), fontsize=16, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=col("Euler explicite"), lw=2))
ax1.annotate("dissipe (−énergie)", xy=(10.4, 0.11), xytext=(5.8, 6e-3),
             color=col("Euler implicite"), fontsize=16, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=col("Euler implicite"), lw=2))

zoom = ["Euler symplectique", "Leap-Frog", "RK4"]
for nom in zoom:
    _, _, c, ls = next(m for m in METHODES if m[0] == nom)
    ax2.plot(t, R[nom][2], ls, color=c, label=nom)
ax2.axhline(E0, color="k", ls="--", label=r"Exact : $E_0 = 2$")
ax2.set_title("Zoom : Euler symplectique, Leap-Frog et RK4", fontsize=18, fontweight="bold")
ax2.set_xlabel(r"$t$", fontsize=18)
ax2.set_ylabel(r"$E(t)$", fontsize=18)
ax2.set_ylim(1.6, 2.5)
txt = "\n".join(
    rf"{nom} : $\Delta E/E_0$ = {100*(R[nom][2].max()-R[nom][2].min())/E0:.2g} %"
    for nom in zoom)
ax2.text(0.98, 0.97, txt, transform=ax2.transAxes, ha="right", va="top", fontsize=15,
         bbox=dict(boxstyle="round", fc="#fffbe6", ec="#e0b400"))

for ax in (ax1, ax2):
    for k in range(1, 5):
        ax.axvline(k * T, color="gray", ls=":", lw=1, alpha=0.6)
        ax.text(k * T, ax.get_ylim()[1], rf"$T_{k}$", color="gray", fontsize=13,
                ha="right", va="top")
    ax.set_xlim(0, 4 * T)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=14)
    ax.tick_params(labelsize=14)
plt.tight_layout()
fig.savefig(os.path.join(IMG, "energieC.png"), dpi=200)
plt.close(fig)

# --- portrait de phase --------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 11))
th = np.linspace(0, 2 * np.pi, 400)
a, b = np.sqrt(2 * E0 / omega2), np.sqrt(2 * E0)
ax.fill(a * np.cos(th), b * np.sin(th), color="lightgray", alpha=0.5)
ax.plot(a * np.cos(th), b * np.sin(th), "k--", lw=4,
        label=r"Exact : $\frac{1}{2}z_2^2 + \frac{3}{2}z_1^2 = 2$")
ordre = ["Euler explicite", "Euler implicite", "Euler symplectique", "Leap-Frog", "RK4"]
for nom in ordre:
    _, _, c, ls = next(m for m in METHODES if m[0] == nom)
    z1, z2, _ = R[nom]
    ax.plot(z1, z2, ls, color=c, lw=2.5 if nom != "RK4" else 2, label=nom)
ax.plot([y0], [v0], "ko", ms=14)
ax.annotate(r"C.I. $(1,\,-1)$", xy=(y0, v0), xytext=(1.5, -2.0), fontsize=17,
            arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(a, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="dimgray", lw=2))
ax.annotate("", xy=(0, b), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="dimgray", lw=2))
ax.text(0.25, 0.25, rf"$a \approx {a:.3f}$", color="dimgray", fontsize=14)
ax.text(0.15, 1.1, rf"$b = {b:.0f}$", color="dimgray", fontsize=14)
ax.text(0.97, 0.97, "Système conservatif\n" + r"$E = \frac{1}{2}z_2^2 + \frac{3}{2}z_1^2 = \mathbf{2}$",
        transform=ax.transAxes, ha="right", va="top", fontsize=18,
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="olive", lw=2))
ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
ax.set_aspect("equal")
ax.set_xlabel(r"$z_1 = y$", fontsize=20)
ax.set_ylabel(r"$z_2 = y'$", fontsize=20)
ax.set_title(r"Portrait de phase — $y'' + 3y = 0$,  $dt = 0.1$", fontsize=20, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=15)
ax.tick_params(labelsize=15)
plt.tight_layout()
fig.savefig(os.path.join(IMG, "portraitP.png"), dpi=200)
plt.close(fig)

# ============================================================================
# RESULTATS NUMERIQUES
# ============================================================================
for dt, t_max in [(0.05, 10.0), (0.1, 4 * T)]:
    t, R = solve_all(dt, t_max)
    print("=" * 72)
    print(f"y'' + 3y = 0,  y(0)=1, y'(0)=-1,  dt = {dt},  t in [0, {t_max:.2f}]")
    print("-" * 72)
    print(f"{'Methode':20s} {'y(t_max)':>14s} {'erreur y':>10s} {'E min':>8s} {'E max':>8s} {'E fin':>8s}")
    print(f"{'Exacte':20s} {exact(t[-1]):14.8f} {0:10.1e} {E0:8.4f} {E0:8.4f} {E0:8.4f}")
    for nom, _, _, _ in METHODES:
        z1, z2, E = R[nom]
        print(f"{nom:20s} {z1[-1]:14.8f} {abs(z1[-1]-exact(t[-1])):10.1e} "
              f"{E.min():8.4f} {E.max():8.4f} {E[-1]:8.4f}")
    # verification de l'energie fantome de l'Euler symplectique
    z1, z2, _ = R["Euler symplectique"]
    Etilde = 0.5 * z2**2 + 0.5 * omega2 * z1**2 + 0.5 * omega2 * dt * z1 * z2
    print(f"Euler symplectique : E~ = E + (3/2) dt z1 z2  varie de "
          f"{Etilde.min():.10f} a {Etilde.max():.10f}  (conservee exactement)")
print("=" * 72)
