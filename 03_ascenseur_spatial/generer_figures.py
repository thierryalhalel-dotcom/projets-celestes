"""
Génération des figures de l'article "L'ascenseur spatial" - BUP n°994
Thierry ALHALEL, Vol. 111, Mai 2017
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, Arc, FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# ─── Constantes physiques ────────────────────────────────────────────────────
G      = 6.67e-11    # m³/(kg·s²)
MT     = 5.98e24     # kg
RT     = 6.4e6       # m  (6 400 km)
omega  = 7.27e-5     # rad/s
g0     = G * MT / RT**2          # ≈ 9,77 m/s²
RGEO   = (G * MT / omega**2)**(1/3)  # ≈ 42 260 km
hGEO   = RGEO - RT               # ≈ 35 860 km

# Matériau X (nanotubes de carbone)
mu_X   = 1400        # kg/m³
Re_X   = 45e9        # Pa
h_car  = Re_X / (mu_X * g0)  # ≈ 3 290 km

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

print(f"g0   = {g0:.3f} m/s²")
print(f"RGEO = {RGEO/1e6:.3f} Mm = {RGEO/RT:.4f} RT")
print(f"hGEO = {hGEO/1e3:.0f} km = {hGEO/RT:.3f} RT")
print(f"h_car= {h_car/1e3:.0f} km")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Forces d'inertie
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6, 5))
ax  = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

# Axes
orig = np.zeros(3)
ax.quiver(0,0,0, 1.4,0,0, arrow_length_ratio=0.1, color='k', linewidth=1.2)
ax.quiver(0,0,0, 0,1.4,0, arrow_length_ratio=0.1, color='k', linewidth=1.2)
ax.quiver(0,0,0, 0,0,1.6, arrow_length_ratio=0.08, color='k', linewidth=1.2)
ax.text(1.5, 0, 0, 'x', fontsize=11)
ax.text(0, 1.5, 0, 'y', fontsize=11)
ax.text(0, 0, 1.7, 'z', fontsize=11)
ax.text(0.05, 0.05, 0.05, 'O', fontsize=10)

# Vecteur ω le long de z
ax.quiver(0,0,0.2, 0,0,0.6, arrow_length_ratio=0.15, color='k', linewidth=1.5)
ax.text(-0.15, 0, 0.85, r'$\vec{\omega}$', fontsize=12)

# Point A
A = np.array([0.7, 0.7, 1.1])
ax.scatter(*A, color='k', s=25, zorder=5)
ax.text(A[0]+0.05, A[1]+0.05, A[2]+0.05, 'A', fontsize=11)

# H = projection de A sur Oz
H = np.array([0, 0, A[2]])
ax.scatter(*H, color='k', s=20, zorder=5)
ax.text(-0.18, 0, A[2]+0.05, 'H', fontsize=10)

# P = projection de A sur Oxy
P = np.array([A[0], A[1], 0])
ax.scatter(*P, color='k', s=20, zorder=5)
ax.text(P[0]+0.05, P[1]+0.05, -0.1, 'P', fontsize=10)

# Lignes de projection en tirets
ax.plot([A[0],A[0]], [A[1],A[1]], [0,A[2]], 'k--', linewidth=0.7, alpha=0.6)
ax.plot([A[0],0],    [A[1],0],    [A[2],A[2]], 'k--', linewidth=0.7, alpha=0.6)
ax.plot([A[0],A[0]], [A[1],A[1]], [A[2],A[2]], 'k-', linewidth=0.3, alpha=0.3)
ax.plot([0,A[0]],   [0,A[1]],    [0,0],  'k--', linewidth=0.7, alpha=0.6)

# Force de Coriolis (rouge) : -2m ω×v_A → supposons v_A selon z (montée)
# ω×v_A : ω selon z, v selon z → ω×v = 0 ... pour rendre visible, prenons v selon y
# En pratique: force Coriolis ≈ vers l'Est (y>0) lors d'une montée
F_cor = np.array([0.5, 0, 0])  # en rouge, vers x
ax.quiver(*A, *F_cor, arrow_length_ratio=0.2, color='red', linewidth=2)
ax.text(A[0]+F_cor[0]+0.05, A[1]+0.05, A[2]-0.05, r'$\vec{F}_{Cor}$', fontsize=10, color='red')

# Force centrifuge (bleue) : mω²·HA, de H vers A dans le plan Oxy
HA = np.array([A[0]-H[0], A[1]-H[1], 0])
HA_norm = HA / np.linalg.norm(HA) * 0.5
ax.quiver(*H, *HA_norm, arrow_length_ratio=0.2, color='blue', linewidth=2)
ax.text(H[0]+HA_norm[0]+0.05, H[1]+HA_norm[1], H[2]+0.1, r'$\vec{F}_{cf}$', fontsize=10, color='blue')

ax.set_xlim(-0.2, 1.8); ax.set_ylim(-0.2, 1.8); ax.set_zlim(-0.1, 2.0)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.grid(False)
ax.set_title("Figure 1 – Forces d'inertie", fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure1.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure1.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Intérêt de l'ascenseur spatial (rapport A et B)
# ──────────────────────────────────────────────────────────────────────────────
h_arr = np.linspace(0.01, 7.8, 2000)        # altitude en unité de RT
r_arr = (1 + h_arr) * RT                     # distance au centre en m

v_sat  = np.sqrt(G * MT / r_arr)             # vitesse orbitale circulaire
v_cab  = omega * r_arr                       # vitesse cabine (rotation terrestre)
v_lib  = np.sqrt(2 * G * MT / r_arr)         # vitesse de libération

A_ratio = v_cab / v_sat
B_ratio = v_cab / v_lib

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(h_arr, A_ratio, 'r-', linewidth=2, label=r'$A = v\prime/v$')
ax.plot(h_arr, B_ratio, 'b-', linewidth=2, label=r'$B = v\prime/v_{lib}$')
ax.axhline(1, color='gray', linestyle='--', linewidth=0.8)

# Points particuliers
idx_GEO = np.argmin(np.abs(A_ratio - 1.0))
h_geo_RT = h_arr[idx_GEO]
idx_lib = np.argmin(np.abs(B_ratio - 1.0))
h_lib_RT = h_arr[idx_lib]

ax.plot(h_geo_RT, 1.0, 'ko', markersize=6, zorder=5)
ax.plot(h_lib_RT, 1.0, 'ko', markersize=6, zorder=5)

ax.set_xlabel("altitude en unité de rayon terrestre", fontsize=11)
ax.set_ylabel("rapport A et B", fontsize=11)
ax.set_title("adéquation entre la vitesse cabine et les vitesses\n"
             "de satellisation et libération", fontsize=10)
ax.legend(fontsize=11)
ax.set_xlim(0, 8); ax.set_ylim(0, 1.8)
ax.set_xticks(range(0, 9))
ax.grid(True, alpha=0.3)

# Annotation boîte rouge (texte simplifié)
bbox_props = dict(boxstyle="round,pad=0.4", facecolor='#fff0f0', edgecolor='red', linewidth=1.2)
ax.annotate(
    "Courbe superieure : $A=v'/v$\n"
    "$A=1$ pour $h_{GEO}=5.6\\,R_T$\n\n"
    "Courbe inferieure : $B=v'/v_{lib}$\n"
    "$B=1$ pour $h>7.2\\,R_T$",
    xy=(4.5, 1.35), fontsize=8, bbox=bbox_props, color='darkred')

ax.set_title("Figure 2 – Intérêt de l'ascenseur spatial", fontsize=10, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure2.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure2.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Comparaison énergie fusée / ascenseur (Δ)
# ──────────────────────────────────────────────────────────────────────────────
# Δ = (E1 - E0)/(E2 - E0)
# z = r/RT (z=1 au sol, z>1 en altitude)
# E0 = -g0*RT + 0.5*RT²*ω²  (par unité de masse)
# E1 = -g0*RT²/(2*r) = -g0/(2z)  [orbite circulaire par fusée]
# E2 = -g0*RT²/r + 0.5*(ω*r)² = -g0/z + 0.5*ω²*z²*RT²  [ascenseur]

k = omega**2 * RT / g0   # ≈ 3.46e-3 (petit)

# En unités adimensionnées (par g0*RT) :
# E0 = -1 + k/2
# E1 = -1/(2z)
# E2 = -1/z + k*z²/2
# Δ = (E1-E0)/(E2-E0) = (-1/(2z) + 1 - k/2) / (-1/z + k*z²/2 + 1 - k/2)
#   = (1 - 1/(2z) - k/2) / (1 - 1/z + k*(z²-1)/2)

# Attention : E2-E0 = 0 pour z=1 (singularité)
# On trace pour z de 1.16 (h=1000km) à ~16 RT

z_arr = np.linspace(1.16, 16, 2000)
num  = 1 - 1.0/(2*z_arr) - k/2
den  = 1 - 1.0/z_arr + k*(z_arr**2 - 1)/2
Delta = num / den

# Ne garder que les valeurs raisonnables
mask = (den > 0.01) & (Delta < 5) & (Delta > 0)
h_plot = z_arr[mask] - 1   # altitude en RT

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(h_plot, Delta[mask], 'r-', linewidth=2)
ax.axhline(1, color='gray', linestyle='--', linewidth=0.8)

# Point bleu : Δ=1 à h=5,6 RT
idx_blue = np.argmin(np.abs(Delta[mask] - 1.0))
ax.plot(h_plot[idx_blue], 1.0, 'bo', markersize=8, zorder=5)
ax.annotate(f"$h=5{{,}}6$ et $\\Delta=1$",
            xy=(h_plot[idx_blue], 1.0), xytext=(7, 1.3),
            fontsize=9, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue'))

ax.set_xlabel("altitude $h$ en rayon terrestre", fontsize=11)
ax.set_ylabel("rapport $\\Delta$ (sans dimension)", fontsize=11)
ax.set_title("Figure 3 – Comparaison énergie fusée / énergie ascenseur",
             fontsize=10, style='italic')
ax.set_xlim(0, 16); ax.set_ylim(0.4, 4.0)
ax.set_xticks(range(0, 17, 2))
ax.grid(True, alpha=0.3)

# Légende boîte
bbox2 = dict(boxstyle="round,pad=0.4", facecolor='#f0f0ff', edgecolor='gray')
ax.text(8, 3.3,
        "Premier point ≈ alt. 1 000 km\n"
        "Point bleu : $h=5{,}6$ ; $\\Delta=1$\n"
        "Abscisse = altitude $h$ en $R_T$\n"
        "Ordonnée sans dimension",
        fontsize=8, bbox=bbox2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure3.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure3.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Câble équatorial et référentiel R
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_aspect('equal')
ax.axis('off')

# Terre (cercle)
earth = Circle((0, 0), 0.5, color='lightblue', ec='steelblue', linewidth=1.5, zorder=3)
ax.add_patch(earth)
ax.text(0, 0, 'C', ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

# Point O (sur l'équateur, à la surface)
ax.plot(0.5, 0, 'ko', markersize=5, zorder=4)
ax.text(0.52, 0.04, 'O', fontsize=10)

# Fil dans le plan de l'équateur (vers la droite)
L = 2.2
ax.annotate('', xy=(0.5+L, 0), xytext=(0.5, 0),
            arrowprops=dict(arrowstyle='-', color='k', lw=2))
ax.plot([0.5, 0.5+L], [0, 0], 'k-', linewidth=2.5)

# Flèche indiquant la longueur L
ax.annotate('', xy=(0.5+L, -0.18), xytext=(0.5, -0.18),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1))
ax.text(0.5 + L/2, -0.25, 'Longueur $L$ du fil', ha='center', fontsize=10)

# Vecteur ω (vers le haut = axe de rotation = Oz)
ax.annotate('', xy=(0, 1.1), xytext=(0, 0.55),
            arrowprops=dict(arrowstyle='->', color='k', lw=1.5))
ax.text(-0.12, 1.0, r'$\vec{\omega}$', fontsize=13)

# Axe Oz
ax.plot([0, 0], [-0.6, 0.55], 'k-', linewidth=0.8, linestyle=':')

# Label fil
ax.text(0.5+L+0.08, 0.05, 'Fil dans\nle plan\nde l\'équateur', fontsize=8, va='center')

ax.set_xlim(-0.8, 3.2)
ax.set_ylim(-0.5, 1.3)
ax.set_title("Figure 4 – Câble équatorial et référentiel $\\mathbf{R}$",
             fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure4.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure4.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Allure de la fonction α(z)
# ──────────────────────────────────────────────────────────────────────────────
# α(z) = g0/z² − ω²·RT·z   avec z en unités de RT
# Numériquement : 9,77/z² − 0,0338·z

coeff_centrifuge = omega**2 * RT   # ≈ 0,0338 m/s²

z_arr2 = np.linspace(0.3, 18, 2000)
alpha = g0 / z_arr2**2 - coeff_centrifuge * z_arr2

# Zéro de α : z_GEO = (g0/(ω²·RT))^(1/3)
z_zero = (g0 / coeff_centrifuge)**(1/3)
print(f"Zéro de α : z = {z_zero:.3f} RT → {z_zero*RT/1e6:.3f} Mm")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(z_arr2, alpha, 'b-', linewidth=2)
ax.axhline(0, color='k', linewidth=0.8)
ax.axvline(z_zero, color='gray', linewidth=0.8, linestyle='--')

ax.set_xlabel("$z$ en unité de $R_T$", fontsize=11)
ax.set_ylabel(r"$\alpha(z)$ (m·s$^{-2}$)", fontsize=11)
ax.set_xlim(0, 18); ax.set_ylim(-1, 6)
ax.set_xticks(range(0, 19, 2))
ax.grid(True, alpha=0.3)

# Annotations
ax.annotate(f"Racine approchée :\n$z \\approx 6{{,}}60\\,R_T = 42\\,240$ km\n$h = 5{{,}}6\\,R_T = h_{{GEO}}$",
            xy=(z_zero, 0), xytext=(10, 2.5),
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.text(11, -0.6,
        "En dessous de $h_{GEO}$ : $dT/dz > 0$\n"
        "En dessus de $h_{GEO}$ : $dT/dz < 0$",
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray'))

ax.set_title(r"Figure 5 – Allure de la fonction $\alpha(z)$ exprimée en rayons terrestres $R_T$",
             fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure5.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure5.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – Courbe contrainte σ vs déformation ε (schéma)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_aspect('auto')

# Coordonnées des zones clés (normalisées)
eps0 = 0.005   # limite linéaire
epsA = 0.008   # limite linéaire (A)
epsB = 0.012   # limite élastique (B)
epsC = 0.20    # contrainte maximale (C)
epsD = 0.30    # rupture (D)

sig0 = 0.5     # σ0
sigA = 0.55    # contrainte limite linéaire
sigB = 0.70    # limite élastique
sigC = 1.0     # résistance à la traction
sigD = 0.85    # contrainte à la rupture

# Courbe
E_mod  = sigA / epsA   # module de Young (pente)
eps = np.concatenate([
    np.linspace(0, epsA, 50),
    np.linspace(epsA, epsB, 30),
    np.linspace(epsB, epsC, 80),
    np.linspace(epsC, epsD, 30)
])
sig = np.zeros_like(eps)

# Région 1 : linéaire 0 → epsA
m1 = len(np.linspace(0, epsA, 50))
sig[:m1] = E_mod * eps[:m1]

# Région 2 : non-linéaire élastique epsA → epsB (courbure)
m2 = m1 + len(np.linspace(epsA, epsB, 30))
t = (eps[m1:m2] - epsA) / (epsB - epsA)
sig[m1:m2] = sigA + (sigB - sigA) * (3*t**2 - 2*t**3)

# Région 3 : plastique epsB → epsC
m3 = m2 + len(np.linspace(epsB, epsC, 80))
t = (eps[m2:m3] - epsB) / (epsC - epsB)
sig[m2:m3] = sigB + (sigC - sigB) * np.sqrt(t)

# Région 4 : striction epsC → epsD
sig[m3:] = sigC - (sigC - sigD) * ((eps[m3:] - epsC) / (epsD - epsC))**0.5

ax.plot(eps, sig, 'k-', linewidth=2)

# Lignes de repère
ax.axvline(epsA, color='gray', linestyle=':', linewidth=0.8)
ax.axvline(epsB, color='gray', linestyle=':', linewidth=0.8)
ax.plot([0, epsA+0.01], [0, (epsA+0.01)*E_mod], 'gray', linewidth=0.7, linestyle='--')

# Points remarquables
for ep, sg, lbl, dx, dy in [
    (epsA,  sigA,  'A',  0.003,  0.02),
    (epsB,  sigB,  'B',  0.003,  0.02),
    (epsC,  sigC,  'C',  0.003,  0.02),
    (epsD,  sigD,  'D',  0.003,  0.02),
]:
    ax.plot(ep, sg, 'ko', markersize=5)
    ax.text(ep + dx, sg + dy, lbl, fontsize=10, fontweight='bold')

ax.plot([0], [sig0], 'ko', markersize=5)
ax.text(-0.008, sig0, r'$\sigma_0$', fontsize=10, ha='right')

# Annotations des zones
ax.text(epsA/2, 0.05, 'Limite\nlinéaire', ha='center', fontsize=7.5, color='gray')
ax.annotate('', xy=(epsA, -0.05), xytext=(0, -0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
ax.text((epsA+epsB)/2, 0.05, 'Limite\nélastique', ha='center', fontsize=7.5, color='gray')
ax.annotate('', xy=(epsB, -0.05), xytext=(epsA, -0.05),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))

# Module de Young (étiquette sur la droite tangente)
ax.text(epsA/2 - 0.003, sigA/2 + 0.05, "$B$ : module\nde Young $E$",
        fontsize=8, rotation=52, color='gray')

# Zones domaine élastique / plastique
ax.text(epsB/2, -0.15, 'Domaine élastique\n= Domaine linéaire\n$\\sigma = E\\varepsilon$',
        ha='center', fontsize=7.5, color='steelblue')
ax.annotate('', xy=(epsB, -0.22), xytext=(0, -0.22),
            arrowprops=dict(arrowstyle='<->', color='steelblue', lw=1))
ax.text((epsB+epsD)/2, -0.15, 'Domaine plastique\n= Domaine non\nlinéaire',
        ha='center', fontsize=7.5, color='steelblue')
ax.annotate('', xy=(epsD, -0.22), xytext=(epsB, -0.22),
            arrowprops=dict(arrowstyle='<->', color='steelblue', lw=1))

# Limite de résistance à la traction
ax.annotate('Limite de résistance\nà la traction', xy=(epsC, sigC),
            xytext=(epsC+0.02, sigC+0.07), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
ax.annotate('Rupture', xy=(epsD, sigD), xytext=(epsD+0.02, sigD+0.07), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

ax.set_xlabel(r"$\varepsilon_n$ (%)", fontsize=11)
ax.set_ylabel(r"$\sigma_n$", fontsize=11)
ax.set_xlim(-0.01, 0.36)
ax.set_ylim(-0.30, 1.25)
ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title(r"Figure 6 – Contrainte $\sigma$ d'un matériau en fonction"
             "\nde sa déformation relative $\\varepsilon$ (%)",
             fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure6.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure6.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 7 – Variations du rapport S/S0 (matériau X)
# ──────────────────────────────────────────────────────────────────────────────
# S/S0 = exp(a * [1/RT - 1/z + RT³/(2*RGEO³) * (1 - z²/RT²)])
# Numériquement (z en RT) :
# S/S0 = exp(1.945 * (1 - 1/z + 1.72e-3 * (1 - z²)))

coeff_a = mu_X * g0 * RT / Re_X            # ≈ 1.945
coeff_c = RT**3 / (2 * RGEO**3)            # ≈ RT/(2*RGEO) * (RT/RGEO)² ≈ 1.72e-3
print(f"coeff_a = {coeff_a:.4f}  (article: 1.945)")
print(f"coeff_c = {coeff_c:.4e} (article: 1.72e-3)")

# La courbe de l'article s'arrête là où elle repasse sous le bas du cadre
# (S/S0 ≈ 0,68 à z ≈ 26 RT) ; on reproduit cette plage.
z3 = np.linspace(1, 26, 2000)   # z en RT
S_S0 = np.exp(coeff_a * (1 - 1/z3 + coeff_c * (1 - z3**2)))

# Maximum à z_GEO
z_geo = RGEO / RT
S_max = np.exp(coeff_a * (1 - 1/z_geo + coeff_c * (1 - z_geo**2)))
print(f"S/S0 max = {S_max:.3f} à z={z_geo:.3f} RT  (article: ≈ 4.5 = e^1.5)")

fig, ax = plt.subplots(figsize=(9.5, 4.8))
# Le graphe occupe la moitié gauche, l'encart la moitié droite (comme l'article)
fig.subplots_adjust(left=0.08, right=0.52, top=0.95, bottom=0.12)
ax.plot(z3, S_S0, color='#2222cc', linewidth=2)

# Échelles identiques à la figure 7 de l'article
ax.set_xlabel("z en unité de RT", fontsize=10)
ax.set_ylabel("rapport Section / Section au sol", fontsize=10)
ax.set_xlim(0, 30)
ax.set_ylim(0.5, 5)
ax.set_xticks(range(0, 31, 5))
ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
ax.grid(True, color='0.75', linewidth=0.5)
ax.tick_params(labelsize=9)

# Encart explicatif à droite du graphe (texte de l'article)
fig.text(
    0.57, 0.55,
    "La courbe ci-contre montre la variation\n"
    "du quotient $S/S_0$ en fonction de la distance\n"
    "au centre de la Terre exprimée\n"
    "en rayon terrestre $R_T$.\n\n"
    "Le rapport vaut au maximum $e^{1,5}\\approx 4{,}5$\n"
    "pour $z = 6{,}5\\,R_T \\approx 42\\,000$ km,\n"
    "c'est-à-dire pour $h = h_{GEO}$.\n\n"
    "Au maximum la section du câble sera\n"
    "donc de 45 mm$^2$ si la section initiale\n"
    "vaut 10 mm$^2$.",
    fontsize=11, va='center', ha='left')
plt.savefig(os.path.join(OUT, "figure7.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure7.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 8 – Variation de T(RT) avec Z et M
# ──────────────────────────────────────────────────────────────────────────────
# Section constante S0 = 10 mm², matériau X
S0_m2 = 10e-6          # m²
rho   = mu_X * S0_m2   # ≈ 0.014 kg/m (masse linéique)

def T_at_RT(Z_m, M_kg):
    """Tension au sol T(RT) en fonction de la position Z du contrepoids (m) et de sa masse M (kg)."""
    return (M_kg * (omega**2 * Z_m - g0 * RT**2 / Z_m**2)
            - rho * omega**2 * (Z_m**2 - RT**2) / 2
            + rho * g0 * RT
            - rho * g0 * RT**2 / Z_m)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# ─ Panneau supérieur : Z_init=45 000 km, M de 5 000 à 15 000 t ─────────────
Z_top = np.linspace(45e6, 65e6, 400)   # m
M_top = np.arange(5000, 16000, 1000) * 1e3  # kg (5 000 à 15 000 t, pas 1 000 t)

ax = axes[0]
for M in M_top:
    T_vals = np.array([T_at_RT(Z, M) for Z in Z_top])
    ax.plot(Z_top / RGEO, T_vals, 'r-', linewidth=0.9)

ax.axhline(0, color='k', linewidth=0.8)
ax.set_xlim(Z_top[0]/RGEO, Z_top[-1]/RGEO)
ax.set_title("variation de T(RT) avec Z et M : 5 000 t minimum, Z : 45 000 km",
             fontsize=9)
ax.set_ylabel("$T(R_T)$ (N)", fontsize=10)
ax.ticklabel_format(axis='x', style='plain')
ax.set_xticks(np.round(np.linspace(Z_top[0]/RGEO, Z_top[-1]/RGEO, 11), 2))
ax.xaxis.set_tick_params(labelsize=8)
ax.grid(True, alpha=0.2)
# Échelle y
y_max = max(abs(T_at_RT(Z_top[-1], M_top[-1])), abs(T_at_RT(Z_top[0], M_top[0])))
ax.set_ylim(-y_max*0.4, y_max*1.05)
# Notation ×10^8
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.text(0.01, 0.95, r"$\times 10^8$", transform=ax.transAxes, fontsize=9, va='top')

# ─ Panneau inférieur : Z_init=85 000 km, M de 700 à 1 700 t ─────────────────
Z_bot = np.linspace(85e6, 105e6, 400)
M_bot = np.arange(700, 1800, 100) * 1e3  # 700 à 1 700 t, pas 100 t

ax = axes[1]
for M in M_bot:
    T_vals = np.array([T_at_RT(Z, M) for Z in Z_bot])
    axes[1].plot(Z_bot / RGEO, T_vals, 'r-', linewidth=0.9)

axes[1].axhline(0, color='k', linewidth=0.8)
axes[1].set_xlim(Z_bot[0]/RGEO, Z_bot[-1]/RGEO)
axes[1].set_title("variation de T(RT) avec Z et M : 700 t minimum, Z : 85 000 km",
                  fontsize=9)
axes[1].set_xlabel("$Z$ en valeurs du rayon de l'orbite géostationnaire\n"
                   "(abscisse = 1 correspond à $R_{GEO}$)", fontsize=9)
axes[1].set_ylabel("$T(R_T)$ (N)", fontsize=10)
axes[1].set_xticks(np.round(np.linspace(Z_bot[0]/RGEO, Z_bot[-1]/RGEO, 11), 2))
axes[1].xaxis.set_tick_params(labelsize=8)
axes[1].grid(True, alpha=0.2)
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
axes[1].text(0.01, 0.05, r"$\times 10^9$", transform=axes[1].transAxes, fontsize=9, va='bottom')

fig.suptitle("Figure 8 – Variation de $T(R_T)$ en newtons en fonction de la valeur de $Z$,\n"
             "exprimée en valeurs du rayon de l'orbite géostationnaire",
             fontsize=10, style='italic')
plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure8.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure8.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 9 – Force de Coriolis et oscillations
# ──────────────────────────────────────────────────────────────────────────────
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect('equal')
ax.axis('off')

# --- Terre avec dégradé océanique radial + éclairage haut-gauche ---
ng = 600
gy, gx = np.mgrid[-1:1:ng*1j, -1:1:ng*1j]
gr = np.sqrt(gx**2 + gy**2)
light = ((-gx + gy) / np.sqrt(2) + 1) / 2          # clair en haut-gauche
shade = np.clip(0.30 + 0.70 * light - 0.18 * gr, 0, 1)
shade = np.ma.masked_where(gr > 1.0, shade)
ocean = LinearSegmentedColormap.from_list(
    'ocean', ['#0b3d66', '#1f6fb2', '#4ea3e0', '#bfe6ff'])
ax.imshow(shade, extent=(-1, 1, -1, 1), origin='lower', cmap=ocean,
          zorder=2, interpolation='bilinear')
ax.add_patch(Circle((0, 0), 1.0, fill=False, ec='#08334f', linewidth=2.5, zorder=3))
# Méridiens / équateur stylisés
ax.add_patch(mpatches.Arc((0, 0), 1.1, 2.0, angle=0, theta1=90, theta2=270,
                          color='#cfeeff', linewidth=0.8, alpha=0.6, zorder=3))
ax.add_patch(mpatches.Arc((0, 0), 0.55, 2.0, angle=0, theta1=90, theta2=270,
                          color='#cfeeff', linewidth=0.8, alpha=0.6, zorder=3))
ax.plot([-1, 1], [0, 0], color='#cfeeff', linewidth=0.8, alpha=0.6, zorder=3)

# --- Axe de rotation (Nord) ---
ax.annotate('', xy=(0, 1.55), xytext=(0, 1.02),
            arrowprops=dict(arrowstyle='-|>', color='#222222', lw=2.2))
ax.text(0.10, 1.46, r'$\vec{\omega}$', fontsize=16, color='#222222')

# --- Câble équatorial ---
ax.plot([1.0, 3.7], [0, 0], color='#3a3a3a', linewidth=3, zorder=2,
        solid_capstyle='round')
ax.text(3.8, 0.07, 'Câble équatorial', fontsize=11, color='#3a3a3a')

# --- Cabine ---
cab_x, cab_y = 2.2, 0.0
cab = mpatches.FancyBboxPatch((cab_x-0.11, cab_y-0.13), 0.22, 0.26,
                              boxstyle="round,pad=0.02",
                              facecolor='#ffb733', edgecolor='#7a4b00',
                              linewidth=1.8, zorder=4)
ax.add_patch(cab)
ax.text(cab_x, cab_y, 'm', fontsize=11, ha='center', va='center',
        fontweight='bold', color='#5a3500', zorder=5)

# --- Vecteur vitesse v : selon le câble, vers l'Est ---
ax.annotate('', xy=(cab_x+0.70, cab_y), xytext=(cab_x+0.11, cab_y),
            arrowprops=dict(arrowstyle='-|>', color='#1f4fd6', lw=3))
ax.text(cab_x+0.62, cab_y+0.12, r'vitesse $\vec{v}$',
        fontsize=12, color='#1f4fd6', fontweight='bold', ha='center')

# --- Vecteur vitesse angulaire ω : perpendiculaire au câble en m ---
ax.annotate('', xy=(cab_x, cab_y+0.62), xytext=(cab_x, cab_y+0.13),
            arrowprops=dict(arrowstyle='-|>', color='#7a1fa2', lw=3))
ax.text(cab_x+0.18, cab_y+0.62, r'vitesse angulaire $\vec{\omega}$',
        fontsize=12, color='#7a1fa2', fontweight='bold', ha='right')

# --- Titre interne + encart ---
ax.text(0, 2.05, "Ascenseur en montée / descente :", fontsize=11, ha='center',
        fontweight='bold', color='#1a1a1a')
ax.text(3.8, 0.9,
        "La force subie en montée\n"
        "est orientée vers l'Ouest\n"
        "($\\vec{u}$ orienté sortant\n"
        "perpendiculairement à la feuille).\n"
        "En descente cette force\n"
        "est orientée vers l'Est.",
        fontsize=10, va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8d6',
                  edgecolor='#c9a227', linewidth=1.2))

ax.set_xlim(-1.7, 5.4)
ax.set_ylim(-1.5, 2.4)
ax.set_title("Figure 9 – Force de Coriolis et oscillations", fontsize=13,
             style='italic', color='#08334f')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure9.png"), dpi=300, bbox_inches='tight')
plt.close()
print("figure9.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 10 – Déplacement géométrique lors de la montée
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 7))
ax.set_aspect('equal')
ax.axis('off')

# Terre (cercle)
earth = Circle((0, 0), 1.2, color='#d0e8ff', ec='k', linewidth=2, zorder=3)
ax.add_patch(earth)

# Câble vertical (référence, en tirets)
cable_len = 5.5
ax.plot([0, 0], [1.2, 1.2+cable_len], 'k--', linewidth=1, alpha=0.5)

# Câble réel : déviation δ (angle β en bas, angle α en haut)
# Convention : β angle du bas, α angle du haut
# Pour la figure : le câble dévie vers la droite (vers l'Est)
# Ancrage en bas à x=0, y=R_T (bord de la Terre)
# Point GEO à x=delta, y=R_T+h_geo (relativement)
# Sommet à x=delta_sommet, y=R_T+L

delta = 0.45     # déplacement horizontal au niveau GEO
h_geo_disp = 2.5 # en unités de la figure
L_disp = 5.5     # longueur totale du câble
R_T_disp = 1.2   # rayon de la Terre

# Câble en deux segments (angle β en bas, angle α en haut)
# Point d'ancrage au sol
x0, y0 = 0.0, R_T_disp

# Point GEO
x_geo = delta
y_geo = R_T_disp + h_geo_disp

# Sommet (on garde l'angle α plus petit)
x_top = delta * 1.6
y_top = R_T_disp + L_disp

# Tracé du câble (courbure simulée)
from matplotlib.path import Path
import matplotlib.patches as patches

# Câble comme deux segments
ax.plot([x0, x_geo, x_top], [y0, y_geo, y_top], 'k-', linewidth=2, zorder=4)

# Ligne verticale de référence depuis le point GEO
ax.plot([x_geo, x_geo], [y_geo, y_top+0.3], 'k--', linewidth=0.8, alpha=0.7)

# Flèche δ (déplacement horizontal au niveau GEO)
ax.annotate('', xy=(x_geo, y_geo), xytext=(0, y_geo),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1.5))
ax.text(x_geo/2, y_geo+0.15, r'$\delta$', ha='center', fontsize=12, color='k')

# Angle β (bas) avec une arc
beta_arc = Arc((x0, y0), 0.5, 0.5, angle=0,
               theta1=90, theta2=90+np.degrees(np.arctan2(x_geo-x0, y_geo-y0)),
               color='k', lw=1.5)
ax.add_patch(beta_arc)
ax.text(0.12, R_T_disp+0.35, r'$\beta$', fontsize=11, color='k')

# Angle α (haut) avec une arc
alpha_val = np.degrees(np.arctan2(x_top-x_geo, y_top-y_geo))
alpha_arc = Arc((x_geo, y_geo), 0.5, 0.5, angle=0,
                theta1=90, theta2=90+alpha_val,
                color='k', lw=1.5)
ax.add_patch(alpha_arc)
ax.text(x_geo+0.15, y_geo+0.35, r'$\alpha$', fontsize=11, color='k')

# Flèche L (longueur totale)
ax.annotate('', xy=(-0.7, y_top), xytext=(-0.7, y0),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
ax.text(-1.0, (y0+y_top)/2, '$L$', fontsize=12, ha='center', va='center')

# Flèche hGEO
ax.annotate('', xy=(-0.3, y_geo), xytext=(-0.3, y0),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1))
ax.text(-0.55, (y0+y_geo)/2, r'$h_{GEO}$', fontsize=10, ha='center', va='center', color='k')

# Sommet (flèche vers le haut = contrepoids/ancrage)
ax.annotate('', xy=(x_top, y_top+0.5), xytext=(x_top, y_top),
            arrowprops=dict(arrowstyle='->', color='k', lw=1))
ax.text(x_top+0.1, y_top+0.3, 'contrepoids', fontsize=8)

ax.set_xlim(-1.5, 3.5)
ax.set_ylim(-1.4, 7.5)
ax.set_title("Figure 10 – Déplacement géométrique lors de la montée",
             fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure10.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure10.png OK")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 11 – Structure du champ magnétique dipolaire terrestre
# ──────────────────────────────────────────────────────────────────────────────
# Lignes de champ d'un dipôle magnétique : r = r0 * sin²(θ)
# θ = colatitude (0 = pôle Nord, π/2 = équateur)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# Étendue de la vue (zoom : la Terre apparaît plus grosse)
LIM = 5.5

# Inclinaison de l'axe magnétique p/r à l'axe géographique (~11° sur Terre)
tilt = np.radians(11)
ct, st = np.cos(tilt), np.sin(tilt)
def rot(x, y):
    """Rotation de l'axe magnétique par rapport à l'axe géographique (vertical)."""
    return x * ct - y * st, x * st + y * ct

# Lignes de champ (dipôle magnétique incliné)
theta = np.linspace(0.05, np.pi - 0.05, 500)
r0_values = [1.4, 1.9, 2.6, 3.5, 4.6]  # en unités R_T

for r0 in r0_values:
    r = r0 * np.sin(theta)**2
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    xr, yr = rot(x, y);   ax.plot(xr, yr, 'k-', linewidth=0.9)
    xr, yr = rot(-x, y);  ax.plot(xr, yr, 'k-', linewidth=0.9)

# Lignes de champ internes (dans la Terre, simulées)
for r0 in [1.1, 1.25]:
    r = r0 * np.sin(theta)**2
    mask = r >= 1.0
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    xr, yr = rot(x[mask], y[mask]);   ax.plot(xr, yr, 'k-', linewidth=0.7, alpha=0.5)
    xr, yr = rot(-x[mask], y[mask]);  ax.plot(xr, yr, 'k-', linewidth=0.7, alpha=0.5)

# Terre (cercle) — plus grosse grâce au zoom
earth = Circle((0, 0), 1.0, color='#c8e0f0', ec='steelblue', linewidth=2, zorder=5)
ax.add_patch(earth)

# Axe de rotation géographique (vertical)
ax.plot([0, 0], [-LIM, LIM], 'k-', linewidth=0.9, linestyle='dotted', alpha=0.6, zorder=4)

# Axe magnétique (incliné) — pour montrer le décalage N géo / N mag
mx_top, my_top = rot(0,  LIM)
mx_bot, my_bot = rot(0, -LIM)
ax.plot([mx_bot, mx_top], [my_bot, my_top], color='purple',
        linewidth=0.9, linestyle='--', alpha=0.7, zorder=4)

# Annotation de l'angle (~11°) entre axe géographique et axe magnétique
angle_arc = Arc((0, 0), 5.4, 5.4, angle=0, theta1=90, theta2=90+np.degrees(tilt),
                color='purple', lw=1.2, zorder=6)
ax.add_patch(angle_arc)
amid = np.radians(90 + np.degrees(tilt)/2)
ax.text(3.05*np.cos(amid), 3.05*np.sin(amid), r'$\approx 11°$',
        fontsize=9, color='purple', ha='center', va='bottom', zorder=6)

# Pôle géographique Nord (axe vertical) — placé plus haut
ax.annotate('', xy=(0, 2.4), xytext=(0, 1.05),
            arrowprops=dict(arrowstyle='->', color='k', lw=1.6, zorder=6))
ax.text(0.10, 2.30, 'N géo', fontsize=9, zorder=6)

# Pôles magnétiques le long de l'axe incliné (légèrement décalés de N/S géo)
nmx, nmy = rot(0, 1.95)
ax.text(nmx + 0.12, nmy, 'N mag.', fontsize=9, zorder=6, color='red', va='center')
smx, smy = rot(0, -1.6)
ax.text(smx, smy - 0.05, 'S mag.', fontsize=9, ha='center', va='top',
        zorder=6, color='blue')

# Flèches de sens des lignes de champ : du Nord vers le Sud
def field_arrow(r0, th0, sign):
    dth = 0.05
    def pt(t):
        rr = r0 * np.sin(t)**2
        return rot(sign * rr * np.sin(t), rr * np.cos(t))
    x1, y1 = pt(th0 - dth)   # côté Nord
    x2, y2 = pt(th0 + dth)   # côté Sud (θ croissant -> vers le Sud)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2), zorder=6)

for r0 in r0_values:
    for sign in (+1, -1):
        field_arrow(r0, np.pi/3, sign)      # hémisphère Nord
        field_arrow(r0, np.pi/2, sign)      # passage équatorial
        field_arrow(r0, 2*np.pi/3, sign)    # hémisphère Sud

ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_title("Figure 11 – Structure du champ magnétique dipolaire terrestre",
             fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "figure11.png"), dpi=150, bbox_inches='tight')
plt.close()
print("figure11.png OK")

print("\n=== Toutes les figures ont été générées ===")
