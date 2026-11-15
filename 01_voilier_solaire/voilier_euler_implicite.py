import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------------------------
#  constantes physiques (SI)
# -------------------------------------------------
G = 6.67430e-11          #  (m^3 kg^-1 s^-2)
M = 5.9722e24           # masse Terre(kg)
R = 6371e3               # rayon Terre (m)
p = 4.6e-6               # pression solaire 1 AU (N/m^2)
jour = 5

# -------------------------------------------------
# parametres de vol
# -------------------------------------------------
r0 = 6.6316*R        # distance initiale
vr0 = 0.0                # vitesse radiale (m/s)
vt0 = math.sqrt(G * M / r0)   # norme pour r0
phi0 = 3 * np.pi / 2     # angle polaire initial (rad)
alpha0 = np.pi / 2       # angle vol initial (rad)
m_ship = 2000.0          #  masse (kg)
A_sail = 2.13e6          # surface voile (m^2)
a_m = 2 * A_sail * p / m_ship   # facteur acceleration (m/s^2)

# -------------------------------------------------
#  parametres numeriques
# -------------------------------------------------
dt = 10.0               # pas de temps (secondes)
NN = int(jour*86400/dt)   # nombre de pas
print('Total simulation (jours):', dt * NN / 86400)

# Fonctions ombre /gravite/ pression radiation
def ombre(angle, rayon):
    return 0 if abs(angle) < math.asin(R/rayon) else 1

def gravite(r):
    return -G*M/(r*r)

def radiation1(A, B, H):
    return a_m/2.0*H*(np.cos(A)+ np.cos(B))

def radiation2(A, B, H):
    return a_m/2.0*H*(-np.sin(A) + np.sin(B))

def euler_implicite(u_n, s_n, r_n, phi_n, dt, n_iter=3):
    # Calcul des termes de radiation avec les valeurs au temps n
    H = ombre(phi_n, r_n)
    alpha_pred = math.atan2(s_n, u_n)
    rad1 = radiation1(phi_n, alpha_pred, H)
    rad2 = radiation2(phi_n, alpha_pred, H)
    # Initialisation avec les valeurs au temps n
    u_temp = u_n
    s_temp = s_n
    r_temp = r_n
    phi_temp = phi_n
    # Iterations de point fixe pour resoudre le systeme implicite
    for j in range(n_iter):
        # Mise a jour position et angle avec velocites temporaires
        r_temp = r_n + dt * u_temp
        phi_temp = phi_n + dt * s_temp / r_temp
        # Mise a jour des vitesses (schema implicite)
        u_temp = u_n + dt * (s_temp*s_temp/r_temp + gravite(r_temp) + rad1)
        s_temp = s_n + dt * (-u_temp*s_temp/r_temp + rad2)
    return u_temp, s_temp, r_temp, phi_temp
# valeurs de depart:
u_n = vr0                # vitesse radiale
s_n = vt0                # vitesse tangentielle
r_n = r0                 # distance radiale
phi_n = phi0             # angle polaire
alpha_n = alpha0         # angle de vol
t = 0.0                  #  secondes
flagE_n = 0              # flag test energie de liberation
print('Vitesse tangentielle initiale:', vt0)

# Boucle principale de simulation
for i in range(NN):
    # Calcul de l'energie
    E_n = u_n * u_n + s_n * s_n - 2.0 * G * M / r_n
    if E_n > 0 and flagE_n == 0:
        print('Temps liberation:', t / 3600, 'heures')
        flagE_n = 1
    # Application du schema d'Euler implicite
    u_n, s_n, r_n, phi_n = euler_implicite(u_n, s_n, r_n, phi_n, dt)
    # Normalisation de l'angle
    phi_n = phi_n % (2 * np.pi)
    # Calcul de l'angle de vol
    alpha_n = math.atan2(s_n, u_n)
    # Calcul des coordonnees cartesiennes (si necessaire)
    x_n, y_n = r_n*np.cos(phi_n), r_n*np.sin(phi_n)
    t += dt
    
    # Affichage progression tous les 10000 pas
    if i % 10000 == 0:
        print(f'Progression: {100*i/NN:.1f}%, r={r_n/R:.2f}R, E={E_n/1e6:.2f}MJ/kg')

print('Simulation terminee')
print(f'Distance finale: {r_n/R:.2f} rayons terrestres')
print(f'Energie finale: {E_n/1e6:.2f} MJ/kg')