# =================================================================
#  Projets celestes - Chapitre 1 : Une course de voiliers solaires
#  Programme du cours (paragraphes 1.7 et 1.8)
#
#  Mouvement d'un voilier solaire autour de la Terre, resolu par la
#  methode d'Euler explicite. Les trois parties du code imprimees dans
#  le livre (constantes, fonctions, boucle principale) sont reprises
#  a l'identique ; seuls ont ete ajoutes l'enregistrement des valeurs
#  dans la boucle et les traces de la fin du fichier.
#
#  Affiche le temps de liberation, puis trace :
#    - la trajectoire (cf. figure 1.4 du livre),
#    - l'energie reduite h au cours du temps (cf. figure 1.9),
#    - la distance au centre de la Terre (cf. figure 1.12).
# =================================================================
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
jour=5.0
# -------------------------------------------------
# parametres
# -------------------------------------------------
r0 =    6.6316*R
u0 = 0.0                # vitesse radiale (m/s)
s0 = math.sqrt(G * M / r0)  # vitesse circulaire orbitale
phi0 =  3 * np.pi / 2     # angle polaire initial (rad)
alpha0 = np.pi / 2       # voile perpendiculaire rayons sol (rad)
m = 2000.0          #  masse (kg)
S_voile= 2.13e6          # surface voile (m^2)
a_m = 2 * S_voile* p / m   #  acceleration (m/s^2)
# -------------------------------------------------
#  parametres numeriques
# -------------------------------------------------
dt = 1.0                # pas de temps (s)
NN = int(jour*86400/dt)   # nombre de pas
print('Total simulation  (jours):', dt * NN / 86400)

def ombre(angle, rayon):
	return 0 if abs(angle) < math.asin(R/rayon) else 1
def gravite(r):
	return -G*M/(r*r)
def radiation1(A,B,H):
	return  a_m/2.0*H*(np.cos(A)+ np.cos(B))
def radiation2(A,B,H):
	return  a_m/2.0*H*(-np.sin(A) + np.sin(B))

#valeurs de depart:
u_n = u0                # vitesse radiale
s_n = s0                # vitesse tangentielle
r_n = r0                 # distance radiale
phi_n = phi0             # angle polaire
alpha_n = alpha0
t = 0.0
liberation_atteinte = 0              #  test energie de liberation
# listes pour les traces (ajout par rapport au livre)
X, Y, T, E, D = [], [], [], [], []
# Schema euler explicite
for i in range(NN):
	E_n = u_n * u_n + s_n * s_n - 2.0 * G * M / r_n
	if E_n > 0 and liberation_atteinte == 0:
		print('Temps liberation (heure):', t / 3600)
		liberation_atteinte = 1
	H = ombre(phi_n, r_n)
	Du_n = s_n*s_n/r_n +gravite(r_n)+ radiation1(phi_n,alpha_n,H)
	Ds_n = -u_n*s_n/r_n + radiation2(phi_n,alpha_n,H)

	u_n += Du_n * dt
	s_n += Ds_n * dt
	phi_n += dt * s_n / r_n
	r_n += u_n * dt

	phi_n = phi_n %(2 * np.pi)
	alpha_n = math.atan2(s_n, u_n)
	x_n, y_n = r_n*np.cos(phi_n), r_n*np.sin(phi_n)
	t += dt
	# enregistrement (ajout par rapport au livre)
	if i % 60 == 0:                      # une valeur par minute suffit
		X.append(x_n / R); Y.append(y_n / R)
		T.append(t / 3600); E.append(E_n); D.append(r_n / R)

# -------------------------------------------------
#  traces (ajout par rapport au livre)
# -------------------------------------------------
X, Y, T, E, D = map(np.array, (X, Y, T, E, D))
pas5h = int(5 * 60)                      # un point toutes les 5 heures

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(X, Y, 'k-', lw=1, label='trajectoire')
ax.plot(X[::pas5h], Y[::pas5h], 'ro', ms=4, label='points (5 h)')
ax.add_patch(plt.Circle((0, 0), 1.0, color='black', label='Terre'))
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
ax.set_xlabel('x (rayons terrestres)'); ax.set_ylabel('y (rayons terrestres)')
ax.set_title("Trajectoire du voilier (Euler explicite, dt = %g s)" % dt)

fig2, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
a1.plot(T, E, 'r-'); a1.axhline(0, color='k', lw=0.8)
a1.set_ylabel('energie reduite h (SI)'); a1.grid(alpha=0.3)
a1.set_title('Energie reduite : la liberation est atteinte quand h = 0')
a2.plot(T, D, 'b-'); a2.grid(alpha=0.3)
a2.set_xlabel('temps (heures)'); a2.set_ylabel('distance (rayons terrestres)')
plt.tight_layout()
plt.show()
