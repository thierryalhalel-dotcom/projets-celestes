# =================================================================
#  Projets celestes - Chapitre 1 : Une course de voiliers solaires
#  Exercice 1 : le meme probleme resolu par la methode d'Euler implicite
#
#  Le systeme implicite est resolu a chaque pas par quelques iterations
#  de point fixe (3 suffisent pour dt <= 10 s). Les noms des variables
#  et des fonctions sont ceux du livre ; la fonction euler_implicite et
#  la boucle principale sont celles imprimees dans l'exercice.
#
#  Comme dans tous les exercices du chapitre, dt = 10 s (dix fois le pas
#  du cours) : on compare le temps de liberation et la trajectoire avec
#  ceux du programme voilier_solaire.py (Euler explicite).
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
jour = 5.0
# -------------------------------------------------
# parametres
# -------------------------------------------------
r0 = 6.6316*R
u0 = 0.0                # vitesse radiale (m/s)
s0 = math.sqrt(G * M / r0)  # vitesse circulaire orbitale
phi0 = 3 * np.pi / 2     # angle polaire initial (rad)
alpha0 = np.pi / 2       # voile perpendiculaire rayons sol (rad)
m = 2000.0          #  masse (kg)
S_voile = 2.13e6          # surface voile (m^2)
a_m = 2 * S_voile * p / m   #  acceleration (m/s^2)
# -------------------------------------------------
#  parametres numeriques
# -------------------------------------------------
dt = 10.0               # pas de temps (s) : celui des exercices
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

def euler_implicite(u_n, s_n, r_n, phi_n, dt, n_iter=3):
	# Calcul des fonctions
	H = ombre(phi_n, r_n)
	alpha_pred = math.atan2(s_n, u_n)
	rad1 = radiation1(phi_n, alpha_pred, H)
	rad2 = radiation2(phi_n, alpha_pred, H)
	#  valeurs au temps n
	u_temp = u_n
	s_temp = s_n
	r_temp = r_n
	phi_temp = phi_n
	#  3 iterations de point fixe (dt < 10 s)
	for j in range(n_iter):
		# Mise a jour position et angle avec vitesses temporaires
		r_temp = r_n + dt * u_temp
		phi_temp = phi_n + dt * s_temp / r_temp
		# Mise a jour des vitesses (schema implicite)
		u_temp = u_n + dt * (s_temp*s_temp/r_temp + gravite(r_temp) + rad1)
		s_temp = s_n + dt * (-u_temp*s_temp/r_temp + rad2)
	return u_temp, s_temp, r_temp, phi_temp

#valeurs de depart:
u_n = u0                # vitesse radiale
s_n = s0                # vitesse tangentielle
r_n = r0                 # distance radiale
phi_n = phi0             # angle polaire
alpha_n = alpha0
t = 0.0
liberation_atteinte = 0              #  test energie de liberation
X, Y = [], []                        # pour le trace (ajout par rapport au livre)

# Boucle principale
for i in range(NN):
	# energie
	E_n = u_n * u_n + s_n * s_n - 2.0 * G * M / r_n
	if E_n > 0 and liberation_atteinte == 0:
		print('Temps liberation:', t / 3600, 'heures')
		liberation_atteinte = 1
	# schema d'Euler implicite
	u_n, s_n, r_n, phi_n = euler_implicite(u_n, s_n, r_n, phi_n, dt)
	# on calcule modulo 2PI
	phi_n = phi_n % (2 * np.pi)
	# orientation optimale
	alpha_n = math.atan2(s_n, u_n)
	# coordonnees cartesiennes pour trace
	x_n, y_n = r_n*np.cos(phi_n), r_n*np.sin(phi_n)
	X.append(x_n / R); Y.append(y_n / R)
	t += dt

	# Affichage
	if i % 10000 == 0:
		print(f'Progression: {100*i/NN:.1f}%, r={r_n/R:.2f}R, E={E_n/1e6:.2f}MJ/kg')
print('Simulation terminee')
print(f'Distance finale: {r_n/R:.2f} rayons terrestres')
print(f'Energie finale: {E_n/1e6:.2f} MJ/kg')

# -------------------------------------------------
#  trace de la trajectoire (ajout par rapport au livre)
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(X, Y, 'k-', lw=1, label='trajectoire')
ax.add_patch(plt.Circle((0, 0), 1.0, color='black', label='Terre'))
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
ax.set_xlabel('x (rayons terrestres)'); ax.set_ylabel('y (rayons terrestres)')
ax.set_title("Trajectoire du voilier (Euler implicite, dt = %g s)" % dt)
plt.show()
