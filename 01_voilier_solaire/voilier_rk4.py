# =================================================================
#  Projets celestes - Chapitre 1 : Une course de voiliers solaires
#  Exercice 2 : le meme probleme resolu par la methode de Runge-Kutta 4
#
#  La fonction runge_kutta_4 est celle imprimee dans l'exercice du livre.
#  Le programme detecte aussi une eventuelle collision avec la Terre
#  (r < R), ce qui sert dans l'exercice sur les "ilots" de collision.
#
#  Usage :
#    python3 voilier_rk4.py                 cas du cours : r0 = 6,6316 RT, phi0 = 270 deg
#    python3 voilier_rk4.py 6.01 0          orbite 1 de collision (figure 1.15)
#    python3 voilier_rk4.py 6.98 270        orbite 2 de collision (figure 1.16)
#  Arguments facultatifs : r0 en rayons terrestres, phi0 en degres,
#  duree simulee en jours (5 par defaut ; 9,3 pour les figures 1.15-1.16).
# =================================================================
import sys
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
# -------------------------------------------------
# parametres (modifiables en ligne de commande)
# -------------------------------------------------
r0_RT    = float(sys.argv[1]) if len(sys.argv) > 1 else 6.6316
phi0_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 270.0
jour     = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
r0 = r0_RT * R
u0 = 0.0                # vitesse radiale (m/s)
s0 = math.sqrt(G * M / r0)  # vitesse circulaire orbitale
phi0 = math.radians(phi0_deg)   # angle polaire initial (rad)
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
	# garde ajoutee : sous la surface (collision) arcsin(R/rayon) n'existe pas ;
	# la collision est detectee dans la boucle principale (r < R)
	if rayon <= R:
		return 1
	return 0 if abs(angle) < math.asin(R/rayon) else 1
def gravite(r):
	return -G*M/(r*r)
def radiation1(A,B,H):
	return  a_m/2.0*H*(np.cos(A)+ np.cos(B))
def radiation2(A,B,H):
	return  a_m/2.0*H*(-np.sin(A) + np.sin(B))

def runge_kutta_4(u_n, s_n, r_n, phi_n, dt):
	#  etape k1  au temps: n
	H1 = ombre(phi_n, r_n)
	alpha1 = math.atan2(s_n, u_n)
	rad1_1 = radiation1(phi_n, alpha1, H1)
	rad2_1 = radiation2(phi_n, alpha1, H1)
	k1_u = s_n * s_n / r_n + gravite(r_n) + rad1_1
	k1_s = -u_n * s_n / r_n + rad2_1
	k1_r = u_n
	k1_phi = s_n / r_n
	#  etape k2 au temps :n + dt/2
	u_temp2 = u_n + 0.5 * dt * k1_u
	s_temp2 = s_n + 0.5 * dt * k1_s
	r_temp2 = r_n + 0.5 * dt * k1_r
	phi_temp2 = phi_n + 0.5 * dt * k1_phi
	H2 = ombre(phi_temp2, r_temp2)
	alpha2 = math.atan2(s_temp2, u_temp2)
	rad1_2 = radiation1(phi_temp2, alpha2, H2)
	rad2_2 = radiation2(phi_temp2, alpha2, H2)
	k2_u = s_temp2 * s_temp2 / r_temp2 + gravite(r_temp2) + rad1_2
	k2_s = -u_temp2 * s_temp2 / r_temp2 + rad2_2
	k2_r = u_temp2
	k2_phi = s_temp2 / r_temp2
	#etape k3 au temps n+dt/2  avec k2
	u_temp3 = u_n + 0.5 * dt * k2_u
	s_temp3 = s_n + 0.5 * dt * k2_s
	r_temp3 = r_n + 0.5 * dt * k2_r
	phi_temp3 = phi_n + 0.5 * dt * k2_phi
	H3 = ombre(phi_temp3, r_temp3)
	alpha3 = math.atan2(s_temp3, u_temp3)
	rad1_3 = radiation1(phi_temp3, alpha3, H3)
	rad2_3 = radiation2(phi_temp3, alpha3, H3)
	k3_u = s_temp3 * s_temp3 / r_temp3 + gravite(r_temp3) + rad1_3
	k3_s = -u_temp3 * s_temp3 / r_temp3 + rad2_3
	k3_r = u_temp3
	k3_phi = s_temp3 / r_temp3
	# etape k4 au temps n + dt avec k3
	u_temp4 = u_n + dt * k3_u
	s_temp4 = s_n + dt * k3_s
	r_temp4 = r_n + dt * k3_r
	phi_temp4 = phi_n + dt * k3_phi
	H4 = ombre(phi_temp4, r_temp4)
	alpha4 = math.atan2(s_temp4, u_temp4)
	rad1_4 = radiation1(phi_temp4, alpha4, H4)
	rad2_4 = radiation2(phi_temp4, alpha4, H4)
	k4_u = s_temp4 * s_temp4 / r_temp4 + gravite(r_temp4) + rad1_4
	k4_s = -u_temp4 * s_temp4 / r_temp4 + rad2_4
	k4_r = u_temp4
	k4_phi = s_temp4 / r_temp4
	#combinaison / ponderation
	u_np1 = u_n + (dt / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
	s_np1 = s_n + (dt / 6.0) * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
	r_np1 = r_n + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
	phi_np1 = phi_n + (dt / 6.0) * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi)

	return u_np1, s_np1, r_np1, phi_np1

#valeurs de depart:
u_n = u0                # vitesse radiale
s_n = s0                # vitesse tangentielle
r_n = r0                 # distance radiale
phi_n = phi0             # angle polaire
t = 0.0
liberation_atteinte = 0              #  test energie de liberation
X, Y = [], []

# Boucle principale
for i in range(NN):
	E_n = u_n * u_n + s_n * s_n - 2.0 * G * M / r_n
	if E_n > 0 and liberation_atteinte == 0:
		print('Temps liberation:', t / 3600, 'heures')
		liberation_atteinte = 1
	u_n, s_n, r_n, phi_n = runge_kutta_4(u_n, s_n, r_n, phi_n, dt)
	phi_n = phi_n % (2 * np.pi)
	X.append(r_n*np.cos(phi_n) / R); Y.append(r_n*np.sin(phi_n) / R)
	t += dt
	if r_n < R:                                   # collision avec la Terre
		print('Collision avec la Terre apres %.1f heures' % (t / 3600))
		break
print(f'Distance finale: {r_n/R:.2f} rayons terrestres')

# -------------------------------------------------
#  trace de la trajectoire
# -------------------------------------------------
X, Y = np.array(X), np.array(Y)
pas5h = int(5 * 3600 / dt)                        # un point toutes les 5 heures
fig, ax = plt.subplots(figsize=(6, 7))
ax.plot(X, Y, 'k-', lw=1, label='trajectoire')
ax.plot(X[::pas5h], Y[::pas5h], 'ro', ms=4, label='points (5 h)')
ax.add_patch(plt.Circle((0, 0), 1.0, color='black', label='Terre'))
ax.plot(r0_RT*np.cos(phi0), r0_RT*np.sin(phi0), 'g^', ms=8, label='depart $r_0$')
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
ax.set_xlabel('x (rayons terrestres)'); ax.set_ylabel('y (rayons terrestres)')
ax.set_title('RK4, dt = %g s : $r_0$ = %.3f RT, $\\varphi_0$ = %g deg' % (dt, r0_RT, phi0_deg))
plt.show()
