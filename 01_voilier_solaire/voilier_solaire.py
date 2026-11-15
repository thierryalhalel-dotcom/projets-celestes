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

