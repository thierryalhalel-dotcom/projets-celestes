#!/usr/bin/env python3
"""
Resolution numerique de l'equation differentielle y'' + 5y' + 100y = 0
Conditions initiales : y(0) = 1, y'(0) = -1
Methodes : Euler explicite, Euler implicite, Leap-Frog, Runge-Kutta 4
"""

import numpy as np
import matplotlib.pyplot as plt

# Parametres
dt = 0.01           # Pas de temps
t_max = 2.0         # Temps final
y0 = 1              # Condition initiale y(0)
v0 = -1             # Condition initiale y'(0)
gamma = 5           # Coefficient d'amortissement
omega0_squared = 100  # Coefficient de rappel

# Grille temporelle
t = np.arange(0, t_max + dt, dt)
n = len(t)

# Solution analytique exacte
# y'' + 5y' + 100y = 0
# Equation caracteristique : r^2 + 5r + 100 = 0
# Delta = 25 - 400 = -375 < 0 => Regime sous-amorti (pseudo-periodique)
# r = (-5 +/- i*sqrt(375)) / 2 = -2.5 +/- i*9.682...
# alpha = 2.5 (coefficient d'amortissement)
# omega_d = sqrt(375)/2 = 5*sqrt(15)/2 (pseudo-pulsation)

alpha = gamma / 2.0
omega_d = np.sqrt(omega0_squared - alpha**2)

# Conditions initiales : y(0) = 1, y'(0) = -1
# y(t) = e^(-alpha*t) * [A*cos(omega_d*t) + B*sin(omega_d*t)]
# y(0) = A = 1
# y'(0) = -alpha*A + omega_d*B = -1
# => omega_d*B = -1 + alpha*A = -1 + 2.5 = 1.5
# => B = 1.5 / omega_d

A = y0
B = (v0 + alpha * A) / omega_d

y_exact = np.exp(-alpha * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))


# ============================================================================
# METHODE D'EULER EXPLICITE
# ============================================================================
def euler_explicite(y0, v0, t, dt, gamma, omega0_squared):
    """
    Schema d'Euler explicite pour systeme d'ordre 1
    y' = v
    v' = -gamma * v - omega0_squared * y
    
    y(n+1) = y(n) + dt * v(n)
    v(n+1) = v(n) + dt * (-gamma * v(n) - omega0_squared * y(n))
    """
    n = len(t)
    y = np.zeros(n)
    v = np.zeros(n)
    y[0] = y0
    v[0] = v0
    
    for i in range(n - 1):
        y[i + 1] = y[i] + dt * v[i]
        v[i + 1] = v[i] + dt * (-gamma * v[i] - omega0_squared * y[i])
    
    return y, v


# ============================================================================
# METHODE D'EULER IMPLICITE
# ============================================================================
def euler_implicite(y0, v0, t, dt, gamma, omega0_squared):
    """
    Schema d'Euler implicite
    y(n+1) = y(n) + dt * v(n+1)
    v(n+1) = v(n) + dt * (-gamma * v(n+1) - omega0_squared * y(n+1))
    
    Resolution du systeme :
    y(n+1) - dt * v(n+1) = y(n)
    omega0_squared * dt * y(n+1) + (1 + gamma * dt) * v(n+1) = v(n)
    
    Forme matricielle :
    [1              -dt          ] [y(n+1)]   [y(n)]
    [omega0^2*dt   1+gamma*dt    ] [v(n+1)] = [v(n)]
    """
    n = len(t)
    y = np.zeros(n)
    v = np.zeros(n)
    y[0] = y0
    v[0] = v0
    
    # Matrice du systeme
    det = (1 + gamma * dt) + omega0_squared * dt**2
    
    for i in range(n - 1):
        # Resolution du systeme 2x2
        y[i + 1] = ((1 + gamma * dt) * y[i] + dt * v[i]) / det
        v[i + 1] = (-omega0_squared * dt * y[i] + v[i]) / det
    
    return y, v


# ============================================================================
# METHODE LEAP-FROG (Saute-mouton)
# ============================================================================
def leap_frog(y0, v0, t, dt, gamma, omega0_squared):
    """
    Schema Leap-Frog pour equations avec amortissement
    
    y(n+1) = y(n-1) + 2*dt * v(n)
    v(n+1) = v(n-1) + 2*dt * (-gamma * v(n) - omega0_squared * y(n))
    """
    n = len(t)
    y = np.zeros(n)
    v = np.zeros(n)
    y[0] = y0
    v[0] = v0
    
    # Premier pas avec Euler explicite pour obtenir y(1) et v(1)
    y[1] = y[0] + dt * v[0]
    v[1] = v[0] + dt * (-gamma * v[0] - omega0_squared * y[0])
    
    # Leap-Frog pour les autres pas
    for i in range(1, n - 1):
        y[i + 1] = y[i - 1] + 2 * dt * v[i]
        v[i + 1] = v[i - 1] + 2 * dt * (-gamma * v[i] - omega0_squared * y[i])
    
    return y, v


# ============================================================================
# METHODE RUNGE-KUTTA D'ORDRE 4 (RK4)
# ============================================================================
def runge_kutta_4(y0, v0, t, dt, gamma, omega0_squared):
    """
    Schema de Runge-Kutta d'ordre 4 pour systeme
    y' = v
    v' = -gamma * v - omega0_squared * y
    """
    n = len(t)
    y = np.zeros(n)
    v = np.zeros(n)
    y[0] = y0
    v[0] = v0
    
    for i in range(n - 1):
        # k1
        k1_y = v[i]
        k1_v = -gamma * v[i] - omega0_squared * y[i]
        
        # k2
        k2_y = v[i] + 0.5 * dt * k1_v
        k2_v = -gamma * (v[i] + 0.5 * dt * k1_v) - omega0_squared * (y[i] + 0.5 * dt * k1_y)
        
        # k3
        k3_y = v[i] + 0.5 * dt * k2_v
        k3_v = -gamma * (v[i] + 0.5 * dt * k2_v) - omega0_squared * (y[i] + 0.5 * dt * k2_y)
        
        # k4
        k4_y = v[i] + dt * k3_v
        k4_v = -gamma * (v[i] + dt * k3_v) - omega0_squared * (y[i] + dt * k3_y)
        
        # Mise a jour
        y[i + 1] = y[i] + (dt / 6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        v[i + 1] = v[i] + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    return y, v


# ============================================================================
# CALCUL DES SOLUTIONS
# ============================================================================
y_euler_exp, v_euler_exp = euler_explicite(y0, v0, t, dt, gamma, omega0_squared)
y_euler_imp, v_euler_imp = euler_implicite(y0, v0, t, dt, gamma, omega0_squared)
y_leapfrog, v_leapfrog = leap_frog(y0, v0, t, dt, gamma, omega0_squared)
y_rk4, v_rk4 = runge_kutta_4(y0, v0, t, dt, gamma, omega0_squared)


# ============================================================================
# AFFICHAGE DES RESULTATS
# ============================================================================
# Configuration generale des graphiques
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2.5

# Creation de la figure avec 3 sous-graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Resolution de l'equation y'' + 5y' + 100y = 0 avec y(0) = 1, y'(0) = -1", 
             fontsize=16, fontweight='bold')

# ============================================================================
# GRAPHIQUE 1 : Euler explicite et implicite vs Solution exacte
# ============================================================================
axes[0].plot(t, y_exact, 'k-', label='Solution exacte', linewidth=2.5)
axes[0].plot(t, y_euler_exp, 'b--', label='Euler explicite', linewidth=2.5)
axes[0].plot(t, y_euler_imp, 'r-.', label='Euler implicite', linewidth=2.5)
axes[0].set_xlabel('Temps t', fontweight='bold')
axes[0].set_ylabel('y(t)', fontweight='bold')
axes[0].set_title('Methodes d\'Euler', fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, t_max])

# ============================================================================
# GRAPHIQUE 2 : Leap-Frog vs Solution exacte
# ============================================================================
axes[1].plot(t, y_exact, 'k-', label='Solution exacte', linewidth=2.5)
axes[1].plot(t, y_leapfrog, 'g--', label='Leap-Frog', linewidth=2.5)
axes[1].set_xlabel('Temps t', fontweight='bold')
axes[1].set_ylabel('y(t)', fontweight='bold')
axes[1].set_title('Methode Leap-Frog', fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, t_max])

# ============================================================================
# GRAPHIQUE 3 : Runge-Kutta 4 vs Solution exacte
# ============================================================================
axes[2].plot(t, y_exact, 'k-', label='Solution exacte', linewidth=2.5)
axes[2].plot(t, y_rk4, 'm--', label='Runge-Kutta 4', linewidth=2.5)
axes[2].set_xlabel('Temps t', fontweight='bold')
axes[2].set_ylabel('y(t)', fontweight='bold')
axes[2].set_title('Methode Runge-Kutta 4', fontweight='bold')
axes[2].legend(loc='best', fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, t_max])

plt.tight_layout()
plt.savefig('oscillateur_amorti_solutions.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# CALCUL DE L'ENERGIE (pour equations dissipatives)
# ============================================================================
def compute_energy(y, v, omega0_squared):
    """
    Energie mecanique E = (1/2) * v^2 + (1/2) * omega0_squared * y^2
    Pour un systeme amorti, l'energie decroit exponentiellement
    """
    E = 0.5 * v**2 + 0.5 * omega0_squared * y**2
    return E

# Calcul de la vitesse exacte
v_exact = np.exp(-alpha * t) * (
    -alpha * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t)) +
    omega_d * (-A * np.sin(omega_d * t) + B * np.cos(omega_d * t))
)

# Energie pour chaque methode
E_exact = compute_energy(y_exact, v_exact, omega0_squared)
E_euler_exp = compute_energy(y_euler_exp, v_euler_exp, omega0_squared)
E_euler_imp = compute_energy(y_euler_imp, v_euler_imp, omega0_squared)
E_leapfrog = compute_energy(y_leapfrog, v_leapfrog, omega0_squared)
E_rk4 = compute_energy(y_rk4, v_rk4, omega0_squared)

# Energie theorique : E(t) = E0 * exp(-gamma * t)
E0_theory = 0.5 * v0**2 + 0.5 * omega0_squared * y0**2
E_theory = E0_theory * np.exp(-gamma * t)

# Graphique de l'energie
fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(t, E_theory, 'k-', label='Energie theorique (decroissance exp.)', linewidth=3)
ax.plot(t, E_exact, 'k--', label='Energie exacte (calculee)', linewidth=2.5, alpha=0.7)
ax.plot(t, E_euler_exp, 'b--', label='Euler explicite', linewidth=2.5)
ax.plot(t, E_euler_imp, 'r-.', label='Euler implicite', linewidth=2.5)
ax.plot(t, E_leapfrog, 'g:', label='Leap-Frog', linewidth=2.5)
ax.plot(t, E_rk4, 'm--', label='Runge-Kutta 4', linewidth=2.5)
ax.set_xlabel('Temps t', fontweight='bold')
ax.set_ylabel('Energie E', fontweight='bold')
ax.set_title('Dissipation de l\'energie pour y\'\' + 5y\' + 100y = 0', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('oscillateur_amorti_energie.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# GRAPHIQUE SUPPLEMENTAIRE : Enveloppe exponentielle
# ============================================================================
fig3, ax = plt.subplots(figsize=(12, 6))

# Enveloppe exponentielle
envelope_pos = np.exp(-alpha * t) * np.sqrt(A**2 + B**2)
envelope_neg = -envelope_pos

ax.plot(t, y_exact, 'k-', label='Solution exacte', linewidth=2.5)
ax.plot(t, envelope_pos, 'r--', label='Enveloppe exp(-2.5t)', linewidth=2, alpha=0.7)
ax.plot(t, envelope_neg, 'r--', linewidth=2, alpha=0.7)
ax.fill_between(t, envelope_neg, envelope_pos, alpha=0.1, color='red')

ax.set_xlabel('Temps t', fontweight='bold')
ax.set_ylabel('y(t)', fontweight='bold')
ax.set_title('Solution exacte et enveloppe exponentielle decroissante', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('oscillateur_amorti_enveloppe.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# GRAPHIQUE SUPPLEMENTAIRE : Portrait de phase
# ============================================================================
fig4, ax = plt.subplots(figsize=(10, 10))

ax.plot(y_exact, v_exact, 'k-', label='Solution exacte', linewidth=2.5)
ax.plot(y_euler_exp, v_euler_exp, 'b--', label='Euler explicite', linewidth=2, alpha=0.7)
ax.plot(y_euler_imp, v_euler_imp, 'r-.', label='Euler implicite', linewidth=2, alpha=0.7)
ax.plot(y_leapfrog, v_leapfrog, 'g:', label='Leap-Frog', linewidth=2, alpha=0.7)
ax.plot(y_rk4, v_rk4, 'm--', label='Runge-Kutta 4', linewidth=2, alpha=0.7)

# Point initial
ax.plot(y0, v0, 'ro', markersize=10, label='Point initial', zorder=5)

ax.set_xlabel('Position y', fontweight='bold')
ax.set_ylabel('Vitesse v = y\'', fontweight='bold')
ax.set_title('Portrait de phase : oscillateur amorti', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('oscillateur_amorti_portrait_phase.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# AFFICHAGE DES RESULTATS NUMERIQUES
# ============================================================================
print("=" * 70)
print("RESOLUTION DE L'EQUATION y'' + 5y' + 100y = 0")
print("Conditions initiales : y(0) = 1, y'(0) = -1")
print("=" * 70)
print(f"Pas de temps : dt = {dt}")
print(f"Intervalle : [0, {t_max}]")
print(f"Nombre de pas : {n}")
print(f"Coefficient d'amortissement : gamma = {gamma}")
print(f"Coefficient de rappel : omega0^2 = {omega0_squared}")
print(f"Alpha (taux d'amortissement) = {alpha:.6f}")
print(f"Pseudo-pulsation omega_d = {omega_d:.6f} rad/s")
print(f"Pseudo-periode T_d = 2*pi/omega_d = {2*np.pi/omega_d:.6f} s")
print(f"Temps de relaxation tau = 1/alpha = {1/alpha:.6f} s")
print(f"Facteur d'amortissement zeta = {gamma/(2*np.sqrt(omega0_squared)):.6f}")
print("=" * 70)
print("\nVALEURS A t = {:.1f} :".format(t_max))
print("-" * 70)
print(f"Solution exacte       : y({t_max}) = {y_exact[-1]:.10f}")
print(f"Euler explicite       : y({t_max}) = {y_euler_exp[-1]:.10f}")
print(f"Euler implicite       : y({t_max}) = {y_euler_imp[-1]:.10f}")
print(f"Leap-Frog             : y({t_max}) = {y_leapfrog[-1]:.10f}")
print(f"Runge-Kutta 4         : y({t_max}) = {y_rk4[-1]:.10f}")
print("=" * 70)
print("\nERREURS ABSOLUES A t = {:.1f} :".format(t_max))
print("-" * 70)
print(f"Euler explicite       : {abs(y_euler_exp[-1] - y_exact[-1]):.2e}")
print(f"Euler implicite       : {abs(y_euler_imp[-1] - y_exact[-1]):.2e}")
print(f"Leap-Frog             : {abs(y_leapfrog[-1] - y_exact[-1]):.2e}")
print(f"Runge-Kutta 4         : {abs(y_rk4[-1] - y_exact[-1]):.2e}")
print("=" * 70)
print("\nERREURS RELATIVES A t = {:.1f} (%) :".format(t_max))
print("-" * 70)
print(f"Euler explicite       : {100*abs(y_euler_exp[-1] - y_exact[-1])/abs(y_exact[-1]):.4f} %")
print(f"Euler implicite       : {100*abs(y_euler_imp[-1] - y_exact[-1])/abs(y_exact[-1]):.4f} %")
print(f"Leap-Frog             : {100*abs(y_leapfrog[-1] - y_exact[-1])/abs(y_exact[-1]):.4f} %")
print(f"Runge-Kutta 4         : {100*abs(y_rk4[-1] - y_exact[-1])/abs(y_exact[-1]):.4f} %")
print("=" * 70)
print("\nDISSIPATION DE L'ENERGIE :")
print("-" * 70)
E0_exact = E_exact[0]
E_theory_final = E0_exact * np.exp(-gamma * t_max)
print(f"Energie initiale E0            : {E0_exact:.6f}")
print(f"Energie theorique finale       : {E_theory_final:.6f}")
print(f"Dissipation theorique          : {100*(1 - np.exp(-gamma * t_max)):.2f} %")
print("-" * 70)
print(f"Euler explicite - finale       : {E_euler_exp[-1]:.6f}")
print(f"  Dissipation                  : {100*(E0_exact - E_euler_exp[-1])/E0_exact:.2f} %")
print(f"Euler implicite - finale       : {E_euler_imp[-1]:.6f}")
print(f"  Dissipation                  : {100*(E0_exact - E_euler_imp[-1])/E0_exact:.2f} %")
print(f"Leap-Frog - finale             : {E_leapfrog[-1]:.6f}")
print(f"  Dissipation                  : {100*(E0_exact - E_leapfrog[-1])/E0_exact:.2f} %")
print(f"Runge-Kutta 4 - finale         : {E_rk4[-1]:.6f}")
print(f"  Dissipation                  : {100*(E0_exact - E_rk4[-1])/E0_exact:.2f} %")
print("=" * 70)
print("\nERREUR SUR LA DISSIPATION D'ENERGIE (par rapport a la theorie) :")
print("-" * 70)
print(f"Euler explicite       : {abs(E_euler_exp[-1] - E_theory_final):.2e}")
print(f"Euler implicite       : {abs(E_euler_imp[-1] - E_theory_final):.2e}")
print(f"Leap-Frog             : {abs(E_leapfrog[-1] - E_theory_final):.2e}")
print(f"Runge-Kutta 4         : {abs(E_rk4[-1] - E_theory_final):.2e}")
print("=" * 70)
