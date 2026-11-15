#!/usr/bin/env python3
"""
Resolution numerique de l'equation differentielle y' + 3y = 0
Condition initiale : y(0) = 7
Methodes : Euler explicite, Euler implicite, Leap-Frog, Runge-Kutta 4
"""

import numpy as np
import matplotlib.pyplot as plt

# Parametres
dt = 0.1           # Pas de temps
t_max = 1.0         # Temps final
y0 = 7              # Condition initiale
alpha = 3           # Coefficient de l'equation y' + alpha*y = 0

# Grille temporelle
t = np.arange(0, t_max + dt, dt)
n = len(t)

# Solution analytique exacte : y(t) = y0 * exp(-alpha * t)
y_exact = y0 * np.exp(-alpha * t)


# ============================================================================
# METHODE D'EULER EXPLICITE
# ============================================================================
def euler_explicite(y0, t, dt, alpha):
    """
    Schema d'Euler explicite : y(n+1) = y(n) + dt * f(y(n))
    Avec f(y) = -alpha * y
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    
    for i in range(n - 1):
        y[i + 1] = y[i] + dt * (-alpha * y[i])
    
    return y


# ============================================================================
# METHODE D'EULER IMPLICITE
# ============================================================================
def euler_implicite(y0, t, dt, alpha):
    """
    Schema d'Euler implicite : y(n+1) = y(n) + dt * f(y(n+1))
    Avec f(y) = -alpha * y
    
    On obtient : y(n+1) = y(n) + dt * (-alpha * y(n+1))
                 y(n+1) * (1 + dt * alpha) = y(n)
                 y(n+1) = y(n) / (1 + dt * alpha)
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    
    # Facteur de recurrence
    factor = 1 / (1 + dt * alpha)
    
    for i in range(n - 1):
        y[i + 1] = y[i] * factor
    
    return y


# ============================================================================
# METHODE LEAP-FROG (Saute-mouton)
# ============================================================================
def leap_frog(y0, t, dt, alpha):
    """
    Schema Leap-Frog : y(n+1) = y(n-1) + 2*dt * f(y(n))
    
    Necessite deux valeurs initiales : y(0) et y(1)
    On utilise Euler explicite pour calculer y(1)
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    
    # Premier pas avec Euler explicite
    y[1] = y[0] + dt * (-alpha * y[0])
    
    # Leap-Frog pour les autres pas
    for i in range(1, n - 1):
        y[i + 1] = y[i - 1] + 2 * dt * (-alpha * y[i])
    
    return y


# ============================================================================
# METHODE RUNGE-KUTTA D'ORDRE 4 (RK4)
# ============================================================================
def runge_kutta_4(y0, t, dt, alpha):
    """
    Schema de Runge-Kutta d'ordre 4
    Avec f(y) = -alpha * y
    """
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    
    for i in range(n - 1):
        k1 = -alpha * y[i]
        k2 = -alpha * (y[i] + 0.5 * dt * k1)
        k3 = -alpha * (y[i] + 0.5 * dt * k2)
        k4 = -alpha * (y[i] + dt * k3)
        
        y[i + 1] = y[i] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y


# ============================================================================
# CALCUL DES SOLUTIONS
# ============================================================================
y_euler_exp = euler_explicite(y0, t, dt, alpha)
y_euler_imp = euler_implicite(y0, t, dt, alpha)
y_leapfrog = leap_frog(y0, t, dt, alpha)
y_rk4 = runge_kutta_4(y0, t, dt, alpha)


# ============================================================================
# AFFICHAGE DES RESULTATS
# ============================================================================
# Configuration generale des graphiques
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2.5  # Toutes les courbes en gras

# Creation de la figure avec 3 sous-graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Resolution de l'equation y' + 3y = 0 avec y(0) = 7", 
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
plt.show()


# ============================================================================
# AFFICHAGE DES ERREURS A t = 1
# ============================================================================
print("=" * 70)
print("RESOLUTION DE L'EQUATION y' + 3y = 0 avec y(0) = 7")
print("=" * 70)
print(f"Pas de temps : dt = {dt}")
print(f"Intervalle : [0, {t_max}]")
print(f"Nombre de pas : {n}")
print("=" * 70)
print("\nVALEURS A t = 1.0 :")
print("-" * 70)
print(f"Solution exacte       : y(1) = {y_exact[-1]:.10f}")
print(f"Euler explicite       : y(1) = {y_euler_exp[-1]:.10f}")
print(f"Euler implicite       : y(1) = {y_euler_imp[-1]:.10f}")
print(f"Leap-Frog             : y(1) = {y_leapfrog[-1]:.10f}")
print(f"Runge-Kutta 4         : y(1) = {y_rk4[-1]:.10f}")
print("=" * 70)
print("\nERREURS ABSOLUES A t = 1.0 :")
print("-" * 70)
print(f"Euler explicite       : {abs(y_euler_exp[-1] - y_exact[-1]):.2e}")
print(f"Euler implicite       : {abs(y_euler_imp[-1] - y_exact[-1]):.2e}")
print(f"Leap-Frog             : {abs(y_leapfrog[-1] - y_exact[-1]):.2e}")
print(f"Runge-Kutta 4         : {abs(y_rk4[-1] - y_exact[-1]):.2e}")
print("=" * 70)