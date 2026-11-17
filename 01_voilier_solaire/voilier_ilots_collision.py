# =================================================================
#  Projets celestes - Chapitre 1 : Une course de voiliers solaires
#  Paragraphe "Altitudes interdites" : ilots de collision (RK4)
#
#  Balaye un ensemble de distances initiales r0 et integre chaque
#  trajectoire par Runge-Kutta 4 (dt = 10 s). Certaines conditions
#  initiales conduisent a une collision avec la Terre : elles forment
#  des "ilots" tres etroits. Un second balayage, 50 fois plus fin,
#  mesure la largeur de chaque ilot.
#
#  Usage :
#    python3 voilier_ilots_collision.py          orbite 1 : phi0 = 0,   r0 de 5,01 a 6,20 RT
#    python3 voilier_ilots_collision.py 2        orbite 2 : phi0 = 270, r0 de 6,96 a 7,00 RT
#    python3 voilier_ilots_collision.py 1 5.8 6.2 0.001
#        choix libre : orbite, r0 min, r0 max, pas (en rayons terrestres)
#  Les figures 1.17 et 1.18 du livre utilisent les plages
#  5,8-6,2 RT (pas 0,001), 6,8-7,0 RT (pas 0,0005), 7,0-7,5 RT (pas 0,0005)
#  et 7,0-8,5 RT (pas 0,005).
#
#  Attention : le calcul est long. Le cas par defaut de l'orbite 2 demande
#  environ 6 minutes sur un ordinateur courant ; un balayage fin peut
#  prendre bien davantage (chaque trajectoire represente ~84 000 pas RK4).
# =================================================================
import sys
# -------------------------------------------------
#  Solar-sail three-body simulation (Earth-Sun-Sail)
#  Multiple trajectories for different initial altitudes
#  + determination de la largeur Delta_R de l'ilot de collision
#  Integrateur : Runge-Kutta ordre 4 (RK4 classique)
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

# -------------------------------------------------
#  Physical constants (SI)
# -------------------------------------------------
G    = 6.67430e-11   # gravitational constant (m^3 kg^-1 s^-2)
M    = 5.9722e24     # mass of the Earth (kg)
R    = 6371e3        # radius of the Earth (m)
p    = 4.6e-6        # solar radiation pressure at 1 AU (N/m^2)

# -------------------------------------------------
#  Sail and spacecraft parameters
# -------------------------------------------------
m_ship = 2000.0      # spacecraft mass (kg)
A_sail = 2.13e6      # sail area (m^2)
a_m    = 2 * A_sail * p / m_ship   # radiation acceleration factor (m/s^2)

# -------------------------------------------------
#  Numerical parameters
# -------------------------------------------------
dt    = 10.0                     # time step (s) - coarse scan
NN    = int(6500 * 130 / dt)     # integration steps per trajectory

# Choix de l'orbite et de la plage de r0 (ligne de commande)
orbite = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if orbite == 1:
    r0_min, r0_max, r0_step, phi0 = 5.01, 6.20, 0.05, 0.0
else:
    r0_min, r0_max, r0_step, phi0 = 6.96, 7.00, 0.005, 3 * np.pi / 2
if len(sys.argv) > 4:
    r0_min, r0_max, r0_step = map(float, sys.argv[2:5])
r0_min, r0_max, r0_step = r0_min * R, r0_max * R, r0_step * R

# Fixed initial conditions
vr0    = 0.0           # initial radial velocity (m/s)
alpha0 = np.pi / 2     # initial flight path angle (rad)

print('Total simulation time per trajectory (days):', dt * NN / 86400)
print('Coarse scan: r0 from {:.2f} to {:.2f} RT, step {:.3f} RT'.format(
    r0_min/R, r0_max/R, r0_step/R))

# -------------------------------------------------
#  Helper functions
# -------------------------------------------------

def shadow(phi, r):
    """Return 0 if in Earth's shadow, 1 otherwise."""
    if r > R and abs(phi) < math.asin(R / r):
        return 0
    return 1


def gravity(r):
    """Gravitational acceleration directed toward Earth (m/s^2)."""
    return -G * M / (r * r)


def radiation_radial(phi, alpha, H):
    """Radial component of radiation pressure acceleration."""
    return a_m / 2.0 * H * (np.cos(phi) + np.cos(alpha))


def radiation_tangential(phi, alpha, H):
    """Tangential component of radiation pressure acceleration."""
    return a_m / 2.0 * H * (-np.sin(phi) + np.sin(alpha))


def collision(r):
    """True if the craft has hit the Earth."""
    return r < R


def compute_derivatives(u, s, r, phi, alpha):
    """
    Compute time derivatives of the state [u, s, r, phi].

    Returns
    -------
    Du, Ds, Dr, Dphi : derivatives
    H                : shadow function at this state
    """
    H    = shadow(phi, r)
    Du   = s*s/r   + gravity(r) + radiation_radial(phi, alpha, H)
    Ds   = -u*s/r              + radiation_tangential(phi, alpha, H)
    Dr   = u
    Dphi = s / r
    return Du, Ds, Dr, Dphi, H


def rk4_step(u, s, r, phi, alpha, dt):
    """
    One classical RK4 step for the solar-sail equations of motion.

    State vector  y = [u, s, r, phi]  (radial vel, tangential vel,
    distance, polar angle).  alpha is recomputed from (u, s) at each
    sub-step so the sail orientation tracks the velocity direction.

    Four derivative evaluations per step (k1..k4), combined with
    Simpson weights 1-2-2-1.  Global error O(dt^4).

    Returns the same 6-tuple as the former euler_step for full
    drop-in compatibility with simulate_trajectory().

    Physical notes
    --------------
    * Shadow function H is re-evaluated at every sub-step position
      (r, phi), so the sail correctly switches off inside Earth shadow.
    * alpha is updated as atan2(s, u) at each sub-step, meaning the
      sail always faces the velocity direction — consistent with the
      flight-path-angle control law used elsewhere.
    * No guard needed for r->0: collision() is tested in the main loop
      before calling rk4_step, so r stays > R throughout.
    """
    def f(u_, s_, r_, phi_):
        """Derivatives vector F(y) — re-evaluates alpha and shadow."""
        alpha_ = math.atan2(s_, u_)
        Du, Ds, Dr, Dphi, _ = compute_derivatives(u_, s_, r_, phi_, alpha_)
        return Du, Ds, Dr, Dphi

    # k1 : slope at (t, y_n)
    k1u, k1s, k1r, k1p = f(u, s, r, phi)

    # k2 : slope at (t + dt/2, y_n + dt/2 * k1)
    k2u, k2s, k2r, k2p = f(u  + 0.5*dt*k1u,
                             s  + 0.5*dt*k1s,
                             r  + 0.5*dt*k1r,
                             phi + 0.5*dt*k1p)

    # k3 : slope at (t + dt/2, y_n + dt/2 * k2)
    k3u, k3s, k3r, k3p = f(u  + 0.5*dt*k2u,
                             s  + 0.5*dt*k2s,
                             r  + 0.5*dt*k2r,
                             phi + 0.5*dt*k2p)

    # k4 : slope at (t + dt, y_n + dt * k3)
    k4u, k4s, k4r, k4p = f(u  + dt*k3u,
                             s  + dt*k3s,
                             r  + dt*k3r,
                             phi + dt*k3p)

    # Weighted combination (Simpson 1/6 rule)
    u_new   = u   + dt/6.0 * (k1u + 2*k2u + 2*k3u + k4u)
    s_new   = s   + dt/6.0 * (k1s + 2*k2s + 2*k3s + k4s)
    r_new   = r   + dt/6.0 * (k1r + 2*k2r + 2*k3r + k4r)
    phi_new = phi + dt/6.0 * (k1p + 2*k2p + 2*k3p + k4p)

    H_n       = shadow(phi, r)          # shadow at time n (for diagnostics)
    alpha_new = math.atan2(s_new, u_new)
    return u_new, s_new, r_new, phi_new, alpha_new, H_n


# -------------------------------------------------
#  Single-trajectory simulator
# -------------------------------------------------

def simulate_trajectory(r0_initial, dt_sim=None, nn=None):
    """
    Simulate one solar-sail trajectory starting from r0_initial.

    Parameters
    ----------
    r0_initial : initial radial distance (m)
    dt_sim     : time step override (default: global dt)
    nn         : step count override (default: global NN)

    Returns
    -------
    dict with keys:
        X, Y            : trajectory in Earth radii
        liberation_time : hours to escape (None if not reached)
        collision       : bool
        r0_RT           : r0_initial / R
    """
    if dt_sim is None:
        dt_sim = dt
    if nn is None:
        nn = NN

    vt0     = math.sqrt(G * M / r0_initial)
    u_n     = vr0
    s_n     = vt0
    r_n     = r0_initial
    phi_n   = phi0
    alpha_n = alpha0
    t       = 0.0
    flagE   = 0
    liberation_time  = None
    collision_flag   = False

    X, Y = [], []

    for _ in range(nn):
        # Energy and liberation check
        E_n = u_n**2 + s_n**2 - 2.0 * G * M / r_n
        if E_n > 0 and flagE == 0:
            liberation_time = t / 3600.0
            flagE = 1

        # Collision test
        if collision(r_n):
            collision_flag = True
            break

        # One Euler step
        u_n, s_n, r_n, phi_n, alpha_n, _ = rk4_step(u_n, s_n, r_n, phi_n, alpha_n, dt_sim)

        phi_n = phi_n % (2 * np.pi)

        X.append(r_n * np.cos(phi_n) / R)
        Y.append(r_n * np.sin(phi_n) / R)

        t += dt_sim

    return {
        'X': X,
        'Y': Y,
        'liberation_time': liberation_time,
        'collision': collision_flag,
        'r0_RT': r0_initial / R,
    }


# =================================================================
#  PHASE 1 : Coarse scan  (step = r0_step)
# =================================================================
r0_values    = np.arange(r0_min, r0_max + r0_step / 2, r0_step)
num_traj     = len(r0_values)
trajectories = []
liberation_times    = []
initial_distances   = []

print('\n--- Phase 1 : coarse scan ({} trajectories) ---'.format(num_traj))

for idx, r0_cur in enumerate(r0_values):
    result = simulate_trajectory(r0_cur)
    trajectories.append(result)

    status = 'COLLISION' if result['collision'] else (
        'liberation {:.1f} h'.format(result['liberation_time'])
        if result['liberation_time'] else 'no liberation')
    print('  [{:2d}/{}] r0 = {:.3f} RT  ->  {}'.format(
        idx + 1, num_traj, r0_cur / R, status))

    if result['liberation_time'] is not None:
        liberation_times.append(result['liberation_time'])
        initial_distances.append(r0_cur / R)

# Identify coarse collision points and surrounding intervals
# Each isolated collision point becomes an interval centred on it,
# padded by 2*r0_step on each side so the fine scan always brackets
# the true boundary even when only one coarse point shows collision.
collision_indices = [idx for idx, traj in enumerate(trajectories)
                     if traj['collision']]

collision_intervals = []
if collision_indices:
    # Group consecutive collision indices into intervals
    group_start = collision_indices[0]
    group_end   = collision_indices[0]
    for idx in collision_indices[1:]:
        if idx == group_end + 1:
            group_end = idx
        else:
            collision_intervals.append(
                (r0_values[group_start], r0_values[group_end]))
            group_start = group_end = idx
    collision_intervals.append(
        (r0_values[group_start], r0_values[group_end]))

print('\nCoarse collision interval(s) detected:')
if collision_intervals:
    for lo, hi in collision_intervals:
        print('  [{:.3f} – {:.3f}] RT'.format(lo / R, hi / R))
else:
    print('  None')


# =================================================================
#  PHASE 2 : Fine scan inside each collision interval
#  IMPORTANT : keep dt_fine == dt  (Euler trajectories depend on dt)
# =================================================================
dt_fine      = dt             # same dt as coarse scan
r0_step_fine = r0_step / 50  # 50x finer spatial resolution

fine_results = []   # list of (r0_RT, collision_bool)

print('\n--- Phase 2 : fine scan (step = {:.5f} RT, dt = {} s) ---'.format(
    r0_step_fine / R, dt_fine))

for lo, hi in collision_intervals:
    # Pad by 2 coarse steps on each side to bracket the true boundary
    lo_ext  = max(r0_min, lo - 2 * r0_step)
    hi_ext  = min(r0_max, hi + 2 * r0_step)
    fine_r0 = np.arange(lo_ext, hi_ext + r0_step_fine / 2, r0_step_fine)

    print('  Scanning [{:.4f} – {:.4f}] RT ({} points)'.format(
        lo_ext / R, hi_ext / R, len(fine_r0)))

    for r0_cur in fine_r0:
        result = simulate_trajectory(r0_cur, dt_sim=dt_fine)
        fine_results.append((r0_cur / R, result['collision']))

# Extract collision boundaries at fine resolution
fine_collision_r0 = [r0 for r0, col in fine_results if col]

Delta_R      = None
Delta_R_frac = None

if fine_collision_r0:
    r_col_min = min(fine_collision_r0)
    r_col_max = max(fine_collision_r0)
    Delta_R      = (r_col_max - r_col_min) * R    # in metres
    Delta_R_RT   = r_col_max - r_col_min           # in Earth radii
    Delta_R_frac = Delta_R_RT / (r0_max / R) * 100  # % of scan range
    Delta_R_pct  = Delta_R / R * 100               # % of Earth radius

    print('\n=== Collision island (fine resolution) ===')
    print('  r0 range : [{:.4f} – {:.4f}] RT'.format(r_col_min, r_col_max))
    print('  ΔR        = {:.2f} km  ({:.4f} RT)'.format(
        Delta_R / 1e3, Delta_R_RT))
    print('  ΔR / R    = {:.4f} %'.format(Delta_R_pct))
else:
    print('\nNo collision detected in fine scan.')


# =================================================================
#  PLOTS
# =================================================================

# ---- Figure 1 : All trajectories (coarse scan) ----
fig1, ax1 = plt.subplots(figsize=(12, 12))

non_collision = [t for t in trajectories if not t['collision'] and t['X']]
cmap = plt.get_cmap('viridis')

for k, traj in enumerate(non_collision):
    color = cmap(k / max(1, len(non_collision) - 1))
    ax1.plot(traj['X'], traj['Y'], color=color, linewidth=1.0, alpha=0.7)

for traj in trajectories:
    if traj['collision'] and traj['X']:
        ax1.plot(traj['X'], traj['Y'], 'r-', linewidth=1.5, alpha=0.8)

ax1.add_patch(plt.Circle((0, 0), 1.0, color='black', fill=True, zorder=10, label='Terre'))
ax1.set_xlabel('x (rayon Terre)', fontsize=12)
ax1.set_ylabel('y (rayon Terre)', fontsize=12)
ax1.set_title(
    'Trajectoires voilier solaire (RK4 ordre 4)\n'
    'r0={:.2f}-{:.2f} RT, pas={:.3f} RT, dt={:.0f} s'.format(
        r0_min/R, r0_max/R, r0_step/R, dt), fontsize=13)
ax1.axis('equal')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=r0_min/R, vmax=r0_max/R))
sm.set_array([])
plt.colorbar(sm, ax=ax1, label='Distance initiale r0 (rayons terrestres)')
plt.tight_layout()
plt.show()

# ---- Figure 2 : Liberation time vs r0 ----
if liberation_times:
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(initial_distances, liberation_times, 'b-o', linewidth=2, markersize=7,
             label='Temps de libération')

    # Mark collision zone
    for lo, hi in collision_intervals:
        ax2.axvspan(lo/R, hi/R, alpha=0.2, color='red', label='Zone collision (scan grossier)')

    ax2.set_xlabel('Distance initiale r0 (rayons terrestres)', fontsize=12)
    ax2.set_ylabel('Temps de libération (heures)', fontsize=12)
    ax2.set_title('Temps de libération en fonction de r0', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# ---- Figure 3 : Fine-scan collision map ----
if fine_results:
    r0_fine_arr = np.array([r for r, _ in fine_results])
    col_arr     = np.array([int(c) for _, c in fine_results])

    fig3, ax3 = plt.subplots(figsize=(13, 4))
    ax3.scatter(r0_fine_arr, col_arr, c=col_arr, cmap='RdYlGn_r',
                s=30, zorder=5, label='0=libéré, 1=collision')

    if Delta_R is not None:
        ax3.axvspan(r_col_min, r_col_max, alpha=0.15, color='red')
        ax3.annotate(
            '', xy=(r_col_max, 0.5), xytext=(r_col_min, 0.5),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax3.text((r_col_min + r_col_max) / 2, 0.65,
                 'ΔR = {:.2f} km\nΔR/R = {:.4f} %'.format(Delta_R/1e3, Delta_R_pct),
                 ha='center', fontsize=11, color='red',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='red', alpha=0.9))

    ax3.set_xlabel('Distance initiale r0 (rayons terrestres)', fontsize=12)
    ax3.set_ylabel('Collision (1 = oui)', fontsize=12)
    ax3.set_title('Carte de collision (scan fin, pas = {:.5f} RT, dt = {} s)'.format(
        r0_step_fine/R, dt_fine), fontsize=12)  # RK4 ordre 4
    ax3.set_ylim(-0.15, 0.95)
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# =================================================================
#  BILAN FINAL
# =================================================================
print('\n' + '='*60)
print('BILAN FINAL')
print('='*60)
print('Trajectoires simulées (scan grossier) : {}'.format(num_traj))
print('  dont avec libération : {}'.format(len(liberation_times)))
print('  dont avec collision  : {}'.format(
    sum(1 for t in trajectories if t['collision'])))

if liberation_times:
    idx_min = liberation_times.index(min(liberation_times))
    idx_max = liberation_times.index(max(liberation_times))
    print('\nTemps de libération :')
    print('  Min : {:.2f} h  à r0 = {:.3f} RT'.format(
        min(liberation_times), initial_distances[idx_min]))
    print('  Max : {:.2f} h  à r0 = {:.3f} RT'.format(
        max(liberation_times), initial_distances[idx_max]))
    print('  Moy : {:.2f} h'.format(
        sum(liberation_times) / len(liberation_times)))

if Delta_R is not None:
    print('\nIlot de collision (scan fin) :')
    print('  Intervalle : [{:.4f} – {:.4f}] RT'.format(r_col_min, r_col_max))
    print('  ΔR         = {:.3f} km'.format(Delta_R / 1e3))
    print('  ΔR / R     = {:.4f} %'.format(Delta_R_pct))
    print('  ΔR (RT)    = {:.5f} rayons terrestres'.format(Delta_R_RT))
else:
    print('\nAucun ilot de collision détecté.')

print('='*60)
