# Baseline model uses the general rigid-body form (radius of gyration k, pivot distance ell),
# with specialisations to (i) rod-only (light cross-bar) and (ii) rod + massive cross-bar.
import os
import matplotlib
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "0" # set to 0 to show plots, 1 otherwise
if not SHOW_PLOTS:
    matplotlib.use("Agg")  # file-only (non-interactive)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Definitions

def omega_small(k2: float, ell: float, g: float = 9.81) -> float:
    """Small-angle angular frequency for the rocking pendulum.

    Model: theta_ddot + (g*ell/(k^2 + ell^2)) * theta = 0.
    """
    return np.sqrt(g * ell / (k2 + ell**2))


def period_small(k2: float, ell: float, g: float = 9.81) -> float:
    """Small-angle period."""
    return 2.0 * np.pi / omega_small(k2, ell, g)


def Icm_rod(m: float, L: float) -> float:
    """Moment of inertia of a uniform rod about its centre (axis out of plane)."""
    return (1.0 / 12.0) * m * L**2


def Icm_bar(M: float, a: float) -> float:
    """Moment of inertia of a uniform cross-bar of length 2a about its centre (axis out of plane)."""
    return (1.0 / 12.0) * M * (2.0 * a) ** 2  # = (1/3) M a^2


def k2_rod_only(m: float, L: float) -> float:
    """Radius of gyration squared k^2 = I_G/mu for rod-only model."""
    mu = m
    return Icm_rod(m, L) / mu


def k2_rod_plus_bar(m: float, L: float, M: float, a: float) -> float:
    """Radius of gyration squared k^2 for rod + massive cross-bar."""
    mu = m + M
    return (Icm_rod(m, L) + Icm_bar(M, a)) / mu


# Nonlinear extension: theta_ddot + omega0^2 sin(theta) = 0,
# where omega0^2 = g*ell/(k^2 + ell^2) (same as small-angle linearisation).

def simulate_nonlinear(theta0: float, omega0: float, tmax: float, dt: float):
    """Simulate nonlinear pendulum equation using a symplectic velocity-Verlet scheme."""
    n = int(np.ceil(tmax / dt)) + 1
    t = np.linspace(0.0, tmax, n)
    theta = np.empty(n)
    theta_dot = np.empty(n)

    theta[0] = theta0
    theta_dot[0] = 0.0

    def acc(th):
        return -(omega0**2) * np.sin(th)

    for i in range(n - 1):
        v_half = theta_dot[i] + 0.5 * dt * acc(theta[i])
        theta[i + 1] = theta[i] + dt * v_half
        theta_dot[i + 1] = v_half + 0.5 * dt * acc(theta[i + 1])

    return t, theta, theta_dot


def find_zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Times where y crosses zero, using linear interpolation."""
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    tz = []
    for i in idx:
        t0, t1 = t[i], t[i + 1]
        y0, y1 = y[i], y[i + 1]
        tz.append(t0 - y0 * (t1 - t0) / (y1 - y0))
    return np.asarray(tz)


def find_local_max_times(t: np.ndarray, y: np.ndarray):
    """Local maxima times for y(t) based on sign change of first difference."""
    dy = np.diff(y)
    idx = np.where((dy[:-1] > 0.0) & (dy[1:] < 0.0))[0] + 1
    return t[idx], y[idx]


# Parameters
L = 1.0 # rod length (m)
a = 0.2 # cross-bar half-length
ell = a # pivot distance (m)
m = 1.0 # rod mass (kg)
M_baseline = 0.5 # cross-bar mass (kg)
g = 9.81

# ICs (small oscillations)
omega_init = 0.5 # initial angular speed at equilibrium (rad/s)
theta0 = 0.1 # initial tilt angle (rad)

# Time resolution
num_periods = 4
N_time = 2000

# Sweep settings
M_over_m_max = 2.0
M_over_m_points = 250

# Extension settings
theta0_max_extension = 1.2 # rad
extension_points = 25


# Baseline choice (general model)
# Baseline: rod + massive cross-bar
k2_base = k2_rod_plus_bar(m=m, L=L, M=M_baseline, a=a)
omega_base = omega_small(k2_base, ell, g)
T_base = period_small(k2_base, ell, g)

print("Baseline parameters:")
print(f"  L = {L:.3g} m, a = {a:.3g} m (ell = a)")
print(f"  m = {m:.3g} kg, M = {M_baseline:.3g} kg")
print(f"  k^2 = {k2_base:.6g} m^2")
print(f"  omega = {omega_base:.6g} rad/s")
print(f"  T = {T_base:.6g} s,  T/2 = {0.5*T_base:.6g} s")


# Fig 1- small-angle solutions theta(t)
t = np.linspace(0.0, num_periods * T_base, N_time)
t_scaled = t / T_base

# Case 1- start at equilibrium with small angular speed omega_init
# theta(t) = (omega_init/omega) sin(omega t)
theta_case1 = (omega_init / omega_base) * np.sin(omega_base * t)

# Case 2- start from theta0 and release from rest
# theta(t) = theta0 cos(omega t)
theta_case2 = theta0 * np.cos(omega_base * t)

# Numerical verification (from analytic arrays):
zero_times = find_zero_crossings(t, theta_case2)
max_times, max_vals = find_local_max_times(t, theta_case2)

verification = {
    "T_small": T_base,
    "T_over_2": 0.5 * T_base,
    "first_zero_crossings_case2_s": zero_times[:8].tolist(),
    "first_maxima_case2_s": max_times[:8].tolist(),
}

pd.DataFrame([verification]).to_csv("verification_times.csv", index=False)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
})

fig1, ax1 = plt.subplots(figsize=(5.3, 3.2))
ax1.plot(t_scaled, theta_case1, color="navy", label=r"Case 1: $(\omega_0/\omega)\sin(\omega t)$")
ax1.plot(t_scaled, theta_case2, color="crimson", label=r"Case 2: $\theta_0\cos(\omega t)$")

# Mark equilibrium crossings every T/2
for tc in np.arange(0.0, num_periods * T_base + 1e-12, T_base / 2.0):
    ax1.axvline(tc / T_base, color="0.85", linestyle=":", linewidth=0.8)

ax1.set_xlabel(r"$t/T$")
ax1.set_ylabel(r"$\theta(t)$ [rad]")
ax1.legend(frameon=False,loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2)
fig1.subplots_adjust(bottom=0.28)

fig1.tight_layout()
fig1.savefig("fig1_timeseries.pdf", dpi=1000, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig1)


# Fig 2- sensitivity sweep, T vs M/m

M_over_m = np.linspace(0.0, M_over_m_max, M_over_m_points)
T_rod_only = np.full_like(M_over_m, period_small(k2_rod_only(m, L), ell, g))
T_rod_plus_bar = np.empty_like(M_over_m)

for i, r in enumerate(M_over_m):
    Mi = r * m
    k2_i = k2_rod_plus_bar(m=m, L=L, M=Mi, a=a)
    T_rod_plus_bar[i] = period_small(k2_i, ell, g)

sweep_df = pd.DataFrame({
    "M_over_m": M_over_m,
    "T_rod_only_s": T_rod_only,
    "T_rod_plus_bar_s": T_rod_plus_bar,
})

sweep_df.to_csv("sweep_T_vs_M_over_m.csv", index=False)

fig2, ax2 = plt.subplots(figsize=(5.3, 3.2))
ax2.plot(M_over_m, T_rod_plus_bar, color="black", label=r"Rod + cross-bar")
ax2.plot(M_over_m, T_rod_only, color="tab:blue", linestyle="--", label=r"Rod only")
ax2.axvline(M_baseline / m, color="tab:red", linestyle=":", linewidth=1.2, label="Baseline")

ax2.set_xlabel(r"Cross-bar mass ratio $M/m$")
ax2.set_ylabel(r"Small-angle period $T$ [s]")
ax2.legend(frameon=False, loc="best")
fig2.tight_layout()
fig2.savefig("fig2_sweep_T_vs_M_over_m.pdf", dpi=1000, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig2)


# Extension
# Fig 3- finite-amplitude period vs theta0

# Nonlinear period depends on amplitude. We estimate it by simulating
# theta_ddot + omega^2 sin(theta) = 0 and measuring time between successive maxima.

theta0_vals = np.linspace(0.05, theta0_max_extension, extension_points)
T_nonlinear = np.empty_like(theta0_vals)
T_small = T_base

for i, th0 in enumerate(theta0_vals):
    dt = T_small / 5000.0
    tmax = 12.0 * T_small
    t_sim, th_sim, thdot_sim = simulate_nonlinear(th0, omega_base, tmax=tmax, dt=dt)

    tmaxs, ymaxs = find_local_max_times(t_sim, th_sim)
    # ignore early numerical transients close to t=0
    tmaxs = tmaxs[tmaxs > 0.1 * T_small]

    if len(tmaxs) >= 2:
        # use median of first few differences for robustness
        T_est = np.median(np.diff(tmaxs[:6]))
    else:
        T_est = np.nan

    T_nonlinear[i] = T_est

ext_df = pd.DataFrame({
    "theta0_rad": theta0_vals,
    "T_nonlinear_s": T_nonlinear,
    "T_small_angle_s": np.full_like(theta0_vals, T_small),
    "ratio_T_over_Tsmall": T_nonlinear / T_small,
})

ext_df.to_csv("extension_period_vs_amplitude.csv", index=False)

fig3, ax3 = plt.subplots(figsize=(5.3, 3.2))
ax3.plot(theta0_vals, T_nonlinear / T_small, "o-", color="tab:red", markersize=3, label="Nonlinear (simulated)")
ax3.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Small-angle")
ax3.set_xlabel(r"Release angle $\theta_0$ [rad]")
ax3.set_ylabel(r"$T(\theta_0)/T_{\rm small}$")
ax3.legend(frameon=False, loc="upper left")
fig3.tight_layout()
fig3.savefig("fig3_extension_period_vs_amplitude.pdf", dpi=1000, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
plt.close(fig3)


# Save baseline time-series data for reproducibility

ts_df = pd.DataFrame({
    "t_s": t,
    "t_over_T": t_scaled,
    "theta_case1_rad": theta_case1,
    "theta_case2_rad": theta_case2,
})

ts_df.to_csv("timeseries_baseline.csv", index=False)

print("\nOutputs written:")
print("  fig1_timeseries.pdf")
print("  fig2_sweep_T_vs_M_over_m.pdf")
print("  fig3_extension_period_vs_amplitude.pdf")
print("  timeseries_baseline.csv")
print("  sweep_T_vs_M_over_m.csv")
print("  extension_period_vs_amplitude.csv")
print("  verification_times.csv")
