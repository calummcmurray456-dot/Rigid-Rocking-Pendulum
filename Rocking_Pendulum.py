import numpy as np
import matplotlib.pyplot as plt

#rod length, half distance of crossbar to pivot centre of mass and gravity value
L=1.0
a=0.2
g=9.81

#moment of intertia at contact, small-osc angular freq and period
I_p=(1/12)*L**2 + a**2
om=np.sqrt(g*a / I_p)
T=2*np.pi / om

num_periods= 4
N=2000
t=np.linspace(0, num_periods*T, N)
t_scaled = t/T

# SC1 initial angular speed
om_0= 0.5
theta_1= (om_0/om)*np.sin(om*t)

# SC2 initial tilt from rest
theta_0= 0.1
theta_2 = theta_0*np.cos(om*t)

plt.rcParams.update({"font.size":18, "axes.labelsize": 18, "legend.fontsize": 15, "lines.linewidth": 1.5,})

# F1 θ(t) for SC1
fig1, ax1 = plt.subplots(figsize = (10,6))
ax1.plot(t_scaled, theta_1, color="navy")
crossings = np.arange(0, num_periods*T + 1e-9, T/2)
for tc in crossings:
    ax1.axvline(tc/T, color="black", linestyle=":", linewidth=0.7)

ax1.set_xlabel(r"$t\ /\ T$")
ax1.set_ylabel(r"$\theta(t) = (\omega_0\omega)\sin(\omega t)$")
ax1.legend([r"$\theta(t) = (\omega_0/\omega)\sin(\omega t)$"],loc="upper right", frameon=False, bbox_to_anchor=(1, 1))
fig1.tight_layout()
fig1.savefig("Rocking_Pendulum_Case1.pdf", dpi=1000, bbox_inches="tight")
plt.show()

# F2 θ(t) for SC2
fig2, ax2 = plt.subplots(figsize = (10,6))
ax1.plot(t_scaled, theta_2, color="crimson", label=r"$\theta(t) = \theta_0\cos(\omega t)$")

for tc in crossings:
        ax2.axvline(tc/T, color="black", linestyle=":", linewidth=0.7)

maxima= np.arange(0, num_periods*T +1e-9, T/2)
theta_max= theta_0*np.cos(om*maxima)
ax2.plot(maxima/T, theta_max, "o", color="red", markersize=3)
ax2.text(0.6, 0.85, r"successive maxima of $|\theta|$", transform=ax2.transAxes)
ax2.set_xlabel(r"$t\ /\ T$")
ax2.set_ylabel(r"$\theta(t)$ radians")
ax2.legend(loc="right", frameon=False)
fig2.tight_layout()
fig2.savefig("Rocking_Pendulum_Case2.pdf", dpi=1000, bbox_inches="tight")
plt.show()
