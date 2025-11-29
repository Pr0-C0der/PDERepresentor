import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
nu = 0.1                  # viscosity
nx = 100                  # spatial points
x0, x1 = 0.0, 2.0 * np.pi
L = x1 - x0
dx = L / nx
x = np.linspace(x0, x1 - dx, nx)   # periodic grid (exclude endpoint so roll works cleanly)

t0, t1 = 0.0, 0.1
nt = 100                  # number of time points
t = np.linspace(t0, t1, nt)
dt = t[1] - t[0]

print(f"nx={nx}, nt={nt}, dx={dx:.4e}, dt={dt:.4e}")

# ---------------- Initial condition ----------------
u = np.zeros((nx, nt))        # columns: time index, rows: space
u[:, 0] = np.sin(x)           # u(x,0) = sin(x)

# ---------------- Finite-difference operators (periodic via np.roll) ----------------
# central difference for first derivative: ux_i = (u_{i+1} - u_{i-1}) / (2 dx)
# central second derivative: uxx_i = (u_{i+1} - 2 u_i + u_{i-1}) / dx^2

# Time-stepping: forward Euler (explicit)
for n in range(nt - 1):
    u_n = u[:, n]

    # periodic neighbors via roll
    u_ip1 = np.roll(u_n, -1)   # i+1
    u_im1 = np.roll(u_n, 1)    # i-1

    ux = (u_ip1 - u_im1) / (2.0 * dx)
    uxx = (u_ip1 - 2.0 * u_n + u_im1) / (dx * dx)

    # nonlinear term: - u * ux
    adv_term = - u_n * ux

    # rhs
    rhs = adv_term + nu * uxx

    # advance
    u[:, n + 1] = u_n + dt * rhs

# ---------------- Plotting & saving ----------------
# Prepare U for plotting with shape (nt, nx): rows=time, columns=space
U_Tx = u.T   # shape (nt, nx)

# Meshgrid for surface: both shape (nt, nx)
Xgrid, Tgrid = np.meshgrid(x, t)

# 1) Surface plot
ds_x = max(1, nx // 120)
ds_t = max(1, nt // 120)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Xgrid[::ds_t, ::ds_x], Tgrid[::ds_t, ::ds_x], U_Tx[::ds_t, ::ds_x],
                linewidth=0, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Burgers 1D: u(x,t)')
plt.tight_layout()
fig.savefig('burgers_surface.png', dpi=200)
plt.close(fig)
print("Saved burgers_surface.png")

# 2) Heatmap (rows=time, columns=space)
fig2, ax2 = plt.subplots(figsize=(8, 5))
im = ax2.imshow(U_Tx, aspect='auto', origin='lower',
                extent=[x0, x1 - dx, t0, t1],
                interpolation='nearest')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title('Heatmap of u(x,t)')
cb = fig2.colorbar(im, ax=ax2)
cb.set_label('u')
plt.tight_layout()
fig2.savefig('burgers_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved burgers_heatmap.png")

# 3) Profiles at selected times
times_to_plot = [0.0, 0.02, 0.05, 0.1]
idxs = [np.argmin(np.abs(t - tt)) for tt in times_to_plot]

fig3, ax3 = plt.subplots()
for idx in idxs:
    ax3.plot(x, u[:, idx], label=f"t={t[idx]:.3f}")
ax3.set_xlabel('x')
ax3.set_ylabel('u')
ax3.set_title('Burgers: snapshots at selected times')
ax3.legend()
plt.tight_layout()
fig3.savefig('burgers_profiles.png', dpi=200)
plt.close(fig3)
print("Saved burgers_profiles.png")

# ---------------- Quick diagnostics ----------------
print("Initial mean (should be ~0):", np.mean(u[:, 0]))
print("Final min, max, mean:", u[:, -1].min(), u[:, -1].max(), np.mean(u[:, -1]))
