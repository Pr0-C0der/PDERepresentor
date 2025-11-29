import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------- PARAMETERS (given) -------
D = 0.1
h = 1.0
X_env = 0.2
X0 = 1.0

# spatial domain
L = 1.0
Nz = 101
z = np.linspace(0, L, Nz)
dz = z[1] - z[0]

# time domain
t_max = 1.0
Nt = 400
t = np.linspace(0, t_max, Nt)
dt = t[1] - t[0]

print(f"dz = {dz:.4e}, dt = {dt:.4e}, alpha = D*dt/dz^2 = {D*dt/dz**2:.4f}")

# initialize
u = np.zeros((Nz, Nt))
u[:, 0] = X0
alpha = D * dt / dz**2

def build_tridiagonal(Nz, alpha, dt, D, dz, h, X_env):
    a = np.zeros(Nz)        # lower diag (a[0] unused)
    b = np.zeros(Nz)        # main diag
    c = np.zeros(Nz)        # upper diag (c[-1] unused)
    rhs_const = np.zeros(Nz)

    # interior nodes (i=1..Nz-2): (1 + 2*alpha) on diagonal, -alpha on off-diagonals
    for i in range(1, Nz - 1):
        a[i] = -alpha
        b[i] = 1.0 + 2.0 * alpha
        c[i] = -alpha
        rhs_const[i] = 0.0

    # Left boundary (i=0)
    # M[0,0] = 1 - dt * (D/dz^2 * (-2 + 2*dz*h)) = 1 + dt*D*(2/dz^2 - 2*h/dz)
    b[0] = 1.0 + dt * D * (2.0 / dz**2 - 2.0 * h / dz)
    # M[0,1] = - dt * D * (2 / dz^2)
    c[0] = - dt * D * (2.0 / dz**2)
    # RHS adds dt * b_boundary where b_boundary = - 2*h/dz * D * X_env
    rhs_const[0] = - dt * D * (2.0 * h / dz) * X_env

    # Right boundary (i = Nz-1)  <-- corrected signs (mirror of left)
    # M[-1,-2] = - dt * D * (2 / dz^2)
    a[Nz - 1] = - dt * D * (2.0 / dz**2)
    # M[-1,-1] = 1 - dt * (D/dz^2 * (-2 + 2*dz*h)) = 1 + dt*D*(2/dz^2 - 2*h/dz)
    b[Nz - 1] = 1.0 + dt * D * (2.0 / dz**2 - 2.0 * h / dz)
    # RHS same sign as left
    rhs_const[Nz - 1] = - dt * D * (2.0 * h / dz) * X_env

    return a, b, c, rhs_const


a, b, c, rhs_const = build_tridiagonal(Nz, alpha, dt, D, dz, h, X_env)

def thomas_solve(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x

# time-stepping
for n in range(0, Nt - 1):
    rhs = u[:, n].copy() + rhs_const
    u_new = thomas_solve(a, b, c, rhs)
    u[:, n + 1] = u_new

# --------- PLOTTING (fixed shapes) ----------
# prepare grids: meshgrid(z,t) -> shape (Nt, Nz)
Z_grid, T_grid = np.meshgrid(z, t)  # Z_grid.shape == (Nt, Nz), T_grid.shape == (Nt, Nz)
U_TZ = u.T  # shape (Nt, Nz)

# downsample factors (keep same ordering for all arrays)
ds_z = max(1, Nz // 100)
ds_t = max(1, Nt // 100)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Z_grid[::ds_t, ::ds_z], T_grid[::ds_t, ::ds_z], U_TZ[::ds_t, ::ds_z],
                linewidth=0, antialiased=True)
ax.set_xlabel('z')
ax.set_ylabel('t')
ax.set_zlabel('X(z,t)')
ax.set_title('Surface: X(z,t) (z vs t)')
plt.tight_layout()
fig.savefig('u_surface.png', dpi=200)
plt.close(fig)
print("Saved u_surface.png")

# Heatmap
fig2, ax2 = plt.subplots(figsize=(8, 5))
im = ax2.imshow(U_TZ, aspect='auto', origin='lower',
                extent=[z[0], z[-1], t[0], t[-1]])
ax2.set_xlabel('z')
ax2.set_ylabel('t')
ax2.set_title('Heatmap of X(z,t)')
cb = fig2.colorbar(im, ax=ax2)
cb.set_label('X')
plt.tight_layout()
fig2.savefig('u_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved u_heatmap.png")

# Profiles
times_to_plot = [0.0, 0.01, 0.05, 0.1, t_max]
idxs = [np.argmin(abs(t - tt)) for tt in times_to_plot]
fig3, ax3 = plt.subplots()
for idx in idxs:
    ax3.plot(z, u[:, idx], label=f"t={t[idx]:.3f}")
ax3.set_xlabel('z')
ax3.set_ylabel('X')
ax3.set_title('Profiles X(z) at selected times')
ax3.legend()
plt.tight_layout()
fig3.savefig('u_profiles.png', dpi=200)
plt.close(fig3)
print("Saved u_profiles.png")
