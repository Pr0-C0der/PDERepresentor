import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- PARAMETERS ----------------
nu = 0.5              # diffusivity
x0, x1 = 0.0, 2.0 * np.pi
Nx = 128              # spatial points
dx = (x1 - x0) / Nx
x = np.linspace(x0, x1 - dx, Nx)  # cell-centered periodic grid (exclude endpoint)
# time
t0, t1 = 0.0, 1.0
Nt = 400
t = np.linspace(t0, t1, Nt)
dt = t[1] - t[0]

alpha = nu * dt / dx**2
print(f"Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}, alpha = nu*dt/dx^2 = {alpha:.6f}")

# ---------------- INITIAL CONDITION ----------------
u = np.zeros((Nx, Nt))
u[:, 0] = np.sin(x)   # initial condition u(x,0) = sin(x)

# ---------------- DISCRETE LAPLACIAN (periodic) ----------------
# Build the Nx x Nx matrix L approximating d^2/dx^2 with periodic BCs:
# L[i,i] = -2, L[i,i+1] = 1, L[i,i-1] = 1, and wrap-around at boundaries.
L = np.zeros((Nx, Nx))
for i in range(Nx):
    L[i, i] = -2.0
    L[i, (i - 1) % Nx] = 1.0
    L[i, (i + 1) % Nx] = 1.0
# Note: the actual second derivative is (1/dx^2) * L

# Crank-Nicolson matrices:
# (I - 0.5*alpha*L) u^{n+1} = (I + 0.5*alpha*L) u^n
I = np.eye(Nx)
A = I - 0.5 * alpha * L
B = I + 0.5 * alpha * L

# We will factor A once if desired; here we just use numpy.linalg.solve for simplicity

# ---------------- TIME-STEPPING ----------------
for n in range(Nt - 1):
    rhs = B.dot(u[:, n])
    u[:, n + 1] = np.linalg.solve(A, rhs)

# ---------------- PLOTTING & SAVING ----------------
# Prepare arrays for plotting: we want U_t_x with shape (Nt, Nx) (rows=time, cols=space)
U_Tx = u.T  # shape (Nt, Nx)

# 1) Surface plot (x vs t vs u) -- make grids (Nt x Nx)
X_grid, T_grid = np.meshgrid(x, t)  # both shape (Nt, Nx)

# downsample for speed/clarity in surface
ds_x = max(1, Nx // 120)
ds_t = max(1, Nt // 120)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_grid[::ds_t, ::ds_x], T_grid[::ds_t, ::ds_x], U_Tx[::ds_t, ::ds_x],
                linewidth=0, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Surface: u(x,t)')
plt.tight_layout()
fig.savefig('u_surface.png', dpi=200)
plt.close(fig)
print("Saved u_surface.png")

# 2) Heatmap: rows=time, columns=space (use nearest interpolation to show exact cells)
fig2, ax2 = plt.subplots(figsize=(8, 5))
im = ax2.imshow(U_Tx, aspect='auto', origin='lower',
                extent=[x.min(), x.max(), t.min(), t.max()],
                interpolation='nearest')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title('Heatmap of u(x,t)')
cbar = fig2.colorbar(im, ax=ax2)
cbar.set_label('u')
plt.tight_layout()
fig2.savefig('u_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved u_heatmap.png")

# 3) Profiles at selected times
times_to_plot = [0.0, 0.01, 0.05, 0.1, t1]
idxs = [np.argmin(np.abs(t - tt)) for tt in times_to_plot]
fig3, ax3 = plt.subplots()
for idx in idxs:
    ax3.plot(x, u[:, idx], label=f"t={t[idx]:.3f}")
ax3.set_xlabel('x'); ax3.set_ylabel('u')
ax3.set_title('Profiles u(x) at selected times')
ax3.legend()
plt.tight_layout()
fig3.savefig('u_profiles.png', dpi=200)
plt.close(fig3)
print("Saved u_profiles.png")

# Optional quick checks: mass conservation (should decay for diffusion? For pure diffusion with periodic BC,
# the integral of u is conserved in time if there is no source term. For initial sin(x) integral is zero)
mean_initial = u[:, 0].mean()
mean_final = u[:, -1].mean()
print(f"mean initial = {mean_initial:.6e}, mean final = {mean_final:.6e}")
