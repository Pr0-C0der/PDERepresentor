import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ---------------- Parameters ----------------
D = 0.1          # diffusion coefficient
r = 1.0          # reaction rate
nx = 100
ny = 100

x0, x1 = 0.0, 10.0
y0, y1 = 0.0, 10.0

dx = (x1 - x0) / nx
dy = (y1 - y0) / ny
x = np.linspace(x0, x1 - dx, nx)
y = np.linspace(y0, y1 - dy, ny)

# time domain
t0, t1 = 0.0, 2.0
nt = 100
t = np.linspace(t0, t1, nt)
dt = t[1] - t[0]

print(f"nx={nx}, ny={ny}, nt={nt}, dx={dx:.4e}, dy={dy:.4e}, dt={dt:.4e}")

# ---------------- Initial condition ----------------
# Gaussian centered at (5,5) with sigma=1
X, Y = np.meshgrid(x, y, indexing='ij')
u = np.zeros((nx, ny, nt))
u[:, :, 0] = np.exp(-((X - 5.0)**2 + (Y - 5.0)**2) / 2.0)

# ---------------- Time stepping (explicit) ----------------
for n in range(nt - 1):
    un = u[:, :, n]

    # periodic neighbors (roll in each direction)
    upx = np.roll(un, -1, axis=0)
    umx = np.roll(un, +1, axis=0)
    upy = np.roll(un, -1, axis=1)
    umy = np.roll(un, +1, axis=1)

    # second derivatives (central)
    uxx = (upx - 2 * un + umx) / dx**2
    uyy = (upy - 2 * un + umy) / dy**2

    diffusion = D * (uxx + uyy)
    reaction = r * un * (1.0 - un)

    u[:, :, n + 1] = un + dt * (diffusion + reaction)

# ---------------- Prepare for plotting ----------------
uf = u[:, :, -1]     # final time grid

# 1) 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

dsx = max(1, nx // 80)
dsy = max(1, ny // 80)

ax.plot_surface(X[::dsx, ::dsy], Y[::dsx, ::dsy], uf[::dsx, ::dsy],
                linewidth=0, antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title(f'Reaction-Diffusion u(x,y) at t={t[-1]:.2f}')
plt.tight_layout()
fig.savefig('u_final_surface.png', dpi=200)
plt.close(fig)
print("Saved u_final_surface.png")

# 2) Heatmap of final state
fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(uf.T, origin='lower',
                extent=[x0, x1, y0, y1],
                interpolation='nearest', aspect='equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(f'u(x,y) at t={t[-1]:.2f}')
plt.colorbar(im, ax=ax2, label='u')
plt.tight_layout()
fig2.savefig('u_final_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved u_final_heatmap.png")

# 3) Snapshots at 6 time points
times_to_plot = np.linspace(t0, t1, 6)
idxs = [np.argmin(np.abs(t - tt)) for tt in times_to_plot]

fig3, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.ravel()

for k, idx in enumerate(idxs):
    uk = u[:, :, idx]
    axk = axes[k]
    imk = axk.imshow(uk.T, origin='lower',
                     extent=[x0, x1, y0, y1],
                     interpolation='nearest', aspect='equal')
    axk.set_title(f't = {t[idx]:.2f}')
    axk.set_xlabel('x'); axk.set_ylabel('y')
    fig3.colorbar(imk, ax=axk)

plt.tight_layout()
fig3.savefig('u_snapshots.png', dpi=200)
plt.close(fig3)
print("Saved u_snapshots.png")

# Diagnostics
print("Initial min/max =", u[:, :, 0].min(), u[:, :, 0].max())
print("Final   min/max =", uf.min(), uf.max())
