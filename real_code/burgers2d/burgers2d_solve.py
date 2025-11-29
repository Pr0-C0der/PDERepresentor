import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- Parameters ----------------
nu = 0.1
nx = 100
ny = 100
x0, x1 = 0.0, 2.0 * np.pi
y0, y1 = 0.0, 2.0 * np.pi

dx = (x1 - x0) / nx
dy = (y1 - y0) / ny
x = np.linspace(x0, x1 - dx, nx)
y = np.linspace(y0, y1 - dy, ny)

t0, t1 = 0.0, 0.1
nt = 50
t = np.linspace(t0, t1, nt)
dt = t[1] - t[0]

print(f"nx={nx}, ny={ny}, nt={nt}, dx={dx:.4e}, dy={dy:.4e}, dt={dt:.4e}")

# Stability hint (printed)
max_dt_diff = 0.25 * min(dx*dx, dy*dy) / nu
print(f"recommended dt (diffusive CFL-ish) <= {max_dt_diff:.4e} ; current dt = {dt:.4e}")

# ---------------- Initial condition ----------------
# u is stored as u[:,:,n] where n is time index
u = np.zeros((nx, ny, nt))
X, Y = np.meshgrid(x, y, indexing='ij')
u[:, :, 0] = np.sin(X) * np.sin(Y)   # initial condition

# ---------------- Time stepping (explicit) ----------------
for n in range(nt - 1):
    un = u[:, :, n]

    # periodic neighbors via roll
    up = np.roll(un, -1, axis=0)   # i+1 in x
    um = np.roll(un, +1, axis=0)   # i-1 in x
    vp = np.roll(un, -1, axis=1)   # j+1 in y
    vm = np.roll(un, +1, axis=1)   # j-1 in y

    # derivatives (central)
    ux = (up - um) / (2.0 * dx)
    uy = (vp - vm) / (2.0 * dy)

    uxx = (up - 2.0 * un + um) / (dx * dx)
    uyy = (vp - 2.0 * un + vm) / (dy * dy)

    # rhs
    adv = - un * (ux + uy)
    diff = nu * (uxx + uyy)

    rhs = adv + diff

    u[:, :, n + 1] = un + dt * rhs

# ---------------- Prepare plots ----------------
# Final time slice
uf = u[:, :, -1]

# 1) 3D surface of final solution
Xg, Yg = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# downsample for plotting if grid is large
dsx = max(1, nx // 80)
dsy = max(1, ny // 80)
ax.plot_surface(Xg[::dsx, ::dsy], Yg[::dsx, ::dsy], uf[::dsx, ::dsy],
                linewidth=0, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y,t_final)')
ax.set_title(f'Burgers2D u at t={t[-1]:.3f}')
plt.tight_layout()
fig.savefig('u_final_surface.png', dpi=200)
plt.close(fig)
print("Saved u_final_surface.png")

# 2) Heatmap of final solution
fig2, ax2 = plt.subplots(figsize=(6,5))
im = ax2.imshow(uf.T, origin='lower', extent=[x0, x1, y0, y1],
                aspect='equal', interpolation='nearest')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(f'u(x,y) at t={t[-1]:.3f}')
plt.colorbar(im, ax=ax2, label='u')
plt.tight_layout()
fig2.savefig('u_final_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved u_final_heatmap.png")

# 3) Snapshots grid (6 panels) at selected times including t=0 and t_final
times_to_plot = np.linspace(t0, t1, 6)
idxs = [np.argmin(np.abs(t - tt)) for tt in times_to_plot]

fig3, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.ravel()
for k, idx in enumerate(idxs):
    uk = u[:, :, idx]
    axk = axes[k]
    imk = axk.imshow(uk.T, origin='lower', extent=[x0, x1, y0, y1],
                     aspect='equal', interpolation='nearest')
    axk.set_title(f't = {t[idx]:.3f}')
    axk.set_xlabel('x'); axk.set_ylabel('y')
    fig3.colorbar(imk, ax=axk)
plt.tight_layout()
fig3.savefig('u_snapshots.png', dpi=200)
plt.close(fig3)
print("Saved u_snapshots.png")

# ---------------- Diagnostics ----------------
print("Initial: min, max, mean =", u[:, :, 0].min(), u[:, :, 0].max(), u[:, :, 0].mean())
print("Final:   min, max, mean =", u[:, :, -1].min(), u[:, :, -1].max(), u[:, :, -1].mean())
