import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------- PARAMETERS ----------------
nu = 0.5

x0, x1 = 0.0, 2.0 * np.pi
y0, y1 = 0.0, 2.0 * np.pi

nx = 100
ny = 100

dx = (x1 - x0) / nx
dy = (y1 - y0) / ny

# create grid points (periodic - exclude last point so grid points are cell-centered)
x = np.linspace(x0, x1 - dx, nx)
y = np.linspace(y0, y1 - dy, ny)

# time
t0, t1 = 0.0, 1.0
nt = 100
t = np.linspace(t0, t1, nt)
dt = t[1] - t[0]

print(f"nx={nx}, ny={ny}, nt={nt}, dx={dx:.4e}, dy={dy:.4e}, dt={dt:.4e}")

# ---------------- INITIAL CONDITION ----------------
X, Y = np.meshgrid(x, y, indexing='ij')
u0 = np.sin(X) * np.sin(Y)   # shape (nx, ny)

# Flatten solution to vector of length N = nx*ny (ordering: x-major -> block rows)
N = nx * ny
u_vec = u0.ravel()  # initial vector

# ---------------- BUILD SPARSE 1D PERIODIC LAPLACIAN ----------------
# 1D second-derivative periodic matrix (scaled by 1/dx^2)
def periodic_1d_laplacian(n, dx):
    """Return sparse n x n periodic 1D Laplacian (central difference) / dx^2."""
    main = -2.0 * np.ones(n)
    off = 1.0 * np.ones(n - 1)
    # tridiagonal
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format='csr')
    # periodic connections
    A[0, -1] = 1.0
    A[-1, 0] = 1.0
    return A / (dx * dx)

Lx = periodic_1d_laplacian(nx, dx)
Ly = periodic_1d_laplacian(ny, dy)

# 2D Laplacian via Kronecker sum: L = kron(Iy, Lx) + kron(Ly, Ix)
Ix = sp.eye(nx, format='csr')
Iy = sp.eye(ny, format='csr')

L2 = sp.kron(Iy, Lx, format='csr') + sp.kron(Ly, Ix, format='csr')  # shape (N, N)

# ---------------- CRANK-NICOLSON MATRICES ----------------
# CN: (I - 0.5*dt*nu*L) u^{n+1} = (I + 0.5*dt*nu*L) u^{n}
A_mat = sp.eye(N, format='csr') - 0.5 * dt * nu * L2
B_mat = sp.eye(N, format='csr') + 0.5 * dt * nu * L2

# Factor A once (sparse LU)
print("Factoring LU for CN (this may take a moment)...")
A_fact = spla.splu(A_mat.tocsc())  # use CSC for splu
print("Factorization done.")

# ---------------- TIME STEPPING ----------------
u_history = np.zeros((nt, N))  # store for plotting (rows=time)
u_history[0, :] = u_vec.copy()

for n in range(nt - 1):
    rhs = B_mat.dot(u_vec)
    u_vec = A_fact.solve(rhs)
    u_history[n + 1, :] = u_vec

# reshape final solution
U_final = u_vec.reshape((nx, ny))

# ---------------- PLOTTING ----------------
# Final surface
Xg, Yg = np.meshgrid(x, y, indexing='ij')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
dsx = max(1, nx // 80)
dsy = max(1, ny // 80)
ax.plot_surface(Xg[::dsx, ::dsy], Yg[::dsx, ::dsy], U_final[::dsx, ::dsy],
                linewidth=0, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
ax.set_title(f'Heat eqn u at t={t[-1]:.3f}')
plt.tight_layout()
fig.savefig('heat2d_surface.png', dpi=200)
plt.close(fig)
print("Saved heat2d_surface.png")

# Final heatmap
fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(U_final.T, origin='lower', extent=[x0, x1, y0, y1],
                interpolation='nearest', aspect='equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(f'u(x,y) at t={t[-1]:.3f}')
plt.colorbar(im, ax=ax2)
plt.tight_layout()
fig2.savefig('heat2d_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved heat2d_heatmap.png")

# Snapshots: choose 6 times including t=0 and t_final
times_to_plot = np.linspace(t0, t1, 6)
idxs = [np.argmin(np.abs(t - tt)) for tt in times_to_plot]

fig3, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
for k, idx in enumerate(idxs):
    uk = u_history[idx, :].reshape((nx, ny))
    axk = axes[k]
    imk = axk.imshow(uk.T, origin='lower', extent=[x0, x1, y0, y1],
                     interpolation='nearest', aspect='equal')
    axk.set_title(f't = {t[idx]:.3f}')
    axk.set_xlabel('x'); axk.set_ylabel('y')
    fig3.colorbar(imk, ax=axk)
plt.tight_layout()
fig3.savefig('heat2d_snapshots.png', dpi=200)
plt.close(fig3)
print("Saved heat2d_snapshots.png")

# Diagnostics
print("Initial mean (should be ~0):", u_history[0, :].mean())
print("Final min/max/mean:", U_final.min(), U_final.max(), U_final.mean())
