import numpy as np
import matplotlib.pyplot as plt

# ----------------- PARAMETERS -----------------
a = 1.0        # advection speed
nu = 0.05      # diffusion coefficient
lam = 0.5      # reaction coefficient multiplier (lambda)

# spatial domain
x0, x1 = 0.0, 1.0
Nx = 201            # number of grid points including boundaries
x = np.linspace(x0, x1, Nx)
dx = x[1] - x[0]

# time domain
t0, t1 = 0.0, 1.0
Nt = 500
t = np.linspace(t0, t1, Nt)
dt = t[1] - t[0]

print(f"Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}")

# ----------------- INITIAL CONDITION -----------------
u = np.zeros((Nx, Nt))
u[:, 0] = np.exp(-50.0 * (x - 0.5) ** 2)
# enforce Dirichlet BCs
u[0, :] = 0.0
u[-1, :] = 0.0

# ----------------- DISCRETIZATION CONSTANTS -----------------
# unknowns are interior nodes i = 1..Nx-2 (M = Nx-2)
M = Nx - 2
idx = np.arange(1, Nx - 1)

adv_coef = a / dx
diff_coef = nu / dx**2

# Build tridiagonal coefficients for backward Euler at a given next time t_{n+1}
def build_tridiagonal_for_time(tnext):
    s = np.sin(tnext)
    # coefficients of L (L = -a*d/dx + nu*d2/dx2 + lam*sin(t)*I):
    # L_{i,i-1} = adv_coef + diff_coef
    # L_{i,i  } = -adv_coef - 2*diff_coef + lam * s
    # L_{i,i+1} = diff_coef
    sub = -dt * (adv_coef + diff_coef)           # sub-diagonal (length M)
    main = 1.0 + dt * (adv_coef + 2.0 * diff_coef - lam * s)  # main diagonal (length M)
    sup = -dt * diff_coef                        # super-diagonal (length M)
    # as arrays
    sub_arr = np.full(M, sub)
    main_arr = np.full(M, main)
    sup_arr = np.full(M, sup)
    return sub_arr, main_arr, sup_arr

# ----------------- THOMAS SOLVER -----------------
def thomas_solve(sub, main, sup, rhs):
    """Solve tridiagonal system with sub[0..n-1], main[0..n-1], sup[0..n-1] (sup[-1] unused).
       rhs length n. Returns solution x length n."""
    n = len(rhs)
    # copy arrays so we don't overwrite inputs
    cprime = np.zeros(n)
    dprime = np.zeros(n)
    xsol = np.zeros(n)

    # first row
    denom = main[0]
    if abs(denom) < 1e-14:
        denom = 1e-14
    cprime[0] = sup[0] / denom
    dprime[0] = rhs[0] / denom

    # forward sweep
    for i in range(1, n):
        denom = main[i] - sub[i] * cprime[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cprime[i] = sup[i] / denom if i < n - 1 else 0.0
        dprime[i] = (rhs[i] - sub[i] * dprime[i - 1]) / denom

    # back substitution
    xsol[-1] = dprime[-1]
    for i in range(n - 2, -1, -1):
        xsol[i] = dprime[i] - cprime[i] * xsol[i + 1]

    return xsol

# ----------------- TIME-STEPPING (backward Euler implicit) -----------------
for n in range(Nt - 1):
    tnext = t[n + 1]
    sub_arr, main_arr, sup_arr = build_tridiagonal_for_time(tnext)

    # RHS: u_interior^n. Dirichlet BCs are zero so no extra boundary contributions
    rhs = u[idx, n].copy()

    # Solve tridiagonal system for interior nodes
    u_interior_new = thomas_solve(sub_arr, main_arr, sup_arr, rhs)

    # store
    u[1:-1, n + 1] = u_interior_new
    u[0, n + 1] = 0.0
    u[-1, n + 1] = 0.0

# ----------------- PLOTTING & SAVING -----------------
# Prepare U with shape (Nt, Nx) rows=time, cols=space
U_Tx = u.T  # Nt x Nx

# 1) Surface plot (x vs t vs u)
X_grid, T_grid = np.meshgrid(x, t)  # both shape (Nt, Nx)

# downsample for surface clarity if arrays big
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

# 2) Heatmap (rows=time, cols=space). Use nearest interpolation so cell values are accurate visually.
fig2, ax2 = plt.subplots(figsize=(8, 5))
im = ax2.imshow(U_Tx, aspect='auto', origin='lower',
                extent=[x[0], x[-1], t[0], t[-1]],
                interpolation='nearest')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title('Heatmap of u(x,t)')
cb = fig2.colorbar(im, ax=ax2)
cb.set_label('u')
plt.tight_layout()
fig2.savefig('u_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved u_heatmap.png")

# 3) Profiles at selected times
times_to_plot = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
idxs = [np.argmin(np.abs(t - tt)) for tt in times_to_plot]

fig3, ax3 = plt.subplots()
for idxi in idxs:
    ax3.plot(x, u[:, idxi], label=f"t={t[idxi]:.3f}")
ax3.set_xlabel('x')
ax3.set_ylabel('u')
ax3.set_title('Profiles u(x) at selected times')
ax3.legend()
plt.tight_layout()
fig3.savefig('u_profiles.png', dpi=200)
plt.close(fig3)
print("Saved u_profiles.png")

# Quick diagnostics
print("Initial (t=0): min, max, mean =", u[:,0].min(), u[:,0].max(), u[:,0].mean())
print("Final (t=1):   min, max, mean =", u[:,-1].min(), u[:,-1].max(), u[:,-1].mean())
