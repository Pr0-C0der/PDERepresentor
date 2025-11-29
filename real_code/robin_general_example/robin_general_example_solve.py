import numpy as np
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
nu = 0.1

# domain
x0, x1 = 0.0, 1.0
Nx = 101                  # grid points (including boundaries)
x = np.linspace(x0, x1, Nx)
dx = x[1] - x[0]

# time
t0, t1 = 0.0, 1.0
Nt = 100
t = np.linspace(t0, t1, Nt)
dt = t[1] - t[0]

print(f"Nx={Nx}, Nt={Nt}, dx={dx:.4e}, dt={dt:.4e}, nu={nu}")

# Robin BC parameters
aL, bL, cL = 2.0, 3.0, 5.0         # left: aL*u(0) + bL*u_x(0) = cL (cL constant)
aR, bR = 1.0, 1.0                  # right: aR*u(1) + bR*u_x(1) = cR(t)
def cR(time):
    return np.sin(time)

# ---------------- INITIAL CONDITION ----------------
u = np.zeros((Nx, Nt))
u[:, 0] = np.sin(np.pi * x)        # u(x,0) = sin(pi x)

# indices of interior unknowns (we solve for u[0..Nx-1] directly, but matrix covers all nodes)
# We'll build tridiagonal system of size Nx (including boundary rows modified by Robin BC elimination)
# but simpler: keep arrays length Nx with a[0]=0, c[-1]=0 as usual
Acoef = nu * dt
alpha = nu * dt / dx**2  # common name

# ---------------- Build tridiagonal (template) ----------------
def build_tridiag(Nx, alpha, dt, nu, dx, aL, bL, aR, bR, tnext):
    """
    Return arrays a (sub-diag), b (diag), c (super-diag), rhs_const (contribution from BC constants)
    for the linear system A u^{n+1} = u^n + rhs_const
    """
    a = np.zeros(Nx)   # sub-diagonal (a[0] unused)
    b = np.zeros(Nx)   # main diagonal
    c = np.zeros(Nx)   # super-diagonal (c[-1] unused)
    rhs_const = np.zeros(Nx)

    # interior rows i = 1..Nx-2 (standard BE for diffusion)
    for i in range(1, Nx - 1):
        a[i] = -alpha
        b[i] = 1.0 + 2.0 * alpha
        c[i] = -alpha

    # LEFT boundary (i = 0) using ghost-elimination for Robin aL*u0 + bL*u_x(0)=cL
    # Derived coefficients (see explanation):
    # b0 = 1 + A*(2/dx^2 - 2*aL/(bL*dx))
    # c0 = - A*(2/dx^2)
    # RHS contribution: rhs_const[0] = - A*(2*cL/(bL*dx))
    b[0] = 1.0 + dt * nu * (2.0 / dx**2 - 2.0 * aL / (bL * dx))
    c[0] = - dt * nu * (2.0 / dx**2)
    a[0] = 0.0
    rhs_const[0] = - dt * nu * (2.0 * cL / (bL * dx))

    # RIGHT boundary (i = Nx-1) using ghost-elimination for Robin aR*uN + bR*u_x(1)=cR(t)
    # Derived coefficients:
    # aN = - A*(2/dx^2)
    # bN = 1 + A*(2/dx^2 + 2*aR/(bR*dx))
    # RHS contribution: rhs_const[-1] = + A*(2*cR/(bR*dx))
    a[Nx - 1] = - dt * nu * (2.0 / dx**2)
    b[Nx - 1] = 1.0 + dt * nu * (2.0 / dx**2 + 2.0 * aR / (bR * dx))
    c[Nx - 1] = 0.0
    rhs_const[Nx - 1] = + dt * nu * (2.0 * cR(tnext) / (bR * dx))

    return a, b, c, rhs_const

# Thomas solver for tridiagonal system
def thomas_solve(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    xsol = np.zeros(n)

    # first row
    denom = b[0]
    if abs(denom) < 1e-14:
        denom = 1e-14
    cp[0] = c[0] / denom
    dp[0] = d[0] / denom

    # forward sweep
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-14:
            denom = 1e-14
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # back substitution
    xsol[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        xsol[i] = dp[i] - cp[i] * xsol[i + 1]

    return xsol

# ---------------- Time-stepping ----------------
for n in range(Nt - 1):
    tnext = t[n + 1]
    a_diag, b_diag, c_diag, rhs_const = build_tridiag(Nx, alpha, dt, nu, dx, aL, bL, aR, bR, tnext)

    # RHS = u^n + rhs_const
    rhs = u[:, n].copy() + rhs_const

    # solve A u^{n+1} = rhs
    u_new = thomas_solve(a_diag, b_diag, c_diag, rhs)
    u[:, n + 1] = u_new

# ---------------- PLOTTING & SAVING ----------------
# Prepare arrays for plotting: U_Tx shape (Nt, Nx): rows = time, cols = x
U_Tx = u.T

# meshgrid for surface
X_grid, T_grid = np.meshgrid(x, t)   # shapes (Nt, Nx)

# downsample for surface clarity
ds_x = max(1, Nx // 100)
ds_t = max(1, Nt // 100)

# 1) Surface plot
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

# 2) Heatmap (rows=time, cols=space)
fig2, ax2 = plt.subplots(figsize=(8, 5))
im = ax2.imshow(U_Tx, aspect='auto', origin='lower',
                extent=[x0, x1, t0, t1], interpolation='nearest')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title('Heatmap of u(x,t)')
cb = fig2.colorbar(im, ax=ax2)
cb.set_label('u')
plt.tight_layout()
fig2.savefig('u_heatmap.png', dpi=200)
plt.close(fig2)
print("Saved u_heatmap.png")

# 3) Profile snapshots
times_to_plot = [0.0, 0.1, 0.25, 0.5, 1.0]
idxs = [np.argmin(np.abs(t - tt)) for tt in times_to_plot]
fig3, ax3 = plt.subplots()
for idx in idxs:
    ax3.plot(x, u[:, idx], label=f"t={t[idx]:.3f}")
ax3.set_xlabel('x')
ax3.set_ylabel('u')
ax3.set_title('Profiles u(x) at selected times')
ax3.legend()
plt.tight_layout()
fig3.savefig('u_profiles.png', dpi=200)
plt.close(fig3)
print("Saved u_profiles.png")

# quick diagnostics
print("Initial (t=0): min, max, mean =", u[:, 0].min(), u[:, 0].max(), u[:, 0].mean())
print("Final   (t=1): min, max, mean =", u[:, -1].min(), u[:, -1].max(), u[:, -1].mean())
