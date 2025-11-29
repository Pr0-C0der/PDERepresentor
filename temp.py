import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def solve_diffusion_pde(D, h, X_env, z_points, t_points, z_domain, t_domain, X0):
    L = z_domain[-1] - z_domain[0]
    dz = L/(z_points - 1)

    A = np.zeros((z_points, z_points))

    for i in range(z_points - 1):
        A[i, i-1] = 1
        A[i, i] = -2
        A[i, i+1] = 1

    
    A[0, 0] = -2 - 2*dz*h
    A[0, 1] = 2
    A[-1,-2] = 2
    A[-1, -1] = -2 - 2*dz*h

    A = (D/dz**2)*A

    b = np.zeros(z_points)
    b[0] = 2*dz*h*X_env*(D/dz**2)
    b[-1] = 2*dz*h*X_env*(D/dz**2)

    def ode_system(t, y):
        return A@y + b

    y0 = np.full(z_points, X0)
    t_eval = np.linspace(t_domain[0], t_domain[1], t_points)
    sol = solve_ivp(ode_system, t_domain, y0=y0, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)

    return sol.y

D = 0.1
h = 1.0
X_env = 0.2
z_points = 1000
t_points = 1000
z_domain = np.linspace(0, 1, z_points)
t_domain = np.linspace(0, 1, t_points)
X0 = 1.0

sol = solve_diffusion_pde(D, h, X_env, z_points, t_points, z_domain, t_domain, X0)

plt.plot(z_domain, sol[:, -1])
plt.show()