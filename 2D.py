import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros para la simulación de la ecuación de ondas en 2D
Lx, Ly = 10.0, 10.0  # Dimensiones del dominio
Nx, Ny = 100, 100  # Número de puntos espaciales
dx, dy = Lx / Nx, Ly / Ny  # Pasos espaciales
T = 5.0  # Tiempo total de simulación
dt = 0.01  # Paso temporal
c = 1.0  # Velocidad de la onda
r = c * dt / dx  # Número de Courant (debe ser <= 1)

# Condición inicial: perturbación gaussiana en el centro
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
u0 = np.exp(-100 * ((X - Lx / 2)**2 + (Y - Ly / 2)**2))  # Perturbación inicial
u1 = u0.copy()  # Paso anterior (igual al inicial en t=0)
u2 = np.zeros_like(u0)  # Paso siguiente

# Preparamos la figura para la animación
fig, ax = plt.subplots(figsize=(8, 8))
cmap = ax.imshow(u0, extent=[0, Lx, 0, Ly], cmap='viridis', origin='lower')
ax.set_title("Simulación de Ondas en 2D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
fig.colorbar(cmap, ax=ax, label="Amplitud")

# Actualización de la animación
def update(frame):
    global u0, u1, u2
    # Actualizar los valores de la onda usando diferencias finitas
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u2[i, j] = (
                2 * (1 - 2 * r**2) * u1[i, j]
                - u0[i, j]
                + r**2 * (u1[i+1, j] + u1[i-1, j] + u1[i, j+1] + u1[i, j-1])
            )
    # Desplazar las soluciones en el tiempo
    u0, u1, u2 = u1, u2, u0
    cmap.set_data(u1)
    return [cmap]

# Crear animación
ani = FuncAnimation(fig, update, frames=int(T / dt), interval=dt * 1000, blit=True)

plt.show()
