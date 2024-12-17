import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros para la simulación de la ecuación de ondas en 1D
L = 10.0  # Longitud del dominio
Nx = 100  # Número de puntos espaciales
dx = L / Nx  # Paso espacial
T = 5.0  # Tiempo total de simulación
dt = 0.01  # Paso temporal
c = 1.0  # Velocidad de la onda
r = c * dt / dx  # Número de Courant (debe ser <= 1)

# Condición inicial: perturbación gaussiana
x = np.linspace(0, L, Nx)
u0 = np.exp(-100 * (x - L / 2)**2)  # Perturbación inicial
u1 = u0.copy()  # Paso anterior (igual al inicial en t=0)
u2 = np.zeros(Nx)  # Paso siguiente

# Preparamos la figura para la animación
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, u0, label='Onda')
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_xlabel("Posición")
ax.set_ylabel("Amplitud")
ax.set_title("Simulación de Ondas en 1D")
ax.legend()

# Actualización de la animación
def update(frame):
    global u0, u1, u2
    # Actualizar los valores de la onda usando diferencias finitas
    for i in range(1, Nx - 1):
        u2[i] = 2 * (1 - r**2) * u1[i] - u0[i] + r**2 * (u1[i+1] + u1[i-1])
    # Desplazar las soluciones en el tiempo
    u0, u1, u2 = u1, u2, u0
    line.set_ydata(u1)
    return line,

# Crear animación
ani = FuncAnimation(fig, update, frames=int(T / dt), interval=dt * 1000, blit=True)

plt.show()
