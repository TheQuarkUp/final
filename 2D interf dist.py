import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros iniciales del dominio y resolución
Lx, Ly = 10.0, 10.0  # Dimensiones del dominio
Nx, Ny = 200, 200  # Resolución espacial
dx, dy = Lx / Nx, Ly / Ny  # Pasos espaciales
T = 5.0  # Tiempo total de simulación
dt = 0.005  # Paso temporal pequeño para mayor precisión
c = 1.0  # Velocidad de la onda

# Definición de fuentes (pueden modificarse o añadirse más)
fuentes = [(Lx / 4, Ly / 2), (3 * Lx / 4, Ly / 2)]  # Coordenadas de las fuentes
n_fuentes = len(fuentes)

# Entradas del usuario para cada fuente
frecuencias = []
amplitudes = []
for i in range(n_fuentes):
    print(f"\nConfiguración para la fuente {i + 1}:")
    frecuencia = float(input(f"Ingrese la frecuencia para la fuente {i + 1} (valor sugerido: 1.0): "))
    amplitud = float(input(f"Ingrese la amplitud para la fuente {i + 1} (valor sugerido: 1.0): "))
    frecuencias.append(frecuencia)
    amplitudes.append(amplitud)

# Coeficiente de amortiguamiento (común para todas las paredes)
coef_amortiguamiento = float(input("\nIngrese el coeficiente de amortiguamiento de las paredes (0 a 1, sugerido: 0.9): "))

# Calcular parámetros derivados
r = c * dt / dx  # Número de Courant
if r > 1:
    print("Advertencia: el número de Courant (r) > 1. Reduce el dt o aumenta dx.")
    exit()

# Crear la malla espacial
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Configuración inicial: onda cero en todo el dominio
u0 = np.zeros((Nx, Ny))  # Estado inicial
u1 = np.zeros_like(u0)  # Paso anterior
u2 = np.zeros_like(u0)  # Paso siguiente

# Máscara para las fuentes
fuente_mask = np.zeros_like(u0, dtype=bool)
for fx, fy in fuentes:
    i, j = int(fx / dx), int(fy / dy)
    fuente_mask[i, j] = True

# Preparamos la figura para la animación
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
cmap = ax.imshow(
    u0, extent=[0, Lx, 0, Ly], cmap='gray', origin='lower', animated=True, vmin=-max(amplitudes), vmax=max(amplitudes)
)
ax.set_title("Simulación de Emisión Continua de Ondas (Personalizada)", color='white')
ax.set_xlabel("X", color='white')
ax.set_ylabel("Y", color='white')
ax.tick_params(colors='white')

# Crear la barra de color
colorbar = fig.colorbar(cmap, ax=ax)
colorbar.ax.yaxis.set_tick_params(color='white')
plt.setp(colorbar.ax.yaxis.get_ticklabels(), color='white')  # Cambiar color del texto del colorbar
colorbar.set_label("Amplitud", color='white')

# Actualización de la animación
def update(frame):
    global u0, u1, u2
    # Generar emisión continua desde las fuentes
    t = frame * dt
    for k, (fx, fy) in enumerate(fuentes):
        i, j = int(fx / dx), int(fy / dy)
        u1[i, j] = amplitudes[k] * np.sin(2 * np.pi * frecuencias[k] * t)  # Emisión sinusoidal individual

    # Actualizar los valores de la onda usando diferencias finitas
    u2[1:-1, 1:-1] = (
        2 * (1 - 2 * r**2) * u1[1:-1, 1:-1]
        - u0[1:-1, 1:-1]
        + r**2 * (
            u1[2:, 1:-1] + u1[:-2, 1:-1] + u1[1:-1, 2:] + u1[1:-1, :-2]
        )
    )

    # Amortiguamiento en los bordes
    u2[0, :] *= coef_amortiguamiento
    u2[-1, :] *= coef_amortiguamiento
    u2[:, 0] *= coef_amortiguamiento
    u2[:, -1] *= coef_amortiguamiento

    # Desplazar las soluciones en el tiempo
    u0[:], u1[:], u2[:] = u1, u2, u0
    cmap.set_data(u1)
    return [cmap]

# Crear animación
ani = FuncAnimation(fig, update, frames=int(T / dt), interval=dt * 1000, blit=True)

plt.show()
