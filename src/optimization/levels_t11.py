import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def objective(point):
    x1 = point[0]
    x2 = point[1]
    return (
        0.2 * x1**2 + 0.7 * x2**2
        - 0.2 * x1 * x2 - 0.9 * x1 - 1.2 * x2 + 1.8
    )


# 1. Область рисования: горизонтальные координаты X и Y.
x1_values = np.linspace(-1, 6.5, 100)
x2_values = np.linspace(-1, 3.5, 100)

# 2. Полная сетка пар координат.
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

# 3. Высота Z в каждой точке сетки.
z_grid = objective([x1_grid, x2_grid])

# 4. Минимум ищется независимо от сетки.
result = minimize(objective, [2, 1], method="BFGS")
print(f"x1 = {result.x[0]:.5f}")
print(f"x2 = {result.x[1]:.5f}")
print(f"f_min = {result.fun:.5f}")
print(f"Успех: {result.success}")

# 5. Двумерные линии уровня (вид сверху на чашу).
fig2d, ax2d = plt.subplots(figsize=(8, 5))
contours = ax2d.contour(
    x1_grid, x2_grid, z_grid,
    levels=[0.1, 0.5, 1.0, 1.5]
)
ax2d.clabel(contours, inline=True, fmt="%.1f")
ax2d.scatter(2, 1, color="orange", label="Старт")
ax2d.scatter(*result.x, color="red", marker="*", s=150, label="Минимум")
ax2d.set(xlabel="x1 (X)", ylabel="x2 (Y)", title="Линии уровня")
ax2d.set_aspect("equal")
ax2d.grid(alpha=0.3)
ax2d.legend()

# 6. Трёхмерная поверхность: X, Y и высота Z.
fig3d = plt.figure(figsize=(9, 6))
ax3d = fig3d.add_subplot(111, projection="3d")
ax3d.plot_surface(x1_grid, x2_grid, z_grid, cmap="viridis", alpha=0.8)
ax3d.scatter(
    result.x[0], result.x[1], result.fun,
    color="red", s=100, label="Минимум"
)
ax3d.set(xlabel="x1 (X)", ylabel="x2 (Y)", zlabel="Z = f(x1, x2)")
ax3d.legend()

plt.show()