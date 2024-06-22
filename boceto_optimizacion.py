import numpy as np
import matplotlib.pyplot as plt

np.random.seed(36645)

n = 5
d = 100
A = np.random.randint(-100, 100, size=(n, d))
b = np.random.randint(-100, 100, size=(n))
x_rand = np.random.randint(-100, 100, size=(d))
x_zero = np.zeros(d)

max_iter = 1000

def F(x, A, b):
    return np.dot((A @ x - b).T, (A @ x - b))

def grad_F(x, A, b):
    return 2 * A.T @ (A @ x - b)

# Calcular el Hessiano
H = 2 * np.dot(A.T, A)

# Calcular los autovalores del Hessiano
eigenvalues = np.linalg.eigvals(H)

# Encontrar el autovalor más grande
max_eigenvalue = np.max(eigenvalues)

s = 1 / max_eigenvalue.real  # s = 1 / λ_max

def metodo_gradiente_descendiente(x_vector):
    x_gd = x_vector.copy()
    for _ in range(max_iter):
        delta = s * grad_F(x_vector, A, b)
        x_gd = x_gd - delta
    return x_gd

alpha = metodo_gradiente_descendiente(x_rand)
print(alpha)
omega = metodo_gradiente_descendiente(x_zero)
print(omega)

print(b)
complejos = A @ omega
numeros_reales = [f"{num.real:.2f}" for num in complejos]

print(numeros_reales)
print(A @ omega)

delta2 = 1e-2 * np.linalg.norm(A, ord=2)

def F2(x, A, b, delta2):
    return F(x, A, b) + delta2 * np.linalg.norm(x) ** 2

def grad_F2(x, A, b, delta2):
    return grad_F(x, A, b) + 2 * delta2 * x

x_gd_reg = x_rand.copy()

for _ in range(max_iter):
    x_gd_reg = x_gd_reg - s * grad_F2(x_gd_reg, A, b, delta2)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
x_svd = np.linalg.pinv(A) @ b

def metodo_gradiente_descendiente(x_vector, iter=max_iter, steps=s):
    x_gd = x_vector.copy()
    historial_F_costo = [F(x_gd, A, b)]
    historial_x_gd = [x_gd.copy()]

    for _ in range(iter):
        x_gd = x_gd - steps * grad_F(x_vector, A, b)
        historial_F_costo.append(F(x_gd, A, b))
        historial_x_gd.append(x_gd.copy())

    return x_gd, historial_x_gd, historial_F_costo

x_min_gd, hist_x_gd, hist_F_cost = metodo_gradiente_descendiente(x_rand, max_iter, s)

norm_diff_gd = [np.linalg.norm(A @ x_min_gd - b)]
norm_diff_gd_reg = [np.linalg.norm(A @ x_gd_reg - b)]

for _ in range(max_iter):
    x_min_gd -= s * grad_F(x_min_gd, A, b)
    x_gd_reg -= s * grad_F2(x_gd_reg, A, b, delta2)
    norm_diff_gd.append(np.linalg.norm(A @ x_min_gd - b))
    norm_diff_gd_reg.append(np.linalg.norm(A @ x_gd_reg - b))

plt.plot(norm_diff_gd, label='Gradiente Descendente')
plt.plot(norm_diff_gd_reg, label='Gradiente Descendente Regularizado')
plt.axhline(y=np.linalg.norm(A @ x_svd - b), color='r', linestyle='-', label='SVD')
plt.xlabel('Iteraciones')
plt.ylabel('Norma de la diferencia')
plt.legend()
plt.show()