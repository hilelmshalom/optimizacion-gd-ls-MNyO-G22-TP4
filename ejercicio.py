import numpy as np
import matplotlib.pyplot as plt

np.random.seed(36645)

n = 5
d = 100
max_iter = 1000

A = np.random.randint(-100, 100, size=(n, d))
b = np.random.randint(-100, 100, size=(n))
x_rand = np.random.randint(-100, 100, size=(d))
x_zero = np.zeros(d)


# Función costo
def F(x, A, b):
    return np.dot((A @ x - b).T, (A @ x - b))


def F2(x, A, b, delta2):
    return F(x, A, b) + delta2 * np.linalg.norm(x) ** 2


def grad_F(x, A, b):
    return 2 * A.T @ (A @ x - b)


def grad_F2(x, A, b, delta2):
    return grad_F(x, A, b) + 2 * delta2 * x


# Valor del Hessiano
H = 2 * (A.T @ A)

# Encontrar el autovalor más grande del Hessiano
lambda_max = np.max(np.linalg.eigvals(H))

# Stepsize mas optimizado
s = 1 / lambda_max.real  # s = 1 / λ_max

# Mejor valor aproximado por SVD
U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
x_svd = np.linalg.pinv(A) @ b

# parametro δ_2
sigma_max = np.max(Sigma)
delta_2 = 10 ** (-2) * sigma_max


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


def metodo_gradiente_descendiente_reg(x_vector, iter=max_iter, steps=s):
    x_reg_gd = x_vector.copy()
    historial_F2_costo = [F2(x_reg_gd, A, b, delta_2)]
    historial_x_reg_gd = [x_reg_gd.copy()]

    for _ in range(iter):
        x_reg_gd = x_reg_gd - steps * grad_F2(x_vector, A, b, delta_2)
        historial_F2_costo.append(F2(x_reg_gd, A, b, delta_2))
        historial_x_reg_gd.append(x_reg_gd.copy())

    return x_reg_gd, historial_x_reg_gd, historial_F2_costo


x_min_reg_gd, hist_x_reg_gd, hist_F2_cost = metodo_gradiente_descendiente_reg(x_rand, max_iter, s)

norm_diff_gd = [np.linalg.norm(A @ x_min_gd - b)]
norm_diff_gd_reg = [np.linalg.norm(A @ x_min_reg_gd - b)]

for _ in range(max_iter):
    x_min_gd -= s * grad_F(x_min_gd, A, b)
    x_min_reg_gd -= s * grad_F2(x_min_reg_gd, A, b, delta_2)
    norm_diff_gd.append(np.linalg.norm(A @ x_min_gd - b))
    norm_diff_gd_reg.append(np.linalg.norm(A @ x_min_reg_gd - b))

ancla = np.full(100, 500)
ancla_comp = np.full(100, 501)

plt.figure(figsize=(8, 6))
plt.plot(hist_x_gd[::-1])
plt.scatter(ancla, x_svd)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(ancla_comp, hist_x_gd[0])
plt.scatter(ancla, x_svd)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(hist_x_reg_gd[::-1])
plt.scatter(ancla, x_svd)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(ancla_comp, hist_x_reg_gd[0])
plt.scatter(ancla, x_svd)
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.show()

plt.plot(norm_diff_gd, label='Gradiente Descendente')
plt.plot(norm_diff_gd_reg, label='Gradiente Descendente Regularizado')
plt.axhline(y=np.linalg.norm(A @ x_svd - b), color='r', linestyle='-', label='SVD')
plt.xlabel('Iteraciones')
plt.ylabel('Norma de la diferencia')
plt.legend()
plt.show()