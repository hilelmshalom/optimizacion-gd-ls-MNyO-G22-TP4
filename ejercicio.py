import numpy as np
import matplotlib.pyplot as plt

np.random.seed(36645+35526)

n = 5
d = 100
max_iter = 1000

A = np.random.rand(n, d)
b = np.random.rand(n)
x_rand = np.random.rand(d)

# Función costo
def F(x, A, b):
    return np.linalg.norm(A @ x - b)**2


def F2(x, A, b, delta2):
    return np.linalg.norm(A @ x - b)**2 + delta2 * np.linalg.norm(x)**2


def grad_F(x, A, b):
    return 2 * A.T @ (A @ x - b)


def grad_F2(x, A, b, delta2):
    return grad_F(x, A, b) + 2 * delta2 * x


# Valor del Hessiano
H = 2 * (A.T @ A)

# Encontrar el autovalor más grande del Hessiano
lambda_max = np.max(np.linalg.eigvals(H))

# Stepsize mas optimizado
s = 1 / lambda_max  # s = 1 / λ_max

# Mejor valor aproximado por SVD
_, Sigma, _ = np.linalg.svd(A, full_matrices=False)

# parametro δ_2
sigma_max = np.max(Sigma)
delta_2 = 10 ** (-2) * sigma_max
# aumentar delta2 hace que converga mas rapido

def metodo_gradiente_descendiente(x_vector, iter=max_iter, steps=s):
    historial_F_costo = []
    historial_x_gd = []

    for _ in range(iter):
        x_vector = x_vector - steps * grad_F(x_vector, A, b)
        historial_F_costo.append(F(x_vector, A, b))
        historial_x_gd.append(x_vector.copy())

    return x_vector, historial_x_gd, historial_F_costo


x_min_gd, hist_x_gd, hist_F_cost = metodo_gradiente_descendiente(x_rand, max_iter, s)


def metodo_gradiente_descendiente_reg(x_vector, iter=max_iter, steps=s):
    historial_F2_costo = []
    historial_x_reg_gd = []

    for _ in range(iter):
        x_vector = x_vector - steps * grad_F2(x_vector, A, b, delta_2)
        historial_F2_costo.append(F2(x_vector, A, b, delta_2))
        historial_x_reg_gd.append(x_vector.copy())

    return x_vector, historial_x_reg_gd, historial_F2_costo


x_min_reg_gd, hist_x_reg_gd, hist_F2_cost = metodo_gradiente_descendiente_reg(x_rand, max_iter, s)

# norm_diff_gd = [np.linalg.norm(A @ x_min_gd - b)]
# norm_diff_gd_reg = [np.linalg.norm(A @ x_min_reg_gd - b)]
#
# for _ in range(max_iter):
#     x_min_gd -= s * grad_F(x_min_gd, A, b)
#     x_min_reg_gd -= s * grad_F2(x_min_reg_gd, A, b, delta_2)
#     norm_diff_gd.append(np.linalg.norm(A @ x_min_gd - b))
#     norm_diff_gd_reg.append(np.linalg.norm(A @ x_min_reg_gd - b))

x_svd = np.linalg.pinv(A) @ b
svd_cost = F(x_svd, A, b)

#plt.figure(figsize=(8, 6))
plt.semilogy(hist_F_cost, label='F(x)', linewidth=2, color='red')
plt.semilogy(hist_F2_cost, label='F2(x)', linewidth=2, color='green')
plt.axhline(svd_cost, label='SVD', linewidth=2, color='blue')
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.legend()
plt.show()


# ancla = np.full(100, 500)
# ancla_comp = np.full(100, 501)
#
# plt.figure(figsize=(8, 6))
# plt.scatter(ancla_comp, hist_x_gd[0])
# plt.scatter(ancla, x_svd)
# plt.xlabel('Iteraciones')
# plt.ylabel('Costo')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.plot(hist_x_reg_gd[::-1])
# plt.scatter(ancla, x_svd)
# plt.xlabel('Iteraciones')
# plt.ylabel('Costo')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.scatter(ancla_comp, hist_x_reg_gd[0])
# plt.scatter(ancla, x_svd)
# plt.xlabel('Iteraciones')
# plt.ylabel('Costo')
# plt.legend()
# plt.show()
#
# plt.plot(norm_diff_gd, label='Gradiente Descendente')
# plt.plot(norm_diff_gd_reg, label='Gradiente Descendente Regularizado')
# plt.axhline(y=np.linalg.norm(A @ x_svd - b), color='r', linestyle='-', label='SVD')
# plt.xlabel('Iteraciones')
# plt.ylabel('Norma de la diferencia')
# plt.legend()
# plt.show()