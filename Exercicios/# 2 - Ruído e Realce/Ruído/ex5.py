# Descreva cada etapa para ilustrar o processo de aplicação de ruídos em uma imagem A
# (5x5), com níveis de cinza definidos aleatoriamente. Para tanto, determine os parâmetros
# iniciais para produzir uma imagem B (representativa do ruído em questão) e, em seguida,
# apresente:
#
# a) o resultado da função p(z);
# b) a imagem B representativa do ruído;
# c) o histograma de B para mostrar a característica do ruído;
# d) a matriz A após ser degrada por B;
# e) o histograma do resultado obtido em (d). Esses experimentos devem ser realizados
# com os ruídos gaussiano e poisson (shot noise) ou salt-and-pepper noise.


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# --------------------------------------------
# Resposta A
# --------------------------------------------
# Cria a imagem A (5x5) com níveis de cinza aleatórios
A = np.random.randint(0, 256, (5, 5))
print("Imagem A:")
print(A)

# --------------------------------------------
# Resposta B - Ruído Gaussiano
# --------------------------------------------
# Cria imagem B representativa do ruído gaussiano
mean = 0
std_dev = 10
B_gauss = np.random.normal(mean, std_dev, (5, 5))
print("\nImagem B - Ruído Gaussiano:")
print(B_gauss)

# --------------------------------------------
# Resposta C - Ruído Gaussiano
# --------------------------------------------
# Mostra histograma de B
plt.hist(B_gauss.ravel(), bins=256, color='gray', alpha=0.7)
plt.title("Histograma da imagem B (ruído gaussiano)")
plt.show()

# --------------------------------------------
# Resposta D - Ruído Gaussiano
# --------------------------------------------
# Degrada a imagem A com o ruído B
A_noisy_gauss = np.clip(A + B_gauss, 0, 255)
print("\nImagem A degradada pelo ruído gaussiano:")
print(A_noisy_gauss)

# --------------------------------------------
# Resposta E - Ruído Gaussiano
# --------------------------------------------
# Mostra histograma de A após ser degradada por B
plt.hist(A_noisy_gauss.ravel(), bins=256, color='gray', alpha=0.7)
plt.title("Histograma da imagem A degradada pelo ruído gaussiano")
plt.show()

# --------------------------------------------
# Resposta B - Ruído de Poisson
# --------------------------------------------
# Cria imagem B representativa do ruído Poisson
lam = 10
B_poisson = np.random.poisson(lam, (5, 5))
print("\nImagem B - Ruído de Poisson:")
print(B_poisson)

# --------------------------------------------
# Resposta C - Ruído de Poisson
# --------------------------------------------
# Mostra histograma de B
plt.hist(B_poisson.ravel(), bins=256, color='gray', alpha=0.7)
plt.title("Histograma da imagem B (ruído de Poisson)")
plt.show()

# --------------------------------------------
# Resposta D - Ruído de Poisson
# --------------------------------------------
# Degrada a imagem A com o ruído B
A_noisy_poisson = np.clip(A + B_poisson, 0, 255)
print("\nImagem A degradada pelo ruído de Poisson:")
print(A_noisy_poisson)

# --------------------------------------------
# Resposta E - Ruído de Poisson
# --------------------------------------------
# Mostra histograma de A após ser degradada por B
plt.hist(A_noisy_poisson.ravel(), bins=256, color='gray', alpha=0.7)
plt.title("Histograma da imagem A degradada pelo ruído de Poisson")
plt.show()

# --------------------------------------------
# Função p(z) - Ruído Gaussiano
# --------------------------------------------
# Mostra a função densidade de probabilidade do ruído gaussiano
x = np.linspace(-50, 50, 100)
p_gauss = stats.norm.pdf(x, mean, std_dev)

plt.plot(x, p_gauss)
plt.title("Função densidade de probabilidade do ruído gaussiano")
plt.show()

# --------------------------------------------
# Função p(z) - Ruído de Poisson
# --------------------------------------------
# Mostra a função densidade de probabilidade do ruído de Poisson
x = np.arange(0, 20)
p_poisson = stats.poisson.pmf(x, lam)

plt.plot(x, p_poisson)
plt.title("Função densidade de probabilidade do ruído de Poisson")
plt.show()
