# ex 4
#
# Crie um programa para gerar discretamente máscaras 3x3, 5x5 e 7x7, representativas de filtros
# gaussianos. Use os coeficientes da expansão binomial de Newton. Para cada máscara, calcule o valor
# de desvio padrao.


import math
from scipy.special import comb


def gerar_mascara_gaussiana(tamanho):
    if tamanho not in [3, 5, 7]:
        raise ValueError("O tamanho da máscara deve ser 3x3, 5x5 ou 7x7.")

    # Gera uma linha da expansão binomial de Newton (coeficientes binomiais)
    # comb(n, k) é a função que calcula o coeficiente binomial C(n, k)
    linha_pascal = [comb(tamanho - 1, i) for i in range(tamanho)]

    # Crie a matriz do filtro gaussiano usando o produto externo
    # Produto externo é o produto de cada elemento da linha_pascal por todos os elementos da mesma linha
    filtro_gaussiano = [
        [linha_pascal[i] * linha_pascal[j] for j in range(tamanho)]
        for i in range(tamanho)
    ]

    # Normalize a matriz para que a soma de todos os elementos seja 1
    soma_total = sum(linha_pascal) ** 2
    filtro_gaussiano_normalizado = [
        [elemento / soma_total for elemento in linha] for linha in filtro_gaussiano
    ]

    # Calcule o desvio padrão
    media = sum(linha_pascal) / tamanho
    variancia = sum([(linha_pascal[i] - media) ** 2 for i in range(tamanho)]) / (
        tamanho - 1
    )
    desvio_padrao = math.sqrt(variancia)

    return filtro_gaussiano_normalizado, desvio_padrao


# Gere as máscaras e calcule os desvios padrão
mascara_3x3, desvio_padrao_3x3 = gerar_mascara_gaussiana(3)
mascara_5x5, desvio_padrao_5x5 = gerar_mascara_gaussiana(5)
mascara_7x7, desvio_padrao_7x7 = gerar_mascara_gaussiana(7)

print("Máscara 3x3:\n", mascara_3x3)
print("Desvio Padrão 3x3:", desvio_padrao_3x3)
print("\n\n")

print("\nMáscara 5x5:\n", mascara_5x5)
print("Desvio Padrão 5x5:", desvio_padrao_5x5)
print("\n\n")

print("\nMáscara 7x7:\n", mascara_7x7)
print("Desvio Padrão 7x7:", desvio_padrao_7x7)
