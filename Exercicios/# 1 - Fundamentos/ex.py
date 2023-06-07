# region Importações

import os
from PIL import Image
import numpy as np
import math
import random

# endregion

# Exercício 6

# Escreva uma programa para reproduzir as imagens apresentadas no slide 41.
# Considere que as imagens têm dimensões: 256x256 com 256 níveis de
# profundidade. Em seguida, o programa deve ser capaz de apresentar a taxa de
# amostragem e a profundidade de cada imagem.

# region funções
caminho = os.path.abspath(os.path.dirname(__file__)) + "\\"


def ExibeInfoImagem(nome):
    # Abre a imagem usando a função "open" da biblioteca Pillow
    img = Image.open(caminho + nome)

    # Exibe a taxa de amostragem da imagem
    taxa_amostragem = img.info.get("dpi")
    print(f"Taxa de amostragem {nome}: {taxa_amostragem[0]} dpi")

    # Exibe a profundidade da imagem
    img_array = np.array(img)
    profundidade = int(math.ceil(math.log2(np.max(img_array) + 1)))

    print(f"Profundidade da imagem {nome}: {profundidade} bits")

    print(" ")


def GeraImagem(matriz, nome):
    # Redimensiona a matriz ampliada para 256x256 pixels
    matriz_ampliada = np.repeat(np.repeat(matriz, 16, axis=0), 16, axis=1)

    # Cria uma nova imagem com 256 pixels de largura e 256 de altura
    img = Image.new("L", (256, 256))

    # Cria uma imagem a partir da matriz redimensionada
    img = Image.fromarray(matriz_ampliada.astype(np.uint8)).convert("L")

    # Salva a imagem em um arquivo PNG
    img.save(caminho + nome)


def MudaValorMatriz(matriz, inicio_linha, fim_linha, inicio_coluna, fim_coluna, valor):
    # Loop para percorrer as células da seção e atribuir um novo valor a cada célula
    for i in range(inicio_linha, fim_linha + 1):
        for j in range(inicio_coluna, fim_coluna + 1):
            # Atribui o valor definido a cada célula na seção
            matriz[i][j] = valor
    return matriz


# endregion

# region imagem A


print("\n\n\n\n\n\n\n\n\n\n\n\n")

imagem_A = np.ones((16, 16)) * 200
GeraImagem(imagem_A, "Imagem_A.bmp")

# endregion

# region imagem B

# Cria uma matriz 4x8 preenchida com valores 1
parte_superior = np.ones((8, 16)) * 200
# Cria uma matriz 4x8 preenchida com valores 0
parte_inferior = np.ones((8, 16)) * 150
# Empilha as duas matrizes verticalmente para criar uma matriz 8x8 dividida horizontalmente
imagem_B = np.vstack((parte_superior, parte_inferior))

GeraImagem(imagem_B, "Imagem_B.bmp")

# endregion

# region imagem C

# Cria uma matriz 8x16 preenchida com valores 1
parte_superior = np.ones((8, 16)) * 200
# Cria uma matriz 8x16 preenchida com valores 0
parte_inferior = np.ones((8, 16)) * 150

# Empilha as duas matrizes verticalmente para criar uma matriz 16x16 dividida horizontalmente
imagem_C = np.vstack((parte_superior, parte_inferior))

# retângulos superiores
# (matriz, inicio_linha, fim_linha, inicio_coluna, fim_coluna, valor)
imagem_C = MudaValorMatriz(imagem_C, 2, 5, 2, 6, 150)
imagem_C = MudaValorMatriz(imagem_C, 2, 5, 9, 13, 150)

# retângulos inferiores
imagem_C = MudaValorMatriz(imagem_C, 10, 13, 2, 6, 200)
imagem_C = MudaValorMatriz(imagem_C, 10, 13, 9, 13, 200)

GeraImagem(imagem_C, "Imagem_C.bmp")

# endregion

# region imagem D

matriz1 = np.ones((8, 8)) * 100
matriz2 = np.ones((8, 8)) * 150
matriz3 = np.ones((8, 8)) * 200
matriz4 = np.ones((8, 8)) * 250

imagem_D = np.block([[matriz1, matriz2], [matriz3, matriz4]])

GeraImagem(imagem_D, "Imagem_D.bmp")

# endregion

# region imagem E

# Cria uma matriz de zeros com dimensão
imagem_E = np.zeros((4, 4), dtype=int)

# Define o valor inicial como 30
valor = 100

# Percorre as células da matriz e atribui os valores de acordo com a regra definida
for i in range(imagem_E.shape[0]):
    for j in range(imagem_E.shape[1]):
        imagem_E[i, j] = valor
        valor += 10

# amplia a matriz
matriz_ampliada = np.repeat(np.repeat(imagem_E, 4, axis=0), 4, axis=1)

GeraImagem(matriz_ampliada, "Imagem_E.bmp")

# endregion

# region Exibi informações das Imagens

ExibeInfoImagem("Imagem_A.bmp")
ExibeInfoImagem("Imagem_B.bmp")
ExibeInfoImagem("Imagem_C.bmp")
ExibeInfoImagem("Imagem_D.bmp")
ExibeInfoImagem("Imagem_E.bmp")

# endregion

# Exercício 13

# Considere as imagens produzidas no Exercício 6 e implemente um
# programa para realizar a rotulagem de componentes conexos
# (cluster/aglomerado). A rotulagem deve ser realizada por meio do
# “Hoshen–Kopelman algorithm”. O programa deve fornecer o total de
# componentes conexos e os rótulos atribuídos em cada região da
# imagem dada como entrada. Use vizinhança-8 como critério. Por fim,
# considerando a imagem (e) após a rotulagem, o programa deve
# apresentar as distâncias (DE, D4 e D8) entre os centros de dois
# componentes conexos (definidos (sorteados) aleatoriamente).

# Função para encontrar a raiz de um rótulo


def achar_raiz(x, rotulos):
    # Encontra a raiz do rótulo x na árvore de rótulos
    while rotulos[x] < x:
        x = rotulos[x]
    return x


# Função para unir dois componentes conexos


def uniao(x, y, rotulos):
    # Encontra as raízes dos rótulos x e y na árvore de rótulos
    raiz_x = achar_raiz(x, rotulos)
    raiz_y = achar_raiz(y, rotulos)

    # Se as raízes são diferentes, atualiza a árvore de rótulos com a união dos componentes
    if raiz_x != raiz_y:
        if raiz_x < raiz_y:
            rotulos[raiz_y] = raiz_x
        else:
            rotulos[raiz_x] = raiz_y


# Algoritmo de Hoshen-Kopelman para rotular componentes conexos


def hoshen_kopelman(matriz):
    # Cria uma lista de rótulos, cada elemento inicializado com um rótulo diferente
    rotulos = np.arange(1, matriz.size + 1)

    # Percorre a matriz de pixels
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i, j] == 0:
                continue

            # Verifica os pixels acima e à esquerda do pixel atual
            acima = matriz[i - 1, j] if i > 0 else 0
            esquerda = matriz[i, j - 1] if j > 0 else 0

            # Se ambos os pixels são zeros, cria um novo rótulo para o pixel atual
            if acima == 0 and esquerda == 0:
                matriz[i, j] = np.max(rotulos)
                rotulos = np.append(rotulos, rotulos[-1] + 1)
            # Se apenas o pixel acima é zero, o rótulo do pixel atual é o mesmo do pixel à esquerda
            elif acima == 0:
                matriz[i, j] = esquerda
            # Se apenas o pixel à esquerda é zero, o rótulo do pixel atual é o mesmo do pixel acima
            elif esquerda == 0:
                matriz[i, j] = acima
            # Se ambos os pixels não são zero, o rótulo do pixel atual é o rótulo mínimo dos dois pixels
            # e os componentes são unidos
            else:
                matriz[i, j] = min(acima, esquerda)
                uniao(acima, esquerda, rotulos)

    # Percorre a matriz de pixels novamente para atualizar os rótulos
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i, j] != 0:
                matriz[i, j] = achar_raiz(matriz[i, j], rotulos)

    return matriz


# Função para processar a imagem e obter os componentes conexos rotulados


def processar_imagem(imagem):
    # Define um limiar para binarizar a imagem
    limiar = 180
    binarizada = (imagem >= limiar).astype(int)

    # Rotula os componentes conexos utilizando o algoritmo de Hoshen-Kopelman
    rotulada = hoshen_kopelman(binarizada)
    return rotulada


def Informacoes_componente_imagem(imagem):
    # Carrega a imagem
    imagem_E = Image.open(caminho + imagem)
    imagem_E_array = np.array(imagem_E)

    # Processa a imagem e obtém os componentes conexos rotulados
    imagem_rotulada = processar_imagem(imagem_E_array)
    print(f"Imagem rotulada {imagem}:")
    print(imagem_rotulada)

    # Conta o número total de componentes conexos
    rotulos_unicos = np.unique(imagem_rotulada)
    n_clusters = len(rotulos_unicos) - 1
    print(f"Número total de componentes conexos: {n_clusters}")
    print(
        "\n\n****************************************************************************************"
    )


if __name__ == "__main__":
    Informacoes_componente_imagem("Imagem_A.bmp")
    Informacoes_componente_imagem("Imagem_B.bmp")
    Informacoes_componente_imagem("Imagem_C.bmp")
    Informacoes_componente_imagem("Imagem_D.bmp")

    # Carrega a imagem
    imagem_E = Image.open(caminho + "Imagem_E.bmp")
    imagem_E_array = np.array(imagem_E)

    # Processa a imagem e obtém os componentes conexos rotulados
    imagem_rotulada = processar_imagem(imagem_E_array)
    print("Imagem rotulada Imagem_E.bmp:")
    print(imagem_rotulada)

    # Conta o número total de componentes conexos
    rotulos_unicos = np.unique(imagem_rotulada)
    n_clusters = len(rotulos_unicos) - 1
    print(f"Número total de componentes conexos: {n_clusters}")
    print(
        "\n\n****************************************************************************************"
    )

    # Escolhe dois componentes conexos aleatoriamente
    componente1 = random.choice(rotulos_unicos[1:])
    componente2 = random.choice(rotulos_unicos[1:])
    while componente2 == componente1 and n_clusters > 1:
        componente2 = random.choice(rotulos_unicos[1:])

    # Obtém as coordenadas dos pixels de cada componente
    y1, x1 = np.where(imagem_rotulada == componente1)
    y2, x2 = np.where(imagem_rotulada == componente2)

    # Calcula o centro de massa de cada componente
    centro1 = (np.mean(y1), np.mean(x1))
    centro2 = (np.mean(y2), np.mean(x2))

    # Calcula as três medidas de distância entre os centros dos componentes
    DE = np.linalg.norm(np.array(centro1) - np.array(centro2))
    D4 = np.abs(centro1[0] - centro2[0]) + np.abs(centro1[1] - centro2[1])
    D8 = max(np.abs(centro1[0] - centro2[0]), np.abs(centro1[1] - centro2[1]))

    # Imprime as distâncias calculadas
    print(
        "\n\n****************************************************************************************"
    )
    print(f"Distância Euclidiana (DE) entre os centros dos componentes: {DE}")
    print(f"Distância D4 entre os centros dos componentes: {D4}")
    print(f"Distância D8 entre os centros dos componentes: {D8}")
