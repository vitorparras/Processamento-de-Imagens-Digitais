# 6. Aplique sobre a imagem(e) os ruídos aditivos: sal e pimenta e gaussiano. As distribuições devem
# ser fornecidas pelo usuário. Aplique os filtros apresentados abaixo.
#
# a) Suavização da imagem(Média, Mediana, Gaussiano e Moda) com janelas(w) 3x3;
#
# b) Filtro Passa-Alta com as máscaras:
#
#     h1=[[0, -1,  0],
#         [-1, 4, -1],
#         [0, -1,  0]]
#
#     h2=[[-1, -1, -1],
#         [-1,  8, -1],
#         [-1, -1, -1]]
#
# c) Considerando i como sendo cada imagem dada como entrada, determine qual filtro indicou o
# melhor resultado visual î. Em seguida, use uma métrica para avaliar a qualidade de î e confirmar sua
# hipótese.

# region Importações

import os
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import convolve
from scipy.signal import medfilt2d
from skimage.metrics import structural_similarity as ssim
import random
import matplotlib.pyplot as plt

# endregion

# region gera a imagem E

caminho = os.path.abspath(os.path.dirname(__file__)) + "\\"


def GeraImagem(matriz, nome):
    # Redimensiona a matriz ampliada para 256x256 pixels
    matriz_ampliada = np.repeat(np.repeat(matriz, 16, axis=0), 16, axis=1)

    # Cria uma nova imagem com 256 pixels de largura e 256 de altura
    img = Image.new("L", (256, 256))

    # Cria uma imagem a partir da matriz redimensionada
    img = Image.fromarray(matriz_ampliada.astype(np.uint8)).convert("L")

    # Salva a imagem em um arquivo PNG
    img.save(nome)

    return img


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

imgOriginal = GeraImagem(matriz_ampliada, caminho + "Imagem_E.bmp")

# endregion

# region funções para aplicar ruídos


def sal_pimenta(imagem, probabilidade):
    imagem_ruidosa = np.copy(imagem)
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            r = random.random()
            if r < probabilidade / 2:
                imagem_ruidosa[i, j] = 0
            elif r < probabilidade:
                imagem_ruidosa[i, j] = 255
    return imagem_ruidosa


def ruido_gaussiano(imagem, media, desvio_padrao):
    imagem_ruidosa = imagem + np.random.normal(media, desvio_padrao, imagem.shape)
    return np.clip(imagem_ruidosa, 0, 255).astype(np.uint8)


# endregion

# region funções para aplicar os filtros


def filtro_media(imagem, tamanho_kernel):
    return cv2.blur(imagem, (tamanho_kernel, tamanho_kernel))


def filtro_mediana(imagem, tamanho_kernel):
    return medfilt2d(imagem, tamanho_kernel)


def filtro_gaussiano(imagem, tamanho_kernel, desvio_padrao):
    imagem_uint8 = imagem.astype(np.uint8)
    return cv2.GaussianBlur(
        imagem_uint8, (tamanho_kernel, tamanho_kernel), desvio_padrao
    )


def filtro_moda(imagem, tamanho_kernel):
    imagem_uint8 = imagem.astype(np.uint8)
    return cv2.medianBlur(imagem_uint8, tamanho_kernel)


def filtro_passa_alta(imagem, mascara):
    return convolve(imagem, mascara, mode="reflect")


# endregion

# region Aplica os ruídos e filtros na imagem E:


def aplicar_filtro(imagem_Filtro, titulo, matriz_ampliada):
    # Aplique os filtros de suavização
    imagem_media = filtro_media(imagem_Filtro, 3)
    imagem_mediana = filtro_mediana(imagem_Filtro, 3)
    imagem_gaussiana_suavizada = filtro_gaussiano(imagem_Filtro, 3, 1)
    imagem_moda = filtro_moda(imagem_Filtro, 3)

    # Aplique os filtros passa-alta
    h1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    h2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    imagem_passa_alta_h1 = filtro_passa_alta(imagem_Filtro, h1)
    imagem_passa_alta_h2 = filtro_passa_alta(imagem_Filtro, h2)

    # Calcule o SSIM para cada imagem filtrada
    ssim_original = ssim(matriz_ampliada, matriz_ampliada, data_range=255)
    ssim_media = ssim(matriz_ampliada, imagem_media, data_range=255)
    ssim_mediana = ssim(matriz_ampliada, imagem_mediana, data_range=255)
    ssim_gaussiana_suavizada = ssim(
        matriz_ampliada, imagem_gaussiana_suavizada, data_range=255
    )
    ssim_moda = ssim(matriz_ampliada, imagem_moda, data_range=255)
    ssim_passa_alta_h1 = ssim(matriz_ampliada, imagem_passa_alta_h1, data_range=255)
    ssim_passa_alta_h2 = ssim(matriz_ampliada, imagem_passa_alta_h2, data_range=255)

    # Encontre o melhor resultado visual
    max_ssim = max(
        ssim_media,
        ssim_mediana,
        ssim_gaussiana_suavizada,
        ssim_moda,
        ssim_passa_alta_h1,
        ssim_passa_alta_h2,
    )
    if max_ssim == ssim_media:
        melhor_filtro = "Média"
    elif max_ssim == ssim_mediana:
        melhor_filtro = "Mediana"
    elif max_ssim == ssim_gaussiana_suavizada:
        melhor_filtro = "Gaussiana suavizada"
    elif max_ssim == ssim_moda:
        melhor_filtro = "Moda"
    elif max_ssim == ssim_passa_alta_h1:
        melhor_filtro = "Passa-alta h1"
    else:
        melhor_filtro = "Passa-alta h2"

    # Imprima os valores de SSIM
    print("imagem " + titulo + " -->SSIM Original: ", ssim_original)
    print("imagem " + titulo + " -->SSIM Média: ", ssim_media)
    print("imagem " + titulo + " -->SSIM Mediana: ", ssim_mediana)
    print(
        "imagem " + titulo + " -->SSIM Gaussiana suavizada: ", ssim_gaussiana_suavizada
    )
    print("imagem " + titulo + " -->SSIM Moda: ", ssim_moda)
    print("imagem " + titulo + " -->SSIM Passa-alta h1: ", ssim_passa_alta_h1)
    print("imagem " + titulo + " -->SSIM Passa-alta h2: ", ssim_passa_alta_h2)
    print(
        "imagem " + titulo + " -->Melhor resultado visual: ",
        melhor_filtro,
        " com SSIM=",
        max_ssim,
    )

    # Aplique os filtros passa-alta

    plt.subplot(2, 2, 2), plt.imshow(imagem_media, cmap="gray")
    plt.title("imagem " + titulo + " --> filtro media"), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 1), plt.imshow(imagem_mediana, cmap="gray")
    plt.title("imagem " + titulo + " --> filtro mediana"), plt.xticks([]), plt.yticks(
        []
    )

    plt.subplot(2, 2, 3), plt.imshow(imagem_gaussiana_suavizada, cmap="gray")
    plt.title("imagem " + titulo + " --> filtro gaussiano"), plt.xticks([]), plt.yticks(
        []
    )

    plt.subplot(2, 2, 4), plt.imshow(imagem_moda, cmap="gray")
    plt.title("imagem " + titulo + " --> filtro moda"), plt.xticks([]), plt.yticks([])

    plt.show()

    plt.subplot(335), plt.imshow(imagem_passa_alta_h1, cmap="gray")
    plt.title("imagem " + titulo + " --> filtro passa alta h1"), plt.xticks(
        []
    ), plt.yticks([])

    plt.subplot(336), plt.imshow(imagem_passa_alta_h2, cmap="gray")
    plt.title("imagem " + titulo + " --> filtro passa alta h2"), plt.xticks(
        []
    ), plt.yticks([])

    plt.show()


# Aplique os ruídos aditivos
imagem_sal_pimenta = sal_pimenta(matriz_ampliada, 0.05)
imagem_gaussiana = ruido_gaussiano(matriz_ampliada, 0, 10)

plt.subplot(331), plt.imshow(imgOriginal, cmap="gray")
plt.title("imagem original"), plt.xticks([]), plt.yticks([])

plt.subplot(332), plt.imshow(imagem_sal_pimenta, cmap="gray")
plt.title("imagem sal pimenta"), plt.xticks([]), plt.yticks([])

plt.subplot(333), plt.imshow(imagem_gaussiana, cmap="gray")
plt.title("imagem gaussiana"), plt.xticks([]), plt.yticks([])

plt.show()

print("\n\n\n")
aplicar_filtro(imagem_sal_pimenta, "Sal e pimenta", matriz_ampliada)

print("\n\n\n")
aplicar_filtro(imagem_gaussiana, "gaussiana", matriz_ampliada)

# endregion
