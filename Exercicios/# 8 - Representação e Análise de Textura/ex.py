### Construa um código para receber imagens monocromáticas como
### entrada, 8 bits de quantização. O código deve ser capaz de fornecer os
### valores de:
### -Haralick, com segundo momento angular, entropia e contraste. Use d=1 e  =0;
### -LBP (especificar as condições utilizadas);
### - Dimensão fractal (DF), usando Box-couting. A DF deve ser definida via
### coeficiente angular da regressão log x log. Os dois primeiros valores parciais de
### DF, em função das iterações 1 e 2, também devem ser apresentados.
### Os descritores devem ser organizados como vetores de características,
### respeitando a ordem posicional: momento angular; entropia; contraste; LBP; DF
### (coeficiente logxlog); DF iteração 1; DF iteração 2.
### Apresente os vetores para cada imagem. As imagens são apresentadas nos
### próximos slides.
### Em seguida, observe os resultados numéricos e indique quais descritores
### apresentam as maiores diferenças para separar as imagens R0 de R3. Apresente
### os gráficos para ilustrar as posições espaciais dos descritores. Por exemplo, eixo
### x representa momento angular, eixo y a entropia e eixo z o contraste. Use a
### mesma estratégia para DF. Cada imagem é um ponto espacial em função das
### suas coordenadas/descritores. Quais as dificuldades neste tipo de análise? Quais
### as soluções?

# Importação de bibliotecas
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
from scipy import ndimage
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Caminhos para suas imagens
caminho = os.path.abspath(os.path.dirname(__file__)) + "\\"

caminhos = [
    caminho + "R0_caso1.JPG",
    caminho + "R0_caso2.JPG",
    caminho + "R3_caso1.JPG",
    caminho + "R3_caso2.JPG",
]


# Função para calcular características de Haralick
def calcular_recursos_haralick(imagem):
    glcm = graycomatrix(imagem, [1], [0], 256, symmetric=True, normed=True)
    segundo_momento = graycoprops(glcm, "ASM")[0, 0]
    contraste = graycoprops(glcm, "contrast")[0, 0]
    entropia = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
    return segundo_momento, entropia, contraste


# Função para calcular LBP
def calcular_lbp(imagem):
    lbp = local_binary_pattern(imagem, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    return hist


# Função para calcular dimensão fractal
def calcular_dimensao_fractal(imagem):
    def boxcount(imagem, k):
        S = np.add.reduceat(
            np.add.reduceat(imagem, np.arange(0, imagem.shape[0], k), axis=0),
            np.arange(0, imagem.shape[1], k),
            axis=1,
        )
        return len(np.where((S > 0) & (S < k * k))[0])

    imagem = imagem > 0
    p = min(imagem.shape)
    n = 2 ** np.floor(np.log(p) / np.log(2))
    n = int(np.log(n) / np.log(2))
    sizes = 2 ** np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(imagem, size))
    coeficientes = np.polyfit(np.log(sizes), np.log(counts), 1)
    return coeficientes[0], counts[0], counts[1]


# Função para criar vetor de características
def vetor_de_caracteristicas(imagem):
    segundo_momento, entropia, contraste = calcular_recursos_haralick(imagem)
    lbp = calcular_lbp(imagem)
    df, df_iteracao1, df_iteracao2 = calcular_dimensao_fractal(imagem)
    return (
        [segundo_momento, entropia, contraste]
        + lbp.tolist()
        + [df, df_iteracao1, df_iteracao2]
    )


# Função para processar todas as imagens
def processar_imagens(caminhos):
    caracteristicas = []
    for caminho in caminhos:
        image_pil = Image.open(caminho)
        image = np.array(image_pil)
        imagem = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        caracteristicas.append(vetor_de_caracteristicas(imagem))
    return caracteristicas


caracteristicas = processar_imagens(caminhos)


# Função para analisar as diferenças nos descritores
def analisar_descritores(caracteristicas):
    caracteristicas = np.array(caracteristicas)
    diferenca_media = np.abs(
        np.mean(caracteristicas[:2, :], axis=0)
        - np.mean(caracteristicas[2:, :], axis=0)
    )
    descritores = (
        ["segundo momento angular", "entropia", "contraste"]
        + [f"LBP{i}" for i in range(1, 9)]
        + ["DF", "DF iteração 1", "DF iteração 2"]
    )
    for descritor, diferenca in zip(descritores, diferenca_media):
        print(f"{descritor}: {diferenca}")


analisar_descritores(caracteristicas)


# Função para plotar gráfico tridimensional dos descritores
def plotar_descritores_3d(caracteristicas, descritores_indices):
    caracteristicas = np.array(caracteristicas)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, caracteristicas_imagem in enumerate(caracteristicas):
        ax.scatter(
            caracteristicas_imagem[descritores_indices[0]],
            caracteristicas_imagem[descritores_indices[1]],
            caracteristicas_imagem[descritores_indices[2]],
            label=f"Imagem {i+1}",
        )

    ax.set_xlabel("Descritor 1")
    ax.set_ylabel("Descritor 2")
    ax.set_zlabel("Descritor 3")
    ax.legend()
    plt.show()


# Exemplo de uso da função
# Os índices 0, 1, 2 correspondem ao segundo momento angular, entropia e contraste respectivamente
plotar_descritores_3d(caracteristicas, [0, 1, 2])
