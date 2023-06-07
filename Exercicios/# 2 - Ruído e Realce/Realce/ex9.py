# Considere a imagem (e) com ruído gaussiano, exercício 3 da Aula 4, e aplique a correção gama
# com c=1 e  (0.04; 0.4; 2,5; 10). Visualmente, esse tipo de realce permitiu melhorar a qualidade
# da imagem com ruído? Caso sim, indique o valor de  e o apromixado da correção. Em seguida,
# calcule ao menos duas métricas indicadas no exercício 4 (Aula 4) e avalie se é possível
# comprovar quantitativamente as verificações iniciais. Por fim, conclua sobre a efetividade do
# realce para corrigir a imagem degradada.


import os
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função para realizar a correção gama na imagem


def gama(imagem, c=1):
    # Inverso do valor de gama
    invGama = 1 / c
    # Constrói a tabela de pesquisa para a correção gama
    correcao_gama = np.array(
        [((i / 255) ** invGama) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    # Aplica a correção gama na imagem usando a tabela de pesquisa
    return cv2.LUT(imagem, correcao_gama)


# Carrega a imagem usando PIL
caminho = os.path.abspath(os.path.dirname(__file__)) + "\\Imagem_E.bmp"
pil_imagem = Image.open(caminho).convert("L")

# Converte para array NumPy e depois para imagem OpenCV
imagem = np.array(pil_imagem)

# Função para calcular a similaridade de Jaccard entre duas imagens


def jaccard_binary(x, y):
    # Calcula a interseção das duas imagens
    intersection = np.logical_and(x, y)
    # Calcula a união das duas imagens
    union = np.logical_or(x, y)
    # Calcula a similaridade de Jaccard
    similaridade = intersection.sum() / float(union.sum())
    return similaridade


# Loop sobre diferentes valores de gama
for i in [0.04, 0.4, 2.5, 10]:
    # Desativa os eixos do gráfico
    plt.axis("off")
    # Realiza a correção gama na imagem
    imagem1 = gama(imagem, i)
    # Concatena horizontalmente a imagem original e a imagem após a correção gama
    img = np.hstack((imagem, imagem1))
    # Define o título do gráfico
    plt.title("       Original                     Correção Gama = " + str(i))
    # Calcula a similaridade de Jaccard
    similaridade = jaccard_binary(imagem, imagem1)
    # Calcula o erro médio quadrático (EMQ)
    emq = np.square(np.subtract(imagem, imagem1)).mean()
    # Mostra a imagem no gráfico
    plt.imshow(img, cmap="gray")
    plt.show()
    # Imprime o coeficiente de Jaccard
    print("Coeficiente de Jaccard = ", similaridade)
    # Imprime o erro médio quadrático
    print("Erro médio quadrático = ", emq)
