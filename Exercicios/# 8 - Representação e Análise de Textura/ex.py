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
from skimage import feature
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def calcular_recursos_haralick(imagem, d=1, theta=0):
    glcm = feature.graycomatrix(imagem, [d], [theta], 256, symmetric=True, normed=True)
    contraste = feature.graycoprops(glcm, "contrast")[0, 0]
    segundo_momento_angular = feature.graycoprops(glcm, "ASM")[0, 0]
    entropia = stats.entropy(glcm.ravel())
    return segundo_momento_angular, entropia, contraste


def calcular_recursos_lbp(imagem, P=8, R=1):
    lbp = feature.local_binary_pattern(imagem, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    return hist.mean()


def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k),
        axis=1,
    )
    return len(np.where((S > 0) & (S < k * k))[0])


def calcular_dimensao_fractal(imagem, threshold=0.9):
    imagem = imagem < threshold

    p = min(imagem.shape)
    escalas = 2 ** np.arange(13, -1, -1)

    contagens = []
    for escala in escalas:
        contagens.append(boxcount(imagem, escala))

    coeficientes = np.polyfit(np.log(escalas), np.log(contagens), 1)
    return coeficientes[0]


def processar_imagem(caminho_imagem):
    image_pil = Image.open(caminho_imagem)
    image = np.array(image_pil)
    imagem = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    asm, entropia, contraste = calcular_recursos_haralick(imagem)
    lbp = calcular_recursos_lbp(imagem)
    df = calcular_dimensao_fractal(imagem)
    return [asm, entropia, contraste, lbp, df]


caminho = os.path.abspath(os.path.dirname(__file__)) + "\\"

caminhos_para_suas_imagens = [
    caminho + "R0_caso1.JPG",
    caminho + "R0_caso2.JPG",
    caminho + "R3_caso1.JPG",
    caminho + "R3_caso2.JPG",
]

# E estes com os rótulos correspondentes
rotulos_de_suas_imagens = ["R01", "R02", "R31", "R32"]

# Codificar rótulos como números
le = LabelEncoder()
rotulos_codificados = le.fit_transform(rotulos_de_suas_imagens)

recursos_imagens = [processar_imagem(img) for img in caminhos_para_suas_imagens]
recursos_imagens = np.array(recursos_imagens)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    recursos_imagens[:, 0],
    recursos_imagens[:, 1],
    recursos_imagens[:, 2],
    c=rotulos_codificados,
)
ax.set_xlabel("Segundo Momento Angular")
ax.set_ylabel("Entropia")
ax.set_zlabel("Contraste")
plt.show()
