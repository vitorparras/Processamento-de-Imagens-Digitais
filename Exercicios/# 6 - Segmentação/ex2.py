# Dada a imagem à direita, desenvolva um método capaz de
# segmentar as regiões “circulares” em azul/violeta. Descreva
# cada etapa utilizada para caracterizar o método. Em
# seguida, considere algumas regiões de controle e calcule as
# taxas de acerto e erro do método. O programa deve fornecer
# como saída uma imagem com as regiões circulares
# segmentadas e as taxas obtidas.

# Importação das bibliotecas necessárias
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from PIL import Image

# Função para redimensionar imagem


def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# Nome do arquivo de imagem a ser processado
file_name = os.path.abspath(os.path.dirname(__file__))+'\\ImgEx2.jpg'

# Abre a imagem com PIL
img_pil = Image.open(file_name)
img_pil.thumbnail((200, 200), Image.LANCZOS)

# Converte a imagem PIL para um array NumPy e depois para o formato de cores BGR
img_np = np.array(img_pil)
image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# Redimensiona a imagem para uma altura de 500 pixels
resize_ratio = 500 / image.shape[0]
original = image.copy()
image = opencv_resize(image, resize_ratio)

# Converte a imagem para o espaço de cores RGB e cria uma máscara para os tons de azul
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.inRange(gray, (100, 90, 90), (130, 255, 255))

# Utiliza a operação de erosão para reduzir o ruído na imagem
rectKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
dilated = cv2.erode(mask, rectKernel)

# Aplica o detector de bordas Canny para encontrar as bordas na imagem
edged = cv2.Canny(dilated, 13, 150, apertureSize=3)

# Encontra os contornos na imagem
contours, hierarchy = cv2.findContours(
    edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenha os contornos encontrados na imagem
image_with_contours = cv2.drawContours(
    image.copy(), contours, -1, (0, 255, 0), 3)

# Detecta círculos na imagem
circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100)

# Se algum círculo foi encontrado
if circles is not None:
    # Converte as coordenadas e raios dos círculos para inteiros
    circles = np.round(circles[0, :]).astype("int")

    # Desenha cada círculo na imagem
    for (x, y, r) in circles:
        cv2.circle(image_with_contours, (x, y), r, (0, 255, 0), 4)

# Cria uma figura para mostrar as imagens
fig, ax = plt.subplots(1, 5, figsize=(20, 4))

# Mostra a imagem original
ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")

# Mostra a máscara
ax[1].imshow(mask, cmap='Greys_r')
ax[1].set_title("Mask")

# Mostra a imagem após a operação de erosão
ax[2].imshow(dilated, cmap='Greys_r')
ax[2].set_title("Erosion")

# Mostra a imagem após a detecção de bordas Canny
ax[3].imshow(edged, cmap='Greys_r')
ax[3].set_title("Canny Edges")

# Mostra a imagem com os contornos desenhados
ax[4].imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
ax[4].set_title("Contours")

# Remove os eixos das subplots
for a in ax:
    a.axis("off")

plt.show()
