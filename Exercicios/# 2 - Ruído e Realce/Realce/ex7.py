# Construa um programa para receber cada imagem indicada a seguir e, em seguida, apresentar os
# resultados após o processo de equalização de histograma. O programa deve apresentar também os
# histogramas das imagens, com e sem a equalização.


import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Lista de nomes de arquivos de imagens
image_files = ["\\frutas.bmp", "\\mammogram.bmp", "\\Moon.bmp", "\\polem.bmp"]
caminho = os.path.abspath(os.path.dirname(__file__)) + "\\"

for file in image_files:
    # Carregar imagem com PIL
    pil_image = Image.open(caminho + file).convert("L")

    # Converter para array do NumPy e então para imagem OpenCV
    img = np.array(pil_image)

    # Equalizar o histograma da imagem
    equ = cv2.equalizeHist(img)

    # Calcular histogramas
    hist_before = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_after = cv2.calcHist([equ], [0], None, [256], [0, 256])

    # Desenhar os histogramas e as imagens
    plt.figure(figsize=(10, 10))

    plt.subplot(221), plt.imshow(img, cmap="gray"), plt.title("Imagem Original")
    plt.subplot(222), plt.imshow(equ, cmap="gray"), plt.title("Imagem Equalizada")
    plt.subplot(223), plt.plot(hist_before), plt.title("Histograma Original")
    plt.subplot(224), plt.plot(hist_after), plt.title("Histograma Equalizado")

    plt.show()
