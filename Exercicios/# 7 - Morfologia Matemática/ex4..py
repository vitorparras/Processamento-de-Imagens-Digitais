# 4. Construa um programa que receba a imagem A, forneça uma imagem limiarizada
# parecida com a indicada em B. Caso necessário, considere o método de Otsu para
# obter a solução mais apropriada. Em seguida, aplique a operação morfológica
# necessária para obter um resultado similar ao indicado em C.
# Indique qual o valor de liminar escolhido e o elemento estruturante aplicado para obter C.


# Importação das bibliotecas necessárias
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Carregamento da imagem
caminho = os.path.abspath(os.path.dirname(__file__))+'\\Img4.bmp'
image_pil = Image.open(caminho)
image = np.array(image_pil)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Limiarização da imagem usando o método de Otsu
valor_limiar, limiarizada = cv2.threshold(
    image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Criação do elemento estruturante
# Neste exemplo, estamos usando um kernel quadrado 5x5.
kernel = np.ones((5, 5), np.uint8)

# Aplicação da operação morfológica
# estamos usando a operação de abertura (erosão seguida de dilatação)
morfologica = cv2.morphologyEx(limiarizada, cv2.MORPH_OPEN, kernel)

# Exibição das imagens
plt.subplot(131), plt.imshow(image_gray, 'gray'), plt.title('Original')
plt.subplot(132), plt.imshow(limiarizada, 'gray'), plt.title('Limiarizada')
plt.subplot(133), plt.imshow(morfologica, 'gray'), plt.title('Morfologica')
plt.show()

# Impressão do valor do limiar e do elemento estruturante
print(f"O valor do limiar escolhido pelo método de Otsu é: {valor_limiar}")
print(f"O elemento estruturante aplicado é um kernel quadrado 5x5:\n {kernel}")
