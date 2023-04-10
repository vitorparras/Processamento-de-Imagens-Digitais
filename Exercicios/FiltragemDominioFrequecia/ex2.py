# ex 2
# Para facilitar a elaboração dessa atividade, converta a
# imagem abaixo para níveis de cinza, 8 bits de quantização.
# Aplique o ruído gaussiano sobre a imagem. Em seguida,
# aplique a DFT sobre a imagem com ruído. Apresente o
# espectro de Fourier com o deslocamento da origem do
# plano de frequências. Proponha dois filtros no domínio da
# frequência. O objetivo é suavizar o ruído inserido
# previamente. Apresente os espectros de Fourier antes e após
# a etapa de processamento, bem como as imagens
# reconstruídas após cada processo de filtragem. Explique
# detalhadamente cada etapa e os filtros propostos. Indique
# qual foi o filtro que forneceu o melhor resultado em termos
# de minimização da presença. É permitido o uso de pacote
# DFT, disponível em ferramentas de PDI, a fim de facilitar o
# processamento da transformada e exibição de cada espectro.
# Não é permitido o uso de filtros disponíveis em ferramentas
# de PDI.


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Função para converter uma imagem colorida em escala de cinza


def converter_para_cinza(imagem):
    # Utiliza a função cvtColor do OpenCV para converter a imagem para escala de cinza
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Função para adicionar ruído gaussiano a uma imagem


def adicionar_ruido_gaussiano(imagem, media=0, desvio_padrao=25):
    # Utiliza a função add do OpenCV para adicionar ruído gaussiano à imagem
    return cv2.add(imagem, cv2.randn(np.zeros(imagem.shape), media, desvio_padrao))

# Função para calcular a Transformada Discreta de Fourier (DFT) de uma imagem


def calcular_dft(imagem):
    # Utiliza a função dft do OpenCV para calcular a DFT da imagem
    dft = cv2.dft(np.float32(imagem), flags=cv2.DFT_COMPLEX_OUTPUT)
    # Desloca a DFT para centralizar as frequências baixas
    dft_deslocado = np.fft.fftshift(dft)
    return dft_deslocado

# Função para aplicar um filtro no espectro de Fourier deslocado


def aplicar_filtro(dft_deslocado, filtro):
    # Multiplica o espectro de Fourier deslocado pelo filtro
    dft_filtrado = dft_deslocado * filtro[:, :, np.newaxis]
    return dft_filtrado

# Função para calcular a Transformada Inversa de Fourier (IDFT) de um espectro de Fourier filtrado


def idft(dft_filtrado):
    # Desloca o espectro de Fourier filtrado para a posição original
    idft_deslocado = np.fft.ifftshift(dft_filtrado)
    # Utiliza a função idft do OpenCV para calcular a IDFT do espectro de Fourier filtrado
    return cv2.idft(idft_deslocado)

# Função para criar um filtro passa-baixa de média


def criar_filtro_media(imagem_shape):
    # Cria uma matriz de uns com o mesmo tamanho da imagem e divide por sua área
    filtro = np.ones(imagem_shape) / (imagem_shape[0] * imagem_shape[1])
    return filtro

# Função para criar um filtro passa-baixa gaussiano


def criar_filtro_gaussiano(imagem_shape, sigma):
    # Cria um espaço linear no intervalo [-1, 1] com o mesmo número de elementos que as dimensões da imagem
    x = np.linspace(-1, 1, imagem_shape[1])
    y = np.linspace(-1, 1, imagem_shape[0])
    # Cria uma grade de coordenadas a partir dos espaços lineares
    X, Y = np.meshgrid(x, y)
    # Calcula a função gaussiana usando as coordenadas da grade
    filtro = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return filtro


caminho_atual = os.path.abspath(os.path.dirname(__file__))
caminho_imagem = caminho_atual + "\\exercicio2.png"

# Lê a imagem do arquivo
imagem = cv2.imread(caminho_imagem)

# Converte a imagem para escala de cinza
imagem_cinza = converter_para_cinza(imagem)

# Adiciona ruído gaussiano à imagem em escala de cinza
imagem_ruidosa = adicionar_ruido_gaussiano(imagem_cinza)

# Calcula a Transformada Discreta de Fourier (DFT) da imagem ruidosa
dft_deslocado = calcular_dft(imagem_ruidosa)

# Cria um filtro passa-baixa de média
filtro_media = criar_filtro_media(imagem_ruidosa.shape)

# Aplica o filtro de média no espectro de Fourier deslocado
dft_filtrado1 = aplicar_filtro(dft_deslocado, filtro_media)

# Calcula a Transformada Inversa de Fourier (IDFT) da imagem filtrada com filtro de média
idft1 = idft(dft_filtrado1)

# Cria um filtro passa-baixa gaussiano
filtro_gaussiano = criar_filtro_gaussiano(imagem_ruidosa.shape, sigma=0.5)

# Aplica o filtro gaussiano no espectro de Fourier deslocado
dft_filtrado2 = aplicar_filtro(dft_deslocado, filtro_gaussiano)

# Calcula a Transformada Inversa de Fourier (IDFT) da imagem filtrada com filtro gaussiano
idft2 = idft(dft_filtrado2)

# Exibir resultados
plt.figure()
plt.subplot(221), plt.imshow(imagem_cinza, cmap='gray')
plt.title('Imagem Original em Cinza'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(imagem_ruidosa, cmap='gray')
plt.title('Imagem com Ruído'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(cv2.magnitude(
    idft1[:, :, 0], idft1[:, :, 1]), cmap='gray')
plt.title('Imagem Filtrada (Média)'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(cv2.magnitude(
    idft2[:, :, 0], idft2[:, :, 1]), cmap='gray')
plt.title('Imagem Filtrada (Gaussiano)'), plt.xticks([]), plt.yticks([])
plt.show()
