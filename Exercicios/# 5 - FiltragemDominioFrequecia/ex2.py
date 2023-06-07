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


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# Função para converter a imagem para níveis de cinza
def converte_para_cinza(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


# Função para aplicar o ruído gaussiano na imagem


def aplica_ruido_gaussiano(imagem, media=0, desvio_padrao=10):
    ruido = np.random.normal(media, desvio_padrao, imagem.shape)
    imagem_com_ruido = np.clip(imagem + ruido, 0, 255).astype(np.uint8)
    return imagem_com_ruido


# Função para calcular a DFT e retornar o espectro de Fourier com o deslocamento da origem


def calcula_dft(imagem):
    dft = np.fft.fft2(imagem)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


# Função para aplicar o filtro passa-baixa ideal


# Função para aplicar o filtro passa-baixa ideal
def filtro_passa_baixa_ideal(dft_shift, raio):
    # Obter as dimensões da imagem (altura e largura)
    altura, largura = dft_shift.shape

    # Encontrar o ponto central (origem) da imagem
    centro_x, centro_y = largura // 2, altura // 2

    # Criar uma matriz de zeros (máscara) com as mesmas dimensões da imagem
    mascara = np.zeros((altura, largura), np.uint8)

    # Desenhar um círculo preenchido com valor 255 (branco) na máscara, centrado na origem e com raio especificado
    cv2.circle(mascara, (centro_x, centro_y), raio, 255, -1)

    # Multiplicar a matriz da DFT deslocada pela máscara, eliminando as frequências fora do círculo
    return mascara * dft_shift


# Função para aplicar o filtro passa-baixa Butterworth


def filtro_passa_baixa_butterworth(dft_shift, raio, ordem):
    # Obter as dimensões da imagem (altura e largura)
    altura, largura = dft_shift.shape

    # Encontrar o ponto central (origem) da imagem
    centro_x, centro_y = largura // 2, altura // 2

    # Criar uma matriz de zeros (máscara) com as mesmas dimensões da imagem
    mascara = np.zeros((altura, largura), np.float32)

    # Preencher a matriz da máscara com os valores do filtro Butterworth
    for y in range(altura):
        for x in range(largura):
            # Calcular a distância entre o ponto (x, y) e o centro da imagem
            distancia = np.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)

            # Calcular o valor do filtro Butterworth para o ponto (x, y) e atribuir à máscara
            mascara[y, x] = 1 / (1 + (distancia / raio) ** (2 * ordem))

    # Multiplicar a matriz da DFT deslocada pela máscara, aplicando o filtro Butterworth
    return mascara * dft_shift


# Função para calcular a inversa da DFT e obter a imagem reconstruída


def calcula_idft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)
    imagem_reconstruida = np.fft.ifft2(dft_ishift)
    return np.abs(imagem_reconstruida)


# Carregar a imagem
caminho_atual = os.path.abspath(os.path.dirname(__file__))
imagem = cv2.imread(caminho_atual + "\\exercicio2.png")

# Converter a imagem para níveis de cinza
imagem_cinza = converte_para_cinza(imagem)

# Aplicar ruído gaussiano na imagem
imagem_com_ruido = aplica_ruido_gaussiano(imagem_cinza)

# Calcular a DFT da imagem com ruído
dft_shift = calcula_dft(imagem_com_ruido)

# Aplicar os filtros no domínio da frequência
dft_filtrado_ideal = filtro_passa_baixa_ideal(dft_shift, 30)
dft_filtrado_butterworth = filtro_passa_baixa_butterworth(dft_shift, 30, 2)

# Calcular a inversa da DFT para obter as imagens reconstruídas
imagem_reconstruida_ideal = calcula_idft(dft_filtrado_ideal)
imagem_reconstruida_butterworth = calcula_idft(dft_filtrado_butterworth)

# Exibir as imagens e os espectros de Fourier
plt.subplot(331), plt.imshow(imagem_cinza, cmap="gray")
plt.title("Imagem Original"), plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(imagem_com_ruido, cmap="gray")
plt.title("Imagem com Ruído"), plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(np.log(1 + np.abs(dft_shift)), cmap="gray")
plt.title("Espectro de Fourier"), plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(imagem_reconstruida_ideal, cmap="gray")
plt.title("Filtragem Ideal"), plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(imagem_reconstruida_butterworth, cmap="gray")
plt.title("Filtragem Butterworth"), plt.xticks([]), plt.yticks([])

plt.show()

#
# O filtro passa-baixa Butterworth é mais suave que o filtro ideal, e sua resposta em frequência é mais suave,
# o que pode resultar em uma melhor performance para eliminar ruídos.
#
# A função filtro_passa_baixa_butterworth implementa esse filtro. O parâmetro 'ordem' define a "suavidade" da transição
# entre as frequências preservadas e as eliminadas. Um valor maior de ordem resulta em uma transição mais acentuada.
