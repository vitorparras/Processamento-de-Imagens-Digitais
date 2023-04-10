# Escreva um programa que receba as imagens abaixo, converta para o padrão HSI e
# aplique a equalização de histograma. Apresentar os histogramas equalizados.

import cv2
import numpy as np
import os

# Função para converter imagem RGB para HSI


def rgb_para_hsi(imagem):
    # Separa os canais de cor vermelho (R), verde (G) e azul (B) da imagem
    r, g, b = cv2.split(imagem)

    # Calcula a intensidade (I) como a média dos três canais de cor
    intensidade = (r + g + b) / 3

    # Calcula o valor mínimo entre os três canais de cor
    valor_minimo = np.minimum(r, np.minimum(g, b))

    # Calcula a saturação (S) usando a fórmula S = 1 - (3 / (R + G + B)) * valor mínimo
    saturacao = 1 - (3 / (r + g + b + 1e-6)) * valor_minimo

    # Calcula o ângulo que será usado para calcular o matiz (H)
    raiz_quadrada = np.sqrt((r - g) * (r - g) + (r - b) * (g - b))
    angulo = np.arccos((0.5 * ((r - g) + (r - b))) / (raiz_quadrada + 1e-6))

    # Calcula o matiz (H) convertendo o ângulo para graus
    matiz = angulo * (180 / np.pi)

    # Ajusta o matiz caso o valor de G seja menor que B
    matiz[g < b] = 360 - matiz[g < b]

    # Normaliza o matiz dividindo-o por 360
    matiz /= 360

    # Mescla os canais H, S e I de volta em uma única imagem HSI
    imagem_hsi = cv2.merge((matiz, saturacao, intensidade))
    return imagem_hsi

# Função para converter imagem HSI para RGB


def hsi_para_rgb(imagem_hsi):
    h, s, i = cv2.split(imagem_hsi)
    h = h * 2 * np.pi

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # R > G > B
    mask = (h < 2 * np.pi / 3)
    b[mask] = i[mask] * (1 - s[mask])
    r[mask] = i[mask] * (1 + s[mask] * np.cos(h[mask]) /
                         np.cos(np.pi / 3 - h[mask]))
    g[mask] = 3 * i[mask] - (r[mask] + b[mask])

    # G > R > B
    mask = (h >= 2 * np.pi / 3) & (h < 4 * np.pi / 3)
    h_temp = h[mask] - 2 * np.pi / 3
    r[mask] = i[mask] * (1 - s[mask])
    g[mask] = i[mask] * (1 + s[mask] * np.cos(h_temp) /
                         np.cos(np.pi / 3 - h_temp))
    b[mask] = 3 * i[mask] - (r[mask] + g[mask])

    # B > R > G
    mask = (h >= 4 * np.pi / 3)
    h_temp = h[mask] - 4 * np.pi / 3
    g[mask] = i[mask] * (1 - s[mask])
    b[mask] = i[mask] * (1 + s[mask] * np.cos(h_temp) /
                         np.cos(np.pi / 3 - h_temp))
    r[mask] = 3 * i[mask] - (g[mask] + b[mask])

    imagem_rgb = cv2.merge((r, g, b)).clip(0, 1) * 255
    return imagem_rgb.astype(np.uint8)


# Função para aplicar equalização de histograma na intensidade (I) do espaço HSI


def equalizar_histograma(imagem):
    # Separa os canais H, S e I da imagem HSI
    h, s, i = cv2.split(imagem)

    # Normaliza o canal I no intervalo [0, 255] usando a função cv2.normalize
    i = cv2.normalize(i, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Aplica equalização de histograma no canal I normalizado usando a função cv2.equalizeHist
    intensidade_equalizada = cv2.equalizeHist(i.astype(np.uint8))

    # Mescla os canais H, S e I equalizado de volta em uma única imagem HSI
    hsi_equalizado = cv2.merge(
        (h.astype(np.uint8), s.astype(np.uint8), intensidade_equalizada))

    # Converte a imagem HSI equalizada para float32 e normaliza-a
    hsi_equalizado = hsi_equalizado.astype(np.float32) / 255
    return hsi_equalizado


# Função principal que carrega a imagem, realiza as conversões e exibe os resultados


def main(nome):
    # Carrega a imagem de entrada a partir do arquivo 'input_image.jpg'
    imagem_entrada = cv2.imread(nome+'.jpg')

    # Converte a imagem de entrada para o espaço HSI usando a função rgb_para_hsi
    imagem_hsi = rgb_para_hsi(imagem_entrada)

    # Aplica equalização de histograma na imagem HSI usando a função equalizar_histograma
    imagem_hsi_equalizada = equalizar_histograma(imagem_hsi)

    # Converte a imagem HSI equalizada de volta para o espaço de cores RGB usando a função hsi_para_rgb
    imagem_rgb_equalizada = hsi_para_rgb(imagem_hsi_equalizada)

    # Salva a imagem RGB equalizada no arquivo 'rgb_equalized_image.jpg'
    cv2.imwrite('rgb_equalized_image.jpg', imagem_rgb_equalizada)

    # Exibe a imagem de entrada e a imagem RGB equalizada em janelas separadas
    cv2.imshow('Imagem de Entrada', imagem_entrada)
    cv2.imshow('Imagem RGB Equalizada', imagem_rgb_equalizada)

    # Aguarda o usuário pressionar uma tecla para fechar as janelas
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Executa a função principal quando o script é executado
if __name__ == '__main__':

    caminho_atual = os.path.abspath(os.path.dirname(__file__))

    main(caminho_atual+'\\Img1')
    main(caminho_atual+'\\Img2')
    main(caminho_atual+'\\Img3')
