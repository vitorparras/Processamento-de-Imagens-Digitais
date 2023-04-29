# 4. Dada a imagem (e) com e sem a presença de ruído gaussiano (exercício 3), compare-as por
# meio das métricas erro máximo, erro médio absoluto, erro médio quadrático, raiz do erro
# médio quadrático, erro médio quadrático normalizado e coeficiente de Jaccard para
# identificar os níveis de similaridades existentes.
# i) Observando os valores obtidos, é possível definir algum comportamento padrão entre as
# métricas a partir dos ruídos aplicados em cada imagem?
# ii) Neste cenário, qual métrica permite evidenciar melhor a degradação da imagem em razão
# da presença de ruído? Justifique sua resposta.


import os
import numpy as np
from skimage import io, util
from skimage.metrics import mean_squared_error
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

# Carregando a imagem original
caminho = os.path.abspath(os.path.dirname(__file__))+'\\Imagem_E.bmp'
image_orig = io.imread(caminho, as_gray=True)


# Adicionando ruído gaussiano
image_noise = util.random_noise(image_orig, mode='gaussian')

# Plotando as imagens
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Imagem original
axs[0].imshow(image_orig, cmap='gray')
axs[0].set_title('Original')
axs[0].axis('off')

# Imagem com ruído
axs[1].imshow(image_noise, cmap='gray')
axs[1].set_title('Gaussian Noise')
axs[1].axis('off')

plt.show()

# Função para cálculo das métricas


def calcula_metricas(img1, img2):
    mae = np.mean(np.abs(img1 - img2))  # Erro médio absoluto
    mse_val = mean_squared_error(img1, img2)  # Erro médio quadrático
    rmse_val = np.sqrt(mse_val)  # Raiz do erro médio quadrático
    max_err = np.max(np.abs(img1 - img2))  # Erro máximo
    # Erro médio quadrático normalizado
    nmse_val = mse_val / np.mean(np.square(img1))
    jaccard_val = jaccard_score(
        img1.flatten() > 0.5, img2.flatten() > 0.5)  # Coeficiente de Jaccard

    return max_err, mae, mse_val, rmse_val, nmse_val, jaccard_val


# Calculando e imprimindo as métricas
max_err, mae, mse_val, rmse_val, nmse_val, jaccard_val = calcula_metricas(
    image_orig, image_noise)

print('Erro Máximo:', max_err)
print('Erro Médio Absoluto:', mae)
print('Erro Médio Quadrático:', mse_val)
print('Raiz do Erro Médio Quadrático:', rmse_val)
print('Erro Médio Quadrático Normalizado:', nmse_val)
print('Coeficiente de Jaccard:', jaccard_val)


# Pergunta i: Observando os valores obtidos, é possível definir
# algum comportamento padrão entre as métricas a partir dos ruídos aplicados em cada imagem?

# R: Sim, é possível definir um comportamento padrão entre as métricas a partir dos ruídos aplicados
# em cada imagem. Todas as métricas de erro aumentam com a adição de ruído à imagem, o que indica uma
# maior diferença entre a imagem original e a imagem com ruído. Isso é esperado, já que o ruído altera
# os valores dos pixels da imagem original, aumentando assim todas as métricas de erro.

# Pergunta ii: Neste cenário, qual métrica permite evidenciar melhor a
# degradação da imagem em razão da presença de ruído? Justifique sua resposta.

# R: A métrica que melhor evidencia a degradação da imagem em razão da presença de ruído é
# o Erro Médio Quadrático. pois ele é utilizado para medir a diferença
# entre duas imagens porque considera a distância quadrática entre os pixels correspondentes
# nas duas imagens, enfatizando as discrepâncias maiores. Assim, ele é sensível a variações
# intensas de ruído, tornando-o particularmente útil para evidenciar a degradação da imagem
# devido à presença de ruído.
