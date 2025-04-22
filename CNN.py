# -*- coding: utf-8 -*-

#==============================================================================
# INTELIGÊNCIA ARTIFICIAL APLICADA
# REDES NEURAIS - SEMANA 4
# REDE NEURAL CONVOLUCIONAL (CNN)
# PROF. EDSON RUSCHEL
#==============================================================================
#------------------------------------------------------------------------------
# IMPORTAÇÃO DE BIBLIOTECAS
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

#------------------------------------------------------------------------------
# DEFINIÇÃO DO PERCENTUAL DE DADOS PARA TREINAMENTO
percentual_treinamento = 0.7 # 0.7 significa 70% para treinamento e 30% para teste

#------------------------------------------------------------------------------
# Carregar dados MNIST
(x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

#------------------------------------------------------------------------------
# Determinar o tamanho do conjunto de treinamento com base na proporção especificada
train_size = int(len(x_train_full) * percentual_treinamento)

#------------------------------------------------------------------------------
# Dividir os dados em conjuntos de treinamento e teste
x_train = x_train_full[:train_size]
y_train = y_train_full[:train_size]
x_test = x_train_full[train_size:]
y_test = y_train_full[train_size:]

#------------------------------------------------------------------------------
# Pré-processamento dos dados
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#------------------------------------------------------------------------------
# DEFINIÇÃO DAS FUNÇÕES DE ATIVAÇÃO
    # relu: Rectified Linear Unit, retorna valor positivo e zero caso contrário.
    # sigmoid: Função logística, retorna valores entre 0 e 1.
    # tanh: Tangente hiperbólica, que retorna valores entre -1 e 1.
    # softmax: Normaliza as saídas em uma distribuição de probabilidade.

fa_C1 = 'relu' # função de ativação da primeira camada de convolução
fa_C2 = 'relu' # função de ativação da segunda camada de convolução
fa_D1 = 'relu' # função de ativação da camada densa oculta
fa_D2 = 'softmax' # função de ativação da camada densa de saída

#------------------------------------------------------------------------------
# DEFINIÇÃO DA ESTRUTURA DA REDE

# DEFINA O TAMANHO MxN DAS MATRIZES DE CONVOLUÇÃO
m_C1 = 3 # 9 significa uma matriz 9x9 para a primeira camada de convolução
m_C2 = 3 # 9 significa uma matriz 9x9 para a segunda camada de convolução

# DEFINA O NÚMERO DE NEURÔNIOS DAS CAMADAS DA REDE CNN
n_C1 = 32 # número de neurônios da primeira camada de convolução
n_C2 = 64 # número de neurônios da segunda camada de convolução
n_D1 = 10 # número de neurônios da camada densa oculta

#------------------------------------------------------------------------------
# CONSTRUÇÃO DO MODELO CNN
model = Sequential()

model.add(Conv2D(n_C1, (m_C1, m_C1), activation = fa_C1, input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(n_C2, (m_C2, m_C2), activation = fa_C2))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(n_D1, activation = fa_D1))
model.add(Dense(10, activation = fa_D2))

#------------------------------------------------------------------------------
# DEFINIÇÃO DO OTIMIZADOR
# adam
# sgd
# rmsprop
# adadelta
# adagrad
# adamax
# nadam

otimizador = 'rmsprop'

#------------------------------------------------------------------------------
# DEFINIÇÃO DA FUNÇÃO DE PERDA
# categorical_crossentropy
# binary_crossentropy
# mean_squared_error
# mean_absolute_error
# categorical_hinge
# logcosh

funcao_perda = 'categorical_crossentropy'

#------------------------------------------------------------------------------
# DEFINIÇÃO DA MÉTRICA DE DESEMPENHO
# accuracy
# mean_squared_error
# mean_absolute_error

metrica = 'accuracy'

#------------------------------------------------------------------------------
# Compilar o modelo
model.compile(optimizer = otimizador,
              loss = funcao_perda,
              metrics = [metrica])

#------------------------------------------------------------------------------
# DEFINIÇÃO DO NÚMERO DE ÉPOCAS E AMOSTRAS DE TREINAMENTO
epocas = 12
amostras = 512

#------------------------------------------------------------------------------
# Treinar o modelo
history = model.fit(x_train, y_train, \
                    epochs = epocas, \
                    batch_size = amostras, \
                    validation_data = (x_test, y_test))

#------------------------------------------------------------------------------
# Exibir resultados na aba "Plots"
plt.plot(history.history[metrica], label='Métrica de Treinamento')
plt.plot(history.history['val_' + metrica], label='Métrica de Validação')
plt.plot(history.history['loss'], label='Função de Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Função de Perda de Validação')
plt.xlabel('Épocas')
plt.ylabel('Métrica / Função de Perda')
plt.legend()
plt.show()

#==============================================================================
# Carregar e pré-processar as imagens de teste
HD = 'C' # Digite a letra da Unidade de Disco
pasta = 'Scripts\\Machine Learning\\data' # Digite a pasta onde estao as imagens

imagem0 = 'zero0b.png'
imagem1 = 'um1b.png'
imagem2 = 'dois2b.png'
imagem3 = 'tres3b.png'
imagem4 = 'quatro4b.png'
imagem5 = 'cinco5b.png'
imagem6 = 'seis6b.png'
imagem7 = 'sete7b.png'
imagem8 = 'oito8b.png'
imagem9 = 'nove9b.png'

image_paths = {
    HD + ':\\' + pasta + '\\' + imagem0 :0,
    HD + ':\\' + pasta + '\\' + imagem1 :1,
    HD + ':\\' + pasta + '\\' + imagem2 :2,
    HD + ':\\' + pasta + '\\' + imagem3 :3,
    HD + ':\\' + pasta + '\\' + imagem4 :4,
    HD + ':\\' + pasta + '\\' + imagem5 :5,
    HD + ':\\' + pasta + '\\' + imagem6 :6,
    HD + ':\\' + pasta + '\\' + imagem7 :7,
    HD + ':\\' + pasta + '\\' + imagem8 :8,
    HD + ':\\' + pasta + '\\' + imagem9 :9,
}

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

fig, axs = plt.subplots(2, 5, figsize=(8, 5))
axs = axs.flatten()

for i, (image_path, real_label) in enumerate(image_paths.items()):
    # Carregar a imagem
    image = Image.open(image_path).convert('L')  # Converter para escala de cinza
    image = image.resize((28, 28))  # Redimensionar para 28x28 pixels
    image = np.array(image)  # Converter para matriz numpy
    image = image.reshape(1, 28, 28, 1)  # Adicionar dimensão de lote e canal

    # Normalizar a imagem
    image = image.astype('float32')
    image /= 255.0

    # Fazer a previsão da classe
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Exibir a imagem e as informações sobre o número real e previsto
    axs[i].imshow(np.squeeze(image), cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f'Real: {class_names[real_label]}\nPrevisto: {class_names[predicted_class]}')

plt.tight_layout()
plt.show()

#==============================================================================
# AVALIACAO DO MODELO (PERDA E METRICA DE DESEMPENHO)

print('\n\n\n' + '=' * 70)
print('CALCULANDO FUNCAO DE PERDA E METRICA DE DESEMPENHO...\n')
loss, metric = model.evaluate(x_test, y_test)

print('\n' + '=' * 70)
print('*** DESEMPENHO DO MODELO APOS O TREINAMENTO ***\n')


print("Funcao de Perda utilizada: " + funcao_perda)
print("Valor obtido: " + f" = {loss:.4f}" + '\n')

print('-' * 70 + '\n')

print("Metrica de Desempenho utilizada: " + metrica)
print("Valor obtido: " + f" = {metric:.4f}\n")

print('=' * 70)

#==============================================================================