# -*- coding: utf-8 -*-

#==============================================================================
# INTELIGENCIA ARTIFICIAL APLICADA
# REDES NEURAIS - SEMANA 4
# REDE NEURAL MLP (MULTI-LAYER PERCEPTRON)
# PROF. EDSON RUSCHEL
#==============================================================================
#------------------------------------------------------------------------------
# IMPORTACAO DE BIBLIOTECAS
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from PIL import Image

#------------------------------------------------------------------------------
# CRIACAO DO MODELO MLP
model = Sequential()

#------------------------------------------------------------------------------
# DEFINICAO DA QUANTIDADE DE NEURONIOS DAS CAMADAS
n1 = 784  # Quantidade de neuronios da Camada Oculta 1
n2 = 360  # Quantidade de neuronios da Camada Oculta 2
#n3 = 1000  # Quantidade de neuronios da Camada Oculta 3
ns = 10  # Quantidade de neuronios da Camada de Sai­da

#------------------------------------------------------------------------------
# DEFINICAO DAS FUNCOES DE ATIVACAO
    # relu    -> Rectified Linear Unit (Unidade Linear Retificada)
    # sigmoid -> sigmoid(x) = 1 / (1 + exp(-x))
    # tanh    -> Tangente hiperbolica
    # softmax -> Utilizada na camada de sai­da
    
# Selecione a Funcao de Ativacao da Camada Oculta 1
fa1 = 'relu'

# Selecione a Funcao de Ativacao da Camada Oculta 2
fa2 = 'relu'

# Selecione a Funcao de Ativacao da Camada Oculta 2
fa3 = 'relu'

# Selecione a Funcao de Ativacao da Camada de Sai­da
fas = 'softmax'

#------------------------------------------------------------------------------
# ADICAO DE CAMADAS A REDE NEURAL

# Primeira Camada Oculta:
# Com 784 neuronios na camada de entrada -> 28x28 pixels (tamanho das imagens)
model.add(Dense(units=n1, activation=fa1, input_dim=784))

# Segunda Camada Oculta (desabilitada por padrao)
model.add(Dense(units=n2, activation=fa2))

# Segunda Camada Oculta (desabilitada por padrao)
#model.add(Dense(units=n3, activation=fa3))

# Camada de Sai­da
model.add(Dense(units=ns, activation=fas))

#==============================================================================
#------------------------------------------------------------------------------
# DEFINICAO DA FUNCAO DE PERDA
    # mean_squared_error       -> Erro quadratico medio.
    # binary_crossentropy      -> Entropia cruzada binaria.
    # categorical_crossentropy -> Entropia cruzada categorica.
    # mean_absolute_error      -> Erro absoluto medio.
    
# INSIRA ABAIXO A FUNCAO DE PERDA DE SUA ESCOLHA
fp = 'categorical_crossentropy'

#------------------------------------------------------------------------------
# DEFINICAO DO OTIMIZADOR
    # sgd     -> Descida de gradiente estocastico (SGD).
    # adam    -> SGD com adaptacao de taxa de aprendizado.
    # rmsprop -> Baseado em Root Mean Square Propagation.
    # adagrad -> Adapta a taxa de aprendizado para cada parametro.
    
# INSIRA ABAIXO O OTIMIZADOR DE SUA ESCOLHA
otimizador = 'rmsprop'

#------------------------------------------------------------------------------
# DEFINICAO DA METRICA DE DESEMPENHO
    # accuracy            -> Acuracia.
    # mean_squared_error  -> Erro quadratico medio.
    # mean_absolute_error -> Erro absoluto medio.
    
# INSIRA ABAIXO A METRICA DE SUA ESCOLHA
metrica = 'accuracy'

#------------------------------------------------------------------------------
# COMPILACAO DO MODELO

model.compile(loss=fp, optimizer=otimizador, metrics=[metrica])

#==============================================================================
#------------------------------------------------------------------------------
# CARREGAMENTO DOS DADOS DE TREINAMENTO E TESTE

(x_train, y_train), (x_test, y_test) =\
    tensorflow.keras.datasets.mnist.load_data()

#------------------------------------------------------------------------------
# PRE-PROCESSAMENTO DOS DADOS

# Redimensionar as imagens para um vetor unidimensional
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Converter para tipo float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizar os valores dos pixels para o intervalo [0, 1]
x_train /= 255.0
x_test /= 255.0 

#------------------------------------------------------------------------------
# TRANSFORMACAO DOS ROTULOS EM CODIFICACAO ONE-HOT

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=ns)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=ns)

#------------------------------------------------------------------------------
# DEFINICAO DO PERCENTUAL DO CONJUNTO DE DADOS DE TESTE

dados_teste = 0.80 # 0.3 significa 30%

#------------------------------------------------------------------------------
# DIVISAO DOS DADOS EM CONJUNTOS DE TREINAMENTO E TESTE

x_train, x_test, y_train, y_test =\
    train_test_split(x_train, y_train, \
    test_size=dados_teste, random_state=42)

#------------------------------------------------------------------------------
# DEFINICAO DO NUMERO DE EPOCAS E NUMERO DE AMOSTRAS

epocas = 30
amostras = 512

#------------------------------------------------------------------------------
# TREINAMENTO DA REDE NEURAL MLP

print('\n' + '=' * 70)
print('INICIANDO O TREINAMENTO DO MODELO... \n')

model.fit(x_train, y_train, epochs = epocas, batch_size = amostras)

#==============================================================================
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
    image = image.reshape(1, 784)  # Redimensionar para (1, 784)

    # Normalizar a imagem
    image = image.astype('float32')
    image /= 255.0

    # Fazer a previsão da classe
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Exibir a imagem e as informações sobre o número real e previsto
    axs[i].imshow(np.squeeze(image.reshape(28, 28)), cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f'Real: {class_names[real_label]}\nPrevisto: {class_names[predicted_class]}')

plt.tight_layout()
plt.show()

#==============================================================================
# AVALIACAO DO MODELO (PERDA E METRICA DE DESEMPENHO)

print('\n' + '=' * 70)
print('CALCULANDO FUNCAO DE PERDA E METRICA DE DESEMPENHO...\n')
loss, metric = model.evaluate(x_test, y_test)

# Exibir a Perda e a PrecisÃ£o
print('\n' + '=' * 70)
print('*** DESEMPENHO DO MODELO APOS O TREINAMENTO ***\n')

print("Funcao de Perda utilizada: " + fp)
print("Valor obtido: " + f" = {loss:.4f}" + '\n')

print('-' * 70 + '\n')

print("Metrica de Desempenho utilizada: " + metrica)
print("Valor obtido: " + f" = {metric:.4f} \n")

print('=' * 70)

#==============================================================================