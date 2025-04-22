#==============================================================================
# INTELIGÊNCIA ARTIFICIAL APLICADA
# REDES NEURAIS - SEMANA 2
# REDE NEURAL MLP (MULTI-LAYER PERCEPTRON)
# PROF. EDSON RUSCHEL
#==============================================================================
#------------------------------------------------------------------------------
# IMPORTAÇÃO DE BIBLIOTECAS
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#------------------------------------------------------------------------------
# CRIAÇÃO DO MODELO MLP

model = Sequential()

#------------------------------------------------------------------------------
# DEFINIÇÃO DA QUANTIDADE DE NEURÔNIOS DAS CAMADAS

# Quantidade de neurônios da Camada Oculta 1
n1 = 10000

# Quantidade de neurônios da Camada de Saída
ns = 10

#------------------------------------------------------------------------------
# DEFINIÇÃO DAS FUNÇÕES DE ATIVAÇÃO
    # relu    -> Rectified Linear Unit (Unidade Linear Retificada)
    # sigmoid -> sigmoid(x) = 1 / (1 + exp(-x))
    # tanh    -> Tangente hiperbólica
    # softmax -> Utilizada na camada de saída
    
# Função de Ativação da Camada Oculta 1
fa1 = 'relu'

# Função de Ativação da Camada de Saída
fas = 'softmax'

#------------------------------------------------------------------------------
# ADIÇÃO DE CAMADAS À REDE NEURAL

# Primeira Camada Oculta (o input_dim é a própria camada de entrada)
model.add(Dense(units=n1, activation=fa1, input_dim=784))

# Camada de Saída
model.add(Dense(units=ns, activation=fas))

#==============================================================================
#------------------------------------------------------------------------------
# DEFINIÇÃO DA FUNÇÃO DE PERDA
    # mean_squared_error       -> Erro quadrático médio.
    # binary_crossentropy      -> Entropia cruzada binária.
    # categorical_crossentropy -> Entropia cruzada categórica.
    # mean_absolute_error      -> Erro absoluto médio.
    
fp = 'mean_absolute_error'

#------------------------------------------------------------------------------
# DEFINIÇÃO DO OTIMIZADOR
    # sgd     -> Descida de gradiente estocástico (SGD).
    # adam    -> SGD com adaptação de taxa de aprendizado.
    # rmsprop -> Baseado em Root Mean Square Propagation.
    # adagrad -> Adapta a taxa de aprendizado para cada parâmetro.
    
otimizador = 'adam'

#------------------------------------------------------------------------------
# DEFINIÇÃO DAS MÉTRICAS
    # accuracy            -> Acurácia.
    # precision           -> Precisão.
    # recall              -> Revocação.
    # f1-score            -> Pontuação F1.
    # mean_squared_error  -> Erro quadrático médio.
    # mean_absolute_error -> Erro absoluto médio.
    
metrica = 'accuracy'

#------------------------------------------------------------------------------
# COMPILAÇÃO DO MODELO

model.compile(loss=fp, optimizer=otimizador, metrics=[metrica])

#==============================================================================
#------------------------------------------------------------------------------
# CARREGAMENTO DOS DADOS DE TREINAMENTO E TESTE

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

#------------------------------------------------------------------------------
# PRÉ-PROCESSAMENTO DOS DADOS

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
# TRANSFORMAÇÃO DOS RÓTULOS EM CODIFICAÇÃO ONE-HOT

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=10)

#------------------------------------------------------------------------------
# DEFINIÇÃO DO PERCENTUAL DO CONJUNTO DE DADOS DE TESTE

dados_teste = 0.90 # 0.90 significa 90% para teste e 10% para treinamento

#------------------------------------------------------------------------------
# DIVISÃO DOS DADOS EM CONJUNTOS DE TREINAMENTO E TESTE

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=dados_teste, random_state=42)

#------------------------------------------------------------------------------
# DEFINIÇÃO DO NÚMERO DE ÉPOCAS E NÚMERO DE AMOSTRAS

epocas = 2
amostras = 256

#------------------------------------------------------------------------------
# TREINAMENTO DA REDE NEURAL MLP

model.fit(x_train, y_train, epochs = epocas, batch_size = amostras)

#------------------------------------------------------------------------------
# AVALIAÇÃO DO MODELO (PERDA E ACURÁCIA)

loss, accuracy = model.evaluate(x_test, y_test)

#==============================================================================
#------------------------------------------------------------------------------
# REALIZAÇÃO DE PREVISÕES COM DADOS ORIGINAIS

predictions = model.predict(x_test)

#------------------------------------------------------------------------------
# SELEÇÃO DO NÚMERO DE AMOSTRAS PARA TESTE

# Selecione valores maiores que zero para testar
amostras_previsao = 20

# Sorteio de amostras de acordo com o número de amostras (amostras_previsao)
samples = np.random.choice(len(x_test), amostras_previsao)
idx=0

#------------------------------------------------------------------------------
# EXIBIÇÃO DAS IMAGENS E PREVISÕES CORRESPONDENTES

for idx in samples:
    image = x_test[idx].reshape((28, 28))  # Redimensionar a imagem para o formato original
    label_true = np.argmax(y_test[idx])  # Rótulo do número real
    label_pred = np.argmax(predictions[idx])  # Rótulo do número previsto
    plt.imshow(image, cmap='gray')
    plt.title(f'Real: {label_true}, Reconhecido: {label_pred}')
    plt.axis('off')
    plt.show(block=True)

#------------------------------------------------------------------------------
# Exibir a Perda e a Precisão
print(f"Função de Perda: {loss:.4f}")
print(f"Precisão: {accuracy:.4f}")