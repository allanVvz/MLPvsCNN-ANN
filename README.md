# Análise Técnica Comparativa: MLP vs CNN no MNIST

---

## 1. Multilayer Perceptron (MLP)

### 1.1. Evolução Arquitetural e Hiperparâmetros

| Teste | Camadas Ocultas | Ativação            | Épocas | Otimizador | Batch size | Acurácia Validação (%) | Observações                               |
| ----- | --------------- | ------------------- | ------ | ---------- | ---------- | ---------------------- | ----------------------------------------- |
| 1     | 1×(n=8)         | sigmoid             | 5      | RMSprop    | 512        | 28.0                   | Underfitting; capacidade insuficiente     |
| 2     | 1×(n=50)        | sigmoid             | 5      | RMSprop    | 512        | 79.0                   | Aumento de neurônios → ganho expressivo   |
| 4     | 1×(n=784)       | softmax / cross‑ent | 2      | RMSprop    | 512        | 86.22                  | Transição para softmax + cross‑entropy    |
| 7     | 2×(784→360)     | ReLU                | 5      | RMSprop    | 512        | 92.24                  | ReLU reduz saturação, +6 p.p. de acurácia |
| 9     | 2×(784→360)     | ReLU                | 10     | RMSprop    | 512        | 96.20                  | +5 épocas consolidam aprendizado          |
| 10    | 2×(784→360)     | ReLU                | 30     | RMSprop    | 512        | **96.72**              | Overfitting discreto; performance máxima  |

#### Figura 1. Curvas de treinamento e validação do MLP (Teste 10)


### 1.2. Análise Técnica (Causa e Efeito)

- **Capacidade de Representação:** ampliar de 8→784 neurônios (Teste 1→4) elevou a acurácia de ~27% a 86.2%, saindo do underfitting.
- **Profundidade vs Saturação:** inserir segunda camada (n2=360) com ReLU (Teste 4→7) mitigou gradientes saturados do sigmoid, acelerando convergência em +6 p.p. de acurácia.
- **Épocas de Treino:** estender de 5→30 épocas (Testes 7→10) refinou os pesos; a 10ᵃ época (Teste 9) consolidou aprendizado e a 30ᵃ (Teste 10) adicionou +0.5 p.p., mas aumentou val_loss (overfitting).
- **Otimizador & Batch:** RMSprop mostrou estabilidade; batches muito pequenos (<64) geram ruído, enquanto 512 equilibra estabilidade e velocidade.

**🔑 Conclusão MLP:** Melhor configuração (Teste 10):

- Arquitetura: Dense(784, ReLU) → Dense(360, ReLU) → Dense(10, Softmax)
- Épocas: 30 | Batch size: 512 | Otimizador: RMSprop
- Acurácia Validação: **96.72%**

---

## 2. Convolutional Neural Network (CNN)

### 2.1. Evolução de Camadas e Hiperparâmetros

| Teste | Convs (kernel)     | Filtros C1→C2 | Dense pós-Conv | Ativação | Épocas | % Treino | Otimizador + Perda  | Acurácia Validação (%) | Observações                                |
| ----- | ------------------ | ------------- | -------------- | -------- | ------ | -------- | ------------------- | ---------------------- | ------------------------------------------ |
| 1     | 9×9 (tanh/sigmoid) | 2→2           | 2              | tanh     | 2      | 10%      | Adadelta + hinge    | ~9.0                   | Underfitting extremo                       |
| 3     | 3×3 (tanh/sigmoid) | 4→8           | 10             | tanh     | 5      | 10%      | RMSprop + cross-ent | ~25                    | Saturação ainda presente                   |
| 6     | 3×3 (ReLU)         | 32→64         | 10             | ReLU     | 10     | 70%      | RMSprop + cross-ent | ~97.8                  | Deep stack + melhora generalização         |
| 8     | 3×3 (ReLU)         | 32→64         | 256            | ReLU     | 15     | 70%      | RMSprop + cross-ent | **98.61**              | Melhor balanço capacidade vs generalização |

#### Figura 2. Curvas de treinamento e validação da CNN (Teste 8)


### 2.2. Análise Técnica (Causa e Efeito)

- **Kernel 3×3 vs 9×9:** kernels menores preservam detalhes locais, aumentando acurácia em ~15 p.p. ao migrar Teste 1→3.
- **Filtros Expandidos:** de 4→32 e 8→64 (Testes 3→6) ampliou a extração de features, elevando acurácia de ~25% a ~97.8%.
- **Proporção de Treino:** 10%→70% de treino (Teste 3→6) reduziu variação, melhorando generalização.
- **Batch & Épocas:** batch 256 + 15 épocas (Teste 8) refinou padrões finos, atingindo 98.61% com curvas estáveis.

**🔑 Conclusão CNN:** Melhor configuração (Teste 8):

- Arquitetura: [Conv3×3→ReLU→Pool]×2 → Flatten → Dense(256, ReLU) → Dense(10, Softmax)
- Épocas: 15 | Batch size: 256 | Otimizador: RMSprop
- Acurácia Validação: **98.61%**

---

## 3. Comparativo Técnico MLP vs CNN

| Métrica             | MLP (Teste 10) | CNN (Teste 8)   | Diferença  | Interpretação Técnica                            |
| ------------------- | -------------- | --------------- | ---------- | ------------------------------------------------ |
| Acurácia Validação  | 96.72%         | 98.61%          | +1.89 p.p. | CNN captura hierarquia espacial de pixels melhor |
| Épocas Convergência | 30             | 15              | –50%       | CNN converge mais rápido por parâmetros locais   |
| Generalização       | <60% (Conj. 2) | ~70% (Conj. 2)  | +10 p.p.   | CNN generaliza melhor em dados fora do treino    |

> **Insight:** Convoluções 3×3 + pooling permitem à CNN extrair padrões locais antes de combinações densas, resultando em acurácia superior e melhor generalização com menos épocas, enquanto a MLP requer maior profundidade densa e mais ciclos de treino.

---

*Este documento foi revisado para manter fidelidade ao conteúdo original, remover marcações internas e uniformizar formatação e arquitetura.*

