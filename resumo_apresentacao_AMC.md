# Classificação Automática de Modulação (AMC) de Sinais de Radar com MLP

---

## 1. A Base de Dados: DeepRadar2022

- Dataset público criado especificamente para AMC de sinais de radar.
- Sinais I/Q (In-Phase e Quadrature) amostrados a 100 MHz, com formato
  **1024 × 2** (1024 amostras temporais, 2 canais).
- **23 classes de modulação** no total (LFM, famílias FSK, PSK, e diversos
  códigos de fase: Barker, Huffman, Frank, P1–P4, Px, Zadoff-Chu, T1–T4),
  além de classe de não-modulação (NM) e ruído puro (Noise).
- SNR variando de **-12 dB a +20 dB**, em passos de 2 dB — cobrindo desde
  cenários de ruído dominante até sinal bem mais forte que o ruído.
- Já vem dividido em conjuntos de treino (60%), validação e teste (20% cada).

### Por que reduzir para 11 classes?

O dataset completo soma **782.000 sinais**. Diante de limitações de hardware
e tempo de processamento, foi necessário reduzir o escopo — a mesma decisão
adotada pelos autores do artigo de referência. Foram selecionadas as
**11 classes mais representativas** dos principais tipos de modulação usados
em radar:

| Selecionadas no projeto | Descartadas (por limitação de recursos) |
|---|---|
| LFM | 2FSK, 4FSK, 8FSK |
| FM_Costas | 4PSK, 8PSK |
| 2PSK (BPSK) | PM_Huffman |
| PM_Barker | PM_Px, PM_ZadoffChu |
| PM_Frank | PM_T1, PM_T2, PM_T3, PM_T4 |
| PM_P1, PM_P2, PM_P3, PM_P4 | |
| NM (não-modulado) | |
| Noise (ruído puro) | |

Após o filtro, o conjunto de dados ficou com:

| Conjunto | Antes | Depois (11 classes) |
|---|---|---|
| Treino | 469.200 | 224.400 |
| Validação | 156.400 | 74.800 |
| Teste | 156.400 | 74.800 |

---

## 2. Pipeline de Processamento

1. **Carregamento** dos arquivos `.mat` (formato HDF5 para os sinais, formato
   padrão MATLAB para rótulos e metadados).
2. **Filtragem** das 11 classes de interesse e **remapeamento** dos rótulos
   originais (0–22) para o intervalo sequencial (0–10), exigido pela camada
   de saída da rede.
3. **Análise exploratória (EDA)**: estatísticas de amplitude por classe,
   distribuição de classes e de SNR, visualização de sinais I/Q no tempo,
   espectro de potência por modulação.
4. **Engenharia de features**: ao invés de alimentar o MLP com o sinal bruto
   achatado, foram extraídas características que tornam os padrões de cada
   modulação mais acessíveis para uma rede densa (ver Seção 4).
5. **Normalização (z-score)**, com média e desvio padrão calculados
   exclusivamente no conjunto de treino (evitando vazamento de dados).
6. **Treinamento** do MLP, com Cyclical Learning Rate e Early Stopping.
7. **Avaliação**: acurácia geral, relatório de classificação, matriz de
   confusão, e curvas de acurácia em função do SNR (geral e por modulação).

---

## 3. Por que um MLP não pode receber o sinal bruto diretamente

Um MLP processa cada entrada como um vetor de números independentes — ele não
tem noção de "ordem no tempo", diferente de uma LSTM (memória sequencial) ou
uma CNN (kernels que exploram vizinhança local). Isso é uma limitação
importante quando o que diferencia duas modulações é justamente o padrão de
variação ao longo do tempo (como nos códigos de fase polifásicos).

Para compensar essa limitação, em vez de simplesmente achatar o sinal
(1024×2 → vetor de 2048), foram extraídas features que tornam essas
propriedades temporais explícitas:

| Feature | O que captura | Tamanho |
|---|---|---|
| Amplitude instantânea | Envoltória do sinal | 1024 |
| Diferença de fase (frequência instantânea) | Variação da fase ponto a ponto | 1023 |
| Espectro de potência (FFT normalizada) | Conteúdo em frequência | 1024 |
| **Autocorrelação normalizada** | "Assinatura" de repetição do sinal — o que matematicamente distingue códigos polifásicos (Frank, P1–P4) entre si | 64 |

Total: **3.135 features** por amostra, normalizadas por z-score antes de
entrar na rede.

A adição da autocorrelação foi motivada por uma observação durante os testes:
mesmo com as primeiras features (amplitude, fase, espectro), o modelo ainda
tinha dificuldade significativa nos códigos polifásicos — uma limitação
também relatada no artigo de referência para LSTM e CNN.

---

## 4. Arquitetura do Modelo

MLP com 3 camadas ocultas densas, seguido de BatchNormalization e Dropout em
cada camada (para estabilizar o treinamento e reduzir overfitting):

```
Input (3135) → [Dense(512) → BatchNorm → Dropout(0.3)]
             → [Dense(256) → BatchNorm → Dropout(0.3)]
             → [Dense(128) → BatchNorm → Dropout(0.3)]
             → Dense(11, softmax)
```

- Otimizador: Adam, com Cyclical Learning Rate (1e-7 a 1e-3)
- Class weights balanceados (compensando classes mais difíceis de aprender)
- Early Stopping com paciência de 15 épocas, monitorando acurácia de validação
- O treinamento convergiu e parou automaticamente na **época 54** (early
  stopping), restaurando os pesos da melhor época (39).

---

## 5. Resultados

### 5.1 Desempenho geral no conjunto de teste

**Acurácia geral: 80,15%**

| Classe | Precisão | Recall | F1-score |
|---|---|---|---|
| NM | 0,990 | 0,998 | 0,994 |
| PM_Barker | 0,972 | 0,970 | 0,971 |
| FM_Costas | 0,951 | 0,954 | 0,952 |
| Noise | 0,904 | 0,963 | 0,932 |
| LFM | 0,910 | 0,890 | 0,900 |
| 2PSK | 0,909 | 0,859 | 0,883 |
| PM_P3 | 0,708 | 0,703 | 0,706 |
| PM_Frank | 0,670 | 0,711 | 0,690 |
| PM_P4 | 0,628 | 0,549 | 0,585 |
| PM_P1 | 0,652 | 0,515 | 0,575 |
| PM_P2 | 0,550 | 0,706 | 0,618 |

### 5.2 SNR mínimo para manter 90% de acurácia

| Critério | Resultado |
|---|---|
| **SNR mínimo (acurácia geral)** | **8 dB** |
| Modulação limitante | **PM_P1** (16 dB) |

Acurácia mínima por modulação (dB):

| Modulação | SNR mín. (dB) |
|---|---|
| NM | -12 |
| Noise | -12 |
| PM_Barker | -8 |
| LFM | -4 |
| FM_Costas | -4 |
| 2PSK | 0 |
| PM_P3 | 6 |
| PM_Frank | 8 |
| PM_P4 | 14 |
| PM_P1 | 16 |
| PM_P2 | 16 |

### 5.3 Padrão observado

- Modulações estruturalmente distintas entre si (LFM, FM_Costas, 2PSK,
  PM_Barker, NM, Noise) atingem acurácia próxima de 100% rapidamente, mesmo
  em SNR moderado/baixo.
- As modulações que permanecem mais difíceis são justamente os **códigos de
  fase polifásicos** (Frank, P1, P2, P3, P4) — eles compartilham estatísticas
  de amplitude muito parecidas e se diferenciam por propriedades sutis de
  autocorrelação na sequência de fase.
- Esse padrão **coincide com o relatado no artigo de referência**: tanto o
  LSTM quanto o CNN do paper original também apresentaram mais dificuldade
  exatamente nesse mesmo subconjunto de modulações (Frank e códigos P).

---

## 6. Conclusão

- O MLP, mesmo sem capacidade nativa de capturar estrutura temporal, alcançou
  desempenho satisfatório (80% de acurácia geral) ao ser alimentado com
  features manuais que tornam explícitas as propriedades relevantes do sinal
  (amplitude, fase, espectro, autocorrelação).
- A principal limitação encontrada — dificuldade em distinguir códigos
  polifásicos — **confirma uma fragilidade já relatada no artigo de
  referência**, mesmo em arquiteturas mais sofisticadas (LSTM/CNN), reforçando
  que essa é uma dificuldade intrínseca da tarefa e não apenas da arquitetura
  escolhida.
- O resultado reforça a importância de arquiteturas que preservam estrutura
  sequencial (LSTM) ou local (CNN) para extrair o máximo de informação
  discriminativa desse tipo de sinal — e, ao mesmo tempo, demonstra que
  engenharia de features pode reduzir consideravelmente essa desvantagem em
  arquiteturas mais simples como o MLP.

---

## 7. Possíveis trabalhos futuros

- Comparar diretamente o MLP com LSTM/CNN treinados sob as mesmas condições
  (mesmo subconjunto de 11 classes, mesmo hardware).
- Investigar `max_lag` da autocorrelação alinhado ao comprimento real das
  sequências de fase de cada código polifásico.
- Avaliar arquiteturas híbridas (ex: CNN 1D rasa + MLP) como meio-termo entre
  simplicidade computacional e capacidade de capturar estrutura local.
