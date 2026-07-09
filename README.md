# Prova Substitutiva - Data Analytics Fase 5

Projeto de classificação de textos com duas abordagens NLP:

1. **Deep Learning** — Word2Vec pré-treinado + LSTM
2. **Word2Vec** — Gensim + média de vetores + Regressão Logística

Dataset utilizado: [BBC News (`bbc-text.csv`)](https://github.com/FIAP/Pos_Tech_DTAT/blob/DeepLearning/Aula%204%20-%20Redes%20Neurais%20Recorrentes/bbc-text.csv)

## Estrutura do projeto

```
project_nlp_dtat_fase5/
├── bbc-text.csv                          # Dataset (baixar manualmente)
├── main.py                               # Script principal
├── requirements.txt                      # Dependências Python
├── README.md                             # Este arquivo
└── outputs/                              # Gerado após executar main.py
    ├── confusion_matrix_deep_learning.png
    ├── confusion_matrix_word2vec.png
    └── relatorio_comparativo.txt
```

## Pré-requisitos

- Python 3.10 ou superior
- Windows PowerShell (ou terminal equivalente)

## Como executar

### 1. Baixar o dataset

Coloque o arquivo `bbc-text.csv` na mesma pasta do `main.py`.

### 2. Instalar dependências

```powershell
py -m pip install -r requirements.txt
```

### 3. Executar o projeto

```powershell
py main.py
```

O script pode levar alguns minutos por causa do treinamento do modelo LSTM.

## O que o projeto faz

### Coleta e pré-processamento

- Carrega o dataset BBC News (`category`, `text`)
- Aplica tokenização, remoção de pontuação e stopwords em inglês

### Abordagem 1 — Deep Learning

- Treina Word2Vec (Gensim) no conjunto de treino
- Usa os vetores Word2Vec como **embeddings pré-treinados** na camada inicial
- Classifica com rede **LSTM**
- Monitora accuracy/loss por época com early stopping
- Avalia no conjunto de teste com precision, recall, F1-score e matriz de confusão

### Abordagem 2 — Word2Vec

- Utiliza o mesmo Word2Vec treinado no conjunto de treino
- Representa cada documento pela **média dos vetores das palavras**
- Classifica com **Regressão Logística**
- Avalia no **mesmo conjunto de teste** com as mesmas métricas

### Saída esperada

- Relatório comparativo no terminal
- Arquivo `outputs/relatorio_comparativo.txt`
- Matrizes de confusão em PNG para ambos os modelos
- Conclusão indicando qual abordagem teve melhor desempenho

## Resultados obtidos

Os resultados oficiais deste projeto estão em `outputs/relatorio_comparativo.txt`, gerado pela última execução de `py main.py`.

| Modelo | Acurácia | F1 macro | Tempo |
|---|---:|---:|---:|
| Deep Learning (Word2Vec + LSTM) | 60,9% | 0,534 | ~45 s |
| Word2Vec + Regressão Logística | **92,4%** | **0,922** | **~1 s** |

**Melhor abordagem neste experimento:** Word2Vec + Regressão Logística

> **Nota:** o modelo LSTM pode variar levemente entre execuções por causa da aleatoriedade no treinamento neural. O Word2Vec + Regressão Logística permanece estável (~92%). Para entrega, use os arquivos em `outputs/` da execução escolhida e evite rodar `main.py` novamente sem necessidade.
## Dependências principais

- `pandas`, `numpy`, `scikit-learn`
- `nltk` (tokenização e stopwords)
- `gensim` (Word2Vec)
- `tensorflow` (LSTM)
- `matplotlib`, `seaborn` (matrizes de confusão)
