# Classificação de Diagnósticos Médicos com KNN

Este projeto utiliza Machine Learning para prever o diagnóstico de novos pacientes com base em um histórico de dados médicos catalogados. O algoritmo utilizado é o **K-Nearest Neighbors (KNN)**, que classifica cada novo paciente comparando suas características com as dos pacientes mais parecidos da base de treinamento.

Este é um projeto de estudo, criado para praticar os primeiros passos em Machine Learning.

## Funcionalidades

* Carregamento e exploração de dados médicos usando `pandas`.
* Separação de dados em conjuntos de treino (70%) e teste (30%).
* Padronização das variáveis para a mesma escala usando `StandardScaler`.
* Treinamento de um modelo classificador KNN com `k = 5`.
* Avaliação do modelo pela acurácia nos dados de teste.
* Previsão de diagnósticos para uma base de pacientes inéditos.

## Estrutura do projeto

```
.
├── main.py                       # Código principal
├── pacientes_catalogados.csv     # Base de treino: pacientes com diagnóstico (170 registros)
├── pacientes_novos.csv           # Pacientes sem diagnóstico, para previsão (20 registros)
└── README.md
```

## Dados

### `pacientes_catalogados.csv`

Contém 170 pacientes já diagnosticados, divididos igualmente entre as duas classes (85 de cada).

| Coluna              | Descrição                          | Tipo     |
|---------------------|------------------------------------|----------|
| `idade`             | Idade do paciente, em anos         | inteiro  |
| `imc`               | Índice de Massa Corporal           | decimal  |
| `glicose`           | Glicose em jejum, em mg/dL         | inteiro  |
| `pressao_sistolica` | Pressão arterial sistólica, em mmHg| inteiro  |
| `diagnostico`       | Classe a ser prevista: `Saudavel` ou `Risco` | texto |

### `pacientes_novos.csv`

Contém 20 pacientes com as mesmas colunas de características, **sem** a coluna `diagnostico`. É nessa base que o modelo faz as previsões.

## Como funciona

1. Os dados catalogados são lidos e a coluna `diagnostico` é separada como alvo (`y`). As demais colunas formam as características (`x`).
2. A base é dividida em treino e teste (`train_test_split`, 70/30).
3. O `StandardScaler` é ajustado **apenas nos dados de treino** e depois aplicado ao teste e aos novos pacientes.
4. O modelo KNN é treinado e avaliado nos dados de teste, e a acurácia é exibida.
5. Os novos pacientes são padronizados com o mesmo padronizador e o modelo prevê o diagnóstico de cada um. O resultado aparece na coluna `diagnostico_previsto`.

A padronização é necessária porque o KNN se baseia em distâncias: sem ela, variáveis com valores maiores, como a glicose, dominariam o cálculo em relação a variáveis menores, como o IMC.

## Limitações

* A base é pequena e sintética, então a acurácia alta não reflete o desempenho em um cenário real.
* A divisão treino/teste é aleatória, então a acurácia pode variar entre execuções.
* A acurácia sozinha pode ser enganosa em contextos médicos, onde errar um caso de risco (falso negativo) costuma ser mais grave. Outras métricas seriam mais adequadas.

## Próximos passos

* Avaliar com matriz de confusão, precisão, recall e F1-score.
* Comparar com outros algoritmos, como Regressão Logística e Random Forest.
