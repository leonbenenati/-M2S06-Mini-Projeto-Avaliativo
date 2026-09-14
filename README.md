# Classificação de Dígitos Manuscritos - MNIST

> Projeto de Machine Learning e Deep Learning desenvolvido para classificar dígitos manuscritos utilizando o conjunto de dados MNIST. O projeto contempla análise exploratória de imagens, pré-processamento, treinamento e avaliação de diferentes modelos, testes com classes ocultadas e classificação de uma imagem externa ao conjunto de dados.

-----

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- TensorFlow
- Keras
- Pillow (PIL)

-----

## Modelos Utilizados

Foram utilizados três modelos de classificação com características distintas:

- **K-Nearest Neighbors (KNN)**
- **LightGBM**
- **Multilayer Perceptron (MLP)** utilizando TensorFlow/Keras

A utilização desses três modelos permite comparar diferentes abordagens de classificação: um algoritmo baseado em distância, um modelo de Gradient Boosting baseado em árvores e uma rede neural artificial.

-----

## Divisão das Partes do Projeto

O projeto foi dividido em cinco etapas principais:

### Parte 1 – Carregamento e Análise Exploratória de Imagens (EDA)

Nesta etapa foi carregado o conjunto de dados **MNIST**, composto por 70.000 imagens de dígitos manuscritos pertencentes às classes de 0 a 9.

Cada imagem possui dimensão de **28 × 28 pixels**, totalizando **784 características**. Os valores dos pixels originalmente variam entre 0 e 255, representando diferentes níveis de intensidade.

Foram realizadas:

- análise das dimensões das variáveis preditoras (X) e da variável alvo (y);
- análise da distribuição das classes;
- verificação do balanceamento entre os dígitos;
- visualização de exemplos das dez classes do conjunto de dados.

-----

### Parte 2 – Pipeline de Pré-processamento e Divisão dos Dados

Nesta etapa foi realizada a preparação dos dados para o treinamento dos modelos.

O conjunto foi dividido em:

- **80% para treinamento e validação**;
- **20% para teste**.

A divisão foi realizada de forma estratificada, preservando a distribuição das dez classes.

Os valores dos pixels também foram normalizados do intervalo original de **0–255 para 0–1**.

A normalização é especialmente importante para o **KNN**, que utiliza cálculos de distância, e para o **MLP**, auxiliando no processo de otimização dos pesos da rede neural.

Para o **LightGBM**, a normalização não é necessária, pois modelos baseados em árvores não são sensíveis à escala das características. Os dados normalizados foram mantidos para padronizar o conjunto utilizado pelos três modelos.

-----

### Parte 3 – Implementação, Treinamento e Avaliação de 3 Modelos

Nesta etapa foram implementados, treinados e avaliados três algoritmos: **KNN, LightGBM e MLP**.

#### K-Nearest Neighbors (KNN)

Para o KNN foram avaliados os hiperparâmetros:

- **n_neighbors:** quantidade de vizinhos considerados na classificação;
- **weights:** forma de ponderação dos votos dos vizinhos (`uniform` ou `distance`).

A otimização foi realizada utilizando **RandomizedSearchCV** e validação cruzada.

#### LightGBM

Para o LightGBM foram avaliados:

- **n_estimators:** número de árvores utilizadas durante o processo de boosting;
- **learning_rate:** contribuição de cada nova árvore para a atualização do modelo.

Também foi utilizado **RandomizedSearchCV** com validação cruzada para selecionar a melhor configuração.

#### Multilayer Perceptron (MLP)

Foi desenvolvida uma rede neural utilizando **TensorFlow/Keras**, composta por:

- camada de entrada com 784 características;
- primeira camada oculta com quantidade variável de neurônios e função ReLU;
- segunda camada oculta com 64 neurônios e função ReLU;
- camada de saída com 10 neurônios e função Softmax.

Durante a otimização foram avaliados dois hiperparâmetros:

- **número de neurônios da primeira camada oculta**;
- **learning rate do otimizador Adam**.

Foram avaliadas cinco configurações utilizando validação cruzada com cinco folds, totalizando **25 treinamentos**.

#### Avaliação dos Modelos

Os três modelos foram avaliados utilizando um conjunto de teste independente.

As métricas utilizadas foram:

- Accuracy;
- Precision ponderada;
- Recall ponderado;
- F1-Score ponderado.

Também foram construídas matrizes de confusão para analisar os principais erros de classificação entre os dígitos.

Os resultados obtidos foram:

| Modelo | Tempo de Tunagem | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| KNN | 14,47 min | 0,9732 | 0,9734 | 0,9732 | 0,9732 |
| LightGBM | 92,49 min | **0,9811** | **0,9812** | **0,9811** | **0,9811** |
| MLP | **8,68 min** | 0,9773 | 0,9775 | 0,9773 | 0,9773 |

O **LightGBM** apresentou o melhor desempenho preditivo, alcançando aproximadamente **98,11% de acurácia**, porém apresentou também o maior custo computacional durante a otimização.

O **MLP** apresentou desempenho próximo ao LightGBM, com aproximadamente **97,73% de acurácia**, e o menor tempo de tunagem neste experimento.

O **KNN** apresentou aproximadamente **97,32% de acurácia**.

Os tempos apresentados correspondem ao ambiente computacional e à estratégia de busca utilizados neste projeto, não representando uma comparação universal de velocidade entre os algoritmos.

-----

### Parte 4 – Testando a Capacidade do Modelo MLP de Lidar com Cenários Fora do Padrão Ideal de Laboratório

Nesta etapa foi analisado o comportamento do MLP diante de classes que foram completamente removidas durante o treinamento.

Os dígitos **4 e 7** foram selecionados como classes ocultas e removidos do conjunto utilizado para treinamento e validação.

Dessa forma, o novo MLP foi treinado apenas com as classes:

`0, 1, 2, 3, 5, 6, 8 e 9`

O processo de otimização de hiperparâmetros foi realizado novamente, garantindo que os dígitos 4 e 7 não fossem utilizados durante o ajuste dos pesos ou durante a validação.

Após o treinamento, o modelo foi submetido a um conjunto de teste contendo exclusivamente imagens dos dígitos **4 e 7**, caracterizando um cenário de inferência **Out-of-Distribution (OOD)**.

O modelo apresentou maior tendência de atribuir as imagens desconhecidas às classes **2, 8 e 0**, sendo a classe 2 a mais frequentemente prevista.

Na análise das probabilidades das previsões foram obtidos:

- confiança média: **15,47%**;
- mediana da confiança: **14,76%**;
- confiança mínima: **10,74%**;
- confiança máxima: **33,39%**;
- previsões com confiança igual ou superior a 90%: **0%**;
- previsões com confiança igual ou superior a 95%: **0%**.

Apesar de o classificador ser obrigado a selecionar uma das classes disponíveis, não foi observado um comportamento acentuado de **overconfidence** neste experimento.

As probabilidades permaneceram relativamente distribuídas entre diferentes classes, indicando incerteza do modelo diante das imagens pertencentes às classes ocultadas.

O experimento demonstra que a maior probabilidade produzida por um classificador não deve ser interpretada automaticamente como garantia de que a entrada pertence a uma classe conhecida pelo modelo.

-----

### Parte 5 – Desenhando e Prevendo

Na última etapa foi realizado um teste utilizando uma imagem criada externamente ao conjunto MNIST.

Foi desenhado o dígito **5**, e a imagem passou por um processo de preparação para aproximá-la do formato utilizado pelo MNIST:

- conversão para escala de cinza;
- redimensionamento do dígito;
- centralização em uma imagem de **28 × 28 pixels**;
- transformação da imagem em 784 características;
- normalização dos pixels para o intervalo de 0 a 1.

A mesma imagem foi então submetida aos três modelos treinados anteriormente.

Os resultados foram:

| Modelo | Previsão | Confiança |
|---|---:|---:|
| KNN | 1 | 100,0% |
| LightGBM | 7 | 60,4% |
| MLP | **5** | 30,2% |

O **MLP foi o único modelo que classificou corretamente o dígito 5**, apesar de apresentar a menor confiança entre os três modelos.

O teste também evidencia que um valor elevado de confiança não representa necessariamente uma previsão correta. O KNN, por exemplo, classificou incorretamente a imagem como o dígito 1, mesmo apresentando 100% na proporção de votos utilizada para a classificação.

A imagem externa também apresentou diferenças em relação ao padrão de escrita encontrado nas imagens do MNIST, demonstrando a dificuldade adicional dos modelos ao receber dados produzidos fora do conjunto utilizado durante o treinamento.

-----

# Informações Adicionais

O projeto utiliza o conjunto de dados **MNIST**, contendo 70.000 imagens de dígitos manuscritos de 0 a 9.

As imagens possuem resolução de **28 × 28 pixels** e são representadas como vetores contendo **784 características**.

O projeto busca não apenas comparar o desempenho de diferentes algoritmos de classificação, mas também analisar seu comportamento diante de classes não observadas durante o treinamento e de uma imagem produzida externamente ao conjunto de dados.
