# Classificador de Sentimentos com Python

## 📌 Sobre o projeto

Este projeto utiliza técnicas de Processamento de Linguagem Natural (NLP — *Natural Language Processing*) e Machine Learning para classificar avaliações textuais em três categorias:

- **Positivo:** avaliações que expressam satisfação.
- **Negativo:** avaliações que expressam insatisfação.
- **Neutro:** avaliações que apresentam informações sem sentimento claramente positivo ou negativo.

O objetivo é desenvolver conhecimentos práticos em análise de texto, classificação automática e avaliação de modelos de Machine Learning.

## 🎯 Objetivos

- Carregar e analisar dados textuais utilizando Pandas.
- Transformar textos em representações numéricas com TF-IDF.
- Treinar um modelo de classificação com Regressão Logística.
- Separar dados de treinamento e teste.
- Avaliar o desempenho do modelo.
- Classificar novas frases automaticamente.

## 🛠️ Tecnologias utilizadas

- Python 3
- Pandas
- Scikit-learn
- TF-IDF
- Regressão Logística
- Git e GitHub

## 📂 Estrutura do projeto

```text
classificador_sentimentos/
├── avaliacoes.csv
├── classificador.py
└── README.md
```

## 📊 Base de dados

O projeto utiliza um arquivo CSV com avaliações textuais previamente classificadas.

Cada registro contém:

| Coluna | Descrição |
|---|---|
| `texto` | Avaliação escrita em português |
| `sentimento` | Categoria atribuída à avaliação |

A base inicial contém 15 exemplos distribuídos entre as classes positiva, negativa e neutra.

## ⚙️ Como funciona

O processamento é dividido nas seguintes etapas:

1. **Carregamento dos dados:** leitura do arquivo CSV com Pandas.
2. **Preparação:** separação dos textos e dos rótulos.
3. **Divisão dos dados:** criação dos conjuntos de treinamento e teste.
4. **Vetorização:** conversão dos textos em características numéricas usando TF-IDF.
5. **Treinamento:** ajuste do modelo de Regressão Logística.
6. **Predição:** classificação de frases não utilizadas diretamente no treinamento.
7. **Avaliação:** cálculo da acurácia e análise de métricas de classificação.

## ▶️ Como executar

### 1. Instalar as dependências

```bash
python -m pip install pandas scikit-learn
```

### 2. Executar o programa

Na pasta do projeto, execute:

```bash
python classificador.py
```

O programa exibirá a base de avaliações, as métricas de avaliação e os sentimentos previstos para novas frases.

## 📈 Avaliação do modelo

O projeto utiliza as seguintes métricas:

- **Acurácia:** proporção de previsões corretas.
- **Precisão:** proporção das previsões de uma classe que estão corretas.
- **Recall:** proporção dos exemplos de uma classe identificados corretamente.
- **F1-score:** média harmônica entre precisão e recall.

Os resultados devem ser interpretados com cautela, pois a base inicial é pequena e não representa a diversidade da linguagem natural.

## 🚀 Possíveis melhorias

- Aumentar a quantidade de avaliações.
- Utilizar bases de dados reais em português.
- Comparar diferentes algoritmos de classificação.
- Analisar uma matriz de confusão.
- Criar uma interface para digitar frases.
- Desenvolver gráficos para visualizar os resultados.
- Investigar técnicas para lidar com negações, ironia e variações linguísticas.

## 📚 Aprendizados

Este projeto permite praticar conceitos fundamentais de Ciência de Dados, incluindo preparação de dados, vetorização de texto, aprendizado supervisionado, classificação multiclasse e avaliação de modelos.

## 👨‍💻 Autor

Projeto desenvolvido como parte dos estudos práticos em Python, Machine Learning, NLP e Ciência de Dados.

**Status:** projeto educacional em desenvolvimento.
