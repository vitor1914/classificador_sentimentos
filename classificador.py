
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Carregar os dados
dados = pd.read_csv("avaliacoes.csv")

print("BASE DE AVALIAÇÕES")
print(dados)
print("\nQuantidade de avaliações:", len(dados))

# 2. Separar textos e sentimentos
X = dados["texto"]
y = dados["sentimento"]

# 3. Separar treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y
)

# 4. Criar o modelo
modelo = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classificador", LogisticRegression(max_iter=1000))
])

# 5. Treinar o modelo
modelo.fit(X_treino, y_treino)

# 6. Fazer previsões
previsoes = modelo.predict(X_teste)

# 7. Avaliar o desempenho
print("\nACURÁCIA:")
print(accuracy_score(y_teste, previsoes))

print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(
    y_teste,
    previsoes,
    zero_division=0
))

# 8. Testar frases novas
frases = [
    "Gostei muito do produto",
    "O atendimento foi horrível",
    "O pedido chegou hoje"
]

print("\nCLASSIFICAÇÃO DE NOVAS FRASES")

for frase in frases:
    sentimento = modelo.predict([frase])[0]
    print(f"Frase: {frase}")
    print(f"Sentimento previsto: {sentimento}\n")
