import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pasta_atual = os.path.dirname(os.path.abspath(__file__))
caminho_catalogados = os.path.join(pasta_atual, "pacientes_catalogados.csv")
caminho_novos = os.path.join(pasta_atual, "pacientes_novos.csv")
tabela = pd.read_csv(caminho_catalogados)
print("\nDados dos pacientes já catalogados")
print(tabela,"\n")
tabela.info()
y = tabela["diagnostico"]
x = tabela.drop(columns=["diagnostico"])

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

padronizador = StandardScaler()
x_treino_pad = padronizador.fit_transform(x_treino)
x_teste_pad = padronizador.transform(x_teste)

modelo_knn = KNeighborsClassifier(n_neighbors=5)
modelo_knn.fit(x_treino_pad, y_treino)

previsao_knn = modelo_knn.predict(x_teste_pad)
precisao = accuracy_score(y_teste, previsao_knn)
print("###"*50)
print(f"\nPrecisão do modelo KNN nos dados de teste: {precisao * 100:.2f}%\n")
print("###"*50)

novos = pd.read_csv(caminho_novos)
print("\nDados dos pacientes ainda não catalogados")
print(novos,"\n")

# Mesmas colunas e mesma ordem do treino, com a mesma padronização
novos_pad = padronizador.transform(novos[x.columns])
previsao = modelo_knn.predict(novos_pad)

resultado = novos.copy()
resultado["diagnostico_previsto"] = previsao
print("\nPrevisão do diagnostico dos novos pacientes")
print(resultado)