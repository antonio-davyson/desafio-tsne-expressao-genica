# %% [markdown]
# # Desafio: t-SNE em dados de expressão gênica de células individuais (sintéticos)
# %%
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

SEED = 42

# %% 1. Carregar e explorar
df = pd.read_csv("synthetic_cell_data.csv")
genes = [c for c in df.columns if c.startswith("gene_")]
X, y = df[genes], df["cell_type"]
print(df.shape, df.isna().sum().sum(), "valores ausentes")
print(y.value_counts())
print(X.describe().T[["mean", "std", "min", "max"]].describe().round(2))   # escalas dos genes

# %% 2. Pré-processamento: padronização (t-SNE usa distâncias, então escalas diferentes distorcem o mapa)
X_s = StandardScaler().fit_transform(X)
tipos = sorted(y.unique()); cores = dict(zip(tipos, plt.cm.tab10(range(len(tipos)))))

# %% 3. t-SNE para 2 e 3 componentes (50 dimensões -> 2D e 3D)
tsne2 = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(X_s)
tsne3 = TSNE(n_components=3, perplexity=30, init="pca", random_state=SEED).fit_transform(X_s)

# %% 4. Gráfico 2D (componente 1 em X, componente 2 em Y, cor = cell_type)
plt.figure(figsize=(8, 6.5))
for t in tipos:
    m = (y == t).values
    plt.scatter(tsne2[m, 0], tsne2[m, 1], s=14, alpha=.8, color=cores[t], label=t)
plt.xlabel("t-SNE componente 1"); plt.ylabel("t-SNE componente 2"); plt.title("t-SNE 2D — expressão gênica por tipo celular")
plt.legend(title="cell_type"); plt.tight_layout(); plt.savefig("tsne_2d.png", dpi=130); plt.close()

# %% 5. Gráfico 3D (componentes 1, 2 e 3; cor = cell_type), em dois ângulos
fig = plt.figure(figsize=(14, 6.5))
for i, (elev, azim) in enumerate([(20, 45), (25, 135)], start=1):
    ax = fig.add_subplot(1, 2, i, projection="3d")
    for t in tipos:
        m = (y == t).values
        ax.scatter(tsne3[m, 0], tsne3[m, 1], tsne3[m, 2], s=10, alpha=.8, color=cores[t], label=t)
    ax.set_xlabel("componente 1"); ax.set_ylabel("componente 2"); ax.set_zlabel("componente 3"); ax.view_init(elev, azim)
ax.legend(title="cell_type", loc="upper left"); fig.suptitle("t-SNE 3D — expressão gênica por tipo celular")
plt.tight_layout(); plt.savefig("tsne_3d.png", dpi=130); plt.close()

# %% 6. Análise: os tipos celulares formam agrupamentos claros e distintos?
# Visual: ver os gráficos. Quantitativo (apoio à análise visual): silhueta nos espaços 2D/3D e acerto de um KNN (vizinhos mais próximos) com validação cruzada.
def resumo(nome, E):
    sil = silhouette_score(E, y)
    knn = cross_val_score(KNeighborsClassifier(15), E, y, cv=5).mean()
    print(f"{nome}: silhueta={sil:.3f}  KNN(cv=5)={knn:.3f}")
resumo("original 50D (padronizado)", X_s); resumo("t-SNE 2D", tsne2); resumo("t-SNE 3D", tsne3)

# Sensibilidade à perplexidade (parâmetro principal do t-SNE)
for p in [5, 15, 30, 50, 100]:
    resumo(f"t-SNE 2D perplexity={p}", TSNE(n_components=2, perplexity=p, init="pca", random_state=SEED).fit_transform(X_s))

# Separação por par de tipos (distância entre centroides / dispersão média) no 2D
cent = pd.DataFrame(tsne2).groupby(y.values).mean()
print(cent.round(1))

# %% [markdown]
# ## 7. Análise visual dos resultados
# - **Dados:** 1.200 células, 50 genes, 5 tipos celulares equilibrados (237 a 242 células cada), sem valores ausentes. Os genes têm escalas bem diferentes
#   (desvio-padrão de ~1 a ~15), por isso foram padronizados antes do t-SNE.
# - **Gráfico 2D:** os 5 tipos celulares formam agrupamentos claros e distintos. type_1 (canto superior esquerdo), type_2 (direita) e type_3 (parte inferior) ficam bem isolados;
#   type_0 e type_4 são vizinhos e é onde está a maior proximidade entre grupos, com poucas células "fora do lugar" (pontos de um tipo dentro do grupo de outro).
#   Silhueta no mapa 2D ≈ 0,54, contra ≈ 0,08 nas 50 dimensões originais, e o KNN com validação cruzada acerta ≈ 97% do tipo celular, então a separação visual reflete estrutura real nos dados.
# - **Gráfico 3D:** os grupos continuam separados (silhueta ≈ 0,34), mas aparecem mais próximos e sobrepostos, principalmente nos ângulos em que type_4 fica no centro entre os demais.
#   O 3D não trouxe ganho de separação em relação ao 2D (KNN ≈ 96%), e é mais difícil de ler em imagem estática.
# - **Perplexidade:** com valores de 15 a 100 o resultado é estável (silhueta 0,51–0,54); só com perplexidade 5 os grupos ficam mais fragmentados (silhueta 0,46).
# - **Cuidados:** no t-SNE, a distância entre clusters e o tamanho deles não têm interpretação direta (só a vizinhança local é confiável), os eixos não têm significado próprio
#   e o resultado depende da semente e da perplexidade. Por isso, a afirmação é sobre "formam grupos distintos", não sobre quão distantes os tipos são entre si.
#   Os dados são sintéticos, então a separação clara é esperada e não garante o mesmo comportamento em dados reais de células.
