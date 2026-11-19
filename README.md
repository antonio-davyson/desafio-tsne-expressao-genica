# t-SNE em dados de expressão gênica de células individuais (sintéticos)

Desafio Rocketseat: reduzir 50 dimensões de expressão gênica com t-SNE, plotar em 2D e 3D coloridos por `cell_type`
e avaliar visualmente se os tipos celulares formam agrupamentos claros.

## Conteúdo
- `tsne_expressao_genica.py` — carregamento, padronização, t-SNE 2D/3D, gráficos e análise (no final do arquivo)
- `synthetic_cell_data.csv` — dataset (1.200 células, 50 genes, 5 tipos celulares)
- `tsne_2d.png` e `tsne_3d.png` — gráficos gerados

## Como executar
```bash
pip install -r requirements.txt
python tsne_expressao_genica.py
```

## Resultados
- Os 5 tipos celulares formam agrupamentos claros no t-SNE 2D (silhueta ≈ 0,54 contra ≈ 0,08 nas 50 dimensões originais; KNN com validação cruzada ≈ 97%).
- No 3D os grupos seguem separados, mas mais próximos entre si (silhueta ≈ 0,34).
- Resultado estável para perplexidades de 15 a 100. No t-SNE, distâncias entre clusters não têm interpretação direta.
