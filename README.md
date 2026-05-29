# Rede Neural — Wine Quality Classification

Trabalho da disciplina de **Machine Learning** utilizando TensorFlow/Keras para classificar a qualidade de vinhos tintos.

## Dupla

| Aluno | RA |
|---|---|
| Gabriel Dias do Prado | 1002260326 |
| José Marcos dos Santos Medeiros | 1002261005 |

## Dataset

**Wine Quality Red** — [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

- 1.599 amostras de vinho tinto português
- 11 atributos físico-químicos numéricos (acidez, pH, álcool, etc.)
- Variável alvo: qualidade ≥ 6 → boa (1) / qualidade < 6 → baixa (0)

## Arquitetura da Rede Neural

```
Entrada (11 atributos)
     │
     ▼
Dense(64, ReLU)   ← Camada Oculta 1
     │
     ▼
Dense(32, ReLU)   ← Camada Oculta 2
     │
     ▼
Dense(1, Sigmoid) ← Saída
```

- **Otimizador:** Adam
- **Função de perda:** Binary Crossentropy
- **Épocas:** 50 | **Batch size:** 32

## Resultados

| Métrica | Valor |
|---|---|
| Acurácia (teste) | ~75–80% |
| Loss (teste) | ~0.48 |

## Como executar

```bash
# Clonar o repositório
git clone https://github.com/gabriel-days/rede-neural-wine-quality.git
cd rede-neural-wine-quality

# Criar ambiente virtual e instalar dependências
python3 -m venv venv
source venv/bin/activate
pip install tensorflow-macos tensorflow-metal pandas numpy matplotlib seaborn scikit-learn visualkeras Pillow jupyter ipykernel

# Abrir o notebook
jupyter notebook trabalho_redes_neurais.ipynb
```

> **Nota para Apple Silicon (M1/M2/M3):** usar `tensorflow-macos` + `tensorflow-metal` ao invés de `tensorflow`.

## Estrutura do Notebook

| Seção | Conteúdo |
|---|---|
| 1. Banco de Dados | Descrição e análise exploratória do Wine Quality Red |
| 2. Preparação | Normalização e divisão treino/teste |
| 3. Rede Neural | Modelo Keras com 2 camadas ocultas |
| 4. Gráficos | Diagrama da arquitetura + curvas de treinamento |
| 5. Erro do Modelo | Loss, acurácia, matriz de confusão, relatório de classificação |
| 6. Conclusão | Análise dos resultados e considerações finais |
