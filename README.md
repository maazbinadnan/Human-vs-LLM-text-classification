# Human vs LLM Text Classification

A machine learning project for binary classification of text as either **human-written** or **machine-generated (LLM)**. Built as part of the SCC 453 course project, based on the [SemEval 2024 Task 8](https://semeval.github.io/SemEval2024/tasks) — Subtask A (Monolingual).

---

## Task

Given a piece of text, classify it as:
- **0** — Human-written
- **1** — Machine-generated (LLM)

---

## Dataset

**SemEval 2024 Task 8 — Subtask A (Monolingual)**

| Split | Rows |
|-------|------|
| Train | monolingual train set |
| Dev   | 5,000 (2,500 human / 2,500 machine) |

Dataset files (not included in this repo — download from the official SemEval task page):
- `subtaskA_train_monolingual.jsonl`
- `subtaskA_dev_monolingual.jsonl`

---

## Models & Results

Three model variants were explored:

### Model 1 — RoBERTa-Base (Fine-tuned)

| Metric | Human (0) | Machine (1) | Macro Avg |
|--------|-----------|-------------|-----------|
| Precision | 0.707 | 0.847 | 0.777 |
| Recall    | 0.886 | 0.634 | 0.760 |
| F1-Score  | 0.787 | 0.725 | 0.756 |
| **Accuracy** | — | — | **75.96%** |

---

### Model 2 — DistilBERT-base-uncased (Fine-tuned)

| Metric | Human (0) | Machine (1) | Macro Avg |
|--------|-----------|-------------|-----------|
| Precision | 0.713 | 0.890 | 0.801 |
| Recall    | 0.922 | 0.629 | 0.775 |
| F1-Score  | 0.804 | 0.737 | 0.770 |
| **Accuracy** | — | — | **77.54%** |

---

### Model 3 — DistilBERT + Custom Classifier + Post-Processing

A custom architecture on top of DistilBERT with a multi-layer classification head, dropout regularization, and a textual post-processing step using readability features (Flesch-Kincaid grade level, stopword ratio, etc.).

| Metric | Value |
|--------|-------|
| Macro F1 | 0.7092 |
| Test Loss | 1.2498 |
| Accuracy (baseline) | 72.30% |

Post-processing with varying confidence threshold (θ) did not improve over the baseline.

---

## Architecture

**Model 3 Custom Classifier:**
- Backbone: `distilbert-base-uncased`
- Two dropout layers
- Dense classification head
- Post-processing using `textstat` Flesch-Kincaid grade level, NLTK stopword analysis, and confidence thresholding

---

## Project Structure

```
├── 453_Project_Maaz_Adnan.ipynb   # Main notebook (EDA + all 3 models)
├── 453_Project_Maaz_Adnan - Colab.pdf  # PDF export of the notebook
└── xlm-roberta-base/              # XLM-RoBERTa model artifacts
    └── subtaskA/
```

---

## Setup & Requirements

The project was developed on **Google Colab** with GPU acceleration.

**Key dependencies:**
```
torch
transformers
datasets
evaluate
scikit-learn
pandas
numpy
matplotlib
seaborn
textstat
nltk
wordcloud
```

Install via:
```bash
pip install transformers datasets evaluate textstat nltk wordcloud language-tool-python
```

---

## Usage

Open `453_Project_Maaz_Adnan.ipynb` in Google Colab and:

1. Mount your Google Drive and place the SemEval dataset under `SemEval_Dataset/` in your Drive project folder.
2. Run the **EDA** section for dataset exploration and visualizations.
3. Run **Model 1 (RoBERTa-Base)** or **Model 2 (DistilBERT)** sections for fine-tuning with the Hugging Face `Trainer` API.
4. Run **Model 3** for the custom classifier with post-processing.

Switch between `train_and_test`, `train`, and `test` modes via the `RUN_PIPELINE` dict in each model section.

