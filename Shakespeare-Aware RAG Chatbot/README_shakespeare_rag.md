# 🎭 Shakespeare-Aware RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=flat)
![Course](https://img.shields.io/badge/UOW-CSCI933%20ML-103a75?style=flat)

> A domain-adapted Retrieval-Augmented Generation (RAG) system for question-answering over Shakespeare's plays — benchmarked against a TF-IDF baseline with rigorous multi-criteria evaluation and production-grade safeguards.

---

## 📌 Overview

This project builds a complete **RAG pipeline** capable of answering questions about three Shakespeare plays — *Hamlet*, *Macbeth*, and *Romeo and Juliet* — grounded in the actual text rather than relying on a language model's parametric memory alone.

Two retrieval systems are implemented and rigorously compared head-to-head:
- **Dense semantic retrieval** using sentence-transformer embeddings (cosine similarity)
- **Sparse keyword retrieval** using TF-IDF (bigram, 10K vocabulary baseline)

The system also includes a **stylised response mode** that generates answers in Early Modern English (*thou/thee/dost*), staying true to the Shakespearean register.

---

## 📂 Project Structure

```
shakespeare-rag/
├── data/
│   ├── hamlet.txt
│   ├── macbeth.txt
│   └── romeo_and_juliet.txt
├── embeddings/
│   └── embeddings_cache.npy       # Cached MiniLM embeddings
├── notebooks/
│   └── shakespeare_rag.ipynb
├── outputs/
│   ├── evaluation_results.csv
│   └── retrieval_comparison_chart.png
├── requirements.txt
└── README.md
```

---

## 🏗 Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│         Retrieval Layer              │
│  ┌──────────────┐  ┌─────────────┐  │
│  │ Dense (MiniLM│  │ TF-IDF      │  │
│  │ cosine sim)  │  │ (bigrams)   │  │
│  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────┘
     │  Top-k scene chunks
     ▼
┌─────────────────────────────────────┐
│       Generation Layer               │
│   google/flan-t5-large (770M)        │
│   Beam search · Length penalty       │
│   No-repeat n-gram constraints       │
└─────────────────────────────────────┘
     │
     ▼
  Answer (Standard or Early Modern English mode)
```

---

## 🔑 Key Design Decisions

### Chunking Strategy
Text is split at the **scene level** rather than fixed-size windows — preserving dramatic context and character continuity within each retrieved chunk.

### Embedding & Caching
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are computed once and cached as `.npy` files for reproducible, fast retrieval across runs

### Generation Config
| Parameter | Value | Rationale |
|---|---|---|
| Backbone | `google/flan-t5-large` | Strong instruction-following at 770M params |
| Beam search | 4 beams | Balance between diversity and coherence |
| Length penalty | > 1.0 | Encourages complete answers |
| No-repeat n-gram | 3 | Prevents repetitive output |

### Production Safeguards
- **Out-of-domain guard** — detects and handles questions unrelated to the three plays
- **Unknown-play guard** — gracefully responds when a play outside the corpus is mentioned
- **Token-budget-aware truncation** — ensures prompt + context fits within model limits without silent truncation

---

## 📊 Evaluation

### Methodology
- **11 questions** evaluated (6 instructor-set + 5 group-designed)
- **4 criteria** scored 1–5 by human evaluators:

| Criterion | Description |
|---|---|
| Correctness | Factual accuracy of the answer |
| Grounding | Is the answer supported by retrieved text? |
| Retrieval Relevance | Did retrieval surface the right chunks? |
| Usefulness | Is the answer practically helpful? |

### Results

| Metric | RAG (Dense) | TF-IDF Baseline |
|---|---|---|
| Avg. Correctness | **4.0** | 1.9 |
| Retrieval Relevance | **4.4** | 2.3 |

**RAG outperformed the TF-IDF baseline across all four criteria**, with the largest gains in correctness (+110%) and retrieval relevance (+91%). Results are logged in `outputs/evaluation_results.csv`.

---

## 🛠 Setup & Usage

```bash
# Clone the repo
git clone https://github.com/sohan-kumar/shakespeare-rag.git
cd shakespeare-rag

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/shakespeare_rag.ipynb
```

**Requirements:**
```
torch
transformers
sentence-transformers
scikit-learn
numpy
pandas
matplotlib
jupyter
```

---

## 💬 Example

```python
question = "Why does Hamlet hesitate to kill Claudius?"

# Standard mode
answer = rag_answer(question, style="standard")
# → "Hamlet delays because he finds Claudius praying and believes killing him
#    in prayer would send him to heaven rather than damnation."

# Stylised mode
answer = rag_answer(question, style="early_modern")
# → "He doth hesitate, for Claudius kneeleth in prayer, and to slay him thus
#    wouldst grant him passage to heaven, not the damnation Hamlet seeketh."
```

---

## 🎓 Academic Context

Developed as part of **CSCI933 Machine Learning Algorithms and Applications**
University of Wollongong, 2026

---

## 👤 Author

**Sohan Kumar**
[LinkedIn](https://linkedin.com/in/sohan-kumar-006599220) · [GitHub](https://github.com/sohan-kumar)
