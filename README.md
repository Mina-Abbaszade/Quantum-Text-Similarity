# Quantum-Text-Similarity

## Overview

This repository implements a quantum-enhanced semantic similarity model for text using Quantum Natural Language Processing (QNLP).

Text is converted into quantum circuits using the DisCoCirc framework, and similarity between texts is computed using:

- Quantum state fidelity  
- Cosine similarity  

---

## Method

Each text is mapped to a quantum circuit using DisCoCirc and a parameterised ansatz (Sim4).

Similarity between two texts is computed using:

- Fidelity between quantum states  
- Cosine similarity of circuit outputs  

---

## Experiments

We evaluate the model using:

- Train / validation / test split (for hyperparameter tuning)  
- 5-fold cross-validation (main results)  

---

## Usage

Run the fidelity-based model:

```
PYTHONPATH=. python src/training/train_kfold.py
```

Run the cosine similarity model:

```
PYTHONPATH=. python src/training/train_kfold_cosine.py
```
---

## Notes

- Dataset is not included  
- This repository focuses on similarity learning only  
- The recommender system is not included  

---

## Author

Mina Abbaszadeh  
UCL — Quantum Machine Learning / QNLP
