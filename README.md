![Build Status](https://github.com/KaterinaKorre/ai-text-classification-ml/actions/workflows/ci-python.yml/badge.svg)

# 🎬 Sentiment Classification on IMDB Dataset

This project implements several **Machine Learning algorithms from scratch** and compares them against **Scikit-learn** and **Keras** implementations.  
It was developed for the *Artificial Intelligence* course at Athens University of Economics and Business.

---

## 📌 Task
Binary classification of movie reviews (**positive vs negative**) using the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

---

## 🛠️ Algorithms Implemented

### From scratch
- Naive Bayes (Multinomial):contentReference[oaicite:9]{index=9}
- Logistic Regression with SGD + regularization:contentReference[oaicite:10]{index=10}
- AdaBoost with decision stumps:contentReference[oaicite:11]{index=11}

### Using libraries
- Scikit-learn:
  - `MultinomialNB`
  - `LogisticRegression`
  - `AdaBoostClassifier`:contentReference[oaicite:12]{index=12}
- Keras:
  - `MLP` with word embeddings:contentReference[oaicite:13]{index=13}

---

## 📂 File Overview
src/
main.py # runs experiments, plots results
Naives.py # custom Naive Bayes
LogisticRegresion.py # logistic regression baseline
LogisticRegresionimpl.py# improved logistic regression
AdaBoost.py # custom AdaBoost
sklearn_adaboost.py # scikit-learn AdaBoost
mlp_word_embedings.py # MLP with embeddings (Keras)


---

## 📊 Results (from Report)
- Custom models reached ~0.61 max test accuracy
- MLP (Keras) achieved ~0.86 test accuracy:contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}

Metrics reported: Accuracy, Precision, Recall, F1 for increasing training sizes  
`[500, 2500, 4500, 6500, ..., 24500]`

---

## 🚀 How to Run
1. Install dependencies:
```bash
pip install numpy scikit-learn keras matplotlib tensorflow

Run experiments:
python src/main.py

Or run individual models:
python src/Naives.py
python src/LogisticRegresion.py
python src/AdaBoost.py
python src/mlp_word_embedings.py

📖 What I Learned

Implementing ML algorithms from scratch

Designing regularization & SGD optimizers

Building boosting methods (AdaBoost)

Comparing custom models with library implementations

Applying deep learning (MLPs with embeddings) for NLP

Evaluating models with accuracy, precision, recall, F1 curves

🔧 Continuous Integration

This repo includes a GitHub Actions workflow (ci-python.yml) that:

Sets up Python

Installs dependencies

Runs a quick check (python src/Naives.py on a small subset)
