# 📘 YouTube Toxic Comment Detection

A project that detects whether a YouTube comment is **toxic** or **non-toxic** using both **Traditional Machine Learning** and **Deep Learning** approaches.

---

## 🧠 Introduction

Online platforms such as YouTube contain large amounts of user-generated content, where toxic language, harassment, and hate speech frequently appear.  
The goal of this project is to build a **binary classification model** that automatically classifies comments into:

- **0 — Non-toxic**  
- **1 — Toxic**

We explore two major approaches:  
1. **Traditional Machine Learning (TF-IDF based models)**  
2. **Deep Learning (Neural Network models)**

This project provides a comparison between classic NLP pipelines and modern neural architectures on the same dataset.

---

## 📂 Dataset

The dataset is sourced from Kaggle:  
🔗 https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data

It contains YouTube comments along with toxicity labels.  
Columns used:
- `Text`
- `IsToxic` (0 or 1)

---

# 🅐 Approach A — Traditional Machine Learning

This pipeline uses classical NLP preprocessing and statistical feature extraction.

### 💡 Why Traditional ML?

Traditional ML models:

- work well on **small datasets**  
- train extremely fast  
- provide good interpretability  
- require simple and efficient text representations (TF-IDF)

Although they cannot capture semantic meaning as deeply as neural networks, they often perform surprisingly well when the dataset is limited.

---

## 🛠 Workflow for Traditional ML

### **1. Text Cleaning**

- Lowercasing  
- Removing URLs  
- Removing usernames/mentions  
- Removing non-letter characters  
- Normalizing whitespace  

### **2. Feature Extraction — TF-IDF**

TF-IDF transforms each comment into a sparse numerical vector, weighting words based on how important they are to the document relative to the entire dataset.

### **3. Models Tried**

#### ✔ Logistic Regression  
- Strong baseline for NLP  
- Performs well with high-dimensional TF-IDF  
- Efficient and interpretable  
- Best performer among the three traditional models in this project  

#### ✔ Random Forest  
- Works well for structured/tabular data  
- Tends to **overfit** on TF-IDF text features  
- Performed worse on this dataset  

#### ✔ Support Vector Machine (SVM)  
- Effective classifier for many text problems  
- Slower to train with high-dimensional sparse vectors  
- Requires careful hyperparameter tuning  
- Moderate performance

---

# 🅑 Approach B — Deep Learning

We also explored neural models that process sequences of embedded word vectors.  
Deep learning can model:

- semantic similarity  
- contextual relationships  
- local and long-range dependencies  

We implemented **three neural architectures**:

---

## 1️⃣ LSTM (Long Short-Term Memory)

- Embedding layer → Bidirectional LSTM  
- Captures long-range dependencies  
- Dropout to reduce overfitting  
- Sigmoid output for binary classification  

---

## 2️⃣ CNN (1D Convolutional Neural Network)

- Embedding layer  
- Conv1D + Global Max Pooling  
- Captures important n-gram patterns  
- Trains faster than RNNs  
- Works well for shorter comments  

---

## 3️⃣ GloVe Embeddings + CNN

Instead of learning embeddings from scratch, we use **pretrained GloVe embeddings** from Stanford NLP.

### Why GloVe?

- Trained on a massive corpus  
- Provides meaningful, semantically rich word vectors  
- Helps generalize better on small datasets  

We used:
glove.6B.100d.txt

## 📥 How to Download & Use GloVe

GloVe files are large (≈800MB zipped → 1GB unzipped), so they are **not included in the repository**.

### 1️⃣ Download the embeddings

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip

## 📥 How to Download & Use GloVe

GloVe files are large (≈800MB zipped → 1GB unzipped), so they are **not included in the repository**.

unzip glove.6B.zip

Place the following file manually into the project:

deep_learning/glove.6B.100d.txt
