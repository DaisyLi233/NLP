# 📘 YouTube Toxic Comment Detection

A project that detects whether a YouTube comment is **toxic** or **non-toxic** using both **Traditional Machine Learning** and **Deep Learning** approaches.

---

## Introduction

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

## 🅐 Approach A — Traditional Machine Learning (TF-IDF Pipeline)

Traditional machine learning methods rely on **statistical text representations** such as TF-IDF, combined with classical supervised learning algorithms.  
Unlike deep learning models that learn dense semantic embeddings, traditional ML works with **sparse, high-dimensional vectors**, offering interpretability and efficiency.

---

### Workflow for Traditional ML

#### 1. Exploratory Data Analysis (EDA) & Preprocessing
We begin by understanding dataset characteristics and preparing the text for feature extraction. Steps include:

- Inspecting class distribution  
- Examining comment length statistics  
- Lowercasing  
- Removing URLs  
- Removing usernames/mentions  
- Removing non-alphabetic characters  
- Normalizing whitespace
- Removing unrelated columns  

This ensures consistent and clean input for modeling.

---

#### 2. Feature Extraction — TF-IDF
We convert each comment into numerical vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

TF-IDF:

- represents text as a sparse high-dimensional vector  
- down-weights common or uninformative words  
- highlights discriminative terms  
- serves as an efficient and widely used baseline in NLP  

These TF-IDF vectors are used as input for all traditional ML models in this project.

---

#### 3. Models Tried

##### ✔ Logistic Regression
A linear classifier that learns an optimal decision boundary between toxic and non-toxic comments.  
Highly effective for high-dimensional sparse features and widely used as a baseline in NLP.

##### ✔ Random Forest
An ensemble method composed of multiple decision trees.  
Captures non-linear feature interactions but is less optimized for extremely sparse TF-IDF matrices.

##### ✔ Support Vector Machine (SVM)
A margin-based classifier that identifies the best separating hyperplane between classes.  
Widely used in text classification tasks, especially with linear kernels.

---


## 🅑 Approach B — Deep Learning

In this approach, we explore **neural network models** that operate on sequences of embedded word vectors.  
Deep learning architectures can capture:

- semantic similarity  
- contextual meaning  
- local and long-range dependencies  

We implemented three neural network models: **LSTM**, **CNN**, and **GloVe Embeddings + CNN**.

---

### 1️⃣ LSTM (Long Short-Term Memory)

LSTM networks are a type of recurrent neural network (RNN) designed to capture long-range dependencies in sequential data.  
They are effective for text tasks where the order and context of words matter.

In this project, we used the following pipeline:

- **Embedding Layer**  
  Learned from scratch during training.

- **Bidirectional LSTM**  
  Processes the text sequence in both forward and backward directions to capture richer context.

- **Dropout**  
  Added to reduce overfitting and improve generalization.

- **Sigmoid Output Layer**  
  Produces a probability for binary classification (toxic vs non-toxic).

---

### 2️⃣ CNN (1D Convolutional Neural Network)

CNNs can extract meaningful local patterns (such as n-grams) from text sequences.  
They are computationally efficient and often perform well on shorter comments or when long-range context is less critical.

The CNN pipeline used:

- **Embedding Layer**  
  Learned jointly with the model.

- **Conv1D Layer**  
  Applies convolutional filters to detect local patterns in the text.

- **Global Max Pooling**  
  Selects the most salient features from the convolution step.

- **Sigmoid Output Layer**  
  Generates a binary classification output.

---

### 3️⃣ GloVe Embeddings + CNN

For the third model, we incorporate **pretrained word embeddings** instead of training embeddings from scratch.

**What is GloVe?**  
GloVe (Global Vectors for Word Representation) is a pretrained embedding model trained on **6 billion tokens** of text.  
Because it is trained on such a massive and diverse corpus, it captures broad semantic relationships between words and provides rich, meaningful vector representations.

**Why we used it:**  
We expected that using pretrained embeddings would provide stronger semantic understanding and potentially improve model performance, especially given our dataset size.

**Model pipeline:**

- **Load GloVe pretrained vectors (glove.6B.100d.txt)**  
  Initialize the embedding matrix with pretrained word vectors.

- **Freeze or fine-tune embeddings**  
  (Depending on configuration) to preserve GloVe semantics.

- **CNN architecture**  
  Same Conv1D + Global Max Pooling structure as the previous CNN model.

- **Sigmoid Output Layer**  
  Produces the final binary prediction.

---

## How to Download & Use GloVe

We used:
glove.6B.100d.txt

The GloVe embedding files are large (≈800MB zipped → about 1GB unzipped), so they are **not included in this repository**.  
To reproduce the GloVe experiment, follow the steps below.

### 1. Download the embeddings

Run in your terminal:

"wget http://nlp.stanford.edu/data/glove.6B.zip"


### 2. Unzip the file

unzip glove.6B.zip


You will get:

- glove.6B.50d.txt  
- glove.6B.100d.txt  
- glove.6B.200d.txt  
- glove.6B.300d.txt  

This project uses:

**`glove.6B.100d.txt`**

### 3. Move the file into the project

Put the file into the deep_learning folder.

## Results & Discussion

Across all models, the **test accuracy generally falls between 60% and the low 70% range**.  
In contrast, most models, especially the deep learning architectures, and it achieved **very high training accuracy**, in some cases approaching **nearly 100%**.  
This indicates **significant overfitting**, where the models learn the training data well but fail to generalize to new comments.

Even the model using **GloVe pretrained embeddings**, which are trained on a massive **6-billion-token** corpus and provide rich semantic word representations, achieved only around **~70% test accuracy**.  
While GloVe provides strong understanding at the word level, the model must still learn **how words combine into sentences** to form toxic meaning.  
This requires patterns specific to the toxicity dataset, not just general semantic knowledge.

We believe the fundamental limitation lies in the **small dataset size**:

- only around **1,000 rows** in total  
- only **~30%** labeled as toxic  

This amount of data is insufficient for models — especially deep learning models — to reliably capture toxicity patterns or extract meaningful linguistic features.  
NLP tasks typically require **large datasets**, because models must learn patterns of a language rather than simple numerical relationships.

Overall, we think the reason is that the dataset size is the primary bottleneck, rather than the choice of model or embedding method.

However, this project represents our **first hands-on experience with NLP**, and it gave us a clear understanding of how modern text classification pipelines work.  
Despite the dataset limitations, achieving these results was exciting and rewarding for us, and it motivates us to explore more advanced NLP techniques in the future.



