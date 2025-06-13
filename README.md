# Arabic Tweet Sentiment Classification

## Project Overview

This project focuses on classifying the sentiment of Arabic tweets into four categories: Objective, Negative, Neutral, and Positive. It involves preprocessing, feature extraction, classical and deep learning models (FFNN, LSTM), and transformer fine-tuning using MARBERTv2. Performance is evaluated across models and feature sets.

---

## Data Preprocessing

- **Translation**: English tweets are translated into Arabic (`Translated_Cleaned_Tweet`).
- **Label Mapping**: Sentiment labels are mapped to integers:
  - `OBJ` → 0  
  - `NEG` → 1  
  - `NEUTRAL` → 2  
  - `POS` → 3  
- **Data Source**: Processed from `clean_train_data.csv`.

---

##  Feature Engineering

- **Classical NLP Features**:
  - Binary Bag of Words (BBoW)
  - Frequency Bag of Words (FBoW)
  - TF-IDF

- **Arabic Word Embeddings**:
  - MARBERTv2 tokenizer `input_ids` (max_length=256)

---

## Model Architectures

- Optimizer: Adam  
- Loss: `categorical_crossentropy`  
- Metric: Accuracy  
- Techniques: EarlyStopping, `class_weight` for class imbalance

### Feed Forward Neural Network (FFNN)

- Dense layers with ReLU, BatchNormalization, and Dropout

### Long Short-Term Memory (LSTM)

- Masking layer → LSTM layers → Dense output

---

## Competition Model: Fine-Tuning MARBERTv2

- Model: `UBC-NLP/MARBERTv2`
- Framework: Hugging Face Transformers


