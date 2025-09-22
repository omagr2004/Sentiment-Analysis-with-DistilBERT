# Sentiment Analysis with DistilBERT

## Project Overview

This project aims to fine-tune a lightweight transformer (DistilBERT) to classify text into emotion categories (sadness, joy, love, anger, fear, surprise) using the [emotion dataset](https://huggingface.co/datasets/emotion).

---

## Phase 1: Data Preparation & Exploration

**Status:** ✅ Completed

### Steps Done

- **Dataset Loaded:** Used Hugging Face `datasets` library to load the emotion dataset.
- **Dataset Structure:**  
  - Train: 16,000 samples  
  - Validation: 2,000 samples  
  - Test: 2,000 samples  
  - Classes: sadness, joy, love, anger, fear, surprise

- **Class Distribution:**
  - sadness: 4666
  - joy: 5362
  - love: 1304
  - anger: 2159
  - fear: 1937
  - surprise: 572

- **Text Length Statistics:**
  - Average: 19.17 words
  - Median: 17 words
  - Std: 10.99 words
  - Max: 66 words
  - Min: 2 words

- **Unique Labels:** All 6 classes present in the training set.

### Notes

- The dataset is imbalanced (e.g., "surprise" and "love" are minority classes).
- No missing labels or obvious data issues found.
- Ready to proceed to tokenization and model fine-tuning.

---

## Next Steps

- **Phase 2:** Model fine-tuning and evaluation (with per-class metrics and imbalance handling).

---

*This README will be updated as each phase is completed.*