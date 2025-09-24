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

## Phase 2: Model Fine-Tuning & Evaluation

**Status:** ✅ Completed

### Steps Done

- **Model & Tokenizer:** Loaded `distilbert-base-uncased` and tokenizer with 6 output labels.
- **Tokenization:** Tokenized all splits with truncation and dynamic padding.
- **Data Formatting:** Removed original text column, set format for PyTorch.
- **Training Arguments:** Set batch size 8, 2 epochs, learning rate 2e-5, weight decay 0.01, saved best model by macro F1.
- **Metrics:** Used accuracy, macro F1, and per-class F1 scores for evaluation.
- **Trainer & Callbacks:** Used Hugging Face `Trainer` with `EarlyStoppingCallback`.
- **Training:** Fine-tuned the model on the training set.
- **Evaluation:** Evaluated on validation and test sets.
- **Detailed Metrics:** Printed classification report and confusion matrix for test set.
- **Model Saving:** Saved the trained model and tokenizer to `sentiment_model/`.

### Results

- **Validation Accuracy:** ~0.94
- **Validation Macro F1:** ~0.92
- **Per-Class F1 (Validation):**
  - sadness: 0.97
  - joy: 0.96
  - love: 0.89
  - anger: 0.94
  - fear: 0.89
  - surprise: 0.87

- **Test Set Performance:**  
  Similar to validation, with strong results across all classes, including minorities.

### Notes

- The model handles class imbalance well, with only a slight drop for minority classes.
- Early stopping and best model selection by macro F1 ensured robust generalization.
- The model and tokenizer are ready for deployment.

---

## Phase 3: Model Deployment & Inference

**Status:** ✅ Completed

### Inference via Notebook

- See [`inference.ipynb`](inference.ipynb) for example usage.
- Example:
    ```python
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
    sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    texts = ["I am so happy today!", "She smiled at me, and my heart skipped a beat."]
    predictions = sentiment_pipeline(texts)
    for text, preds in zip(texts, predictions):
        print(f"Text: {text}")
        for i, pred in enumerate(preds):
            print(f"  {label_names[i]}: {pred['score']:.3f}")
        print()
    ```

### Streamlit App

- Run the interactive app with:
    ```sh
    streamlit run streamlit_app.py
    ```
- Enter text and get emotion predictions in your browser.

### Requirements

- Python 3.8+
- Install dependencies:
    ```sh
    pip install transformers datasets streamlit
    ```

---

## Outputs

- Trained model and tokenizer: [`sentiment_model/`](sentiment_model)
- Example input files: [`sample_input.txt`](sample_input.txt), [`sample_input1.txt`](sample_input1.txt)
- **Sample Dashboard Output:**

    ![Dashboard Screenshot 1](output/Screenshot%202025-09-24%20133036.png)
    ![Dashboard Screenshot 2](output/Screenshot%202025-09-24%20133239.png)
    ![Dashboard Screenshot 3](output/Screenshot%202025-09-24%20133331.png)

---

## Output Images Explained

- **Screenshot 1:**  
  Shows the main dashboard interface. Users can input text or upload `.txt`/`.pdf` files. The sidebar provides navigation, model info, and appearance settings.

- **Screenshot 2:**  
  Displays per-sentence emotion predictions and confidence scores. Each sentence is analyzed and visualized with metrics, progress bars, and summary charts (bar and pie charts) showing emotion distribution and confidence.

- **Screenshot 3 (Session History):**  
  Demonstrates the "Session History" page. This page lists previous analysis runs, showing the last five sessions. For each run, you can expand details to see the sentences analyzed, their predicted emotions, and confidence scores. This helps users track and review past analyses for comparison or record-keeping.

---

*These images illustrate the interactive features and visualization capabilities of your Streamlit Emotion Detection Dashboard.*

*This README is updated after each phase. Project is now fully complete and ready for use.*