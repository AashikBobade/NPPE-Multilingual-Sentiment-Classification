# NPPE-1 Multilingual Sentiment Analysis

This repository contains my Kaggle notebook solution for the **NPPE-1 Multilingual Sentiment Classification** challenge.
The objective is to fine-tune **`google/gemma-3-1b-it`** for robust classification of text across multiple low-resource Indian languages.

---

## Competition Objective

The task is to predict the sentiment class for multilingual text samples across 13 Indian languages.
Each input sentence must be classified into one of two labels:

- **Positive**
- **Negative**

---

## Evaluation

Submissions are evaluated using **Macro F1-score** on the private leaderboard.

Macro F1 is especially suitable for this task because it gives equal importance to both classes and provides a balanced view of performance even when label distribution varies across languages.

---

## My Results

- **Public leaderboard score:** `0.95454`
- **Private leaderboard score:** `0.95238`

These scores were obtained from a Kaggle Notebook workflow built around prompt-based fine-tuning with parameter-efficient adaptation.

---

## Dataset Overview

The dataset contains:

- **Train set:** 900 labeled samples
- **Test set:** 100 unlabeled samples
- **Languages:** Assamese, Bodo, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, and Urdu

### Train file columns

- `ID` — unique row identifier
- `sentence` — input text
- `label` — target class (`Positive` or `Negative`)
- `language` — language code

### Test file columns

- `ID` — unique row identifier
- `sentence` — input text
- `language` — language code

---

## Solution Summary

My approach follows a **prompt-based multilingual fine-tuning pipeline**:

1. Load the competition dataset from Kaggle.
2. Normalize language codes into readable language names.
3. Convert each labeled row into an instruction-style prompt.
4. Fine-tune **Gemma-3-1B-IT** using **QLoRA / LoRA**.
5. Generate predictions on the test set with the same prompt format.
6. Parse the model output into the required labels.
7. Export the final `submission.csv` in Kaggle format.


---

## Approach Details

### 1) Prompt Engineering

Each training example is reformatted into a compact instruction prompt that asks the model to act as a multilingual sentiment classifier.
The prompt includes:

- the language name,
- the input sentence,
- a strict output constraint to return only the label.

This framing helps the generative model behave like a classifier while still benefiting from instruction tuning.

### 2) Multi-script Preprocessing

The notebook maps short language codes such as `bn`, `ta`, and `pa` to full language names before training.
This makes the prompt more human-readable and gives the model clearer language context.

### 3) Parameter-Efficient Fine-Tuning

To keep training feasible inside Kaggle GPU limits, the notebook uses:

- **4-bit quantization** for memory efficiency,
- **LoRA adapters** on attention projection layers,
- **SFT-style training** for supervised instruction tuning.

This makes the pipeline lightweight while preserving strong performance.

### 4) Inference and Post-Processing

At inference time, the test rows are converted to the same prompt structure used during training.
The generated output is then cleaned and mapped back to one of the required labels:

- `Positive`
- `Negative`

---

## Dataset Notes

The training set in the notebook is nearly balanced:

- **Positive:** 456
- **Negative:** 444

This reduces the risk of bias toward one class and makes the F1 score more meaningful.

---

## Training Configuration


The notebook uses a compact fine-tuning setup.
The main ingredients are:

- **Base model:** `google/gemma-3-1b-it`
- **Training method:** QLoRA / LoRA
- **Precision:** 4-bit loading
- **Trainer:** supervised fine-tuning workflow
- **Objective:** multilingual sentiment classification

---

## Strengths of the Solution

- Works well with a **small labeled dataset**.
- Handles **multiple Indian languages** in a single pipeline.
- Is efficient enough to run within **Kaggle Notebook constraints**.
- Produces a clean end-to-end submission workflow.

---

## Limitations

- The solution depends on prompt quality and output parsing.
- Because the dataset is small, the model can still be sensitive to seed and decoding choices.
- Generation-based classification can be less stable than a dedicated classifier head.

---

## Reproducibility

To reproduce the notebook flow:

1. Install the required libraries.
2. Run the notebook cells in order:
   - environment setup
   - data loading
   - preprocessing
   - prompt creation
   - model loading
   - LoRA configuration
   - training
   - inference
   - submission generation

The final output is:

```text
submission.csv
```

---

## Data Access

The dataset is already included in this repository for convenience.

Alternatively, you can download it directly from Kaggle using `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.competition_download('nppe-dlp-2026-term-1')

print("Path to competition files:", path)
```
---

## Requirements

The notebook was developed for a Kaggle GPU environment with the following core libraries:

- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `bitsandbytes`
- `trl`
- `torch`

Example installation:

```bash
pip install transformers datasets peft accelerate bitsandbytes trl
```

---

## File Structure

```text
.
├── main.ipynb
├── data/
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
└── README.md
```

---

## Acknowledgment

Built for the **NPPE-1 Multilingual Sentiment Classification** Kaggle competition.
