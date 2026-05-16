# Phroura

Phroura is a master's-level computer science project developed in fulfillment of the requirements for DLMCSPCSP01 - Project: Computer Science. The project focuses on phishing URL detection using lexical feature engineering, classical machine learning, and a lightweight full-stack interface.

## Project Goal

The goal of this project is to design, implement, and evaluate a phishing detection platform that can classify suspicious URLs in real time using lightweight, explainable methods.

## Current Status

The project now includes:

- lexical URL feature extraction
- heuristic phishing baseline
- reproducible train/test splitting
- Logistic Regression, Random Forest, and SVM model evaluation
- multi-source Random Forest deployment model artifact
- backend inference pipeline
- Streamlit demo app with prediction output and explanation support

The Streamlit app uses the multi-source Random Forest selected during Phase 3 external validation.

## Repository Structure

- `app/` - Streamlit interface and user-facing application code
- `data/` - dataset files and dataset notes
- `models/` - trained model artifacts and related metadata
- `notebooks/` - exploratory analysis and experiments
- `src/` - core source code for features, training, evaluation, and inference
- `tests/` - tests for feature extraction, inference, and other core logic

## Setup

Install project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Run The Streamlit App

```bash
python3 -m streamlit run app/streamlit_app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

The app lets a user submit a URL and returns:

- benign or phishing classification
- Random Forest phishing probability
- heuristic rule-based notes
- top model feature signals
- full lexical feature table
- explanation pages for the pipeline, features, model, and limitations

## Model Artifact

The hosted demo uses:

```text
models/multisource_random_forest_model.joblib
```

This file is intentionally committed so the Streamlit app can run immediately without retraining the model during deployment. The app uses a phishing probability threshold of `0.40`, selected from the Phase 3 threshold evaluation.

To regenerate the original single-source model artifact from the local dataset:

```bash
python3 -m src.inference --save-model
```

To regenerate the multi-source deployment artifact:

```bash
python3 -m src.evaluation --run-multisource-retraining
```

The local training dataset is expected at:

```text
data/Dataset.csv
```

The dataset file is ignored by Git because it is a local data dependency.

## Command-Line Prediction

After the model artifact exists, a URL can be classified from the terminal:

```bash
python3 -m src.inference --url https://www.rmit.edu.au/
```

## Core Workflows

1. Select and document phishing and benign URL datasets.
2. Extract lexical and structural features from URLs.
3. Compare heuristic and machine learning phishing detectors.
4. Integrate the selected model into a usable application.
5. Evaluate the system and document the results in the final report.

## Scope And Limitations

Phroura uses lexical URL features only. It does not inspect webpage content, redirect behavior, WHOIS records, domain reputation, blacklist membership, or live threat intelligence.

This means the system should be understood as a research prototype and decision-support tool, not a guarantee that a URL is safe or unsafe.
