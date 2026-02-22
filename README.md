# DeepSignal — AI Safety Intelligence Engine

## Overview

DeepSignal is a transformer-based multi-label AI Safety Intelligence System designed to analyze social media text and detect:

- Psychological distress (Depression Risk)
- Toxic language
- Manipulative intent

The system integrates multi-label classification, class imbalance correction, threshold calibration, composite risk scoring, and rule-based escalation logic into a deployable Streamlit application.


## Problem Statement

Online platforms face increasing challenges related to:

- Mental health crisis detection
- Toxic interactions
- Emotional manipulation

DeepSignal performs multi-label safety detection and computes a composite psychological risk score to categorize content into actionable safety levels.


## System Architecture

User Text  
↓  
DistilBERT Encoder  
↓  
Multi-Label Classification (3 Outputs)  
↓  
Class-Weighted BCEWithLogitsLoss  
↓  
Threshold Calibration  
↓  
Composite Risk Scoring Engine  
↓  
Rule-Based Escalation Layer  
↓  
Streamlit Deployment  


## Model Details

- Base Model: DistilBERT
- Task: Multi-label classification (3 labels)
- Loss Function: BCEWithLogitsLoss with class weighting
- Optimizer: AdamW (LR = 2e-5)
- Batch Size: 16
- GPU: T4 (Google Colab)

### Labels
| Label | Description |
|--------|------------|
| Depression | Psychological distress indicators |
| Toxicity | Offensive / harmful language |
| Manipulation | Emotional coercion patterns |



##  Handling Class Imbalance

Class imbalance was addressed using positive class weighting:
pos_weight = total_samples / (3 * label_positive_counts)

This improved recall for rare safety-critical categories.

## Threshold Calibration

Instead of default 0.5 thresholds, per-label threshold optimization was performed:
| Label | Optimized Threshold |
|--------|-------------------|
| Depression | 0.45 |
| Toxicity | 0.20 |
| Manipulation | 0.10 |


## Model Performance

### ROC-AUC Scores

| Label | ROC-AUC |
|--------|----------|
| Depression | 0.93+ |
| Toxicity | 0.94+ |
| Manipulation | 0.99 

High ROC-AUC values indicate strong discriminative performance across all labels.

## Composite Risk Engine

Risk Score =  
0.5 × Depression +  0.3 × Toxicity +  0.2 × Manipulation  

### Risk Categories

- Safe
- Monitor
- Moderate Risk
- Critical Risk

## Escalation Policy Layer

To prevent severe signals from being diluted:

- Depression > 0.9 → Critical Risk  
- Toxicity > 0.8 → High Toxicity Risk  
- Manipulation > 0.9 → High Manipulation Risk  

This ensures high-risk cases are escalated immediately.

## Deployment

- Framework: Streamlit
- Live inference with probability outputs
- Real-time risk categorization
- Model loaded from saved PyTorch weights

## Project Structure
DeepSignal/
│
├── app.py
├── requirements.txt
├── deepsignal_model.ipynb
├── deepsignal_dataset.ipynb
└── README.md


---

## 🧩 Key Technical Highlights

- Multi-label transformer fine-tuning
- Imbalance-aware training
- Threshold calibration
- Composite risk scoring
- Rule-based escalation logic
- Streamlit deployment
- ROC-AUC evaluation

## Author

Bhawana Kapri — B.Tech CSE (Data Science)

