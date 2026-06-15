# Federated Learning-Based Privacy-Preserving Disease Prediction System

This project is for privacy-preserving disease prediction using federated learning.

It simulates multiple hospitals/clients. Each client trains locally on its own patient data and sends only model parameters to the server. The server combines the models using Federated Averaging. Raw patient records are never shared.

## Features

- Synthetic healthcare dataset for diabetes/heart-risk style binary prediction
- Multiple simulated hospitals/clients
- Non-IID data split to make the experiment realistic
- Federated Averaging
- Optional differential privacy noise on local model updates
- Centralized training baseline
- Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
- Optional Streamlit prediction interface

## Project Structure

```text
federated-disease-prediction/
  app.py
  main.py
  requirements.txt
  src/
    data.py
    federated.py
    metrics.py
    model.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Federated Learning Experiment

```bash
python main.py
```

Useful options:

```bash
python main.py --clients 5 --rounds 30 --local-epochs 4 --dp-noise 0.02
python main.py --clients 8 --rounds 40 --non-iid-strength 0.9
```

## Run Web Interface

```bash
streamlit run app.py
```

## Suggested Thesis Title

Privacy-Preserving Disease Prediction Using Federated Learning and Differential Privacy

## Methodology Summary

1. Generate or load healthcare records.
2. Split data across multiple simulated hospitals.
3. Train local disease prediction models at each hospital.
4. Add optional differential privacy noise to local model weights.
5. Aggregate local weights at the server using FedAvg.
6. Evaluate the global model on a held-out test set.
7. Compare against centralized learning.

