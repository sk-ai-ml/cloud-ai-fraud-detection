# Cloud AI Framework for Credit Card Fraud Detection

COS7054-B Cloud AI, Coursework 1, University of Bradford.

This repository contains the code and evidence for a cloud AI framework that detects credit card fraud and is deployed on **Microsoft Azure Machine Learning**. The work follows the three stages of the AI lifecycle: data preparation, model training and tuning, and deployment and monitoring.

## Results at a glance

| Item | Result |
| --- | --- |
| Dataset | ULB credit card fraud (284,807 transactions, 0.17% fraud) |
| Selected model | XGBoost with square-root class weighting |
| Test PR-AUC | 0.819 |
| Test recall / precision | 77.9% / 85.1% (threshold 0.386) |
| False alerts | 13 of 56,651 legitimate test transactions (0.02%) |
| Cloud deployment | Azure ML managed online endpoint, key authentication, Standard_F4s_v2 |
| Server-side scoring time | 6 ms per transaction |

## Repository structure

| Path | Contents |
| --- | --- |
| `01_eda_and_preprocessing.ipynb` | Data ingestion, cleaning, class distribution, amount and time analysis, correlation, clustering, preprocessing pipeline |
| `02_modelling.ipynb` | 5-fold cross-validation of eight models, threshold tuning, test evaluation, SHAP explainability, model export |
| `03_azure_deployment.ipynb` | Data asset and model registration, scoring script, environment, managed online endpoint, live tests, logs, clean-up |
| `deploy/src/score.py` | Scoring script run by the Azure endpoint |
| `deploy/conda.yaml` | Pinned inference environment |
| `models/` | Trained model (XGBoost JSON), scaler, metadata, sample request and endpoint test results |
| `figures/` | All figures and result tables used in the report |

## Reproducing the work

1. Download `creditcard.csv` from [Kaggle (mlg-ulb/creditcardfraud)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`. The dataset is not included in this repository under its licence.
2. Install the dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order (01, 02, 03). Notebook 03 also needs an Azure subscription, an Azure Machine Learning workspace and its `config.json` in the project folder.

All random steps use a fixed seed (42).

## Note on the dataset

The dataset was published by the Machine Learning Group of Université Libre de Bruxelles (ULB). Its features V1 to V28 are anonymised principal components, so no personal data is processed.
