# Explainable Sex-Stratified Machine Learning for Tuberculosis (TB) Risk Profiling

[![R Version](https://img.shields.io/badge/R-v4.3%2B-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Parallel Computing](https://img.shields.io/badge/Parallel-doParallel-orange.svg)](https://cran.r-project.org/package=doParallel)

This repository contains the master R implementation script (`analyze_tb_risk_machine_learning_master.R`) for a sex-stratified, explainable machine learning framework designed to predict population-level Tuberculosis (TB) risk using structured patient demographic and clinical data.

The computational pipeline implements end-to-end data ingestion, multivariable binomial logistic regression modeling ($\text{aOR}$ estimation), cost-sensitive class weighting, stratified 5-fold cross-validation across multiple algorithms, TreeSHAP feature attributions, model calibration diagnostics, Decision Curve Analysis (DCA), and demographic fairness auditing across age brackets.

---

## 📌 Repository Overview & Key Features

- **Sex-Stratified Analytical Framework**: Partitions patient records into female ($\mathcal{D}_{\text{Female}}$) and male ($\mathcal{D}_{\text{Male}}$) strata to capture biological and epidemiological variations in disease manifestation.
- **Epidemiological Risk Modeling**: Fits multivariable logistic regressions to derive Adjusted Odds Ratios ($\text{aOR}$) and 95% Wald confidence intervals for clinical predictors.
- **Cost-Sensitive Class Weighting**: Embeds inverse class frequency weights ($\omega_1 = N_0 / N_1$) directly into optimization loss functions to mitigate class imbalance without discarding data or introducing synthetic noise.
- **Multi-Model Machine Learning Paradigms**: Comparative-benchmarks Weighted Logistic Regression, Fast Parallel Random Forest (`ranger`), and Gradient Boosted Decision Trees (`XGBoost`).
- **Parallel Processing Engine**: Automatically registers multi-core processing (`doParallel` and `foreach`) to accelerate tree building and cross-validation iterations.
- **Explainable AI (XAI)**: Quantifies global feature importance via XGBoost Fractional Gain and local directional attributions using TreeSHAP beeswarm plots and Partial Dependence Plots (PDP).
- **Clinical Utility & Equity Diagnostics**: Evaluates probability calibration via Brier reliability scores, estimates net clinical benefit across decision thresholds via Decision Curve Analysis (DCA), and audits predictive equity across age brackets ($<25$, $25\text{--}44$, $45\text{--}64$, $65+$).

---

## 📁 Dataset Source & Expected Structure

The pipeline is configured to ingest structured tabular records (e.g., `tuberculosis_xray_dataset.csv`), available via the Kaggle Synthetic Tuberculosis Dataset:
- **Data Link**: [Kaggle Tuberculosis X-Ray Dataset (Synthetic)](https://www.kaggle.com/datasets/miadul/tuberculosis-x-ray-dataset-synthetic)

### Input Variables Handled:
- **Target Outcome**: `TB_Status` / `Class` (`Tuberculosis` vs. `Normal`)
- **Demographics**: `Sex` / `Gender`, `Age`
- **Clinical Biomarkers & Symptoms**: `Cough_Severity`, `Breathlessness`, `Fatigue`, `Weight_Loss`, `Fever`, `Night_Sweats`, `Sputum_Production`, `Blood_in_Sputum`, `Smoking_History`, `Previous_TB_History`

---

## ⚙️ Dependencies & Prerequisites

The execution script automatically checks for, installs, and loads all necessary R packages. Ensure you have **R (v4.3 or higher)** installed.

### Required R Packages:
```R
tidyverse, caret, ranger, xgboost, pROC, shapviz, 
gridExtra, e1071, PRROC, viridis, themis, recipes, doParallel
