# 💳 Real-Time Credit Card Fraud Detection Pipeline

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end MLOps solution for identifying fraudulent credit card transactions in highly imbalanced datasets using **Random Forest**, **SMOTE**, **FastAPI**, and **Docker**.

---

## 📌 Architecture Overview

```text
[ Raw Transaction Data ] ➔ [ SMOTE Resampling ] ➔ [ Random Forest Classifier ]
                                                                 │
                                                        [ joblib Artifact ]
                                                                 │
[ Client Request ] ➔ [ FastAPI REST API ] ➔ [ Real-Time Fraud Score ]
