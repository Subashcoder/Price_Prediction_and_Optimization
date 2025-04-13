# 🏡 Airbnb Smart Solution – Data Intelligence Platform

A containerized, end-to-end data platform that delivers actionable pricing and expansion insights to Airbnb hosts using machine learning and big data tools.

---

## 🔧 Tech Stack

- **Programming**: Python, PySpark
- **Data Engineering**: Apache Airflow, Spark, AWS S3, AWS Redshift
- **ML Models**: Gradient Boosting, Collaborative Filtering
- **Deployment**: Docker, AWS ECR
- **Visualization**: Power BI

---

## 🚀 Project Overview

This project builds a complete pipeline for analyzing Airbnb data, generating recommendations for:
- Optimal **listing prices**
- Ideal **expansion locations**
- **Personalized suggestions** for hosts based on booking behavior

The system is fully automated and production-ready, using Airflow for orchestration and Docker for deployment.

---

## 🛠️ Pipeline Architecture

```mermaid
graph TD
    A[Data Extraction - Python] --> B[Data Processing - PySpark]
    B --> C[Data Storage - AWS S3]
    C --> D[AWS Redshift - Analytics Layer]
    D --> E[ML Models - GBM & CF]
    E --> F[Docker Containerization]
    F --> G[Deployment on AWS ECR]
    G --> H[Dashboard - Power BI]
