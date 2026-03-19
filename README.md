# 🏡 Airbnb Smart Solution – Data Intelligence Platform

![WhatsApp Image 2026-03-19 at 1 58 13 PM](https://github.com/user-attachments/assets/38baf503-dba3-4148-b5e1-876225fb6a30)


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

This project builds a complete pipeline for analyzing Airbnb data, Optimizing the price and generating recommendations for AirBnb hosts.
- Analysis Page
  ![WhatsApp Image 2026-03-19 at 1 58 13 PM](https://github.com/user-attachments/assets/f3dfaedd-3e3d-4aa1-9b66-8e770e71e6e0)

- **Price Optimization**
  ![WhatsApp Image 2026-03-19 at 1 58 50 PM](https://github.com/user-attachments/assets/6a751613-8a0b-4a26-b963-0224504c1f69)

- **Personalized Recomendation Engine**
  ![WhatsApp Image 2026-03-19 at 1 58 50 PM](https://github.com/user-attachments/assets/9ed14a50-2c8e-4d39-919a-3d0deb6b5a00)


The system is fully automated and production-ready, using Airflow for orchestration and Docker for deployment.

## 🛠️ Pipeline Architecture

- **Deployment Stratergy**
- Data Extraction:Collect Airbnb datasets for key insights.
- Spark Data Processing: Process and transform data efficiently.
- Model Training: Train machine learning models for predictions.
- Frontend Integration: Present insights through a user interface.
- Orchestration with Airflow: Automate and monitor workflow tasks.
- Docker Containerization: Ensure scalable and consistent deployment.

graph TD
    A[Data Extraction - Python] --> B[Data Processing - PySpark]
    B --> C[Data Storage - AWS S3]
    C --> D[AWS Redshift - Analytics Layer]
    D --> E[ML Models - GBM & CF]
    E --> F[Docker Containerization]
    F --> G[Deployment on AWS ECR]
    G --> H[Dashboard - Power BI]
