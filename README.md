# Diabetes Prediction Service

A containerized ML inference service for predicting diabetes risk based on input features.
Built as a backend API with a focus on latency, scalability, and deployment trade-offs.

## Tech Stack
- Python (Flask)
- scikit-learn
- Docker
- AWS ECS (Fargate)

## Deployment Note
This service was deployed on **AWS ECS (Fargate)** during development and testing to validate containerization, request handling, and inference performance under load.

The live deployment was intentionally shut down after validation to avoid ongoing cloud costs. The repository retains the full **Docker configuration, service setup, and deployment workflow** used during ECS deployment.

## Project Structure
- `app.py` — Inference API
- `diabetes_predictor.pkl` — Trained ML model
- `Dockerfile` — Production container image
- `buildspec.yml` — CI/CD pipeline configuration
- `requirements.txt` — Python dependencies
