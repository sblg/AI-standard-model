# ANN Prediction Model with Flask API

This project implements an Artificial Neural Network (ANN) prediction model using Keras and deploys it as an API using Flask. The model is trained on a simple dataset and then converted to the ONNX format for cross-platform compatibility. This README provides instructions on setting up and running the project.

## Introduction

The goal of this project is to demonstrate a simple implementation of an ANN prediction model using Keras and provide a Flask API for serving predictions. The model is trained on a synthetic dataset and saved as an ONNX file, which can be used in different environments.

## Dependencies

To run this project, you need the following dependencies:

- Python 3.6+
- scikit-learn
- Keras
- Flask
- onnxruntime

Install the dependencies using the following command:

```bash
pip install -r requirements.txt

## Usage
1. Clone the repository:
git clone https://github.com/your_username/your_repository.git
cd your_repository

2. Train the ANN model and convert to ONNX:
python train_model.py

3. Start the Flask API:
python app.py
