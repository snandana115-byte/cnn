Face Emotion Recognition with CNN😊😢 Happy vs Sad Image Classification
A simple and interactive Streamlit web application that uses a trained Convolutional Neural Network (CNN) model to recognize human facial emotions and classify them as Happy or Sad.
🚀 Features
📸 Upload a facial image (JPG, PNG, JPEG)
🤖 Predicts emotion using a trained CNN model
😊😢 Classifies images into Happy or Sad
📊 Displays prediction confidence
📉 Shows probability distribution for both classes
⚡ Fast, simple, and user-friendly interface
🧩 Classes
The model predicts one of the following emotions:
Happy 😊
Sad 😢
🛠️ Tech Stack
Python 🐍
TensorFlow / Keras
Streamlit
NumPy
PIL (Python Imaging Library)
📂 Project Structure
project/
│── app.py                # Streamlit app
│── emotion_model.h5      # Trained CNN model
│── train_model.py        # Model training script
│── dataset/              # Emotion image dataset
📌 Project Overview
This project focuses on facial emotion recognition, where a CNN model is trained on facial images to learn emotional patterns such as smiles or sadness. The trained model is integrated into a Streamlit web app that allows users to upload images and instantly receive emotion predictions along with confidence scores.
