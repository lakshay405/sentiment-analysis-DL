# sentiment-analysis-DL
Toxic Comment Classification using LSTM Neural Network
This project focuses on classifying toxic comments using LSTM (Long Short-Term Memory) neural networks, aimed at identifying various types of toxicity such as threats, obscenity, insults, and more. The dataset used is sourced from the Jigsaw Toxic Comment Classification Challenge on Kaggle, which contains labeled comments with multiple toxicity categories.

Project Overview
The project encompasses the following major components:

Data Loading and Preparation
Model Development with LSTM
Training, Evaluation, and Metrics
Model Persistence and Inference
Interactive Scoring Interface using Gradio
Data Loading and Preparation
The dataset (train.csv) is loaded into a Pandas DataFrame, where each comment is associated with multiple toxicity labels. The text data is vectorized using the TensorFlow TextVectorization layer to convert comments into numerical sequences suitable for LSTM input.

Model Development with LSTM
The LSTM-based neural network is constructed using TensorFlow/Keras:

Layers: Embedding layer for word embeddings, Bidirectional LSTM layer for capturing sequence dependencies, followed by fully connected Dense layers for feature extraction.
Activation Functions: ReLU for Dense layers and sigmoid for the output layer to predict toxicity probabilities.
Loss Function: Binary cross-entropy is utilized as the loss function to train the model for multi-label classification.
Training, Evaluation, and Metrics
The model is trained on a dataset optimized for performance, with batch processing and prefetching for efficiency. Training progress and validation metrics (e.g., accuracy, loss) are monitored and visualized using Matplotlib.

Model Persistence and Inference
Upon training completion, the model is saved as toxicity_model.h5 for future inference. The saved model is reloaded and used to predict toxicity levels of new comments, showcasing how to use the trained model for real-world applications.

Interactive Scoring Interface using Gradio
Gradio is employed to create an interactive web-based interface for scoring comments in real-time. Users can input comments through a textbox, and the model predicts toxicity probabilities across different categories, displaying results dynamically.

Installation
To run this project, ensure Python and the necessary libraries are installed:

TensorFlow (including TensorFlow GPU for GPU support)
Pandas
Matplotlib
Scikit-learn
Gradio
