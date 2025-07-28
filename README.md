# Real-time Facial Emotion Recognition using ImageNet and VGGNet

## Overview

This repository contains the code and resources for a **Real-time Facial Emotion Recognition** system. Developed as part of a B.Tech Computer Science and Engineering (CSE) curriculum with a specialization in Artificial Intelligence and Machine Learning, this project leverages deep learning techniques, specifically transfer learning with VGGNet and ImageNet, to detect and classify human emotions from live video feeds or static images.

The system is capable of recognizing various emotional states with an accuracy of **70%**.

## Features

* **Real-time Emotion Detection:** Processes video streams to identify emotions as they occur.
* **Facial Detection:** Utilizes Haar Cascades (or similar method) to accurately locate faces within frames.
* **Transfer Learning:** Employs a pre-trained VGGNet model (fine-tuned on ImageNet) to extract robust features for emotion classification.
* **Emotion Classification:** Predicts a range of emotions (e.g., Happy, Sad, Angry, Neutral, Surprise, Disgust, Fear).
* **Python & TensorFlow/Keras:** Built using popular and powerful AI/ML frameworks.

## Project Structure

* `emotiondetector.json`: The JSON file containing the architecture of the trained deep learning model.
* `emotiondetector.h5`: The H5 file containing the trained weights of the facial emotion recognition model.
* `faceemotion.ipynb`: A Jupyter Notebook containing the full code for data loading, preprocessing, model definition, training, evaluation, and visualization of the training process.
* `realtimedetect.py`: A Python script designed for performing real-time emotion detection on a live webcam feed.
* `images/`: (Optional) Directory to store example images for testing or demonstration.
* `README.md`: This README file.

## Technologies Used

* **Python**
* **TensorFlow / Keras:** For building and training the deep learning model.
* **OpenCV (cv2):** For image/video processing and facial detection.
* **Numpy:** For numerical operations.
* **Jupyter Notebook:** For interactive development and experimentation.
* **VGGNet Architecture:** Utilized for transfer learning.
* **ImageNet Dataset:** Used as the base for pre-trained weights.

## Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/LokendraPandey1/Facial-Emotion-Recognition-using-ImageNet-and-VGGNet.git](https://github.com/LokendraPandey1/Facial-Emotion-Recognition-using-ImageNet-and-VGGNet.git)
    cd Facial-Emotion-Recognition-using-ImageNet-and-VGGNet
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install tensorflow keras opencv-python numpy matplotlib
    ```
    *(Note: `keras` is often included with `tensorflow` now, but it's good to list explicitly if your code uses `keras` directly.)*

4.  **Download Model Weights (if not already in repo):**
    Ensure `emotiondetector.json` and `emotiondetector.h5` are present in the root directory. If they are very large files and not directly committed to Git, you might need to provide instructions on where to download them from (e.g., a Google Drive link). *(Based on your current repo, they are committed, which is fine for their size.)*

## Usage

### 1. Training and Evaluation (Jupyter Notebook)

Open and run the `faceemotion.ipynb` notebook to understand the model training process, data preparation, and evaluation metrics.
```bash
jupyter notebook faceemotion.ipynb
```

### 2. Real-time Emotion Detection

To run the real-time detection script using your webcam:
```bash
python realtimedetect.py
```
_Ensure you have a working webcam connected._

### 3. Predicting on a Single Image

You can use the provided predict_emotion.py script (if you create one based on our earlier discussion) or integrate similar logic into your notebook.

## Future Work & Improvements

* Increase model accuracy by experimenting with different architectures, datasets, or advanced training techniques (e.g., data augmentation, regularization).
* Explore more robust facial alignment methods.
* Implement a user interface (GUI) for easier interaction.
* Optimize for deployment on edge devices.
* Extend to recognize more nuanced emotional states or compound emotions.
* Integrate with other applications (e.g., sentiment analysis, human-computer interaction).
